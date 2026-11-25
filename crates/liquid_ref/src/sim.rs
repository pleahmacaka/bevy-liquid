use glam::{DVec3, UVec3};

use crate::adhesion::adhere;
use crate::extrapolate::extrapolate_mac;
use crate::grid::{Field3, Grid, MacField};
use crate::idp::{density_error, displace, fluid_fraction, particle_density, solve_displacement};
use crate::levelset::{
    face_owners, face_weights, fill_solid_cells, fraction_inside_square, liquid_phi, redistance,
    solid_phi_nodes,
};
use crate::narrow_band::{
    NarrowBand, Resample, VolumeControl, advect_level_set, advect_velocity, combine_level_sets,
    combine_velocity, initial_velocity,
};
use crate::particles::{CellIndex, Particle, hash_u32, seed_cell};
use crate::pressure::{
    Level, Multigrid, PressureSystem, SolveStats, apply_gradient, body_couplings, divergence_rhs,
    floor_ghost, floor_surface, pcg,
};
use crate::rigid::{RigidBody, Shape, apply_inverse_mass};
use crate::surface_tension::{
    PressureJump, capillary_damping, capillary_dt, capillary_viscosity, curvature,
};
use crate::transfer::{GridTransfer, Transfer, g2p, p2g};
use crate::viscosity::{Fractions, ViscositySolve};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    pub gravity: DVec3,

    pub density: f64,

    pub cfl: f64,

    pub max_dt: f64,

    pub transfer: Transfer,

    pub tolerance: f64,

    pub max_iterations: usize,

    pub density_projection: bool,

    pub extrapolation_layers: usize,

    pub multigrid_min_dim: u32,

    pub solid_margin_cells: f64,

    pub narrow_band: Option<NarrowBand>,

    pub viscosity: f64,

    pub surface_tension: f64,

    pub adhesion: f64,

    pub boundary_width: u32,

    pub boundary_sweeps: usize,

    pub block_smoothing: bool,

    pub open_floor: Option<f64>,

    pub tall_cells: Option<u32>,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            gravity: DVec3::new(0.0, -9.81, 0.0),
            density: 1000.0,
            cfl: 1.0,
            max_dt: 1.0 / 60.0,
            transfer: Transfer::Apic,
            tolerance: 1e-8,
            max_iterations: 200,
            density_projection: true,
            extrapolation_layers: 4,
            multigrid_min_dim: 4,
            solid_margin_cells: 0.1,
            narrow_band: None,
            viscosity: 0.0,
            surface_tension: 0.0,
            adhesion: 0.0,
            boundary_width: 0,
            boundary_sweeps: 0,
            block_smoothing: false,
            open_floor: None,
            tall_cells: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StepReport {
    pub dt: f64,

    pub pressure: SolveStats,

    pub density: Option<SolveStats>,

    pub viscosity: Option<SolveStats>,

    pub capillary: Option<SolveStats>,

    pub max_divergence: f64,

    pub unknowns: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Reservoir {
    pub level: f64,

    pub rate: f64,
}

pub struct Simulation {
    pub grid: Grid,

    pub params: Params,

    pub particles: Vec<Particle>,

    pub bodies: Vec<RigidBody>,

    pub velocity: MacField,

    pub pressure: Vec<f64>,

    pub phi: Field3,

    pub time: f64,

    pub report: StepReport,

    pub volume_control: Option<VolumeControl>,

    pub ticks: u32,

    pub reservoir: Option<Reservoir>,
}

impl Simulation {
    pub fn new(
        grid: Grid,
        params: Params,
        particles: Vec<Particle>,
        bodies: Vec<RigidBody>,
    ) -> Self {
        let mut sim = Self {
            grid,
            params,
            particles,
            bodies,
            velocity: MacField::new(&grid),
            pressure: vec![0.0; grid.cell_count()],
            phi: grid.cells(),
            time: 0.0,
            report: StepReport::default(),
            volume_control: None,
            ticks: 0,
            reservoir: None,
        };

        if params.narrow_band.is_some() {
            sim.phi = Field3::new(grid.dims, f64::INFINITY);
            sim.velocity = initial_velocity(&grid, &sim.particles, params.extrapolation_layers);
        }

        sim.refresh_phi();

        sim
    }

    pub fn refresh_phi(&mut self) {
        let index = CellIndex::build(&self.grid, &self.particles);
        self.phi = self.geometry(&index, self.params.open_floor.is_some()).0;
    }

    pub fn cfl_dt(&self) -> f64 {
        let vmax = self
            .particles
            .iter()
            .fold(0.0f64, |m, p| m.max(p.v.length()));
        let bmax = self
            .bodies
            .iter()
            .fold(0.0f64, |m, b| m.max(b.velocity.length()));
        let grid_speed = if self.params.narrow_band.is_some() {
            self.velocity.max_abs()
        } else {
            0.0
        };
        let g = self.params.gravity.length();
        let speed = vmax.max(bmax).max(grid_speed) + (5.0 * self.grid.dx * g).sqrt();
        let dt = (self.params.cfl * self.grid.dx / speed.max(1e-12)).min(self.params.max_dt);

        if self.params.surface_tension > 0.0 {
            dt.min(capillary_dt(
                &self.grid,
                self.params.density,
                self.params.surface_tension,
            ))
        } else {
            dt
        }
    }

    pub fn advance(&mut self, duration: f64) -> usize {
        let mut remaining = duration;
        let mut steps = 0;

        while remaining > 1e-12 {
            let dt = self.cfl_dt();
            let dt = if dt >= remaining * (1.0 - 1e-5) {
                remaining
            } else if 2.0 * dt > remaining {
                0.5 * remaining
            } else {
                dt
            };
            self.substep(dt);
            remaining -= dt;
            steps += 1;
        }

        steps
    }

    pub fn substep(&mut self, dt: f64) {
        let grid = self.grid;
        let p = self.params;
        self.ticks = self.ticks.wrapping_add(1);

        let index = CellIndex::build(&grid, &self.particles);
        let transfer = p2g(&grid, &self.particles, &index);
        let (phi, weights, owners, _) = self.geometry(&index, p.open_floor.is_some());
        let (mut velocity, mut valid) = self.transferred_velocity(transfer, &phi);

        extrapolate_mac(&mut velocity, &mut valid, 2);

        for axis in 0..3 {
            let g = p.gravity[axis] * dt;
            velocity.axes[axis].data.iter_mut().for_each(|v| *v += g);
        }

        for b in self.bodies.iter_mut().filter(|b| b.is_dynamic()) {
            b.velocity += p.gravity * dt;
        }

        if p.adhesion > 0.0 {
            adhere(&grid, &self.bodies, &weights, &mut velocity, p.adhesion, dt);
        }

        let solid = solid_velocity(&grid, &self.bodies, &owners);

        let scale = dt / (p.density * grid.dx * grid.dx);
        let mut level = Level::from_fine(&grid, &phi, &weights, scale);

        if let Some(reach) = p.open_floor {
            let weight = -p.density * p.gravity.y;
            let surface = floor_surface(&level, grid.dx);
            let mean = surface.iter().sum::<f64>() / surface.len() as f64;
            let offset = self.reservoir_offset(mean);

            level.ghost = floor_ghost(&level, grid.dx, reach, weight);
            level.ghost.iter_mut().for_each(|g| *g += weight * offset);
        }

        let multigrid = Multigrid::new(level, p.multigrid_min_dim)
            .with_boundary_smoothing(p.boundary_width, p.boundary_sweeps)
            .with_tall(&weights, p.tall_cells)
            .with_block_smoothing(p.block_smoothing);
        let couplings =
            body_couplings(&grid, &multigrid.levels[0], &weights, &owners, &self.bodies);
        let coupling_scale = dt / grid.dx.powi(3);
        let mut rhs = divergence_rhs(&grid, &multigrid.levels[0], &velocity, &weights, &solid);
        multigrid.levels[0].add_floor_rhs(&mut rhs);

        if let Some(target) = self.volume_divergence(&phi, dt) {
            let level = &multigrid.levels[0];

            for (n, r) in rhs.iter_mut().enumerate() {
                if level.active[n] {
                    *r += target * open_fraction(&weights, level.coords(n));
                }
            }
        }

        let distance = (p.surface_tension > 0.0).then(|| redistance(&phi, grid.dx));
        let kappa = distance.as_ref().map(|d| curvature(&grid, d));
        let jump = kappa
            .as_ref()
            .map(|kappa| PressureJump::new(&grid, &multigrid.levels[0], kappa, p.surface_tension));

        if let Some(jump) = &jump {
            jump.add_to_rhs(&mut rhs);
        }

        let tolerance = multigrid.levels[0].gather_rhs(&mut rhs, p.tolerance);

        let system = PressureSystem {
            multigrid: &multigrid,
            couplings: &couplings,
            bodies: &self.bodies,
            coupling_scale,
        };
        let solve = pcg(
            &system,
            &rhs,
            &mut self.pressure,
            tolerance,
            p.max_iterations,
            true,
        );

        multigrid.levels[0].expand(&mut self.pressure);

        for c in &couplings {
            let impulse = c.gather(&self.pressure).map(|v| v * dt);
            let dv = apply_inverse_mass(&self.bodies[c.body], impulse);
            let body = &mut self.bodies[c.body];

            body.velocity += DVec3::new(dv[0], dv[1], dv[2]);
            body.angular_velocity += DVec3::new(dv[3], dv[4], dv[5]);
        }

        let mut valid = apply_gradient(
            &grid,
            &multigrid.levels[0],
            &self.pressure,
            dt / (p.density * grid.dx),
            &mut velocity,
            &weights,
        );

        if let Some(jump) = &jump {
            jump.apply_gradient(dt / (p.density * grid.dx), &mut velocity, &weights);
        }

        let solid_after = solid_velocity(&grid, &self.bodies, &owners);

        if p.narrow_band.is_some() {
            blend_partial(&mut velocity, &mut valid, &weights, &solid_after);
        }

        self.record_floor_flux(&multigrid.levels[0], &velocity, scale);

        extrapolate_mac(&mut velocity, &mut valid, p.extrapolation_layers);
        constrain(&mut velocity, &weights, &solid_after);

        let boundary = (&weights, &solid_after);
        let damping = distance.as_ref().map(|d| {
            let nu = capillary_viscosity(&grid, p.density, p.surface_tension, dt);

            (capillary_damping(&grid, d), nu)
        });
        let (capillary, viscosity) = if p.viscosity > 0.0 {
            let nu = p.viscosity / p.density;
            let liquid = Fractions::new(&grid, &phi);
            let stress = match &damping {
                Some((damping, capillary)) => liquid.blend(damping, capillary / nu),
                None => liquid,
            };

            (
                None,
                Some(self.diffuse(&stress, nu, boundary, &mut velocity, dt)),
            )
        } else {
            let capillary = damping
                .map(|(damping, nu)| self.diffuse(&damping, nu, boundary, &mut velocity, dt));

            (capillary, None)
        };

        let mut divergence = divergence_rhs(
            &grid,
            &multigrid.levels[0],
            &velocity,
            &weights,
            &solid_after,
        );
        multigrid.levels[0].gather(&mut divergence, true);

        let max_divergence = divergence.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let unknowns = multigrid.levels[0].unknowns();

        if p.open_floor.is_some() && p.narrow_band.is_none() {
            self.spawn_floor();
        }

        g2p(&grid, &velocity, &mut self.particles, p.transfer);

        if let Some(band) = p.narrow_band {
            self.resample_band(band, &phi, &velocity);
        }

        advect(&grid, &velocity, &mut self.particles, dt, p.cfl);
        self.cull_floor();

        for b in &mut self.bodies {
            b.integrate(dt);
        }

        self.keep_particles_inside();

        let density =
            (p.density_projection && p.narrow_band.is_none()).then(|| self.project_density());

        if p.narrow_band.is_some() {
            blend_solid(&mut velocity, &valid, &weights, &solid_after);
        }

        self.velocity = velocity;
        self.phi = phi;
        self.time += dt;
        self.report = StepReport {
            dt,
            pressure: solve,
            density,
            viscosity,
            capillary,
            max_divergence,
            unknowns,
        };
    }

    fn transferred_velocity(&self, transfer: GridTransfer, phi: &Field3) -> (MacField, MacField) {
        let Some(band) = self.params.narrow_band else {
            return (transfer.velocity, transfer.mass);
        };

        let dt = self.report.dt;
        let advected = if dt > 0.0 {
            advect_velocity(&self.grid, &self.velocity, dt)
        } else {
            self.velocity.clone()
        };

        combine_velocity(&self.grid, phi, &advected, &transfer, band)
    }

    fn liquid_surface(&self, index: &CellIndex) -> Field3 {
        let particle_phi = liquid_phi(&self.grid, &self.particles, index);

        if self.params.narrow_band.is_none() {
            return particle_phi;
        }

        let dt = self.report.dt;
        let advected = if dt > 0.0 {
            advect_level_set(&self.grid, &self.phi, &self.velocity, dt)
        } else {
            self.phi.clone()
        };

        combine_level_sets(&self.grid, &advected, &particle_phi)
    }

    fn volume_divergence(&mut self, phi: &Field3, dt: f64) -> Option<f64> {
        self.params.narrow_band?;

        if self.params.open_floor.is_some() {
            return None;
        }

        let volume = level_set_volume(&self.grid, phi);
        let control = self
            .volume_control
            .get_or_insert_with(|| VolumeControl::new(volume));

        Some(control.divergence(volume, dt))
    }

    fn resample_band(&mut self, band: NarrowBand, phi: &Field3, velocity: &MacField) {
        let bits = self.time.to_bits();

        Resample {
            grid: &self.grid,
            phi,
            velocity,
            transfer: self.params.transfer,
            band,
            salt: hash_u32(bits as u32 ^ (bits >> 32) as u32),
        }
        .apply(&mut self.particles);
    }

    fn diffuse(
        &self,
        fractions: &Fractions,
        kinematic_viscosity: f64,
        boundary: (&MacField, &MacField),
        velocity: &mut MacField,
        dt: f64,
    ) -> SolveStats {
        let p = self.params;
        let (weights, solid) = boundary;
        let stats = ViscositySolve {
            grid: &self.grid,
            fractions,
            weights,
            kinematic_viscosity,
            dt,
            tolerance: p.tolerance,
            max_iterations: p.max_iterations,
        }
        .apply(velocity);

        let mut liquid = fractions.faces.clone();
        extrapolate_mac(velocity, &mut liquid, p.extrapolation_layers);
        constrain(velocity, weights, solid);

        stats
    }

    fn geometry(&self, index: &CellIndex, open_floor: bool) -> (Field3, MacField, [Vec<i32>; 3], Field3) {
        let mut phi = self.liquid_surface(index);
        let nodes = solid_phi_nodes(&self.grid, &self.bodies);

        fill_solid_cells(&mut phi, &nodes);

        let mut weights = face_weights(&self.grid, &nodes);

        if open_floor {
            open_floor_weights(&self.grid, &nodes, &mut weights);
        }

        let owners = face_owners(&self.grid, &self.bodies, &weights);

        (phi, weights, owners, nodes)
    }

    fn spawn_floor(&mut self) {
        let grid = self.grid;
        let salt = hash_u32(self.ticks);

        for k in 0..grid.dims.z {
            for i in 0..grid.dims.x {
                for x in seed_cell(&grid, UVec3::new(i, 0, k), 1.0, salt) {
                    self.particles
                        .push(Particle::at_rest(x - DVec3::Y * grid.dx));
                }
            }
        }
    }

    fn project_density(&mut self) -> SolveStats {
        let grid = self.grid;
        let index = CellIndex::build(&grid, &self.particles);
        let (phi, weights, _, nodes) = self.geometry(&index, false);
        let level = Level::from_fine(&grid, &phi, &weights, 1.0 / (grid.dx * grid.dx));
        let fluid = fluid_fraction(&grid, &level, &nodes);
        let density = particle_density(&grid, &level, &self.particles, &index, &fluid);
        let error = density_error(&density, &level);
        let multigrid = Multigrid::new(level, self.params.multigrid_min_dim)
            .with_boundary_smoothing(self.params.boundary_width, self.params.boundary_sweeps)
            .with_block_smoothing(self.params.block_smoothing);

        let projection = solve_displacement(
            &grid,
            &multigrid,
            &weights,
            &error,
            self.params.tolerance.max(1e-6),
            self.params.max_iterations,
            2,
        );

        displace(&grid, &projection.displacement, &mut self.particles);
        self.keep_particles_inside();

        projection.stats
    }

    fn reservoir_offset(&mut self, mean: f64) -> f64 {
        let gravity = -self.params.gravity.y;

        if gravity <= 0.0 {
            return 0.0;
        }

        let Reservoir { level, rate } = *self.reservoir.get_or_insert(Reservoir { level: mean, rate: 0.0 });

        level - mean - (level / gravity).sqrt() * rate
    }

    fn record_floor_flux(&mut self, level: &Level, velocity: &MacField, scale: f64) {
        let Some(reservoir) = &mut self.reservoir else {
            return;
        };

        let flux: f64 = (0..level.columns())
            .filter(|&col| level.floor[col] > 0.0 && level.active[level.floor_cell(col)])
            .map(|col| {
                let c = level.coords(level.floor_cell(col));

                level.floor[col] * velocity.axes[1].get(c.x, 0, c.z)
            })
            .sum();

        reservoir.rate = flux / (scale * level.columns() as f64);
    }

    fn cull_floor(&mut self) {
        if self.params.open_floor.is_some() {
            self.particles.retain(|q| q.x.y >= 0.0);
        }
    }

    fn keep_particles_inside(&mut self) {
        let grid = self.grid;
        let margin = self.params.solid_margin_cells * grid.dx;
        let hi = grid.size() - margin;
        let mut lo = DVec3::splat(margin);
        let bodies = &self.bodies;

        if self.params.open_floor.is_some() {
            lo.y = 0.0;
        }

        let free = Free { bodies, margin, lo, hi };

        crate::exec::each_mut(&mut self.particles, |_, p| {
            for b in bodies {
                let s = b.sdf(p.x);

                if s < margin {
                    let out = free.push_out(b, p.x, s, grid.dx * 1e-3);
                    let n = (out - p.x).normalize_or_zero();
                    let inward = (p.v - b.velocity_at(out)).dot(n);

                    if inward < 0.0 {
                        p.v -= inward * n;
                    }

                    p.x = out;
                }
            }

            let clamped = p.x.clamp(lo, hi);

            for a in 0..3 {
                if clamped[a] != p.x[a] {
                    p.v[a] = 0.0;
                }
            }

            p.x = clamped;
        });
    }

    pub fn liquid_volume(&self) -> f64 {
        level_set_volume(&self.grid, &self.phi)
    }

    pub fn kinetic_energy(&self) -> f64 {
        let m = self.params.density * self.grid.dx.powi(3) / 8.0;

        self.particles
            .iter()
            .map(|p| 0.5 * m * p.v.length_squared())
            .sum()
    }

    pub fn momentum(&self) -> DVec3 {
        let m = self.params.density * self.grid.dx.powi(3) / 8.0;

        self.particles.iter().map(|p| p.v * m).sum()
    }
}

pub fn sdf_gradient(body: &RigidBody, x: DVec3, h: f64) -> DVec3 {
    let g = DVec3::new(
        body.sdf(x + DVec3::X * h) - body.sdf(x - DVec3::X * h),
        body.sdf(x + DVec3::Y * h) - body.sdf(x - DVec3::Y * h),
        body.sdf(x + DVec3::Z * h) - body.sdf(x - DVec3::Z * h),
    );

    g.normalize_or_zero()
}

pub struct Free<'a> {
    pub bodies: &'a [RigidBody],

    pub margin: f64,

    pub lo: DVec3,

    pub hi: DVec3,
}

impl Free<'_> {
    pub fn contains(&self, x: DVec3) -> bool {
        x.cmpge(self.lo).all()
            && x.cmple(self.hi).all()
            && self.bodies.iter().all(|b| b.sdf(x) >= 0.99 * self.margin)
    }

    pub fn push_out(&self, body: &RigidBody, x: DVec3, s: f64, h: f64) -> DVec3 {
        let along = x + sdf_gradient(body, x, h) * (self.margin - s);
        let mut best = (along, f64::INFINITY);

        if self.contains(along) {
            best.1 = (along - x).length();
        }

        let p = body.rotation.inverse() * (x - body.position);

        for a in 0..3 {
            for sign in [-1.0, 1.0] {
                let reach = match body.shape {
                    Shape::Sphere { radius } => {
                        let r = radius + self.margin;
                        (r * r - (p.length_squared() - p[a] * p[a])).max(0.0).sqrt()
                    }
                    Shape::Cuboid { half } => half[a] + self.margin,
                };
                let mut q = p;
                q[a] = sign * reach;

                let c = body.position + body.rotation * q;
                let d = (q - p).length();

                if d < best.1 && self.contains(c) {
                    best = (c, d);
                }
            }
        }

        best.0
    }
}

pub fn solid_velocity(grid: &Grid, bodies: &[RigidBody], owners: &[Vec<i32>; 3]) -> MacField {
    let mut solid = MacField::new(grid);

    for axis in 0..3 {
        let field = &mut solid.axes[axis];
        let dims: UVec3 = field.dims;

        let owner = &owners[axis];

        field.data = crate::exec::map(owner.len(), |n| {
            let o = owner[n];

            if o < 0 {
                return 0.0;
            }

            let n = n as u32;
            let f = UVec3::new(n % dims.x, (n / dims.x) % dims.y, n / (dims.x * dims.y));
            let x = grid.face_position(axis, f);

            bodies[o as usize].velocity_at(x)[axis]
        });
    }

    solid
}

pub fn open_floor_weights(grid: &Grid, nodes: &Field3, weights: &mut MacField) {
    let field = &mut weights.axes[1];

    for k in 0..grid.dims.z {
        for i in 0..grid.dims.x {
            let corner = |a: u32, b: u32| nodes.get(i + b, 0, k + a);
            let blocked =
                fraction_inside_square(corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1));

            field.set(i, 0, k, 1.0 - blocked);
        }
    }
}

pub fn open_fraction(weights: &MacField, c: UVec3) -> f64 {
    (0..3)
        .map(|axis| {
            let field = &weights.axes[axis];
            let mut hi = c;
            hi[axis] += 1;

            field.get(c.x, c.y, c.z) + field.get(hi.x, hi.y, hi.z)
        })
        .sum::<f64>()
        / 6.0
}

pub fn constrain(velocity: &mut MacField, weights: &MacField, solid: &MacField) {
    for axis in 0..3 {
        let w = &weights.axes[axis].data;
        let s = &solid.axes[axis].data;

        for (n, v) in velocity.axes[axis].data.iter_mut().enumerate() {
            if w[n] <= 0.0 {
                *v = s[n];
            }
        }
    }
}

pub fn blend_partial(velocity: &mut MacField, valid: &mut MacField, weights: &MacField, solid: &MacField) {
    for axis in 0..3 {
        let w = &weights.axes[axis].data;
        let s = &solid.axes[axis].data;
        let marks = &mut valid.axes[axis].data;

        for (n, v) in velocity.axes[axis].data.iter_mut().enumerate() {
            if marks[n] > 0.0 {
                let w = w[n].min(1.0);
                *v = w * *v + (1.0 - w) * s[n];
                marks[n] = 2.0;
            }
        }
    }
}

pub fn blend_solid(velocity: &mut MacField, valid: &MacField, weights: &MacField, solid: &MacField) {
    for axis in 0..3 {
        let w = &weights.axes[axis].data;
        let s = &solid.axes[axis].data;
        let marks = &valid.axes[axis].data;

        for (n, v) in velocity.axes[axis].data.iter_mut().enumerate() {
            if marks[n] <= 1.5 {
                let w = w[n].clamp(0.0, 1.0);
                *v = w * *v + (1.0 - w) * s[n];
            }
        }
    }
}

pub fn advect(grid: &Grid, velocity: &MacField, particles: &mut [Particle], dt: f64, cfl: f64) {
    crate::exec::each_mut(particles, |_, p| {
        let mut v = velocity.sample(grid, p.x);
        let cycles = if cfl > 1.0 { (dt * v.length() / grid.dx).ceil().clamp(1.0, ADVECT_CYCLES as f64) as usize } else { 1 };
        let h = dt / cycles as f64;

        for k in 0..cycles {
            if k > 0 {
                v = velocity.sample(grid, p.x);
            }

            p.x += h * velocity.sample(grid, p.x + 0.5 * h * v);
        }
    });
}

pub const ADVECT_CYCLES: usize = 8;

pub fn level_set_volume(grid: &Grid, phi: &Field3) -> f64 {
    let cell = grid.dx.powi(3);

    phi.data
        .iter()
        .map(|phi| (0.5 - phi / grid.dx).clamp(0.0, 1.0) * cell)
        .sum()
}
