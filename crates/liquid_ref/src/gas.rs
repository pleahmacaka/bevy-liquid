use glam::{DVec3, IVec3, UVec3};

use crate::exec;
use crate::grid::{Field3, Grid, MacField, trilinear_corners};
use crate::pressure::{
    Level, Multigrid, PressureSystem, SolveStats, face_pressure_difference, max_abs, pcg,
};
use crate::rigid::Shape;

const CELL_OFFSET: DVec3 = DVec3::splat(0.5);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasParams {
    pub ambient_temperature: f64,

    pub density_buoyancy: f64,

    pub thermal_buoyancy: f64,

    pub vorticity_confinement: f64,

    pub density_decay: f64,

    pub cooling: f64,

    pub air_density: f64,

    pub cfl: f64,

    pub max_dt: f64,

    pub tolerance: f64,

    pub max_iterations: usize,

    pub multigrid_min_dim: u32,

    pub maccormack: bool,

    pub open_low: [bool; 3],

    pub open_high: [bool; 3],
}

impl Default for GasParams {
    fn default() -> Self {
        Self {
            ambient_temperature: 293.15,
            density_buoyancy: 0.05,
            thermal_buoyancy: 9.81 / 293.15,
            vorticity_confinement: 0.5,
            density_decay: 0.1,
            cooling: 0.5,
            air_density: 1.2,
            cfl: 1.0,
            max_dt: 1.0 / 60.0,
            tolerance: 1e-8,
            max_iterations: 200,
            multigrid_min_dim: 4,
            maccormack: true,
            open_low: [false; 3],
            open_high: [false, true, false],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Region {
    pub shape: Shape,

    pub center: DVec3,
}

impl Region {
    pub fn contains(&self, x: DVec3) -> bool {
        self.shape.sdf_local(x - self.center) <= 0.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitter {
    pub region: Region,

    pub density_rate: f64,

    pub heat_rate: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GasReport {
    pub dt: f64,

    pub pressure: SolveStats,

    pub input_divergence: f64,
}

#[derive(Clone, Debug)]
pub struct GasDomain {
    pub grid: Grid,

    pub origin: DVec3,

    pub params: GasParams,

    pub emitters: Vec<Emitter>,

    pub obstacles: Vec<Region>,

    pub velocity: MacField,

    pub density: Field3,

    pub temperature: Field3,

    pub time: f64,

    pub report: GasReport,
}

impl GasDomain {
    pub fn new(grid: Grid, origin: DVec3, params: GasParams) -> Self {
        Self {
            grid,
            origin,
            params,
            emitters: Vec::new(),
            obstacles: Vec::new(),
            velocity: MacField::new(&grid),
            density: grid.cells(),
            temperature: Field3::new(grid.dims, params.ambient_temperature),
            time: 0.0,
            report: GasReport::default(),
        }
    }

    pub fn cell_position(&self, n: usize) -> DVec3 {
        self.origin + self.grid.cell_center(self.density.coords(n))
    }

    pub fn mass(&self) -> f64 {
        self.density.data.iter().sum::<f64>() * self.grid.dx.powi(3)
    }

    pub fn solid(&self) -> Vec<bool> {
        exec::map(self.grid.cell_count(), |n| self.is_solid(n))
    }

    fn is_solid(&self, n: usize) -> bool {
        let x = self.cell_position(n);

        self.obstacles.iter().any(|o| o.contains(x))
    }

    fn cell_index(&self, c: IVec3) -> Option<usize> {
        self.density
            .contains(c)
            .then(|| self.density.index(c.x as u32, c.y as u32, c.z as u32))
    }

    fn face_open(&self, axis: usize, f: UVec3, solid: impl Fn(usize) -> bool) -> bool {
        let hi = f.as_ivec3();
        let mut lo = hi;
        lo[axis] -= 1;

        [
            (lo, self.params.open_low[axis]),
            (hi, self.params.open_high[axis]),
        ]
        .into_iter()
        .all(|(c, open)| self.cell_index(c).map_or(open, |n| !solid(n)))
    }

    fn face_weights(&self, solid: &[bool]) -> MacField {
        let mut weights = MacField::new(&self.grid);

        for (axis, field) in weights.axes.iter_mut().enumerate() {
            field.data = exec::map(field.data.len(), |n| {
                if self.face_open(axis, field.coords(n), |m| solid[m]) {
                    1.0
                } else {
                    0.0
                }
            });
        }

        weights
    }

    pub fn cfl_dt(&self) -> f64 {
        let speed = self.velocity.max_abs();

        (self.params.cfl * self.grid.dx / speed.max(1e-12)).min(self.params.max_dt)
    }

    pub fn advance(&mut self, duration: f64) -> usize {
        let mut remaining = duration;
        let mut substeps = 0;

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
            substeps += 1;
        }

        substeps
    }

    pub fn substep(&mut self, dt: f64) {
        let solid = self.solid();
        let weights = self.face_weights(&solid);

        self.emit(dt);
        self.advect(dt);
        self.dissipate(&solid, dt);
        self.add_forces(dt);
        self.enforce(&weights);

        let (pressure, input_divergence) = self.project(&weights);

        self.time += dt;
        self.report = GasReport {
            dt,
            pressure,
            input_divergence,
        };
    }

    fn emit(&mut self, dt: f64) {
        for n in 0..self.grid.cell_count() {
            let x = self.cell_position(n);

            for e in self.emitters.iter().filter(|e| e.region.contains(x)) {
                self.density.data[n] += e.density_rate * dt;
                self.temperature.data[n] += e.heat_rate * dt;
            }
        }
    }

    fn advect(&mut self, dt: f64) {
        let density = self.advect_field(&self.density, CELL_OFFSET, dt);
        let temperature = self.advect_field(&self.temperature, CELL_OFFSET, dt);
        let axes = std::array::from_fn(|axis| {
            self.advect_field(&self.velocity.axes[axis], Grid::face_offset(axis), dt)
        });

        self.density = density;
        self.temperature = temperature;
        self.velocity = MacField { axes };
    }

    fn backtrace(&self, x: DVec3, dt: f64) -> DVec3 {
        let mid = x - 0.5 * dt * self.velocity.sample(&self.grid, x);

        x - dt * self.velocity.sample(&self.grid, mid)
    }

    fn trace(&self, source: &Field3, offset: DVec3, dt: f64) -> Vec<(f64, f64, f64)> {
        let dx = self.grid.dx;

        exec::map(source.data.len(), |n| {
            let x = (source.coords(n).as_dvec3() + offset) * dx;
            let g = self.backtrace(x, dt) / dx - offset;

            trilinear_corners(source, g).iter().fold(
                (0.0, f64::INFINITY, f64::NEG_INFINITY),
                |(v, lo, hi), (w, u)| (v + w * u, lo.min(*u), hi.max(*u)),
            )
        })
    }

    fn advect_field(&self, field: &Field3, offset: DVec3, dt: f64) -> Field3 {
        let forward = self.trace(field, offset, dt);
        let ahead = Field3 {
            dims: field.dims,
            data: forward.iter().map(|t| t.0).collect(),
        };

        if !self.params.maccormack {
            return ahead;
        }

        let behind = self.trace(&ahead, offset, -dt);
        let data = forward
            .iter()
            .zip(&behind)
            .zip(&field.data)
            .map(|((&(a, lo, hi), b), u)| (a + 0.5 * (u - b.0)).clamp(lo, hi))
            .collect();

        Field3 {
            dims: field.dims,
            data,
        }
    }

    fn dissipate(&mut self, solid: &[bool], dt: f64) {
        let p = self.params;
        let keep = (-p.density_decay * dt).exp();
        let cool = (-p.cooling * dt).exp();
        let ambient = p.ambient_temperature;

        for r in &mut self.density.data {
            *r *= keep;
        }

        for t in &mut self.temperature.data {
            *t = ambient + (*t - ambient) * cool;
        }

        for n in (0..solid.len()).filter(|&n| solid[n]) {
            self.density.data[n] = 0.0;
            self.temperature.data[n] = ambient;
        }
    }

    fn add_forces(&mut self, dt: f64) {
        let lift = self.lift();
        let swirl = (self.params.vorticity_confinement > 0.0).then(|| self.confinement());

        add_to_faces(&mut self.velocity.axes[1], 1, &lift, dt);

        if let Some(swirl) = swirl {
            for (axis, force) in swirl.iter().enumerate() {
                add_to_faces(&mut self.velocity.axes[axis], axis, force, dt);
            }
        }
    }

    fn lift(&self) -> Field3 {
        let p = self.params;
        let mut lift = self.grid.cells();

        lift.data = self
            .density
            .data
            .iter()
            .zip(&self.temperature.data)
            .map(|(r, t)| p.thermal_buoyancy * (t - p.ambient_temperature) - p.density_buoyancy * r)
            .collect();

        lift
    }

    pub fn vorticity(&self) -> Vec<DVec3> {
        let u = self.cell_velocity();
        let dx = self.grid.dx;

        exec::map(self.grid.cell_count(), |n| {
            let c = self.density.coords(n);
            let d = |component: usize, axis: usize| derivative(&u[component], c, axis, dx);

            DVec3::new(d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1))
        })
    }

    fn cell_velocity(&self) -> [Field3; 3] {
        std::array::from_fn(|axis| {
            let faces = &self.velocity.axes[axis];
            let mut out = self.grid.cells();

            out.data = exec::map(out.data.len(), |n| {
                let c = self.density.coords(n);
                let mut hi = c;
                hi[axis] += 1;

                0.5 * (faces.get(c.x, c.y, c.z) + faces.get(hi.x, hi.y, hi.z))
            });

            out
        })
    }

    fn confinement(&self) -> [Field3; 3] {
        let omega = self.vorticity();
        let dx = self.grid.dx;
        let scale = self.params.vorticity_confinement * dx;
        let mut magnitude = self.grid.cells();

        magnitude.data = omega.iter().map(|w| w.length()).collect();

        let force = exec::map(omega.len(), |n| {
            let c = magnitude.coords(n);
            let eta = DVec3::from_array(std::array::from_fn(|axis| {
                derivative(&magnitude, c, axis, dx)
            }));

            scale * eta.normalize_or_zero().cross(omega[n])
        });

        std::array::from_fn(|axis| Field3 {
            dims: self.grid.dims,
            data: force.iter().map(|f| f[axis]).collect(),
        })
    }

    fn enforce(&mut self, weights: &MacField) {
        for (faces, open) in self.velocity.axes.iter_mut().zip(&weights.axes) {
            for (v, w) in faces.data.iter_mut().zip(&open.data) {
                if *w <= 0.0 {
                    *v = 0.0;
                }
            }
        }
    }

    pub fn divergence(&self) -> Field3 {
        let inv_dx = 1.0 / self.grid.dx;
        let mut out = self.grid.cells();

        out.data = exec::map(out.data.len(), |n| {
            let c = self.density.coords(n);
            let flux: f64 = (0..3)
                .map(|axis| {
                    let faces = &self.velocity.axes[axis];
                    let mut hi = c;
                    hi[axis] += 1;

                    faces.get(hi.x, hi.y, hi.z) - faces.get(c.x, c.y, c.z)
                })
                .sum();

            flux * inv_dx
        });

        out
    }

    fn project(&mut self, weights: &MacField) -> (SolveStats, f64) {
        let level = self.pressure_level(weights);
        let divergence = self.divergence();
        let mut rhs = vec![0.0; level.len()];

        for (n, d) in divergence.data.iter().enumerate() {
            let e = level.index(divergence.coords(n) + 1);

            if level.active[e] {
                rhs[e] = -d;
            }
        }

        let multigrid = Multigrid::new(level, self.params.multigrid_min_dim);
        let system = PressureSystem {
            multigrid: &multigrid,
            couplings: &[],
            bodies: &[],
            coupling_scale: 0.0,
        };
        let mut pressure = vec![0.0; rhs.len()];
        let stats = pcg(
            &system,
            &rhs,
            &mut pressure,
            self.params.tolerance,
            self.params.max_iterations,
            true,
        );

        self.subtract_gradient(&multigrid.levels[0], &pressure, weights);

        (stats, max_abs(&rhs))
    }

    fn pressure_level(&self, weights: &MacField) -> Level {
        let dx = self.grid.dx;
        let padded = Grid::new(self.grid.dims + 2, dx);
        let mut phi = padded.cells();
        let mut open = MacField::new(&padded);

        phi.data = (0..phi.data.len())
            .map(|n| self.pressure_phi(phi.coords(n)))
            .collect();

        for (axis, w) in weights.axes.iter().enumerate() {
            for (n, value) in w.data.iter().enumerate() {
                let f = w.coords(n) + 1;
                open.axes[axis].set(f.x, f.y, f.z, *value);
            }
        }

        Level::from_fine(&padded, &phi, &open, 1.0 / (dx * dx))
    }

    fn pressure_phi(&self, e: UVec3) -> f64 {
        let last = self.grid.dims + 1;
        let beyond_open = (0..3).any(|a| {
            (e[a] == 0 && self.params.open_low[a]) || (e[a] == last[a] && self.params.open_high[a])
        });

        if beyond_open { 1.0 } else { -1.0 }
    }

    fn subtract_gradient(&mut self, level: &Level, pressure: &[f64], weights: &MacField) {
        let inv_dx = 1.0 / self.grid.dx;

        for (axis, faces) in self.velocity.axes.iter_mut().enumerate() {
            for n in 0..faces.data.len() {
                if weights.axes[axis].data[n] <= 0.0 {
                    continue;
                }

                let hi = faces.coords(n) + 1;
                let mut lo = hi;
                lo[axis] -= 1;

                let dp =
                    face_pressure_difference(level, pressure, level.index(lo), level.index(hi));

                if let Some(dp) = dp {
                    faces.data[n] -= dp * inv_dx;
                }
            }
        }
    }

    pub fn inject(
        &mut self,
        position: DVec3,
        mass: f64,
        temperature: f64,
        velocity: DVec3,
    ) -> bool {
        let x = position - self.origin;

        if mass <= 0.0 || x.cmplt(DVec3::ZERO).any() || x.cmpgt(self.grid.size()).any() {
            return false;
        }

        let g = x / self.grid.dx;
        let cells = spread(self.grid.dims, g - CELL_OFFSET, |c| {
            !self.is_solid(self.density.index(c.x, c.y, c.z))
        });

        if cells.is_empty() {
            return false;
        }

        let volume = self.grid.dx.powi(3);
        let air = self.params.air_density * volume;

        for (c, w) in cells {
            let n = self.density.index(c.x, c.y, c.z);
            let added = w * mass;
            let t = &mut self.temperature.data[n];

            self.density.data[n] += added / volume;
            *t += added / (air + added) * (temperature - *t);
        }

        for axis in 0..3 {
            let faces = spread(
                self.grid.face_dims(axis),
                g - Grid::face_offset(axis),
                |f| self.face_open(axis, f, |m| self.is_solid(m)),
            );
            let field = &mut self.velocity.axes[axis];

            for (f, w) in faces {
                let added = w * mass;
                let n = field.index(f.x, f.y, f.z);

                field.data[n] += added / (air + added) * (velocity[axis] - field.data[n]);
            }
        }

        true
    }
}

fn add_to_faces(faces: &mut Field3, axis: usize, cells: &Field3, scale: f64) {
    let last = cells.dims[axis] - 1;

    for n in 0..faces.data.len() {
        let f = faces.coords(n);
        let mut lo = f;
        let mut hi = f;
        lo[axis] = f[axis].saturating_sub(1);
        hi[axis] = f[axis].min(last);

        faces.data[n] += scale * 0.5 * (cells.get(lo.x, lo.y, lo.z) + cells.get(hi.x, hi.y, hi.z));
    }
}

fn derivative(field: &Field3, c: UVec3, axis: usize, dx: f64) -> f64 {
    let mut lo = c;
    let mut hi = c;
    lo[axis] = c[axis].saturating_sub(1);
    hi[axis] = (c[axis] + 1).min(field.dims[axis] - 1);

    let span = (hi[axis] - lo[axis]) as f64 * dx;

    if span == 0.0 {
        return 0.0;
    }

    (field.get(hi.x, hi.y, hi.z) - field.get(lo.x, lo.y, lo.z)) / span
}

fn corners(dims: UVec3, g: DVec3) -> [(UVec3, f64); 8] {
    let max = (dims.as_dvec3() - 1.0).max(DVec3::ZERO);
    let g = g.clamp(DVec3::ZERO, max);
    let base = g.floor().min((max - 1.0).max(DVec3::ZERO));
    let t = g - base;
    let b = base.as_uvec3();
    let hi = (b + 1).min(dims - 1);

    std::array::from_fn(|corner| {
        let corner = corner as u32;
        let pick = UVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1).cmpeq(UVec3::ONE);
        let w = DVec3::select(pick, t, 1.0 - t);

        (UVec3::select(pick, hi, b), w.x * w.y * w.z)
    })
}

fn spread(dims: UVec3, g: DVec3, open: impl Fn(UVec3) -> bool) -> Vec<(UVec3, f64)> {
    let kept: Vec<(UVec3, f64)> = corners(dims, g)
        .into_iter()
        .filter(|&(c, w)| w > 0.0 && open(c))
        .collect();
    let total: f64 = kept.iter().map(|k| k.1).sum();

    kept.into_iter().map(|(c, w)| (c, w / total)).collect()
}
