use glam::{DVec3, IVec3, UVec3};

use crate::exec;
use crate::extrapolate::extrapolate_mac;
use crate::grid::{Field3, Grid, MacField, sample_cells, trilinear_corners};
use crate::levelset::redistance;
use crate::particles::{CellIndex, Particle, seed_cell};
use crate::scene::SEED_JITTER;
use crate::transfer::{GridTransfer, Transfer, g2p, p2g};

pub const PARTICLES_PER_CELL: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NarrowBand {
    pub band_cells: f64,
}

impl Default for NarrowBand {
    fn default() -> Self {
        Self { band_cells: 3.0 }
    }
}

impl NarrowBand {
    pub fn depth(&self, grid: &Grid) -> f64 {
        self.band_cells * grid.dx
    }

    pub fn combination_depth(&self, grid: &Grid) -> f64 {
        (self.band_cells - 1.0) * grid.dx
    }
}

fn backtrace(grid: &Grid, velocity: &MacField, x: DVec3, dt: f64) -> DVec3 {
    let k1 = velocity.sample(grid, x);
    let k2 = velocity.sample(grid, x - 0.5 * dt * k1);
    let k3 = velocity.sample(grid, x - 0.5 * dt * k2);
    let k4 = velocity.sample(grid, x - dt * k3);

    x - dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
}

pub fn advect_level_set(grid: &Grid, phi: &Field3, velocity: &MacField, dt: f64) -> Field3 {
    let mut out = phi.clone();

    out.data = exec::map(phi.data.len(), |n| {
        let x = grid.cell_center(phi.coords(n));
        sample_cells(phi, grid, backtrace(grid, velocity, x, dt))
    });

    out
}

fn trace_faces(
    grid: &Grid,
    field: &MacField,
    velocity: &MacField,
    dt: f64,
) -> [Vec<(f64, f64, f64)>; 3] {
    std::array::from_fn(|axis| {
        let source = &field.axes[axis];

        exec::map(source.data.len(), |n| {
            let x = grid.face_position(axis, source.coords(n));
            let g = backtrace(grid, velocity, x, dt) / grid.dx - Grid::face_offset(axis);
            let corners = trilinear_corners(source, g);

            corners.iter().fold(
                (0.0, f64::INFINITY, f64::NEG_INFINITY),
                |(v, lo, hi), (w, u)| (v + w * u, lo.min(*u), hi.max(*u)),
            )
        })
    })
}

pub fn advect_velocity(grid: &Grid, velocity: &MacField, dt: f64) -> MacField {
    let forward = trace_faces(grid, velocity, velocity, dt);
    let mut predicted = MacField::new(grid);

    for (axis, traced) in forward.iter().enumerate() {
        predicted.axes[axis].data = traced.iter().map(|t| t.0).collect();
    }

    let backward = trace_faces(grid, &predicted, velocity, -dt);
    let mut out = MacField::new(grid);

    for axis in 0..3 {
        let original = &velocity.axes[axis].data;

        out.axes[axis].data = exec::map(original.len(), |n| {
            let (ahead, lo, hi) = forward[axis][n];
            let corrected = ahead + 0.5 * (original[n] - backward[axis][n].0);

            corrected.clamp(lo, hi)
        });
    }

    out
}

pub fn combine_level_sets(grid: &Grid, advected: &Field3, particle_phi: &Field3) -> Field3 {
    let mut phi = particle_phi.clone();

    for (p, a) in phi.data.iter_mut().zip(&advected.data) {
        *p = p.min(a + grid.dx);
    }

    redistance(&phi, grid.dx)
}

fn face_phi(phi: &Field3, axis: usize, f: UVec3) -> f64 {
    let hi = f.as_ivec3();
    let mut lo = hi;
    lo[axis] -= 1;

    match (phi.at(lo), phi.at(hi)) {
        (Some(a), Some(b)) => 0.5 * (a + b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => unreachable!(),
    }
}

pub fn combine_velocity(
    grid: &Grid,
    phi: &Field3,
    advected: &MacField,
    transfer: &GridTransfer,
    band: NarrowBand,
) -> (MacField, MacField) {
    let r = band.combination_depth(grid);
    let mut velocity = MacField::new(grid);
    let mut valid = MacField::new(grid);

    for axis in 0..3 {
        let mass = &transfer.mass.axes[axis];
        let particle = &transfer.velocity.axes[axis].data;
        let grid_velocity = &advected.axes[axis].data;

        let picked: Vec<Option<f64>> = exec::map(mass.data.len(), |n| {
            let depth = face_phi(phi, axis, mass.coords(n));

            if mass.data[n] > 0.0 && depth >= -r {
                Some(particle[n])
            } else if depth < 0.0 {
                Some(grid_velocity[n])
            } else {
                None
            }
        });

        for (n, v) in picked.into_iter().enumerate() {
            if let Some(v) = v {
                velocity.axes[axis].data[n] = v;
                valid.axes[axis].data[n] = 1.0;
            }
        }
    }

    (velocity, valid)
}

pub fn initial_velocity(grid: &Grid, particles: &[Particle], layers: usize) -> MacField {
    let index = CellIndex::build(grid, particles);
    let transfer = p2g(grid, particles, &index);
    let mut velocity = transfer.velocity;
    let mut valid = transfer.mass;

    extrapolate_mac(&mut velocity, &mut valid, layers);

    velocity
}

pub struct Resample<'a> {
    pub grid: &'a Grid,

    pub phi: &'a Field3,

    pub velocity: &'a MacField,

    pub transfer: Transfer,

    pub band: NarrowBand,

    pub salt: u32,
}

impl Resample<'_> {
    pub fn apply(&self, particles: &mut Vec<Particle>) {
        let deep = self.band.depth(self.grid);

        particles.retain(|p| sample_cells(self.phi, self.grid, p.x) >= -deep);
        self.thin_crowded_cells(particles);

        let index = CellIndex::build(self.grid, particles);
        let mut fresh: Vec<Particle> = (0..self.grid.cell_count())
            .flat_map(|n| self.seed(n, &index, particles))
            .collect();

        g2p(self.grid, self.velocity, &mut fresh, self.transfer);
        particles.extend(fresh);
    }

    fn in_band(&self, x: DVec3) -> bool {
        let depth = -sample_cells(self.phi, self.grid, x);

        (self.grid.dx..=self.band.depth(self.grid)).contains(&depth)
    }

    fn resampled(&self, n: usize) -> bool {
        self.in_band(self.grid.cell_center(self.phi.coords(n)))
    }

    fn thin_crowded_cells(&self, particles: &mut Vec<Particle>) {
        let index = CellIndex::build(self.grid, particles);
        let mut keep = vec![true; particles.len()];

        for n in (0..self.grid.cell_count()).filter(|n| self.resampled(*n)) {
            for &pi in index.cell(n).iter().skip(2 * PARTICLES_PER_CELL) {
                keep[pi as usize] = false;
            }
        }

        let mut flags = keep.into_iter();
        particles.retain(|_| flags.next().unwrap_or(true));
    }

    fn seed(&self, n: usize, index: &CellIndex, particles: &[Particle]) -> Vec<Particle> {
        let cell = self.phi.coords(n);
        let members = index.cell(n);

        if members.len() >= PARTICLES_PER_CELL || !self.resampled(n) {
            return Vec::new();
        }

        let occupied = members.iter().fold(0u32, |mask, &pi| {
            mask | 1 << octant(self.grid, cell, particles[pi as usize].x)
        });

        seed_cell(self.grid, cell, SEED_JITTER, self.salt)
            .into_iter()
            .enumerate()
            .filter(|(s, x)| occupied & (1 << s) == 0 && self.in_band(*x))
            .take(PARTICLES_PER_CELL - members.len())
            .map(|(_, x)| Particle::at_rest(x))
            .collect()
    }
}

fn octant(grid: &Grid, cell: UVec3, x: DVec3) -> u32 {
    let local = (x / grid.dx - cell.as_dvec3()) * 2.0;
    let o = local.floor().as_ivec3().clamp(IVec3::ZERO, IVec3::ONE);

    (o.x + 2 * o.y + 4 * o.z) as u32
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VolumeControl {
    pub target: f64,

    pub integral: f64,
}

impl VolumeControl {
    pub fn new(target: f64) -> Self {
        Self {
            target,
            integral: 0.0,
        }
    }

    pub fn divergence(&mut self, volume: f64, dt: f64) -> f64 {
        let x = (volume - self.target) / self.target;
        let kp = 2.3 / (25.0 * dt);
        let ki = (kp / 8.0).powi(2);

        self.integral += x * dt;

        -(kp * x + ki * self.integral) / (x + 1.0)
    }
}
