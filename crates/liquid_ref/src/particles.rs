use glam::{DVec3, UVec3};

use crate::grid::Grid;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Particle {
    pub x: DVec3,

    pub v: DVec3,

    pub c: [DVec3; 3],
}

impl Particle {
    pub fn at_rest(x: DVec3) -> Self {
        Self {
            x,
            v: DVec3::ZERO,
            c: [DVec3::ZERO; 3],
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CellIndex {
    pub start: Vec<u32>,

    pub count: Vec<u32>,

    pub order: Vec<u32>,
}

impl CellIndex {
    pub fn build(grid: &Grid, particles: &[Particle]) -> Self {
        let cells = grid.cell_count();
        let keys: Vec<usize> = particles.iter().map(|p| cell_key(grid, p.x)).collect();

        let mut count = vec![0u32; cells];

        for &k in &keys {
            count[k] += 1;
        }

        let mut start = vec![0u32; cells];
        let mut running = 0u32;

        for c in 0..cells {
            start[c] = running;
            running += count[c];
        }

        let mut cursor = start.clone();
        let mut order = vec![0u32; particles.len()];

        for (p, &k) in keys.iter().enumerate() {
            order[cursor[k] as usize] = p as u32;
            cursor[k] += 1;
        }

        Self {
            start,
            count,
            order,
        }
    }

    pub fn cell(&self, n: usize) -> &[u32] {
        let s = self.start[n] as usize;

        &self.order[s..s + self.count[n] as usize]
    }
}

pub fn cell_key(grid: &Grid, x: DVec3) -> usize {
    let c = grid
        .cell_of(x)
        .clamp(glam::IVec3::ZERO, grid.dims.as_ivec3() - 1)
        .as_uvec3();

    (c.x + grid.dims.x * (c.y + grid.dims.y * c.z)) as usize
}

pub fn hash_u32(mut s: u32) -> u32 {
    s ^= 2747636419;
    s = s.wrapping_mul(2654435769);
    s ^= s >> 16;
    s = s.wrapping_mul(2654435769);
    s ^= s >> 16;
    s.wrapping_mul(2654435769)
}

pub fn hash_unit(seed: u32) -> f64 {
    hash_u32(seed) as f64 / u32::MAX as f64
}

pub fn jitter(cell: UVec3, sub: u32, salt: u32) -> DVec3 {
    let seed = cell.x.wrapping_mul(73856093)
        ^ cell.y.wrapping_mul(19349663)
        ^ cell.z.wrapping_mul(83492791)
        ^ sub.wrapping_mul(2654435761)
        ^ salt;

    DVec3::new(
        hash_unit(seed),
        hash_unit(seed ^ 0x9E37_79B9),
        hash_unit(seed ^ 0x7F4A_7C15),
    )
}

pub fn seed_cell(grid: &Grid, cell: UVec3, jitter_amount: f64, salt: u32) -> [DVec3; 8] {
    let mut out = [DVec3::ZERO; 8];

    for (s, slot) in out.iter_mut().enumerate() {
        let sub = DVec3::new((s & 1) as f64, ((s >> 1) & 1) as f64, ((s >> 2) & 1) as f64);
        let j = (jitter(cell, s as u32, salt) - 0.5) * jitter_amount;
        let local = (sub + 0.5 + j) * 0.5;

        *slot = (cell.as_dvec3() + local) * grid.dx;
    }

    out
}

pub fn seed_box(grid: &Grid, lo: DVec3, hi: DVec3, jitter_amount: f64) -> Vec<Particle> {
    let mut out = Vec::new();

    for k in 0..grid.dims.z {
        for j in 0..grid.dims.y {
            for i in 0..grid.dims.x {
                let cell = UVec3::new(i, j, k);

                for x in seed_cell(grid, cell, jitter_amount, 0) {
                    if x.cmpge(lo).all() && x.cmplt(hi).all() {
                        out.push(Particle::at_rest(x));
                    }
                }
            }
        }
    }

    out
}
