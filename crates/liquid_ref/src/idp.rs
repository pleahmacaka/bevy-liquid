use glam::{DVec3, IVec3};

use crate::extrapolate::extrapolate_mac;
use crate::grid::{Field3, Grid, MacField};
use crate::kernel::bspline2_weight;
use crate::levelset::solid_phi_at_cell;
use crate::particles::{CellIndex, Particle};
use crate::pressure::{Level, Multigrid, PressureSystem, SolveStats, apply_gradient, pcg};

pub const PARTICLES_PER_CELL: f64 = 8.0;

pub const KERNEL_CELL_SHARE: [f64; 3] = [0.156_25, 0.6875, 0.156_25];

pub fn particle_density(
    grid: &Grid,
    level: &Level,
    particles: &[Particle],
    index: &CellIndex,
    fluid: &[f64],
) -> Vec<f64> {
    crate::exec::map(level.len(), |n| {
        if !level.active[n] {
            return 0.0;
        }

        let c = level.coords(n).as_ivec3();
        let lo = (c - 1).max(IVec3::ZERO);
        let hi = (c + 1).min(grid.dims.as_ivec3() - 1);
        let node = c.as_dvec3();
        let mut sum = 0.0;

        for cz in lo.z..=hi.z {
            for cy in lo.y..=hi.y {
                for cx in lo.x..=hi.x {
                    let cell = level.index(IVec3::new(cx, cy, cz).as_uvec3());

                    for &pi in index.cell(cell) {
                        let g = particles[pi as usize].x / grid.dx - 0.5;
                        sum += bspline2_weight(g - node);
                    }
                }
            }
        }

        match kernel_coverage(level, fluid, c) {
            Some(coverage) => sum / (PARTICLES_PER_CELL * coverage),
            None => 1.0,
        }
    })
}

pub fn fluid_fraction(grid: &Grid, level: &Level, nodes: &Field3) -> Vec<f64> {
    crate::exec::map(level.len(), |n| {
        let phi = solid_phi_at_cell(nodes, level.coords(n));

        (0.5 + phi / grid.dx).clamp(0.0, 1.0)
    })
}

pub fn kernel_coverage(level: &Level, fluid: &[f64], c: IVec3) -> Option<f64> {
    let mut sum = 0.0;

    for o in 0..27 {
        let d = IVec3::new(o % 3, (o / 3) % 3, o / 9);
        let p = c + d - 1;

        if p.cmplt(IVec3::ZERO).any() || p.cmpge(level.dims.as_ivec3()).any() {
            continue;
        }

        let k = KERNEL_CELL_SHARE[d.x as usize]
            * KERNEL_CELL_SHARE[d.y as usize]
            * KERNEL_CELL_SHARE[d.z as usize];
        let f = fluid[level.index(p.as_uvec3())];

        if f < 1.0 {
            return None;
        }

        sum += k;
    }

    Some(sum)
}

pub fn density_error(density: &[f64], level: &Level) -> Vec<f64> {
    (0..level.len())
        .map(|n| {
            if !level.active[n] {
                return 0.0;
            }

            let e = density[n] - 1.0;

            if is_interior(level, n) { e } else { e.max(0.0) }
        })
        .collect()
}

pub fn is_interior(level: &Level, n: usize) -> bool {
    let c = level.coords(n).as_ivec3();

    (0..27).all(|o| {
        let p = c + IVec3::new(o % 3, (o / 3) % 3, o / 9) - 1;
        let inside = p.cmpge(IVec3::ZERO).all() && p.cmplt(level.dims.as_ivec3()).all();

        !inside || level.active[level.index(p.as_uvec3())]
    })
}

pub struct Projection {
    pub displacement: MacField,

    pub stats: SolveStats,
}

pub fn solve_displacement(
    grid: &Grid,
    multigrid: &Multigrid,
    weights: &MacField,
    error: &[f64],
    tolerance: f64,
    max_iterations: usize,
    layers: usize,
) -> Projection {
    let system = PressureSystem {
        multigrid,
        couplings: &[],
        bodies: &[],
        coupling_scale: 0.0,
    };

    let mut q = vec![0.0; error.len()];
    let stats = pcg(&system, error, &mut q, tolerance, max_iterations, true);

    let mut displacement = MacField::new(grid);
    let mut valid = apply_gradient(
        grid,
        &multigrid.levels[0],
        &q,
        1.0 / grid.dx,
        &mut displacement,
        weights,
    );

    extrapolate_mac(&mut displacement, &mut valid, layers);

    for (d, w) in displacement.axes.iter_mut().zip(&weights.axes) {
        for (v, w) in d.data.iter_mut().zip(&w.data) {
            if *w <= 0.0 {
                *v = 0.0;
            }
        }
    }

    Projection {
        displacement,
        stats,
    }
}

pub fn displace(grid: &Grid, displacement: &MacField, particles: &mut [Particle]) {
    crate::exec::each_mut(particles, |_, p| {
        let d: DVec3 = displacement.sample(grid, p.x);
        p.x += d;
    });
}
