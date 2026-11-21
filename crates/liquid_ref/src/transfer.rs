use glam::{DMat3, DVec3, IVec3, UVec3};

use crate::grid::{Field3, Grid, MacField};
use crate::kernel::{APIC_D_INV_SCALE, Stencil, bspline2_weight};
use crate::particles::{CellIndex, Particle};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer {
    Apic,
    Pic,
}

pub struct GridTransfer {
    pub velocity: MacField,

    pub mass: MacField,
}

pub fn p2g(grid: &Grid, particles: &[Particle], index: &CellIndex) -> GridTransfer {
    let mut velocity = MacField::new(grid);
    let mut mass = MacField::new(grid);

    for axis in 0..3 {
        let (v, m) = p2g_axis(grid, particles, index, axis);
        velocity.axes[axis] = v;
        mass.axes[axis] = m;
    }

    GridTransfer { velocity, mass }
}

fn p2g_axis(
    grid: &Grid,
    particles: &[Particle],
    index: &CellIndex,
    axis: usize,
) -> (Field3, Field3) {
    let dims = grid.face_dims(axis);
    let mut velocity = Field3::new(dims, 0.0);
    let mut mass = Field3::new(dims, 0.0);
    let offset = Grid::face_offset(axis);

    let gathered: Vec<(f64, f64)> = crate::exec::map(velocity.data.len(), |n| {
        let f = velocity.coords(n);
        gather_face(grid, particles, index, axis, f, offset)
    });

    for (n, (mom, m)) in gathered.into_iter().enumerate() {
        mass.data[n] = m;
        velocity.data[n] = if m > 0.0 { mom / m } else { 0.0 };
    }

    (velocity, mass)
}

fn gather_face(
    grid: &Grid,
    particles: &[Particle],
    index: &CellIndex,
    axis: usize,
    f: UVec3,
    offset: DVec3,
) -> (f64, f64) {
    let node = f.as_dvec3();
    let x_node = (node + offset) * grid.dx;

    let mut lo = f.as_ivec3() - 1;
    let mut hi = f.as_ivec3() + 1;
    lo[axis] -= 1;
    lo = lo.max(IVec3::ZERO);
    hi = hi.min(grid.dims.as_ivec3() - 1);

    let mut mom = 0.0;
    let mut m = 0.0;

    for cz in lo.z..=hi.z {
        for cy in lo.y..=hi.y {
            for cx in lo.x..=hi.x {
                let cell =
                    (cx as u32 + grid.dims.x * (cy as u32 + grid.dims.y * cz as u32)) as usize;

                for &pi in index.cell(cell) {
                    let p = &particles[pi as usize];
                    let g = p.x / grid.dx - offset;
                    let w = bspline2_weight(g - node);

                    if w > 0.0 {
                        m += w;
                        mom += w * (p.v[axis] + p.c[axis].dot(x_node - p.x));
                    }
                }
            }
        }
    }

    (mom, m)
}

pub fn g2p(grid: &Grid, velocity: &MacField, particles: &mut [Particle], transfer: Transfer) {
    crate::exec::each_mut(particles, |_, p| {
        for axis in 0..3 {
            let (v, c) = interpolate_axis(grid, &velocity.axes[axis], axis, p.x);
            p.v[axis] = v;
            p.c[axis] = match transfer {
                Transfer::Apic => c,
                Transfer::Pic => DVec3::ZERO,
            };
        }
    });
}

pub fn interpolate_axis(grid: &Grid, field: &Field3, axis: usize, x: DVec3) -> (f64, DVec3) {
    let offset = Grid::face_offset(axis);
    let g = x / grid.dx - offset;
    let stencil = Stencil::new(g);

    let mut samples = [(0.0, 0.0, DVec3::ZERO); 27];
    let mut count = 0;
    let mut v = 0.0;
    let mut wsum = 0.0;

    for o in Stencil::offsets() {
        let node = stencil.base + o;

        let Some(value) = field.at(node) else {
            continue;
        };

        let w = stencil.weight(o);
        let r = (node.as_dvec3() + offset) * grid.dx - x;

        samples[count] = (w, value, r);
        count += 1;
        wsum += w;
        v += w * value;
    }

    if wsum <= 0.0 {
        return (0.0, DVec3::ZERO);
    }

    v /= wsum;

    if count == 27 {
        let b = samples
            .iter()
            .fold(DVec3::ZERO, |acc, (w, u, r)| acc + *w * *u * *r);

        return (v, b * APIC_D_INV_SCALE / (grid.dx * grid.dx));
    }

    let mut b = DVec3::ZERO;
    let mut d = DMat3::ZERO;

    for (w, u, r) in &samples[..count] {
        b += *w * (*u - v) * *r;
        d += DMat3::from_cols(*r * r.x, *r * r.y, *r * r.z) * *w;
    }

    if d.determinant().abs() < 1e-12 * grid.dx.powi(6) {
        return (v, DVec3::ZERO);
    }

    (v, d.inverse().transpose() * b)
}
