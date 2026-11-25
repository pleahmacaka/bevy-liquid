use std::array::from_fn;

use glam::{DVec3, IVec3, UVec3};

use crate::exec;
use crate::grid::{Field3, Grid, MacField, sample_cells};
use crate::pressure::{SolveStats, dot, max_abs};

pub const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

pub struct Fractions {
    pub cells: Field3,

    pub faces: MacField,

    pub edges: [Field3; 3],
}

impl Fractions {
    pub fn new(grid: &Grid, phi: &Field3) -> Self {
        Self::weighted(grid, phi, |_| 1.0)
    }

    pub fn weighted(grid: &Grid, phi: &Field3, weight: impl Fn(f64) -> f64 + Sync) -> Self {
        let stress = |dims: UVec3, offset: DVec3| sampled(grid, phi, dims, offset, &weight);

        Self {
            cells: stress(grid.dims, DVec3::splat(0.5)),
            faces: MacField {
                axes: from_fn(|a| {
                    sampled(grid, phi, grid.face_dims(a), Grid::face_offset(a), &|_| 1.0)
                }),
            },
            edges: from_fn(|e| {
                let (a, b) = PAIRS[e];
                stress(edge_dims(grid, a, b), edge_offset(a, b))
            }),
        }
    }

    pub fn blend(&self, other: &Fractions, ratio: f64) -> Self {
        let mix = |a: &Field3, b: &Field3| Field3 {
            dims: a.dims,
            data: a
                .data
                .iter()
                .zip(&b.data)
                .map(|(x, y)| x + ratio * y)
                .collect(),
        };

        Self {
            cells: mix(&self.cells, &other.cells),
            faces: self.faces.clone(),
            edges: from_fn(|e| mix(&self.edges[e], &other.edges[e])),
        }
    }
}

pub fn edge_dims(grid: &Grid, a: usize, b: usize) -> UVec3 {
    let mut d = grid.dims;
    d[a] += 1;
    d[b] += 1;

    d
}

fn edge_offset(a: usize, b: usize) -> DVec3 {
    let mut o = DVec3::splat(0.5);
    o[a] = 0.0;
    o[b] = 0.0;

    o
}

fn sampled(
    grid: &Grid,
    phi: &Field3,
    dims: UVec3,
    offset: DVec3,
    weight: &(impl Fn(f64) -> f64 + Sync),
) -> Field3 {
    let mut field = Field3::new(dims, 0.0);

    field.data = exec::map(field.data.len(), |n| {
        let x = (field.coords(n).as_dvec3() + offset) * grid.dx;
        liquid_fraction(grid, phi, x) * weight(sample_cells(phi, grid, x))
    });

    field
}

pub fn liquid_fraction(grid: &Grid, phi: &Field3, x: DVec3) -> f64 {
    let h = 0.5 * grid.dx;

    let sum: f64 = (0..8)
        .map(|corner| {
            let o = DVec3::new(
                (corner & 1) as f64,
                ((corner >> 1) & 1) as f64,
                ((corner >> 2) & 1) as f64,
            ) - 0.5;

            (0.5 - sample_cells(phi, grid, x + o * h) / h).clamp(0.0, 1.0)
        })
        .sum();

    sum / 8.0
}

fn unit(axis: usize) -> IVec3 {
    let mut e = IVec3::ZERO;
    e[axis] = 1;

    e
}

fn pair(a: usize, b: usize) -> usize {
    a + b - 1
}

fn value(field: &Field3, p: IVec3) -> f64 {
    field.at(p).unwrap_or(0.0)
}

pub struct ViscousSystem<'a> {
    pub grid: Grid,

    pub fractions: &'a Fractions,

    pub scale: f64,

    pub mass: Vec<f64>,

    pub diagonal: Vec<f64>,
}

impl<'a> ViscousSystem<'a> {
    pub fn new(
        grid: &Grid,
        fractions: &'a Fractions,
        weights: &MacField,
        coefficient: f64,
    ) -> Self {
        let scale = coefficient / (grid.dx * grid.dx);
        let mut diagonal = MacField::new(grid);

        for a in 0..3 {
            let w = &weights.axes[a];

            diagonal.axes[a].data = exec::map(w.data.len(), |n| {
                if w.data[n] <= 0.0 {
                    return 0.0;
                }

                let f = w.coords(n).as_ivec3();
                let cells = &fractions.cells;
                let mut stress = 2.0 * (value(cells, f - unit(a)) + value(cells, f));

                for b in (0..3).filter(|b| *b != a) {
                    let edges = &fractions.edges[pair(a, b)];
                    stress += value(edges, f) + value(edges, f + unit(b));
                }

                fractions.faces.axes[a].data[n] + scale * stress
            });
        }

        Self {
            grid: *grid,
            mass: flatten(&fractions.faces),
            fractions,
            scale,
            diagonal: flatten(&diagonal),
        }
    }

    pub fn stress_divergence(&self, u: &MacField) -> MacField {
        let cells = &self.fractions.cells;

        let normal: [Field3; 3] = from_fn(|a| {
            let mut s = cells.clone();

            s.data = exec::map(cells.data.len(), |n| {
                let c = cells.coords(n).as_ivec3();
                2.0 * cells.data[n] * (value(&u.axes[a], c + unit(a)) - value(&u.axes[a], c))
            });

            s
        });

        let shear: [Field3; 3] = from_fn(|e| {
            let (a, b) = PAIRS[e];
            let edges = &self.fractions.edges[e];
            let mut s = edges.clone();

            s.data = exec::map(edges.data.len(), |n| {
                let c = edges.coords(n).as_ivec3();
                let along_b = value(&u.axes[a], c) - value(&u.axes[a], c - unit(b));
                let along_a = value(&u.axes[b], c) - value(&u.axes[b], c - unit(a));

                edges.data[n] * (along_b + along_a)
            });

            s
        });

        let mut out = MacField::new(&self.grid);

        for a in 0..3 {
            let field = &u.axes[a];

            out.axes[a].data = exec::map(field.data.len(), |n| {
                let f = field.coords(n).as_ivec3();
                let mut v = value(&normal[a], f - unit(a)) - value(&normal[a], f);

                for b in (0..3).filter(|b| *b != a) {
                    let edges = &shear[pair(a, b)];
                    v += value(edges, f) - value(edges, f + unit(b));
                }

                v
            });
        }

        out
    }

    fn apply(&self, x: &[f64]) -> Vec<f64> {
        let stress = flatten(&self.stress_divergence(&unflatten(&self.grid, x)));

        exec::map(x.len(), |n| {
            if self.diagonal[n] > 0.0 {
                self.mass[n] * x[n] + self.scale * stress[n]
            } else {
                0.0
            }
        })
    }
}

pub fn flatten(m: &MacField) -> Vec<f64> {
    m.axes.iter().flat_map(|a| a.data.iter().copied()).collect()
}

pub fn unflatten(grid: &Grid, v: &[f64]) -> MacField {
    let mut m = MacField::new(grid);
    let mut start = 0;

    for a in &mut m.axes {
        let len = a.data.len();
        a.data.copy_from_slice(&v[start..start + len]);
        start += len;
    }

    m
}

pub struct ViscositySolve<'a> {
    pub grid: &'a Grid,

    pub fractions: &'a Fractions,

    pub weights: &'a MacField,

    pub kinematic_viscosity: f64,

    pub dt: f64,

    pub tolerance: f64,

    pub max_iterations: usize,
}

impl ViscositySolve<'_> {
    pub fn apply(&self, velocity: &mut MacField) -> SolveStats {
        let system = ViscousSystem::new(
            self.grid,
            self.fractions,
            self.weights,
            self.dt * self.kinematic_viscosity,
        );
        let diagonal = &system.diagonal;
        let current = flatten(velocity);

        let fixed: Vec<f64> = current
            .iter()
            .zip(diagonal)
            .map(|(v, d)| if *d > 0.0 { 0.0 } else { *v })
            .collect();
        let mut x: Vec<f64> = current
            .iter()
            .zip(diagonal)
            .map(|(v, d)| if *d > 0.0 { *v } else { 0.0 })
            .collect();

        let coupling = flatten(&system.stress_divergence(&unflatten(self.grid, &fixed)));
        let b: Vec<f64> = (0..x.len())
            .map(|n| {
                if diagonal[n] > 0.0 {
                    system.mass[n] * current[n] - system.scale * coupling[n]
                } else {
                    0.0
                }
            })
            .collect();

        let stats = conjugate_gradient(
            |v| system.apply(v),
            diagonal,
            &b,
            &mut x,
            self.tolerance,
            self.max_iterations,
        );

        let solved: Vec<f64> = x.iter().zip(&fixed).map(|(x, f)| x + f).collect();
        *velocity = unflatten(self.grid, &solved);

        stats
    }
}

pub fn conjugate_gradient(
    apply: impl Fn(&[f64]) -> Vec<f64>,
    diagonal: &[f64],
    b: &[f64],
    x: &mut [f64],
    tolerance: f64,
    max_iterations: usize,
) -> SolveStats {
    let precondition = |r: &[f64]| -> Vec<f64> {
        r.iter()
            .zip(diagonal)
            .map(|(r, d)| if *d > 0.0 { r / d } else { 0.0 })
            .collect()
    };

    let ax = apply(x);
    let mut r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
    let initial_residual = max_abs(&r);
    let threshold = tolerance * max_abs(b).max(f64::MIN_POSITIVE);

    let mut stats = SolveStats {
        iterations: 0,
        residual: initial_residual,
        initial_residual,
    };

    if initial_residual <= threshold {
        return stats;
    }

    let mut z = precondition(&r);
    let mut d = z.clone();
    let mut rz = dot(&r, &z);

    for it in 1..=max_iterations {
        let q = apply(&d);
        let alpha = rz / dot(&d, &q);

        for n in 0..x.len() {
            x[n] += alpha * d[n];
            r[n] -= alpha * q[n];
        }

        stats.iterations = it;
        stats.residual = max_abs(&r);

        if stats.residual <= threshold {
            break;
        }

        z = precondition(&r);

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;

        for n in 0..x.len() {
            d[n] = z[n] + beta * d[n];
        }
    }

    stats
}
