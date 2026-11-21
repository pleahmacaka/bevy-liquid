use glam::{DVec3, IVec3, UVec3};
use rayon::prelude::*;

use crate::grid::{Field3, Grid, MacField};
use crate::rigid::{RigidBody, apply_inverse_mass};
use crate::tall::TallColumn;

pub const THETA_MIN: f64 = 0.01;

pub const COARSE_REGULARIZATION: f64 = 1e-6;

pub const PARALLEL_MIN: usize = 1 << 15;

#[derive(Clone, Debug)]
pub struct Level {
    pub dims: UVec3,

    pub phi: Vec<f64>,

    pub coef: [Vec<f64>; 3],

    pub active: Vec<bool>,

    pub diag: Vec<f64>,

    pub off: [Vec<f64>; 3],

    pub floor: Vec<f64>,

    pub ghost: Vec<f64>,

    pub tall: Vec<TallColumn>,
}

impl Level {
    pub fn len(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize
    }

    pub fn columns(&self) -> usize {
        (self.dims.x * self.dims.z) as usize
    }

    pub fn column(&self, c: UVec3) -> usize {
        (c.x + self.dims.x * c.z) as usize
    }

    pub fn floor_cell(&self, col: usize) -> usize {
        let col = col as u32;

        self.index(UVec3::new(col % self.dims.x, 0, col / self.dims.x))
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn index(&self, c: UVec3) -> usize {
        (c.x + self.dims.x * (c.y + self.dims.y * c.z)) as usize
    }

    pub fn coords(&self, n: usize) -> UVec3 {
        let n = n as u32;

        UVec3::new(
            n % self.dims.x,
            (n / self.dims.x) % self.dims.y,
            n / (self.dims.x * self.dims.y),
        )
    }

    pub fn neighbor(&self, c: UVec3, axis: usize, step: i32) -> Option<usize> {
        let mut p = c.as_ivec3();
        p[axis] += step;

        let inside = p.cmpge(IVec3::ZERO).all() && p.cmplt(self.dims.as_ivec3()).all();

        inside.then(|| self.index(p.as_uvec3()))
    }

    pub fn from_fine(grid: &Grid, phi: &Field3, weights: &MacField, scale: f64) -> Self {
        let dims = grid.dims;
        let len = grid.cell_count();
        let mut coef: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; len]);

        for (axis, k) in coef.iter_mut().enumerate() {
            for (n, slot) in k.iter_mut().enumerate() {
                let c = phi.coords(n);

                if c[axis] + 1 < dims[axis] {
                    let mut f = c;
                    f[axis] += 1;
                    *slot = weights.axes[axis].get(f.x, f.y, f.z) * scale;
                }
            }
        }

        let columns = (dims.x * dims.z) as usize;
        let floor = (0..columns)
            .map(|col| {
                let col = col as u32;

                weights.axes[1].get(col % dims.x, 0, col / dims.x) * scale
            })
            .collect();

        let mut level = Self {
            dims,
            phi: phi.data.clone(),
            coef,
            active: vec![false; len],
            diag: vec![0.0; len],
            off: std::array::from_fn(|_| vec![0.0; len]),
            floor,
            ghost: vec![0.0; columns],
            tall: Vec::new(),
        };

        level.finalize(0.0);

        level
    }

    fn floor_coef(&self, c: UVec3) -> f64 {
        if c.y == 0 {
            self.floor[self.column(c)]
        } else {
            0.0
        }
    }

    fn finalize(&mut self, regularization: f64) {
        let len = self.len();

        for n in 0..len {
            let c = self.coords(n);
            let connected = self.floor_coef(c) > 0.0
                || (0..3).any(|a| {
                    self.coef[a][n] > 0.0
                        || self
                            .neighbor(c, a, -1)
                            .is_some_and(|m| self.coef[a][m] > 0.0)
                });

            self.active[n] = self.phi[n] < 0.0 && connected;
        }

        for n in 0..len {
            self.diag[n] = 0.0;

            for off in &mut self.off {
                off[n] = 0.0;
            }

            if !self.active[n] {
                continue;
            }

            let c = self.coords(n);

            for a in 0..3 {
                for step in [-1, 1] {
                    let Some(m) = self.neighbor(c, a, step) else {
                        continue;
                    };

                    let k = if step > 0 {
                        self.coef[a][n]
                    } else {
                        self.coef[a][m]
                    };

                    if k <= 0.0 {
                        continue;
                    }

                    if self.active[m] {
                        self.diag[n] += k;

                        if step > 0 {
                            self.off[a][n] = k;
                        }
                    } else if self.phi[m] >= 0.0 {
                        let theta = (self.phi[n] / (self.phi[n] - self.phi[m])).max(THETA_MIN);
                        self.diag[n] += k / theta;
                    }
                }
            }

            self.diag[n] += self.floor_coef(c);
            self.diag[n] *= 1.0 + regularization;
        }
    }

    pub fn add_floor_rhs(&self, rhs: &mut [f64]) {
        for col in 0..self.columns() {
            let n = self.floor_cell(col);

            if self.active[n] {
                rhs[n] += self.floor[col] * self.ghost[col];
            }
        }
    }

    pub fn floor_difference(&self, p: &[f64], col: usize) -> Option<f64> {
        let n = self.floor_cell(col);

        (self.floor[col] > 0.0 && self.active[n]).then(|| p[n] - self.ghost[col])
    }

    fn couplings(&self, n: usize) -> usize {
        let c = self.coords(n);

        (0..3)
            .map(|a| {
                let below = self
                    .neighbor(c, a, -1)
                    .is_some_and(|m| self.off[a][m] != 0.0);

                usize::from(self.off[a][n] != 0.0) + usize::from(below)
            })
            .sum()
    }

    pub fn boundary_band(&self, width: u32) -> Vec<bool> {
        let len = self.len();
        let boundary: Vec<bool> = (0..len)
            .map(|n| self.active[n] && self.couplings(n) < 6)
            .collect();
        let reach = width as i32 - 1;

        (0..len)
            .map(|n| {
                if !self.active[n] || width == 0 {
                    return false;
                }

                let c = self.coords(n).as_ivec3();
                let lo = (c - reach).max(IVec3::ZERO);
                let hi = (c + reach).min(self.dims.as_ivec3() - 1);

                (lo.z..=hi.z).any(|z| {
                    (lo.y..=hi.y).any(|y| {
                        (lo.x..=hi.x)
                            .any(|x| boundary[self.index(IVec3::new(x, y, z).as_uvec3())])
                    })
                })
            })
            .collect()
    }

    pub fn coarsen(&self) -> Self {
        let dims = (self.dims + 1) / 2;
        let len = (dims.x * dims.y * dims.z) as usize;
        let mut phi = vec![0.0; len];
        let mut coef: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; len]);

        let columns = (dims.x * dims.z) as usize;
        let mut coarse = Self {
            dims,
            phi: vec![0.0; len],
            coef: std::array::from_fn(|_| vec![0.0; len]),
            active: vec![false; len],
            diag: vec![0.0; len],
            off: std::array::from_fn(|_| vec![0.0; len]),
            floor: vec![0.0; columns],
            ghost: vec![0.0; columns],
            tall: Vec::new(),
        };

        for col in 0..columns {
            let cc = coarse.coords(coarse.floor_cell(col));
            let mut sum = 0.0;

            for o in 0..4u32 {
                let f = cc * 2 + UVec3::new(o & 1, 0, o >> 1);

                if f.cmplt(self.dims).all() {
                    sum += self.floor[self.column(f)];
                }
            }

            coarse.floor[col] = sum / 16.0;
        }

        for (n, slot) in phi.iter_mut().enumerate() {
            let cc = coarse.coords(n);
            let mut sum = 0.0;
            let mut count = 0.0;

            for o in 0..8u32 {
                let f = cc * 2 + UVec3::new(o & 1, (o >> 1) & 1, (o >> 2) & 1);

                if f.cmplt(self.dims).all() {
                    sum += self.phi[self.index(f)];
                    count += 1.0;
                }
            }

            *slot = sum / count;
        }

        for (axis, k) in coef.iter_mut().enumerate() {
            let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);

            for (n, slot) in k.iter_mut().enumerate() {
                let cc = coarse.coords(n);
                let mut sum = 0.0;

                for o in 0..4u32 {
                    let mut f = cc * 2;
                    f[axis] += 1;
                    f[t1] += o & 1;
                    f[t2] += o >> 1;

                    let mut g = f;
                    g[axis] += 1;

                    if g.cmplt(self.dims).all() {
                        sum += self.coef[axis][self.index(f)];
                    }
                }

                *slot = sum / 16.0;
            }
        }

        coarse.phi = phi;
        coarse.coef = coef;
        coarse.finalize(COARSE_REGULARIZATION);

        coarse
    }

    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        fill_cells(y, |n| {
            if !self.unknown(n) {
                return 0.0;
            }

            match self.endpoint(n) {
                Some((col, upper)) => self.tall_row(col, x)[usize::from(upper)],
                None => self.diag[n] * x[n] - self.neighbor_sum(n, x),
            }
        });
    }

    fn neighbor_sum(&self, n: usize, x: &[f64]) -> f64 {
        let c = self.coords(n);
        let mut s = 0.0;

        for a in 0..3 {
            if self.off[a][n] != 0.0 {
                s += self.off[a][n] * self.value(x, self.neighbor(c, a, 1).unwrap_or(n));
            }

            if let Some(m) = self.neighbor(c, a, -1)
                && self.off[a][m] != 0.0
            {
                s += self.off[a][m] * self.value(x, m);
            }
        }

        s
    }

    pub fn smooth(&self, x: &mut [f64], b: &[f64], color: u32) {
        self.relax(x, b, color, &self.active);
    }

    pub fn sweep(&self, x: &mut [f64], b: &[f64], forward: bool) {
        for k in 0..2 {
            let color = if forward { k } else { 1 - k };

            if self.tall.is_empty() {
                self.smooth(x, b, color);
            } else {
                self.relax_lines(x, b, color);
            }
        }
    }

    pub fn relax(&self, x: &mut [f64], b: &[f64], color: u32, cells: &[bool]) {
        if self.len() < PARALLEL_MIN {
            for n in 0..self.len() {
                let c = self.coords(n);

                if cells[n] && (c.x + c.y + c.z) % 2 == color && self.regular(n) {
                    x[n] = (b[n] + self.neighbor_sum(n, x)) / self.diag[n];
                }
            }

            return;
        }

        let updates: Vec<(usize, f64)> = (0..self.len())
            .into_par_iter()
            .filter_map(|n| {
                let c = self.coords(n);

                if !cells[n] || (c.x + c.y + c.z) % 2 != color || !self.regular(n) {
                    return None;
                }

                Some((n, (b[n] + self.neighbor_sum(n, x)) / self.diag[n]))
            })
            .collect();

        for (n, v) in updates {
            x[n] = v;
        }
    }

    pub fn block_smooth(&self, x: &mut [f64], b: &[f64], forward: bool, sweeps: usize) {
        let frozen = x.to_vec();

        for _ in 0..sweeps {
            for k in 0..2 {
                let color = if forward { k } else { 1 - k };

                for n in 0..self.len() {
                    let c = self.coords(n);

                    if self.active[n] && (c.x + c.y + c.z) % 2 == color && self.regular(n) {
                        x[n] = (b[n] + self.block_sum(n, x, &frozen)) / self.diag[n];
                    }
                }
            }
        }
    }

    fn block_sum(&self, n: usize, x: &[f64], frozen: &[f64]) -> f64 {
        let c = self.coords(n);
        let home = c / SMOOTHING_BLOCK;
        let mut s = 0.0;

        for a in 0..3 {
            for side in [-1, 1] {
                let Some(m) = self.neighbor(c, a, side) else {
                    continue;
                };
                let k = if side > 0 {
                    self.off[a][n]
                } else {
                    self.off[a][m]
                };
                let v = if self.coords(m) / SMOOTHING_BLOCK == home {
                    x[m]
                } else {
                    frozen[m]
                };

                s += k * v;
            }
        }

        s
    }
}

pub const SMOOTHING_BLOCK: u32 = 8;

pub const FLOOR_SAMPLES: i32 = 32;

pub fn floor_surface(level: &Level, dx: f64) -> Vec<f64> {
    (0..level.columns())
        .map(|col| {
            let c = level.coords(level.floor_cell(col));
            let phi = |j: u32| level.phi[level.index(UVec3::new(c.x, j, c.z))];

            if phi(0) >= 0.0 {
                return 0.0;
            }

            (1..level.dims.y)
                .find(|&j| phi(j) >= 0.0)
                .map(|j| {
                    let theta = (phi(j - 1) / (phi(j - 1) - phi(j))).max(THETA_MIN);
                    (j as f64 - 0.5 + theta) * dx
                })
                .unwrap_or(level.dims.y as f64 * dx)
        })
        .collect()
}

fn mirror(e: i32, n: i32) -> i32 {
    let m = e.rem_euclid(2 * n);

    if m >= n { 2 * n - 1 - m } else { m }
}

pub fn floor_ghost(level: &Level, dx: f64, reach: f64, hydrostatic: f64) -> Vec<f64> {
    let eta = floor_surface(level, dx);
    let (nx, nz) = (level.dims.x as i32, level.dims.z as i32);
    let mean = eta.iter().sum::<f64>() / eta.len() as f64;

    (0..eta.len())
        .map(|col| {
            let (i, k) = (col as i32 % nx, col as i32 / nx);
            let h = (eta[col] + 0.5 * dx).max(dx);
            let span = (reach * h / dx).ceil() as i32;
            let stride = (span + FLOOR_SAMPLES - 1) / FLOOR_SAMPLES;
            let mut seen = eta[col];

            if span > 0 {
                let count = (span + stride - 1) / stride;
                let area = (stride as f64 * dx).powi(2);
                let (mut sum, mut total) = (0.0, 0.0);

                for mz in -count..=count {
                    for mx in -count..=count {
                        let (ox, oz) = (mx * stride, mz * stride);
                        let q = (ox * ox + oz * oz) as f64 * dx * dx + h * h;
                        let w = h * area / (std::f64::consts::TAU * q * q.sqrt());
                        let src = mirror(i + ox, nx) + nx * mirror(k + oz, nz);

                        sum += w * eta[src as usize];
                        total += w;
                    }
                }

                let tail = (1.0 - total).max(0.0);
                seen = (sum + tail * mean) / (total + tail);
            }

            hydrostatic * (seen + 0.5 * dx)
        })
        .collect()
}

pub const TRANSFER_WEIGHTS: [f64; 4] = [0.25, 0.75, 0.75, 0.25];

pub fn restrict(fine: &Level, coarse: &Level, r: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; coarse.len()];

    let gather = |n: usize| {
        let cc = coarse.coords(n).as_ivec3();
        let mut sum = 0.0;

        for oz in 0..4 {
            for oy in 0..4 {
                for ox in 0..4 {
                    let f = cc * 2 + IVec3::new(ox, oy, oz) - 1;

                    if f.cmplt(IVec3::ZERO).any() || f.cmpge(fine.dims.as_ivec3()).any() {
                        continue;
                    }

                    let w = TRANSFER_WEIGHTS[ox as usize]
                        * TRANSFER_WEIGHTS[oy as usize]
                        * TRANSFER_WEIGHTS[oz as usize];
                    sum += w * r[fine.index(f.as_uvec3())];
                }
            }
        }

        sum / 8.0
    };

    fill_cells(&mut out, |n| {
        if !coarse.unknown(n) {
            return 0.0;
        }

        match coarse.endpoint(n) {
            Some((col, upper)) => coarse.hat_sum(col, upper, gather),
            None => gather(n),
        }
    });

    out
}

pub fn prolong_add(fine: &Level, coarse: &Level, e: &[f64], x: &mut [f64]) {
    let mut correction = vec![0.0; x.len()];

    fill_cells(&mut correction, |n| {
        if !fine.unknown(n) {
            return 0.0;
        }

        let f = fine.coords(n).as_ivec3();
        let half = f / 2;
        let odd = f - half * 2;
        let dir = odd * 2 - 1;

        let mut acc = 0.0;

        for corner in 0..8 {
            let pick = IVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
            let c = half + pick * dir;

            if c.cmplt(IVec3::ZERO).any() || c.cmpge(coarse.dims.as_ivec3()).any() {
                continue;
            }

            let w = DVec3::select(
                pick.cmpeq(IVec3::ONE),
                DVec3::splat(0.25),
                DVec3::splat(0.75),
            );
            acc += w.x * w.y * w.z * coarse.value(e, coarse.index(c.as_uvec3()));
        }

        acc
    });

    for (v, c) in x.iter_mut().zip(correction) {
        *v += c;
    }
}

pub fn fill_cells(out: &mut [f64], f: impl Fn(usize) -> f64 + Sync) {
    if out.len() < PARALLEL_MIN {
        out.iter_mut().enumerate().for_each(|(n, o)| *o = f(n));
    } else {
        out.par_iter_mut().enumerate().for_each(|(n, o)| *o = f(n));
    }
}

#[derive(Clone, Debug)]
pub struct Multigrid {
    pub levels: Vec<Level>,

    pub smoothing: usize,

    pub coarse_sweeps: usize,

    pub boundary_sweeps: usize,

    pub bands: Vec<Vec<bool>>,

    pub block_smoothing: bool,
}

impl Multigrid {
    pub fn new(fine: Level, min_dim: u32) -> Self {
        let mut levels = vec![fine];

        while levels
            .last()
            .is_some_and(|l| l.dims.max_element() > min_dim)
        {
            let next = levels
                .last()
                .map(Level::coarsen)
                .unwrap_or_else(|| unreachable!());
            levels.push(next);
        }

        Self {
            levels,
            smoothing: 2,
            coarse_sweeps: 16,
            boundary_sweeps: 0,
            bands: Vec::new(),
            block_smoothing: false,
        }
    }

    pub fn with_block_smoothing(mut self, on: bool) -> Self {
        self.block_smoothing = on;

        self
    }

    fn blocks(&self) -> bool {
        self.block_smoothing && self.boundary_sweeps == 0 && self.levels[0].tall.is_empty()
    }

    fn smooth(&self, level: &Level, x: &mut [f64], b: &[f64], forward: bool) {
        if self.blocks() {
            level.block_smooth(x, b, forward, self.smoothing);
            return;
        }

        for _ in 0..self.smoothing {
            level.sweep(x, b, forward);
        }
    }

    pub fn with_boundary_smoothing(mut self, width: u32, sweeps: usize) -> Self {
        self.boundary_sweeps = if width > 0 { sweeps } else { 0 };
        self.bands = if self.boundary_sweeps > 0 {
            self.levels.iter().map(|l| l.boundary_band(width)).collect()
        } else {
            Vec::new()
        };

        self
    }

    pub fn precondition(&self, r: &[f64]) -> Vec<f64> {
        self.vcycle(0, r)
    }

    fn vcycle(&self, l: usize, b: &[f64]) -> Vec<f64> {
        let level = &self.levels[l];
        let mut x = vec![0.0; level.len()];

        if l + 1 == self.levels.len() {
            for _ in 0..self.coarse_sweeps {
                level.sweep(&mut x, b, true);
            }

            for _ in 0..self.coarse_sweeps {
                level.sweep(&mut x, b, false);
            }

            return x;
        }

        self.smooth(level, &mut x, b, true);

        for _ in 0..self.boundary_sweeps {
            level.relax(&mut x, b, 0, &self.bands[l]);
            level.relax(&mut x, b, 1, &self.bands[l]);
        }

        let mut ax = vec![0.0; level.len()];
        level.apply(&x, &mut ax);

        let r: Vec<f64> = b.iter().zip(&ax).map(|(b, a)| b - a).collect();
        let coarse = &self.levels[l + 1];
        let bc = restrict(level, coarse, &r);
        let ec = self.vcycle(l + 1, &bc);

        prolong_add(level, coarse, &ec, &mut x);

        for _ in 0..self.boundary_sweeps {
            level.relax(&mut x, b, 1, &self.bands[l]);
            level.relax(&mut x, b, 0, &self.bands[l]);
        }

        self.smooth(level, &mut x, b, false);

        x
    }
}

#[derive(Clone, Debug)]
pub struct BodyCoupling {
    pub body: usize,

    pub rows: Vec<(usize, [f64; 6])>,
}

impl BodyCoupling {
    pub fn gather(&self, p: &[f64]) -> [f64; 6] {
        let mut out = [0.0; 6];

        for (n, j) in &self.rows {
            for (o, v) in out.iter_mut().zip(j) {
                *o += v * p[*n];
            }
        }

        out
    }
}

pub fn body_couplings(
    grid: &Grid,
    level: &Level,
    weights: &MacField,
    owners: &[Vec<i32>; 3],
    bodies: &[RigidBody],
) -> Vec<BodyCoupling> {
    let area = grid.dx * grid.dx;

    bodies
        .iter()
        .enumerate()
        .filter(|(_, b)| b.is_dynamic())
        .map(|(bi, body)| {
            let mut rows = Vec::new();

            for n in 0..level.len() {
                if !level.active[n] {
                    continue;
                }

                let c = level.coords(n);
                let mut j = [0.0; 6];
                let mut touched = false;

                for (axis, w_axis) in weights.axes.iter().enumerate() {
                    for (high, sign) in [(0u32, -1.0), (1u32, 1.0)] {
                        let mut f = c;
                        f[axis] += high;

                        let fi = w_axis.index(f.x, f.y, f.z);

                        if owners[axis][fi] != bi as i32 {
                            continue;
                        }

                        let open = 1.0 - w_axis.data[fi];

                        if open <= 0.0 {
                            continue;
                        }

                        let mut e = DVec3::ZERO;
                        e[axis] = 1.0;

                        let r = grid.face_position(axis, f) - body.position;
                        let torque = r.cross(e);
                        let s = sign * open * area;

                        for d in 0..3 {
                            j[d] += s * e[d];
                            j[d + 3] += s * torque[d];
                        }

                        touched = true;
                    }
                }

                if touched {
                    rows.push((n, j));
                }
            }

            BodyCoupling { body: bi, rows }
        })
        .collect()
}

pub struct PressureSystem<'a> {
    pub multigrid: &'a Multigrid,

    pub couplings: &'a [BodyCoupling],

    pub bodies: &'a [RigidBody],

    pub coupling_scale: f64,
}

impl PressureSystem<'_> {
    pub fn level(&self) -> &Level {
        &self.multigrid.levels[0]
    }

    pub fn apply(&self, x: &[f64], y: &mut [f64]) {
        self.level().apply(x, y);

        for c in self.couplings {
            let gathered = c.gather(x);
            let response = apply_inverse_mass(&self.bodies[c.body], gathered);

            for (n, j) in &c.rows {
                let dot: f64 = j.iter().zip(&response).map(|(a, b)| a * b).sum();
                y[*n] += self.coupling_scale * dot;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolveStats {
    pub iterations: usize,

    pub residual: f64,

    pub initial_residual: f64,
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn max_abs(a: &[f64]) -> f64 {
    a.iter().fold(0.0, |m, v| m.max(v.abs()))
}

pub fn pcg(
    system: &PressureSystem,
    b: &[f64],
    x: &mut [f64],
    tolerance: f64,
    max_iterations: usize,
    precondition: bool,
) -> SolveStats {
    let len = b.len();
    let level = system.level();

    for (n, v) in x.iter_mut().enumerate() {
        if !level.unknown(n) {
            *v = 0.0;
        }
    }

    let mut q = vec![0.0; len];
    system.apply(x, &mut q);

    let mut r: Vec<f64> = b.iter().zip(&q).map(|(b, q)| b - q).collect();
    let initial_residual = level.residual_norm(&r);
    let threshold = tolerance * level.residual_norm(b).max(f64::MIN_POSITIVE);

    let mut stats = SolveStats {
        iterations: 0,
        residual: initial_residual,
        initial_residual,
    };

    if initial_residual <= threshold {
        return stats;
    }

    let apply_m = |r: &[f64]| -> Vec<f64> {
        if precondition {
            system.multigrid.precondition(r)
        } else {
            r.to_vec()
        }
    };

    let mut z = apply_m(&r);
    let mut d = z.clone();
    let mut rz = dot(&r, &z);

    for it in 1..=max_iterations {
        system.apply(&d, &mut q);

        let alpha = rz / dot(&d, &q);

        for n in 0..len {
            x[n] += alpha * d[n];
            r[n] -= alpha * q[n];
        }

        stats.iterations = it;
        stats.residual = level.residual_norm(&r);

        if stats.residual <= threshold {
            break;
        }

        z = apply_m(&r);

        let rz_new = dot(&r, &z);
        let beta = rz_new / rz;
        rz = rz_new;

        for n in 0..len {
            d[n] = z[n] + beta * d[n];
        }
    }

    stats
}

pub fn divergence_rhs(
    grid: &Grid,
    level: &Level,
    velocity: &MacField,
    weights: &MacField,
    solid: &MacField,
) -> Vec<f64> {
    let inv_dx = 1.0 / grid.dx;

    crate::exec::map(level.len(), |n| {
        if !level.active[n] {
            return 0.0;
        }

        let c = level.coords(n);
        let mut div = 0.0;

        for axis in 0..3 {
            let mut hi = c;
            hi[axis] += 1;

            let flux = |f: UVec3| {
                let w = weights.axes[axis].get(f.x, f.y, f.z);
                w * velocity.axes[axis].get(f.x, f.y, f.z)
                    + (1.0 - w) * solid.axes[axis].get(f.x, f.y, f.z)
            };

            div += flux(hi) - flux(c);
        }

        -div * inv_dx
    })
}

pub fn apply_gradient(
    grid: &Grid,
    level: &Level,
    p: &[f64],
    scale: f64,
    velocity: &mut MacField,
    weights: &MacField,
) -> MacField {
    let mut valid = MacField::new(grid);

    for axis in 0..3 {
        let w = &weights.axes[axis];
        let dims = w.dims;

        let results: Vec<Option<f64>> = crate::exec::map(w.data.len(), |fi| {
            if w.data[fi] <= 0.0 {
                return None;
            }

            let f = w.coords(fi);

            if axis == 1 && f.y == 0 {
                return level
                    .floor_difference(p, level.column(f))
                    .map(|dp| dp * scale);
            }

            if f[axis] == 0 || f[axis] == dims[axis] - 1 {
                return None;
            }

            let hi = level.index(f);
            let mut lo_c = f;
            lo_c[axis] -= 1;
            let lo = level.index(lo_c);

            face_pressure_difference(level, p, lo, hi).map(|dp| dp * scale)
        });

        for (fi, r) in results.into_iter().enumerate() {
            if let Some(dp) = r {
                velocity.axes[axis].data[fi] -= dp;
                valid.axes[axis].data[fi] = 1.0;
            }
        }
    }

    valid
}

pub fn face_pressure_difference(level: &Level, p: &[f64], lo: usize, hi: usize) -> Option<f64> {
    let (a_lo, a_hi) = (level.active[lo], level.active[hi]);

    match (a_lo, a_hi) {
        (true, true) => Some(p[hi] - p[lo]),
        (true, false) if level.phi[hi] >= 0.0 => {
            let theta = (level.phi[lo] / (level.phi[lo] - level.phi[hi])).max(THETA_MIN);
            Some(-p[lo] / theta)
        }
        (false, true) if level.phi[lo] >= 0.0 => {
            let theta = (level.phi[hi] / (level.phi[hi] - level.phi[lo])).max(THETA_MIN);
            Some(p[hi] / theta)
        }
        _ => None,
    }
}
