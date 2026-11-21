use glam::UVec3;

use crate::grid::MacField;
use crate::pressure::{COARSE_REGULARIZATION, Level, Multigrid, THETA_MIN, max_abs};

pub const TALL_SLOPE: i64 = 4;

pub const TALL_LEVELS: usize = 3;

const LATERAL: [(usize, i32); 4] = [(0, 1), (2, 1), (0, -1), (2, -1)];

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TallColumn {
    pub top: u32,

    pub aa: f64,

    pub bb: f64,

    pub ab: f64,

    pub cross: [[f64; 4]; 2],
}

impl Level {
    pub fn top(&self, col: usize) -> u32 {
        self.tall.get(col).map_or(0, |t| t.top)
    }

    pub fn column_cell(&self, col: usize, j: u32) -> usize {
        let base = self.coords(self.floor_cell(col));

        self.index(UVec3::new(base.x, j, base.z))
    }

    pub fn interior(&self, n: usize) -> bool {
        if self.tall.is_empty() {
            return false;
        }

        let c = self.coords(n);

        c.y > 0 && c.y < self.top(self.column(c))
    }

    pub fn endpoint(&self, n: usize) -> Option<(usize, bool)> {
        if self.tall.is_empty() {
            return None;
        }

        let c = self.coords(n);
        let col = self.column(c);
        let top = self.top(col);

        (top >= 2 && (c.y == 0 || c.y == top)).then_some((col, c.y == top))
    }

    pub fn unknown(&self, n: usize) -> bool {
        self.active[n] && !self.interior(n)
    }

    pub fn regular(&self, n: usize) -> bool {
        self.tall.is_empty() || (!self.interior(n) && self.endpoint(n).is_none())
    }

    pub fn unknowns(&self) -> usize {
        (0..self.len()).filter(|&n| self.unknown(n)).count()
    }

    pub fn gather_rhs(&self, rhs: &mut [f64], tolerance: f64) -> f64 {
        if self.tall.is_empty() {
            return tolerance;
        }

        let scale = max_abs(rhs);
        self.gather(rhs, false);

        tolerance * scale / self.residual_norm(rhs).max(f64::MIN_POSITIVE)
    }

    pub fn residual_norm(&self, v: &[f64]) -> f64 {
        if self.tall.is_empty() {
            return max_abs(v);
        }

        (0..v.len()).fold(0.0f64, |m, n| {
            let mass = self.endpoint(n).map_or(1.0, |(col, _)| 0.5 * (self.top(col) as f64 + 1.0));

            m.max(v[n].abs() / mass)
        })
    }

    pub fn value(&self, x: &[f64], n: usize) -> f64 {
        if !self.interior(n) {
            return x[n];
        }

        let c = self.coords(n);
        let col = self.column(c);
        let top = self.top(col);
        let s = c.y as f64 / top as f64;

        (1.0 - s) * x[self.column_cell(col, 0)] + s * x[self.column_cell(col, top)]
    }

    pub fn hat_sum(&self, col: usize, upper: bool, f: impl Fn(usize) -> f64) -> f64 {
        let top = self.top(col);

        (0..=top)
            .map(|j| {
                let s = j as f64 / top as f64;
                let hat = if upper { s } else { 1.0 - s };

                hat * f(self.column_cell(col, j))
            })
            .sum()
    }

    pub fn gather(&self, v: &mut [f64], mean: bool) {
        for col in 0..self.tall.len() {
            let top = self.top(col);

            if top < 2 {
                continue;
            }

            let scale = if mean { 2.0 / (top as f64 + 1.0) } else { 1.0 };
            let lower = scale * self.hat_sum(col, false, |n| v[n]);
            let upper = scale * self.hat_sum(col, true, |n| v[n]);

            for j in 0..=top {
                v[self.column_cell(col, j)] = 0.0;
            }

            v[self.column_cell(col, 0)] = lower;
            v[self.column_cell(col, top)] = upper;
        }
    }

    pub fn expand(&self, x: &mut [f64]) {
        for col in 0..self.tall.len() {
            let top = self.top(col);
            let (a, b) = (x[self.column_cell(col, 0)], x[self.column_cell(col, top)]);

            for j in 1..top {
                let s = j as f64 / top as f64;
                x[self.column_cell(col, j)] = (1.0 - s) * a + s * b;
            }
        }
    }

    fn theta(&self, n: usize, m: usize) -> f64 {
        (self.phi[n] / (self.phi[n] - self.phi[m])).max(THETA_MIN)
    }

    fn lateral(&self, n: usize, axis: usize, sign: i32) -> Option<(usize, f64)> {
        let m = self.neighbor(self.coords(n), axis, sign)?;
        let k = if sign > 0 { self.coef[axis][n] } else { self.coef[axis][m] };

        (k > 0.0).then_some((m, k))
    }

    fn side(&self, col: usize, axis: usize, sign: i32) -> Option<(usize, usize)> {
        let m = self.neighbor(self.coords(self.floor_cell(col)), axis, sign)?;

        Some((m, self.column(self.coords(m))))
    }

    pub fn place_tall(&mut self, weights: &MacField, band: u32) {
        let d = self.dims;
        let plain = |c: UVec3| {
            let n = self.index(c);

            self.active[n]
                && self.phi[n] < 0.0
                && (0..3).all(|a| {
                    (0..2).all(|h| {
                        let mut f = c;
                        f[a] += h;

                        f[a] == 0 || f[a] == d[a] || weights.axes[a].get(f.x, f.y, f.z) >= 1.0
                    })
                })
        };
        let low: Vec<i64> = (0..self.columns())
            .map(|col| {
                let base = self.coords(self.floor_cell(col));

                (0..d.y).find(|&j| !plain(UVec3::new(base.x, j, base.z))).unwrap_or(d.y) as i64
            })
            .collect();
        let mut cap: Vec<i64> = (0..self.columns())
            .map(|col| {
                let around = LATERAL.iter().filter_map(|&(axis, sign)| self.side(col, axis, sign));

                around.fold(low[col], |m, (_, other)| m.min(low[other])) - 1 - band as i64
            })
            .collect();

        envelope(&mut cap, d.x as usize, d.z as usize);

        let tops: Vec<u32> = cap.iter().map(|&t| if t >= 2 { t as u32 } else { 0 }).collect();

        self.assemble_tall(&tops, 0.0);
    }

    pub fn assemble_tall(&mut self, tops: &[u32], regularization: f64) {
        self.tall = tops.iter().map(|&top| TallColumn { top, ..TallColumn::default() }).collect();
        self.tall = (0..tops.len()).map(|col| self.tall_column(col, regularization)).collect();
    }

    fn tall_column(&self, col: usize, regularization: f64) -> TallColumn {
        let top = self.top(col);

        if top < 2 {
            return TallColumn::default();
        }

        let mut t = TallColumn { top, ..TallColumn::default() };
        let height = top as f64;
        let at = |j: u32| self.column_cell(col, j);
        let vertical = (0..top).map(|j| self.coef[1][at(j)]).sum::<f64>() / (height * height);

        t.aa = vertical + self.floor[col];
        t.bb = vertical;
        t.ab = -vertical;

        let n = at(top);

        if top + 1 < self.dims.y && self.coef[1][n] > 0.0 {
            let m = at(top + 1);

            if self.active[m] {
                t.bb += self.coef[1][n];
            } else if self.phi[m] >= 0.0 {
                t.bb += self.coef[1][n] / self.theta(n, m);
            }
        }

        for (dir, (axis, sign)) in LATERAL.into_iter().enumerate() {
            let Some((_, other_col)) = self.side(col, axis, sign) else {
                continue;
            };
            let other = self.top(other_col);

            for j in 0..=top {
                let n = at(j);
                let Some((m, k)) = self.lateral(n, axis, sign) else {
                    continue;
                };
                let s = j as f64 / height;
                let hat = [1.0 - s, s];
                let g = if self.active[m] {
                    k
                } else if self.phi[m] >= 0.0 {
                    k / self.theta(n, m)
                } else {
                    continue;
                };

                t.aa += g * hat[0] * hat[0];
                t.bb += g * hat[1] * hat[1];
                t.ab += g * hat[0] * hat[1];

                if self.active[m] && dir < 2 && other >= 2 && j <= other {
                    let s = j as f64 / other as f64;
                    let near = [1.0 - s, s];

                    for (w, (p, q)) in t.cross[dir].iter_mut().zip([(0, 0), (0, 1), (1, 0), (1, 1)]) {
                        *w += k * hat[p] * near[q];
                    }
                }
            }
        }

        let share = regularization / (1.0 + regularization);

        for j in 0..=top {
            let d = self.diag[at(j)] * share;
            let s = j as f64 / height;

            t.aa += d * (1.0 - s) * (1.0 - s);
            t.bb += d * s * s;
            t.ab += d * (1.0 - s) * s;
        }

        t
    }

    fn tall_coupling(&self, col: usize, x: &[f64], vertical: bool) -> [f64; 2] {
        let t = &self.tall[col];
        let top = t.top;
        let at = |j: u32| self.column_cell(col, j);
        let mut s = [0.0; 2];

        if vertical && top + 1 < self.dims.y {
            s[1] += self.off[1][at(top)] * x[at(top + 1)];
        }

        for (dir, (axis, sign)) in LATERAL.into_iter().enumerate() {
            let Some((side, other_col)) = self.side(col, axis, sign) else {
                continue;
            };
            let other = self.top(other_col);

            if other >= 2 {
                let w = if dir < 2 {
                    t.cross[dir]
                } else {
                    let w = self.tall[other_col].cross[dir - 2];
                    [w[0], w[2], w[1], w[3]]
                };
                let (a, b) = (x[side], x[self.column_cell(other_col, other)]);

                s[0] += w[0] * a + w[1] * b;
                s[1] += w[2] * a + w[3] * b;
            }

            let first = if other >= 2 { other + 1 } else { 0 };

            for j in first..=top {
                let Some((m, k)) = self.lateral(at(j), axis, sign) else {
                    continue;
                };

                if self.active[m] {
                    let s_j = j as f64 / top as f64;
                    s[0] += k * (1.0 - s_j) * x[m];
                    s[1] += k * s_j * x[m];
                }
            }
        }

        s
    }

    pub fn tall_row(&self, col: usize, x: &[f64]) -> [f64; 2] {
        let t = &self.tall[col];
        let (a, b) = (x[self.column_cell(col, 0)], x[self.column_cell(col, t.top)]);
        let s = self.tall_coupling(col, x, true);

        [t.aa * a + t.ab * b - s[0], t.ab * a + t.bb * b - s[1]]
    }

    fn line_lateral(&self, n: usize, x: &[f64]) -> f64 {
        let c = self.coords(n);
        let mut s = 0.0;

        for a in [0, 2] {
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

    fn line(&self, col: usize, x: &[f64], b: &[f64]) -> Vec<(usize, f64)> {
        let top = self.top(col);
        let at = |j: u32| self.column_cell(col, j);
        let mut rows = Vec::new();
        let (mut cp, mut dp) = (0.0, 0.0);
        let mut first = 0;

        if top >= 2 {
            let t = &self.tall[col];
            let s = self.tall_coupling(col, x, false);
            let (lo, hi) = (at(0), at(top));
            let (cp_lo, dp_lo) = (t.ab / t.aa, (b[lo] + s[0]) / t.aa);
            let m = t.bb - t.ab * cp_lo;

            cp = -self.off[1][hi] / m;
            dp = (b[hi] + s[1] - t.ab * dp_lo) / m;
            rows.extend([(lo, cp_lo, dp_lo), (hi, cp, dp)]);
            first = top + 1;
        }

        for j in first..self.dims.y {
            let n = at(j);

            if !self.active[n] {
                continue;
            }

            let lower = if j > 0 { -self.off[1][at(j - 1)] } else { 0.0 };
            let m = self.diag[n] - lower * cp;

            dp = (b[n] + self.line_lateral(n, x) - lower * dp) / m;
            cp = -self.off[1][n] / m;
            rows.push((n, cp, dp));
        }

        let mut next = 0.0;

        rows.into_iter()
            .rev()
            .map(|(n, cp, dp)| {
                next = dp - cp * next;

                (n, next)
            })
            .collect()
    }

    pub(crate) fn relax_lines(&self, x: &mut [f64], b: &[f64], parity: u32) {
        let updates = crate::exec::map(self.columns(), |col| {
            let c = self.coords(self.floor_cell(col));

            if (c.x + c.z) % 2 == parity { self.line(col, x, b) } else { Vec::new() }
        });

        for (n, v) in updates.into_iter().flatten() {
            x[n] = v;
        }
    }
}

fn envelope(v: &mut [i64], nx: usize, nz: usize) {
    let mut relax = |n: usize, m: usize| v[n] = v[n].min(v[m] + TALL_SLOPE);

    for k in 0..nz {
        for i in 1..nx {
            relax(i + nx * k, i - 1 + nx * k);
        }

        for i in (0..nx - 1).rev() {
            relax(i + nx * k, i + 1 + nx * k);
        }
    }

    for i in 0..nx {
        for k in 1..nz {
            relax(i + nx * k, i + nx * (k - 1));
        }

        for k in (0..nz - 1).rev() {
            relax(i + nx * k, i + nx * (k + 1));
        }
    }
}

impl Multigrid {
    pub fn with_tall(mut self, weights: &MacField, band: Option<u32>) -> Self {
        let Some(band) = band.filter(|_| self.levels.len() > 1) else {
            return self;
        };

        self.levels[0].place_tall(weights, band);

        for l in 1..TALL_LEVELS.min(self.levels.len() - 1) {
            let tops = coarse_tops(&self.levels[l - 1], &self.levels[l]);
            self.levels[l].assemble_tall(&tops, COARSE_REGULARIZATION);
        }

        self
    }
}

fn coarse_tops(fine: &Level, coarse: &Level) -> Vec<u32> {
    (0..coarse.columns())
        .map(|col| {
            let c = coarse.coords(coarse.floor_cell(col));
            let lowest = (0..4u32)
                .filter_map(|o| {
                    let f = UVec3::new(2 * c.x + (o & 1), 0, 2 * c.z + (o >> 1));

                    f.cmplt(fine.dims).all().then(|| fine.top(fine.column(f)))
                })
                .min()
                .unwrap_or(0);
            let top = lowest / 2;

            if top >= 2 { top } else { 0 }
        })
        .collect()
}
