use glam::IVec3;

use crate::exec;
use crate::grid::{Field3, Grid, MacField};
use crate::pressure::{Level, THETA_MIN};
use crate::viscosity::Fractions;

pub const CURVATURE_BAND_CELLS: f64 = 3.0;

pub const LEVEL_SET_SMOOTHING_PASSES: usize = 8;

pub const DAMPING_HALF_WIDTH_CELLS: f64 = 1.5;

pub const JUMP_SMOOTHING_PASSES: usize = 2;

pub const RESOLVED_CURVATURE: f64 = 0.45;

pub const THIN_CURVATURE: f64 = 0.6;

pub fn capillary_dt(grid: &Grid, density: f64, sigma: f64) -> f64 {
    (density * grid.dx.powi(3) / (2.0 * std::f64::consts::PI * sigma)).sqrt()
}

pub fn capillary_viscosity(grid: &Grid, density: f64, sigma: f64, dt: f64) -> f64 {
    sigma * dt / (density * grid.dx)
}

pub fn capillary_damping(grid: &Grid, distance: &Field3) -> Fractions {
    let width = DAMPING_HALF_WIDTH_CELLS * grid.dx;

    Fractions::weighted(grid, distance, |d| {
        if d.abs() < width {
            grid.dx * (1.0 + (std::f64::consts::PI * d / width).cos()) / width
        } else {
            0.0
        }
    })
}

pub fn curvature(grid: &Grid, distance: &Field3) -> Field3 {
    let mut distance = distance.clone();

    for _ in 0..LEVEL_SET_SMOOTHING_PASSES {
        distance = blur(&distance);
    }

    let band = CURVATURE_BAND_CELLS * grid.dx;
    let mut kappa = distance.clone();

    kappa.data = exec::map(distance.data.len(), |n| {
        if distance.data[n].abs() > band {
            return 0.0;
        }

        mean_curvature(&distance, distance.coords(n).as_ivec3(), grid.dx)
            .clamp(-1.0 / grid.dx, 1.0 / grid.dx)
    });

    kappa
}

fn clamped(phi: &Field3, p: IVec3) -> f64 {
    let p = p.clamp(IVec3::ZERO, phi.dims.as_ivec3() - 1);

    phi.get(p.x as u32, p.y as u32, p.z as u32)
}

fn blur(phi: &Field3) -> Field3 {
    let mut out = phi.clone();

    out.data = exec::map(phi.data.len(), |n| {
        let c = phi.coords(n).as_ivec3();
        let neighbors: f64 = [IVec3::X, IVec3::Y, IVec3::Z]
            .iter()
            .map(|e| clamped(phi, c + *e) + clamped(phi, c - *e))
            .sum();

        0.5 * phi.data[n] + neighbors / 12.0
    });

    out
}

fn mean_curvature(phi: &Field3, c: IVec3, dx: f64) -> f64 {
    let at = |o: IVec3| clamped(phi, c + o);
    let e = [IVec3::X, IVec3::Y, IVec3::Z];
    let center = at(IVec3::ZERO);

    let g: [f64; 3] = std::array::from_fn(|a| (at(e[a]) - at(-e[a])) / (2.0 * dx));
    let h = |a: usize, b: usize| {
        if a == b {
            (at(e[a]) - 2.0 * center + at(-e[a])) / (dx * dx)
        } else {
            (at(e[a] + e[b]) - at(e[a] - e[b]) - at(e[b] - e[a]) + at(-e[a] - e[b]))
                / (4.0 * dx * dx)
        }
    };

    let norm2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];

    if norm2 < 1e-12 {
        return 0.0;
    }

    let mut numerator = 0.0;

    for a in 0..3 {
        let (b, d) = ((a + 1) % 3, (a + 2) % 3);
        numerator += h(a, a) * (g[b] * g[b] + g[d] * g[d]);
        numerator -= 2.0 * g[a] * g[b] * h(a, b);
    }

    numerator / norm2.powf(1.5)
}

pub struct PressureJump<'a> {
    pub level: &'a Level,

    pub kappa: &'a Field3,

    pub smooth: Vec<f64>,

    pub sigma: f64,

    pub dx: f64,
}

fn theta(level: &Level, liquid: usize, air: usize) -> Option<f64> {
    let ghost = level.active[liquid] && !level.active[air] && level.phi[air] >= 0.0;

    ghost.then(|| (level.phi[liquid] / (level.phi[liquid] - level.phi[air])).max(THETA_MIN))
}

struct Cut {
    air: usize,

    theta: f64,

    coef: f64,
}

fn cuts(level: &Level, n: usize) -> impl Iterator<Item = Cut> + '_ {
    let c = level.coords(n);

    (0..3)
        .flat_map(|a| [(a, -1), (a, 1)])
        .filter_map(move |(a, side)| {
            let air = level.neighbor(c, a, side)?;
            let coef = if side > 0 {
                level.coef[a][n]
            } else {
                level.coef[a][air]
            };

            if coef <= 0.0 {
                return None;
            }

            theta(level, n, air).map(|theta| Cut { air, theta, coef })
        })
}

fn interface_curvature(level: &Level, kappa: &Field3, n: usize) -> Option<f64> {
    let k = &kappa.data;
    let (sum, count) = cuts(level, n).fold((0.0, 0.0), |(sum, count), cut| {
        (sum + (1.0 - cut.theta) * k[n] + cut.theta * k[cut.air], count + 1.0)
    });

    (count > 0.0).then(|| sum / count)
}

fn neighborhood_mean(level: &Level, values: &[Option<f64>], n: usize) -> Option<f64> {
    if !level.active[n] {
        return None;
    }

    let c = level.coords(n).as_ivec3();
    let dims = level.dims.as_ivec3();
    let mut sum = 0.0;
    let mut count = 0.0;

    for o in 0..27 {
        let p = c + IVec3::new(o % 3 - 1, (o / 3) % 3 - 1, o / 9 - 1);

        if p.cmplt(IVec3::ZERO).any() || p.cmpge(dims).any() {
            continue;
        }

        if let Some(v) = values[level.index(p.as_uvec3())] {
            sum += v;
            count += 1.0;
        }
    }

    (count > 0.0).then(|| sum / count)
}

impl<'a> PressureJump<'a> {
    pub fn new(grid: &Grid, level: &'a Level, kappa: &'a Field3, sigma: f64) -> Self {
        let mut values: Vec<Option<f64>> =
            exec::map(level.len(), |n| interface_curvature(level, kappa, n));

        for _ in 0..JUMP_SMOOTHING_PASSES {
            values = exec::map(level.len(), |n| neighborhood_mean(level, &values, n));
        }

        Self {
            level,
            kappa,
            smooth: values.into_iter().map(|v| v.unwrap_or(0.0)).collect(),
            sigma,
            dx: grid.dx,
        }
    }

    pub fn theta(&self, liquid: usize, air: usize) -> Option<f64> {
        theta(self.level, liquid, air)
    }

    pub fn interface_pressure(&self, liquid: usize, air: usize, theta: f64) -> f64 {
        let k = &self.kappa.data;
        let smooth = self.smooth[liquid];
        let t = ((smooth.abs() * self.dx - RESOLVED_CURVATURE)
            / (THIN_CURVATURE - RESOLVED_CURVATURE))
            .clamp(0.0, 1.0);
        let thin = t * t * (3.0 - 2.0 * t);
        let sharp = (1.0 - theta) * k[liquid] + theta * k[air];

        self.sigma * (sharp + thin * (smooth - sharp))
    }

    pub fn add_to_rhs(&self, rhs: &mut [f64]) {
        for (n, slot) in rhs.iter_mut().enumerate() {
            for cut in cuts(self.level, n) {
                *slot += cut.coef / cut.theta * self.interface_pressure(n, cut.air, cut.theta);
            }
        }
    }

    pub fn apply_gradient(&self, scale: f64, velocity: &mut MacField, weights: &MacField) {
        let level = self.level;

        for axis in 0..3 {
            let w = &weights.axes[axis];
            let dims = w.dims;

            for (fi, v) in velocity.axes[axis].data.iter_mut().enumerate() {
                let f = w.coords(fi);

                if w.data[fi] <= 0.0 || f[axis] == 0 || f[axis] == dims[axis] - 1 {
                    continue;
                }

                let hi = level.index(f);
                let mut lo_c = f;
                lo_c[axis] -= 1;
                let lo = level.index(lo_c);

                if let Some(theta) = self.theta(lo, hi) {
                    *v -= scale * self.interface_pressure(lo, hi, theta) / theta;
                } else if let Some(theta) = self.theta(hi, lo) {
                    *v += scale * self.interface_pressure(hi, lo, theta) / theta;
                }
            }
        }
    }
}
