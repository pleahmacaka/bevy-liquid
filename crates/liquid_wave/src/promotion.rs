use crate::airy::SurfaceMode;
use crate::dispersive::gradient;
use crate::grid::{Bathymetry, Grid, State};

pub const SCORE_CAP: f64 = 4.0;
pub const SCORE_SCALE: f64 = 256.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Criteria {
    pub breaking: f64,
    pub acceleration: f64,
    pub slope: f64,
    pub dry_depth: f64,
}

impl Default for Criteria {
    fn default() -> Self {
        Self {
            breaking: 0.85,
            acceleration: 0.5,
            slope: 1.0,
            dry_depth: 0.01,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Body {
    pub center: [f64; 3],
    pub radius: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Indicators {
    pub breaking: f64,
    pub acceleration: f64,
    pub falling: f64,
    pub piercing: f64,
}

impl Indicators {
    pub fn score(&self) -> f64 {
        self.breaking
            .max(self.acceleration)
            .max(self.falling)
            .max(self.piercing)
    }
}

pub struct Surface<'a> {
    pub grid: &'a Grid,
    pub bathymetry: &'a Bathymetry,
    pub state: &'a State,
    pub modes: Option<&'a [SurfaceMode]>,
    pub gravity: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Promotion {
    pub flagged: u32,
    pub bounds: [u32; 4],
    pub window: [u32; 4],
    pub weight: u32,
}

pub struct Detector {
    pub criteria: Criteria,
    pub previous: Option<Vec<f64>>,
}

impl Detector {
    pub fn new(criteria: Criteria) -> Self {
        Self {
            criteria,
            previous: None,
        }
    }

    pub fn evaluate(&mut self, surface: &Surface, dt: f64, bodies: &[Body]) -> Vec<Indicators> {
        let g = surface.grid;
        let c = &self.criteria;
        let wet: Vec<bool> = surface.state.h.iter().map(|&h| h > c.dry_depth).collect();
        let velocity: Vec<[f64; 2]> = (0..g.cells())
            .map(|cell| surface_speed(surface, cell, wet[cell]))
            .collect();
        let level: Vec<f64> = (0..g.cells())
            .map(|cell| {
                surface
                    .bathymetry
                    .cell(cell)
                    .level(surface.state.h[cell])
            })
            .collect();
        let vertical = vertical_velocity(surface, &wet, &velocity, &level);
        let mut out = vec![Indicators::default(); g.cells()];

        for k in 0..g.nz {
            for i in 0..g.nx {
                let cell = g.cell(i, k);

                if !wet[cell] {
                    continue;
                }

                let h = surface.state.h[cell];
                let u = velocity[cell];
                let speed = u[0].hypot(u[1]);
                let mode = surface.modes.map_or(SurfaceMode::default(), |m| m[cell]);
                let slope = masked_gradient(g, &level, &wet, i, k);
                let bed = gradient(g, &surface.bathymetry.center, i, k);
                let mut indicators = Indicators {
                    breaking: speed / mode.celerity(h, surface.gravity) / c.breaking,
                    ..Indicators::default()
                };

                if let Some(previous) = &self.previous {
                    let dw = masked_gradient(g, &vertical, &wet, i, k);
                    let acceleration =
                        (vertical[cell] - previous[cell]) / dt + u[0] * dw[0] + u[1] * dw[1];
                    indicators.acceleration = acceleration.abs() / surface.gravity / c.acceleration;
                }

                if speed > 1e-3 {
                    let along = (u[0] * slope[0] + u[1] * slope[1]) / speed;
                    let drop = -(u[0] * bed[0] + u[1] * bed[1]) / speed;
                    let steep = if along < 0.0 {
                        slope[0].hypot(slope[1])
                    } else {
                        0.0
                    };
                    indicators.falling = steep.max(drop) / c.slope;
                }

                let [x, z] = g.center(i, k);

                if bodies.iter().any(|b| {
                    let d = [x - b.center[0], level[cell] - b.center[1], z - b.center[2]];
                    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] < b.radius * b.radius
                }) {
                    indicators.piercing = 1.0;
                }

                out[cell] = indicators;
            }
        }

        self.previous = Some(vertical);

        out
    }
}

fn surface_speed(surface: &Surface, cell: usize, wet: bool) -> [f64; 2] {
    if !wet {
        return [0.0; 2];
    }

    let h = surface.state.h[cell];
    let mode = surface.modes.map_or(SurfaceMode::default(), |m| m[cell]);
    let crest = mode.velocity(0.0, h);

    [
        (surface.state.hu[cell] - mode.flux[0]) / h + crest[0],
        (surface.state.hv[cell] - mode.flux[1]) / h + crest[2],
    ]
}

fn vertical_velocity(
    surface: &Surface,
    wet: &[bool],
    velocity: &[[f64; 2]],
    level: &[f64],
) -> Vec<f64> {
    let g = surface.grid;
    let mut out = vec![0.0; g.cells()];

    for k in 0..g.nz {
        for i in 0..g.nx {
            let cell = g.cell(i, k);

            if !wet[cell] {
                continue;
            }

            let qx = masked_gradient(g, &surface.state.hu, wet, i, k)[0];
            let qz = masked_gradient(g, &surface.state.hv, wet, i, k)[1];
            let slope = masked_gradient(g, level, wet, i, k);
            let u = velocity[cell];

            out[cell] = -(qx + qz) + u[0] * slope[0] + u[1] * slope[1];
        }
    }

    out
}

fn masked_gradient(g: &Grid, f: &[f64], wet: &[bool], i: usize, k: usize) -> [f64; 2] {
    let at = |ii: usize, kk: usize| {
        let c = g.cell(ii, kk);
        if wet[c] { f[c] } else { f[g.cell(i, k)] }
    };
    let (il, ir) = (i.saturating_sub(1), (i + 1).min(g.nx - 1));
    let (kl, kr) = (k.saturating_sub(1), (k + 1).min(g.nz - 1));
    let gx = if ir > il {
        (at(ir, k) - at(il, k)) / ((ir - il) as f64 * g.dx)
    } else {
        0.0
    };
    let gz = if kr > kl {
        (at(i, kr) - at(i, kl)) / ((kr - kl) as f64 * g.dz)
    } else {
        0.0
    };

    [gx, gz]
}

pub fn weight(score: f64) -> u32 {
    if score < 1.0 {
        return 0;
    }

    (score.min(SCORE_CAP) * SCORE_SCALE + 0.5).floor() as u32
}

pub fn promote(grid: &Grid, indicators: &[Indicators], window: [usize; 2]) -> Option<Promotion> {
    let (nx, nz) = (grid.nx, grid.nz);
    let (wx, wz) = (window[0].clamp(1, nx), window[1].clamp(1, nz));
    let row = nx + 1;
    let mut table = vec![0u32; row * (nz + 1)];
    let mut flagged = 0u32;
    let mut bounds = [u32::MAX, u32::MAX, 0, 0];

    for k in 0..nz {
        for i in 0..nx {
            let w = weight(indicators[grid.cell(i, k)].score());

            if w > 0 {
                flagged += 1;
                bounds = [
                    bounds[0].min(i as u32),
                    bounds[1].min(k as u32),
                    bounds[2].max(i as u32 + 1),
                    bounds[3].max(k as u32 + 1),
                ];
            }

            table[(k + 1) * row + i + 1] = table[(k + 1) * row + i] + w;
        }
    }

    if flagged == 0 {
        return None;
    }

    for k in 0..nz {
        for i in 0..=nx {
            table[(k + 1) * row + i] += table[k * row + i];
        }
    }

    let centre = [bounds[0] + bounds[2], bounds[1] + bounds[3]];
    let mut best = (0u32, i64::MAX, usize::MAX);

    for k0 in 0..=nz - wz {
        for i0 in 0..=nx - wx {
            let at = |i: usize, k: usize| table[k * row + i];
            let sum = at(i0 + wx, k0 + wz) + at(i0, k0) - at(i0, k0 + wz) - at(i0 + wx, k0);
            let dx = (2 * i0 + wx) as i64 - centre[0] as i64;
            let dz = (2 * k0 + wz) as i64 - centre[1] as i64;
            let candidate = (sum, dx * dx + dz * dz, k0 * nx + i0);

            if better(candidate, best) {
                best = candidate;
            }
        }
    }

    let (i0, k0) = ((best.2 % nx) as u32, (best.2 / nx) as u32);

    Some(Promotion {
        flagged,
        bounds,
        window: [i0, k0, i0 + wx as u32, k0 + wz as u32],
        weight: best.0,
    })
}

fn better(a: (u32, i64, usize), b: (u32, i64, usize)) -> bool {
    a.0 > b.0 || (a.0 == b.0 && (a.1 < b.1 || (a.1 == b.1 && a.2 < b.2)))
}
