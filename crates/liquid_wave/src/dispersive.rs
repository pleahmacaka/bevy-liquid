use crate::airy::{self, SurfaceMode};
use crate::fft::{self, Complex64, Fft2};
use crate::grid::{Grid, State};
use crate::kp07;
use crate::swe::{self, FaceField, Swe};

#[derive(Clone, Debug)]
pub struct DispersiveParams {
    pub reference_depths: Vec<f64>,
    pub filter_width: f64,
    pub gradient_penalty: f64,
    pub amplification_damping: f64,
}

impl Default for DispersiveParams {
    fn default() -> Self {
        Self {
            reference_depths: vec![1.0, 4.0, 16.0, 64.0],
            filter_width: 4.0,
            gradient_penalty: 0.25,
            amplification_damping: 0.25,
        }
    }
}

pub const MASK_FLOOR: f64 = 1e-5;

pub const SURFACE_SPEED_LIMIT: f64 = 0.5;

#[derive(Clone, Debug)]
pub struct Decomposition {
    pub bulk: State,
    pub surface: State,
}

#[derive(Clone, Debug)]
pub struct AiryOutput {
    pub flux: FaceField,
    pub qx: Vec<f64>,
    pub qz: Vec<f64>,
}

#[derive(Clone, Copy)]
pub struct Parity {
    pub odd_x: bool,
    pub odd_z: bool,
}

pub const EVEN: Parity = Parity {
    odd_x: false,
    odd_z: false,
};
pub const ODD_X: Parity = Parity {
    odd_x: true,
    odd_z: false,
};
pub const ODD_Z: Parity = Parity {
    odd_x: false,
    odd_z: true,
};

pub struct MirrorFft {
    pub grid: Grid,
    pub fft: Fft2,
}

impl MirrorFft {
    pub fn new(grid: Grid) -> Self {
        Self {
            fft: Fft2::new(2 * grid.nx, 2 * grid.nz),
            grid,
        }
    }

    pub fn ex(&self) -> usize {
        2 * self.grid.nx
    }

    pub fn ez(&self) -> usize {
        2 * self.grid.nz
    }

    pub fn transform(&self, field: &[f64], parity: Parity) -> Vec<Complex64> {
        let (nx, nz) = (self.grid.nx, self.grid.nz);
        let ex = self.ex();
        let mut data = vec![Complex64::default(); ex * self.ez()];

        for b in 0..self.ez() {
            for a in 0..ex {
                let (i, mirrored_x) = if a < nx {
                    (a, false)
                } else {
                    (2 * nx - 1 - a, true)
                };
                let (k, mirrored_z) = if b < nz {
                    (b, false)
                } else {
                    (2 * nz - 1 - b, true)
                };
                let sx = if parity.odd_x && mirrored_x {
                    -1.0
                } else {
                    1.0
                };
                let sz = if parity.odd_z && mirrored_z {
                    -1.0
                } else {
                    1.0
                };

                data[b * ex + a] = Complex64::new(sx * sz * field[self.grid.cell(i, k)], 0.0);
            }
        }

        self.fft.forward(&mut data);

        data
    }

    pub fn restore(&self, mut spectrum: Vec<Complex64>) -> Vec<f64> {
        self.fft.inverse(&mut spectrum);

        let ex = self.ex();
        let mut out = vec![0.0; self.grid.cells()];

        for k in 0..self.grid.nz {
            for i in 0..self.grid.nx {
                out[self.grid.cell(i, k)] = spectrum[k * ex + i].re;
            }
        }

        out
    }

    pub fn wavevector(&self, index: usize) -> [f64; 2] {
        let (ex, ez) = (self.ex(), self.ez());
        let [lx, lz] = self.grid.length();

        [
            fft::wavenumber(index % ex, ex, 2.0 * lx),
            fft::wavenumber(index / ex, ez, 2.0 * lz),
        ]
    }

    pub fn is_nyquist(&self, index: usize) -> bool {
        fft::is_nyquist(index % self.ex(), self.ex())
            || fft::is_nyquist(index / self.ex(), self.ez())
    }
}

pub fn bracket(nodes: &[f64], depth: f64) -> (usize, f64) {
    let last = nodes.len() - 1;

    if depth <= nodes[0] {
        return (0, 0.0);
    }

    if depth >= nodes[last] {
        return (last, 0.0);
    }

    let j = nodes.iter().rposition(|&n| n <= depth).unwrap_or(0);

    (j, (depth - nodes[j]) / (nodes[j + 1] - nodes[j]))
}

pub fn node_weight(nodes: &[f64], node: usize, depth: f64) -> f64 {
    let (j, t) = bracket(nodes, depth);

    if node == j {
        1.0 - t
    } else if node == j + 1 {
        t
    } else {
        0.0
    }
}

pub fn half_cell_correction(theta: f64) -> f64 {
    if theta.abs() < 1e-12 {
        1.0
    } else {
        theta / theta.sin()
    }
}

pub struct Dispersive {
    pub swe: Swe,
    pub params: DispersiveParams,
    pub mirror: MirrorFft,
}

impl Dispersive {
    pub fn new(swe: Swe, params: DispersiveParams) -> Self {
        assert!(
            !params.reference_depths.is_empty(),
            "at least one reference depth is required"
        );

        Self {
            mirror: MirrorFft::new(swe.grid),
            swe,
            params,
        }
    }

    pub fn step(&mut self, dt_max: f64) -> f64 {
        let grid = self.swe.grid;
        let state = self.swe.state.clone();
        let parts = self.decompose(&state);
        let wet: Vec<bool> = state.h.iter().map(|&h| h > 0.0).collect();

        let mut bulk = parts.bulk.clone();
        let mut flux = FaceField::zeros(&grid);
        let dt = self.swe.heun(&mut bulk, dt_max, Some(&mut flux));

        let airy = self.airy(&parts.surface, &parts.bulk.h, &wet, dt);
        let velocity = self.mean_velocity(&parts.bulk, &bulk);
        let (transport, qx, qz) =
            self.transport(&parts.surface.h, [&airy.qx, &airy.qz], &velocity, &wet, dt);

        for (f, (a, t)) in flux.x.iter_mut().zip(airy.flux.x.iter().zip(&transport.x)) {
            *f += a + t;
        }

        for (f, (a, t)) in flux.z.iter_mut().zip(airy.flux.z.iter().zip(&transport.z)) {
            *f += a + t;
        }

        swe::limit_outflow(&grid, &state.h, &mut flux);
        self.swe.interface_mass += self.swe.interface_flux(&flux);

        let mut next = State::zeros(grid.cells());
        let dry_depth = self.swe.params.dry_depth;
        let gravity = self.swe.params.gravity;

        for k in 0..grid.nz {
            for i in 0..grid.nx {
                let c = grid.cell(i, k);
                let divergence = (flux.x[grid.x_face(i + 1, k)] - flux.x[grid.x_face(i, k)])
                    / grid.dx
                    + (flux.z[grid.z_face(i, k + 1)] - flux.z[grid.z_face(i, k)]) / grid.dz;
                let h = (state.h[c] - divergence).max(0.0);
                let keep_surface = if h > dry_depth { 1.0 } else { 0.0 };

                next.h[c] = h;

                if h > 0.0 {
                    let cap = SURFACE_SPEED_LIMIT * h * (gravity * h).sqrt();
                    let scale = keep_surface * (cap / qx[c].hypot(qz[c]).max(1e-300)).min(1.0);
                    let hu = bulk.hu[c] + scale * qx[c];
                    let hv = bulk.hv[c] + scale * qz[c];
                    let limit = h * self.speed_bound(bulk.h[c].max(h), bulk.hu[c], bulk.hv[c]);
                    let total = (limit / hu.hypot(hv).max(1e-300)).min(1.0);

                    next.hu[c] = total * hu;
                    next.hv[c] = total * hv;
                }
            }
        }

        self.swe.state = next;
        self.swe.finish_step(dt);

        dt
    }

    pub fn decompose(&self, state: &State) -> Decomposition {
        let grid = &self.swe.grid;
        let bathymetry = &self.swe.bathymetry;
        let cells = grid.cells();

        let mask: Vec<f64> = state
            .h
            .iter()
            .map(|&h| if h > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let surface_level: Vec<f64> = (0..cells)
            .map(|c| bathymetry.cell(c).level(state.h[c]))
            .collect();
        let level: Vec<f64> = (0..cells).map(|c| surface_level[c] * mask[c]).collect();

        let spectra = [
            self.mirror.transform(&level, EVEN),
            self.mirror.transform(&mask, EVEN),
            self.mirror.transform(&state.hu, ODD_X),
            self.mirror.transform(&state.hv, ODD_Z),
        ];

        let mut nodes = vec![0.0];
        nodes.extend(&self.params.reference_depths);

        let identity = [surface_level, state.hu.clone(), state.hv.clone()];
        let mut smooth = [vec![0.0; cells], vec![0.0; cells], vec![0.0; cells]];

        for (node, &depth) in nodes.iter().enumerate() {
            let weights: Vec<f64> = state
                .h
                .iter()
                .map(|&h| node_weight(&nodes, node, h))
                .collect();

            if weights.iter().all(|&w| w == 0.0) {
                continue;
            }

            let filtered = if node == 0 {
                identity.clone()
            } else {
                self.filtered(&spectra, depth, &identity)
            };

            for (acc, value) in smooth.iter_mut().zip(&filtered) {
                for c in 0..cells {
                    acc[c] += weights[c] * value[c];
                }
            }
        }

        let penalty = self.gradient_penalty(&state.h);
        let mut bulk = State::zeros(cells);
        let mut surface = State::zeros(cells);

        for c in 0..cells {
            if mask[c] == 0.0 {
                continue;
            }

            let p = penalty[c];
            let w = identity[0][c] + p * (smooth[0][c] - identity[0][c]);
            let h = bathymetry.cell(c).depth(w);

            if h > 0.0 {
                let qx = state.hu[c] + p * (smooth[1][c] - state.hu[c]);
                let qz = state.hv[c] + p * (smooth[2][c] - state.hv[c]);
                let limit = h * self.speed_bound(state.h[c], state.hu[c], state.hv[c]);
                let scale = (limit / qx.hypot(qz).max(1e-300)).min(1.0);

                bulk.h[c] = h;
                bulk.hu[c] = scale * qx;
                bulk.hv[c] = scale * qz;
            }

            surface.h[c] = state.h[c] - bulk.h[c];
            surface.hu[c] = state.hu[c] - bulk.hu[c];
            surface.hv[c] = state.hv[c] - bulk.hv[c];
        }

        Decomposition { bulk, surface }
    }

    pub fn surface_modes(&self, surface: &State) -> Vec<SurfaceMode> {
        let g = &self.swe.grid;
        let eta = &surface.h;
        let flow = divergence(g, &surface.hu, &surface.hv);
        let eta_curvature = laplacian(g, eta);
        let flow_curvature = laplacian(g, &flow);
        let spacing = 0.5 * (g.dx + g.dz);
        let dry = self.swe.params.dry_depth;
        let gravity = self.swe.params.gravity;

        (0..g.cells())
            .map(|c| {
                let h = self.swe.state.h[c];

                if h <= dry {
                    return SurfaceMode::default();
                }

                let (i, k) = (c % g.nx, c / g.nx);
                let [ea, eb] = mode_energies(g, eta, &eta_curvature, i, k);
                let [fa, fb] = mode_energies(g, &flow, &flow_curvature, i, k);
                let weight = gravity / h;

                SurfaceMode {
                    k: airy::local_wavenumber(weight * ea + fa, weight * eb + fb, spacing),
                    flux: [surface.hu[c], surface.hv[c]],
                    divergence: flow[c],
                }
            })
            .collect()
    }

    pub fn column_velocity(&self, modes: &[SurfaceMode], i: usize, k: usize, height: f64) -> [f64; 3] {
        let g = &self.swe.grid;
        let c = g.cell(i, k);
        let h = self.swe.state.h[c];

        if h <= self.swe.params.dry_depth || height > h {
            return [0.0; 3];
        }

        let bulk = |c: usize| {
            let h = self.swe.state.h[c].max(self.swe.params.dry_depth);
            [
                (self.swe.state.hu[c] - modes[c].flux[0]) / h,
                (self.swe.state.hv[c] - modes[c].flux[1]) / h,
            ]
        };
        let (il, ir) = (i.saturating_sub(1), (i + 1).min(g.nx - 1));
        let (kl, kr) = (k.saturating_sub(1), (k + 1).min(g.nz - 1));
        let spread = |a: usize, b: usize, d: f64| if b > a { (b - a) as f64 * d } else { 1.0 };
        let bulk_divergence = (bulk(g.cell(ir, k))[0] - bulk(g.cell(il, k))[0]) / spread(il, ir, g.dx)
            + (bulk(g.cell(i, kr))[1] - bulk(g.cell(i, kl))[1]) / spread(kl, kr, g.dz);
        let surface = modes[c].velocity(height - h, h);
        let u = bulk(c);

        [
            u[0] + surface[0],
            -height * bulk_divergence + surface[1],
            u[1] + surface[2],
        ]
    }

    fn filtered(
        &self,
        spectra: &[Vec<Complex64>; 4],
        depth: f64,
        identity: &[Vec<f64>; 3],
    ) -> [Vec<f64>; 3] {
        let sigma = self.params.filter_width * depth;

        let apply = |spectrum: &Vec<Complex64>| {
            let filtered = spectrum
                .iter()
                .enumerate()
                .map(|(idx, v)| {
                    let [kx, kz] = self.mirror.wavevector(idx);
                    v * (-0.5 * sigma * sigma * (kx * kx + kz * kz)).exp()
                })
                .collect();

            self.mirror.restore(filtered)
        };

        let level = apply(&spectra[0]);
        let mask = apply(&spectra[1]);
        let qx = apply(&spectra[2]);
        let qz = apply(&spectra[3]);

        let cells = level.len();
        let mut out = [vec![0.0; cells], vec![0.0; cells], vec![0.0; cells]];

        for c in 0..cells {
            if mask[c] > MASK_FLOOR {
                out[0][c] = level[c] / mask[c];
                out[1][c] = qx[c] / mask[c];
                out[2][c] = qz[c] / mask[c];
            } else {
                out[0][c] = identity[0][c];
                out[1][c] = identity[1][c];
                out[2][c] = identity[2][c];
            }
        }

        out
    }

    fn speed_bound(&self, h: f64, hu: f64, hv: f64) -> f64 {
        let epsilon = self.swe.params.epsilon();
        let flow = kp07::desingularize(h, hu, epsilon).hypot(kp07::desingularize(h, hv, epsilon));

        flow + (self.swe.params.gravity * h).sqrt()
    }

    fn gradient_penalty(&self, h: &[f64]) -> Vec<f64> {
        let g = &self.swe.grid;
        let d = self.params.gradient_penalty;
        let mut out = vec![1.0; g.cells()];

        for k in 0..g.nz {
            for i in 0..g.nx {
                let at = |ii: usize, kk: usize| h[g.cell(ii, kk)];
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

                out[g.cell(i, k)] = (-d * (gx * gx + gz * gz)).exp();
            }
        }

        out
    }

    pub fn airy(&self, surface: &State, bulk_depth: &[f64], wet: &[bool], dt: f64) -> AiryOutput {
        let grid = &self.swe.grid;
        let gravity = self.swe.params.gravity;
        let depths = &self.params.reference_depths;

        let eta = self.mirror.transform(&surface.h, EVEN);
        let qx_hat = self.mirror.transform(&surface.hu, ODD_X);
        let qz_hat = self.mirror.transform(&surface.hv, ODD_Z);

        let mut out = AiryOutput {
            flux: FaceField::zeros(grid),
            qx: vec![0.0; grid.cells()],
            qz: vec![0.0; grid.cells()],
        };

        for (node, &depth) in depths.iter().enumerate() {
            let cell_weight = |c: usize| node_weight(depths, node, bulk_depth[c]);

            if (0..grid.cells()).all(|c| cell_weight(c) == 0.0) {
                continue;
            }

            let spectra = self.airy_spectra(&eta, &qx_hat, &qz_hat, depth, gravity, dt);
            let [qx, qz, fx, fz] = spectra.map(|s| self.mirror.restore(s));

            for c in 0..grid.cells() {
                out.qx[c] += cell_weight(c) * qx[c];
                out.qz[c] += cell_weight(c) * qz[c];
            }

            for k in 0..grid.nz {
                for i in 1..grid.nx {
                    let (l, r) = (grid.cell(i - 1, k), grid.cell(i, k));
                    let w = node_weight(depths, node, 0.5 * (bulk_depth[l] + bulk_depth[r]));
                    out.flux.x[grid.x_face(i, k)] += w * fx[l];
                }
            }

            for k in 1..grid.nz {
                for i in 0..grid.nx {
                    let (l, r) = (grid.cell(i, k - 1), grid.cell(i, k));
                    let w = node_weight(depths, node, 0.5 * (bulk_depth[l] + bulk_depth[r]));
                    out.flux.z[grid.z_face(i, k)] += w * fz[l];
                }
            }
        }

        self.mask_dry(&mut out, wet);

        out
    }

    fn airy_spectra(
        &self,
        eta: &[Complex64],
        qx: &[Complex64],
        qz: &[Complex64],
        depth: f64,
        gravity: f64,
        dt: f64,
    ) -> [Vec<Complex64>; 4] {
        let i = Complex64::new(0.0, 1.0);
        let (dx, dz) = (self.swe.grid.dx, self.swe.grid.dz);
        let len = eta.len();
        let mut out = [
            vec![Complex64::default(); len],
            vec![Complex64::default(); len],
            vec![Complex64::default(); len],
            vec![Complex64::default(); len],
        ];

        for idx in 0..len {
            let [kx, kz] = self.mirror.wavevector(idx);
            let k = kx.hypot(kz);

            if self.mirror.is_nyquist(idx) || k == 0.0 {
                out[0][idx] = qx[idx];
                out[1][idx] = qz[idx];
                continue;
            }

            let omega = crate::airy::omega(k, depth, gravity);
            let (s, c) = (omega * dt).sin_cos();
            let (nx, nz) = (kx / k, kz / k);

            let p = qx[idx] * nx + qz[idx] * nz;
            let tx = qx[idx] - p * nx;
            let tz = qz[idx] - p * nz;

            let p_next = p * c - i * (omega / k * s) * eta[idx];
            let p_integral = p * (s / omega) - i * ((1.0 - c) / k) * eta[idx];

            let shift_x = Complex64::from_polar(half_cell_correction(0.5 * kx * dx), 0.5 * kx * dx);
            let shift_z = Complex64::from_polar(half_cell_correction(0.5 * kz * dz), 0.5 * kz * dz);

            out[0][idx] = p_next * nx + tx;
            out[1][idx] = p_next * nz + tz;
            out[2][idx] = (p_integral * nx + tx * dt) * shift_x;
            out[3][idx] = (p_integral * nz + tz * dt) * shift_z;
        }

        out
    }

    fn mask_dry(&self, out: &mut AiryOutput, wet: &[bool]) {
        let g = &self.swe.grid;

        for c in 0..g.cells() {
            if !wet[c] {
                out.qx[c] = 0.0;
                out.qz[c] = 0.0;
            }
        }

        for k in 0..g.nz {
            for i in 1..g.nx {
                if !(wet[g.cell(i - 1, k)] && wet[g.cell(i, k)]) {
                    out.flux.x[g.x_face(i, k)] = 0.0;
                }
            }
        }

        for k in 1..g.nz {
            for i in 0..g.nx {
                if !(wet[g.cell(i, k - 1)] && wet[g.cell(i, k)]) {
                    out.flux.z[g.z_face(i, k)] = 0.0;
                }
            }
        }
    }

    fn mean_velocity(&self, start: &State, end: &State) -> [Vec<f64>; 2] {
        let epsilon = self.swe.params.epsilon();
        let speed = |s: &State, q: &[f64], c: usize| kp07::desingularize(s.h[c], q[c], epsilon);
        let cells = self.swe.grid.cells();

        [
            (0..cells)
                .map(|c| 0.5 * (speed(start, &start.hu, c) + speed(end, &end.hu, c)))
                .collect(),
            (0..cells)
                .map(|c| 0.5 * (speed(start, &start.hv, c) + speed(end, &end.hv, c)))
                .collect(),
        ]
    }

    fn transport(
        &self,
        surface_h: &[f64],
        q: [&Vec<f64>; 2],
        velocity: &[Vec<f64>; 2],
        wet: &[bool],
        dt: f64,
    ) -> (FaceField, Vec<f64>, Vec<f64>) {
        let g = &self.swe.grid;
        let [u, v] = velocity;
        let mut flux = FaceField::zeros(g);

        for k in 0..g.nz {
            for i in 1..g.nx {
                let (l, r) = (g.cell(i - 1, k), g.cell(i, k));

                if wet[l] && wet[r] {
                    let uf = 0.5 * (u[l] + u[r]);
                    let vf = 0.5 * (v[l] + v[r]);
                    let x = i as f64 * g.dx - 0.5 * dt * uf;
                    let z = (k as f64 + 0.5) * g.dz - 0.5 * dt * vf;
                    flux.x[g.x_face(i, k)] = dt * uf * sample(g, surface_h, x, z);
                }
            }
        }

        for k in 1..g.nz {
            for i in 0..g.nx {
                let (l, r) = (g.cell(i, k - 1), g.cell(i, k));

                if wet[l] && wet[r] {
                    let uf = 0.5 * (u[l] + u[r]);
                    let vf = 0.5 * (v[l] + v[r]);
                    let x = (i as f64 + 0.5) * g.dx - 0.5 * dt * uf;
                    let z = k as f64 * g.dz - 0.5 * dt * vf;
                    flux.z[g.z_face(i, k)] = dt * vf * sample(g, surface_h, x, z);
                }
            }
        }

        let gamma = self.params.amplification_damping;
        let divergence = divergence(g, u, v);
        let growth: Vec<f64> = divergence.iter().map(|&d| (-d).min(-gamma * d)).collect();
        let amplified = |field: &Vec<f64>| -> Vec<f64> {
            field
                .iter()
                .zip(&growth)
                .map(|(f, &gr)| f * (gr * dt).exp())
                .collect()
        };
        let ax = amplified(q[0]);
        let az = amplified(q[1]);

        let mut qx = vec![0.0; g.cells()];
        let mut qz = vec![0.0; g.cells()];

        for k in 0..g.nz {
            for i in 0..g.nx {
                let c = g.cell(i, k);

                if wet[c] {
                    let [x, z] = g.center(i, k);
                    let (x, z) = (x - dt * u[c], z - dt * v[c]);
                    qx[c] = sample(g, &ax, x, z);
                    qz[c] = sample(g, &az, x, z);
                }
            }
        }

        (flux, qx, qz)
    }
}

pub fn gradient(g: &Grid, f: &[f64], i: usize, k: usize) -> [f64; 2] {
    let (il, ir) = (i.saturating_sub(1), (i + 1).min(g.nx - 1));
    let (kl, kr) = (k.saturating_sub(1), (k + 1).min(g.nz - 1));
    let gx = if ir > il {
        (f[g.cell(ir, k)] - f[g.cell(il, k)]) / ((ir - il) as f64 * g.dx)
    } else {
        0.0
    };
    let gz = if kr > kl {
        (f[g.cell(i, kr)] - f[g.cell(i, kl)]) / ((kr - kl) as f64 * g.dz)
    } else {
        0.0
    };

    [gx, gz]
}

pub fn mode_energies(g: &Grid, f: &[f64], curvature: &[f64], i: usize, k: usize) -> [f64; 2] {
    let c = g.cell(i, k);
    let slope = gradient(g, f, i, k);
    let bend = gradient(g, curvature, i, k);

    [
        slope[0] * slope[0] + slope[1] * slope[1] - f[c] * curvature[c],
        curvature[c] * curvature[c] - (slope[0] * bend[0] + slope[1] * bend[1]),
    ]
}

pub fn laplacian(g: &Grid, f: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; g.cells()];

    for k in 0..g.nz {
        for i in 0..g.nx {
            let at = |i: usize, k: usize| f[g.cell(i, k)];
            let (il, ir) = (i.saturating_sub(1), (i + 1).min(g.nx - 1));
            let (kl, kr) = (k.saturating_sub(1), (k + 1).min(g.nz - 1));
            let centre = at(i, k);

            out[g.cell(i, k)] = (at(ir, k) - 2.0 * centre + at(il, k)) / (g.dx * g.dx)
                + (at(i, kr) - 2.0 * centre + at(i, kl)) / (g.dz * g.dz);
        }
    }

    out
}

pub fn divergence(g: &Grid, u: &[f64], v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; g.cells()];

    for k in 0..g.nz {
        for i in 0..g.nx {
            let (il, ir) = (i.saturating_sub(1), (i + 1).min(g.nx - 1));
            let (kl, kr) = (k.saturating_sub(1), (k + 1).min(g.nz - 1));

            let du = if ir > il {
                (u[g.cell(ir, k)] - u[g.cell(il, k)]) / ((ir - il) as f64 * g.dx)
            } else {
                0.0
            };
            let dv = if kr > kl {
                (v[g.cell(i, kr)] - v[g.cell(i, kl)]) / ((kr - kl) as f64 * g.dz)
            } else {
                0.0
            };

            out[g.cell(i, k)] = du + dv;
        }
    }

    out
}

pub fn sample(g: &Grid, field: &[f64], x: f64, z: f64) -> f64 {
    let fx = (x / g.dx - 0.5).clamp(0.0, (g.nx - 1) as f64);
    let fz = (z / g.dz - 0.5).clamp(0.0, (g.nz - 1) as f64);

    let i0 = (fx.floor() as usize).min(g.nx - 1);
    let k0 = (fz.floor() as usize).min(g.nz - 1);
    let i1 = (i0 + 1).min(g.nx - 1);
    let k1 = (k0 + 1).min(g.nz - 1);
    let (tx, tz) = (fx - i0 as f64, fz - k0 as f64);

    let at = |i: usize, k: usize| field[g.cell(i, k)];
    let bottom = at(i0, k0) * (1.0 - tx) + at(i1, k0) * tx;
    let top = at(i0, k1) * (1.0 - tx) + at(i1, k1) * tx;

    bottom * (1.0 - tz) + top * tz
}
