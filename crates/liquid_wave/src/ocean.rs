use std::f64::consts::PI;

use crate::airy::{AmbientField, WaveComponent};
use crate::fft::{self, Complex64, Fft2};
use crate::random;
use crate::spectrum::SeaState;

#[derive(Clone, Copy, Debug)]
pub struct FoamParams {
    pub threshold: f64,
    pub gain: f64,
    pub decay: f64,
}

impl Default for FoamParams {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            gain: 4.0,
            decay: 2.5,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OceanParams {
    pub sea: SeaState,
    pub resolution: usize,
    pub patch_sizes: Vec<f64>,
    pub choppiness: f64,
    pub seed: u64,
    pub foam: FoamParams,
}

impl Default for OceanParams {
    fn default() -> Self {
        Self {
            sea: SeaState::default(),
            resolution: 256,
            patch_sizes: vec![1013.0, 251.0, 61.0, 15.7],
            choppiness: 1.0,
            seed: 1,
            foam: FoamParams::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Cascade {
    pub patch: f64,
    pub n: usize,
    pub band: [f64; 2],
    pub h0: Vec<Complex64>,
    pub omega: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct CascadeFields {
    pub eta: Vec<f64>,
    pub dx: Vec<f64>,
    pub dz: Vec<f64>,
    pub slope_x: Vec<f64>,
    pub slope_z: Vec<f64>,
    pub dxx: Vec<f64>,
    pub dzz: Vec<f64>,
    pub dxz: Vec<f64>,
    pub jacobian: Vec<f64>,
}

impl Cascade {
    pub fn wavevector(&self, a: usize, b: usize) -> [f64; 2] {
        [
            fft::wavenumber(a, self.n, self.patch),
            fft::wavenumber(b, self.n, self.patch),
        ]
    }

    pub fn mirror(&self, index: usize) -> usize {
        let (a, b) = (index % self.n, index / self.n);

        ((self.n - b) % self.n) * self.n + (self.n - a) % self.n
    }

    pub fn spectrum(&self, t: f64) -> Vec<Complex64> {
        (0..self.n * self.n)
            .map(|i| {
                let rotation = Complex64::from_polar(1.0, -self.omega[i] * t);
                self.h0[i] * rotation + (self.h0[self.mirror(i)] * rotation).conj()
            })
            .collect()
    }

    pub fn spectral_variance(&self, t: f64) -> f64 {
        self.spectrum(t).iter().map(|h| h.norm_sqr()).sum()
    }

    pub fn expected_variance(&self) -> f64 {
        2.0 * self.h0.iter().map(|h| h.norm_sqr()).sum::<f64>()
    }

    pub fn band_variance_integral(&self, sea: &SeaState) -> f64 {
        let dk = 2.0 * PI / self.patch;

        sea.band_variance(self.band[0].max(dk), self.band[1], 400)
    }

    pub fn fields(&self, t: f64, choppiness: f64, fft: &Fft2) -> CascadeFields {
        let h = self.spectrum(t);
        let i = Complex64::new(0.0, 1.0);

        let field = |weight: &dyn Fn([f64; 2]) -> Complex64| -> Vec<f64> {
            let mut data: Vec<Complex64> = (0..h.len())
                .map(|idx| {
                    let k = self.wavevector(idx % self.n, idx / self.n);
                    weight(k) * h[idx]
                })
                .collect();

            fft.inverse_unnormalized(&mut data);

            data.iter().map(|v| v.re).collect()
        };

        let unit = |k: [f64; 2], axis: usize| {
            let m = k[0].hypot(k[1]);
            if m > 0.0 { k[axis] / m } else { 0.0 }
        };

        let lambda = choppiness;

        let eta = field(&|_| Complex64::new(1.0, 0.0));
        let dx = field(&|k| i * lambda * unit(k, 0));
        let dz = field(&|k| i * lambda * unit(k, 1));
        let slope_x = field(&|k| i * k[0]);
        let slope_z = field(&|k| i * k[1]);
        let dxx = field(&|k| Complex64::new(-lambda * k[0] * unit(k, 0), 0.0));
        let dzz = field(&|k| Complex64::new(-lambda * k[1] * unit(k, 1), 0.0));
        let dxz = field(&|k| Complex64::new(-lambda * k[0] * unit(k, 1), 0.0));

        let jacobian = (0..eta.len())
            .map(|c| (1.0 + dxx[c]) * (1.0 + dzz[c]) - dxz[c] * dxz[c])
            .collect();

        CascadeFields {
            eta,
            dx,
            dz,
            slope_x,
            slope_z,
            dxx,
            dzz,
            dxz,
            jacobian,
        }
    }

    pub fn components(&self) -> impl Iterator<Item = WaveComponent> + '_ {
        (0..self.n * self.n)
            .filter(|&i| self.h0[i].norm_sqr() > 0.0)
            .map(move |i| WaveComponent {
                k: self.wavevector(i % self.n, i / self.n),
                amplitude: 2.0 * self.h0[i].norm(),
                phase: self.h0[i].arg(),
                omega: self.omega[i],
            })
    }
}

pub struct Ocean {
    pub params: OceanParams,
    pub cascades: Vec<Cascade>,
    fft: Fft2,
}

impl Ocean {
    pub fn new(params: OceanParams) -> Self {
        let n = params.resolution;
        assert!(
            n.is_power_of_two(),
            "ocean resolution must be a power of two"
        );

        let bands = cascade_bands(&params.patch_sizes, n);
        let cascades = params
            .patch_sizes
            .iter()
            .zip(bands)
            .enumerate()
            .map(|(index, (&patch, band))| build_cascade(&params, index, patch, band))
            .collect();

        Self {
            fft: Fft2::new(n, n),
            params,
            cascades,
        }
    }

    pub fn fields(&self, cascade: usize, t: f64) -> CascadeFields {
        self.cascades[cascade].fields(t, self.params.choppiness, &self.fft)
    }

    pub fn ambient(&self) -> AmbientField {
        AmbientField::new(
            self.cascades.iter().flat_map(Cascade::components).collect(),
            self.params.sea.depth,
        )
    }

    pub fn fft(&self) -> &Fft2 {
        &self.fft
    }
}

pub fn cascade_bands(patch_sizes: &[f64], n: usize) -> Vec<[f64; 2]> {
    let nyquist = |l: f64| PI * n as f64 / l;
    let start = |l: f64| 6.0 * 2.0 * PI / l;

    (0..patch_sizes.len())
        .map(|i| {
            let lo = if i == 0 {
                0.0
            } else {
                start(patch_sizes[i]).min(nyquist(patch_sizes[i - 1]))
            };
            let hi = match patch_sizes.get(i + 1) {
                Some(&next) => start(next).min(nyquist(patch_sizes[i])),
                None => nyquist(patch_sizes[i]),
            };

            [lo, hi]
        })
        .collect()
}

fn build_cascade(params: &OceanParams, index: usize, patch: f64, band: [f64; 2]) -> Cascade {
    let n = params.resolution;
    let dk = 2.0 * PI / patch;
    let mut cascade = Cascade {
        patch,
        n,
        band,
        h0: vec![Complex64::default(); n * n],
        omega: vec![0.0; n * n],
    };

    for b in 0..n {
        for a in 0..n {
            let idx = b * n + a;
            let k = cascade.wavevector(a, b);
            let magnitude = k[0].hypot(k[1]);

            cascade.omega[idx] = params.sea.omega(magnitude);

            let in_band = magnitude >= band[0] && magnitude < band[1] && magnitude > 0.0;

            if !in_band || fft::is_nyquist(a, n) || fft::is_nyquist(b, n) {
                continue;
            }

            let density = params.sea.wavevector_density(k[0], k[1]);
            let [xr, xi] = random::gaussian_pair(&[params.seed, index as u64, a as u64, b as u64]);
            let amplitude = (0.5 * density * dk * dk).sqrt() / std::f64::consts::SQRT_2;

            cascade.h0[idx] = Complex64::new(xr, xi) * amplitude;
        }
    }

    cascade
}

#[derive(Clone, Debug)]
pub struct Foam {
    pub value: Vec<f64>,
}

impl Foam {
    pub fn new(cells: usize) -> Self {
        Self {
            value: vec![0.0; cells],
        }
    }

    pub fn update(&mut self, jacobian: &[f64], dt: f64, params: &FoamParams) {
        let keep = (-dt / params.decay).exp();

        for (f, &j) in self.value.iter_mut().zip(jacobian) {
            let produced = params.gain * dt * (params.threshold - j).max(0.0);
            *f = (*f * keep + produced).clamp(0.0, 1.0);
        }
    }
}
