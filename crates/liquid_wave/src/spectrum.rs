use std::f64::consts::PI;

use crate::airy;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spreading {
    Hasselmann,
    DonelanBanner,
}

#[derive(Clone, Copy, Debug)]
pub struct SeaState {
    pub gravity: f64,
    pub wind_speed: f64,
    pub wind_direction: f64,
    pub fetch: f64,
    pub depth: f64,
    pub gamma: f64,
    pub swell: f64,
    pub spreading: Spreading,
}

impl Default for SeaState {
    fn default() -> Self {
        Self {
            gravity: crate::GRAVITY,
            wind_speed: 10.0,
            wind_direction: 0.0,
            fetch: 100_000.0,
            depth: 200.0,
            gamma: 3.3,
            swell: 0.0,
            spreading: Spreading::Hasselmann,
        }
    }
}

impl SeaState {
    pub fn peak_omega(&self) -> f64 {
        let g = self.gravity;

        22.0 * (g * g / (self.wind_speed * self.fetch)).cbrt()
    }

    pub fn alpha(&self) -> f64 {
        0.076 * (self.wind_speed * self.wind_speed / (self.fetch * self.gravity)).powf(0.22)
    }

    pub fn jonswap(&self, omega: f64) -> f64 {
        if omega <= 0.0 {
            return 0.0;
        }

        let g = self.gravity;
        let wp = self.peak_omega();
        let sigma = if omega <= wp { 0.07 } else { 0.09 };
        let r = (-(omega - wp).powi(2) / (2.0 * sigma * sigma * wp * wp)).exp();

        self.alpha() * g * g / omega.powi(5)
            * (-1.25 * (wp / omega).powi(4)).exp()
            * self.gamma.powf(r)
    }

    pub fn depth_attenuation(&self, omega: f64) -> f64 {
        let wh = omega * (self.depth / self.gravity).sqrt();

        if wh <= 1.0 {
            0.5 * wh * wh
        } else if wh < 2.0 {
            1.0 - 0.5 * (2.0 - wh).powi(2)
        } else {
            1.0
        }
    }

    pub fn tma(&self, omega: f64) -> f64 {
        self.jonswap(omega) * self.depth_attenuation(omega)
    }

    pub fn spreading_exponent(&self, omega: f64) -> f64 {
        let wp = self.peak_omega();
        let ratio = omega / wp;

        let base = if omega <= wp {
            6.97 * ratio.powf(4.06)
        } else {
            let mu = -2.33 - 1.45 * (self.wind_speed * wp / self.gravity - 1.17);
            9.77 * ratio.powf(mu)
        };

        base + 16.0 * (wp / omega).tanh() * self.swell * self.swell
    }

    pub fn direction(&self, omega: f64, theta: f64) -> f64 {
        let theta = wrap_angle(theta - self.wind_direction);

        match self.spreading {
            Spreading::Hasselmann => {
                let s = self.spreading_exponent(omega);
                cosine_power_normalization(s) * (0.5 * theta).cos().abs().powf(2.0 * s)
            }
            Spreading::DonelanBanner => {
                let beta = donelan_banner_beta(omega / self.peak_omega());
                beta / (2.0 * (beta * PI).tanh()) / (beta * theta).cosh().powi(2)
            }
        }
    }

    pub fn omega(&self, k: f64) -> f64 {
        airy::omega(k, self.depth, self.gravity)
    }

    pub fn wavevector_density(&self, kx: f64, kz: f64) -> f64 {
        let k = kx.hypot(kz);

        if k <= 0.0 {
            return 0.0;
        }

        let w = self.omega(k);
        let dw_dk = airy::group_derivative(k, self.depth, self.gravity);

        self.tma(w) * self.direction(w, kz.atan2(kx)) * dw_dk / k
    }

    pub fn band_variance(&self, k_min: f64, k_max: f64, samples: usize) -> f64 {
        let radial = samples.max(16);
        let angular = (2 * samples).max(64);
        let (lo, hi) = (k_min.max(1e-9).ln(), k_max.ln());
        let dlnk = (hi - lo) / radial as f64;
        let dtheta = 2.0 * PI / angular as f64;

        let mut total = 0.0;

        for r in 0..radial {
            let k = (lo + (r as f64 + 0.5) * dlnk).exp();

            for a in 0..angular {
                let theta = -PI + (a as f64 + 0.5) * dtheta;
                total += self.wavevector_density(k * theta.cos(), k * theta.sin())
                    * k
                    * k
                    * dlnk
                    * dtheta;
            }
        }

        total
    }
}

pub fn wrap_angle(theta: f64) -> f64 {
    (theta + PI).rem_euclid(2.0 * PI) - PI
}

pub fn cosine_power_normalization(s: f64) -> f64 {
    (((2.0 * s - 1.0) * 2f64.ln()) + 2.0 * ln_gamma(s + 1.0) - ln_gamma(2.0 * s + 1.0)).exp() / PI
}

fn donelan_banner_beta(ratio: f64) -> f64 {
    if ratio < 0.95 {
        2.61 * ratio.powf(1.3)
    } else if (0.95..1.6).contains(&ratio) {
        2.28 * ratio.powf(-1.3)
    } else {
        let epsilon = -0.4 + 0.8393 * (-0.567 * (ratio * ratio).ln()).exp();
        10f64.powf(epsilon)
    }
}

pub fn ln_gamma(x: f64) -> f64 {
    const COEFFICIENTS: [f64; 8] = [
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if x < 0.5 {
        return (PI / (PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }

    let x = x - 1.0;
    let t = x + 7.5;
    let series = COEFFICIENTS
        .iter()
        .enumerate()
        .fold(0.999_999_999_999_809_9, |acc, (i, c)| {
            acc + c / (x + i as f64 + 1.0)
        });

    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + series.ln()
}
