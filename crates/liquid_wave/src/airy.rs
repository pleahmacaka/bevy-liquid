pub fn omega(k: f64, depth: f64, gravity: f64) -> f64 {
    (gravity * k * (k * depth).tanh()).sqrt()
}

pub fn group_derivative(k: f64, depth: f64, gravity: f64) -> f64 {
    let w = omega(k, depth, gravity);

    if w <= 0.0 {
        return (gravity * depth).sqrt();
    }

    let kh = k * depth;
    let sech = 1.0 / kh.cosh();

    gravity * (kh.tanh() + kh * sech * sech) / (2.0 * w)
}

pub fn shallow_omega(k: f64, depth: f64, gravity: f64) -> f64 {
    k * (gravity * depth).sqrt()
}

pub fn horizontal_profile(k: f64, z: f64, depth: f64) -> f64 {
    let a = (k * z).exp();
    let b = (-k * (z + 2.0 * depth)).exp();
    let d = 1.0 - (-2.0 * k * depth).exp();

    (a + b) / d
}

pub fn vertical_profile(k: f64, z: f64, depth: f64) -> f64 {
    let a = (k * z).exp();
    let b = (-k * (z + 2.0 * depth)).exp();
    let d = 1.0 - (-2.0 * k * depth).exp();

    (a - b) / d
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SurfaceMode {
    pub k: f64,
    pub flux: [f64; 2],
    pub divergence: f64,
}

impl SurfaceMode {
    pub fn velocity(&self, z: f64, depth: f64) -> [f64; 3] {
        if depth <= 0.0 {
            return [0.0; 3];
        }

        let (horizontal, vertical) = if self.k * depth < 1e-6 {
            (1.0 / depth, (z + depth) / depth)
        } else {
            (
                self.k * horizontal_profile(self.k, z, depth),
                vertical_profile(self.k, z, depth),
            )
        };

        [
            horizontal * self.flux[0],
            -self.divergence * vertical,
            horizontal * self.flux[1],
        ]
    }

    pub fn celerity(&self, depth: f64, gravity: f64) -> f64 {
        if self.k * depth < 1e-6 {
            return (gravity * depth).sqrt();
        }

        (gravity * (self.k * depth).tanh() / self.k).sqrt()
    }
}

pub fn local_wavenumber(gradient_energy: f64, curvature_energy: f64, spacing: f64) -> f64 {
    if gradient_energy <= 0.0 || curvature_energy <= 0.0 {
        return 0.0;
    }

    let half = 0.5 * spacing * (curvature_energy / gradient_energy).sqrt();

    2.0 / spacing * half.min(1.0).asin()
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveComponent {
    pub k: [f64; 2],
    pub amplitude: f64,
    pub phase: f64,
    pub omega: f64,
}

impl WaveComponent {
    pub fn wavenumber(&self) -> f64 {
        self.k[0].hypot(self.k[1])
    }

    pub fn angle(&self, x: f64, z: f64, t: f64) -> f64 {
        self.k[0] * x + self.k[1] * z - self.omega * t + self.phase
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AmbientSample {
    pub elevation: f64,
    pub velocity: [f64; 3],
}

#[derive(Clone, Debug)]
pub struct AmbientField {
    pub components: Vec<WaveComponent>,
    pub depth: f64,
}

impl AmbientField {
    pub fn new(components: Vec<WaveComponent>, depth: f64) -> Self {
        Self { components, depth }
    }

    pub fn elevation(&self, x: f64, z: f64, t: f64) -> f64 {
        self.components
            .iter()
            .map(|c| c.amplitude * c.angle(x, z, t).cos())
            .sum()
    }

    pub fn flux(&self, x: f64, z: f64, t: f64) -> [f64; 2] {
        self.components.iter().fold([0.0; 2], |acc, c| {
            let k = c.wavenumber();
            let q = c.amplitude * c.omega / k * c.angle(x, z, t).cos();

            [acc[0] + q * c.k[0] / k, acc[1] + q * c.k[1] / k]
        })
    }

    pub fn sample(&self, x: f64, y: f64, z: f64, t: f64) -> AmbientSample {
        let elevation = self.elevation(x, z, t);
        let stretched = (y - elevation) * self.depth / (self.depth + elevation);

        let velocity = self.components.iter().fold([0.0; 3], |acc, c| {
            let k = c.wavenumber();
            let angle = c.angle(x, z, t);
            let horizontal =
                c.amplitude * c.omega * horizontal_profile(k, stretched, self.depth) * angle.cos();
            let vertical =
                c.amplitude * c.omega * vertical_profile(k, stretched, self.depth) * angle.sin();

            [
                acc[0] + horizontal * c.k[0] / k,
                acc[1] + vertical,
                acc[2] + horizontal * c.k[1] / k,
            ]
        });

        AmbientSample {
            elevation,
            velocity,
        }
    }

    pub fn strongest(&self, count: usize) -> Self {
        let mut components = self.components.clone();
        components.sort_by(|a, b| b.amplitude.total_cmp(&a.amplitude));
        components.truncate(count);

        Self {
            components,
            depth: self.depth,
        }
    }
}
