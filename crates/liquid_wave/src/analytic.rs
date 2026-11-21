#[derive(Clone, Copy, Debug)]
pub struct DamBreak {
    pub gravity: f64,
    pub dam: f64,
    pub upstream: f64,
    pub downstream: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct StokerState {
    pub depth: f64,
    pub velocity: f64,
    pub shock_speed: f64,
}

impl DamBreak {
    pub fn depth(&self, x: f64, t: f64) -> f64 {
        if t <= 0.0 {
            return if x < self.dam {
                self.upstream
            } else {
                self.downstream
            };
        }

        if self.downstream <= 0.0 {
            self.ritter(x, t)
        } else {
            self.stoker(x, t)
        }
    }

    fn rarefaction(&self, xi: f64) -> f64 {
        let c0 = (self.gravity * self.upstream).sqrt();
        let c = (2.0 * c0 - xi) / 3.0;

        c * c / self.gravity
    }

    fn ritter(&self, x: f64, t: f64) -> f64 {
        let c0 = (self.gravity * self.upstream).sqrt();
        let xi = (x - self.dam) / t;

        if xi <= -c0 {
            self.upstream
        } else if xi < 2.0 * c0 {
            self.rarefaction(xi)
        } else {
            0.0
        }
    }

    fn stoker(&self, x: f64, t: f64) -> f64 {
        let c0 = (self.gravity * self.upstream).sqrt();
        let middle = self.stoker_middle();
        let cm = (self.gravity * middle.depth).sqrt();
        let xi = (x - self.dam) / t;

        if xi <= -c0 {
            self.upstream
        } else if xi <= middle.velocity - cm {
            self.rarefaction(xi)
        } else if xi < middle.shock_speed {
            middle.depth
        } else {
            self.downstream
        }
    }

    pub fn stoker_middle(&self) -> StokerState {
        let g = self.gravity;
        let (h0, h1) = (self.upstream, self.downstream);
        let c0 = (g * h0).sqrt();

        let shock = |hm: f64| (g * hm * (hm + h1) / (2.0 * h1)).sqrt();
        let mismatch = |hm: f64| 2.0 * (c0 - (g * hm).sqrt()) - shock(hm) * (1.0 - h1 / hm);

        let (mut lo, mut hi) = (h1, h0);

        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);

            if mismatch(mid) > 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        let depth = 0.5 * (lo + hi);

        StokerState {
            depth,
            velocity: 2.0 * (c0 - (g * depth).sqrt()),
            shock_speed: shock(depth),
        }
    }
}
