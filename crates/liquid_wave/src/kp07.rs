use std::f64::consts::SQRT_2;

#[derive(Clone, Copy, Debug)]
pub struct Kp07Params {
    pub gravity: f64,
    pub theta: f64,
    pub cfl: f64,
    pub dry_depth: f64,
    pub manning: f64,
}

impl Default for Kp07Params {
    fn default() -> Self {
        Self {
            gravity: crate::GRAVITY,
            theta: 1.3,
            cfl: 0.25,
            dry_depth: 1e-3,
            manning: 0.0,
        }
    }
}

impl Kp07Params {
    pub fn epsilon(&self) -> f64 {
        self.dry_depth.powi(4)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Cell {
    pub h: f64,
    pub qn: f64,
    pub qt: f64,
    pub level: f64,
    pub full: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point {
    pub h: f64,
    pub qn: f64,
    pub qt: f64,
    pub un: f64,
    pub ut: f64,
}

impl Point {
    pub fn new(h: f64, qn: f64, qt: f64, epsilon: f64) -> Self {
        let un = desingularize(h, qn, epsilon);
        let ut = desingularize(h, qt, epsilon);

        Self {
            h,
            qn: h * un,
            qt: h * ut,
            un,
            ut,
        }
    }

    pub fn from_velocity(h: f64, un: f64, ut: f64) -> Self {
        Self {
            h,
            qn: h * un,
            qt: h * ut,
            un,
            ut,
        }
    }

    pub fn mirrored(&self) -> Self {
        Self {
            qn: -self.qn,
            un: -self.un,
            ..*self
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Reconstruction {
    pub minus: Point,
    pub plus: Point,
    pub dw: f64,
}

pub fn desingularize(h: f64, q: f64, epsilon: f64) -> f64 {
    let h4 = h * h * h * h;

    if h4 >= epsilon {
        q / h
    } else {
        SQRT_2 * h * q / (h4 + epsilon).sqrt()
    }
}

pub fn minmod(a: f64, b: f64, c: f64) -> f64 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        a.min(b).min(c)
    } else if a < 0.0 && b < 0.0 && c < 0.0 {
        a.max(b).max(c)
    } else {
        0.0
    }
}

pub fn limited_half_slope(m: f64, c: f64, p: f64, theta: f64) -> f64 {
    0.5 * minmod(theta * (c - m), 0.5 * (p - m), theta * (p - c))
}

pub fn pressure(h: f64, gravity: f64) -> f64 {
    0.5 * gravity * h * h
}

pub fn reconstruct(
    m: Cell,
    c: Cell,
    p: Cell,
    b_minus: f64,
    b_plus: f64,
    params: &Kp07Params,
) -> Reconstruction {
    let epsilon = params.epsilon();

    if c.h <= 0.0 {
        return Reconstruction::default();
    }

    let w = c.level;

    if !c.full {
        let un = desingularize(c.h, c.qn, epsilon);
        let ut = desingularize(c.h, c.qt, epsilon);

        return Reconstruction {
            minus: Point::from_velocity((w - b_minus).max(0.0), un, ut),
            plus: Point::from_velocity((w - b_plus).max(0.0), un, ut),
            dw: 0.0,
        };
    }

    let sw = limited_half_slope(m.level, w, p.level, params.theta);
    let mut w_plus = w + sw;
    let mut w_minus = w - sw;

    if w_plus < b_plus {
        w_plus = b_plus;
        w_minus = 2.0 * w - b_plus;
    } else if w_minus < b_minus {
        w_minus = b_minus;
        w_plus = 2.0 * w - b_minus;
    }

    let sqn = limited_half_slope(m.qn, c.qn, p.qn, params.theta);
    let sqt = limited_half_slope(m.qt, c.qt, p.qt, params.theta);

    Reconstruction {
        minus: Point::new(
            (w_minus - b_minus).max(0.0),
            c.qn - sqn,
            c.qt - sqt,
            epsilon,
        ),
        plus: Point::new((w_plus - b_plus).max(0.0), c.qn + sqn, c.qt + sqt, epsilon),
        dw: w_plus - w_minus,
    }
}

fn physical_flux(s: &Point, gravity: f64) -> [f64; 3] {
    [s.qn, s.qn * s.un + pressure(s.h, gravity), s.qt * s.un]
}

pub fn central_upwind(l: &Point, r: &Point, gravity: f64) -> ([f64; 3], f64) {
    let cl = (gravity * l.h).sqrt();
    let cr = (gravity * r.h).sqrt();

    let ap = (l.un + cl).max(r.un + cr).max(0.0);
    let am = (l.un - cl).min(r.un - cr).min(0.0);
    let spread = ap - am;

    if spread <= 0.0 {
        return ([0.0; 3], 0.0);
    }

    let fl = physical_flux(l, gravity);
    let fr = physical_flux(r, gravity);
    let ul = [l.h, l.qn, l.qt];
    let ur = [r.h, r.qn, r.qt];

    let flux = std::array::from_fn(|n| {
        fl[n] + (am * (fl[n] - fr[n]) + ap * am * (ur[n] - ul[n])) / spread
    });

    (flux, ap.max(-am))
}

pub fn friction_factor(h: f64, hu: f64, hv: f64, dt: f64, params: &Kp07Params) -> f64 {
    if params.manning <= 0.0 || h <= 0.0 {
        return 1.0;
    }

    let epsilon = params.epsilon();
    let speed = desingularize(h, hu, epsilon).hypot(desingularize(h, hv, epsilon));
    let drag = params.gravity * params.manning * params.manning * speed / h.powf(4.0 / 3.0);

    1.0 / (1.0 + dt * drag)
}

pub fn bed_source(r: &Reconstruction, gravity: f64, spacing: f64) -> f64 {
    let hydrostatic = (pressure(r.plus.h, gravity) - pressure(r.minus.h, gravity)) / spacing;
    let surface = gravity * 0.5 * (r.plus.h + r.minus.h) * r.dw / spacing;

    hydrostatic - surface
}
