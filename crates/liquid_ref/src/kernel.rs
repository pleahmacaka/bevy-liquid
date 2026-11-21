use glam::{DVec3, IVec3};

pub const APIC_D_INV_SCALE: f64 = 4.0;

pub fn bspline2(d: f64) -> f64 {
    let a = d.abs();

    if a < 0.5 {
        0.75 - a * a
    } else if a < 1.5 {
        let t = 1.5 - a;
        0.5 * t * t
    } else {
        0.0
    }
}

pub fn bspline2_weight(d: DVec3) -> f64 {
    bspline2(d.x) * bspline2(d.y) * bspline2(d.z)
}

pub struct Stencil {
    pub base: IVec3,

    pub weights: [[f64; 3]; 3],
}

impl Stencil {
    pub fn new(g: DVec3) -> Self {
        let base = (g - 0.5).floor();
        let f = g - base;
        let mut weights = [[0.0; 3]; 3];

        for axis in 0..3 {
            let x = f[axis];
            weights[axis] = [
                0.5 * (1.5 - x) * (1.5 - x),
                0.75 - (x - 1.0) * (x - 1.0),
                0.5 * (x - 0.5) * (x - 0.5),
            ];
        }

        Self {
            base: base.as_ivec3(),
            weights,
        }
    }

    pub fn weight(&self, o: IVec3) -> f64 {
        self.weights[0][o.x as usize]
            * self.weights[1][o.y as usize]
            * self.weights[2][o.z as usize]
    }

    pub fn offsets() -> impl Iterator<Item = IVec3> {
        (0..27).map(|n| IVec3::new(n % 3, (n / 3) % 3, n / 9))
    }
}
