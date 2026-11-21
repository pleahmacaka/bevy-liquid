use glam::IVec3;

use crate::grid::{Field3, MacField};

pub fn extrapolate(field: &mut Field3, valid: &mut Field3, layers: usize) {
    let dims = field.dims.as_ivec3();

    for _ in 0..layers {
        let mut updates = Vec::new();

        for n in 0..field.data.len() {
            if valid.data[n] > 0.0 {
                continue;
            }

            let c = field.coords(n).as_ivec3();
            let mut sum = 0.0;
            let mut count = 0.0;

            for axis in 0..3 {
                for step in [-1, 1] {
                    let mut p = c;
                    p[axis] += step;

                    if p.cmplt(IVec3::ZERO).any() || p.cmpge(dims).any() {
                        continue;
                    }

                    let m = field.index(p.x as u32, p.y as u32, p.z as u32);

                    if valid.data[m] > 0.0 {
                        sum += field.data[m];
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                updates.push((n, sum / count));
            }
        }

        if updates.is_empty() {
            return;
        }

        for (n, v) in updates {
            field.data[n] = v;
            valid.data[n] = 1.0;
        }
    }
}

pub fn extrapolate_mac(velocity: &mut MacField, valid: &mut MacField, layers: usize) {
    for axis in 0..3 {
        extrapolate(&mut velocity.axes[axis], &mut valid.axes[axis], layers);
    }
}
