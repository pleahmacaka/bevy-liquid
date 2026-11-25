use crate::grid::{Grid, MacField};
use crate::rigid::RigidBody;
use crate::sim::sdf_gradient;

pub const RANGE_CELLS: f64 = 3.0;

const SAMPLES: usize = 8;

const AREA: f64 = 0.12789;

fn kernel(u: f64) -> f64 {
    let w = 4.0 * u - 3.0;

    (1.0 - w * w).max(0.0).sqrt().sqrt()
}

pub fn profile(q: f64) -> f64 {
    if q <= 0.0 || q >= 1.0 {
        return 0.0;
    }

    let lo = q.max(0.5);
    let h = (1.0 - lo) / SAMPLES as f64;
    let sum: f64 = (0..SAMPLES).map(|k| kernel(lo + (k as f64 + 0.5) * h)).sum();

    q * sum * h / AREA
}

pub fn adhere(
    grid: &Grid,
    bodies: &[RigidBody],
    weights: &MacField,
    velocity: &mut MacField,
    strength: f64,
    dt: f64,
) {
    let reach = RANGE_CELLS * grid.dx;

    for (axis, field) in velocity.axes.iter_mut().enumerate() {
        let open = &weights.axes[axis].data;

        for n in 0..field.data.len() {
            if open[n] <= 0.0 {
                continue;
            }

            let x = grid.face_position(axis, field.coords(n));
            let mut nearest: Option<(f64, &RigidBody)> = None;

            for b in bodies {
                let s = b.sdf(x);

                if nearest.is_none_or(|(best, _)| s < best) {
                    nearest = Some((s, b));
                }
            }

            let Some((s, body)) = nearest.filter(|(s, _)| *s < reach) else {
                continue;
            };

            let normal = sdf_gradient(body, x, 1e-2 * grid.dx);
            field.data[n] -= dt * strength / reach * profile(s / reach) * normal[axis];
        }
    }
}
