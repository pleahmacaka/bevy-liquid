override ADHESION: f32 = 0.0;

const ADHESION_RANGE: f32 = 3.0;
const ADHESION_SAMPLES: u32 = 8u;
const ADHESION_AREA: f32 = 0.12789;

fn adhesion_kernel(u: f32) -> f32 {
    let w = 4.0 * u - 3.0;
    return sqrt(sqrt(max(1.0 - w * w, 0.0)));
}

fn adhesion_profile(q: f32) -> f32 {
    if q <= 0.0 || q >= 1.0 {
        return 0.0;
    }

    let lo = max(q, 0.5);
    let h = (1.0 - lo) / f32(ADHESION_SAMPLES);
    var sum = 0.0;

    for (var k = 0u; k < ADHESION_SAMPLES; k = k + 1u) {
        sum = sum + adhesion_kernel(lo + (f32(k) + 0.5) * h);
    }

    return q * sum * h / ADHESION_AREA;
}

@compute @workgroup_size(256)
fn adhere(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() || weights[n] <= 0.0 {
        return;
    }

    let axis = face_axis(n);
    let x = face_position(axis, face_coords(axis, n));
    var nearest = BIG;
    var body = 0u;

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        let s = body_sdf(bodies[k], x);

        if s < nearest {
            nearest = s;
            body = k;
        }
    }

    let reach = ADHESION_RANGE * params.dx;

    if nearest >= reach {
        return;
    }

    let pull = ADHESION / reach * adhesion_profile(nearest / reach);
    velocity[n] = velocity[n] - state.dt * pull * sdf_gradient(bodies[body], x)[axis];
}

@compute @workgroup_size(256)
fn adhere_static(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() || weights[n] <= 0.0 {
        return;
    }

    let axis = face_axis(n);
    let x = face_position(axis, face_coords(axis, n));
    let nearest = solid_at(x);
    let reach = ADHESION_RANGE * params.dx;

    if nearest >= reach {
        return;
    }

    let pull = ADHESION / reach * adhesion_profile(nearest / reach);
    velocity[n] = velocity[n] - state.dt * pull * solid_normal(x)[axis];
}
