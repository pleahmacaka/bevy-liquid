override EXTRAP_GRAVITY: bool = false;
override EXTRAP_FACES: bool = false;
override EXTRAP_BODIES: bool = false;
override EXTRAP_CONSTRAIN: u32 = 0u;
override GRADIENT_REFRESH: bool = false;
override GRADIENT_BLEND: bool = false;

fn source_velocity(m: u32, back: bool) -> f32 {
    if back {
        return scratch[m];
    }

    return velocity[m];
}

fn source_valid(m: u32, back: bool) -> f32 {
    if back {
        return scratch_valid[m];
    }

    return valid[m];
}

fn extrapolate_face(n: u32, back: bool) -> vec2<f32> {
    let own = vec2<f32>(source_velocity(n, back), source_valid(n, back));

    if own.y > 0.0 {
        return own;
    }

    let axis = face_axis(n);
    let f = face_coords(axis, n);
    var sum = 0.0;
    var count = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var step = -1; step <= 1; step = step + 2) {
            let p = bump(f, a, step);

            if !in_faces(axis, p) {
                continue;
            }

            let m = face_step(n, axis, f, a, step);

            if source_valid(m, back) > 0.0 {
                sum = sum + source_velocity(m, back);
                count = count + 1.0;
            }
        }
    }

    if count > 0.0 {
        return vec2<f32>(sum / count, 1.0);
    }

    return own;
}

@compute @workgroup_size(256)
fn extrapolate_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        let r = extrapolate_face(n, false);
        scratch[n] = r.x;
        scratch_valid[n] = r.y;
    }
}

fn extrapolate_back_face(n: u32) {
    let r = extrapolate_face(n, true);
    var v = r.x;

    if EXTRAP_GRAVITY {
        v = v + params.gravity[face_axis(n)] * state.dt;
    }

    valid[n] = r.y;

    if EXTRAP_FACES {
        settle_face(n, true);
    }

    if EXTRAP_CONSTRAIN != 0u && weights[n] <= 0.0 {
        v = select(0.0, solid_velocity[n], EXTRAP_CONSTRAIN == 1u);
    }

    velocity[n] = v;
}

fn finish_bodies(t: u32) {
    if EXTRAP_BODIES && t == 0u {
        add_body_gravity();
    }
}

@compute @workgroup_size(256)
fn extrapolate_back(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        extrapolate_back_face(n);
    }

    finish_bodies(gid.x);
}

fn add_body_gravity() {
    for (var k = 0u; k < params.body_count; k = k + 1u) {
        if bodies[k].velocity.w > 1.5 {
            bodies[k].velocity = vec4<f32>(bodies[k].velocity.xyz + params.gravity.xyz * state.dt, bodies[k].velocity.w);
        }
    }
}

fn integrate_bodies() {
    let dt = state.dt;

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        var b = bodies[k];

        if b.velocity.w < 0.5 {
            continue;
        }

        b.position = vec4<f32>(b.position.xyz + b.velocity.xyz * dt, b.position.w);

        let w = b.angular.xyz * dt;
        let angle = length(w);

        if angle > 0.0 {
            let axis = w / angle;
            let dq = vec4<f32>(axis * sin(0.5 * angle), cos(0.5 * angle));
            let q = b.rotation;
            let r = vec4<f32>(
                dq.w * q.xyz + q.w * dq.xyz + cross(dq.xyz, q.xyz),
                dq.w * q.w - dot(dq.xyz, q.xyz),
            );
            b.rotation = normalize(r);
        }

        bodies[k] = b;
    }
}

@compute @workgroup_size(256)
fn constrain_solid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    if weights[n] <= 0.0 {
        velocity[n] = solid_velocity[n];
    }
}

@compute @workgroup_size(256)
fn constrain_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    if weights[n] <= 0.0 {
        velocity[n] = 0.0;
    }
}

@compute @workgroup_size(256)
fn zero_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    velocity[n] = 0.0;
    valid[n] = 0.0;
}

fn face_flux(axis: u32, f: vec3<i32>) -> f32 {
    let i = face_index(axis, f);
    let w = weights[i];
    return w * velocity[i] + (1.0 - w) * solid_velocity[i];
}

@compute @workgroup_size(256)
fn divergence(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        divergence_cell(n);
    }
}

@compute @workgroup_size(256)
fn divergence_volume(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        divergence_cell(n);
    }

    volume_partial(n, wid.x, lane);
}

fn divergence_cell(n: u32) {
    if l_active[n] == 0u {
        l_b[n] = 0.0;
        return;
    }

    let c = cell_coords(n);
    var div = 0.0;

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        div = div + face_flux(axis, bump(c, axis, 1)) - face_flux(axis, c);
    }

    l_b[n] = -div * params.inv_dx;
}

fn pressure_difference(lo: u32, hi: u32) -> vec2<f32> {
    let a_lo = l_active[lo] != 0u;
    let a_hi = l_active[hi] != 0u;

    if a_lo && a_hi {
        return vec2<f32>(l_x[hi] - l_x[lo], 1.0);
    }

    if a_lo && !a_hi && l_phi[hi] >= 0.0 {
        let theta = max(l_phi[lo] / (l_phi[lo] - l_phi[hi]), THETA_MIN);
        return vec2<f32>(-l_x[lo] / theta, 1.0);
    }

    if !a_lo && a_hi && l_phi[lo] >= 0.0 {
        let theta = max(l_phi[hi] / (l_phi[hi] - l_phi[lo]), THETA_MIN);
        return vec2<f32>(l_x[hi] / theta, 1.0);
    }

    return vec2<f32>(0.0, 0.0);
}

@compute @workgroup_size(256)
fn apply_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        gradient_only(n);
    }
}

@compute @workgroup_size(256)
fn gradient_flow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        if GRADIENT_REFRESH {
            refresh_face(n, false);
        }

        gradient_face(n);
    }
}

fn gradient_face(n: u32) {
    gradient_only(n);

    if GRADIENT_BLEND {
        blend_face(n);
    }
}

fn gradient_only(n: u32) {
    valid[n] = 0.0;

    if weights[n] <= 0.0 {
        return;
    }

    let axis = face_axis(n);
    let f = face_coords(axis, n);

    if OPEN_FLOOR && axis == 1u && f.y == 0 {
        let col = l_column(f);
        let above = cell_index(f);

        if l_coef[floor_slot(col)] > 0.0 && l_active[above] != 0u {
            velocity[n] = velocity[n] - (l_x[above] - l_coef[ghost_slot(col)]) * scalars[S_GRADIENT];
            valid[n] = 1.0;
        }

        return;
    }

    if f[axis] == 0 || f[axis] == i32(params.dims[axis]) {
        return;
    }

    let dp = pressure_difference(cell_index(bump(f, axis, -1)), cell_index(f));

    if dp.y > 0.0 {
        velocity[n] = velocity[n] - dp.x * scalars[S_GRADIENT];
        valid[n] = 1.0;
    }
}

@compute @workgroup_size(256)
fn blend_partial(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        blend_face(n);
    }
}

fn blend_face(n: u32) {
    if valid[n] <= 0.0 {
        return;
    }

    let w = min(weights[n], 1.0);
    velocity[n] = w * velocity[n] + (1.0 - w) * solid_velocity[n];
    valid[n] = 2.0;
}

@compute @workgroup_size(256)
fn displace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= state.particle_count {
        return;
    }

    let x = particles[i].x.xyz;
    particles[i].x = vec4<f32>(x + sample_velocity(x), particles[i].x.w);
}

@compute @workgroup_size(256)
fn copy_scratch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        velocity[n] = scratch[n];
        valid[n] = scratch_valid[n];
    }
}

@compute @workgroup_size(256)
fn save_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        pressure_keep[n] = l_x[n];
    }
}

@compute @workgroup_size(256)
fn restore_pressure(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        l_x[n] = pressure_keep[n];
    }
}

