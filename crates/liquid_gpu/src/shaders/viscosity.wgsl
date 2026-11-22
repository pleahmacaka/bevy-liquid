override KIND: u32 = 0u;
override FIRST: bool = false;

const DAMPING_HALF_WIDTH: f32 = 1.5;
const PI: f32 = 3.14159265358979;
const SOURCE_DIRECTION: u32 = 0u;
const SOURCE_FULL: u32 = 1u;

fn viscous_source(n: u32) -> f32 {
    if KIND == 0u {
        return phi[n];
    }

    return signed_distance[n];
}

fn source_sample(x: vec3<f32>) -> f32 {
    let k = cell_corners(x);
    var acc = 0.0;

    for (var i = 0; i < 8; i = i + 1) {
        acc = acc + k.weight[i] * viscous_source(k.index[i]);
    }

    return acc;
}

fn liquid_fraction(x: vec3<f32>) -> f32 {
    let h = 0.5 * params.dx;
    var sum = 0.0;

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let o = vec3<f32>(f32(corner & 1), f32((corner >> 1) & 1), f32((corner >> 2) & 1)) - 0.5;
        sum = sum + clamp(0.5 - source_sample(x + o * h) / h, 0.0, 1.0);
    }

    return sum / 8.0;
}

fn stress_fraction(x: vec3<f32>) -> f32 {
    let fraction = liquid_fraction(x);

    if KIND == 0u {
        return fraction;
    }

    let d = source_sample(x);
    let width = DAMPING_HALF_WIDTH * params.dx;
    var weight = 0.0;

    if abs(d) < width {
        weight = params.dx * (1.0 + cos(PI * d / width)) / width;
    }

    return fraction * weight;
}

fn edge_axes(e: u32) -> vec3<u32> {
    switch e {
        case 0u: {
            return vec3<u32>(0u, 1u, 2u);
        }
        case 1u: {
            return vec3<u32>(0u, 2u, 1u);
        }
        default: {
            return vec3<u32>(1u, 2u, 0u);
        }
    }
}

fn edge_inside(e: u32, p: vec3<i32>) -> bool {
    let axes = edge_axes(e);
    let d = vec3<i32>(params.dims.xyz);

    return all(p >= vec3<i32>(0)) && p[axes.x] <= d[axes.x] && p[axes.y] <= d[axes.y] && p[axes.z] < d[axes.z];
}

@compute @workgroup_size(1)
fn viscous_scale() {
    var nu = params.physics.y / params.density;

    if KIND == 1u {
        nu = params.physics.x * state.dt / (params.density * params.dx);
    }

    scalars[S_VISCOUS_SCALE] = state.dt * nu / (params.dx * params.dx);
}

@compute @workgroup_size(256)
fn viscous_cell_fractions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        v_cells[n] = stress_fraction(cell_center(cell_coords(n)));
    }
}

@compute @workgroup_size(256)
fn viscous_edge_fractions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_node(gid.x);

    if n >= node_count() {
        return;
    }

    let p = node_coords(n);

    for (var e = 0u; e < 3u; e = e + 1u) {
        var value = 0.0;

        if edge_inside(e, p) {
            let offset = with_axis(vec3<f32>(0.0), edge_axes(e).z, 0.5);
            value = stress_fraction((vec3<f32>(p) + offset) * params.dx);
        }

        v_edges[e * node_count() + n] = value;
    }
}

@compute @workgroup_size(256)
fn viscous_face_fractions(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    let axis = face_axis(n);
    v_mass[n] = liquid_fraction(face_position(axis, face_coords(axis, n)));
}

fn cell_fraction(c: vec3<i32>) -> f32 {
    if !in_cells(c) {
        return 0.0;
    }

    return v_cells[cell_index(c)];
}

fn edge_fraction(e: u32, p: vec3<i32>) -> f32 {
    if !edge_inside(e, p) {
        return 0.0;
    }

    return v_edges[e * node_count() + node_index(p)];
}

@compute @workgroup_size(256)
fn viscous_diagonal(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    if weights[n] <= 0.0 {
        v_diag[n] = 0.0;
        return;
    }

    let a = face_axis(n);
    let f = face_coords(a, n);
    var stress = 2.0 * (cell_fraction(bump(f, a, -1)) + cell_fraction(f));

    for (var b = 0u; b < 3u; b = b + 1u) {
        if b == a {
            continue;
        }

        let e = a + b - 1u;
        stress = stress + (edge_fraction(e, f) + edge_fraction(e, bump(f, b, 1)));
    }

    v_diag[n] = v_mass[n] + scalars[S_VISCOUS_SCALE] * stress;
}

@compute @workgroup_size(256)
fn viscous_load(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    let u = velocity[n];
    let free = v_diag[n] > 0.0;
    v_x[n] = select(0.0, u, free);
    v_d[n] = select(u, 0.0, free);
}

fn viscous_u(b: u32, p: vec3<i32>, source: u32) -> f32 {
    if !in_faces(b, p) {
        return 0.0;
    }

    let m = face_index(b, p);

    if source == SOURCE_DIRECTION {
        return v_d[m];
    }

    return v_x[m] + v_d[m];
}

fn normal_stress(a: u32, c: vec3<i32>, source: u32) -> f32 {
    if !in_cells(c) {
        return 0.0;
    }

    return 2.0 * v_cells[cell_index(c)] * (viscous_u(a, bump(c, a, 1), source) - viscous_u(a, c, source));
}

fn shear_stress(e: u32, c: vec3<i32>, source: u32) -> f32 {
    if !edge_inside(e, c) {
        return 0.0;
    }

    let axes = edge_axes(e);
    let cb = bump(c, axes.y, -1);
    let ca = bump(c, axes.x, -1);
    let along_b = viscous_u(axes.x, c, source) - viscous_u(axes.x, cb, source);
    let along_a = viscous_u(axes.y, c, source) - viscous_u(axes.y, ca, source);

    return v_edges[e * node_count() + node_index(c)] * (along_b + along_a);
}

fn stress_divergence(a: u32, f: vec3<i32>, source: u32) -> f32 {
    var v = normal_stress(a, bump(f, a, -1), source) - normal_stress(a, f, source);

    for (var b = 0u; b < 3u; b = b + 1u) {
        if b == a {
            continue;
        }

        let e = a + b - 1u;
        v = v + (shear_stress(e, f, source) - shear_stress(e, bump(f, b, 1), source));
    }

    return v;
}

@compute @workgroup_size(256)
fn viscous_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    if v_diag[n] <= 0.0 {
        v_q[n] = 0.0;
        v_r[n] = 0.0;
        return;
    }

    let a = face_axis(n);
    let f = face_coords(a, n);
    let scale = scalars[S_VISCOUS_SCALE];
    v_q[n] = v_mass[n] * v_x[n] - scale * stress_divergence(a, f, SOURCE_DIRECTION);
    v_r[n] = -(scale * stress_divergence(a, f, SOURCE_FULL));
}

fn viscous_pair_value(n: u32) -> f32 {
    switch PAIR {
        case 0u: {
            var value = 0.0;

            if v_diag[n] > 0.0 {
                value = v_r[n] * (v_r[n] / v_diag[n]);
            }

            return value;
        }
        case 1u: {
            return v_d[n] * v_q[n];
        }
        case 2u: {
            return abs(v_r[n]);
        }
        default: {
            return abs(v_q[n]);
        }
    }
}

fn face_groups() -> u32 {
    if SPARSE {
        return (brick_list[0] * BRICK_FACES + WG - 1u) / WG;
    }

    return (face_total() + WG - 1u) / WG;
}

@compute @workgroup_size(256)
fn viscous_reduce_partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    let n = brick_face(gid.x);
    var value = 0.0;

    if n < face_total() {
        value = viscous_pair_value(n);
    }

    let total = reduce_workgroup(lane, value);

    if lane.index == 0u {
        partials[wid.x] = total;
    }
}

@compute @workgroup_size(256)
fn viscous_reduce_final(lane: Lane) {
    let count = face_groups();
    var acc = 0.0;

    for (var i = lane.index; i < count; i = i + WG) {
        acc = combine(acc, partials[i]);
    }

    let total = reduce_workgroup(lane, acc);

    if lane.index == 0u {
        scalars[TARGET] = total;
    }
}

@compute @workgroup_size(256)
fn viscous_direction(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    var z = 0.0;

    if v_diag[n] > 0.0 {
        z = v_r[n] / v_diag[n];
    }

    if FIRST {
        v_d[n] = z;
    } else {
        v_d[n] = z + scalars[S_BETA] * v_d[n];
    }
}

@compute @workgroup_size(256)
fn viscous_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    if v_diag[n] <= 0.0 {
        v_q[n] = 0.0;
        return;
    }

    let a = face_axis(n);
    v_q[n] = v_mass[n] * v_d[n] + scalars[S_VISCOUS_SCALE] * stress_divergence(a, face_coords(a, n), SOURCE_DIRECTION);
}

@compute @workgroup_size(256)
fn viscous_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    let alpha = scalars[S_ALPHA];
    v_x[n] = v_x[n] + alpha * v_d[n];
    v_r[n] = v_r[n] - alpha * v_q[n];
}

@compute @workgroup_size(256)
fn viscous_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    if v_diag[n] > 0.0 {
        velocity[n] = v_x[n];
    }

    valid[n] = v_mass[n];
}

fn viscous_counter() -> u32 {
    return S_VISCOUS_ITERATIONS + KIND;
}

@compute @workgroup_size(1)
fn viscous_open() {
    state.converged = 0u;
    scalars[viscous_counter()] = -1.0;
    write_slots(SOLVE_FIRST, state.running == 1u);
}

@compute @workgroup_size(1)
fn viscous_check() {
    if state.converged == 1u {
        write_slots(SOLVE_FIRST, false);
        return;
    }

    if scalars[S_RNORM] <= scalars[S_TOLERANCE] * max(scalars[S_BNORM], 1.0e-30) {
        state.converged = 1u;
    }

    scalars[viscous_counter()] = scalars[viscous_counter()] + 1.0;
    write_slots(SOLVE_FIRST, state.running == 1u && state.converged == 0u);
}
