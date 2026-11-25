override KIND: u32 = 0u;
override FIRST: bool = false;

const DAMPING_HALF_WIDTH: f32 = 1.5;
const PI: f32 = 3.14159265358979;

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

fn octant_bits(o: u32) -> vec3<i32> {
    return vec3<i32>(vec3<u32>(o & 1u, (o >> 1u) & 1u, o >> 2u));
}

fn liquid_fraction(p: vec3<i32>, node: vec3<bool>) -> f32 {
    let top = vec3<i32>(params.dims.xyz) - 1;
    var sum = 0.0;

    for (var corner = 0u; corner < 8u; corner = corner + 1u) {
        let bits = octant_bits(corner);
        let c = select(p, p - 1 + bits, node);
        var side = select(bits, 1 - bits, node);
        side = select(side, vec3<i32>(0), c < vec3<i32>(0));
        side = select(side, vec3<i32>(1), c > top);
        let q = u32(side.x + 2 * side.y + 4 * side.z);
        sum = sum + v_samples[8u * cell_index(clamp(c, vec3<i32>(0), top)) + q];
    }

    return sum / 8.0;
}

fn stress_fraction(p: vec3<i32>, node: vec3<bool>) -> f32 {
    return band_weighted(liquid_fraction(p, node), p, node);
}

fn band_weighted(fraction: f32, p: vec3<i32>, node: vec3<bool>) -> f32 {
    if KIND == 0u {
        return fraction;
    }

    let d = source_sample((vec3<f32>(p) + select(vec3<f32>(0.5), vec3<f32>(0.0), node)) * params.dx);
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

fn viscous_scale() -> f32 {
    var nu = params.physics.y / params.density;

    if KIND == 1u {
        nu = params.physics.x * state.dt / (params.density * params.dx);
    }

    return state.dt * nu / (params.dx * params.dx);
}

@compute @workgroup_size(256)
fn viscous_samples(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    let p = cell_coords(n);
    let c = vec3<f32>(p);
    let h = 0.5 * params.dx;
    var sum = 0.0;

    for (var o = 0u; o < 8u; o = o + 1u) {
        let k = index_corners(c - 0.25 + 0.5 * vec3<f32>(octant_bits(o)));
        var acc = 0.0;

        for (var i = 0; i < 8; i = i + 1) {
            acc = acc + k.weight[i] * viscous_source(k.index[i]);
        }

        let value = clamp(0.5 - acc / h, 0.0, 1.0);
        v_samples[8u * n + o] = value;
        sum = sum + value;
    }

    v_cells[n] = band_weighted(sum / 8.0, p, vec3<bool>(false));
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
            value = stress_fraction(p, !axis_mask(edge_axes(e).z));
        }

        v_edges[e * node_count() + n] = value;
    }
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

struct Couplings {
    lo: vec3<f32>,
    hi: vec3<f32>,
}

fn coupling_at(n: u32) -> Couplings {
    let count = face_total();

    return Couplings(
        vec3<f32>(v_k[n], v_k[count + n], v_k[2u * count + n]),
        vec3<f32>(v_k[3u * count + n], v_k[4u * count + n], v_k[5u * count + n]),
    );
}

fn coupling_low(a: u32, b: u32, f: vec3<i32>) -> f32 {
    if b == a {
        return 2.0 * viscous_scale() * cell_fraction(bump(f, a, -1));
    }

    return viscous_scale() * edge_fraction(a + b - 1u, f);
}

fn coupling_high(a: u32, b: u32, f: vec3<i32>) -> f32 {
    if b == a {
        return 2.0 * viscous_scale() * cell_fraction(f);
    }

    return viscous_scale() * edge_fraction(a + b - 1u, bump(f, b, 1));
}

override ADD_COUPLINGS: bool = false;
override COUPLINGS_ONLY: bool = false;

@compute @workgroup_size(256)
fn viscous_couplings(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    let a = face_axis(n);
    let f = face_coords(a, n);
    let solid = weights[n] <= 0.0;
    var c = Couplings(vec3<f32>(0.0), vec3<f32>(0.0));

    if !solid {
        c = Couplings(
            vec3<f32>(coupling_low(a, 0u, f), coupling_low(a, 1u, f), coupling_low(a, 2u, f)),
            vec3<f32>(coupling_high(a, 0u, f), coupling_high(a, 1u, f), coupling_high(a, 2u, f)),
        );
    }

    if ADD_COUPLINGS {
        let before = coupling_at(n);
        c = Couplings(c.lo + before.lo, c.hi + before.hi);
    }

    let count = face_total();
    v_k[n] = c.lo.x;
    v_k[count + n] = c.lo.y;
    v_k[2u * count + n] = c.lo.z;
    v_k[3u * count + n] = c.hi.x;
    v_k[4u * count + n] = c.hi.y;
    v_k[5u * count + n] = c.hi.z;

    if COUPLINGS_ONLY {
        return;
    }

    let mass = liquid_fraction(f, axis_mask(a));
    let diag = select(mass + (c.lo.x + c.lo.y + c.lo.z) + (c.hi.x + c.hi.y + c.hi.z), 0.0, solid);
    v_mass[n] = mass;
    v_diag[n] = diag;

    let u = velocity[n];
    let free = diag > 0.0;
    var guess = u;

    if KIND == 0u {
        guess = u + v_w[n];
    }

    v_x[n] = select(0.0, guess, free);
    v_d[n] = select(u, 0.0, free);
}

override VISCOUS_SLOT: u32 = 0u;

const LOCAL_SWEEPS: u32 = 4u;
const LOCAL_LOW: f32 = 0.15;
const LOCAL_HIGH: f32 = 2.0;
const LOCAL_EMPTY: u32 = BRICK_FACE_BOX;
const LOCAL_SPAN: u32 = BRICK_FACE_BOX + 1u;
const LOCAL_SHARE: u32 = (BRICK_FACE_BOX + WG - 1u) / WG;
const V_BN: u32 = 0u;
const V_DQ: u32 = 1u;
const V_RZ: u32 = 2u;
const V_RN: u32 = 3u;

var<workgroup> local_z: array<f32, 1154>;
var<private> search_beta: f32;

fn viscous_u(b: u32, p: vec3<i32>, search: bool) -> vec2<f32> {
    if !in_faces(b, p) {
        return vec2<f32>(0.0);
    }

    let m = face_index(b, p);

    if search {
        return vec2<f32>(search_at(m));
    }

    let fixed = v_d[m];

    return vec2<f32>(fixed, v_x[m] + fixed);
}

fn coupled_stress(a: u32, f: vec3<i32>, c: Couplings, search: bool) -> vec2<f32> {
    let own = viscous_u(a, f, search);
    var v = c.lo[a] * (own - viscous_u(a, bump(f, a, -1), search)) - c.hi[a] * (viscous_u(a, bump(f, a, 1), search) - own);

    for (var b = 0u; b < 3u; b = b + 1u) {
        if b == a {
            continue;
        }

        let up = bump(f, b, 1);
        let lower = c.lo[b] * (own - viscous_u(a, bump(f, b, -1), search) + viscous_u(b, f, search) - viscous_u(b, bump(f, a, -1), search));
        let upper = c.hi[b] * (viscous_u(a, up, search) - own + viscous_u(b, up, search) - viscous_u(b, bump(up, a, -1), search));
        v = v + lower - upper;
    }

    return v;
}

fn face_groups() -> u32 {
    if SPARSE {
        return (brick_list[0] * BRICK_FACES + WG - 1u) / WG;
    }

    return (face_total() + WG - 1u) / WG;
}

fn viscous_blocks() -> u32 {
    if SPARSE {
        return 3u * brick_list[0];
    }

    return 3u * brick_total();
}

fn viscous_slot(region: u32, i: u32) -> u32 {
    return region * max(face_groups(), viscous_blocks()) + i;
}

fn viscous_total(lane: Lane, region: u32, count: u32, largest: bool) -> f32 {
    var acc = 0.0;

    for (var i = lane.index; i < count; i = i + WG) {
        let v = partials[viscous_slot(region, i)];
        acc = select(acc + v, max(acc, v), largest);
    }

    return group_reduce(lane, acc, largest);
}

fn viscous_store(lane: Lane, region: u32, w: u32, value: f32) {
    if lane.index == 0u {
        partials[viscous_slot(region, w)] = value;
    }
}

@compute @workgroup_size(256)
fn viscous_residual(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    let n = brick_face(gid.x);
    var bn = 0.0;

    if n < face_total() {
        var r = 0.0;

        if v_diag[n] > 0.0 {
            let a = face_axis(n);
            let f = face_coords(a, n);
            let stress = coupled_stress(a, f, coupling_at(n), false);
            let u = velocity[n];
            bn = abs(v_mass[n] * u - stress.x);
            r = v_mass[n] * (u - v_x[n]) - stress.y;
        }

        v_r[n] = r;
    }

    viscous_store(lane, V_BN, wid.x, group_reduce(lane, bn, true));
}

fn local_origin(w: u32) -> vec3<i32> {
    if SPARSE {
        return slot_origin(listed_slot(w * BRICK_FACES, BRICK_FACES));
    }

    return brick_origin(w);
}

fn local_index(a: u32, p: vec3<i32>, f: vec3<i32>) -> u32 {
    if !in_faces(a, f) || (p[a] == i32(BRICK) && f[a] != i32(params.dims[a])) {
        return NONE;
    }

    return face_index(a, f);
}

fn local_coords(size: vec3<u32>, j: u32) -> vec3<i32> {
    return vec3<i32>(vec3<u32>(j % size.x, (j / size.x) % size.y, j / (size.x * size.y)));
}

fn search_old(m: u32) -> f32 {
    if SWAP {
        return v_d[m];
    }

    return v_e[m];
}

fn search_at(m: u32) -> f32 {
    if FIRST {
        return v_z[m];
    }

    return v_z[m] + search_beta * search_old(m);
}

@compute @workgroup_size(256)
fn viscous_apply(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    search_beta = scalars[S_BETA];

    let n = brick_face(gid.x);
    var dq = 0.0;

    if n < face_total() {
        let d = search_at(n);
        var q = 0.0;

        if v_diag[n] > 0.0 {
            let a = face_axis(n);
            q = v_mass[n] * d + coupled_stress(a, face_coords(a, n), coupling_at(n), true).x;
        }

        if SWAP {
            v_e[n] = d;
        } else {
            v_d[n] = d;
        }

        v_q[n] = q;
        dq = d * q;
    }

    viscous_store(lane, V_DQ, wid.x, group_reduce(lane, dq, false));
}

fn local_slot(a: u32, p: vec3<i32>) -> u32 {
    let size = vec3<i32>(face_box(a));

    if any(p < vec3<i32>(0)) || any(p >= size) {
        return LOCAL_EMPTY;
    }

    return u32(p.x + size.x * (p.y + size.y * p.z));
}

fn local_neighbors(a: u32, p: vec3<i32>, c: Couplings, base: u32) -> f32 {
    var v = 0.0;

    for (var b = 0u; b < 3u; b = b + 1u) {
        v = v + c.lo[b] * local_z[base + local_slot(a, bump(p, b, -1))] + c.hi[b] * local_z[base + local_slot(a, bump(p, b, 1))];
    }

    return v;
}

@compute @workgroup_size(256)
fn viscous_block(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    var alpha = 0.0;

    if !LEAD {
        let dq = viscous_total(lane, V_DQ, face_groups(), false);
        alpha = select(0.0, scalars[S_RZ] / dq, dq != 0.0);
    }

    let a = wid.x % 3u;
    let origin = local_origin(wid.x / 3u);
    var index: array<u32, LOCAL_SHARE>;
    var r: array<f32, LOCAL_SHARE>;
    var z: array<f32, LOCAL_SHARE>;
    var w: array<f32, LOCAL_SHARE>;
    var diag: array<f32, LOCAL_SHARE>;
    var couple: array<Couplings, LOCAL_SHARE>;

    for (var k = 0u; k < LOCAL_SHARE; k = k + 1u) {
        let j = lane.index + k * WG;
        index[k] = NONE;
        r[k] = 0.0;
        diag[k] = 1.0;

        if j < BRICK_FACE_BOX {
            let p = local_coords(face_box(a), j);
            let n = local_index(a, p, origin + p);

            if n != NONE && v_diag[n] > 0.0 {
                index[k] = n;
                diag[k] = v_diag[n];
                couple[k] = coupling_at(n);

                if !LEAD {
                    var d = v_d[n];

                    if SWAP {
                        d = v_e[n];
                    }

                    v_x[n] = v_x[n] + alpha * d;
                    v_r[n] = v_r[n] - alpha * v_q[n];
                }

                r[k] = v_r[n];
            }

            local_z[j] = 0.0;
            local_z[LOCAL_SPAN + j] = 0.0;
        }
    }

    if lane.index == 0u {
        local_z[LOCAL_EMPTY] = 0.0;
        local_z[LOCAL_SPAN + LOCAL_EMPTY] = 0.0;
    }

    let theta = 0.5 * (LOCAL_HIGH + LOCAL_LOW);
    let delta = 0.5 * (LOCAL_HIGH - LOCAL_LOW);
    let sigma = theta / delta;
    var rho = 1.0 / sigma;

    for (var k = 0u; k < LOCAL_SHARE; k = k + 1u) {
        w[k] = r[k] / (theta * diag[k]);
        z[k] = w[k];
    }

    for (var s = 1u; s < LOCAL_SWEEPS; s = s + 1u) {
        let base = (s % 2u) * LOCAL_SPAN;
        let next = 1.0 / (2.0 * sigma - rho);

        for (var k = 0u; k < LOCAL_SHARE; k = k + 1u) {
            let j = lane.index + k * WG;

            if j < BRICK_FACE_BOX {
                local_z[base + j] = z[k];
            }
        }

        workgroupBarrier();

        for (var k = 0u; k < LOCAL_SHARE; k = k + 1u) {
            if index[k] != NONE {
                let p = local_coords(face_box(a), lane.index + k * WG);
                let residual = r[k] - diag[k] * z[k] + local_neighbors(a, p, couple[k], base);
                w[k] = next * rho * w[k] + 2.0 * next / delta * residual / diag[k];
                z[k] = z[k] + w[k];
            }
        }

        rho = next;
    }

    var rz = 0.0;
    var rn = 0.0;

    for (var k = 0u; k < LOCAL_SHARE; k = k + 1u) {
        let j = lane.index + k * WG;

        if j < BRICK_FACE_BOX {
            let p = local_coords(face_box(a), j);
            let n = local_index(a, p, origin + p);

            if n != NONE {
                v_z[n] = z[k];
            }
        }

        rz = rz + r[k] * z[k];
        rn = max(rn, abs(r[k]));
    }

    viscous_store(lane, V_RZ, wid.x, group_reduce(lane, rz, false));
    viscous_store(lane, V_RN, wid.x, group_reduce(lane, rn, true));
}

@compute @workgroup_size(256)
fn viscous_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    let free = v_diag[n] > 0.0;

    if KIND == 0u {
        v_w[n] = select(0.0, v_x[n] - velocity[n], free);
    }

    if free {
        velocity[n] = v_x[n];
    }

    valid[n] = v_mass[n];
}

fn viscous_counter() -> u32 {
    return S_VISCOUS_ITERATIONS + KIND;
}

@compute @workgroup_size(1)
fn viscous_open() {
    let running = state.running == 1u;
    state.converged = 0u;

    if running {
        scalars[viscous_counter()] = -1.0;
    }

    indirect[3u * SLOT_COUNT + VISCOUS_SLOT] = viscous_blocks();
    write_slots(SOLVE_FIRST, running);
}

@compute @workgroup_size(256)
fn viscous_check(lane: Lane) {
    var bn = 0.0;

    if LEAD {
        bn = viscous_total(lane, V_BN, face_groups(), true);
    }

    let rz = viscous_total(lane, V_RZ, viscous_blocks(), false);
    let rn = viscous_total(lane, V_RN, viscous_blocks(), true);

    if lane.index != 0u {
        return;
    }

    if state.converged == 1u || state.running == 0u {
        write_slots(SOLVE_FIRST, false);
        return;
    }

    if LEAD {
        scalars[S_BNORM] = bn;
        scalars[S_BETA] = 0.0;
    } else {
        scalars[S_BETA] = select(0.0, rz / scalars[S_RZ], scalars[S_RZ] != 0.0);
    }

    scalars[S_RZ] = rz;
    scalars[S_RNORM] = rn;

    if rn <= scalars[S_TOLERANCE] * max(scalars[S_BNORM], 1.0e-30) {
        state.converged = 1u;
    }

    scalars[viscous_counter()] = scalars[viscous_counter()] + 1.0;
    write_slots(SOLVE_FIRST, state.converged == 0u);
}
