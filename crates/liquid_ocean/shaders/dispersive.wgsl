struct Params {
    nx: u32,
    nz: u32,
    ex: u32,
    ez: u32,
    dx: f32,
    dz: f32,
    gravity: f32,
    epsilon: f32,
    dry_depth: f32,
    filter_width: f32,
    gradient_penalty: f32,
    damping: f32,
    depth_count: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    depths: vec4<f32>,
}

struct Node {
    index: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct Clock {
    dt: f32,
    remaining: f32,
    time: f32,
    elapsed: f32,
    rate: u32,
    steps: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<uniform> node: Node;
@group(0) @binding(2) var<storage, read_write> total: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> bulk: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> bulk_start: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> surface: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> bed: array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> spectrum: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> work: array<vec2<f32>>;
@group(0) @binding(9) var<storage, read_write> cells: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read_write> faces: array<f32>;
@group(0) @binding(11) var<storage, read_write> limits: array<f32>;
@group(0) @binding(12) var<storage, read_write> accumulated: array<f32>;
@group(0) @binding(13) var<storage, read> clock: Clock;
@group(0) @binding(14) var<storage, read_write> profile: array<vec4<f32>>;

const SQRT_2: f32 = 1.41421356237;
const TAU: f32 = 6.28318530718;
const MASK_FLOOR: f32 = 1e-5;
const SURFACE_SPEED_LIMIT: f32 = 0.5;

const SMOOTH: u32 = 0u;
const AIRY_Q: u32 = 1u;
const AMPLIFIED: u32 = 2u;
const SURFACE_Q: u32 = 3u;

const AIRY_FLUX: u32 = 0u;
const TRANSPORT: u32 = 1u;
const COMBINED: u32 = 2u;

fn cell_count() -> u32 {
    return params.nx * params.nz;
}

fn x_face_count() -> u32 {
    return (params.nx + 1u) * params.nz;
}

fn face_count() -> u32 {
    return x_face_count() + params.nx * (params.nz + 1u);
}

fn ext_count() -> u32 {
    return params.ex * params.ez;
}

fn cell_index(i: u32, k: u32) -> u32 {
    return k * params.nx + i;
}

fn x_face_index(i: u32, k: u32) -> u32 {
    return k * (params.nx + 1u) + i;
}

fn z_face_index(i: u32, k: u32) -> u32 {
    return x_face_count() + k * params.nx + i;
}

fn ext_index(i: u32, k: u32) -> u32 {
    return k * params.ex + i;
}

fn cell_slot(slot: u32, c: u32) -> u32 {
    return slot * cell_count() + c;
}

fn face_slot(slot: u32, f: u32) -> u32 {
    return slot * face_count() + f;
}

fn global_index(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 64u;
}

fn desingularize(h: f32, q: f32) -> f32 {
    let h4 = h * h * h * h;
    if h4 >= params.epsilon {
        return q / h;
    }
    return SQRT_2 * h * q / sqrt(h4 + params.epsilon);
}

fn speed_bound(h: f32, hu: f32, hv: f32) -> f32 {
    let flow = length(vec2<f32>(desingularize(h, hu), desingularize(h, hv)));
    return flow + sqrt(params.gravity * h);
}

fn decomposition_node(j: u32) -> f32 {
    if j == 0u {
        return 0.0;
    }
    return params.depths[j - 1u];
}

fn airy_node(j: u32) -> f32 {
    return params.depths[j];
}

fn node_weight(decomposition: bool, j: u32, depth: f32) -> f32 {
    var count = params.depth_count;
    if decomposition {
        count = count + 1u;
    }
    let last = count - 1u;

    var first_value = airy_node(0u);
    var last_value = airy_node(last);
    if decomposition {
        first_value = decomposition_node(0u);
        last_value = decomposition_node(last);
    }

    var bracket = 0u;
    var t = 0.0;

    if depth <= first_value {
        bracket = 0u;
    } else if depth >= last_value {
        bracket = last;
    } else {
        for (var n = 0u; n < last; n = n + 1u) {
            var lo = airy_node(n);
            var hi = airy_node(n + 1u);
            if decomposition {
                lo = decomposition_node(n);
                hi = decomposition_node(n + 1u);
            }
            if depth >= lo && depth < hi {
                bracket = n;
                t = (depth - lo) / (hi - lo);
            }
        }
    }

    if j == bracket {
        return 1.0 - t;
    }
    if j == bracket + 1u {
        return t;
    }
    return 0.0;
}

fn wavevector(idx: u32) -> vec2<f32> {
    let a = idx % params.ex;
    let b = idx / params.ex;
    let lx = 2.0 * f32(params.nx) * params.dx;
    let lz = 2.0 * f32(params.nz) * params.dz;
    return vec2<f32>(TAU * signed_index(a, params.ex) / lx, TAU * signed_index(b, params.ez) / lz);
}

fn is_nyquist(idx: u32) -> bool {
    return idx % params.ex == params.ex / 2u || idx / params.ex == params.ez / 2u;
}

fn ext_mirror(idx: u32) -> u32 {
    let a = idx % params.ex;
    let b = idx / params.ex;
    return ((params.ez - b) % params.ez) * params.ex + (params.ex - a) % params.ex;
}

struct Source {
    cell: u32,
    flip_x: bool,
    flip_z: bool,
}

fn ext_source(idx: u32) -> Source {
    let a = idx % params.ex;
    let b = idx / params.ex;
    var i = a;
    var k = b;
    var flip_x = false;
    var flip_z = false;
    if a >= params.nx {
        i = 2u * params.nx - 1u - a;
        flip_x = true;
    }
    if b >= params.nz {
        k = 2u * params.nz - 1u - b;
        flip_z = true;
    }
    return Source(cell_index(i, k), flip_x, flip_z);
}

fn odd(flip: bool) -> f32 {
    return select(1.0, -1.0, flip);
}

fn level_of(c: u32) -> f32 {
    return bed_level(bed[c], total[c].x);
}

@compute @workgroup_size(64)
fn decompose_pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let idx = global_index(gid, groups);
    if idx >= ext_count() {
        return;
    }
    let s = ext_source(idx);
    let u = total[s.cell];
    var mask = 0.0;
    if u.x > 0.0 {
        mask = 1.0;
    }
    spectrum[idx] = vec2<f32>(level_of(s.cell) * mask, mask);
    spectrum[ext_count() + idx] = vec2<f32>(odd(s.flip_x) * u.y, odd(s.flip_z) * u.z);
}

@compute @workgroup_size(64)
fn decompose_filter(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let idx = global_index(gid, groups);
    if idx >= ext_count() {
        return;
    }
    let sigma = params.filter_width * decomposition_node(node.index);
    let k = wavevector(idx);
    let f = exp(-0.5 * sigma * sigma * dot(k, k));
    work[idx] = spectrum[idx] * f;
    work[ext_count() + idx] = spectrum[ext_count() + idx] * f;
}

fn identity_of(c: u32) -> vec3<f32> {
    let u = total[c];
    return vec3<f32>(level_of(c), u.y, u.z);
}

@compute @workgroup_size(64)
fn decompose_begin(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let w = node_weight(true, 0u, total[c].x);
    cells[cell_slot(SMOOTH, c)] = vec4<f32>(w * identity_of(c), 0.0);
}

@compute @workgroup_size(64)
fn decompose_accumulate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let w = node_weight(true, node.index, total[c].x);
    if w == 0.0 {
        return;
    }
    let e = ext_index(c % params.nx, c / params.nx);
    let scale = 1.0 / f32(ext_count());
    let a = work[e] * scale;
    let b = work[ext_count() + e] * scale;
    var value = identity_of(c);
    if a.y > MASK_FLOOR {
        value = vec3<f32>(a.x, b.x, b.y) / a.y;
    }
    let slot = cell_slot(SMOOTH, c);
    cells[slot] = cells[slot] + vec4<f32>(w * value, 0.0);
}

fn gradient_penalty(i: u32, k: u32) -> f32 {
    let il = select(i - 1u, 0u, i == 0u);
    let ir = min(i + 1u, params.nx - 1u);
    let kl = select(k - 1u, 0u, k == 0u);
    let kr = min(k + 1u, params.nz - 1u);
    var gx = 0.0;
    var gz = 0.0;
    if ir > il {
        gx = (total[cell_index(ir, k)].x - total[cell_index(il, k)].x) / (f32(ir - il) * params.dx);
    }
    if kr > kl {
        gz = (total[cell_index(i, kr)].x - total[cell_index(i, kl)].x) / (f32(kr - kl) * params.dz);
    }
    return exp(-params.gradient_penalty * (gx * gx + gz * gz));
}

@compute @workgroup_size(64)
fn decompose_finish(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let u = total[c];
    var b = vec4<f32>(0.0);

    if u.x > 0.0 {
        let identity = identity_of(c);
        let smoothed = cells[cell_slot(SMOOTH, c)].xyz;
        let p = gradient_penalty(c % params.nx, c / params.nx);
        let mixed = identity + p * (smoothed - identity);
        let h = bed_depth(bed[c], mixed.x);
        if h > 0.0 {
            let limit = h * speed_bound(u.x, u.y, u.z);
            let scale = min(limit / max(length(mixed.yz), 1e-30), 1.0);
            b = vec4<f32>(h, scale * mixed.y, scale * mixed.z, 0.0);
        }
    }

    bulk[c] = b;
    bulk_start[c] = b;
    surface[c] = vec4<f32>(u.xyz - b.xyz, 0.0);
}

@compute @workgroup_size(64)
fn airy_pack(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let idx = global_index(gid, groups);
    if idx >= ext_count() {
        return;
    }
    let s = ext_source(idx);
    let v = surface[s.cell];
    spectrum[idx] = vec2<f32>(v.x, odd(s.flip_x) * v.y);
    spectrum[ext_count() + idx] = vec2<f32>(odd(s.flip_z) * v.z, 0.0);
}

fn conjugate(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x, -a.y);
}

fn half_cell_shift(theta: f32) -> vec2<f32> {
    var correction = 1.0;
    if abs(theta) > 1e-6 {
        correction = theta / sin(theta);
    }
    return correction * vec2<f32>(cos(theta), sin(theta));
}

@compute @workgroup_size(64)
fn airy_spectra(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let idx = global_index(gid, groups);
    if idx >= ext_count() {
        return;
    }

    let z1 = spectrum[idx];
    let z1m = conjugate(spectrum[ext_mirror(idx)]);
    let eta = 0.5 * (z1 + z1m);
    let qx = -times_i(0.5 * (z1 - z1m));
    let qz = spectrum[ext_count() + idx];

    let kv = wavevector(idx);
    let k = length(kv);

    if is_nyquist(idx) || k == 0.0 {
        work[idx] = qx + times_i(qz);
        work[ext_count() + idx] = vec2<f32>(0.0);
        return;
    }

    let depth = airy_node(node.index);
    let omega = sqrt(params.gravity * k * tanh_positive(k * depth));
    let dt = clock.dt;
    let s = sin(omega * dt);
    let c = cos(omega * dt);
    let n = kv / k;

    let p = qx * n.x + qz * n.y;
    let tx = qx - p * n.x;
    let tz = qz - p * n.y;

    let p_next = p * c - times_i(eta) * (omega / k * s);
    let p_integral = p * (s / omega) - times_i(eta) * ((1.0 - c) / k);

    let qx_next = p_next * n.x + tx;
    let qz_next = p_next * n.y + tz;
    let fx = complex_mul(p_integral * n.x + tx * dt, half_cell_shift(0.5 * kv.x * params.dx));
    let fz = complex_mul(p_integral * n.y + tz * dt, half_cell_shift(0.5 * kv.y * params.dz));

    work[idx] = qx_next + times_i(qz_next);
    work[ext_count() + idx] = fx + times_i(fz);
}

fn wet(c: u32) -> bool {
    return total[c].x > 0.0;
}

@compute @workgroup_size(64)
fn airy_begin(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let index = global_index(gid, groups);
    if index < cell_count() {
        cells[cell_slot(AIRY_Q, index)] = vec4<f32>(0.0);
    }
    if index < face_count() {
        faces[face_slot(AIRY_FLUX, index)] = 0.0;
    }
}

@compute @workgroup_size(64)
fn airy_accumulate_cells(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() || !wet(c) {
        return;
    }
    let w = node_weight(false, node.index, bulk_start[c].x);
    if w == 0.0 {
        return;
    }
    let q = work[ext_index(c % params.nx, c / params.nx)] / f32(ext_count());
    let slot = cell_slot(AIRY_Q, c);
    cells[slot] = cells[slot] + vec4<f32>(w * q, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn airy_accumulate_faces(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let f = global_index(gid, groups);
    if f >= face_count() {
        return;
    }

    var l: u32;
    var r: u32;
    var value: f32;
    let scale = 1.0 / f32(ext_count());

    if f < x_face_count() {
        let i = f % (params.nx + 1u);
        let k = f / (params.nx + 1u);
        if i == 0u || i == params.nx {
            return;
        }
        l = cell_index(i - 1u, k);
        r = cell_index(i, k);
        value = work[ext_count() + ext_index(i - 1u, k)].x * scale;
    } else {
        let g = f - x_face_count();
        let i = g % params.nx;
        let k = g / params.nx;
        if k == 0u || k == params.nz {
            return;
        }
        l = cell_index(i, k - 1u);
        r = cell_index(i, k);
        value = work[ext_count() + ext_index(i, k - 1u)].y * scale;
    }

    if !(wet(l) && wet(r)) {
        return;
    }

    let w = node_weight(false, node.index, 0.5 * (bulk_start[l].x + bulk_start[r].x));
    let slot = face_slot(AIRY_FLUX, f);
    faces[slot] = faces[slot] + w * value;
}

fn mean_velocity(c: u32) -> vec2<f32> {
    let a = bulk_start[c];
    let b = bulk[c];
    let ua = vec2<f32>(desingularize(a.x, a.y), desingularize(a.x, a.z));
    let ub = vec2<f32>(desingularize(b.x, b.y), desingularize(b.x, b.z));
    return 0.5 * (ua + ub);
}

fn sample_slot(slot: u32, use_surface: bool, x: f32, z: f32) -> vec4<f32> {
    let fx = clamp(x / params.dx - 0.5, 0.0, f32(params.nx - 1u));
    let fz = clamp(z / params.dz - 0.5, 0.0, f32(params.nz - 1u));
    let i0 = min(u32(floor(fx)), params.nx - 1u);
    let k0 = min(u32(floor(fz)), params.nz - 1u);
    let i1 = min(i0 + 1u, params.nx - 1u);
    let k1 = min(k0 + 1u, params.nz - 1u);
    let tx = fx - f32(i0);
    let tz = fz - f32(k0);

    var v00: vec4<f32>;
    var v10: vec4<f32>;
    var v01: vec4<f32>;
    var v11: vec4<f32>;
    if use_surface {
        v00 = surface[cell_index(i0, k0)];
        v10 = surface[cell_index(i1, k0)];
        v01 = surface[cell_index(i0, k1)];
        v11 = surface[cell_index(i1, k1)];
    } else {
        v00 = cells[cell_slot(slot, cell_index(i0, k0))];
        v10 = cells[cell_slot(slot, cell_index(i1, k0))];
        v01 = cells[cell_slot(slot, cell_index(i0, k1))];
        v11 = cells[cell_slot(slot, cell_index(i1, k1))];
    }

    let bottom = v00 * (1.0 - tx) + v10 * tx;
    let top = v01 * (1.0 - tx) + v11 * tx;
    return bottom * (1.0 - tz) + top * tz;
}

@compute @workgroup_size(64)
fn transport_faces(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let f = global_index(gid, groups);
    if f >= face_count() {
        return;
    }

    let dt = clock.dt;
    let slot = face_slot(TRANSPORT, f);
    faces[slot] = 0.0;

    if f < x_face_count() {
        let i = f % (params.nx + 1u);
        let k = f / (params.nx + 1u);
        if i == 0u || i == params.nx {
            return;
        }
        let l = cell_index(i - 1u, k);
        let r = cell_index(i, k);
        if !(wet(l) && wet(r)) {
            return;
        }
        let u = 0.5 * (mean_velocity(l) + mean_velocity(r));
        let x = f32(i) * params.dx - 0.5 * dt * u.x;
        let z = (f32(k) + 0.5) * params.dz - 0.5 * dt * u.y;
        faces[slot] = dt * u.x * sample_slot(0u, true, x, z).x;
    } else {
        let g = f - x_face_count();
        let i = g % params.nx;
        let k = g / params.nx;
        if k == 0u || k == params.nz {
            return;
        }
        let l = cell_index(i, k - 1u);
        let r = cell_index(i, k);
        if !(wet(l) && wet(r)) {
            return;
        }
        let u = 0.5 * (mean_velocity(l) + mean_velocity(r));
        let x = (f32(i) + 0.5) * params.dx - 0.5 * dt * u.x;
        let z = f32(k) * params.dz - 0.5 * dt * u.y;
        faces[slot] = dt * u.y * sample_slot(0u, true, x, z).x;
    }
}

fn velocity_divergence(i: u32, k: u32) -> f32 {
    let il = select(i - 1u, 0u, i == 0u);
    let ir = min(i + 1u, params.nx - 1u);
    let kl = select(k - 1u, 0u, k == 0u);
    let kr = min(k + 1u, params.nz - 1u);
    var du = 0.0;
    var dv = 0.0;
    if ir > il {
        du = (mean_velocity(cell_index(ir, k)).x - mean_velocity(cell_index(il, k)).x) / (f32(ir - il) * params.dx);
    }
    if kr > kl {
        dv = (mean_velocity(cell_index(i, kr)).y - mean_velocity(cell_index(i, kl)).y) / (f32(kr - kl) * params.dz);
    }
    return du + dv;
}

@compute @workgroup_size(64)
fn transport_amplify(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let d = velocity_divergence(c % params.nx, c / params.nx);
    let growth = min(-d, -params.damping * d);
    cells[cell_slot(AMPLIFIED, c)] = cells[cell_slot(AIRY_Q, c)] * exp(growth * clock.dt);
}

@compute @workgroup_size(64)
fn transport_advect(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    var q = vec4<f32>(0.0);
    if wet(c) {
        let i = c % params.nx;
        let k = c / params.nx;
        let u = mean_velocity(c);
        let x = (f32(i) + 0.5) * params.dx - clock.dt * u.x;
        let z = (f32(k) + 0.5) * params.dz - clock.dt * u.y;
        q = sample_slot(AMPLIFIED, false, x, z);
    }
    cells[cell_slot(SURFACE_Q, c)] = q;
}

@compute @workgroup_size(64)
fn combine_faces(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let f = global_index(gid, groups);
    if f >= face_count() {
        return;
    }
    faces[face_slot(COMBINED, f)] = accumulated[f] + faces[face_slot(AIRY_FLUX, f)] + faces[face_slot(TRANSPORT, f)];
}

fn combined_x(i: u32, k: u32) -> f32 {
    return faces[face_slot(COMBINED, x_face_index(i, k))];
}

fn combined_z(i: u32, k: u32) -> f32 {
    return faces[face_slot(COMBINED, z_face_index(i, k))];
}

@compute @workgroup_size(64)
fn combine_limit(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let i = c % params.nx;
    let k = c / params.nx;
    let out = (max(combined_x(i + 1u, k), 0.0) - min(combined_x(i, k), 0.0)) * params.dz
        + (max(combined_z(i, k + 1u), 0.0) - min(combined_z(i, k), 0.0)) * params.dx;
    let available = total[c].x * params.dx * params.dz;
    var factor = 1.0;
    if out > available {
        factor = available / out;
    }
    limits[c] = factor;
}

fn limited_x(i: u32, k: u32) -> f32 {
    let f = combined_x(i, k);
    if f > 0.0 && i > 0u {
        return f * limits[cell_index(i - 1u, k)];
    }
    if f <= 0.0 && i < params.nx {
        return f * limits[cell_index(i, k)];
    }
    return f;
}

fn limited_z(i: u32, k: u32) -> f32 {
    let f = combined_z(i, k);
    if f > 0.0 && k > 0u {
        return f * limits[cell_index(i, k - 1u)];
    }
    if f <= 0.0 && k < params.nz {
        return f * limits[cell_index(i, k)];
    }
    return f;
}

@compute @workgroup_size(64)
fn combine_apply(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let i = c % params.nx;
    let k = c / params.nx;
    let divergence = (limited_x(i + 1u, k) - limited_x(i, k)) / params.dx + (limited_z(i, k + 1u) - limited_z(i, k)) / params.dz;
    let h = max(total[c].x - divergence, 0.0);
    var next = vec4<f32>(0.0);
    if h > 0.0 {
        var keep = 0.0;
        if h > params.dry_depth {
            keep = 1.0;
        }
        let q = cells[cell_slot(SURFACE_Q, c)].xy;
        let cap = SURFACE_SPEED_LIMIT * h * sqrt(params.gravity * h);
        let scale = keep * min(cap / max(length(q), 1e-30), 1.0);
        let b = bulk[c];
        let q_total = b.yz + scale * q;
        let limit = h * speed_bound(max(b.x, h), b.y, b.z);
        let total = min(limit / max(length(q_total), 1e-30), 1.0);
        next = vec4<f32>(h, total * q_total, 0.0);
    }
    total[c] = next;
}

@compute @workgroup_size(64)
fn combine_record(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let f = global_index(gid, groups);
    if f >= face_count() {
        return;
    }
    if f < x_face_count() {
        accumulated[f] = limited_x(f % (params.nx + 1u), f / (params.nx + 1u));
    } else {
        let g = f - x_face_count();
        accumulated[f] = limited_z(g % params.nx, g / params.nx);
    }
}

const ETA: u32 = 0u;
const ETA_CURVATURE: u32 = 1u;
const FLOW: u32 = 2u;
const FLOW_CURVATURE: u32 = 3u;

fn profile_field(which: u32, c: u32) -> f32 {
    let scratch = profile[cell_count() + c];
    switch which {
        case 0u: { return surface[c].x; }
        case 1u: { return scratch.x; }
        case 2u: { return scratch.y; }
        default: { return scratch.z; }
    }
}

fn neighbours(i: u32, k: u32) -> vec4<u32> {
    return vec4<u32>(select(i - 1u, 0u, i == 0u), min(i + 1u, params.nx - 1u), select(k - 1u, 0u, k == 0u), min(k + 1u, params.nz - 1u));
}

fn central(lo: f32, hi: f32, a: u32, b: u32, spacing: f32) -> f32 {
    if b > a {
        return (hi - lo) / (f32(b - a) * spacing);
    }
    return 0.0;
}

fn field_gradient(which: u32, i: u32, k: u32) -> vec2<f32> {
    let n = neighbours(i, k);
    return vec2<f32>(
        central(profile_field(which, cell_index(n.x, k)), profile_field(which, cell_index(n.y, k)), n.x, n.y, params.dx),
        central(profile_field(which, cell_index(i, n.z)), profile_field(which, cell_index(i, n.w)), n.z, n.w, params.dz),
    );
}

fn field_laplacian(which: u32, i: u32, k: u32) -> f32 {
    let n = neighbours(i, k);
    let centre = profile_field(which, cell_index(i, k));
    let x = profile_field(which, cell_index(n.y, k)) - 2.0 * centre + profile_field(which, cell_index(n.x, k));
    let z = profile_field(which, cell_index(i, n.w)) - 2.0 * centre + profile_field(which, cell_index(i, n.z));
    return x / (params.dx * params.dx) + z / (params.dz * params.dz);
}

@compute @workgroup_size(64)
fn profile_flow(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let i = c % params.nx;
    let k = c / params.nx;
    let n = neighbours(i, k);
    let flow = central(surface[cell_index(n.x, k)].y, surface[cell_index(n.y, k)].y, n.x, n.y, params.dx)
        + central(surface[cell_index(i, n.z)].z, surface[cell_index(i, n.w)].z, n.z, n.w, params.dz);
    profile[cell_count() + c] = vec4<f32>(field_laplacian(ETA, i, k), flow, 0.0, 0.0);
}

@compute @workgroup_size(64)
fn profile_curvature(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let slot = cell_count() + c;
    profile[slot].z = field_laplacian(FLOW, c % params.nx, c / params.nx);
}

fn mode_energies(field: u32, curvature: u32, i: u32, k: u32) -> vec2<f32> {
    let c = cell_index(i, k);
    let slope = field_gradient(field, i, k);
    let bend = field_gradient(curvature, i, k);
    let value = profile_field(field, c);
    let curve = profile_field(curvature, c);
    return vec2<f32>(dot(slope, slope) - value * curve, curve * curve - dot(slope, bend));
}

fn local_wavenumber(a: f32, b: f32, spacing: f32) -> f32 {
    if a <= 0.0 || b <= 0.0 {
        return 0.0;
    }
    let half = 0.5 * spacing * sqrt(b / a);
    return 2.0 / spacing * asin(min(half, 1.0));
}

@compute @workgroup_size(64)
fn profile_modes(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let h = total[c].x;
    if h <= params.dry_depth {
        profile[c] = vec4<f32>(0.0);
        return;
    }
    let i = c % params.nx;
    let k = c / params.nx;
    let eta = mode_energies(ETA, ETA_CURVATURE, i, k);
    let flow = mode_energies(FLOW, FLOW_CURVATURE, i, k);
    let weight = params.gravity / h;
    let energies = weight * eta + flow;
    let s = surface[c];
    profile[c] = vec4<f32>(local_wavenumber(energies.x, energies.y, 0.5 * (params.dx + params.dz)), s.y, s.z, profile[cell_count() + c].y);
}
