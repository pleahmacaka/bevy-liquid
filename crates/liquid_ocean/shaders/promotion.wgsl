struct Params {
    nx: u32,
    nz: u32,
    window_x: u32,
    window_z: u32,
    dx: f32,
    dz: f32,
    gravity: f32,
    dt: f32,
    breaking: f32,
    acceleration: f32,
    slope: f32,
    dry_depth: f32,
    body_count: u32,
    history: u32,
    pad0: u32,
    pad1: u32,
    bodies: array<vec4<f32>, 8>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> bed: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> profile: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> history: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> indicators: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> table: array<u32>;
@group(0) @binding(7) var<storage, read_write> result: array<atomic<u32>>;

const LEVEL: u32 = 0u;
const FLUX_X: u32 = 1u;
const FLUX_Z: u32 = 2u;
const VERTICAL: u32 = 3u;
const SCORE_CAP: f32 = 4.0;
const SCORE_SCALE: f32 = 256.0;
const WINDOW_GROUP: u32 = 1024u;
const LINE_GROUP: u32 = 256u;

fn cell_count() -> u32 {
    return params.nx * params.nz;
}

fn cell_index(i: u32, k: u32) -> u32 {
    return k * params.nx + i;
}

fn global_index(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 64u;
}

fn wet(c: u32) -> bool {
    return state[c].x > params.dry_depth;
}

fn horizontal_profile(k: f32, z: f32, depth: f32) -> f32 {
    return (exp(k * z) + exp(-k * (z + 2.0 * depth))) / (1.0 - exp(-2.0 * k * depth));
}

fn crest_factor(k: f32, h: f32) -> f32 {
    if k * h < 1e-6 {
        return 1.0 / h;
    }
    return k * horizontal_profile(k, 0.0, h);
}

fn celerity(k: f32, h: f32) -> f32 {
    if k * h < 1e-6 {
        return sqrt(params.gravity * h);
    }
    return sqrt(params.gravity * tanh_positive(k * h) / k);
}

fn surface_speed(c: u32) -> vec2<f32> {
    if !wet(c) {
        return vec2<f32>(0.0);
    }
    let s = state[c];
    let m = profile[c];
    return (s.yz - m.yz) / s.x + crest_factor(m.x, s.x) * m.yz;
}

fn field(which: u32, c: u32) -> f32 {
    switch which {
        case 0u: { return bed_level(bed[c], state[c].x); }
        case 1u: { return state[c].y; }
        case 2u: { return state[c].z; }
        default: { return history[c].y; }
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

fn masked(which: u32, c: u32, own: u32) -> f32 {
    if wet(c) {
        return field(which, c);
    }
    return field(which, own);
}

fn masked_gradient(which: u32, i: u32, k: u32) -> vec2<f32> {
    let own = cell_index(i, k);
    let n = neighbours(i, k);
    return vec2<f32>(
        central(masked(which, cell_index(n.x, k), own), masked(which, cell_index(n.y, k), own), n.x, n.y, params.dx),
        central(masked(which, cell_index(i, n.z), own), masked(which, cell_index(i, n.w), own), n.z, n.w, params.dz),
    );
}

fn bed_gradient(i: u32, k: u32) -> vec2<f32> {
    let n = neighbours(i, k);
    return vec2<f32>(
        central(bed[cell_index(n.x, k)].x, bed[cell_index(n.y, k)].x, n.x, n.y, params.dx),
        central(bed[cell_index(i, n.z)].x, bed[cell_index(i, n.w)].x, n.z, n.w, params.dz),
    );
}

fn weight(score: f32) -> u32 {
    if score < 1.0 {
        return 0u;
    }
    return u32(floor(min(score, SCORE_CAP) * SCORE_SCALE + 0.5));
}

fn reset_result() {
    for (var s = 0u; s < 16u; s = s + 1u) {
        atomicStore(&result[s], 0u);
    }
    atomicStore(&result[1], 0xffffffffu);
    atomicStore(&result[2], 0xffffffffu);
}

@compute @workgroup_size(64)
fn promote_vertical(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c == 0u {
        reset_result();
    }
    if c >= cell_count() {
        return;
    }
    var w = 0.0;
    if wet(c) {
        let i = c % params.nx;
        let k = c / params.nx;
        let qx = masked_gradient(FLUX_X, i, k).x;
        let qz = masked_gradient(FLUX_Z, i, k).y;
        let slope = masked_gradient(LEVEL, i, k);
        w = -(qx + qz) + dot(surface_speed(c), slope);
    }
    history[c].y = w;
}

@compute @workgroup_size(64)
fn promote_cells(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }

    let previous = history[c];
    var out = vec4<f32>(0.0);

    if wet(c) {
        let i = c % params.nx;
        let k = c / params.nx;
        let h = state[c].x;
        let u = surface_speed(c);
        let speed = length(u);
        let m = profile[c];

        out.x = speed / celerity(m.x, h) / params.breaking;

        if params.history != 0u {
            let dw = masked_gradient(VERTICAL, i, k);
            let a = (previous.y - previous.x) / params.dt + dot(u, dw);
            out.y = abs(a) / params.gravity / params.acceleration;
        }

        if speed > 1e-3 {
            let slope = masked_gradient(LEVEL, i, k);
            let along = dot(u, slope) / speed;
            let drop = -dot(u, bed_gradient(i, k)) / speed;
            let steep = select(0.0, length(slope), along < 0.0);
            out.z = max(steep, drop) / params.slope;
        }

        let x = vec3<f32>((f32(i) + 0.5) * params.dx, bed_level(bed[c], h), (f32(k) + 0.5) * params.dz);

        for (var b = 0u; b < params.body_count; b = b + 1u) {
            let body = params.bodies[b];
            let d = x - body.xyz;
            if dot(d, d) < body.w * body.w {
                out.w = 1.0;
            }
        }

        let w = weight(max(max(out.x, out.y), max(out.z, out.w)));

        if w > 0u {
            atomicAdd(&result[0], 1u);
            atomicMin(&result[1], i);
            atomicMin(&result[2], k);
            atomicMax(&result[3], i + 1u);
            atomicMax(&result[4], k + 1u);
        }

        history[c].w = f32(w);
    } else {
        history[c].w = 0.0;
    }

    history[c].x = previous.y;

    indicators[c] = out;
}

var<workgroup> line: array<u32, LINE_GROUP>;

@compute @workgroup_size(256)
fn promote_rows(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let k = wid.x;
    let row = params.nx + 1u;

    if k == 0u {
        for (var i = lid; i < row; i = i + LINE_GROUP) {
            table[i] = 0u;
        }
    }

    if k >= params.nz {
        return;
    }

    if lid == 0u {
        table[(k + 1u) * row] = 0u;
    }

    var carry = 0u;

    for (var start = 0u; start < params.nx; start = start + LINE_GROUP) {
        let i = start + lid;
        var value = 0u;

        if i < params.nx {
            value = u32(history[cell_index(i, k)].w);
        }

        line[lid] = value;
        workgroupBarrier();

        for (var offset = 1u; offset < LINE_GROUP; offset = offset * 2u) {
            let add = select(0u, line[lid - offset], lid >= offset);
            workgroupBarrier();
            line[lid] = line[lid] + add;
            workgroupBarrier();
        }

        if i < params.nx {
            table[(k + 1u) * row + i + 1u] = carry + line[lid];
        }

        carry = carry + line[LINE_GROUP - 1u];
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn promote_columns(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let i = wid.x;
    let row = params.nx + 1u;
    var carry = 0u;

    for (var start = 0u; start < params.nz; start = start + LINE_GROUP) {
        let k = start + lid + 1u;
        var value = 0u;

        if k <= params.nz {
            value = table[k * row + i];
        }

        line[lid] = value;
        workgroupBarrier();

        for (var offset = 1u; offset < LINE_GROUP; offset = offset * 2u) {
            let add = select(0u, line[lid - offset], lid >= offset);
            workgroupBarrier();
            line[lid] = line[lid] + add;
            workgroupBarrier();
        }

        if k <= params.nz {
            table[k * row + i] = carry + line[lid];
        }

        carry = carry + line[LINE_GROUP - 1u];
        workgroupBarrier();
    }
}

var<workgroup> best_sum: array<u32, WINDOW_GROUP>;
var<workgroup> best_distance: array<i32, WINDOW_GROUP>;
var<workgroup> best_index: array<u32, WINDOW_GROUP>;

fn better(sum: u32, distance: i32, index: u32, other: u32) -> bool {
    let s = best_sum[other];
    let d = best_distance[other];
    return sum > s || (sum == s && (distance < d || (distance == d && index < best_index[other])));
}

@compute @workgroup_size(1024)
fn promote_window(@builtin(local_invocation_index) lid: u32) {
    let count = atomicLoad(&result[0]);
    let wx = clamp(params.window_x, 1u, params.nx);
    let wz = clamp(params.window_z, 1u, params.nz);
    let across = params.nx - wx + 1u;
    let positions = across * (params.nz - wz + 1u);
    let row = params.nx + 1u;
    let centre = vec2<i32>(
        i32(atomicLoad(&result[1]) + atomicLoad(&result[3])),
        i32(atomicLoad(&result[2]) + atomicLoad(&result[4])),
    );

    best_sum[lid] = 0u;
    best_distance[lid] = 0x7fffffff;
    best_index[lid] = 0xffffffffu;

    if count > 0u {
        for (var p = lid; p < positions; p = p + WINDOW_GROUP) {
            let i0 = p % across;
            let k0 = p / across;
            let sum = (table[(k0 + wz) * row + i0 + wx] + table[k0 * row + i0]) - (table[(k0 + wz) * row + i0] + table[k0 * row + i0 + wx]);
            let dx = i32(2u * i0 + wx) - centre.x;
            let dz = i32(2u * k0 + wz) - centre.y;
            let distance = dx * dx + dz * dz;
            let index = k0 * params.nx + i0;

            if better(sum, distance, index, lid) {
                best_sum[lid] = sum;
                best_distance[lid] = distance;
                best_index[lid] = index;
            }
        }
    }

    workgroupBarrier();

    for (var span = WINDOW_GROUP / 2u; span > 0u; span = span / 2u) {
        if lid < span && better(best_sum[lid + span], best_distance[lid + span], best_index[lid + span], lid) {
            best_sum[lid] = best_sum[lid + span];
            best_distance[lid] = best_distance[lid + span];
            best_index[lid] = best_index[lid + span];
        }
        workgroupBarrier();
    }

    if lid != 0u || count == 0u {
        return;
    }

    let i0 = best_index[0] % params.nx;
    let k0 = best_index[0] / params.nx;
    atomicStore(&result[5], i0);
    atomicStore(&result[6], k0);
    atomicStore(&result[7], i0 + wx);
    atomicStore(&result[8], k0 + wz);
    atomicStore(&result[9], best_sum[0]);
    atomicStore(&result[10], 1u);
}
