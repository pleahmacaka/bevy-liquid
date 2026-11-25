const WG: u32 = 256u;
const COARSE_ROWS: u32 = 16u;
const TILE: u32 = 10u;
const COARSE_LIMIT: u32 = 1024u;
const MASS_QUANTUM: f32 = 5.9604645e-8;
const TEMPERATURE_SCALE: f32 = 256.0;
const VELOCITY_SCALE: f32 = 4096.0;
const VOLUME_SCALE: f32 = 65536.0;
const WORD: f32 = 4294967296.0;
const HOPS: u32 = 8u;
const WEIGHT_SCALE: f32 = 65536.0;

struct Emitter {
    center: vec4<f32>,
    extent: vec4<f32>,
    rates: vec4<f32>,
}

struct Params {
    dims: vec4<u32>,
    nodes: vec4<u32>,
    faces: vec4<u32>,
    counts: vec4<u32>,
    groups: vec4<u32>,
    grid: vec4<f32>,
    flow: vec4<f32>,
    loss: vec4<f32>,
    solve: vec4<f32>,
    emitters: array<Emitter, 8>,
}

struct Level {
    dims: vec4<u32>,
    coarse: vec4<u32>,
    span: vec4<u32>,
}

struct State {
    dt: f32,
    remaining: f32,
    time: f32,
    running: u32,
    vmax: atomic<u32>,
    divergence: atomic<u32>,
    substeps: u32,
    iterations: u32,
    converged: u32,
    residual: f32,
    rz: f32,
    injected_lo: atomic<u32>,
    injected_hi: atomic<u32>,
    accepted: atomic<u32>,
    rejected: atomic<u32>,
    pad: u32,
}

struct Injection {
    position_mass: vec4<f32>,
    velocity_temperature: vec4<f32>,
    volume: f32,
}

struct Injections {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    items: array<Injection>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: State;
@group(0) @binding(2) var<storage, read_write> density: array<f32>;
@group(0) @binding(3) var<storage, read_write> temperature: array<f32>;
@group(0) @binding(4) var<storage, read_write> velocity: array<f32>;
@group(0) @binding(5) var<storage, read_write> density_next: array<f32>;
@group(0) @binding(6) var<storage, read_write> temperature_next: array<f32>;
@group(0) @binding(7) var<storage, read_write> velocity_next: array<f32>;
@group(0) @binding(8) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(9) var<storage, read_write> flags: array<u32>;
@group(0) @binding(10) var<storage, read_write> coef: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> mg_x: array<f32>;
@group(0) @binding(12) var<storage, read_write> mg_b: array<f32>;
@group(0) @binding(13) var<storage, read_write> krylov: array<f32>;
@group(0) @binding(14) var<storage, read_write> partials: array<f32>;
@group(0) @binding(15) var<storage, read_write> inverse: array<f32>;
@group(0) @binding(16) var<storage, read_write> injections: Injections;
@group(0) @binding(17) var<storage, read_write> accum: array<atomic<u32>>;
@group(0) @binding(18) var<storage, read_write> source: array<f32>;
@group(1) @binding(0) var<uniform> level: Level;
@group(2) @binding(0) var<storage, read_write> indirect: array<u32>;
@group(3) @binding(0) var volume: texture_storage_3d<rgba16float, write>;

override SLOT_COUNT: u32 = 4u;
override SOLVE_FIRST: u32 = 2u;
override BEGIN: bool = false;
override LAST: bool = false;
override SHARE: f32 = 0.0;
override WARM: bool = false;
override EXPANSION: bool = false;
override INITIAL: bool = false;
override MACCORMACK: bool = true;
override VELOCITY_MACCORMACK: bool = true;
override COLOR: u32 = 0u;
override RZ: bool = false;
override INFLOW_AT_REST: bool = false;

var<workgroup> reduction: array<f32, 256>;
var<workgroup> group_max: atomic<u32>;
var<workgroup> coarse_rhs: array<f32, COARSE_LIMIT>;
var<workgroup> tile: array<f32, 1000>;

fn unit(a: u32) -> vec3<u32> {
    var e = vec3<u32>(0u);
    e[a] = 1u;
    return e;
}

fn face_offset(a: u32) -> vec3<f32> {
    var o = vec3<f32>(0.5);
    o[a] = 0.0;
    return o;
}

fn cell_id(c: vec3<u32>) -> u32 {
    return c.x + params.dims.x * (c.y + params.dims.y * c.z);
}

fn cell_coords(n: u32) -> vec3<u32> {
    let d = params.dims;
    return vec3<u32>(n % d.x, (n / d.x) % d.y, n / (d.x * d.y));
}

fn node_coords(n: u32) -> vec3<u32> {
    let d = params.nodes;
    return vec3<u32>(n % d.x, (n / d.x) % d.y, n / (d.x * d.y));
}

fn face_dims(a: u32) -> vec3<u32> {
    return params.dims.xyz + unit(a);
}

fn face_base(a: u32) -> u32 {
    if a == 0u {
        return 0u;
    }

    if a == 1u {
        return params.faces.x;
    }

    return params.faces.y;
}

fn face_id(a: u32, f: vec3<u32>) -> u32 {
    let d = face_dims(a);
    return face_base(a) + f.x + d.x * (f.y + d.y * f.z);
}

fn is_cell(p: vec3<u32>) -> bool {
    return all(p < params.dims.xyz);
}

fn is_face(a: u32, p: vec3<u32>) -> bool {
    return all(p < face_dims(a));
}

fn below(c: vec3<u32>, a: u32) -> vec3<u32> {
    var lo = c;
    lo[a] = max(c[a], 1u) - 1u;
    return lo;
}

fn solid(n: u32) -> bool {
    return (flags[n] & 1u) != 0u;
}

fn sealed(n: u32) -> bool {
    return (flags[n] & 128u) != 0u;
}

fn cell_open(c: vec3<u32>) -> bool {
    return !solid(cell_id(c));
}

fn face_open(a: u32, f: vec3<u32>) -> bool {
    if f[a] < params.dims[a] {
        return (flags[cell_id(f)] & (2u << (2u * a))) != 0u;
    }

    return (flags[cell_id(f - unit(a))] & (4u << (2u * a))) != 0u;
}

struct Stencil {
    lo: vec3<u32>,
    hi: vec3<u32>,
    t: vec3<f32>,
}

fn stencil(d: vec3<u32>, g: vec3<f32>) -> Stencil {
    let top = max(vec3<f32>(d) - 1.0, vec3<f32>(0.0));
    let x = clamp(g, vec3<f32>(0.0), top);
    let base = min(floor(x), max(top - 1.0, vec3<f32>(0.0)));
    let lo = vec3<u32>(base);

    return Stencil(lo, min(lo + 1u, d - 1u), x - base);
}

fn picks(k: u32) -> vec3<bool> {
    return (vec3<u32>(k, k >> 1u, k >> 2u) & vec3<u32>(1u)) == vec3<u32>(1u);
}

fn corner(s: Stencil, k: u32) -> vec3<u32> {
    return select(s.lo, s.hi, picks(k));
}

fn weight(s: Stencil, k: u32) -> f32 {
    let w = select(1.0 - s.t, s.t, picks(k));
    return w.x * w.y * w.z;
}

struct Fetch {
    lo: vec3<u32>,
    reach: vec3<u32>,
    base: u32,
    stride: vec3<u32>,
    t: vec3<f32>,
}

fn fetch(d: vec3<u32>, g: vec3<f32>) -> Fetch {
    let top = max(vec3<f32>(d) - 1.0, vec3<f32>(0.0));
    let x = clamp(g, vec3<f32>(0.0), top);
    let base = min(floor(x), max(top - 1.0, vec3<f32>(0.0)));
    let lo = vec3<u32>(base);
    let reach = min(lo + 1u, d - 1u) - lo;
    let stride = reach * vec3<u32>(1u, d.x, d.x * d.y);

    return Fetch(lo, reach, lo.x + d.x * (lo.y + d.y * lo.z), stride, x - base);
}

fn offsets(f: Fetch, base: u32) -> array<u32, 8> {
    let b = base + f.base;
    let s = f.stride;

    return array<u32, 8>(b, b + s.x, b + s.y, b + s.x + s.y, b + s.z, b + s.x + s.z, b + s.y + s.z, b + s.x + s.y + s.z);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    return a + (b - a) * t;
}

fn blend(c: array<f32, 8>, t: vec3<f32>) -> f32 {
    let y0 = lerp(lerp(c[0], c[1], t.x), lerp(c[2], c[3], t.x), t.y);
    let y1 = lerp(lerp(c[4], c[5], t.x), lerp(c[6], c[7], t.x), t.y);

    return lerp(y0, y1, t.z);
}

fn bounds(c: array<f32, 8>) -> vec2<f32> {
    let lo = min(min(min(c[0], c[1]), min(c[2], c[3])), min(min(c[4], c[5]), min(c[6], c[7])));
    let hi = max(max(max(c[0], c[1]), max(c[2], c[3])), max(max(c[4], c[5]), max(c[6], c[7])));

    return vec2<f32>(lo, hi);
}

fn velocity_corners(a: u32, f: Fetch) -> array<f32, 8> {
    let o = offsets(f, face_base(a));

    return array<f32, 8>(velocity[o[0]], velocity[o[1]], velocity[o[2]], velocity[o[3]], velocity[o[4]], velocity[o[5]], velocity[o[6]], velocity[o[7]]);
}

fn scratch_corners(f: Fetch, base: u32) -> array<f32, 8> {
    let o = offsets(f, base);

    return array<f32, 8>(scratch[o[0]], scratch[o[1]], scratch[o[2]], scratch[o[3]], scratch[o[4]], scratch[o[5]], scratch[o[6]], scratch[o[7]]);
}

fn sample_velocity(a: u32, g: vec3<f32>) -> f32 {
    let f = fetch(face_dims(a), g - face_offset(a));
    return blend(velocity_corners(a, f), f.t);
}

fn velocity_at(g: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(sample_velocity(0u, g), sample_velocity(1u, g), sample_velocity(2u, g));
}

fn traced(g: vec3<f32>, s: f32) -> vec3<f32> {
    var x = g;
    var left = s;

    for (var hop = 1u; hop <= HOPS; hop = hop + 1u) {
        let u = velocity_at(x);
        let speed = max(abs(u.x), max(abs(u.y), abs(u.z)));
        var h = left;

        if hop < HOPS && abs(left) * speed > 1.0 {
            h = sign(left) / speed;
        }

        x = x - h * velocity_at(x - 0.5 * h * u);
        left = left - h;

        if left == 0.0 {
            break;
        }
    }

    return x;
}

fn backtrace(g: vec3<f32>, s: f32) -> vec3<f32> {
    if EXPANSION {
        return traced(g, s);
    }

    let mid = g - 0.5 * s * velocity_at(g);
    return g - s * velocity_at(mid);
}

fn outside_share(d: vec3<u32>, g: vec3<f32>) -> f32 {
    let open = vec3<u32>(params.counts.w) >> vec3<u32>(0u, 1u, 2u);
    let under = select(vec3<f32>(0.0), -g, (open & vec3<u32>(1u)) == vec3<u32>(1u));
    let over = select(vec3<f32>(0.0), g - (vec3<f32>(d) - 1.0), (open & vec3<u32>(8u)) == vec3<u32>(8u));
    let reach = max(under, over);

    return clamp(max(reach.x, max(reach.y, reach.z)), 0.0, 1.0);
}

fn inside(e: Emitter, x: vec3<f32>) -> bool {
    let p = x - e.center.xyz;

    if e.center.w < 0.5 {
        return length(p) - e.extent.x <= 0.0;
    }

    let q = abs(p) - e.extent.xyz;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0) <= 0.0;
}

fn emitted(c: vec3<u32>) -> vec2<f32> {
    let n = cell_id(c);
    let x = (vec3<f32>(c) + 0.5) * params.grid.x;
    let dt = state.dt;
    var v = vec2<f32>(density[n], temperature[n]);

    if EXPANSION {
        v.x = v.x / copies(n);
    }

    for (var i = 0u; i < params.counts.x; i = i + 1u) {
        let e = params.emitters[i];

        if inside(e, x) {
            v = v + e.rates.xy * dt;
        }
    }

    return v;
}

struct Emitted {
    density: array<f32, 8>,
    temperature: array<f32, 8>,
}

fn emitted_corners(f: Fetch) -> Emitted {
    var result = Emitted();

    for (var k = 0u; k < 8u; k = k + 1u) {
        let pick = vec3<u32>(k, k >> 1u, k >> 2u) & vec3<u32>(1u);
        let v = emitted(f.lo + f.reach * pick);
        result.density[k] = v.x;
        result.temperature[k] = v.y;
    }

    return result;
}

fn ahead_cells(g: vec3<f32>) -> vec2<f32> {
    let f = fetch(params.dims.xyz, g);
    return vec2<f32>(blend(scratch_corners(f, 0u), f.t), blend(scratch_corners(f, params.dims.w), f.t));
}

fn ahead_face(a: u32, g: vec3<f32>) -> f32 {
    let f = fetch(face_dims(a), g);
    return blend(scratch_corners(f, 2u * params.dims.w + face_base(a)), f.t);
}

fn bounds_base() -> u32 {
    return 2u * params.dims.w + params.faces.z;
}

fn cell_bounds(n: u32) -> u32 {
    return bounds_base() + 4u * n;
}

fn face_bounds(f: u32) -> u32 {
    return bounds_base() + 4u * params.dims.w + 2u * f;
}

fn group_max_record(lid: u32, value: f32, target_slot: u32) {
    if lid == 0u {
        atomicStore(&group_max, 0u);
    }

    workgroupBarrier();
    atomicMax(&group_max, bitcast<u32>(value));
    workgroupBarrier();

    if lid == 0u {
        if target_slot == 0u {
            atomicMax(&state.vmax, atomicLoad(&group_max));
        } else {
            atomicMax(&state.divergence, atomicLoad(&group_max));
        }
    }
}

fn reduce_sum(lid: u32, value: f32) -> f32 {
    reduction[lid] = value;
    workgroupBarrier();

    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if lid < s {
            reduction[lid] = reduction[lid] + reduction[lid + s];
        }

        workgroupBarrier();
    }

    let total = reduction[0];
    workgroupBarrier();

    return total;
}

fn reduce_max(lid: u32, value: f32) -> f32 {
    reduction[lid] = value;
    workgroupBarrier();

    for (var s = WG / 2u; s > 0u; s = s >> 1u) {
        if lid < s {
            reduction[lid] = max(reduction[lid], reduction[lid + s]);
        }

        workgroupBarrier();
    }

    let total = reduction[0];
    workgroupBarrier();

    return total;
}

fn partial_sum(lid: u32, base: u32, count: u32) -> f32 {
    var acc = 0.0;

    for (var i = lid; i < count; i = i + WG) {
        acc = acc + partials[base + i];
    }

    return reduce_sum(lid, acc);
}

fn partial_max(lid: u32, base: u32, count: u32) -> f32 {
    var acc = 0.0;

    for (var i = lid; i < count; i = i + WG) {
        acc = max(acc, partials[base + i]);
    }

    return reduce_max(lid, acc);
}

fn rz_base() -> u32 {
    return 0u;
}

fn dq_base() -> u32 {
    return params.groups.z;
}

fn rmax_base() -> u32 {
    return 2u * params.groups.z;
}

fn write_slots(first: u32, on: bool) {
    for (var s = first; s < SLOT_COUNT; s = s + 1u) {
        indirect[3u * s] = select(0u, indirect[3u * (SLOT_COUNT + s)], on);
        indirect[3u * s + 1u] = 1u;
        indirect[3u * s + 2u] = 1u;
    }
}

@compute @workgroup_size(1)
fn gas_plan() {
    if BEGIN {
        state.remaining = params.grid.z;
        state.running = 1u;
        state.substeps = 0u;
    }

    let remaining = state.remaining;
    let on = state.running == 1u && remaining > 1.0e-12;

    if on {
        let vmax = bitcast<f32>(atomicLoad(&state.vmax));
        var dt = min(params.flow.x * params.grid.x / max(vmax, 1.0e-12), params.grid.w);

        if SHARE > 0.0 {
            dt = params.grid.z * SHARE;
        } else if LAST || dt >= remaining * (1.0 - 1.0e-5) {
            dt = remaining;
        } else if 2.0 * dt > remaining {
            dt = 0.5 * remaining;
        }

        state.dt = dt;
        state.remaining = remaining - dt;
        state.time = state.time + dt;
        state.substeps = state.substeps + 1u;
        state.converged = 0u;
        state.iterations = 0u;
        state.rz = 0.0;
        state.residual = 0.0;
        atomicStore(&state.vmax, 0u);
        atomicStore(&state.divergence, 0u);
    } else {
        state.running = 0u;
        state.dt = 0.0;
    }

    write_slots(0u, on);
}

@compute @workgroup_size(256)
fn gas_check(@builtin(local_invocation_index) lid: u32) {
    var rmax = 0.0;

    if !INITIAL {
        rmax = partial_max(lid, rmax_base(), params.groups.x);
    }

    if lid != 0u || state.running == 0u || state.converged == 1u {
        return;
    }

    let scale = params.grid.x * params.grid.x;
    let divergence = bitcast<f32>(atomicLoad(&state.divergence));

    if INITIAL {
        rmax = divergence * scale;
    } else {
        state.iterations = state.iterations + 1u;
    }

    state.residual = rmax / scale;

    if rmax <= params.solve.x * max(divergence, 1.0e-30) * scale {
        state.converged = 1u;
        write_slots(SOLVE_FIRST, false);
    }
}

fn weight_slot(n: u32) -> u32 {
    return 3u * (params.dims.w + params.faces.z) + 2u * params.dims.w + n;
}

fn copies(n: u32) -> f32 {
    if source[params.dims.w + n] <= 0.0 {
        return 1.0;
    }

    // a count below one must not amplify: with MacCormack the density then grows without bound
    return max(f32(atomicLoad(&accum[weight_slot(n)])) / WEIGHT_SCALE, 1.0);
}

fn expands(n: u32) -> bool {
    return source[n] > 0.0 && !sealed(n);
}

// interpolation spreads a source over its neighbours; counting copies in the source cell alone leaves 25 to 55 % extra mass
fn expands_nearby(c: vec3<u32>) -> bool {
    let lo = max(c, vec3<u32>(1u)) - 1u;
    let hi = min(c + 1u, params.dims.xyz - 1u);

    for (var z = lo.z; z <= hi.z; z = z + 1u) {
        for (var y = lo.y; y <= hi.y; y = y + 1u) {
            for (var x = lo.x; x <= hi.x; x = x + 1u) {
                if expands(cell_id(vec3<u32>(x, y, z))) {
                    return true;
                }
            }
        }
    }

    return false;
}

@compute @workgroup_size(256)
fn gas_weigh(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dims.w {
        return;
    }

    let p = cell_coords(id.x);
    let f = fetch(params.dims.xyz, backtrace(vec3<f32>(p) + 0.5, state.dt * params.grid.y) - 0.5);
    let o = offsets(f, 0u);

    for (var k = 0u; k < 8u; k = k + 1u) {
        let w = select(1.0 - f.t, f.t, picks(k));

        if source[params.dims.w + o[k]] > 0.0 {
            atomicAdd(&accum[weight_slot(o[k])], u32(round(w.x * w.y * w.z * WEIGHT_SCALE)));
        }
    }
}

@compute @workgroup_size(256)
fn gas_advect(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.nodes.w {
        return;
    }

    let p = node_coords(id.x);
    let s = state.dt * params.grid.y;
    let cells = params.dims.w;

    if is_cell(p) {
        let n = cell_id(p);
        let g = backtrace(vec3<f32>(p) + 0.5, s) - 0.5;
        let f = fetch(params.dims.xyz, g);
        let c = emitted_corners(f);
        var rho = bounds(c.density);
        var heat = bounds(c.temperature);
        var v = vec2<f32>(blend(c.density, f.t), blend(c.temperature, f.t));
        let b = cell_bounds(n);

        if INFLOW_AT_REST {
            v = mix(v, vec2<f32>(0.0, params.flow.y), outside_share(params.dims.xyz, g));
            rho = vec2<f32>(min(rho.x, v.x), max(rho.y, v.x));
            heat = vec2<f32>(min(heat.x, v.y), max(heat.y, v.y));
        }

        scratch[n] = v.x;
        scratch[cells + n] = v.y;
        scratch[b] = rho.x;
        scratch[b + 1u] = rho.y;
        scratch[b + 2u] = heat.x;
        scratch[b + 3u] = heat.y;
    }

    for (var a = 0u; a < 3u; a = a + 1u) {
        if is_face(a, p) {
            let o = face_offset(a);
            let i = face_id(a, p);
            let g = backtrace(vec3<f32>(p) + o, s) - o;
            let f = fetch(face_dims(a), g);
            let c = velocity_corners(a, f);
            var range = bounds(c);
            var u = blend(c, f.t);
            let b = face_bounds(i);

            if INFLOW_AT_REST {
                u = u * (1.0 - outside_share(face_dims(a), g));
                range = vec2<f32>(min(range.x, u), max(range.y, u));
            }

            scratch[2u * cells + i] = u;
            scratch[b] = range.x;
            scratch[b + 1u] = range.y;
        }
    }
}

@compute @workgroup_size(256)
fn gas_correct(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.nodes.w {
        return;
    }

    let p = node_coords(id.x);
    let dt = state.dt;
    let s = dt * params.grid.y;
    let cells = params.dims.w;

    if is_cell(p) {
        let n = cell_id(p);
        var v = vec2<f32>(scratch[n], scratch[cells + n]);

        if MACCORMACK {
            let b = cell_bounds(n);
            let behind = ahead_cells(backtrace(vec3<f32>(p) + 0.5, -s) - 0.5);
            let lo = vec2<f32>(scratch[b], scratch[b + 2u]);
            let hi = vec2<f32>(scratch[b + 1u], scratch[b + 3u]);
            v = clamp(v + 0.5 * (emitted(p) - behind), lo, hi);
        }

        let ambient = params.flow.y;
        var rho = v.x * exp(-params.loss.y * dt);
        var t = ambient + (v.y - ambient) * exp(-params.loss.z * dt);

        if solid(n) {
            rho = 0.0;
            t = ambient;
        }

        density_next[n] = rho;
        temperature_next[n] = t;
    }

    for (var a = 0u; a < 3u; a = a + 1u) {
        if is_face(a, p) {
            let i = face_id(a, p);
            var u = scratch[2u * cells + i];

            if MACCORMACK && VELOCITY_MACCORMACK {
                let o = face_offset(a);
                let b = face_bounds(i);
                let behind = ahead_face(a, backtrace(vec3<f32>(p) + o, -s) - o);
                u = clamp(u + 0.5 * (velocity[i] - behind), scratch[b], scratch[b + 1u]);
            }

            velocity_next[i] = u;
        }
    }
}

fn cell_velocity(c: vec3<u32>, component: u32) -> f32 {
    return 0.5 * (velocity_next[face_id(component, c)] + velocity_next[face_id(component, c + unit(component))]);
}

fn velocity_derivative(component: u32, c: vec3<u32>, a: u32) -> f32 {
    let lo = below(c, a);
    var hi = c;
    hi[a] = min(c[a] + 1u, params.dims[a] - 1u);
    let span = f32(hi[a] - lo[a]) * params.grid.x;

    if span == 0.0 {
        return 0.0;
    }

    return (cell_velocity(hi, component) - cell_velocity(lo, component)) / span;
}

@compute @workgroup_size(256)
fn gas_vorticity(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dims.w {
        return;
    }

    let c = cell_coords(id.x);
    let w = vec3<f32>(
        velocity_derivative(2u, c, 1u) - velocity_derivative(1u, c, 2u),
        velocity_derivative(0u, c, 2u) - velocity_derivative(2u, c, 0u),
        velocity_derivative(1u, c, 0u) - velocity_derivative(0u, c, 1u),
    );
    let n = 4u * id.x;

    scratch[n] = w.x;
    scratch[n + 1u] = w.y;
    scratch[n + 2u] = w.z;
    scratch[n + 3u] = length(w);
}

fn vorticity_magnitude(c: vec3<u32>) -> f32 {
    return scratch[4u * cell_id(c) + 3u];
}

fn magnitude_derivative(c: vec3<u32>, a: u32) -> f32 {
    let lo = below(c, a);
    var hi = c;
    hi[a] = min(c[a] + 1u, params.dims[a] - 1u);
    let span = f32(hi[a] - lo[a]) * params.grid.x;

    if span == 0.0 {
        return 0.0;
    }

    return (vorticity_magnitude(hi) - vorticity_magnitude(lo)) / span;
}

fn lift(c: vec3<u32>) -> f32 {
    let n = cell_id(c);
    return params.flow.w * (temperature_next[n] - params.flow.y) - params.flow.z * density_next[n];
}

@compute @workgroup_size(256)
fn gas_confine(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dims.w {
        return;
    }

    let c = cell_coords(id.x);
    let n = 4u * id.x;
    let eta = vec3<f32>(magnitude_derivative(c, 0u), magnitude_derivative(c, 1u), magnitude_derivative(c, 2u));
    let len = length(eta);
    let omega = vec3<f32>(scratch[n], scratch[n + 1u], scratch[n + 2u]);
    var dir = vec3<f32>(0.0);

    if len > 0.0 && len < 3.4e38 {
        dir = eta * (1.0 / len);
    }

    let f = params.loss.x * params.grid.x * cross(dir, omega);
    let m = 4u * params.dims.w + n;

    scratch[m] = f.x;
    scratch[m + 1u] = f.y;
    scratch[m + 2u] = f.z;
    scratch[m + 3u] = lift(c);
}

fn cell_force(c: vec3<u32>) -> vec4<f32> {
    let m = 4u * (params.dims.w + cell_id(c));
    return vec4<f32>(scratch[m], scratch[m + 1u], scratch[m + 2u], scratch[m + 3u]);
}

fn face_value(a: u32, f: vec3<u32>) -> f32 {
    if !face_open(a, f) {
        return 0.0;
    }

    let lo = below(f, a);
    var hi = f;
    hi[a] = min(f[a], params.dims[a] - 1u);

    let dt = state.dt;
    var u = velocity_next[face_id(a, f)];

    if params.loss.x > 0.0 {
        let flo = cell_force(lo);
        let fhi = cell_force(hi);

        if a == 1u {
            u = u + dt * 0.5 * (flo.w + fhi.w);
        }

        u = u + dt * 0.5 * (flo[a] + fhi[a]);
    } else if a == 1u {
        u = u + dt * 0.5 * (lift(lo) + lift(hi));
    }

    return u;
}

fn pressure_neighbors(c: vec3<u32>, n: u32, cf: vec4<f32>) -> f32 {
    let d = params.dims.xyz;
    let stride = vec3<u32>(1u, d.x, d.x * d.y);
    var s = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        if c[a] + 1u < d[a] {
            s = s + cf[a + 1u] * krylov[n + stride[a]];
        }

        if c[a] > 0u {
            let m = n - stride[a];
            s = s + coef[m][a + 1u] * krylov[m];
        }
    }

    return s;
}

fn zero_guess(n: u32, c: vec3<u32>, r: f32, diag: f32) {
    mg_x[n] = select(0.0, r / diag, diag > 0.0 && ((c.x + c.y + c.z) & 1u) == 0u);
}

@compute @workgroup_size(256)
fn gas_force(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    var divergence = 0.0;

    if id.x < params.nodes.w {
        let p = node_coords(id.x);
        var own = vec3<f32>(0.0);

        for (var a = 0u; a < 3u; a = a + 1u) {
            if is_face(a, p) {
                own[a] = face_value(a, p);
                velocity[face_id(a, p)] = own[a];
            }
        }

        if is_cell(p) {
            let n = cell_id(p);
            density[n] = density_next[n];
            temperature[n] = temperature_next[n];

            var flux = 0.0;

            for (var a = 0u; a < 3u; a = a + 1u) {
                flux = flux + (face_value(a, p + unit(a)) - own[a]);
            }

            if EXPANSION {
                let rate = select(0.0, source[n], expands(n));
                flux = flux - rate * params.grid.x;
                source[params.dims.w + n] = select(0.0, 1.0, expands_nearby(p));
                atomicStore(&accum[weight_slot(n)], 0u);
            }

            let cf = coef[n];
            let diag = cf.x;
            let fluid = diag > 0.0;
            var r = select(0.0, -flux * params.grid.x, fluid);

            if WARM {
                r = select(0.0, r - (diag * krylov[n] - pressure_neighbors(p, n, cf)), fluid);
            } else {
                krylov[n] = 0.0;
            }

            mg_b[n] = r;
            zero_guess(n, p, r, diag);
            divergence = select(0.0, abs(flux * params.grid.y), fluid);
        }
    }

    group_max_record(lid, divergence, 1u);
}

@compute @workgroup_size(256)
fn gas_gradient(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    var speed = 0.0;

    if id.x < params.nodes.w {
        let p = node_coords(id.x);

        for (var a = 0u; a < 3u; a = a + 1u) {
            if is_face(a, p) && face_open(a, p) {
                var dp = 0.0;

                if p[a] == 0u {
                    dp = 2.0 * krylov[cell_id(p)];
                } else if p[a] == params.dims[a] {
                    dp = -2.0 * krylov[cell_id(p - unit(a))];
                } else {
                    dp = krylov[cell_id(p)] - krylov[cell_id(p - unit(a))];
                }

                let f = face_id(a, p);
                let u = velocity[f] - dp * params.grid.y;
                velocity[f] = u;
                speed = max(speed, abs(u));
            }
        }
    }

    group_max_record(lid, speed, 0u);
}

fn level_cell(c: vec3<u32>) -> u32 {
    let d = level.dims;
    return d.w + c.x + d.x * (c.y + d.y * c.z);
}

fn neighbor_sum(c: vec3<u32>, n: u32, cf: vec4<f32>) -> f32 {
    let d = level.dims.xyz;
    let stride = vec3<u32>(1u, d.x, d.x * d.y);
    var s = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        if c[a] + 1u < d[a] {
            s = s + cf[a + 1u] * mg_x[n + stride[a]];
        }

        if c[a] > 0u {
            let m = n - stride[a];
            s = s + coef[m][a + 1u] * mg_x[m];
        }
    }

    return s;
}

fn colored(t: u32, color: u32) -> vec4<u32> {
    let d = level.dims.xyz;
    let hx = level.span.x;
    let i2 = t % hx;
    let j = (t / hx) % d.y;
    let k = t / (hx * d.y);
    let i = 2u * i2 + ((j + k + color) & 1u);

    return vec4<u32>(i, j, k, select(0u, 1u, i < d.x));
}

fn relax(c: vec3<u32>) -> f32 {
    let n = level_cell(c);
    let cf = coef[n];

    if cf.x <= 0.0 {
        return 0.0;
    }

    let z = (mg_b[n] + neighbor_sum(c, n, cf)) / cf.x;
    mg_x[n] = z;

    return z;
}

@compute @workgroup_size(256)
fn mg_smooth(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= level.span.y {
        return;
    }

    let c = colored(id.x, COLOR);

    if c.w == 1u {
        relax(c.xyz);
    }
}

@compute @workgroup_size(256)
fn mg_smooth_rz(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    var acc = 0.0;

    if id.x < level.span.y {
        let c = colored(id.x, 0u);

        if c.w == 1u {
            acc = acc + mg_b[level_cell(c.xyz)] * relax(c.xyz);
        }

        let o = vec3<u32>(c.x ^ 1u, c.y, c.z);

        if o.x < level.dims.x {
            let m = level_cell(o);
            acc = acc + mg_b[m] * mg_x[m];
        }
    }

    let total = reduce_sum(lid, acc);

    if lid == 0u {
        partials[rz_base() + wid.x] = total;
    }
}

fn wall_fold(a: u32, high: bool) -> f32 {
    let bit = select(a, a + 3u, high);
    return select(0.25, -0.25, ((params.counts.w >> bit) & 1u) == 1u);
}

fn transfer_weight(a: u32, o: u32, c: u32) -> f32 {
    if o == 0u || o == 3u {
        return 0.25;
    }

    if o == 1u && c == 0u {
        return 0.75 + wall_fold(a, false);
    }

    if o == 2u && c + 1u >= level.coarse[a] {
        return 0.75 + wall_fold(a, true);
    }

    return 0.75;
}

fn fine_residual(f: vec3<u32>) -> f32 {
    if ((f.x + f.y + f.z) & 1u) == 1u {
        return 0.0;
    }

    let n = level_cell(f);
    let cf = coef[n];

    if cf.x <= 0.0 {
        return 0.0;
    }

    return mg_b[n] - (cf.x * mg_x[n] - neighbor_sum(f, n, cf));
}

@compute @workgroup_size(256)
fn mg_restrict(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let cd = level.coarse.xyz;
    let blocks = (cd + 3u) / 4u;
    let block = vec3<u32>(wid.x % blocks.x, (wid.x / blocks.x) % blocks.y, wid.x / (blocks.x * blocks.y));
    let origin = vec3<i32>(block * 8u) - 1;
    let fd = vec3<i32>(level.dims.xyz);

    for (var i = lid; i < TILE * TILE * TILE; i = i + WG) {
        let f = origin + vec3<i32>(i32(i % TILE), i32((i / TILE) % TILE), i32(i / (TILE * TILE)));
        var r = 0.0;

        if all(f >= vec3<i32>(0)) && all(f < fd) {
            r = fine_residual(vec3<u32>(f));
        }

        tile[i] = r;
    }

    workgroupBarrier();

    let local = vec3<u32>(lid % 4u, (lid / 4u) % 4u, lid / 16u);
    let cc = block * 4u + local;

    if lid >= 64u || any(cc >= cd) {
        return;
    }

    let nc = level.coarse.w + cc.x + cd.x * (cc.y + cd.y * cc.z);
    let diag = coef[nc].x;
    var b = 0.0;

    if diag > 0.0 {
        for (var oz = 0u; oz < 4u; oz = oz + 1u) {
            for (var oy = 0u; oy < 4u; oy = oy + 1u) {
                for (var ox = 0u; ox < 4u; ox = ox + 1u) {
                    let t = 2u * local + vec3<u32>(ox, oy, oz);
                    let w = transfer_weight(0u, ox, cc.x) * transfer_weight(1u, oy, cc.y) * transfer_weight(2u, oz, cc.z);
                    b = b + w * tile[t.x + TILE * (t.y + TILE * t.z)];
                }
            }
        }

        b = b / 8.0;
    }

    mg_b[nc] = b;
    mg_x[nc] = select(0.0, b / diag, diag > 0.0 && ((cc.x + cc.y + cc.z) & 1u) == 0u);
}

fn correction(f: vec3<u32>) -> f32 {
    let cd = vec3<i32>(level.coarse.xyz);
    let fi = vec3<i32>(f);
    let parent = fi / 2;
    let dir = (fi - parent * 2) * 2 - 1;
    let partner = parent + dir;
    var near = vec3<f32>(0.75);
    var far = vec3<f32>(0.25);

    for (var a = 0u; a < 3u; a = a + 1u) {
        if partner[a] < 0 || partner[a] >= cd[a] {
            near[a] = 0.75 + wall_fold(a, partner[a] >= 0);
            far[a] = 0.0;
        }
    }

    var acc = 0.0;

    for (var k = 0u; k < 8u; k = k + 1u) {
        let pick = picks(k);
        let w = select(near, far, pick);

        if w.x * w.y * w.z == 0.0 {
            continue;
        }

        let c = parent + select(vec3<i32>(0), dir, pick);
        let m = level.coarse.w + u32(c.x + cd.x * (c.y + cd.y * c.z));
        acc = acc + w.x * w.y * w.z * mg_x[m];
    }

    return acc;
}

fn prolonged(f: vec3<u32>) -> f32 {
    let n = level_cell(f);

    if coef[n].x <= 0.0 {
        return mg_x[n];
    }

    return mg_x[n] + correction(f);
}

@compute @workgroup_size(256)
fn mg_prolong(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= level.span.y {
        return;
    }

    let c = colored(id.x, 1u);

    if c.w == 0u {
        return;
    }

    let n = level_cell(c.xyz);
    let cf = coef[n];

    if cf.x <= 0.0 {
        return;
    }

    let d = level.dims.xyz;
    var s = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        if c[a] + 1u < d[a] {
            s = s + cf[a + 1u] * prolonged(c.xyz + unit(a));
        }

        if c[a] > 0u {
            let lo = c.xyz - unit(a);
            s = s + coef[level_cell(lo)][a + 1u] * prolonged(lo);
        }
    }

    mg_x[n] = (mg_b[n] + s) / cf.x;
}

@compute @workgroup_size(256)
fn mg_coarse(
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let d = level.dims;
    let n = d.x * d.y * d.z;

    for (var j = lid; j < n; j = j + WG) {
        coarse_rhs[j] = mg_b[d.w + j];
    }

    workgroupBarrier();

    let parts = WG / COARSE_ROWS;
    let row = wid.x * COARSE_ROWS + lid % COARSE_ROWS;
    let part = lid / COARSE_ROWS;
    let chunk = (n + parts - 1u) / parts;
    var acc = 0.0;

    if row < n {
        for (var j = part * chunk; j < min(n, (part + 1u) * chunk); j = j + 1u) {
            acc = acc + inverse[j * n + row] * coarse_rhs[j];
        }
    }

    reduction[lid] = acc;
    workgroupBarrier();

    var rz = 0.0;

    if part == 0u && row < n {
        var z = 0.0;

        for (var q = 0u; q < parts; q = q + 1u) {
            z = z + reduction[lid + q * COARSE_ROWS];
        }

        mg_x[d.w + row] = z;
        rz = coarse_rhs[row] * z;
    }

    workgroupBarrier();

    if RZ {
        let total = reduce_sum(lid, rz);

        if lid == 0u {
            partials[rz_base() + wid.x] = total;
        }
    }
}

@compute @workgroup_size(256)
fn pcg_direction(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let rz_new = partial_sum(lid, rz_base(), params.groups.y);
    let rz_old = state.rz;
    let beta = select(0.0, rz_new / rz_old, rz_old > 0.0);
    let cells = params.dims.w;
    var dq = 0.0;

    if id.x < cells {
        let n = id.x;
        let c = cell_coords(n);
        let cf = coef[n];
        let z = mg_x[n];
        let az = select(0.0, cf.x * z - neighbor_sum(c, n, cf), cf.x > 0.0);
        var d = z;
        var q = az;

        if beta != 0.0 {
            d = z + beta * krylov[cells + n];
            q = az + beta * krylov[2u * cells + n];
        }

        krylov[cells + n] = d;
        krylov[2u * cells + n] = q;
        dq = d * q;
    }

    let total = reduce_sum(lid, dq);

    if lid == 0u {
        partials[dq_base() + wid.x] = total;
    }
}

@compute @workgroup_size(256)
fn pcg_update(
    @builtin(global_invocation_id) id: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let dq = partial_sum(lid, dq_base(), params.groups.x);
    let rz = partial_sum(lid, rz_base(), params.groups.y);
    let alpha = select(0.0, rz / dq, dq > 0.0 && rz > 0.0);
    let cells = params.dims.w;
    var rmax = 0.0;

    if id.x == 0u {
        state.rz = rz;
    }

    if id.x < cells {
        let n = id.x;
        let r = mg_b[n] - alpha * krylov[2u * cells + n];

        krylov[n] = krylov[n] + alpha * krylov[cells + n];
        mg_b[n] = r;
        zero_guess(n, cell_coords(n), r, coef[n].x);
        rmax = abs(r);
    }

    let total = reduce_max(lid, rmax);

    if lid == 0u {
        partials[rmax_base() + wid.x] = total;
    }
}

fn mul_wide(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xffffu;
    let a1 = a >> 16u;
    let b0 = b & 0xffffu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xffffu) + (p10 & 0xffffu);

    return vec2<u32>((p00 & 0xffffu) | (mid << 16u), p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u));
}

fn negate_wide(v: vec2<u32>) -> vec2<u32> {
    let lo = ~v.x + 1u;
    return vec2<u32>(lo, ~v.y + select(0u, 1u, lo == 0u));
}

fn add_wide(i: u32, v: vec2<u32>) {
    let old = atomicAdd(&accum[i], v.x);
    let hi = v.y + select(0u, 1u, old > 0xffffffffu - v.x);

    if hi != 0u {
        atomicAdd(&accum[i + 1u], hi);
    }
}

fn wide_to_f32(lo: u32, hi: u32) -> f32 {
    if (hi & 0x80000000u) != 0u {
        let m = negate_wide(vec2<u32>(lo, hi));
        return -(f32(m.y) * WORD + f32(m.x));
    }

    return f32(hi) * WORD + f32(lo);
}

fn mass_units(m: f32) -> u32 {
    return u32(min(round(m / MASS_QUANTUM), 4294967040.0));
}

fn volume_slot(n: u32) -> u32 {
    return 3u * (params.dims.w + params.faces.z) + 2u * n;
}

fn expansion_rate(n: u32, cell_volume: f32) -> f32 {
    let m = volume_slot(n);
    let added = wide_to_f32(atomicLoad(&accum[m]), atomicLoad(&accum[m + 1u])) * (MASS_QUANTUM / VOLUME_SCALE);
    let dt = params.grid.z;

    if dt <= 0.0 {
        return 0.0;
    }

    atomicStore(&accum[m], 0u);
    atomicStore(&accum[m + 1u], 0u);

    return added / (cell_volume * dt);
}

struct Spread {
    s: Stencil,
    w: array<f32, 8>,
    total: f32,
    last: u32,
}

fn spread_cells(g: vec3<f32>) -> Spread {
    let s = stencil(params.dims.xyz, g);
    var result = Spread(s, array<f32, 8>(), 0.0, 8u);

    for (var k = 0u; k < 8u; k = k + 1u) {
        let w = weight(s, k);

        if w > 0.0 && cell_open(corner(s, k)) {
            result.w[k] = w;
            result.total = result.total + w;
            result.last = k;
        }
    }

    return result;
}

fn spread_faces(a: u32, g: vec3<f32>) -> Spread {
    let s = stencil(face_dims(a), g);
    var result = Spread(s, array<f32, 8>(), 0.0, 8u);

    for (var k = 0u; k < 8u; k = k + 1u) {
        let w = weight(s, k);

        if w > 0.0 && face_open(a, corner(s, k)) {
            result.w[k] = w;
            result.total = result.total + w;
            result.last = k;
        }
    }

    return result;
}

fn shares(sp: Spread, units: u32) -> array<u32, 8> {
    var result = array<u32, 8>();
    var given = 0u;
    var cum = 0.0;

    for (var k = 0u; k < 8u; k = k + 1u) {
        if sp.w[k] > 0.0 {
            cum = cum + sp.w[k];
            var upto = units;

            if k != sp.last {
                upto = min(u32(floor(f32(units) * (cum / sp.total))), units);
            }

            result[k] = upto - given;
            given = upto;
        }
    }

    return result;
}

@compute @workgroup_size(256)
fn gas_inject(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x == 0u {
        atomicStore(&state.vmax, 0u);
    }

    let count = min(atomicLoad(&injections.count), params.counts.y);

    if id.x >= count {
        return;
    }

    let item = injections.items[id.x];
    let x = item.position_mass.xyz;
    let mass = item.position_mass.w;
    let size = vec3<f32>(params.dims.xyz) * params.grid.x;

    if !(mass > 0.0) || any(x < vec3<f32>(0.0)) || any(x > size) {
        atomicAdd(&state.rejected, 1u);
        return;
    }

    let g = x * params.grid.y;
    let cells = spread_cells(g - 0.5);

    if cells.last == 8u {
        atomicAdd(&state.rejected, 1u);
        return;
    }

    let units = mass_units(mass);
    let heat = u32(round(clamp(item.velocity_temperature.w, 0.0, 65535.0) * TEMPERATURE_SCALE));
    let expansion = u32(round(clamp(item.volume / mass, 0.0, 65535.0) * VOLUME_SCALE));
    let cell_shares = shares(cells, units);

    for (var k = 0u; k < 8u; k = k + 1u) {
        if cell_shares[k] > 0u {
            let c = cell_id(corner(cells.s, k));
            let n = 3u * c;
            atomicAdd(&accum[n], cell_shares[k]);
            add_wide(n + 1u, mul_wide(cell_shares[k], heat));

            if expansion > 0u {
                add_wide(volume_slot(c), mul_wide(cell_shares[k], expansion));
            }
        }
    }

    let face_start = 3u * params.dims.w;

    for (var a = 0u; a < 3u; a = a + 1u) {
        let faces = spread_faces(a, g - face_offset(a));

        if faces.last == 8u {
            continue;
        }

        let v = clamp(item.velocity_temperature[a] * VELOCITY_SCALE, -1.0e9, 1.0e9);
        let magnitude = u32(round(abs(v)));
        let face_shares = shares(faces, units);

        for (var k = 0u; k < 8u; k = k + 1u) {
            if face_shares[k] > 0u {
                let m = face_start + 3u * face_id(a, corner(faces.s, k));
                var momentum = mul_wide(face_shares[k], magnitude);

                if v < 0.0 {
                    momentum = negate_wide(momentum);
                }

                atomicAdd(&accum[m], face_shares[k]);
                add_wide(m + 1u, momentum);
            }
        }
    }

    let old = atomicAdd(&state.injected_lo, units);

    if old > 0xffffffffu - units {
        atomicAdd(&state.injected_hi, 1u);
    }

    atomicAdd(&state.accepted, 1u);
}

@compute @workgroup_size(256)
fn gas_apply(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    var speed = 0.0;

    if id.x < params.nodes.w {
        let p = node_coords(id.x);
        let volume = params.grid.x * params.grid.x * params.grid.x;
        let air = params.loss.w * volume;

        if is_cell(p) {
            let n = cell_id(p);
            let units = atomicLoad(&accum[3u * n]);
            var rate = 0.0;

            if units > 0u {
                let added = f32(units) * MASS_QUANTUM;
                let heat = wide_to_f32(atomicLoad(&accum[3u * n + 1u]), atomicLoad(&accum[3u * n + 2u])) * (MASS_QUANTUM / TEMPERATURE_SCALE);

                density[n] = density[n] + added / volume;
                temperature[n] = (air * temperature[n] + heat) / (air + added);
                atomicStore(&accum[3u * n], 0u);
                atomicStore(&accum[3u * n + 1u], 0u);
                atomicStore(&accum[3u * n + 2u], 0u);
                rate = expansion_rate(n, volume);
            }

            source[n] = rate;
        }

        for (var a = 0u; a < 3u; a = a + 1u) {
            if is_face(a, p) {
                let f = face_id(a, p);
                let m = 3u * (params.dims.w + f);
                let units = atomicLoad(&accum[m]);

                if units > 0u {
                    let added = f32(units) * MASS_QUANTUM;
                    let momentum = wide_to_f32(atomicLoad(&accum[m + 1u]), atomicLoad(&accum[m + 2u])) * (MASS_QUANTUM / VELOCITY_SCALE);

                    velocity[f] = (air * velocity[f] + momentum) / (air + added);
                    atomicStore(&accum[m], 0u);
                    atomicStore(&accum[m + 1u], 0u);
                    atomicStore(&accum[m + 2u], 0u);
                }

                speed = max(speed, abs(velocity[f]));
            }
        }
    }

    if id.x == 0u {
        atomicStore(&injections.count, 0u);
    }

    group_max_record(lid, speed, 0u);
}

@compute @workgroup_size(256)
fn gas_pack(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dims.w {
        return;
    }

    let c = cell_coords(id.x);
    textureStore(volume, vec3<i32>(c), vec4<f32>(density[id.x], temperature[id.x] - params.flow.y, 0.0, 0.0));
}
