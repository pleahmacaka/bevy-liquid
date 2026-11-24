struct Params {
    nx: u32,
    nz: u32,
    dx: f32,
    dz: f32,
    gravity: f32,
    theta: f32,
    cfl: f32,
    epsilon: f32,
    dt_max: f32,
    frame_time: f32,
    sponge: u32,
    coupling_blend: f32,
    kinds: vec4<u32>,
    coupling_min: vec2<u32>,
    coupling_max: vec2<u32>,
    manning: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct Clock {
    dt: f32,
    remaining: f32,
    time: f32,
    elapsed: f32,
    rate: atomic<u32>,
    steps: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> state: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> stage: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> bed: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> face_bed: array<f32>;
@group(0) @binding(5) var<storage, read_write> flux: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> factors: array<f32>;
@group(0) @binding(7) var<storage, read_write> clock: Clock;
@group(0) @binding(8) var<storage, read> boundary: array<vec4<f32>>;
@group(0) @binding(9) var<storage, read_write> source: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> coupled: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read> sponge: array<vec4<f32>>;
@group(0) @binding(12) var<storage, read_write> accumulated: array<f32>;
@group(0) @binding(13) var<storage, read_write> ledger: array<f32>;
@group(1) @binding(0) var<storage, read_write> steps: array<u32>;

const WALL: u32 = 0u;
const TRANSMISSIVE: u32 = 1u;
const DIRICHLET: u32 = 2u;

const WEST: u32 = 0u;
const EAST: u32 = 1u;
const SOUTH: u32 = 2u;
const NORTH: u32 = 3u;

const SQRT_2: f32 = 1.41421356237;

struct CellState {
    h: f32,
    qn: f32,
    qt: f32,
    level: f32,
    full: bool,
}

struct Point {
    h: f32,
    qn: f32,
    qt: f32,
    un: f32,
    ut: f32,
}

struct Recon {
    minus: Point,
    plus: Point,
    dw: f32,
}

fn cell_index(i: u32, k: u32) -> u32 {
    return k * params.nx + i;
}

fn x_face_index(i: u32, k: u32) -> u32 {
    return k * (params.nx + 1u) + i;
}

fn z_face_index(i: u32, k: u32) -> u32 {
    return (params.nx + 1u) * params.nz + k * params.nx + i;
}

fn x_face_count() -> u32 {
    return (params.nx + 1u) * params.nz;
}

fn face_count() -> u32 {
    return x_face_count() + params.nx * (params.nz + 1u);
}

fn cell_count() -> u32 {
    return params.nx * params.nz;
}

fn load(from_stage: bool, c: u32) -> vec4<f32> {
    if from_stage {
        return stage[c];
    }
    return state[c];
}

fn desingularize(h: f32, q: f32) -> f32 {
    let h4 = h * h * h * h;
    if h4 >= params.epsilon {
        return q / h;
    }
    return SQRT_2 * h * q / sqrt(h4 + params.epsilon);
}

fn minmod(a: f32, b: f32, c: f32) -> f32 {
    if a > 0.0 && b > 0.0 && c > 0.0 {
        return min(min(a, b), c);
    }
    if a < 0.0 && b < 0.0 && c < 0.0 {
        return max(max(a, b), c);
    }
    return 0.0;
}

fn half_slope(m: f32, c: f32, p: f32) -> f32 {
    return 0.5 * minmod(params.theta * (c - m), 0.5 * (p - m), params.theta * (p - c));
}

fn pressure(h: f32) -> f32 {
    return 0.5 * params.gravity * h * h;
}

fn make_point(h: f32, qn: f32, qt: f32) -> Point {
    let un = desingularize(h, qn);
    let ut = desingularize(h, qt);
    return Point(h, h * un, h * ut, un, ut);
}

fn velocity_point(h: f32, un: f32, ut: f32) -> Point {
    return Point(h, h * un, h * ut, un, ut);
}

fn mirrored(p: Point) -> Point {
    return Point(p.h, -p.qn, p.qt, -p.un, p.ut);
}

fn boundary_offset(side: u32) -> u32 {
    switch side {
        case 0u: { return 0u; }
        case 1u: { return params.nz; }
        case 2u: { return 2u * params.nz; }
        default: { return 2u * params.nz + params.nx; }
    }
}

fn boundary_bottom(side: u32, along: u32) -> f32 {
    switch side {
        case 0u: { return face_bed[x_face_index(0u, along)]; }
        case 1u: { return face_bed[x_face_index(params.nx, along)]; }
        case 2u: { return face_bed[z_face_index(along, 0u)]; }
        default: { return face_bed[z_face_index(along, params.nz)]; }
    }
}

fn interior_cell(from_stage: bool, axis: u32, i: u32, k: u32) -> CellState {
    let c = cell_index(i, k);
    let u = load(from_stage, c);
    let b = bed[c];
    let level = bed_level(b, u.x);
    let full = bed_full(b, u.x);
    if axis == 0u {
        return CellState(u.x, u.y, u.z, level, full);
    }
    return CellState(u.x, u.z, u.y, level, full);
}

fn cell_along(from_stage: bool, axis: u32, i: i32, k: i32) -> CellState {
    var along = i;
    var limit = i32(params.nx);
    if axis == 1u {
        along = k;
        limit = i32(params.nz);
    }

    if along >= 0 && along < limit {
        return interior_cell(from_stage, axis, u32(i), u32(k));
    }

    var side = WEST;
    var ii = i;
    var kk = k;
    if axis == 0u {
        if i < 0 { side = WEST; ii = 0; } else { side = EAST; ii = i32(params.nx) - 1; }
    } else {
        if k < 0 { side = SOUTH; kk = 0; } else { side = NORTH; kk = i32(params.nz) - 1; }
    }

    let inner = interior_cell(from_stage, axis, u32(ii), u32(kk));
    let kind = params.kinds[side];

    if kind == WALL {
        return CellState(inner.h, -inner.qn, inner.qt, inner.level, inner.full);
    }
    if kind == TRANSMISSIVE {
        return inner;
    }

    var row = u32(kk);
    if axis == 1u {
        row = u32(ii);
    }
    let d = boundary[boundary_offset(side) + row];
    var qn = d.y;
    var qt = d.z;
    if axis == 1u {
        qn = d.z;
        qt = d.y;
    }
    return CellState(d.x, qn, qt, d.x + boundary_bottom(side, row), true);
}

fn reconstruct(from_stage: bool, axis: u32, i: i32, k: i32) -> Recon {
    var b_minus: f32;
    var b_plus: f32;
    var m: CellState;
    var p: CellState;

    if axis == 0u {
        b_minus = face_bed[x_face_index(u32(i), u32(k))];
        b_plus = face_bed[x_face_index(u32(i) + 1u, u32(k))];
        m = cell_along(from_stage, axis, i - 1, k);
        p = cell_along(from_stage, axis, i + 1, k);
    } else {
        b_minus = face_bed[z_face_index(u32(i), u32(k))];
        b_plus = face_bed[z_face_index(u32(i), u32(k) + 1u)];
        m = cell_along(from_stage, axis, i, k - 1);
        p = cell_along(from_stage, axis, i, k + 1);
    }

    let c = cell_along(from_stage, axis, i, k);
    let empty = Point(0.0, 0.0, 0.0, 0.0, 0.0);

    if c.h <= 0.0 {
        return Recon(empty, empty, 0.0);
    }

    let w = c.level;

    if !c.full {
        let un = desingularize(c.h, c.qn);
        let ut = desingularize(c.h, c.qt);
        return Recon(velocity_point(max(w - b_minus, 0.0), un, ut), velocity_point(max(w - b_plus, 0.0), un, ut), 0.0);
    }

    let sw = half_slope(m.level, w, p.level);
    var w_plus = w + sw;
    var w_minus = w - sw;

    if w_plus < b_plus {
        w_plus = b_plus;
        w_minus = 2.0 * w - b_plus;
    } else if w_minus < b_minus {
        w_minus = b_minus;
        w_plus = 2.0 * w - b_minus;
    }

    let sqn = half_slope(m.qn, c.qn, p.qn);
    let sqt = half_slope(m.qt, c.qt, p.qt);

    return Recon(
        make_point(max(w_minus - b_minus, 0.0), c.qn - sqn, c.qt - sqt),
        make_point(max(w_plus - b_plus, 0.0), c.qn + sqn, c.qt + sqt),
        w_plus - w_minus,
    );
}

fn ghost_point(side: u32, along: u32, inner: Point) -> Point {
    let kind = params.kinds[side];
    if kind == WALL {
        return mirrored(inner);
    }
    if kind == TRANSMISSIVE {
        return inner;
    }
    let d = boundary[boundary_offset(side) + along];
    if side <= EAST {
        return make_point(d.x, d.y, d.z);
    }
    return make_point(d.x, d.z, d.y);
}

fn physical_flux(s: Point) -> vec3<f32> {
    return vec3<f32>(s.qn, s.qn * s.un + pressure(s.h), s.qt * s.un);
}

fn central_upwind(l: Point, r: Point) -> vec4<f32> {
    let cl = sqrt(params.gravity * l.h);
    let cr = sqrt(params.gravity * r.h);
    let ap = max(max(l.un + cl, r.un + cr), 0.0);
    let am = min(min(l.un - cl, r.un - cr), 0.0);
    let spread = ap - am;

    if spread <= 0.0 {
        return vec4<f32>(0.0);
    }

    let fl = physical_flux(l);
    let fr = physical_flux(r);
    let ul = vec3<f32>(l.h, l.qn, l.qt);
    let ur = vec3<f32>(r.h, r.qn, r.qt);
    let f = fl + (am * (fl - fr) + ap * am * (ur - ul)) / spread;

    return vec4<f32>(f, max(ap, -am));
}

fn bed_source(r: Recon, spacing: f32) -> f32 {
    let hydrostatic = (pressure(r.plus.h) - pressure(r.minus.h)) / spacing;
    let surface = params.gravity * 0.5 * (r.plus.h + r.minus.h) * r.dw / spacing;
    return hydrostatic - surface;
}

fn face_flux(from_stage: bool, axis: u32, i: u32, k: u32) -> vec4<f32> {
    var along = i;
    var limit = params.nx;
    var row = k;
    var low_side = WEST;
    var high_side = EAST;
    var li = i32(i) - 1;
    var lk = i32(k);
    if axis == 1u {
        along = k;
        limit = params.nz;
        row = i;
        low_side = SOUTH;
        high_side = NORTH;
        li = i32(i);
        lk = i32(k) - 1;
    }

    var left: Point;
    var right: Point;

    if along > 0u && along < limit {
        left = reconstruct(from_stage, axis, li, lk).plus;
        right = reconstruct(from_stage, axis, i32(i), i32(k)).minus;
    } else if along == 0u {
        right = reconstruct(from_stage, axis, i32(i), i32(k)).minus;
        left = ghost_point(low_side, row, right);
    } else {
        left = reconstruct(from_stage, axis, li, lk).plus;
        right = ghost_point(high_side, row, left);
    }

    let f = central_upwind(left, right);

    if axis == 0u {
        return f;
    }
    return vec4<f32>(f.x, f.z, f.y, f.w);
}

fn global_index(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * 64u;
}

fn compute_fluxes(from_stage: bool, f: u32) {
    if f >= face_count() {
        return;
    }

    var value: vec4<f32>;
    var rate: f32;

    if f < x_face_count() {
        let i = f % (params.nx + 1u);
        let k = f / (params.nx + 1u);
        value = face_flux(from_stage, 0u, i, k);
        rate = value.w / params.dx;
    } else {
        let r = f - x_face_count();
        let i = r % params.nx;
        let k = r / params.nx;
        value = face_flux(from_stage, 1u, i, k);
        rate = value.w / params.dz;
    }

    flux[f] = value;

    if !from_stage && rate > 0.0 {
        atomicMax(&clock.rate, bitcast<u32>(rate));
    }
}

@compute @workgroup_size(64)
fn flux_state(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    compute_fluxes(false, global_index(gid, groups));
}

@compute @workgroup_size(64)
fn flux_stage(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    compute_fluxes(true, global_index(gid, groups));
}

@compute @workgroup_size(1)
fn begin_frame() {
    clock.remaining = params.frame_time;
    clock.elapsed = 0.0;
    clock.steps = 0u;
    atomicStore(&clock.rate, 0u);
}

const STEP_NOW: u32 = 0u;
const STEP_NEXT: u32 = 9u;

fn write_step_groups(slot: u32, threads: u32, run: bool) {
    let groups = max((threads + 63u) / 64u, 1u);
    steps[slot] = select(0u, min(groups, 65535u), run);
    steps[slot + 1u] = (groups + 65534u) / 65535u;
    steps[slot + 2u] = 1u;
}

fn write_steps(base: u32, run: bool) {
    write_step_groups(base, cell_count(), run);
    write_step_groups(base + 3u, face_count(), run);
    steps[base + 6u] = select(0u, 1u, run);
    steps[base + 7u] = 1u;
    steps[base + 8u] = 1u;
}

@compute @workgroup_size(1)
fn choose_dt() {
    let rate = bitcast<f32>(atomicExchange(&clock.rate, 0u));
    var dt = min(params.dt_max, clock.remaining);
    if rate > 0.0 {
        dt = min(dt, params.cfl / rate);
    }
    dt = max(dt, 0.0);
    clock.dt = dt;
    clock.remaining = clock.remaining - dt;
    clock.elapsed = clock.elapsed + dt;
    clock.time = clock.time + dt;
    if dt > 0.0 {
        clock.steps = clock.steps + 1u;
    }

    write_steps(STEP_NOW, dt > 0.0);
    write_steps(STEP_NEXT, clock.remaining > 0.0);
}

fn outflow(i: u32, k: u32) -> f32 {
    let west = flux[x_face_index(i, k)].x;
    let east = flux[x_face_index(i + 1u, k)].x;
    let south = flux[z_face_index(i, k)].x;
    let north = flux[z_face_index(i, k + 1u)].x;
    return (max(east, 0.0) - min(west, 0.0)) * params.dz + (max(north, 0.0) - min(south, 0.0)) * params.dx;
}

fn compute_drain(from_stage: bool, c: u32) {
    if c >= cell_count() {
        return;
    }
    let i = c % params.nx;
    let k = c / params.nx;
    let out = clock.dt * outflow(i, k);
    let available = load(from_stage, c).x * params.dx * params.dz;
    var factor = 1.0;
    if out > available {
        factor = available / out;
    }
    factors[c] = factor;
}

@compute @workgroup_size(64)
fn drain_state(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    compute_drain(false, global_index(gid, groups));
}

@compute @workgroup_size(64)
fn drain_stage(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    compute_drain(true, global_index(gid, groups));
}

fn donor_factor(axis: u32, i: u32, k: u32, mass: f32) -> f32 {
    var along = i;
    var limit = params.nx;
    if axis == 1u {
        along = k;
        limit = params.nz;
    }
    if mass > 0.0 && along > 0u {
        if axis == 0u {
            return factors[cell_index(i - 1u, k)];
        }
        return factors[cell_index(i, k - 1u)];
    }
    if mass <= 0.0 && along < limit {
        return factors[cell_index(i, k)];
    }
    return 1.0;
}

fn scaled_x(i: u32, k: u32) -> vec3<f32> {
    let f = flux[x_face_index(i, k)];
    return f.xyz * donor_factor(0u, i, k, f.x);
}

fn scaled_z(i: u32, k: u32) -> vec3<f32> {
    let f = flux[z_face_index(i, k)];
    return f.xyz * donor_factor(1u, i, k, f.x);
}

fn advanced(from_stage: bool, c: u32) -> vec4<f32> {
    let i = c % params.nx;
    let k = c / params.nx;
    let west = scaled_x(i, k);
    let east = scaled_x(i + 1u, k);
    let south = scaled_z(i, k);
    let north = scaled_z(i, k + 1u);

    let rx = reconstruct(from_stage, 0u, i32(i), i32(k));
    let rz = reconstruct(from_stage, 1u, i32(i), i32(k));

    let dh = -(east.x - west.x) / params.dx - (north.x - south.x) / params.dz;
    let dhu = -(east.y - west.y) / params.dx - (north.y - south.y) / params.dz + bed_source(rx, params.dx);
    let dhv = -(east.z - west.z) / params.dx - (north.z - south.z) / params.dz + bed_source(rz, params.dz);

    let u = load(from_stage, c);
    let dt = clock.dt;
    return vec4<f32>(u.x + dt * dh, u.y + dt * dhu, u.z + dt * dhv, 0.0);
}

fn with_friction(u: vec4<f32>) -> vec4<f32> {
    if params.manning <= 0.0 || u.x <= 0.0 {
        return u;
    }
    let speed = length(vec2<f32>(desingularize(u.x, u.y), desingularize(u.x, u.z)));
    let drag = params.gravity * params.manning * params.manning * speed / pow(u.x, 4.0 / 3.0);
    let keep = 1.0 / (1.0 + clock.dt * drag);
    return vec4<f32>(u.x, u.y * keep, u.z * keep, 0.0);
}

fn dry_clamped(u: vec4<f32>) -> vec4<f32> {
    if u.x <= 0.0 {
        return vec4<f32>(0.0);
    }
    return u;
}

@compute @workgroup_size(64)
fn update_first(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    stage[c] = dry_clamped(with_friction(advanced(false, c)));
}

@compute @workgroup_size(64)
fn update_second(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }
    let next = with_friction(advanced(true, c));
    let base = state[c];
    state[c] = dry_clamped(vec4<f32>(0.5 * (base.xyz + next.xyz), 0.0));
}

fn accumulate(f: u32) {
    if f >= face_count() {
        return;
    }
    var factor: f32;
    let value = flux[f];
    if f < x_face_count() {
        factor = donor_factor(0u, f % (params.nx + 1u), f / (params.nx + 1u), value.x);
    } else {
        let r = f - x_face_count();
        factor = donor_factor(1u, r % params.nx, r / params.nx, value.x);
    }
    accumulated[f] = accumulated[f] + 0.5 * clock.dt * factor * value.x;
}

@compute @workgroup_size(64)
fn accumulate_flux(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    accumulate(global_index(gid, groups));
}

@compute @workgroup_size(64)
fn clear_accumulated(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let f = global_index(gid, groups);
    if f < face_count() {
        accumulated[f] = 0.0;
    }
}

fn in_coupling(i: u32, k: u32) -> bool {
    return i >= params.coupling_min.x && i < params.coupling_max.x && k >= params.coupling_min.y && k < params.coupling_max.y;
}

@compute @workgroup_size(64)
fn finish(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() {
        return;
    }

    var u = state[c];
    let s = source[c];
    let injected = max(s.x, -u.x);
    u.x = u.x + injected;
    u.y = u.y + s.y;
    u.z = u.z + s.z;
    source[c] = vec4<f32>(0.0);

    if u.x <= 0.0 {
        u = vec4<f32>(0.0);
    }

    let i = c % params.nx;
    let k = c / params.nx;
    let before = u.x;

    if in_coupling(i, k) {
        let target_state = coupled[c];
        u.x = max(u.x + params.coupling_blend * (target_state.x - u.x), 0.0);
        u.y = u.y + params.coupling_blend * (target_state.y - u.y);
        u.z = u.z + params.coupling_blend * (target_state.z - u.z);
    }

    stage[c] = vec4<f32>(injected, u.x - before, 0.0, 0.0);

    if params.sponge != 0u {
        let layer = sponge[c];
        if layer.w > 0.0 {
            let keep = 1.0 / (1.0 + layer.w * clock.elapsed);
            u = vec4<f32>(layer.xyz + (u.xyz - layer.xyz) * keep, 0.0);
        }
    }

    state[c] = vec4<f32>(u.xyz, 0.0);
}

const L_INTERFACE: u32 = 0u;
const L_INJECTED: u32 = 2u;
const L_COUPLED: u32 = 4u;
const L_MASS: u32 = 6u;
const L_OUTSIDE: u32 = 7u;
const L_TIER: u32 = 8u;
const L_FRAMES: u32 = 9u;
const L_REGION: u32 = 10u;
const L_HOLD: u32 = 16u;
const LEDGER_GROUP: u32 = 256u;

var<workgroup> partial: array<vec4<f32>, LEDGER_GROUP>;

fn ledger_add(slot: u32, value: f32) {
    let y = value - ledger[slot + 1u];
    let t = ledger[slot] + y;
    ledger[slot + 1u] = (t - ledger[slot]) - y;
    ledger[slot] = t;
}

fn reduce_partial(lid: u32, value: vec4<f32>) -> vec4<f32> {
    partial[lid] = value;
    workgroupBarrier();

    for (var span = LEDGER_GROUP / 2u; span > 0u; span = span / 2u) {
        if lid < span {
            partial[lid] = partial[lid] + partial[lid + span];
        }
        workgroupBarrier();
    }

    return partial[0];
}

fn stored_region() -> vec4<u32> {
    return vec4<u32>(
        u32(ledger[L_REGION]),
        u32(ledger[L_REGION + 1u]),
        u32(ledger[L_REGION + 2u]),
        u32(ledger[L_REGION + 3u]),
    );
}

fn region_changed() -> bool {
    let r = stored_region();
    return any(r.xy != params.coupling_min) || any(r.zw != params.coupling_max);
}

fn in_stored(i: u32, k: u32) -> bool {
    let r = stored_region();
    return i >= r.x && i < r.z && k >= r.y && k < r.w;
}

@compute @workgroup_size(64)
fn ledger_switch_cells(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = global_index(gid, groups);
    if c >= cell_count() || !region_changed() {
        return;
    }

    let i = c % params.nx;
    let k = c / params.nx;
    let was = in_stored(i, k);
    let now = in_coupling(i, k);
    var transfer = 0.0;

    if was && !now {
        let back = coupled[c];
        state[c] = vec4<f32>(back.xyz, 0.0);
        transfer = -back.x;
    } else if now && !was {
        transfer = state[c].x;
    }

    stage[c] = vec4<f32>(transfer, 0.0, 0.0, 0.0);
}

@compute @workgroup_size(256)
fn ledger_switch(@builtin(local_invocation_index) lid: u32) {
    let changed = region_changed();
    var sum = 0.0;

    if changed {
        for (var c = lid; c < cell_count(); c = c + LEDGER_GROUP) {
            sum = sum + stage[c].x;
        }
    }

    workgroupBarrier();
    let total = reduce_partial(lid, vec4<f32>(sum, 0.0, 0.0, 0.0));

    if lid != 0u || !changed {
        return;
    }

    ledger_add(L_INTERFACE, total.x * params.dx * params.dz);
    ledger[L_REGION] = f32(params.coupling_min.x);
    ledger[L_REGION + 1u] = f32(params.coupling_min.y);
    ledger[L_REGION + 2u] = f32(params.coupling_max.x);
    ledger[L_REGION + 3u] = f32(params.coupling_max.y);
    ledger[L_HOLD] = 0.0;
}

fn interface_extent() -> vec4<u32> {
    let lo = params.coupling_min;
    let hi = min(params.coupling_max, vec2<u32>(params.nx, params.nz));
    return vec4<u32>(lo, max(hi, lo));
}

fn interface_face(e: u32, extent: vec4<u32>) -> vec4<u32> {
    let rows = extent.w - extent.y;
    let columns = extent.z - extent.x;

    if e < rows {
        return vec4<u32>(0u, extent.x, extent.y + e, 1u);
    }
    if e < 2u * rows {
        return vec4<u32>(0u, extent.z, extent.y + e - rows, 0u);
    }

    let r = e - 2u * rows;

    if r < columns {
        return vec4<u32>(1u, extent.x + r, extent.y, 1u);
    }

    return vec4<u32>(1u, extent.x + r - columns, extent.w, 0u);
}

fn interface_mass(from_accumulated: bool, face: vec4<u32>) -> f32 {
    let axis = face.x;
    let i = face.y;
    let k = face.z;
    var along = i;
    var limit = params.nx;
    var f = x_face_index(i, k);
    var width = params.dz;

    if axis == 1u {
        along = k;
        limit = params.nz;
        f = z_face_index(i, k);
        width = params.dx;
    }

    if along == 0u || along == limit {
        return 0.0;
    }

    var mass: f32;

    if from_accumulated {
        mass = accumulated[f];
    } else {
        let value = flux[f].x;
        mass = 0.5 * clock.dt * value * donor_factor(axis, i, k, value);
    }

    let sign = select(-1.0, 1.0, face.w == 1u);
    return sign * mass * width;
}

fn reduce_interface(lid: u32, from_accumulated: bool) {
    let extent = interface_extent();
    let count = 2u * (extent.w - extent.y) + 2u * (extent.z - extent.x);
    var sum = 0.0;

    for (var e = lid; e < count; e = e + LEDGER_GROUP) {
        sum = sum + interface_mass(from_accumulated, interface_face(e, extent));
    }

    let total = reduce_partial(lid, vec4<f32>(sum, 0.0, 0.0, 0.0));

    if lid == 0u && count > 0u {
        ledger_add(L_INTERFACE, total.x);
    }
}

@compute @workgroup_size(256)
fn ledger_faces_flux(@builtin(local_invocation_index) lid: u32) {
    reduce_interface(lid, false);
}

@compute @workgroup_size(256)
fn ledger_faces_accumulated(@builtin(local_invocation_index) lid: u32) {
    reduce_interface(lid, true);
}

struct LedgerCell {
    h: f32,
    book: vec2<f32>,
    held: f32,
    inside: bool,
}

fn ledger_cell(c: u32) -> LedgerCell {
    let inside = in_coupling(c % params.nx, c / params.nx);
    let held = select(0.0, coupled[c].x, inside);

    return LedgerCell(state[c].x, stage[c].xy, held, inside);
}

fn tally_cell(sums: ptr<function, vec4<f32>>, tier: ptr<function, f32>, cell: LedgerCell) {
    *sums = *sums + vec4<f32>(cell.book.x, cell.book.y, cell.h, 0.0);

    if cell.inside {
        *tier = *tier + cell.held;
    } else {
        (*sums).w = (*sums).w + cell.h;
    }
}

@compute @workgroup_size(256)
fn ledger_finish(@builtin(local_invocation_index) lid: u32) {
    var sums = vec4<f32>(0.0);
    var tier = 0.0;
    let count = cell_count();
    var c = lid;

    while c + 7u * LEDGER_GROUP < count {
        let c0 = ledger_cell(c + 0u * LEDGER_GROUP);
        let c1 = ledger_cell(c + 1u * LEDGER_GROUP);
        let c2 = ledger_cell(c + 2u * LEDGER_GROUP);
        let c3 = ledger_cell(c + 3u * LEDGER_GROUP);
        let c4 = ledger_cell(c + 4u * LEDGER_GROUP);
        let c5 = ledger_cell(c + 5u * LEDGER_GROUP);
        let c6 = ledger_cell(c + 6u * LEDGER_GROUP);
        let c7 = ledger_cell(c + 7u * LEDGER_GROUP);
        tally_cell(&sums, &tier, c0);
        tally_cell(&sums, &tier, c1);
        tally_cell(&sums, &tier, c2);
        tally_cell(&sums, &tier, c3);
        tally_cell(&sums, &tier, c4);
        tally_cell(&sums, &tier, c5);
        tally_cell(&sums, &tier, c6);
        tally_cell(&sums, &tier, c7);
        c = c + 8u * LEDGER_GROUP;
    }

    while c < count {
        tally_cell(&sums, &tier, ledger_cell(c));
        c = c + LEDGER_GROUP;
    }

    let totals = reduce_partial(lid, sums);
    workgroupBarrier();
    let tiers = reduce_partial(lid, vec4<f32>(tier, 0.0, 0.0, 0.0));

    if lid != 0u {
        return;
    }

    let area = params.dx * params.dz;
    ledger_add(L_INJECTED, totals.x * area);
    ledger_add(L_COUPLED, totals.y * area);
    ledger[L_MASS] = totals.z * area;
    ledger[L_OUTSIDE] = totals.w * area;
    ledger[L_TIER] = tiers.x * area;
    ledger[L_FRAMES] = ledger[L_FRAMES] + 1.0;
}
