struct CouplingParams {
    offset: vec4<f32>,
    grid: vec4<u32>,
    cell: vec4<f32>,
}

@group(3) @binding(1) var<uniform> coupling: CouplingParams;
@group(3) @binding(2) var<storage, read> hf_state: array<vec4<f32>>;
@group(3) @binding(3) var<storage, read> hf_bed: array<vec4<f32>>;
@group(3) @binding(4) var<storage, read_write> hf_coupled: array<vec4<f32>>;
@group(3) @binding(5) var<storage, read> hf_profile: array<vec4<f32>>;
@group(3) @binding(6) var<storage, read_write> hf_ledger: array<f32>;

const L_INTERFACE: u32 = 0u;
const L_REGION: u32 = 10u;
const L_ARMED: u32 = 14u;
const L_REFERENCE: u32 = 15u;
const L_HOLD: u32 = 16u;
const L_LIVE: u32 = 17u;

struct Column {
    h: f32,
    u: f32,
    v: f32,
    bed: f32,
    mode: vec4<f32>,
}

fn hf_cell(i: i32, k: i32) -> u32 {
    let nx = i32(coupling.grid.x);
    let nz = i32(coupling.grid.y);
    let ci = clamp(i, 0, nx - 1);
    let ck = clamp(k, 0, nz - 1);
    return u32(ck * nx + ci);
}

fn hf_column(xz: vec2<f32>) -> Column {
    let g = xz / coupling.cell.xy - 0.5;
    let b = floor(g);
    let t = g - b;
    let i = i32(b.x);
    let k = i32(b.y);

    let s00 = hf_state[hf_cell(i, k)];
    let s10 = hf_state[hf_cell(i + 1, k)];
    let s01 = hf_state[hf_cell(i, k + 1)];
    let s11 = hf_state[hf_cell(i + 1, k + 1)];
    let s = mix(mix(s00, s10, t.x), mix(s01, s11, t.x), t.y);

    let b00 = hf_bed[hf_cell(i, k)].x;
    let b10 = hf_bed[hf_cell(i + 1, k)].x;
    let b01 = hf_bed[hf_cell(i, k + 1)].x;
    let b11 = hf_bed[hf_cell(i + 1, k + 1)].x;

    let m00 = hf_profile[hf_cell(i, k)];
    let m10 = hf_profile[hf_cell(i + 1, k)];
    let m01 = hf_profile[hf_cell(i, k + 1)];
    let m11 = hf_profile[hf_cell(i + 1, k + 1)];

    var c: Column;
    c.h = s.x;
    c.bed = mix(mix(b00, b10, t.x), mix(b01, b11, t.x), t.y);
    c.mode = mix(mix(m00, m10, t.x), mix(m01, m11, t.x), t.y);

    let h = max(s.x, 1.0e-3);
    c.u = (s.y - c.mode.y) / h;
    c.v = (s.z - c.mode.z) / h;

    if s.x < 1.0e-3 {
        c.u = 0.0;
        c.v = 0.0;
    }

    return c;
}

fn surface_velocity(m: vec4<f32>, z: f32, depth: f32) -> vec3<f32> {
    var horizontal = 1.0 / depth;
    var vertical = (z + depth) / depth;

    if m.x * depth >= 1.0e-6 {
        let a = exp(m.x * z);
        let b = exp(-m.x * (z + 2.0 * depth));
        let d = 1.0 - exp(-2.0 * m.x * depth);
        horizontal = m.x * (a + b) / d;
        vertical = (a - b) / d;
    }

    return vec3<f32>(horizontal * m.y, -m.w * vertical, horizontal * m.z);
}

fn column_target(x: vec3<f32>) -> vec4<f32> {
    let p = x + coupling.offset.xyz;
    let c = hf_column(p.xz);

    if c.h < 1.0e-3 || p.y > c.bed + c.h {
        return vec4<f32>(0.0);
    }

    let e = coupling.cell.xy;
    let cx0 = hf_column(p.xz - vec2<f32>(e.x, 0.0));
    let cx1 = hf_column(p.xz + vec2<f32>(e.x, 0.0));
    let cz0 = hf_column(p.xz - vec2<f32>(0.0, e.y));
    let cz1 = hf_column(p.xz + vec2<f32>(0.0, e.y));
    let divergence = (cx1.u - cx0.u) / (2.0 * e.x) + (cz1.v - cz0.v) / (2.0 * e.y);
    let height = p.y - c.bed;
    let surface = surface_velocity(c.mode, height - c.h, c.h);
    let w = -height * divergence + surface.y;

    return vec4<f32>(c.u + surface.x, w, c.v + surface.z, 1.0);
}

fn side_distance(axis: u32, f: vec3<i32>) -> f32 {
    let x = face_position(axis, f) * params.inv_dx;
    let size = vec3<f32>(params.dims.xyz);
    return min(min(x.x, size.x - x.x), min(x.z, size.z - x.z));
}

fn relax_face(n: u32) {
    if coupling.cell.w < 0.5 {
        return;
    }

    let axis = face_axis(n);
    let f = face_coords(axis, n);
    let d = side_distance(axis, f);
    let band = coupling.offset.w;

    if d >= band {
        return;
    }

    let target_velocity = column_target(face_position(axis, f));

    if target_velocity.w < 0.5 {
        return;
    }

    let r = 1.0 - d / band;
    let weight = coupling.cell.z * r * r;
    velocity[n] = mix(velocity[n], target_velocity[axis], weight);
}

@compute @workgroup_size(256)
fn couple_boundary(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        boundary_face(n);
    }
}

@compute @workgroup_size(256)
fn extrapolate_inflow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        extrapolate_back_face(n);
        relax_face(n);

        if EXTRAP_FACES {
            boundary_face(n);
        }
    }

    finish_bodies(gid.x);
}

@compute @workgroup_size(256)
fn gradient_inflow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        refresh_face(n, false);
        boundary_face(n);
        gradient_face(n);
    }
}

fn boundary_face(n: u32) {
    if coupling.cell.w < 0.5 {
        return;
    }

    let axis = face_axis(n);

    if axis == 1u {
        return;
    }

    let f = face_coords(axis, n);

    if f[axis] != 0 && f[axis] != i32(params.dims[axis]) {
        return;
    }

    let target_velocity = column_target(face_position(axis, f));

    if target_velocity.w > 0.5 {
        solid_velocity[n] = target_velocity[axis];
    }
}

fn interface_reach(inside: f32, outside: f32) -> f32 {
    return inside / (inside - outside);
}

fn column_fraction(c: vec3<i32>) -> f32 {
    let here = phi[cell_index(c)];
    let top = i32(params.dims.y) - 1;
    var above = 3.0 * params.dx;
    var below = -params.dx;

    if c.y < top {
        above = phi[cell_index(c + vec3<i32>(0, 1, 0))];
    }

    if c.y > 0 {
        below = phi[cell_index(c - vec3<i32>(0, 1, 0))];
    }

    if here < 0.0 {
        var up = 0.5;
        var down = 0.5;

        if above >= 0.0 {
            up = min(interface_reach(here, above), 0.5);
        }

        if below >= 0.0 {
            down = min(interface_reach(here, below), 0.5);
        }

        return up + down;
    }

    var extra = 0.0;

    if below < 0.0 {
        extra = extra + max(interface_reach(below, here) - 0.5, 0.0);
    }

    if above < 0.0 {
        extra = extra + max(interface_reach(above, here) - 0.5, 0.0);
    }

    return extra;
}

fn overlap(lo: vec2<f32>, hi: vec2<f32>, a: vec2<f32>, b: vec2<f32>) -> f32 {
    let span = max(min(hi, b) - max(lo, a), vec2<f32>(0.0));
    return span.x * span.y;
}

fn column_cell(cell: vec3<i32>) -> vec3<f32> {
    let fraction = column_fraction(cell);

    if fraction <= 0.0 {
        return vec3<f32>(0.0);
    }

    let u = 0.5 * (velocity[face_index(0u, cell)] + velocity[face_index(0u, cell + vec3<i32>(1, 0, 0))]);
    let v = 0.5 * (velocity[face_index(2u, cell)] + velocity[face_index(2u, cell + vec3<i32>(0, 0, 1))]);

    return fraction * params.dx * vec3<f32>(1.0, u, v);
}

@compute @workgroup_size(256)
fn couple_columns(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    if coupling.cell.w < 0.5 {
        return;
    }

    if all(wid == vec3<u32>(0u)) && lane.index == 0u {
        hf_ledger[L_LIVE] = 1.0;
    }

    let start = vec2<i32>(floor(coupling.offset.xz / coupling.cell.xy)) - 1;
    let column = start + vec2<i32>(wid.xy);

    if any(column < vec2<i32>(0)) || any(column >= vec2<i32>(coupling.grid.xy)) {
        return;
    }

    let c = u32(column.y) * coupling.grid.x + u32(column.x);
    let lo = vec2<f32>(column) * coupling.cell.xy - coupling.offset.xz;
    let hi = lo + coupling.cell.xy;
    let size = vec2<f32>(f32(params.dims.x), f32(params.dims.z)) * params.dx;

    if any(hi <= vec2<f32>(0.0)) || any(lo >= size) {
        return;
    }

    let first = max(vec2<i32>(floor(lo * params.inv_dx)), vec2<i32>(0));
    let last = min(vec2<i32>(ceil(hi * params.inv_dx)), vec2<i32>(i32(params.dims.x), i32(params.dims.z)));
    let span = vec2<u32>(last - first);
    let height = params.dims.y;
    var sum = vec3<f32>(0.0);
    var covered = 0.0;

    for (var t = lane.index; t < span.x * span.y * height; t = t + WG) {
        let j = t % height;
        let q = t / height;
        let ci = first.x + i32(q % span.x);
        let ck = first.y + i32(q / span.x);
        let a = vec2<f32>(f32(ci), f32(ck)) * params.dx;
        let area = overlap(lo, hi, a, a + params.dx);

        if area > 0.0 {
            sum = sum + area * column_cell(vec3<i32>(ci, i32(j), ck));
            covered = covered + select(0.0, area, j == 0u);
        }
    }

    let total = vec4<f32>(group_reduce(lane, sum.x, false), group_reduce(lane, sum.y, false), group_reduce(lane, sum.z, false), group_reduce(lane, covered, false));

    if lane.index != 0u || total.w <= 0.0 {
        return;
    }

    let mean = total.xyz / total.w;
    let bed = hf_bed[c].x;
    let h = max(coupling.offset.y + mean.x - bed, 0.0);
    var scale = 0.0;

    if mean.x > 1.0e-4 {
        scale = h / mean.x;
    }

    hf_coupled[c] = vec4<f32>(h, mean.y * scale, mean.z * scale, total.w / (coupling.cell.x * coupling.cell.y));
}

@compute @workgroup_size(256)
fn band_volume_target(lane: Lane) {
    let total = volume_total(lane);

    if lane.index == 0u {
        retarget();
        settle_volume(total);
    }
}

fn retarget() {
    let exchanged = hf_ledger[L_INTERFACE] - hf_ledger[L_INTERFACE + 1u];
    let current = scalars[S_VOLUME_TARGET];

    if current <= 0.0 {
        if hf_ledger[L_ARMED] > 0.5 {
            hf_ledger[L_HOLD] = 1.0;
        }
        return;
    }

    if hf_ledger[L_ARMED] < 0.5 {
        hf_ledger[L_REFERENCE] = current - exchanged;
        hf_ledger[L_ARMED] = 1.0;
        return;
    }

    if hf_ledger[L_HOLD] < 0.5 {
        scalars[S_VOLUME_TARGET] = hf_ledger[L_REFERENCE] + exchanged;
    }
}

fn previous_region(i: i32, k: i32) -> bool {
    let r = vec4<i32>(
        i32(hf_ledger[L_REGION]),
        i32(hf_ledger[L_REGION + 1u]),
        i32(hf_ledger[L_REGION + 2u]),
        i32(hf_ledger[L_REGION + 3u]),
    );
    return hf_ledger[L_LIVE] > 0.5 && i >= r.x && i < r.z && k >= r.y && k < r.w;
}

fn seed_depth(i: i32, k: i32) -> f32 {
    let c = hf_cell(i, k);
    let ci = i32(c % coupling.grid.x);
    let ck = i32(c / coupling.grid.x);

    if previous_region(ci, ck) {
        return hf_coupled[c].x;
    }

    return hf_state[c].x;
}

fn seed_level(xz: vec2<f32>) -> f32 {
    let g = (xz + coupling.offset.xz) / coupling.cell.xy - 0.5;
    let b = floor(g);
    let t = g - b;
    let i = i32(b.x);
    let k = i32(b.y);
    let l00 = hf_bed[hf_cell(i, k)].x + seed_depth(i, k);
    let l10 = hf_bed[hf_cell(i + 1, k)].x + seed_depth(i + 1, k);
    let l01 = hf_bed[hf_cell(i, k + 1)].x + seed_depth(i, k + 1);
    let l11 = hf_bed[hf_cell(i + 1, k + 1)].x + seed_depth(i + 1, k + 1);

    return mix(mix(l00, l10, t.x), mix(l01, l11, t.x), t.y) - coupling.offset.y;
}

fn seed_position(c: vec3<i32>, s: u32) -> vec3<f32> {
    let sub = vec3<f32>(f32(s & 1u), f32((s >> 1u) & 1u), f32((s >> 2u) & 1u));
    let u = vec3<u32>(c);
    let seed = (u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u) ^ (s * 2654435761u);
    let jitter = vec3<f32>(spawn_hash(seed), spawn_hash(seed ^ 0x9E3779B9u), spawn_hash(seed ^ 0x7F4A7C15u)) - 0.5;
    return (vec3<f32>(c) + (sub + 0.5 + 0.8 * jitter) * 0.5) * params.dx;
}

fn seed_column(column: u32, offset: u32, write: bool) -> u32 {
    let ci = i32(column % params.dims.x);
    let ck = i32(column / params.dims.x);
    let centre = (vec2<f32>(f32(ci), f32(ck)) + 0.5) * params.dx;
    let level = seed_level(centre);
    var n = 0u;

    for (var j = 0; j < i32(params.dims.y); j = j + 1) {
        if f32(j) * params.dx > level + params.dx {
            break;
        }

        for (var s = 0u; s < SPAWN_PER_CELL; s = s + 1u) {
            let x = seed_position(vec3<i32>(ci, j, ck), s);

            if x.y >= seed_level(x.xz) {
                continue;
            }

            if write && offset + n < params.particle_capacity {
                var p: Particle;
                p.x = vec4<f32>(x, -1.0);
                p.v = vec4<f32>(column_target(x).xyz, 0.0);
                p.c0 = vec4<f32>(0.0);
                p.c1 = vec4<f32>(0.0);
                p.c2 = vec4<f32>(0.0);
                particles[offset + n] = p;
            }

            n = n + 1u;
        }
    }

    return n;
}

@compute @workgroup_size(256)
fn couple_seed(lane: Lane) {
    let columns = params.dims.x * params.dims.z;
    var base = 0u;

    for (var start = 0u; start < columns; start = start + WG) {
        let column = start + lane_rank(lane);
        var count = 0u;

        if column < columns {
            count = seed_column(column, 0u, false);
        }

        let scan = block_scan(lane, count);

        if column < columns {
            seed_column(column, base + scan.exclusive, true);
        }

        base = base + scan.total;
    }

    if lane.index == 0u {
        state.particle_count = min(base, params.particle_capacity);
    }
}

const SPAWN_PER_CELL: u32 = 8u;
const SPAWN_FULL: u32 = 6u;

fn boundary_cell_count() -> u32 {
    let d = params.dims;
    return 2u * d.y * d.z + 2u * d.y * d.x;
}

fn boundary_cell(b: u32) -> vec4<i32> {
    let d = params.dims;
    let side_x = d.y * d.z;
    let side_z = d.y * d.x;

    if b < 2u * side_x {
        let side = b / side_x;
        let r = b % side_x;
        let i = select(0, i32(d.x) - 1, side == 1u);
        return vec4<i32>(i, i32(r % d.y), i32(r / d.y), i32(side));
    }

    let rest = b - 2u * side_x;
    let side = rest / side_z;
    let r = rest % side_z;
    let k = select(0, i32(d.z) - 1, side == 1u);
    return vec4<i32>(i32(r % d.x), i32(r / d.x) % i32(d.y), k, 2 + i32(side));
}

fn inflow(side: i32, v: vec3<f32>) -> bool {
    switch side {
        case 0: { return v.x > 0.0; }
        case 1: { return v.x < 0.0; }
        case 2: { return v.z > 0.0; }
        default: { return v.z < 0.0; }
    }
}

fn spawn_hash(s: u32) -> f32 {
    var v = s ^ 2747636419u;
    v = v * 2654435769u;
    v = v ^ (v >> 16u);
    v = v * 2654435769u;
    v = v ^ (v >> 16u);
    v = v * 2654435769u;
    return f32(v) / 4294967295.0;
}

@compute @workgroup_size(256)
fn measure_inflow(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    record_speed(lid, max(particle_speed(gid.x), face_speed(gid.x, groups.x * WG)));

    for (var b = gid.x; b < boundary_cell_count(); b = b + groups.x * WG) {
        spawn_boundary(b);
    }
}

fn spawn_boundary(b: u32) {

    let base = state.particle_count + b * SPAWN_PER_CELL;
    let cell = boundary_cell(b);
    let c = cell.xyz;
    let n = grid_index(c);
    let existing = cell_start[n + 1u] - cell_start[n];
    let target_velocity = column_target(cell_center(c));
    let open = coupling.cell.w > 0.5 && target_velocity.w > 0.5 && inflow(cell.w, target_velocity.xyz) && existing < SPAWN_FULL;
    let need = SPAWN_PER_CELL - min(existing, SPAWN_PER_CELL);
    let salt = bitcast<u32>(state.time) * 7919u;

    for (var s = 0u; s < SPAWN_PER_CELL; s = s + 1u) {
        let slot = base + s;

        if slot >= params.particle_capacity {
            return;
        }

        var p: Particle;
        p.c0 = vec4<f32>(0.0);
        p.c1 = vec4<f32>(0.0);
        p.c2 = vec4<f32>(0.0);
        p.v = vec4<f32>(0.0, 0.0, 0.0, 1.0);

        let sub = vec3<f32>(f32(s & 1u), f32((s >> 1u) & 1u), f32((s >> 2u) & 1u));
        let seed = (b * 8u + s) ^ salt;
        let jitter = vec3<f32>(spawn_hash(seed), spawn_hash(seed ^ 0x9E3779B9u), spawn_hash(seed ^ 0x7F4A7C15u)) - 0.5;
        let x = (vec3<f32>(c) + (sub + 0.5 + 0.8 * jitter) * 0.5) * params.dx;
        p.x = vec4<f32>(x, -1.0);

        if open && s < need && column_target(x).w > 0.5 {
            p.v = vec4<f32>(target_velocity.xyz, 0.0);
        }

        particles[slot] = p;
    }
}

fn cull_particle(i: u32) {
    if coupling.cell.w < 0.5 {
        return;
    }

    let x = particles[i].x.xyz;
    let size = vec3<f32>(params.dims.xyz) * params.dx;

    if x.x < 0.0 || x.z < 0.0 || x.x >= size.x || x.z >= size.z {
        particles[i].v.w = 1.0;
    }
}

@compute @workgroup_size(256)
fn keep_inflow(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i < state.particle_count {
        cull_particle(i);
        keep_particle(i);
    }
}
