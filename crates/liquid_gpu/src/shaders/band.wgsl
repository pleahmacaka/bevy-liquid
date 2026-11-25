override BAND_PER_CELL: u32 = PER_CELL;
override BAND_CROWDED: u32 = 2u * PER_CELL;
const BAND_JITTER: f32 = 0.8;

fn band_depth() -> f32 {
    return params.physics.z * params.dx;
}

fn combination_depth() -> f32 {
    return (params.physics.z - 1.0) * params.dx;
}

fn backtrace(x: vec3<f32>, dt: f32) -> vec3<f32> {
    let k1 = sample_velocity(x);

    if BACKTRACE_ORDER < 4u {
        return x - dt * sample_velocity(x - 0.5 * dt * k1);
    }

    let k2 = sample_velocity(x - 0.5 * dt * k1);
    let k3 = sample_velocity(x - 0.5 * dt * k2);
    let k4 = sample_velocity(x - dt * k3);

    return x - dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

@compute @workgroup_size(256)
fn band_forward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        forward_face(n);
    }

    let stride = dispatched_threads(FACE_SLOT);

    for (var t = gid.x; t < select(cell_count(), brick_list[0] * BRICK_CELLS, SPARSE); t = t + stride) {
        let c = brick_cell(t);

        if c < cell_count() {
            advect_cell_phi(c);
        }
    }

    for (var t = gid.x; t < select(node_count(), brick_list[0] * BRICK_NODES, SPARSE); t = t + stride) {
        let m = brick_node(t);

        if m < node_count() {
            node_solid(m);
        }
    }
}

fn forward_face(n: u32) {
    let axis = face_axis(n);
    let x = face_position(axis, face_coords(axis, n));
    let k = hood_face_corners(axis, backtrace(x, scalars[S_BAND_DT]));
    var v = 0.0;
    var lo = BIG;
    var hi = -BIG;

    for (var i = 0; i < 8; i = i + 1) {
        let u = velocity[k.index[i]];
        v = v + k.weight[i] * u;
        lo = min(lo, u);
        hi = max(hi, u);
    }

    scratch[n] = v;
    valid[n] = lo;
    weights[n] = hi;
}

@compute @workgroup_size(256)
fn band_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        backward_face(n);
    }
}

fn backward_face(n: u32) {
    let axis = face_axis(n);
    let x = face_position(axis, face_coords(axis, n));
    let behind = hood_face_corners(axis, backtrace(x, -scalars[S_BAND_DT]));
    var back = 0.0;

    for (var i = 0; i < 8; i = i + 1) {
        back = back + behind.weight[i] * scratch[behind.index[i]];
    }

    scratch_valid[n] = clamp(scratch[n] + 0.5 * (velocity[n] - back), valid[n], weights[n]);
}

fn advect_cell_phi(n: u32) {
    let x = backtrace(cell_center(cell_coords(n)), scalars[S_BAND_DT]);
    let h = velocity_hood(x);
    let top = max(vec3<f32>(params.dims.xyz) - 1.0, vec3<f32>(0.0));
    let g = clamp(x * params.inv_dx - 0.5, vec3<f32>(0.0), top);
    let base = min(floor(g), max(top - 1.0, vec3<f32>(0.0)));
    let t = g - base;
    let b = vec3<i32>(base);
    let hi = min(b + vec3<i32>(1), vec3<i32>(top));
    var v = 0.0;

    for (var i = 0; i < 8; i = i + 1) {
        let pick = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1) == vec3<i32>(1);
        let w = select(1.0 - t, t, pick);
        v = v + w.x * w.y * w.z * band_phi[hood_cell(h, select(b, hi, pick))];
    }

    sweep_a[n] = v;
}

fn adopted(n: u32, d: f32) -> f32 {
    let solid = has_solids() && solid_phi_at_cell(cell_coords(n)) < 0.0;

    return select(d, BIG, solid);
}

@compute @workgroup_size(256)
fn band_adopt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        phi[n] = adopted(n, signed_distance[n]);
    }
}

fn face_depth(axis: u32, f: vec3<i32>) -> f32 {
    let lo = bump(f, axis, -1);
    let has_lo = lo[axis] >= 0;
    let has_hi = f[axis] < i32(params.dims[axis]);

    if has_lo && has_hi {
        return 0.5 * (phi[cell_index(lo)] + phi[cell_index(f)]);
    }

    if has_lo {
        return phi[cell_index(lo)];
    }

    return phi[cell_index(f)];
}

fn poured(axis: u32, f: vec3<i32>) -> bool {
    if emission.count.x == 0u {
        return false;
    }

    let lo = bump(f, axis, -1);
    let below = lo[axis] < 0 || band_phi[cell_index(lo)] >= 0.0;
    let above = f[axis] >= i32(params.dims[axis]) || band_phi[cell_index(f)] >= 0.0;

    return below && above;
}

@compute @workgroup_size(256)
fn band_pick(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() {
        return;
    }

    let axis = face_axis(n);
    let f = face_coords(axis, n);
    let depth = face_depth(axis, f);
    let first = scalars[S_BAND_DT] <= 0.0;

    if valid[n] > 0.0 && (depth >= -combination_depth() || first || poured(axis, f)) {
        valid[n] = 1.0;
        return;
    }

    if depth < 0.0 {
        velocity[n] = scratch_valid[n];
        valid[n] = 1.0;
        return;
    }

    velocity[n] = 0.0;
    valid[n] = 0.0;
}

fn cell_groups() -> u32 {
    if SPARSE {
        return brick_list[0] * (BRICK_CELLS / WG);
    }

    return (grid_cells() + WG - 1u) / WG;
}

@compute @workgroup_size(256)
fn band_volume_partial(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    volume_partial(brick_cell(gid.x), wid.x, lane);
}

fn volume_partial(n: u32, group: u32, lane: Lane) {
    var value = 0.0;

    if n < cell_count() {
        value = clamp(0.5 - phi[n] * params.inv_dx, 0.0, 1.0);
    }

    let total = reduce_workgroup(lane, value);

    if lane.index == 0u {
        partials[group] = total;
    }
}

fn volume_total(lane: Lane) -> f32 {
    let count = cell_groups();
    var acc = 0.0;

    for (var i = lane.index; i < count; i = i + WG) {
        acc = acc + partials[i];
    }

    return reduce_workgroup(lane, acc);
}

@compute @workgroup_size(256)
fn band_volume_final(lane: Lane) {
    let total = volume_total(lane);

    if lane.index == 0u {
        credit_target();
        settle_volume(total);
    }
}

const S_VOLUME_EMITTED: u32 = 127u;

fn credit_emitted(volume: f32) -> f32 {
    let sum = volume + scalars[S_VOLUME_EMITTED];
    scalars[S_VOLUME_EMITTED] = scalars[S_VOLUME_EMITTED] - (sum - volume);

    return sum;
}

fn credit_target() {
    if scalars[S_VOLUME_TARGET] > 0.0 {
        scalars[S_VOLUME_TARGET] = credit_emitted(scalars[S_VOLUME_TARGET]);
    } else {
        scalars[S_VOLUME_EMITTED] = 0.0;
    }
}

fn settle_volume(total: f32) {
    let volume = total * params.dx * params.dx * params.dx;
    scalars[S_VOLUME] = volume;

    if scalars[S_VOLUME_TARGET] <= 0.0 {
        scalars[S_VOLUME_TARGET] = volume;
    }

    let target_volume = scalars[S_VOLUME_TARGET];

    if target_volume <= 0.0 {
        scalars[S_VOLUME_DIVERGENCE] = 0.0;
        return;
    }

    let dt = state.dt;
    let x = (volume - target_volume) / target_volume;
    let kp = 2.3 / (25.0 * dt);
    let ki = (kp / 8.0) * (kp / 8.0);
    let integral = scalars[S_VOLUME_INTEGRAL] + x * dt;

    scalars[S_VOLUME_INTEGRAL] = integral;
    scalars[S_VOLUME_DIVERGENCE] = -(kp * x + ki * integral) / (x + 1.0);
}

fn open_fraction(c: vec3<i32>) -> f32 {
    var sum = 0.0;

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        sum = sum + weights[face_index(axis, c)] + weights[face_index(axis, bump(c, axis, 1))];
    }

    return sum / 6.0;
}

@compute @workgroup_size(256)
fn band_volume_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() && l_active[n] != 0u {
        l_b[n] = l_b[n] + scalars[S_VOLUME_DIVERGENCE] * open_fraction(cell_coords(n));
    }
}

@compute @workgroup_size(256)
fn band_blend(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() && valid[n] <= 1.5 {
        let w = clamp(weights[n], 0.0, 1.0);
        velocity[n] = w * velocity[n] + (1.0 - w) * solid_velocity[n];
    }

    if BAND_STORE {
        for (var t = gid.x; t < select(cell_count(), brick_list[0] * BRICK_CELLS, SPARSE); t = t + dispatched_threads(FACE_SLOT)) {
            let c = brick_cell(t);

            if c < cell_count() {
                band_phi[c] = phi[c];
            }
        }
    }
}

override BAND_STORE: bool = false;

@compute @workgroup_size(256)
fn band_delete(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= state.particle_count {
        return;
    }

    let x = particles[i].x.xyz;
    home_at(x);

    if phi_at(x) < -band_depth() {
        particles[i].v.w = 1.0;
    }
}

fn in_band(x: vec3<f32>) -> bool {
    let depth = -phi_at(x);
    return depth >= params.dx && depth <= band_depth();
}

fn octant(c: vec3<i32>, x: vec3<f32>) -> u32 {
    let local = (x * params.inv_dx - vec3<f32>(c)) * 2.0;
    let o = clamp(vec3<i32>(floor(local)), vec3<i32>(0), vec3<i32>(1));
    return u32(o.x + 2 * o.y + 4 * o.z);
}

fn band_salt() -> u32 {
    return hash_u32(bitcast<u32>(state.time));
}

fn hash_unit(seed: u32) -> f32 {
    return f32(hash_u32(seed)) / 4294967295.0;
}

fn seed_point(c: vec3<i32>, s: u32, salt: u32) -> vec3<f32> {
    let cell = vec3<u32>(c);
    let seed = (cell.x * 73856093u) ^ (cell.y * 19349663u) ^ (cell.z * 83492791u) ^ (s * 2654435761u) ^ salt;
    let jitter = vec3<f32>(hash_unit(seed), hash_unit(seed ^ 0x9E3779B9u), hash_unit(seed ^ 0x7F4A7C15u));
    let sub = sub_octant(s);
    let local = (sub + 0.5 + (jitter - 0.5) * BAND_JITTER) * 0.5;

    return (vec3<f32>(c) + local) * params.dx;
}

@compute @workgroup_size(256)
fn band_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;

    if g >= grid_cells() {
        return;
    }

    let c = grid_coords(g);
    home_at(cell_center(c));
    let depth = -phi[cell_index(c)];
    let resampled = depth >= params.dx && depth <= band_depth();
    var members = 0u;
    var occupied = 0u;

    for (var i = cell_start[g]; i < cell_start[g + 1u]; i = i + 1u) {
        if particles[i].v.w > 0.5 {
            continue;
        }

        if (resampled || CULL_CROWDED) && members >= BAND_CROWDED {
            particles[i].v.w = 1.0;
            continue;
        }

        members = members + 1u;
        occupied = occupied | (1u << octant(c, particles[i].x.xyz));
    }

    var chosen = 0u;

    if resampled && members < BAND_PER_CELL {
        let salt = band_salt();
        var taken = 0u;

        for (var s = 0u; s < BAND_PER_CELL; s = s + 1u) {
            if taken + members >= BAND_PER_CELL {
                break;
            }

            if (occupied & (1u << seed_octant(s))) != 0u || !in_band(seed_point(c, s, salt)) {
                continue;
            }

            chosen = chosen | (1u << s);
            taken = taken + 1u;
        }
    }

    atomicStore(&cell_counter[g], chosen);
}

@compute @workgroup_size(256)
fn band_scan_blocks(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    let base = wid.x * SCAN_BLOCK + lane_rank(lane) * SCAN_ITEMS;
    var local: array<u32, 4>;
    var sum = 0u;

    for (var i = 0u; i < SCAN_ITEMS; i = i + 1u) {
        let n = base + i;
        var v = 0u;

        if n < grid_cells() {
            v = countOneBits(atomicLoad(&cell_counter[n]));
        }

        local[i] = sum;
        sum = sum + v;
    }

    let scan = block_scan(lane, sum);

    for (var i = 0u; i < SCAN_ITEMS; i = i + 1u) {
        let n = base + i;

        if n < grid_cells() {
            band_offsets[n] = scan.exclusive + local[i];
        }
    }

    if lane.index == 0u {
        partials[wid.x] = bitcast<f32>(scan.total);
    }
}

@compute @workgroup_size(256)
fn band_scan_sums(lane: Lane) {
    let total = scan_partials(lane);

    if lane.index == 0u {
        scalars[S_BAND_SEEDED] = f32(total);
    }
}

@compute @workgroup_size(256)
fn band_spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;

    if g >= grid_cells() {
        return;
    }

    let chosen = atomicLoad(&cell_counter[g]);

    if chosen == 0u {
        return;
    }

    atomicStore(&cell_counter[g], 0u);

    let c = grid_coords(g);
    let salt = band_salt();
    var slot = state.particle_count + band_offsets[g] + bitcast<u32>(partials[g / SCAN_BLOCK]);

    for (var s = 0u; s < BAND_PER_CELL; s = s + 1u) {
        if (chosen & (1u << s)) == 0u {
            continue;
        }

        if slot >= params.particle_capacity {
            return;
        }

        particles[slot].x = vec4<f32>(seed_point(c, s, salt), -1.0);
        slot = slot + 1u;
    }
}

@compute @workgroup_size(256)
fn band_seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let first = state.particle_count;
    let last = min(first + u32(scalars[S_BAND_SEEDED]), params.particle_capacity);

    for (var i = first + gid.x; i < last; i = i + grid_cells()) {
        let x = particles[i].x.xyz;
        home_at(x);
        let h = particle_hood(x);
        let s0 = interpolate_axis(h, 0u, x);
        let s1 = interpolate_axis(h, 1u, x);
        let s2 = interpolate_axis(h, 2u, x);
        particles[i].v = vec4<f32>(s0.v, s1.v, s2.v, 0.0);
        particles[i].c0 = vec4<f32>(s0.c, params.thermal.x);
        particles[i].c1 = vec4<f32>(s1.c, 0.0);
        particles[i].c2 = vec4<f32>(s2.c, 0.0);
    }
}

@compute @workgroup_size(1)
fn band_grow() {
    state.particle_count = min(state.particle_count + u32(scalars[S_BAND_SEEDED]), params.particle_capacity);
    scalars[S_BAND_DT] = state.dt;
}

fn clear_band(t: u32) {
    let s = t / BRICK_CELLS;

    if s >= brick_list[1] {
        return;
    }

    let slot = brick_list[brick_table() + brick_list[BRICK_LIST_HEAD + brick_total() + s]];

    if slot != 0u {
        band_phi[slot * BRICK_CELLS + t % BRICK_CELLS] = BIG;
    }
}
