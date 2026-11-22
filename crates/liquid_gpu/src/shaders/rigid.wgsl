override VECTOR: u32 = 0u;

const BODY_STRIDE: u32 = 48u;

struct Jacobian {
    lin: vec3<f32>,
    ang: vec3<f32>,
}

fn is_dynamic(k: u32) -> bool {
    return bodies[k].velocity.w > 1.5;
}

fn cell_jacobian(n: u32, k: u32) -> Jacobian {
    var j: Jacobian;
    j.lin = vec3<f32>(0.0);
    j.ang = vec3<f32>(0.0);

    let c = cell_coords(n);
    let area = params.dx * params.dx;
    let center = bodies[k].position.xyz;

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        for (var high = 0; high <= 1; high = high + 1) {
            let f = bump(c, axis, high);

            let fi = face_index(axis, f);

            if owners[fi] != i32(k) {
                continue;
            }

            let open = 1.0 - weights[fi];

            if open <= 0.0 {
                continue;
            }

            let e = vec3<f32>(axis_mask(axis));

            let r = face_position(axis, f) - center;
            let s = select(-1.0, 1.0, high == 1) * open * area;

            j.lin = j.lin + s * e;
            j.ang = j.ang + s * cross(r, e);
        }
    }

    return j;
}

fn touches_body(n: u32) -> bool {
    let c = cell_coords(n);

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        for (var high = 0; high <= 1; high = high + 1) {
            let fi = face_index(axis, bump(c, axis, high));
            let k = owners[fi];

            if k >= 0 && u32(k) < params.body_count && is_dynamic(u32(k)) && 1.0 - weights[fi] > 0.0 {
                return true;
            }
        }
    }

    return false;
}

fn tile_flag(t: u32) -> u32 {
    return lv.body_base + t;
}

fn tile_head() -> u32 {
    return lv.body_base + lv.body_groups;
}

fn listed_tiles() -> u32 {
    return bitcast<u32>(partials[tile_head()]);
}

fn tile_listed(w: u32) -> u32 {
    return bitcast<u32>(partials[tile_head() + 1u + w]);
}

@compute @workgroup_size(256)
fn body_mark(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    let n = level_cell(gid.x);
    var hit = 0.0;

    if n < l_len() && l_active[n] != 0u && touches_body(n) {
        hit = 1.0;
    }

    let total = reduce_workgroup(lane, hit);

    if lane.index == 0u {
        partials[tile_flag(wid.x)] = total;
    }
}

fn compact_tiles(lane: Lane) -> u32 {
    let swept = full_size(CELL_SLOT);
    var listed = 0u;

    for (var start = 0u; start < lv.body_groups; start = start + WG) {
        let t = start + lane_rank(lane);
        let hit = t < swept && partials[tile_flag(t)] > 0.0;
        let scan = block_scan(lane, select(0u, 1u, hit));

        if hit {
            partials[tile_head() + 1u + listed + scan.exclusive] = bitcast<f32>(t);
        }

        listed = listed + scan.total;
    }

    if lane.index == 0u {
        partials[tile_head()] = bitcast<f32>(listed);
    }

    return listed;
}

fn body_terms(n: u32, k: u32, v: f32) -> Jacobian {
    var j: Jacobian;
    j.lin = vec3<f32>(0.0);
    j.ang = vec3<f32>(0.0);

    if n < l_len() && l_active[n] != 0u && is_dynamic(k) {
        j = cell_jacobian(n, k);
        j.lin = j.lin * v;
        j.ang = j.ang * v;
    }

    return j;
}

var<workgroup> tile_hit: f32;

fn tile_touched(lane: Lane, t: u32) -> bool {
    if lane.index == 0u {
        tile_hit = partials[tile_flag(t)];
    }

    return workgroupUniformLoad(&tile_hit) > 0.0;
}

var<workgroup> local_response: array<f32, BODY_STRIDE>;

fn response(k: u32, e: u32, local: bool) -> f32 {
    if local {
        return local_response[6u * k + e];
    }

    return scalars[S_BODY_RESPONSE + 6u * k + e];
}

fn body_push(n: u32, local: bool) -> f32 {
    var add = 0.0;

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        if !is_dynamic(k) {
            continue;
        }

        let j = cell_jacobian(n, k);
        let lin = vec3<f32>(response(k, 0u, local), response(k, 1u, local), response(k, 2u, local));
        let ang = vec3<f32>(response(k, 3u, local), response(k, 4u, local), response(k, 5u, local));
        add = add + dot(j.lin, lin) + dot(j.ang, ang);
    }

    return add;
}

@compute @workgroup_size(256)
fn body_gather(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    let w = tile_listed(wid.x);
    let n = level_cell(w * WG + lane.index);
    var v = 0.0;

    if n < l_len() {
        v = select(l_d[n], l_x[n], VECTOR == 0u);
    }

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        let j = body_terms(n, k, v);
        store_body(lane, j.lin, j.ang, w * BODY_STRIDE + k * 6u);
    }
}

var<workgroup> body_sums: array<f32, BODY_STRIDE>;

fn sum_listed(lane: Lane) {
    if lane.index < 6u * params.body_count {
        var s = 0.0;

        for (var w = 0u; w < listed_tiles(); w = w + 1u) {
            s = s + partials[tile_listed(w) * BODY_STRIDE + lane.index];
        }

        body_sums[lane.index] = s;
    }

    workgroupBarrier();
}

fn gathered(k: u32) -> array<f32, 6> {
    let i = 6u * k;

    return array<f32, 6>(body_sums[i], body_sums[i + 1u], body_sums[i + 2u], body_sums[i + 3u], body_sums[i + 4u], body_sums[i + 5u]);
}

fn inverse_mass(k: u32, g: array<f32, 6>) -> array<f32, 6> {
    let b = bodies[k];
    let inv_m = b.position.w;
    let t = vec3<f32>(g[3], g[4], g[5]);
    let w = vec3<f32>(dot(b.inv_inertia0.xyz, t), dot(b.inv_inertia1.xyz, t), dot(b.inv_inertia2.xyz, t));

    return array<f32, 6>(g[0] * inv_m, g[1] * inv_m, g[2] * inv_m, w.x, w.y, w.z);
}

@compute @workgroup_size(256)
fn body_impulse(lane: Lane) {
    sum_listed(lane);

    if lane.index != 0u {
        return;
    }

    let dt = state.dt;

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        if !is_dynamic(k) {
            continue;
        }

        var g = gathered(k);

        for (var e = 0u; e < 6u; e = e + 1u) {
            g[e] = g[e] * dt;
            scalars[S_IMPULSE + k * 6u + e] = scalars[S_IMPULSE + k * 6u + e] + g[e];
        }

        let dv = inverse_mass(k, g);
        var b = bodies[k];
        b.velocity = vec4<f32>(b.velocity.xyz + vec3<f32>(dv[0], dv[1], dv[2]), b.velocity.w);
        b.angular = vec4<f32>(b.angular.xyz + vec3<f32>(dv[3], dv[4], dv[5]), 0.0);
        bodies[k] = b;
    }
}
