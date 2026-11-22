override PAIR: u32 = 0u;
override TARGET: u32 = 0u;
override COUPLED: bool = false;
override LEAD: bool = false;
override TALLY: f32 = 1.0;
override SOLVE_BLOCK_SLOT: u32 = 0u;
override BLOCK_LEVELS: u32 = 0u;

const R_DQ: u32 = 0u;
const R_RN: u32 = 1u;
const R_RZ: u32 = 2u;
const R_BN: u32 = 3u;

var<private> beta: f32;
var<workgroup> solve_live: u32;

fn direction(i: u32) -> f32 {
    if LEAD {
        return l_z[i];
    }

    return l_z[i] + beta * l_e[i];
}

fn pair_is_max() -> bool {
    return PAIR >= 2u;
}

fn combine(a: f32, b: f32) -> f32 {
    if pair_is_max() {
        return max(a, b);
    }
    return a + b;
}

fn reduce_workgroup(lane: Lane, value: f32) -> f32 {
    return group_reduce(lane, value, pair_is_max());
}

fn reduce_total(lane: Lane) -> f32 {
    let groups = lv.groups;
    var acc = 0.0;
    var i = lane.index;

    loop {
        if i >= groups {
            break;
        }

        acc = combine(acc, partials[i]);
        i = i + WG;
    }

    return reduce_workgroup(lane, acc);
}

fn dot_slot(region: u32, w: u32) -> u32 {
    return lv.body_base + 2u * lv.body_groups + 1u + region * (lv.body_groups + 1u) + w;
}

fn store_partial(lane: Lane, region: u32, w: u32, value: f32) {
    if lane.index == 0u {
        partials[dot_slot(region, w)] = value;
    }
}

// the num_workgroups builtin reads stale counts on indirect dispatches, so partial counts come from the grid size and the plan table
fn grid_groups() -> u32 {
    return (grid_cells() + WG - 1u) / WG;
}

fn region_total(lane: Lane, region: u32, count: u32, largest: bool) -> f32 {
    var acc = 0.0;

    for (var i = lane.index; i < count; i = i + WG) {
        let v = partials[dot_slot(region, i)];
        acc = select(acc + v, max(acc, v), largest);
    }

    return group_reduce(lane, acc, largest);
}

fn residual_norm(n: u32) -> f32 {
    return abs(l_r[n]) / tall_mass(n);
}

@compute @workgroup_size(256)
fn pcg_open(lane: Lane) {
    var listed = 0u;

    if COUPLED {
        listed = compact_tiles(lane);
    }

    if lane.index != 0u {
        return;
    }

    let running = state.running == 1u;
    state.converged = 0u;

    if running {
        scalars[S_ITERATION] = 0.0;
    }

    if SPARSE && BLOCK_LEVELS > 0u {
        open_blocks();
    }

    write_slots(SOLVE_FIRST, running);

    if COUPLED {
        let on = indirect[3u * GRID_SLOT] != 0u;
        set_brick_slot(BODY_SLOT, listed, on);
        set_brick_slot(SOLVE_BODY_SLOT, listed, on);
    }
}

@compute @workgroup_size(256)
fn pcg_init_residual(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    let n = level_cell(gid.x);
    var bn = 0.0;
    var rn = 0.0;

    if n < l_len() {
        if l_active[n] == 0u {
            l_x[n] = 0.0;
            l_r[n] = 0.0;
        } else {
            l_r[n] = l_b[n] - l_apply(n, 0u);
        }

        bn = tall_rhs_norm(n);
        rn = residual_norm(n);
    }

    store_partial(lane, R_BN, wid.x, group_reduce(lane, bn, true));

    if !COUPLED {
        store_partial(lane, R_RN, wid.x, group_reduce(lane, rn, true));
    }
}

@compute @workgroup_size(256)
fn pcg_apply(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    if !LEAD {
        let rz = scalars[S_RZ];
        beta = select(0.0, region_total(lane, R_RZ, grid_groups(), false) / rz, rz != 0.0);
    }

    let n = masked_cell(gid.x);
    var dq = 0.0;

    if n < l_len() {
        let v = direction(n);
        var q = 0.0;

        if l_active[n] != 0u {
            q = l_apply(n, 1u);
        }

        l_d[n] = v;
        l_q[n] = q;
        dq = v * q;
    }

    store_partial(lane, R_DQ, wid.x, group_reduce(lane, dq, false));
}

@compute @workgroup_size(256)
fn pcg_dot(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    let n = masked_cell(gid.x);
    var rz = 0.0;

    if n < l_len() {
        rz = l_r[n] * l_z[n];
    }

    store_partial(lane, R_RZ, wid.x, group_reduce(lane, rz, false));
}

@compute @workgroup_size(256)
fn pcg_scalar(lane: Lane) {
    if lane.index == 0u {
        solve_live = select(0u, 1u, state.converged == 0u && state.running == 1u);
    }

    if workgroupUniformLoad(&solve_live) == 0u {
        return;
    }

    var dq = 0.0;
    var rz = 0.0;

    if !LEAD {
        dq = region_total(lane, R_DQ, grid_groups(), false);
        rz = region_total(lane, R_RZ, grid_groups(), false);
    }

    if COUPLED {
        sum_listed(lane);
    }

    if lane.index != 0u {
        return;
    }

    if COUPLED {
        for (var k = 0u; k < MAX_BODIES; k = k + 1u) {
            var out = array<f32, 6>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

            if k < params.body_count && is_dynamic(k) {
                let g = gathered(k);
                out = inverse_mass(k, g);

                for (var e = 0u; e < 6u; e = e + 1u) {
                    out[e] = out[e] * scalars[S_COUPLING];
                    dq = dq + g[e] * out[e];
                }
            }

            for (var e = 0u; e < 6u; e = e + 1u) {
                scalars[S_BODY_RESPONSE + k * 6u + e] = out[e];
            }
        }
    }

    if !LEAD {
        scalars[S_RZ] = rz;
        scalars[S_DQ] = dq;
        scalars[S_ALPHA] = select(0.0, rz / dq, dq != 0.0);
    }
}

@compute @workgroup_size(1)
fn pcg_start() {
    scalars[S_RZ] = scalars[S_RZ_NEW];
}

@compute @workgroup_size(1)
fn pcg_alpha() {
    let dq = scalars[S_DQ];
    scalars[S_ALPHA] = select(0.0, scalars[S_RZ] / dq, dq != 0.0);
}

@compute @workgroup_size(1)
fn pcg_beta() {
    let rz = scalars[S_RZ];
    scalars[S_BETA] = select(0.0, scalars[S_RZ_NEW] / rz, rz != 0.0);
    scalars[S_RZ] = scalars[S_RZ_NEW];
}

@compute @workgroup_size(256)
fn pcg_update(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    let n = level_cell(gid.x);
    var touched = false;
    var rn = 0.0;

    if COUPLED {
        touched = tile_touched(lane, wid.x);
    }

    if n < l_len() {
        if !tall_inside(n) {
            var push = 0.0;

            if touched && l_active[n] != 0u {
                push = body_push(n, false);
            }

            if LEAD {
                l_r[n] = l_r[n] - push;
            } else {
                let alpha = scalars[S_ALPHA];
                l_x[n] = l_x[n] + alpha * l_d[n];
                l_r[n] = l_r[n] - alpha * (l_q[n] + push);
            }
        }

        rn = residual_norm(n);
    }

    store_partial(lane, R_RN, wid.x, group_reduce(lane, rn, true));
}

@compute @workgroup_size(256)
fn pcg_check(lane: Lane) {
    var bn = 0.0;

    if LEAD {
        bn = region_total(lane, R_BN, full_size(SOLVE_LEVEL_SLOT), true);
    }

    let total = region_total(lane, R_RN, full_size(SOLVE_LEVEL_SLOT), true);

    if lane.index != 0u {
        return;
    }

    if state.converged == 1u {
        write_slots(SOLVE_FIRST, false);
        return;
    }

    if state.running == 1u {
        scalars[S_RNORM] = total;

        if LEAD {
            scalars[S_BNORM] = bn;
        }
    }

    if scalars[S_RNORM] <= scalars[S_TOLERANCE] * max(scalars[S_BNORM], 1.0e-30) {
        state.converged = 1u;
    }

    if state.running == 1u {
        scalars[S_ITERATION] = scalars[S_ITERATION] + TALLY;
    }

    write_slots(SOLVE_FIRST, state.running == 1u && state.converged == 0u);
}

fn open_blocks() {
    indirect[3u * SLOT_COUNT + SOLVE_BLOCK_SLOT] = brick_list[0];

    for (var l = 1u; l <= BLOCK_LEVELS; l = l + 1u) {
        let at = level_table() + 1u + 4u * (l - 1u);
        let blocks = (vec3<u32>(brick_list[at], brick_list[at + 1u], brick_list[at + 2u]) + BRICK - 1u) / BRICK;
        let count = brick_list[brick_list[at + 3u]];
        indirect[3u * SLOT_COUNT + SOLVE_BLOCK_SLOT + l] = select(count, blocks.x * blocks.y * blocks.z, count == NONE);
    }
}
