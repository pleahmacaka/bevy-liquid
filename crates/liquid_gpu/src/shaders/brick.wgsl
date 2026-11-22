fn dilated(b: u32) -> bool {
    let dims = vec3<i32>(brick_dims());
    let home = brick_coords(b);

    for (var o = 0; o < 27; o = o + 1) {
        let p = home + vec3<i32>(o % 3, (o / 3) % 3, o / 9) - vec3<i32>(1);

        if any(p < vec3<i32>(0)) || any(p >= dims) {
            continue;
        }

        if (atomicLoad(&brick_flags[u32(p.x + dims.x * (p.y + dims.y * p.z))]) & BRICK_OCCUPIED) != 0u {
            return true;
        }
    }

    return false;
}

fn run_extent(b: u32, table: u32) -> vec2<u32> {
    let x = u32(brick_coords(b).x);
    let width = brick_dims().x;
    var left = 0u;
    var right = 1u;

    while left < x && brick_list[table + b - left - 1u] != 0u {
        left = left + 1u;
    }

    while x + right < width && brick_list[table + b + right] != 0u {
        right = right + 1u;
    }

    return vec2<u32>(left, right);
}

fn set_brick_slot(s: u32, groups: u32, on: bool) {
    indirect[3u * SLOT_COUNT + s] = groups;
    indirect[3u * s] = select(0u, groups, on);
}

fn wants_brick(b: u32, total: u32) -> bool {
    return b < total && (atomicLoad(&brick_flags[b]) & BRICK_ACTIVE) != 0u;
}

fn compact_bricks(lane: Lane) {
    let total = brick_total();
    let changes = BRICK_LIST_HEAD + total;
    let table = brick_table();
    let owners = table + total;
    let free = owners + params.pool.x + 1u;
    var spare = brick_list[2];
    var changed = 0u;

    for (var start = 0u; start < total; start = start + WG) {
        let b = start + lane_rank(lane);

        if b < total && dilated(b) {
            atomicOr(&brick_flags[b], BRICK_ACTIVE);
        }
    }

    storageBarrier();

    for (var start = 0u; start < total; start = start + WG) {
        let b = start + lane_rank(lane);
        var slot = 0u;

        if b < total {
            slot = brick_list[table + b];
        }

        let release = slot != 0u && !wants_brick(b, total);
        let scan = block_scan(lane, select(0u, 1u, release));

        if release {
            brick_list[free + spare + scan.exclusive] = slot;
            brick_list[table + b] = 0u;
            brick_list[changes + changed + scan.exclusive] = b;
        }

        spare = spare + scan.total;
        changed = changed + scan.total;
    }

    storageBarrier();

    var wanted = 0u;

    for (var start = 0u; start < total; start = start + WG) {
        let b = start + lane_rank(lane);
        let fresh = wants_brick(b, total) && brick_list[table + b] == 0u;

        if b < total {
            atomicStore(&brick_flags[b], 0u);
        }

        let scan = block_scan(lane, select(0u, 1u, fresh));
        let rank = wanted + scan.exclusive;

        if fresh && rank < spare {
            let slot = brick_list[free + spare - 1u - rank];
            brick_list[table + b] = slot;
            brick_list[owners + slot] = pack_brick(b);
            brick_list[changes + changed + rank] = b;
        }

        wanted = wanted + scan.total;
    }

    let granted = min(wanted, spare);
    changed = changed + granted;
    spare = spare - granted;
    storageBarrier();

    var listed = 0u;

    for (var start = 0u; start < total; start = start + WG) {
        let b = start + lane_rank(lane);
        let live = b < total && brick_list[table + b] != 0u;
        let scan = block_scan(lane, select(0u, 1u, live));
        let slot = listed + scan.exclusive;

        if live {
            let run = run_extent(b, table);
            brick_list[BRICK_LIST_HEAD + slot] = b;
            brick_list[BRICK_LIST_HEAD + 2u * total + slot] = slot - run.x;
            brick_list[BRICK_LIST_HEAD + 3u * total + slot] = run.x + run.y;
            brick_list[brick_listing() + 2u * slot] = brick_list[table + b];
            brick_list[brick_listing() + 2u * slot + 1u] = pack_brick(b);
        }

        listed = listed + scan.total;
    }

    if lane.index != 0u {
        return;
    }

    brick_list[0] = listed;
    brick_list[1] = changed;
    brick_list[2] = spare;
    state.brick_overflow = wanted - granted;

    let on = indirect[3u * GRID_SLOT] != 0u;
    let cells = listed * (BRICK_CELLS / WG);
    set_brick_slot(CELL_SLOT, cells, on);
    set_brick_slot(LEVEL_SLOT, cells, on);
    set_brick_slot(FACE_SLOT, (listed * BRICK_FACES + WG - 1u) / WG, on);
    set_brick_slot(NODE_SLOT, (listed * BRICK_NODES + WG - 1u) / WG, on);
    set_brick_slot(CLEAR_SLOT, (changed * BRICK_FACES + WG - 1u) / WG, on);
    indirect[3u * SLOT_COUNT + SOLVE_LEVEL_SLOT] = cells;
    indirect[3u * SLOT_COUNT + SOLVE_FACE_SLOT] = (listed * BRICK_FACES + WG - 1u) / WG;
}

fn link_brick(t: u32) {
    let b = t / 27u;

    if b >= brick_total() || brick_list[1] == 0u {
        return;
    }

    let slot = brick_list[brick_table() + b];

    if slot == 0u {
        return;
    }

    let k = t % 27u;
    let p = brick_coords(b) + vec3<i32>(vec3<u32>(k % 3u, (k / 3u) % 3u, k / 9u)) - vec3<i32>(1);
    let dims = vec3<i32>(brick_dims());
    var neighbor = 0u;

    if all(p >= vec3<i32>(0)) && all(p < dims) {
        neighbor = brick_list[brick_table() + u32(p.x + dims.x * (p.y + dims.y * p.z))];
    }

    brick_list[brick_neighbors() + 27u * slot + k] = neighbor;
}

fn covers_live(coords: vec3<u32>, below: u32, below_bricks: vec3<u32>) -> bool {
    for (var o = 0u; o < 8u; o = o + 1u) {
        let p = coords * 2u + vec3<u32>(o & 1u, (o >> 1u) & 1u, o >> 2u);

        if all(p < below_bricks) && brick_list[below + p.x + below_bricks.x * (p.y + below_bricks.y * p.z)] != 0u {
            return true;
        }
    }

    return false;
}

fn level_bricks(lane: Lane) {
    let changed = brick_list[1] != 0u;
    let head = level_table();
    var below = brick_table();
    var below_bricks = brick_dims();
    var dims = params.dims.xyz;

    // FXC rejects barriers in loops bounded by storage reads, so bounds come from params and overrides
    for (var l = 0u; l + 1u < SOLVE_FIRST - LEVEL_SLOT; l = l + 1u) {
        dims = (dims + 1u) / 2u;
        let members = brick_list[head + 1u + 4u * l + 3u];
        let bricks = (dims + BRICK - 1u) / BRICK;
        let total = bricks.x * bricks.y * bricks.z;
        let flags = members + 1u + total;
        var listed = 0u;

        for (var start = 0u; start < total; start = start + WG) {
            let b = start + lane_rank(lane);
            let coords = vec3<u32>(b % bricks.x, (b / bricks.x) % bricks.y, b / (bricks.x * bricks.y));
            let wanted = b < total && covers_live(coords, below, below_bricks);

            if changed && b < total {
                brick_list[flags + b] = select(0u, 1u, wanted);
            }

            let scan = block_scan(lane, select(0u, 1u, wanted));

            if changed && wanted {
                brick_list[members + 1u + listed + scan.exclusive] = coords.x | (coords.y << 10u) | (coords.z << 20u);
            }

            listed = listed + scan.total;
        }

        if changed && lane.index == 0u {
            let full = 4u * listed >= 3u * total;
            let cells = dims.x * dims.y * dims.z;
            brick_list[members] = select(listed, NONE, full);
            indirect[3u * SLOT_COUNT + SOLVE_LEVEL_SLOT + 1u + l] = select(listed * (BRICK_CELLS / WG), (cells + WG - 1u) / WG, full);
        }

        storageBarrier();
        below = flags;
        below_bricks = bricks;
    }
}

fn clear_brick(t: u32) {
    let s = t / BRICK_FACES;

    if s >= brick_list[1] {
        return;
    }

    let b = brick_list[BRICK_LIST_HEAD + brick_total() + s];
    let slot = brick_list[brick_table() + b];
    let r = t % BRICK_FACES;

    if slot == 0u {
        let c = brick_origin(b) + vec3<i32>(vec3<u32>(r % BRICK, (r / BRICK) % BRICK, r / (BRICK * BRICK)));

        if r < BRICK_CELLS && in_cells(c) {
            atomicStore(&surface[grid_index(c)], 0u);
        }

        return;
    }

    if r < BRICK_CELLS {
        let n = slot * BRICK_CELLS + r;
        let len = l_len();
        let empty = 3.0 * params.dx;
        phi[n] = empty;
        pressure_keep[n] = 0.0;
        l_phi[n] = empty;
        l_active[n] = 0u;
        l_diag[n] = 0.0;
        l_x[n] = 0.0;
        l_b[n] = 0.0;
        l_r[n] = 0.0;
        l_z[n] = 0.0;
        l_d[n] = 0.0;
        l_q[n] = 0.0;

        for (var a = 0u; a < 3u; a = a + 1u) {
            l_coef[a * len + n] = 0.0;
            l_off[a * len + n] = 0.0;
        }
    }

    if r < BRICK_NODES {
        nodes[slot * BRICK_NODES + r] = BIG;
    }

    let m = slot * BRICK_FACES + r;
    velocity[m] = 0.0;
    valid[m] = 0.0;
    weights[m] = 1.0;
    owners[m] = -1;
    solid_velocity[m] = 0.0;
}
