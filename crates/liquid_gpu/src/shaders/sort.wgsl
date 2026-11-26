const SCAN_ITEMS: u32 = 4u;
const SCAN_BLOCK: u32 = 1024u;

struct Scan {
    exclusive: u32,
    total: u32,
}

fn occupy(b: u32) {
    if (atomicLoad(&brick_flags[b]) & BRICK_OCCUPIED) == 0u {
        atomicOr(&brick_flags[b], BRICK_OCCUPIED);
    }
}

@compute @workgroup_size(256)
fn sort_count(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;

    if p < state.particle_count && particles[p].v.w <= 0.5 {
        let c = cell_of(particles[p].x.xyz);
        let g = grid_index(c);
        let rank = atomicAdd(&cell_counter[g], 1u);

        if CULL_CROWDED && params.physics.z <= 0.0 && rank >= 8u * PER_CELL {
            atomicSub(&cell_counter[g], 1u);
            particles[p].v.w = 1.0;
            return;
        }

        if SPARSE {
            occupy(brick_of(c));
        }
    }

    if !SPARSE || params.physics.z <= 0.0 {
        return;
    }

    for (var t = gid.x; t < brick_list[0] * BRICK_CELLS; t = t + dispatched_threads(PARTICLE_SLOT)) {
        let n = brick_cell(t);

        if n < cell_count() && band_phi[n] < 0.0 {
            occupy(brick_of(cell_coords(n)));
        }
    }
}

fn scan_block(block: u32, lane: Lane) {
    let base = block * SCAN_BLOCK + lane_rank(lane) * SCAN_ITEMS;
    var local: array<u32, 4>;
    var sum = 0u;

    for (var i = 0u; i < SCAN_ITEMS; i = i + 1u) {
        let n = base + i;
        var v = 0u;

        if n < grid_cells() {
            v = atomicLoad(&cell_counter[n]);
            atomicStore(&cell_counter[n], 0u);
        }

        local[i] = sum;
        sum = sum + v;
    }

    let scan = block_scan(lane, sum);

    for (var i = 0u; i < SCAN_ITEMS; i = i + 1u) {
        let n = base + i;

        if n < grid_cells() {
            cell_start[n] = scan.exclusive + local[i];
        }
    }

    if lane.index == 0u {
        partials[block] = bitcast<f32>(scan.total);
    }
}

var<workgroup> sort_on: u32;

fn sorting(lane: Lane) -> bool {
    if lane.index == 0u {
        sort_on = indirect[3u * GRID_SLOT];
    }

    return workgroupUniformLoad(&sort_on) != 0u;
}

@compute @workgroup_size(256)
fn sort_scan(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, lane: Lane) {
    if !sorting(lane) {
        return;
    }

    let blocks = (grid_cells() + SCAN_BLOCK - 1u) / SCAN_BLOCK;
    let workers = select(groups.x, groups.x - 1u, SPARSE && groups.x > 1u);

    if wid.x < workers {
        for (var block = wid.x; block < blocks; block = block + workers) {
            scan_block(block, lane);
        }
    }

    if SPARSE && wid.x + 1u == groups.x {
        compact_bricks(lane);
    }
}

fn start_of(n: u32) -> u32 {
    if n >= grid_cells() {
        return cell_start[grid_cells()];
    }

    return cell_start[n] + bitcast<u32>(partials[n / SCAN_BLOCK]);
}

fn scan_partials(lane: Lane) -> u32 {
    let blocks = (grid_cells() + SCAN_BLOCK - 1u) / SCAN_BLOCK;
    var carry = 0u;

    for (var start = 0u; start < blocks; start = start + WG) {
        let n = start + lane_rank(lane);
        var v = 0u;

        if n < blocks {
            v = bitcast<u32>(partials[n]);
        }

        let scan = block_scan(lane, v);

        if n < blocks {
            partials[n] = bitcast<f32>(carry + scan.exclusive);
        }

        carry = carry + scan.total;
    }

    return carry;
}

@compute @workgroup_size(256)
fn sort_link(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, lane: Lane) {
    if !sorting(lane) {
        return;
    }

    if wid.x == 0u {
        let total = scan_partials(lane);

        if lane.index == 0u {
            cell_start[grid_cells()] = total;
        }
    }

    if !SPARSE {
        return;
    }

    if wid.x == min(1u, groups.x - 1u) {
        level_bricks(lane);
    }

    let threads = groups.x * WG;
    let first = wid.x * WG + lane.index;
    let changed = brick_list[1];

    for (var t = first; t < 27u * brick_total(); t = t + threads) {
        link_brick(t);
    }

    for (var t = first; t < changed * BRICK_FACES; t = t + threads) {
        clear_brick(t);
    }

    if params.physics.z > 0.0 {
        for (var t = first; t < changed * BRICK_CELLS; t = t + threads) {
            clear_band(t);
        }
    }
}

@compute @workgroup_size(256)
fn scatter_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = gid.x;

    if p >= state.particle_count || particles[p].v.w > 0.5 {
        return;
    }

    let c = grid_index(cell_of(particles[p].x.xyz));
    let slot = start_of(c) + atomicAdd(&cell_counter[c], 1u);
    sorted[slot] = p;
}

@compute @workgroup_size(256)
fn rank_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= cell_start[grid_cells()] {
        return;
    }

    let key = sorted[i];
    let p = particles[key];
    let c = grid_index(cell_of(p.x.xyz));
    let begin = start_of(c);
    let end = start_of(c + 1u);
    var rank = 0u;

    for (var j = begin; j < end; j = j + 1u) {
        rank = rank + select(0u, 1u, sorted[j] < key);
    }

    sorted[params.particle_capacity + begin + rank] = key;
    particles_out[begin + rank] = p;
}

override SORT_PACK: bool = false;

@compute @workgroup_size(256)
fn adopt_band(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    for (var n = gid.x; n < grid_cells(); n = n + groups.x * WG) {
        let m = masked_cell(n);

        if m < cell_count() {
            phi[m] = band_phi[m];
        }
    }
}

@compute @workgroup_size(256)
fn sort_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let total = cell_start[grid_cells()];
    let i = gid.x;

    if i < total {
        let p = particles_out[i];
        particles[i] = p;
        sorted[i] = sorted[params.particle_capacity + i];

        if QUANTIZED {
            packed[i] = pack_position(p.x.xyz, p.c0.w);
            pack_motion(i, p);
        }

        if SORT_PACK {
            ellipsoids[i] = vec4<f32>(p.v.xyz, 0.0);
            ellipsoids[params.particle_capacity + i] = vec4<f32>(p.x.xyz, 0.0);
        }
    }

    if i == 0u {
        state.particle_count = total;
    }

    for (var n = i; n < grid_cells(); n = n + dispatched_threads(PARTICLE_SLOT)) {
        cell_start[n] = cell_start[n] + bitcast<u32>(partials[n / SCAN_BLOCK]);
        atomicStore(&cell_counter[n], 0u);

        if SORT_PACK {
            atomicStore(&surface[n], 0u);

            if params.physics.z > 0.0 {
                let m = masked_cell(n);

                if m < cell_count() {
                    phi[m] = band_phi[m];
                }
            }
        }
    }
}
