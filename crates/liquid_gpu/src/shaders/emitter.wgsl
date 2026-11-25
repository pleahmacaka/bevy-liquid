struct Emitter {
    center: vec4<f32>,
    extent: vec4<f32>,
    velocity: vec4<f32>,
    low: vec4<u32>,
    size: vec4<u32>,
    limit: vec4<u32>,
}

struct Emission {
    count: vec4<u32>,
    emitters: array<Emitter, 4>,
}

struct EmitterState {
    mark: u32,
    particles: u32,
    budget: f32,
    pad: u32,
}

struct EmissionState {
    slots: array<EmitterState, 4>,
    total: u32,
    pad: array<u32, 3>,
}

@group(3) @binding(10) var<uniform> emission: Emission;
@group(3) @binding(11) var<storage, read_write> emitted: EmissionState;

struct EmitCell {
    cell: vec3<i32>,
    open: u32,
}

struct Quota {
    earned: f32,
    spent: u32,
    room: u32,
}

var<workgroup> emit_live: u32;

fn emitting(lane: Lane) -> bool {
    if lane.index == 0u {
        emit_live = state.running;
    }

    return workgroupUniformLoad(&emit_live) != 0u;
}

var<workgroup> emit_room: u32;

fn shared_room(lane: Lane, quota: Quota) -> u32 {
    if lane.index == 0u {
        emit_room = quota.room;
    }

    return workgroupUniformLoad(&emit_room);
}

fn metered(e: Emitter) -> bool {
    return e.velocity.w >= 0.0;
}

fn emitter_holds(e: Emitter, x: vec3<f32>) -> bool {
    let p = x - e.center.xyz;

    if e.center.w < 0.5 {
        return dot(p, p) <= e.extent.x * e.extent.x;
    }

    return all(abs(p) <= e.extent.xyz);
}

fn liquid_free(c: vec3<i32>) -> bool {
    if atomicLoad(&cell_counter[grid_index(c)]) != 0u {
        return false;
    }

    return params.physics.z <= 0.0 || band_phi[cell_index(c)] >= 0.0;
}

fn emit_quota(k: u32) -> Quota {
    let e = emission.emitters[k];
    let kept = emitted.slots[k];
    let same = kept.mark == e.limit.y;
    var quota: Quota;
    quota.earned = 0.0;
    quota.spent = select(0u, kept.particles, same);
    quota.room = NONE;

    if metered(e) {
        quota.earned = select(0.0, kept.budget, same) + e.velocity.w * state.dt;
        quota.room = u32(quota.earned);
    }

    if e.limit.x != NONE {
        quota.room = min(quota.room, e.limit.x - min(quota.spent, e.limit.x));
    }

    return quota;
}

fn emit_cell(k: u32, t: u32) -> EmitCell {
    let e = emission.emitters[k];
    var job: EmitCell;
    job.open = 0u;

    if t >= e.size.w {
        return job;
    }

    var r = t;

    if metered(e) {
        r = (t + hash_u32(state.ticks + k) % e.size.w) % e.size.w;
    }

    let local = vec3<u32>(r % e.size.x, (r / e.size.x) % e.size.y, r / (e.size.x * e.size.y));
    job.cell = vec3<i32>(e.low.xyz + local);

    if emitter_holds(e, cell_center(job.cell)) && liquid_free(job.cell) {
        job.open = (1u << PER_CELL) - 1u;
    }

    return job;
}

fn emit_position(c: vec3<i32>, s: u32) -> vec3<f32> {
    let u = vec3<u32>(c);
    let seed = (u.x * 73856093u) ^ (u.y * 19349663u) ^ (u.z * 83492791u) ^ (s * 2654435761u) ^ hash_u32(state.ticks);
    let jitter = vec3<f32>(hash_unit(seed), hash_unit(seed ^ 0x9E3779B9u), hash_unit(seed ^ 0x7F4A7C15u)) - 0.5;

    return (vec3<f32>(c) + (sub_octant(s) + 0.5 + 0.8 * jitter) * 0.5) * params.dx;
}

fn clear_of_bodies(x: vec3<f32>) -> bool {
    for (var k = 0u; k < params.body_count; k = k + 1u) {
        if body_sdf(bodies[k], x) < params.margin {
            return false;
        }
    }

    return true;
}

fn static_distance(x: vec3<f32>) -> f32 {
    let g = x * params.inv_dx;
    let c = clamp(vec3<i32>(floor(g)), vec3<i32>(0), vec3<i32>(params.dims.xyz) - 1);
    let t = clamp(g - vec3<f32>(c), vec3<f32>(0.0), vec3<f32>(1.0));
    let d = params.dims.xyz + 1u;
    var s = 0.0;

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let o = vec3<i32>(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let w = select(1.0 - t, t, o == vec3<i32>(1));
        let p = vec3<u32>(c + o);
        s = s + w.x * w.y * w.z * static_solid[p.x + d.x * (p.y + d.y * p.z)];
    }

    return s;
}

fn emit_store(k: u32, job: EmitCell, first: u32) {
    let e = emission.emitters[k];
    var slot = first;

    for (var s = 0u; s < PER_CELL; s = s + 1u) {
        if ((job.open >> s) & 1u) == 0u || slot >= params.particle_capacity {
            continue;
        }

        let x = emit_position(job.cell, s);
        let c = cell_of(x);
        var p: Particle;
        p.x = vec4<f32>(x, -1.0 - f32(slot));
        p.v = vec4<f32>(e.velocity.xyz, 0.0);
        p.c0 = vec4<f32>(vec3<f32>(0.0), e.extent.w);
        p.c1 = vec4<f32>(0.0);
        p.c2 = vec4<f32>(0.0);
        particles[slot] = p;
        atomicAdd(&cell_counter[grid_index(c)], 1u);

        if SPARSE {
            occupy(brick_of(c));
        }

        slot = slot + 1u;
    }
}

fn emit_block(lane: Lane, k: u32, cell: EmitCell, first: u32, room: u32) -> u32 {
    let e = emission.emitters[k];
    let own = countOneBits(cell.open);
    let scan = block_scan(lane, own);
    var job = cell;
    var total = scan.total;

    if metered(e) || e.limit.x != NONE {
        if scan.exclusive + own > room {
            job.open = 0u;
        }

        total = block_scan(lane, countOneBits(job.open)).total;
    }

    emit_store(k, job, first + scan.exclusive);
    // a later emitter reads the counters this block filled, so every lane has to finish storing first
    storageBarrier();

    return total;
}

fn held(count: u32) -> u32 {
    return min(count, params.particle_capacity);
}

fn emit_settle(k: u32, quota: Quota, done: u32) {
    let e = emission.emitters[k];
    var budget = 0.0;

    if metered(e) {
        budget = min(quota.earned - f32(done), f32(PER_CELL));
    }

    emitted.slots[k] = EmitterState(e.limit.y, quota.spent + done, budget, 0u);
}

fn emit_finish(next: u32) {
    let added = held(next) - state.particle_count;
    state.particle_count = held(next);
    emitted.total = emitted.total + added;

    if params.physics.z > 0.0 {
        scalars[S_VOLUME_EMITTED] = scalars[S_VOLUME_EMITTED] + f32(added) * params.dx * params.dx * params.dx / f32(PER_CELL);
    }
}

@compute @workgroup_size(256)
fn emit(lane: Lane) {
    if !emitting(lane) {
        return;
    }

    var next = state.particle_count;

    for (var k = 0u; k < emission.count.x; k = k + 1u) {
        let quota = emit_quota(k);
        let room = shared_room(lane, quota);
        let first = next;

        for (var start = 0u; room > 0u && start < emission.emitters[k].size.w; start = start + WG) {
            var job = emit_cell(k, start + lane_rank(lane));

            for (var s = 0u; s < PER_CELL; s = s + 1u) {
                if ((job.open >> s) & 1u) == 1u && !clear_of_bodies(emit_position(job.cell, s)) {
                    job.open = job.open ^ (1u << s);
                }
            }

            next = next + emit_block(lane, k, job, next, room - min(next - first, room));
        }

        if lane.index == 0u && emission.emitters[k].size.w > 0u {
            emit_settle(k, quota, held(next) - held(first));
        }
    }

    if lane.index == 0u {
        emit_finish(next);
    }
}

@compute @workgroup_size(256)
fn emit_static(lane: Lane) {
    if !emitting(lane) {
        return;
    }

    var next = state.particle_count;

    for (var k = 0u; k < emission.count.x; k = k + 1u) {
        let quota = emit_quota(k);
        let room = shared_room(lane, quota);
        let first = next;

        for (var start = 0u; room > 0u && start < emission.emitters[k].size.w; start = start + WG) {
            var job = emit_cell(k, start + lane_rank(lane));

            for (var s = 0u; s < PER_CELL; s = s + 1u) {
                let x = emit_position(job.cell, s);

                if ((job.open >> s) & 1u) == 1u && (!clear_of_bodies(x) || static_distance(x) < params.margin) {
                    job.open = job.open ^ (1u << s);
                }
            }

            next = next + emit_block(lane, k, job, next, room - min(next - first, room));
        }

        if lane.index == 0u && emission.emitters[k].size.w > 0u {
            emit_settle(k, quota, held(next) - held(first));
        }
    }

    if lane.index == 0u {
        emit_finish(next);
    }
}

@compute @workgroup_size(1)
fn emit_grow() {
    if state.running == 1u {
        indirect[3u * PARTICLE_SLOT] = particle_groups();
    }
}
