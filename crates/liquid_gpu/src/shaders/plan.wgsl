@group(3) @binding(0) var<storage, read_write> indirect: array<u32>;

override SLOT_COUNT: u32 = 16u;
override SOLVE_FIRST: u32 = 8u;
override PARTICLE_SLOT: u32 = 0u;
override GRID_SLOT: u32;
override CELL_SLOT: u32;
override FACE_SLOT: u32;
override NODE_SLOT: u32;
override CLEAR_SLOT: u32;
override LEVEL_SLOT: u32;
override SOLVE_LEVEL_SLOT: u32;
override SOLVE_FACE_SLOT: u32;
override BODY_SLOT: u32;
override SOLVE_BODY_SLOT: u32;

fn full_size(s: u32) -> u32 {
    return indirect[3u * SLOT_COUNT + s];
}

fn particle_groups() -> u32 {
    return max((state.particle_count + WG - 1u) / WG, 1u);
}

fn write_slots(first: u32, on: bool) {
    for (var s = first; s < SLOT_COUNT; s = s + 1u) {
        var groups = 0u;

        if on {
            groups = full_size(s);

            if s == PARTICLE_SLOT {
                groups = particle_groups();
            }
        }

        indirect[3u * s] = groups;
    }
}

@compute @workgroup_size(1)
fn begin_frame() {
    state.remaining = params.frame_dt;
    state.substeps = 0u;
    state.running = 1u;
    state.converged = 0u;
    atomicStore(&state.vmax_bits, 0u);

    for (var i = 0u; i < 6u * MAX_BODIES; i = i + 1u) {
        scalars[S_IMPULSE + i] = 0.0;
    }

    write_slots(0u, true);
}

@compute @workgroup_size(1)
fn open_all() {
    write_slots(0u, true);
}

@compute @workgroup_size(1)
fn reset_speed() {
    atomicStore(&state.vmax_bits, 0u);
}

fn measure_particle(p: u32) {
    if p < state.particle_count {
        atomicMax(&state.vmax_bits, bitcast<u32>(length(particles[p].v.xyz)));
    }

    if p < params.body_count {
        atomicMax(&state.vmax_bits, bitcast<u32>(length(bodies[p].velocity.xyz)));
    }
}

fn measure_faces(first: u32, stride: u32) {
    if params.physics.z <= 0.0 {
        return;
    }

    let items = select(face_total(), brick_list[0] * BRICK_FACES, SPARSE);

    for (var t = first; t < items; t = t + stride) {
        let n = brick_face(t);

        if n < face_total() {
            atomicMax(&state.vmax_bits, bitcast<u32>(abs(velocity[n])));
        }
    }
}

@compute @workgroup_size(256)
fn measure_speed(@builtin(global_invocation_id) gid: vec3<u32>) {
    measure_particle(gid.x);
}

@compute @workgroup_size(256)
fn measure_flow(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    measure_particle(gid.x);
    measure_faces(gid.x, groups.x * WG);
}

override PLAN_GROW: bool = false;

@compute @workgroup_size(1)
fn plan_substep() {
    if PLAN_GROW && state.running == 1u {
        state.particle_count = min(state.particle_count + boundary_cell_count() * SPAWN_PER_CELL, params.particle_capacity);
    }

    if state.remaining <= 1.0e-6 * params.frame_dt || state.running == 0u {
        state.running = 0u;
        state.dt = 0.0;
        write_slots(0u, false);
        return;
    }

    let vmax = bitcast<f32>(atomicLoad(&state.vmax_bits));
    let speed = vmax + sqrt(5.0 * params.dx * length(params.gravity.xyz));
    var dt = min(params.cfl * params.dx / max(speed, 1.0e-6), params.max_dt);
    let sigma = params.physics.x;

    if sigma > 0.0 {
        dt = min(dt, sqrt(params.density * params.dx * params.dx * params.dx / (6.28318530718 * sigma)));
    }

    let remaining = state.remaining;

    if dt >= remaining * (1.0 - 1.0e-5) {
        dt = remaining;
    } else if 2.0 * dt > remaining {
        dt = 0.5 * remaining;
    }

    state.dt = dt;
    state.remaining = remaining - dt;
    state.time = state.time + dt;
    state.substeps = state.substeps + 1u;
    state.ticks = state.ticks + 1u;
    state.converged = 0u;
    write_slots(0u, true);
}

@compute @workgroup_size(1)
fn open_solve() {
    state.converged = 0u;
    scalars[S_ITERATION] = 0.0;
    write_slots(SOLVE_FIRST, state.running == 1u);
}

@compute @workgroup_size(256)
fn reduce_check(lane: Lane) {
    let total = reduce_total(lane);

    if lane.index != 0u {
        return;
    }

    if state.converged == 1u {
        write_slots(SOLVE_FIRST, false);
        return;
    }

    if state.running == 1u {
        scalars[S_RNORM] = total;
    }

    if scalars[S_RNORM] <= scalars[S_TOLERANCE] * max(scalars[S_BNORM], 1.0e-30) {
        state.converged = 1u;
    }

    scalars[S_ITERATION] = scalars[S_ITERATION] + 1.0;
    write_slots(SOLVE_FIRST, state.running == 1u && state.converged == 0u);
}

@compute @workgroup_size(1)
fn set_pressure_scale_gravity() {
    add_body_gravity();
    pressure_scale();
}

@compute @workgroup_size(1)
fn set_pressure_scale() {
    pressure_scale();
}

fn pressure_scale() {
    scalars[S_SCALE] = state.dt / (params.density * params.dx * params.dx);
    scalars[S_COUPLING] = state.dt / (params.dx * params.dx * params.dx);
    scalars[S_GRADIENT] = state.dt / (params.density * params.dx);
    scalars[S_FLOOR] = -params.density * params.gravity.y;
}

@compute @workgroup_size(1)
fn set_density_scale() {
    scalars[S_SCALE] = params.inv_dx * params.inv_dx;
    scalars[S_COUPLING] = 0.0;
    scalars[S_GRADIENT] = params.inv_dx;
    scalars[S_FLOOR] = 0.0;
}

@compute @workgroup_size(1)
fn floor_grow() {
    if state.running == 0u {
        return;
    }

    state.particle_count = min(state.particle_count + FLOOR_SPAWN * params.dims.x * params.dims.z, params.particle_capacity);
    indirect[3u * PARTICLE_SLOT] = particle_groups();
}
