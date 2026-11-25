const FLOOR_SAMPLES: i32 = 32;

@compute @workgroup_size(256)
fn floor_surface(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() {
        return;
    }

    let c = vec3<i32>(i32(col % lv.dims.x), 0, i32(col / lv.dims.x));
    let top = i32(lv.dims.y);
    var below = l_phi[l_index(c)];
    var eta = f32(top) * params.dx;

    if below >= 0.0 {
        eta = 0.0;
    } else {
        for (var j = 1; j < top; j = j + 1) {
            let above = l_phi[l_index(vec3<i32>(c.x, j, c.z))];

            if above >= 0.0 {
                let theta = max(below / (below - above), THETA_MIN);
                eta = (f32(j) - 0.5 + theta) * params.dx;
                break;
            }

            below = above;
        }
    }

    l_coef[surface_slot(col)] = eta;
}

@compute @workgroup_size(256)
fn floor_mean(lane: Lane) {
    let cols = l_columns();
    var acc = 0.0;

    for (var col = lane.index; col < cols; col = col + WG) {
        acc = acc + l_coef[surface_slot(col)];
    }

    let total = reduce_workgroup(lane, acc);

    if lane.index == 0u {
        let mean = total / f32(cols);
        scalars[S_FLOOR_MEAN] = mean;

        if state.running == 1u && scalars[S_FLOOR] > 0.0 {
            if scalars[S_FLOOR_SET] == 0.0 {
                scalars[S_FLOOR_LEVEL] = mean;
                scalars[S_FLOOR_RATE] = 0.0;
                scalars[S_FLOOR_SET] = 1.0;
            }

            let level = scalars[S_FLOOR_LEVEL];
            scalars[S_FLOOR_OFFSET] = level - mean - sqrt(level * params.density / scalars[S_FLOOR]) * scalars[S_FLOOR_RATE];
        }
    }
}

fn mirror(e: i32, n: i32) -> i32 {
    let m = ((e % (2 * n)) + 2 * n) % (2 * n);
    return select(m, 2 * n - 1 - m, m >= n);
}

@compute @workgroup_size(256)
fn floor_ghost(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() {
        return;
    }

    let nx = i32(lv.dims.x);
    let nz = i32(lv.dims.z);
    let i = i32(col) % nx;
    let k = i32(col) / nx;
    let dx = params.dx;
    let eta = l_coef[surface_slot(col)];
    let h = max(eta + 0.5 * dx, dx);
    let span = i32(ceil(FLOOR_REACH * h / dx));
    var seen = eta;

    if span > 0 {
        let stride = (span + FLOOR_SAMPLES - 1) / FLOOR_SAMPLES;
        let count = (span + stride - 1) / stride;
        let area = f32(stride) * dx * f32(stride) * dx;
        var sum = 0.0;
        var total = 0.0;

        for (var mz = 0; mz <= 2 * count; mz = mz + 1) {
            for (var mx = 0; mx <= 2 * count; mx = mx + 1) {
                let ox = (mx - count) * stride;
                let oz = (mz - count) * stride;
                let q = f32(ox * ox + oz * oz) * dx * dx + h * h;
                let w = h * area / (6.28318530718 * q * sqrt(q));
                let src = u32(mirror(i + ox, nx) + nx * mirror(k + oz, nz));

                sum = sum + w * l_coef[surface_slot(src)];
                total = total + w;
            }
        }

        let tail = max(1.0 - total, 0.0);
        seen = (sum + tail * scalars[S_FLOOR_MEAN]) / (total + tail);
    }

    l_coef[ghost_slot(col)] = scalars[S_FLOOR] * (seen + scalars[S_FLOOR_OFFSET] + 0.5 * dx);
}

@compute @workgroup_size(256)
fn floor_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() {
        return;
    }

    let n = l_index(vec3<i32>(i32(col % lv.dims.x), 0, i32(col / lv.dims.x)));

    if l_active[n] != 0u {
        l_b[n] = l_b[n] + l_coef[floor_slot(col)] * l_coef[ghost_slot(col)];
    }
}

@compute @workgroup_size(256)
fn floor_spawn(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let slot = state.particle_count + t;

    if state.running == 0u || t >= PER_CELL * params.dims.x * params.dims.z || slot >= params.particle_capacity {
        return;
    }

    let col = t / PER_CELL;
    let s = t % PER_CELL;
    let cell = vec3<u32>(col % params.dims.x, 0u, col / params.dims.x);
    let seed = (cell.x * 73856093u) ^ (cell.z * 83492791u) ^ (s * 2654435761u) ^ hash_u32(state.ticks);
    let jitter = vec3<f32>(hash_unit(seed), hash_unit(seed ^ 0x9E3779B9u), hash_unit(seed ^ 0x7F4A7C15u));
    let sub = sub_octant(s);
    let x = (vec3<f32>(cell) + (sub + jitter) * 0.5 - vec3<f32>(0.0, 1.0, 0.0)) * params.dx;
    var p: Particle;
    p.x = vec4<f32>(x, -1.0);
    p.v = vec4<f32>(0.0);
    p.c0 = vec4<f32>(vec3<f32>(0.0), params.thermal.x);
    p.c1 = vec4<f32>(0.0);
    p.c2 = vec4<f32>(0.0);
    particles[slot] = p;
}

@compute @workgroup_size(256)
fn floor_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i < state.particle_count && particles[i].x.y < 0.0 {
        particles[i].v.w = 1.0;
    }
}

@compute @workgroup_size(256)
fn floor_close(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() && face_axis(n) == 1u && face_coords(1u, n).y == 0 {
        weights[n] = 0.0;
    }
}

@compute @workgroup_size(256)
fn floor_flux(lane: Lane) {
    let cols = l_columns();
    var acc = 0.0;

    for (var col = lane.index; col < cols; col = col + WG) {
        let c = vec3<i32>(i32(col % lv.dims.x), 0, i32(col / lv.dims.x));
        let open = l_coef[floor_slot(col)];

        if open > 0.0 && l_active[l_index(c)] != 0u {
            acc = acc + open * velocity[face_index(1u, c)];
        }
    }

    let total = reduce_workgroup(lane, acc);

    if lane.index == 0u && state.running == 1u {
        scalars[S_FLOOR_RATE] = total / (scalars[S_SCALE] * f32(cols));
    }
}
