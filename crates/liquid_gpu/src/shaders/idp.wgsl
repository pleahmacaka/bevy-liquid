const PARTICLES_PER_CELL: f32 = 8.0;
const KERNEL_SHARE: array<f32, 3> = array<f32, 3>(0.15625, 0.6875, 0.15625);

fn near_body(c: vec3<i32>) -> bool {
    if params.body_count == 0u {
        return false;
    }

    for (var o = 0; o < 27; o = o + 1) {
        let p = c + vec3<i32>(o % 3, (o / 3) % 3, o / 9) - vec3<i32>(1);

        if in_cells(p) && 0.5 + solid_phi_at_cell(p) * params.inv_dx < 1.0 {
            return true;
        }
    }

    return false;
}

fn wall_coverage(c: vec3<i32>) -> f32 {
    var sum = 0.0;

    for (var o = 0; o < 27; o = o + 1) {
        let d = vec3<i32>(o % 3, (o / 3) % 3, o / 9);

        if in_cells(c + d - vec3<i32>(1)) {
            sum = sum + KERNEL_SHARE[d.x] * KERNEL_SHARE[d.y] * KERNEL_SHARE[d.z];
        }
    }

    return sum;
}

fn is_interior(c: vec3<i32>) -> bool {
    for (var o = 0; o < 27; o = o + 1) {
        let p = c + vec3<i32>(o % 3, (o / 3) % 3, o / 9) - vec3<i32>(1);

        if in_cells(p) && l_active[cell_index(p)] == 0u {
            return false;
        }
    }

    return true;
}

@compute @workgroup_size(256)
fn density_error(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = masked_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    l_x[n] = 0.0;

    if l_active[n] == 0u {
        l_b[n] = 0.0;
        return;
    }

    let c = cell_coords(n);

    if near_body(c) {
        l_b[n] = 0.0;
        return;
    }

    let lo = max(c - vec3<i32>(1), vec3<i32>(0));
    let hi = min(c + vec3<i32>(1), vec3<i32>(params.dims.xyz) - 1);
    let node = vec3<f32>(c);
    var sum = 0.0;

    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
                let cell = grid_index(vec3<i32>(cx, cy, cz));

                for (var i = cell_start[cell]; i < cell_start[cell + 1u]; i = i + 1u) {
                    let g = particles[i].x.xyz * params.inv_dx - 0.5;
                    sum = sum + bspline2_weight(g - node);
                }
            }
        }
    }

    let e = sum / (PARTICLES_PER_CELL * wall_coverage(c)) - 1.0;

    if is_interior(c) {
        l_b[n] = e;
    } else {
        l_b[n] = max(e, 0.0);
    }
}
