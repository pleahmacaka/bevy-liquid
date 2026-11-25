override BLUR: u32 = 0u;

const CURVATURE_BAND: f32 = 3.0;

fn blur_read(n: u32) -> f32 {
    switch BLUR {
        case 0u: {
            return signed_distance[n];
        }
        case 1u: {
            return sweep_a[n];
        }
        default: {
            return sweep_b[n];
        }
    }
}

fn blur_write(n: u32, v: f32) {
    if BLUR == 1u {
        sweep_b[n] = v;
    } else {
        sweep_a[n] = v;
    }
}

fn clamped_cell(c: vec3<i32>) -> u32 {
    let p = clamp(c, vec3<i32>(0), vec3<i32>(params.dims.xyz) - 1);

    if SPARSE && !brick_live(p) {
        return NONE;
    }

    return cell_index(p);
}

fn blur_at(c: vec3<i32>, own: f32) -> f32 {
    let m = clamped_cell(c);

    if m == NONE {
        return own;
    }

    return blur_read(m);
}

@compute @workgroup_size(256)
fn blur(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    let c = cell_coords(n);
    let own = blur_read(n);
    let ex = vec3<i32>(1, 0, 0);
    let ey = vec3<i32>(0, 1, 0);
    let ez = vec3<i32>(0, 0, 1);
    var neighbors = blur_at(c + ex, own) + blur_at(c - ex, own);
    neighbors = neighbors + (blur_at(c + ey, own) + blur_at(c - ey, own));
    neighbors = neighbors + (blur_at(c + ez, own) + blur_at(c - ez, own));

    blur_write(n, 0.5 * own + neighbors / 12.0);
}

fn smoothed_at(c: vec3<i32>, own: f32) -> f32 {
    let m = clamped_cell(c);

    if m == NONE {
        return own;
    }

    return sweep_b[m];
}

fn mean_curvature(c: vec3<i32>, center: f32) -> f32 {
    let dx = params.dx;
    var g = vec3<f32>(0.0);

    for (var a = 0u; a < 3u; a = a + 1u) {
        let e = unit_axis(a);
        g = with_axis(g, a, (smoothed_at(c + e, center) - smoothed_at(c - e, center)) / (2.0 * dx));
    }

    let norm2 = g.x * g.x + g.y * g.y + g.z * g.z;

    if norm2 < 1.0e-12 {
        return 0.0;
    }

    var numerator = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        let b = (a + 1u) % 3u;
        let d = (a + 2u) % 3u;
        let ea = unit_axis(a);
        let eb = unit_axis(b);
        let haa = (smoothed_at(c + ea, center) - 2.0 * center + smoothed_at(c - ea, center)) / (dx * dx);
        let hab = (smoothed_at(c + ea + eb, center) - smoothed_at(c + ea - eb, center) - smoothed_at(c + eb - ea, center)
            + smoothed_at(c - ea - eb, center)) / (4.0 * dx * dx);

        numerator = numerator + haa * (g[b] * g[b] + g[d] * g[d]);
        numerator = numerator - 2.0 * g[a] * g[b] * hab;
    }

    return numerator / (norm2 * sqrt(norm2));
}

@compute @workgroup_size(256)
fn curvature(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    let center = sweep_b[n];

    if abs(center) > CURVATURE_BAND * params.dx {
        kappa[n] = 0.0;
        return;
    }

    kappa[n] = clamp(mean_curvature(cell_coords(n), center), -params.inv_dx, params.inv_dx);
}

fn ghost_theta(liquid: u32, air: u32) -> f32 {
    if l_active[liquid] != 0u && l_active[air] == 0u && l_phi[air] >= 0.0 {
        return max(l_phi[liquid] / (l_phi[liquid] - l_phi[air]), THETA_MIN);
    }

    return -1.0;
}

fn cut_curvature(n: u32) -> f32 {
    let c = cell_coords(n);
    let len = l_len();
    var sum = 0.0;
    var count = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var side = -1; side <= 1; side = side + 2) {
            let p = bump(c, a, side);

            if !in_cells(p) {
                continue;
            }

            let m = cell_index(p);
            var k = l_coef[a * len + m];

            if side > 0 {
                k = l_coef[a * len + n];
            }

            if k <= 0.0 {
                continue;
            }

            let theta = ghost_theta(n, m);

            if theta > 0.0 {
                sum = sum + ((1.0 - theta) * kappa[n] + theta * kappa[m]);
                count = count + 1.0;
            }
        }
    }

    if count > 0.0 {
        return sum / count;
    }

    return BIG;
}

@compute @workgroup_size(256)
fn jump_seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    sweep_a[n] = cut_curvature(n);
}

override SMOOTH: u32 = 0u;

fn smooth_read(n: u32) -> f32 {
    if SMOOTH == 0u {
        return sweep_a[n];
    }

    return sweep_b[n];
}

fn neighborhood_mean(n: u32) -> f32 {
    let c = cell_coords(n);
    var sum = 0.0;
    var count = 0.0;

    for (var o = 0; o < 27; o = o + 1) {
        let p = c + vec3<i32>(o % 3, (o / 3) % 3, o / 9) - 1;

        if !in_cells(p) || (SPARSE && !brick_live(p)) {
            continue;
        }

        let v = smooth_read(cell_index(p));

        if v < 0.5 * BIG {
            sum = sum + v;
            count = count + 1.0;
        }
    }

    if count > 0.0 {
        return sum / count;
    }

    return BIG;
}

@compute @workgroup_size(256)
fn jump_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    var mean = BIG;

    if l_active[n] != 0u {
        mean = neighborhood_mean(n);
    }

    if SMOOTH == 0u {
        sweep_b[n] = mean;
    } else {
        sweep_a[n] = select(mean, 0.0, mean > 0.5 * BIG);
    }
}

const RESOLVED_CURVATURE: f32 = 0.45;
const THIN_CURVATURE: f32 = 0.6;

fn interface_pressure(liquid: u32, air: u32, theta: f32) -> f32 {
    let averaged = sweep_a[liquid];
    let thin = smoothstep(RESOLVED_CURVATURE, THIN_CURVATURE, abs(averaged) * params.dx);
    let sharp = (1.0 - theta) * kappa[liquid] + theta * kappa[air];

    return params.physics.x * mix(sharp, averaged, thin);
}

@compute @workgroup_size(256)
fn jump_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() || l_active[n] == 0u {
        return;
    }

    let c = cell_coords(n);
    let len = l_len();
    var rhs = l_b[n];

    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var side = -1; side <= 1; side = side + 2) {
            let p = bump(c, a, side);

            if !in_cells(p) {
                continue;
            }

            let m = cell_index(p);
            var k = l_coef[a * len + m];

            if side > 0 {
                k = l_coef[a * len + n];
            }

            if k <= 0.0 {
                continue;
            }

            let theta = ghost_theta(n, m);

            if theta > 0.0 {
                rhs = rhs + k / theta * interface_pressure(n, m, theta);
            }
        }
    }

    l_b[n] = rhs;
}

@compute @workgroup_size(256)
fn jump_gradient(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n >= face_total() || weights[n] <= 0.0 {
        return;
    }

    let axis = face_axis(n);
    let f = face_coords(axis, n);

    if f[axis] == 0 || f[axis] == i32(params.dims[axis]) {
        return;
    }

    let lo = cell_index(bump(f, axis, -1));
    let hi = cell_index(f);
    let scale = scalars[S_GRADIENT];
    let forward = ghost_theta(lo, hi);

    if forward > 0.0 {
        velocity[n] = velocity[n] - scale * interface_pressure(lo, hi, forward) / forward;
        return;
    }

    let backward = ghost_theta(hi, lo);

    if backward > 0.0 {
        velocity[n] = velocity[n] + scale * interface_pressure(hi, lo, backward) / backward;
    }
}
