@group(2) @binding(0) var<uniform> cv: LevelParams;
@group(2) @binding(1) var<storage, read_write> c_phi: array<f32>;
@group(2) @binding(2) var<storage, read_write> c_coef: array<f32>;
@group(2) @binding(3) var<storage, read_write> c_active: array<u32>;
@group(2) @binding(4) var<storage, read_write> c_diag: array<f32>;
@group(2) @binding(5) var<storage, read_write> c_off: array<f32>;
@group(2) @binding(6) var<storage, read_write> c_x: array<f32>;
@group(2) @binding(7) var<storage, read_write> c_b: array<f32>;
@group(2) @binding(8) var<storage, read_write> c_r: array<f32>;
@group(2) @binding(9) var<storage, read_write> c_z: array<f32>;

override COLOR: u32 = 0u;
override MASK: u32 = 1u;

fn c_len() -> u32 {
    return cv.dims.w;
}

fn c_coords(n: u32) -> vec3<i32> {
    let d = cv.dims.xyz;
    return vec3<i32>(vec3<u32>(n % d.x, (n / d.x) % d.y, n / (d.x * d.y)));
}

fn c_index(c: vec3<i32>) -> u32 {
    let d = vec3<i32>(cv.dims.xyz);
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn c_listed(t: u32) -> u32 {
    if SPARSE && cv.bricks != 0u {
        return set_cell(t, cv.dims.xyz, cv.bricks);
    }

    return t;
}

@compute @workgroup_size(256)
fn level_from_fine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = level_cell(gid.x);

    if n >= l_len() {
        return;
    }

    let c = l_coords(n);
    let len = l_len();
    l_phi[n] = phi[n];

    for (var a = 0u; a < 3u; a = a + 1u) {
        var k = 0.0;

        if c[a] + 1 < i32(lv.dims[a]) {
            k = weights[face_index(a, bump(c, a, 1))] * scalars[S_SCALE];
        }

        l_coef[a * len + n] = k;
    }

    if OPEN_FLOOR && c.y == 0 {
        l_coef[floor_slot(l_column(c))] = weights[face_index(1u, c)] * scalars[S_SCALE];
    }
}

@compute @workgroup_size(256)
fn level_classify(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = level_cell(gid.x);

    if n >= l_len() {
        return;
    }

    let c = l_coords(n);
    let len = l_len();
    var connected = l_floor(c) > 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        if l_coef[a * len + n] > 0.0 {
            connected = true;
        }

        let m = bump(c, a, -1);

        if m[a] >= 0 && l_coef[a * len + l_index(m)] > 0.0 {
            connected = true;
        }
    }

    l_active[n] = select(0u, 1u, l_phi[n] < 0.0 && connected);
}

@compute @workgroup_size(256)
fn level_assemble(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = level_cell(gid.x);

    if n >= l_len() {
        return;
    }

    let len = l_len();
    var diag = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        l_off[a * len + n] = 0.0;
    }

    if l_active[n] == 0u {
        l_diag[n] = 0.0;
        l_x[n] = 0.0;
        l_r[n] = 0.0;
        return;
    }

    let c = l_coords(n);
    var coupled = 0u;

    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var step = -1; step <= 1; step = step + 2) {
            let p = bump(c, a, step);

            if !l_inside(p) {
                continue;
            }

            let m = l_index(p);
            var k = 0.0;

            if step > 0 {
                k = l_coef[a * len + n];
            } else {
                k = l_coef[a * len + m];
            }

            if k <= 0.0 {
                continue;
            }

            if l_active[m] != 0u {
                diag = diag + k;
                coupled = coupled + 1u;

                if step > 0 {
                    l_off[a * len + n] = k;
                }
            } else if l_phi[m] >= 0.0 {
                let theta = max(l_phi[n] / (l_phi[n] - l_phi[m]), THETA_MIN);
                diag = diag + k / theta;
            }
        }
    }

    l_diag[n] = (diag + l_floor(c)) * (1.0 + lv.regularization);

    if coupled < 6u {
        l_active[n] = 3u;
    }
}

override BOUNDARY_REACH: i32 = 1;

@compute @workgroup_size(256)
fn level_band(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = level_cell(gid.x);

    if n >= l_len() || l_active[n] == 0u {
        return;
    }

    let c = l_coords(n);
    let lo = max(c - vec3<i32>(BOUNDARY_REACH), vec3<i32>(0));
    let hi = min(c + vec3<i32>(BOUNDARY_REACH), vec3<i32>(lv.dims.xyz) - 1);

    for (var z = lo.z; z <= hi.z; z = z + 1) {
        for (var y = lo.y; y <= hi.y; y = y + 1) {
            for (var x = lo.x; x <= hi.x; x = x + 1) {
                if (l_active[l_index(vec3<i32>(x, y, z))] & 2u) != 0u {
                    l_active[n] = l_active[n] | 4u;
                    return;
                }
            }
        }
    }
}

fn fine_coef(a: u32, f: vec3<i32>, len: u32) -> f32 {
    if level_bricked() && !brick_live(f) {
        return scalars[S_SCALE];
    }

    return l_coef[a * len + l_index(f)];
}

@compute @workgroup_size(256)
fn coarsen(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;

    if n >= c_len() {
        return;
    }

    let cc = c_coords(n);
    let fine = vec3<i32>(lv.dims.xyz);
    let len_f = l_len();
    let len_c = c_len();
    var sum = 0.0;
    var count = 0.0;

    for (var o = 0; o < 8; o = o + 1) {
        let f = cc * 2 + vec3<i32>(o & 1, (o >> 1) & 1, (o >> 2) & 1);

        if all(f < fine) {
            sum = sum + l_phi[l_index(f)];
            count = count + 1.0;
        }
    }

    c_phi[n] = sum / count;

    for (var a = 0u; a < 3u; a = a + 1u) {
        let t1 = (a + 1u) % 3u;
        let t2 = (a + 2u) % 3u;
        var k = 0.0;

        for (var o = 0; o < 4; o = o + 1) {
            let f = bump(bump(bump(cc * 2, a, 1), t1, o & 1), t2, o >> 1);

            if all(bump(f, a, 1) < fine) {
                k = k + fine_coef(a, f, len_f);
            }
        }

        c_coef[a * len_c + n] = k / 16.0;
    }

    if OPEN_FLOOR && cc.y == 0 {
        var k = 0.0;

        for (var o = 0; o < 4; o = o + 1) {
            let f = cc * 2 + vec3<i32>(o & 1, 0, o >> 1);

            if all(f < fine) {
                k = k + fine_floor(f);
            }
        }

        c_coef[3u * len_c + u32(cc.x) + cv.dims.x * u32(cc.z)] = k / 16.0;
    }
}

fn fine_floor(f: vec3<i32>) -> f32 {
    if level_bricked() && !brick_live(f) {
        return scalars[S_SCALE];
    }

    return l_floor(f);
}

@compute @workgroup_size(256)
fn mg_zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = level_cell(gid.x);

    if n < l_len() {
        l_x[n] = 0.0;
    }
}

fn smooth_cell(n: u32) {
    if (l_active[n] & MASK) == 0u || tall_inside(n) {
        return;
    }

    let c = l_coords(n);

    if u32(c.x + c.y + c.z) % 2u != COLOR || tall_kind(c) != 0u {
        return;
    }

    l_x[n] = (l_b[n] + l_neighbor_sum_x(n)) / l_diag[n];
}

@compute @workgroup_size(256)
fn mg_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = listed_cell(gid.x);

    if n < l_len() {
        smooth_cell(n);
    }
}

@compute @workgroup_size(256)
fn mg_finish(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
    lane: Lane,
) {
    let n = masked_cell(gid.x);
    var rz = 0.0;

    if n < l_len() {
        smooth_cell(n);
        rz = l_b[n] * l_x[n];
    }

    store_partial(lane, R_RZ, wid.x, group_reduce(lane, rz, false));
}

override SWEEPS: u32 = 8u;
override SMOOTHING: u32 = 2u;
override BAND_SWEEPS: u32 = 0u;
override TAIL_COUNT: u32 = 1u;
override TAIL_CELLS: u32 = 1u;

var<workgroup> tail_lds: array<f32, TAIL_WORDS>;

struct Tail {
    dims: vec3<i32>,
    len: u32,
    cell: u32,
}

@compute @workgroup_size(256)
fn tail_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    let len = l_len();

    if n >= len {
        return;
    }

    let p = lv.pack + 4u * n;
    l_pack[p] = select(l_diag[n], -l_diag[n], (l_active[n] & 4u) != 0u);
    l_pack[p + 1u] = l_off[n];
    l_pack[p + 2u] = l_off[len + n];
    l_pack[p + 3u] = l_off[2u * len + n];
}

fn tail_level(i: u32) -> Tail {
    var d = vec3<i32>(lv.dims.xyz);
    var cell = 0u;

    for (var k = 0u; k < i; k = k + 1u) {
        cell = cell + u32(d.x * d.y * d.z);
        d = (d + vec3<i32>(1)) / 2;
    }

    return Tail(d, u32(d.x * d.y * d.z), cell);
}

fn tail_coords(t: Tail, n: u32) -> vec3<i32> {
    let d = vec3<u32>(t.dims);
    return vec3<i32>(vec3<u32>(n % d.x, (n / d.x) % d.y, n / (d.x * d.y)));
}

fn tail_index(t: Tail, c: vec3<i32>) -> u32 {
    return u32(c.x + t.dims.x * (c.y + t.dims.y * c.z));
}

fn tail_coef(t: Tail, n: u32, k: u32) -> f32 {
    return tail_lds[4u * (t.cell + n) + k];
}

fn tail_live(t: Tail, n: u32, mask: u32) -> bool {
    let d = tail_coef(t, n, 0u);

    return select(d != 0.0, d < 0.0, mask == 4u);
}

fn tail_x(t: Tail, n: u32) -> u32 {
    return 4u * TAIL_CELLS + t.cell + n;
}

fn tail_b(t: Tail, n: u32) -> u32 {
    return 5u * TAIL_CELLS + t.cell + n;
}

fn tail_r(n: u32) -> u32 {
    return 6u * TAIL_CELLS + n;
}

fn tail_sum(t: Tail, n: u32, c: vec3<i32>) -> f32 {
    let stride = vec3<u32>(1u, u32(t.dims.x), u32(t.dims.x * t.dims.y));
    let hi = min(vec3<u32>(n) + stride, vec3<u32>(t.len - 1u));
    let lo = select(vec3<u32>(n), vec3<u32>(n) - stride, c > vec3<i32>(0));
    let k_hi = vec3<f32>(tail_coef(t, n, 1u), tail_coef(t, n, 2u), tail_coef(t, n, 3u));
    let k_lo = select(vec3<f32>(0.0), vec3<f32>(tail_coef(t, lo.x, 1u), tail_coef(t, lo.y, 2u), tail_coef(t, lo.z, 3u)), c > vec3<i32>(0));
    let x_hi = vec3<f32>(tail_lds[tail_x(t, hi.x)], tail_lds[tail_x(t, hi.y)], tail_lds[tail_x(t, hi.z)]);
    let x_lo = vec3<f32>(tail_lds[tail_x(t, lo.x)], tail_lds[tail_x(t, lo.y)], tail_lds[tail_x(t, lo.z)]);
    var s = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        s = s + k_hi[a] * x_hi[a];
        s = s + k_lo[a] * x_lo[a];
    }

    return s;
}

fn tail_sweeps(t: Tail, lid: u32) {
    for (var sweep = 0u; sweep < 2u * SWEEPS; sweep = sweep + 1u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            tail_smooth(t, (k + sweep / SWEEPS) % 2u, 1u, lid);
        }
    }
}

fn tail_smooth(t: Tail, color: u32, mask: u32, lid: u32) {
    for (var n = lid; n < t.len; n = n + WG) {
        let c = tail_coords(t, n);

        if tail_live(t, n, mask) && u32(c.x + c.y + c.z) % 2u == color {
            tail_lds[tail_x(t, n)] = (tail_lds[tail_b(t, n)] + tail_sum(t, n, c)) / abs(tail_coef(t, n, 0u));
        }
    }

    workgroupBarrier();
}

fn tail_residual(t: Tail, lid: u32) {
    for (var n = lid; n < t.len; n = n + WG) {
        var r = 0.0;

        if tail_live(t, n, 1u) {
            let c = tail_coords(t, n);
            r = tail_lds[tail_b(t, n)] - (abs(tail_coef(t, n, 0u)) * tail_lds[tail_x(t, n)] - tail_sum(t, n, c));
        }

        tail_lds[tail_r(n)] = r;
    }

    workgroupBarrier();
}

fn tail_value(t: Tail, f: vec3<i32>) -> f32 {
    let g = clamp(f, vec3<i32>(0), t.dims - 1);

    return select(0.0, tail_lds[tail_r(tail_index(t, g))], all(f == g));
}

fn tail_row(t: Tail, f: vec3<i32>) -> f32 {
    let x = vec3<i32>(1, 0, 0);

    return 0.25 * (tail_value(t, f) + tail_value(t, f + 3 * x)) + 0.75 * (tail_value(t, f + x) + tail_value(t, f + 2 * x));
}

fn tail_plane(t: Tail, f: vec3<i32>) -> f32 {
    let y = vec3<i32>(0, 1, 0);

    return 0.25 * (tail_row(t, f) + tail_row(t, f + 3 * y)) + 0.75 * (tail_row(t, f + y) + tail_row(t, f + 2 * y));
}

fn tail_restrict(t: Tail, u: Tail, lid: u32) {
    for (var n = lid; n < u.len; n = n + WG) {
        tail_lds[tail_x(u, n)] = 0.0;

        if !tail_live(u, n, 1u) {
            tail_lds[tail_b(u, n)] = 0.0;
            continue;
        }

        let f = tail_coords(u, n) * 2 - vec3<i32>(1);
        let z = vec3<i32>(0, 0, 1);
        let sum = 0.25 * (tail_plane(t, f) + tail_plane(t, f + 3 * z)) + 0.75 * (tail_plane(t, f + z) + tail_plane(t, f + 2 * z));

        tail_lds[tail_b(u, n)] = sum / 8.0;
    }

    workgroupBarrier();
}

fn tail_corner(u: Tail, half: vec3<i32>, dir: vec3<i32>, pick: vec3<i32>) -> f32 {
    let c = half + pick * dir;
    let g = clamp(c, vec3<i32>(0), u.dims - 1);
    let w = select(vec3<f32>(0.75), vec3<f32>(0.25), pick == vec3<i32>(1));

    return select(0.0, w.x * w.y * w.z * tail_lds[tail_x(u, tail_index(u, g))], all(c == g));
}

fn tail_prolong(t: Tail, u: Tail, lid: u32) {
    for (var n = lid; n < t.len; n = n + WG) {
        if !tail_live(t, n, 1u) {
            continue;
        }

        let f = tail_coords(t, n);
        let half = f / 2;
        let dir = (f - half * 2) * 2 - vec3<i32>(1);
        let low = tail_corner(u, half, dir, vec3<i32>(0, 0, 0)) + tail_corner(u, half, dir, vec3<i32>(1, 0, 0)) + tail_corner(u, half, dir, vec3<i32>(0, 1, 0)) + tail_corner(u, half, dir, vec3<i32>(1, 1, 0));
        let high = tail_corner(u, half, dir, vec3<i32>(0, 0, 1)) + tail_corner(u, half, dir, vec3<i32>(1, 0, 1)) + tail_corner(u, half, dir, vec3<i32>(0, 1, 1)) + tail_corner(u, half, dir, vec3<i32>(1, 1, 1));

        tail_lds[tail_x(t, n)] = tail_lds[tail_x(t, n)] + low + high;
    }

    workgroupBarrier();
}

@compute @workgroup_size(256)
fn mg_tail(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    if wid.x != 0u {
        return;
    }

    let lid = lane.index;

    let first = tail_level(0u);

    for (var n = lid; n < 4u * TAIL_CELLS; n = n + WG) {
        tail_lds[n] = l_pack[lv.pack + n];
    }

    for (var n = lid; n < TAIL_CELLS; n = n + WG) {
        tail_lds[4u * TAIL_CELLS + n] = 0.0;
        tail_lds[5u * TAIL_CELLS + n] = select(0.0, l_b[n], n < first.len);
    }

    workgroupBarrier();

    for (var i = 0u; i + 1u < TAIL_COUNT; i = i + 1u) {
        let t = tail_level(i);

        for (var s = 0u; s < SMOOTHING; s = s + 1u) {
            tail_smooth(t, 0u, 1u, lid);
            tail_smooth(t, 1u, 1u, lid);
        }

        for (var s = 0u; s < BAND_SWEEPS; s = s + 1u) {
            tail_smooth(t, 0u, 4u, lid);
            tail_smooth(t, 1u, 4u, lid);
        }

        tail_residual(t, lid);
        tail_restrict(t, tail_level(i + 1u), lid);
    }

    tail_coarse(tail_level(TAIL_COUNT - 1u), lane);

    for (var i = TAIL_COUNT - 1u; i > 0u; i = i - 1u) {
        let t = tail_level(i - 1u);
        tail_prolong(t, tail_level(i), lid);

        for (var s = 0u; s < BAND_SWEEPS; s = s + 1u) {
            tail_smooth(t, 1u, 4u, lid);
            tail_smooth(t, 0u, 4u, lid);
        }

        for (var s = 0u; s < SMOOTHING; s = s + 1u) {
            tail_smooth(t, 1u, 1u, lid);
            tail_smooth(t, 0u, 1u, lid);
        }
    }

    for (var n = lid; n < first.len; n = n + WG) {
        l_x[n] = tail_lds[4u * TAIL_CELLS + n];
    }
}

@compute @workgroup_size(256)
fn mg_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = listed_cell(gid.x);

    if n >= l_len() {
        return;
    }

    if l_active[n] == 0u {
        l_r[n] = 0.0;
        return;
    }

    l_r[n] = l_b[n] - l_apply(n, 0u);
}

struct Near {
    low: vec3<i32>,
    fine: vec3<i32>,
    a: vec4<u32>,
    b: vec4<u32>,
}

fn near_slot(low: vec3<i32>, o: vec3<i32>, dims: vec3<i32>) -> u32 {
    let b = low + o;

    if any(b >= dims) {
        return 0u;
    }

    return brick_list[brick_table() + u32(b.x + dims.x * (b.y + dims.y * b.z))];
}

fn near_bricks(low: vec3<i32>) -> Near {
    var near = Near(low, vec3<i32>(lv.dims.xyz), vec4<u32>(0u), vec4<u32>(0u));

    if level_bricked() {
        let dims = vec3<i32>(brick_dims());
        near.a = vec4<u32>(near_slot(low, vec3<i32>(0, 0, 0), dims), near_slot(low, vec3<i32>(1, 0, 0), dims), near_slot(low, vec3<i32>(0, 1, 0), dims), near_slot(low, vec3<i32>(1, 1, 0), dims));
        near.b = vec4<u32>(near_slot(low, vec3<i32>(0, 0, 1), dims), near_slot(low, vec3<i32>(1, 0, 1), dims), near_slot(low, vec3<i32>(0, 1, 1), dims), near_slot(low, vec3<i32>(1, 1, 1), dims));
    }

    return near;
}

fn near_value(near: Near, f: vec3<i32>) -> f32 {
    let g = clamp(f, vec3<i32>(0), near.fine - 1);
    var m = 0u;

    if level_bricked() {
        let k = (g >> vec3<u32>(3u)) - near.low;
        let local = vec3<u32>(g & vec3<i32>(7));
        let slot = select(near.a[k.x + 2 * k.y], near.b[k.x + 2 * k.y], k.z == 1);
        m = slot * BRICK_CELLS + local.x + BRICK * (local.y + BRICK * local.z);
    } else {
        m = l_index(g);
    }

    return select(0.0, l_r[m], all(f == g));
}

fn crossing(f: vec3<i32>, m: u32) -> f32 {
    let len = l_len();
    var s = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        if (f[a] & 7) == 7 && f[a] + 1 < i32(lv.dims[a]) {
            s = s + l_off[a * len + m] * l_x[l_index(bump(f, a, 1))];
        }

        if (f[a] & 7) == 0 && f[a] > 0 {
            let lo = l_index(bump(f, a, -1));
            s = s + l_off[a * len + lo] * l_x[lo];
        }
    }

    return s;
}

fn near_row(near: Near, f: vec3<i32>) -> f32 {
    let x = vec3<i32>(1, 0, 0);

    return 0.25 * (near_value(near, f) + near_value(near, f + 3 * x)) + 0.75 * (near_value(near, f + x) + near_value(near, f + 2 * x));
}

fn near_plane(near: Near, f: vec3<i32>) -> f32 {
    let y = vec3<i32>(0, 1, 0);

    return 0.25 * (near_row(near, f) + near_row(near, f + 3 * y)) + 0.75 * (near_row(near, f + y) + near_row(near, f + 2 * y));
}

@compute @workgroup_size(256)
fn mg_restrict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = c_listed(gid.x);

    if n >= c_len() {
        return;
    }

    c_x[n] = 0.0;

    let cc = c_coords(n);

    if c_active[n] == 0u || c_skipped(cc) {
        c_b[n] = 0.0;
        return;
    }

    c_b[n] = restricted(cc);
}

fn restricted(cc: vec3<i32>) -> f32 {
    let f = cc * 2 - vec3<i32>(1);
    let z = vec3<i32>(0, 0, 1);
    let near = near_bricks(max(f, vec3<i32>(0)) >> vec3<u32>(3u));
    let sum = 0.25 * (near_plane(near, f) + near_plane(near, f + 3 * z)) + 0.75 * (near_plane(near, f + z) + near_plane(near, f + 2 * z));

    return sum / 8.0;
}

@compute @workgroup_size(256)
fn mg_prolong(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = listed_cell(gid.x);

    if n >= l_len() || l_active[n] == 0u || tall_inside(n) {
        return;
    }

    l_x[n] = l_x[n] + prolonged(l_coords(n));
}

fn corner(half: vec3<i32>, dir: vec3<i32>, pick: vec3<i32>) -> f32 {
    let c = half + pick * dir;
    let g = clamp(c, vec3<i32>(0), vec3<i32>(cv.dims.xyz) - 1);
    let w = select(vec3<f32>(0.75), vec3<f32>(0.25), pick == vec3<i32>(1));

    return select(0.0, w.x * w.y * w.z * c_value(g), all(c == g));
}

fn prolonged(f: vec3<i32>) -> f32 {
    let half = f / 2;
    let dir = (f - half * 2) * 2 - vec3<i32>(1);
    let low = corner(half, dir, vec3<i32>(0, 0, 0)) + corner(half, dir, vec3<i32>(1, 0, 0)) + corner(half, dir, vec3<i32>(0, 1, 0)) + corner(half, dir, vec3<i32>(1, 1, 0));

    return low + corner(half, dir, vec3<i32>(0, 0, 1)) + corner(half, dir, vec3<i32>(1, 0, 1)) + corner(half, dir, vec3<i32>(0, 1, 1)) + corner(half, dir, vec3<i32>(1, 1, 1));
}

fn corrected(n: u32, c: vec3<i32>) -> f32 {
    let len = l_len();
    var s = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        let k_hi = l_off[a * len + n];

        if k_hi != 0.0 {
            s = s + k_hi * (l_x[l_step(n, c, a, 1)] + prolonged(bump(c, a, 1)));
        }

        if c[a] > 0 {
            let m = l_step(n, c, a, -1);
            let k_lo = l_off[a * len + m];

            if k_lo != 0.0 {
                s = s + k_lo * (l_x[m] + prolonged(bump(c, a, -1)));
            }
        }
    }

    return s;
}

@compute @workgroup_size(256)
fn mg_correct(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = listed_cell(gid.x);

    if n >= l_len() || l_active[n] == 0u {
        return;
    }

    let c = l_coords(n);

    if u32(c.x + c.y + c.z) % 2u == 1u {
        l_x[n] = (l_b[n] + corrected(n, c)) / l_diag[n];
    }
}

@compute @workgroup_size(256)
fn mg_presmooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = listed_cell(gid.x);

    if n >= l_len() {
        return;
    }

    if l_active[n] == 0u {
        l_x[n] = 0.0;
        return;
    }

    let c = l_coords(n);
    var s = 0.0;

    if u32(c.x + c.y + c.z) % 2u == 1u {
        s = l_neighbor_sum(n, 2u);
    }

    l_x[n] = (l_b[n] + s) / l_diag[n];
}

override BLOCK_SWEEPS: u32 = 2u;
override UPDATE: bool = false;
override SWAP: bool = false;
override FINISH: bool = false;

const HALO: u32 = 10u;
const HALO_CELLS: u32 = 1000u;

var<workgroup> block_lds: array<f32, HALO_CELLS>;
var<private> block_n: array<u32, 2>;
var<private> block_live: array<bool, 2>;
var<private> block_h: array<u32, 2>;
var<private> block_d: array<f32, 2>;
var<private> block_b: array<f32, 2>;
var<private> block_hi: array<vec3<f32>, 2>;
var<private> block_lo: array<vec3<f32>, 2>;
var<private> block_parity: u32;
var<workgroup> block_alpha: f32;

fn halo_index(p: vec3<i32>) -> u32 {
    let q = vec3<u32>(p + vec3<i32>(1));

    return q.x + HALO * (q.y + HALO * q.z);
}

fn halo_coords(i: u32) -> vec3<i32> {
    return vec3<i32>(vec3<u32>(i % HALO, (i / HALO) % HALO, i / (HALO * HALO))) - vec3<i32>(1);
}

fn block_local(lane: Lane, j: u32) -> vec3<i32> {
    let row = lane.index / 4u;

    return vec3<i32>(vec3<u32>(2u * (lane.index % 4u) + j, row % BRICK, row / BRICK));
}

fn block_at(s: u32, dims: vec3<u32>, members: u32) -> vec4<i32> {
    let blocks = (dims + BRICK - 1u) / BRICK;
    var count = NONE;

    if SPARSE && members != 0u {
        count = brick_list[members];
    }

    if count == NONE {
        let b = vec3<u32>(s % blocks.x, (s / blocks.x) % blocks.y, s / (blocks.x * blocks.y));

        return vec4<i32>(vec3<i32>(b * BRICK), select(0, 1, s < blocks.x * blocks.y * blocks.z));
    }

    if s >= count {
        return vec4<i32>(0);
    }

    return vec4<i32>(unpack_origin(brick_list[members + 1u + s]), 1);
}

fn block_count(dims: vec3<u32>, members: u32) -> u32 {
    let blocks = (dims + BRICK - 1u) / BRICK;

    if SPARSE && members != 0u && brick_list[members] != NONE {
        return brick_list[members];
    }

    return blocks.x * blocks.y * blocks.z;
}

fn l_block(s: u32) -> vec4<i32> {
    if level_bricked() {
        let slot = listed_slot(s * BRICK_CELLS, BRICK_CELLS);

        return vec4<i32>(home_brick * i32(BRICK), select(0, 1, slot != 0u));
    }

    return block_at(s, lv.dims.xyz, lv.bricks);
}

fn l_blocks() -> u32 {
    if level_bricked() {
        return brick_list[0];
    }

    return block_count(lv.dims.xyz, lv.bricks);
}

fn block_cell(block: vec4<i32>, p: vec3<i32>) -> u32 {
    let c = block.xyz + p;

    if block.w == 0 || !l_inside(c) {
        return NONE;
    }

    if level_bricked() {
        return home_slot * BRICK_CELLS + u32(p.x) + BRICK * (u32(p.y) + BRICK * u32(p.z));
    }

    return l_index(c);
}

fn block_clear(lane: Lane) {
    for (var i = lane.index; i < HALO_CELLS; i = i + WG) {
        block_lds[i] = 0.0;
    }
}

fn block_reset(lane: Lane, j: u32) -> vec3<i32> {
    let p = block_local(lane, j);
    block_n[j] = NONE;
    block_live[j] = false;
    block_h[j] = halo_index(p);
    block_d[j] = 1.0;
    block_b[j] = 0.0;
    block_hi[j] = vec3<f32>(0.0);
    block_lo[j] = vec3<f32>(0.0);

    return p;
}

fn block_setup_l(block: vec4<i32>, lane: Lane) {
    let len = l_len();
    let p0 = block_local(lane, 0u);
    block_parity = u32(p0.y + p0.z) % 2u;

    for (var j = 0u; j < 2u; j = j + 1u) {
        let p = block_reset(lane, j);
        let n = block_cell(block, p);
        block_n[j] = n;

        if n == NONE || (l_active[n] & 1u) == 0u || tall_inside(n) {
            continue;
        }

        let c = block.xyz + p;
        block_live[j] = true;
        block_d[j] = l_diag[n];

        for (var a = 0u; a < 3u; a = a + 1u) {
            block_hi[j][a] = l_off[a * len + n];

            if c[a] > 0 {
                block_lo[j][a] = l_off[a * len + l_step(n, c, a, -1)];
            }
        }
    }
}

fn block_setup_c(block: vec4<i32>, lane: Lane) {
    let len = c_len();
    let p0 = block_local(lane, 0u);
    block_parity = u32(p0.y + p0.z) % 2u;

    for (var j = 0u; j < 2u; j = j + 1u) {
        let p = block_reset(lane, j);
        let cc = block.xyz + p;

        if block.w == 0 || any(cc >= vec3<i32>(cv.dims.xyz)) {
            continue;
        }

        let n = c_index(cc);
        block_n[j] = n;

        if (c_active[n] & 1u) == 0u || c_skipped(cc) {
            continue;
        }

        block_live[j] = true;
        block_d[j] = c_diag[n];

        for (var a = 0u; a < 3u; a = a + 1u) {
            block_hi[j][a] = c_off[a * len + n];

            if cc[a] > 0 {
                block_lo[j][a] = c_off[a * len + c_index(bump(cc, a, -1))];
            }
        }
    }
}

fn block_sum(j: u32) -> f32 {
    let h = block_h[j];
    let hi = block_hi[j];
    let lo = block_lo[j];

    return hi.x * block_lds[h + 1u] + lo.x * block_lds[h - 1u] + hi.y * block_lds[h + HALO] + lo.y * block_lds[h - HALO] + hi.z * block_lds[h + HALO * HALO] + lo.z * block_lds[h - HALO * HALO];
}

fn block_sweeps(forward: bool) {
    for (var sweep = 0u; sweep < BLOCK_SWEEPS; sweep = sweep + 1u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let j = (select(1u - k, k, forward) + block_parity) % 2u;

            if block_live[j] {
                block_lds[block_h[j]] = (block_b[j] + block_sum(j)) / block_d[j];
            }

            workgroupBarrier();
        }
    }
}

fn block_value(j: u32) -> f32 {
    return select(0.0, block_lds[block_h[j]], block_live[j]);
}

fn block_residual(j: u32) -> f32 {
    return select(0.0, block_b[j] - block_d[j] * block_lds[block_h[j]] + block_sum(j), block_live[j]);
}

fn block_tile(s: u32, n: u32, j: u32, lane: Lane) -> u32 {
    if level_bricked() {
        return 2u * s + u32(block_local(lane, j).z >= 4);
    }

    return n / WG;
}

@compute @workgroup_size(256)
fn mg_block_down(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    let block = l_block(wid.x);
    block_clear(lane);
    block_setup_l(block, lane);

    if UPDATE {
        block_scalars(lane, wid.x);
    }

    var rn = 0.0;

    for (var j = 0u; j < 2u; j = j + 1u) {
        let n = block_n[j];

        if n == NONE {
            continue;
        }

        var r = l_b[n];

        if UPDATE && !tall_inside(n) {
            var push = 0.0;

            if COUPLED && l_active[n] != 0u && partials[tile_flag(block_tile(wid.x, n, j, lane))] > 0.0 {
                push = body_push(n, true);
            }

            l_z[n] = l_z[n] + block_alpha * select(l_d[n], l_e[n], SWAP);
            r = r - block_alpha * (l_q[n] + push);
            l_b[n] = r;
        }

        rn = max(rn, abs(r) / tall_mass(n));
        block_b[j] = select(0.0, r, block_live[j]);
    }

    if UPDATE {
        let total = group_reduce(lane, rn, true);
        store_partial(lane, R_RN, wid.x, total);
        store_partial(lane, R_RN, wid.x + l_blocks(), 0.0);
    }

    workgroupBarrier();
    block_sweeps(true);

    for (var j = 0u; j < 2u; j = j + 1u) {
        let n = block_n[j];

        if n != NONE {
            l_r[n] = block_residual(j);
            l_x[n] = block_value(j);
            l_q[n] = block_value(j);
        }
    }
}

@compute @workgroup_size(256)
fn mg_block_coarse(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    let block = block_at(wid.x, cv.dims.xyz, cv.bricks);
    block_clear(lane);
    block_setup_c(block, lane);
    block_restrict(lane, block.xyz, block.w != 0);

    for (var j = 0u; j < 2u; j = j + 1u) {
        let n = block_n[j];

        if n == NONE {
            continue;
        }

        if block_live[j] {
            block_b[j] = block_restricted(block_local(lane, j));
        }

        c_b[n] = block_b[j];

        if RESTRICT_ONLY {
            c_x[n] = 0.0;
        }
    }

    if RESTRICT_ONLY {
        return;
    }

    workgroupBarrier();
    block_sweeps(true);

    for (var j = 0u; j < 2u; j = j + 1u) {
        let n = block_n[j];

        if n != NONE {
            c_r[n] = block_residual(j);
            c_x[n] = block_value(j);
            c_z[n] = block_value(j);
        }
    }
}

@compute @workgroup_size(256)
fn mg_block_up(@builtin(workgroup_id) wid: vec3<u32>, lane: Lane) {
    let block = l_block(wid.x);

    for (var i = lane.index; i < HALO_CELLS; i = i + WG) {
        let c = block.xyz + halo_coords(i);
        var v = 0.0;

        if block.w != 0 && l_inside(c) {
            let m = l_index(c);

            if (l_active[m] & 1u) != 0u && !tall_inside(m) {
                v = select(l_z[m], l_q[m], FINISH) + prolonged(c);
            }
        }

        block_lds[i] = v;
    }

    block_setup_l(block, lane);

    for (var j = 0u; j < 2u; j = j + 1u) {
        if block_live[j] {
            block_b[j] = l_b[block_n[j]];
        }
    }

    workgroupBarrier();
    block_sweeps(false);

    var rz = 0.0;

    for (var j = 0u; j < 2u; j = j + 1u) {
        if block_live[j] {
            let x = block_lds[block_h[j]];
            l_x[block_n[j]] = x;
            rz = rz + block_b[j] * x;
        }
    }

    if FINISH {
        let total = group_reduce(lane, rz, false);
        store_partial(lane, R_RZ, wid.x, total);
        let count = l_blocks();

        if lane.index == 0u {
            for (var w = count + wid.x; w < grid_groups(); w = w + count) {
                partials[dot_slot(R_RZ, w)] = 0.0;
            }
        }
    }
}

@compute @workgroup_size(256)
fn mg_block_complete(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = listed_cell(gid.x);

    if n >= l_len() || (l_active[n] & 1u) == 0u {
        return;
    }

    l_r[n] = l_r[n] + crossing(l_coords(n), n);
}

fn block_scalars(lane: Lane, w: u32) {
    let dq_grid = region_total(lane, R_DQ, grid_groups(), false);
    let rz = region_total(lane, R_RZ, grid_groups(), false);

    if COUPLED {
        sum_listed(lane);
    }

    if lane.index == 0u {
        var dq = dq_grid;

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
                    local_response[6u * k + e] = out[e];
                }
            }
        }

        block_alpha = select(0.0, rz / dq, dq != 0.0);

        if w == 0u {
            scalars[S_RZ] = rz;
            scalars[S_DQ] = dq;
            scalars[S_ALPHA] = block_alpha;
        }
    }

    workgroupBarrier();
}

override RESTRICT_ONLY: bool = false;

const FINE_SPAN: u32 = 18u;
const FINE_X: u32 = 2592u;
const FINE_Y: u32 = 1152u;

var<workgroup> fine_x: array<f32, FINE_X>;
var<workgroup> fine_y: array<f32, FINE_Y>;
var<workgroup> fine_slots: array<u32, 64>;

fn fine_value(f: vec3<i32>, base: vec3<i32>) -> f32 {
    if any(f < vec3<i32>(0)) || any(f >= vec3<i32>(lv.dims.xyz)) {
        return 0.0;
    }

    if level_bricked() {
        let k = vec3<u32>((f >> vec3<u32>(3u)) - base);
        let slot = fine_slots[k.x + 4u * (k.y + 4u * k.z)];
        let local = vec3<u32>(f & vec3<i32>(7));

        return select(0.0, l_r[slot * BRICK_CELLS + local.x + BRICK * (local.y + BRICK * local.z)], slot != 0u);
    }

    return l_r[l_index(f)];
}

fn block_restrict(lane: Lane, origin: vec3<i32>, live: bool) {
    let low = origin * 2 - vec3<i32>(1);
    let base = low >> vec3<u32>(3u);

    if level_bricked() && lane.index < 64u {
        let b = base + vec3<i32>(vec3<u32>(lane.index % 4u, (lane.index / 4u) % 4u, lane.index / 16u));
        var slot = 0u;

        if all(b >= vec3<i32>(0)) && all(b < vec3<i32>(brick_dims())) {
            slot = brick_list[brick_table() + brick_of(b * i32(BRICK))];
        }

        fine_slots[lane.index] = slot;
    }

    workgroupBarrier();

    for (var i = lane.index; i < FINE_X; i = i + WG) {
        let yz = i / BRICK;
        let f = low + vec3<i32>(vec3<u32>(2u * (i % BRICK), yz % FINE_SPAN, yz / FINE_SPAN));
        let x = vec3<i32>(1, 0, 0);
        var s = 0.0;

        if live {
            s = 0.25 * (fine_value(f, base) + fine_value(f + 3 * x, base)) + 0.75 * (fine_value(f + x, base) + fine_value(f + 2 * x, base));
        }

        fine_x[i] = s;
    }

    workgroupBarrier();

    for (var i = lane.index; i < FINE_Y; i = i + WG) {
        let at = i % BRICK + BRICK * (2u * ((i / BRICK) % BRICK) + FINE_SPAN * (i / BRICK_ROWS));
        fine_y[i] = 0.25 * (fine_x[at] + fine_x[at + 3u * BRICK]) + 0.75 * (fine_x[at + BRICK] + fine_x[at + 2u * BRICK]);
    }

    workgroupBarrier();
}

fn block_restricted(p: vec3<i32>) -> f32 {
    let at = u32(p.x) + BRICK * u32(p.y) + 2u * BRICK_ROWS * u32(p.z);

    return (0.25 * (fine_y[at] + fine_y[at + 3u * BRICK_ROWS]) + 0.75 * (fine_y[at + BRICK_ROWS] + fine_y[at + 2u * BRICK_ROWS])) / 8.0;
}
