@group(1) @binding(26) var<storage, read_write> l_tall: array<f32>;
@group(2) @binding(26) var<storage, read_write> c_tall: array<f32>;

override TALL: bool = false;
override TALL_BAND: i32 = 4;

const TALL_STRIDE: u32 = 16u;
const TALL_SLOPE: f32 = 4.0;
const TALL_LOW: u32 = 11u;
const TALL_CAP: u32 = 12u;
const TALL_LO: u32 = 13u;
const TALL_BMAX: u32 = 14u;
const TALL_INSIDE: u32 = 8u;
const TALL_END: u32 = 15u;

fn tall_inside(n: u32) -> bool {
    return TALL && lv.tall != 0u && (l_active[n] & TALL_INSIDE) != 0u;
}

fn c_columns() -> u32 {
    return cv.dims.x * cv.dims.z;
}

fn c_column(c: vec3<i32>) -> u32 {
    return u32(c.x) + cv.dims.x * u32(c.z);
}

fn c_top(col: u32) -> i32 {
    if !TALL || cv.tall == 0u {
        return 0;
    }

    return i32(c_tall[col]);
}

fn c_value(c: vec3<i32>) -> f32 {
    if TALL && c.y > 0 {
        let top = c_top(c_column(c));

        if c.y < top {
            let s = f32(c.y) / f32(top);
            return (1.0 - s) * c_x[c_index(vec3<i32>(c.x, 0, c.z))] + s * c_x[c_index(vec3<i32>(c.x, top, c.z))];
        }
    }

    return c_x[c_index(c)];
}

fn c_skipped(c: vec3<i32>) -> bool {
    if !TALL || cv.tall == 0u || c.y == 0 {
        return false;
    }

    let col = c_column(c);
    let top = c_top(col);

    return top >= 2 && c.y < top && c.y < i32(c_tall[c_columns() + col * TALL_STRIDE + TALL_LO]);
}

@compute @workgroup_size(256)
fn tall_coarse(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= c_columns() {
        return;
    }

    let c = vec2<i32>(i32(col % cv.dims.x), i32(col / cv.dims.x));
    let fine = vec2<i32>(i32(lv.dims.x), i32(lv.dims.z));
    var lowest = 2147483647;
    var near = 2147483647;

    for (var fz = max(2 * c.y - 1, 0); fz <= min(2 * c.y + 2, fine.y - 1); fz = fz + 1) {
        for (var fx = max(2 * c.x - 1, 0); fx <= min(2 * c.x + 2, fine.x - 1); fx = fx + 1) {
            let top = l_top(u32(fx) + lv.dims.x * u32(fz));
            near = min(near, top);

            if fx >> 1u == c.x && fz >> 1u == c.y {
                lowest = min(lowest, top);
            }
        }
    }

    let top = lowest / 2;
    c_tall[col] = select(0.0, f32(top), top >= 2);
    c_tall[c_columns() + col * TALL_STRIDE + TALL_LO] = f32(max(1, (near - 2) / 2));
}

@compute @workgroup_size(256)
fn tall_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() || state.converged == 1u || state.running == 0u || l_top(col) < 2 {
        return;
    }

    let base = tall_base(col);
    let top = l_top(col);
    let bottom = tall_at(base, 0);
    var sum = vec2<f32>(l_b[bottom], 0.0);

    for (var j = max(i32(l_tall[tall_slot(col) + TALL_LO]), 1); j <= top; j = j + 1) {
        let n = tall_at(base, j);
        let s = f32(j) / f32(top);
        sum = sum + l_b[n] * vec2<f32>(1.0 - s, s);
        l_b[n] = 0.0;
    }

    l_b[bottom] = sum.x;
    l_b[tall_at(base, top)] = sum.y;
}

fn l_top(col: u32) -> i32 {
    if !TALL || lv.tall == 0u {
        return 0;
    }

    return i32(l_tall[col]);
}

fn tall_slot(col: u32) -> u32 {
    return l_columns() + col * TALL_STRIDE;
}

fn tall_kind(c: vec3<i32>) -> u32 {
    if !TALL {
        return 0u;
    }

    let top = l_top(l_column(c));

    if top < 2 || c.y > top {
        return 0u;
    }

    if c.y == 0 {
        return 2u;
    }

    if c.y == top {
        return 3u;
    }

    return 1u;
}

fn tall_base(col: u32) -> vec3<i32> {
    return vec3<i32>(i32(col % lv.dims.x), 0, i32(col / lv.dims.x));
}

fn tall_at(base: vec3<i32>, j: i32) -> u32 {
    return l_index(vec3<i32>(base.x, j, base.z));
}

fn lateral_axis(dir: u32) -> u32 {
    return select(2u, 0u, dir % 2u == 0u);
}

fn lateral_sign(dir: u32) -> i32 {
    return select(-1, 1, dir < 2u);
}

fn l_vector(which: u32, i: u32) -> f32 {
    if which == 0u {
        return l_x[i];
    }

    if which == 2u {
        return l_b[i] / l_diag[i];
    }

    return direction(i);
}

fn l_value(which: u32, m: u32, p: vec3<i32>) -> f32 {
    if TALL && p.y > 0 {
        let top = l_top(l_column(p));

        if p.y < top {
            let s = f32(p.y) / f32(top);
            return (1.0 - s) * l_vector(which, tall_at(p, 0)) + s * l_vector(which, tall_at(p, top));
        }
    }

    return l_vector(which, m);
}

fn tall_mass(n: u32) -> f32 {
    if !TALL {
        return 1.0;
    }

    let c = l_coords(n);

    if tall_kind(c) < 2u {
        return 1.0;
    }

    return 0.5 * (f32(l_top(l_column(c))) + 1.0);
}

fn tall_theta(n: u32, m: u32) -> f32 {
    return max(l_phi[n] / (l_phi[n] - l_phi[m]), THETA_MIN);
}

struct Side {
    inside: bool,
    col: u32,
    top: i32,
    lo: f32,
    hi: f32,
}

fn tall_side(base: vec3<i32>, dir: u32, which: u32) -> Side {
    let p = bump(base, lateral_axis(dir), lateral_sign(dir));
    let inside = l_inside(p);
    let q = select(base, p, inside);
    let col = l_column(q);
    let top = select(0, l_top(col), inside);

    return Side(inside, col, top, l_vector(which, tall_at(q, 0)), l_vector(which, tall_at(q, max(top, 0))));
}

fn tall_cross(col: u32, side: Side, dir: u32) -> vec2<f32> {
    if !side.inside || side.top < 2 {
        return vec2<f32>(0.0);
    }

    var w: vec4<f32>;

    if dir < 2u {
        let o = tall_slot(col) + 3u + 4u * dir;
        w = vec4<f32>(l_tall[o], l_tall[o + 1u], l_tall[o + 2u], l_tall[o + 3u]);
    } else {
        let o = tall_slot(side.col) + 3u + 4u * (dir - 2u);
        w = vec4<f32>(l_tall[o], l_tall[o + 2u], l_tall[o + 1u], l_tall[o + 3u]);
    }

    return vec2<f32>(w.x * side.lo + w.y * side.hi, w.z * side.lo + w.w * side.hi);
}

fn tall_lateral(base: vec3<i32>, top: i32, side: Side, dir: u32, which: u32) -> vec2<f32> {
    var s = vec2<f32>(0.0);

    if !side.inside {
        return s;
    }

    let axis = lateral_axis(dir);
    let toward = lateral_sign(dir);
    let len = l_len();

    for (var j = select(0, side.top + 1, side.top >= 2); j <= top; j = j + 1) {
        let n = tall_at(base, j);
        let p = bump(vec3<i32>(base.x, j, base.z), axis, toward);
        let m = l_index(p);
        let k = select(l_off[axis * len + m], l_off[axis * len + n], toward > 0);
        let sj = f32(j) / f32(top);
        s = s + k * l_vector(which, m) * vec2<f32>(1.0 - sj, sj);
    }

    return s;
}

fn tall_coupling(col: u32, which: u32, vertical: bool) -> vec2<f32> {
    let base = tall_base(col);
    let top = l_top(col);
    let east = tall_side(base, 0u, which);
    let north = tall_side(base, 1u, which);
    let west = tall_side(base, 2u, which);
    let south = tall_side(base, 3u, which);
    var s = vec2<f32>(0.0);

    if vertical && top + 1 < i32(lv.dims.y) {
        s.y = l_off[l_len() + tall_at(base, top)] * l_vector(which, tall_at(base, top + 1));
    }

    s = s + tall_cross(col, east, 0u) + tall_cross(col, north, 1u) + tall_cross(col, west, 2u) + tall_cross(col, south, 3u);
    s = s + tall_lateral(base, top, east, 0u, which) + tall_lateral(base, top, north, 1u, which);

    return s + tall_lateral(base, top, west, 2u, which) + tall_lateral(base, top, south, 3u, which);
}

fn tall_row(col: u32, which: u32) -> vec2<f32> {
    let base = tall_base(col);
    let top = l_top(col);
    let t = tall_slot(col);
    let a = l_vector(which, tall_at(base, 0));
    let b = l_vector(which, tall_at(base, top));
    let s = tall_coupling(col, which, true);

    return vec2<f32>(l_tall[t] * a + l_tall[t + 2u] * b - s.x, l_tall[t + 2u] * a + l_tall[t + 1u] * b - s.y);
}

fn l_apply(n: u32, which: u32) -> f32 {
    if tall_inside(n) {
        return 0.0;
    }

    let c = l_coords(n);
    let kind = tall_kind(c);

    if kind == 1u {
        return 0.0;
    }

    if kind >= 2u {
        let row = tall_row(l_column(c), which);
        return select(row.x, row.y, kind == 3u);
    }

    return l_diag[n] * l_vector(which, n) - l_neighbor_sum(n, which);
}

fn line_side(n: u32, c: vec3<i32>, side: Side, dir: u32) -> f32 {
    if !side.inside {
        return 0.0;
    }

    let axis = lateral_axis(dir);
    let toward = lateral_sign(dir);
    let len = l_len();
    let m = l_step(n, c, axis, toward);
    let k = select(l_off[axis * len + m], l_off[axis * len + n], toward > 0);

    if c.y > 0 && c.y < side.top {
        let s = f32(c.y) / f32(side.top);
        return k * ((1.0 - s) * side.lo + s * side.hi);
    }

    return k * l_x[m];
}

override LINE_ROWS: u32 = 258u;

const LINE_SPAN: u32 = 32768u;
const LINE_LANES: u32 = 8u;
const LINE_GROUPS: u32 = 8u;

var<workgroup> line_diag: array<f32, LINE_ROWS>;
var<workgroup> line_lower: array<f32, LINE_ROWS>;
var<workgroup> line_rhs: array<f32, LINE_ROWS>;
var<workgroup> line_live: u32;
var<workgroup> line_total: u32;
var<workgroup> line_rows: array<i32, LINE_GROUPS>;
var<workgroup> line_tall: array<vec2<f32>, 64>;

fn line_column(t: u32) -> u32 {
    let half = (lv.dims.x + 1u) / 2u;
    let z = t / half;
    let x = 2u * (t % half) + (z + COLOR) % 2u;

    if z >= lv.dims.z || x >= lv.dims.x {
        return NONE;
    }

    return x + lv.dims.x * z;
}

fn line_height(k: i32, top: i32) -> i32 {
    if top < 2 {
        return k;
    }

    return select(top + k - 1, top * k, k < 2);
}

@compute @workgroup_size(64)
fn mg_line(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    if lane == 0u {
        line_live = select(0u, 1u, state.converged == 0u && state.running == 1u);
    }

    if workgroupUniformLoad(&line_live) == 0u {
        return;
    }

    let group = lane / LINE_LANES;
    let sub = i32(lane % LINE_LANES);
    let col = line_column((wid.x + LINE_SPAN * wid.y) * LINE_GROUPS + group);
    var rows = 0;
    var top = 0;
    var base = vec3<i32>(0);

    if col != NONE {
        base = tall_base(col);
        top = l_top(col);
        let end = i32(l_tall[tall_slot(col) + TALL_END]);
        rows = select(end + 1, max(end, top) - top + 2, top >= 2);
    }

    if sub == 0 {
        line_rows[group] = rows;
    }

    workgroupBarrier();

    var start = 0;
    var total = 0;

    for (var g = 0u; g < LINE_GROUPS; g = g + 1u) {
        start = start + select(0, line_rows[g], g < group);
        total = total + line_rows[g];
    }

    if lane == 0u {
        line_total = u32(total);
    }

    let rounds = select(LINE_GROUPS, 1u, workgroupUniformLoad(&line_total) <= LINE_ROWS);
    let len = l_len();
    let east = tall_side(base, 0u, 0u);
    let north = tall_side(base, 1u, 0u);
    let west = tall_side(base, 2u, 0u);
    let south = tall_side(base, 3u, 0u);

    for (var round = 0u; round < rounds; round = round + 1u) {
        let mine = rounds == 1u || group == round;
        let o = select(0, start, rounds == 1u);
        let count = select(0, rows, mine);

        for (var k = sub; k < count; k = k + i32(LINE_LANES)) {
            if top >= 2 && k < 2 {
                if k == 0 {
                    let t = tall_slot(col);
                    line_diag[o] = l_tall[t];
                    line_diag[o + 1] = l_tall[t + 1u];
                    line_lower[o] = 0.0;
                    line_lower[o + 1] = l_tall[t + 2u];
                    line_rhs[o] = l_b[tall_at(base, 0)];
                    line_rhs[o + 1] = l_b[tall_at(base, top)];
                }

                continue;
            }

            let j = line_height(k, top);
            let c = vec3<i32>(base.x, j, base.z);
            let n = tall_at(base, j);
            let live = l_active[n] != 0u;
            let lower = select(0.0, -l_off[len + tall_at(base, max(j - 1, 0))], j > 0);
            let lateral = line_side(n, c, east, 0u) + line_side(n, c, west, 2u) + line_side(n, c, north, 1u) + line_side(n, c, south, 3u);

            line_diag[o + k] = select(1.0, l_diag[n], live);
            line_lower[o + k] = select(0.0, lower, live);
            line_rhs[o + k] = select(0.0, l_b[n] + lateral, live);
        }

        if mine && top >= 2 && sub < 4 {
            var side = east;

            if sub == 1 {
                side = north;
            } else if sub == 2 {
                side = west;
            } else if sub == 3 {
                side = south;
            }

            line_tall[lane] = tall_cross(col, side, u32(sub)) + tall_lateral(base, top, side, u32(sub), 0u);
        }

        workgroupBarrier();

        if sub == 0 && count > 0 {
            if top >= 2 {
                let s = line_tall[lane] + line_tall[lane + 1u] + line_tall[lane + 2u] + line_tall[lane + 3u];
                line_rhs[o] = line_rhs[o] + s.x;
                line_rhs[o + 1] = line_rhs[o + 1] + s.y;
            }

            var cp = 0.0;
            var dp = 0.0;

            for (var k = 0; k < count; k = k + 1) {
                let lower = line_lower[o + k];
                let m = line_diag[o + k] - lower * cp;
                var upper = 0.0;

                if k + 1 < count {
                    upper = line_lower[o + k + 1];
                }

                cp = upper / m;
                dp = (line_rhs[o + k] - lower * dp) / m;
                line_diag[o + k] = cp;
                line_rhs[o + k] = dp;
            }

            var next = 0.0;

            for (var k = count - 1; k >= 0; k = k - 1) {
                next = line_rhs[o + k] - line_diag[o + k] * next;
                line_rhs[o + k] = next;
            }
        }

        workgroupBarrier();

        for (var k = sub; k < count; k = k + i32(LINE_LANES)) {
            let n = tall_at(base, line_height(k, top));

            if l_active[n] != 0u {
                l_x[n] = line_rhs[o + k];
            }
        }

        workgroupBarrier();
    }
}

fn tall_plain(c: vec3<i32>) -> bool {
    let n = l_index(c);

    if l_active[n] == 0u || l_phi[n] >= 0.0 {
        return false;
    }

    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var h = 0; h < 2; h = h + 1) {
            let f = bump(c, a, h);

            if f[a] != 0 && f[a] != i32(lv.dims[a]) && weights[face_index(a, f)] < 1.0 {
                return false;
            }
        }
    }

    return true;
}

@compute @workgroup_size(256)
fn tall_low(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() {
        return;
    }

    let base = tall_base(col);
    let rows = i32(lv.dims.y);
    var low = rows;

    for (var j = 0; j < rows; j = j + 1) {
        if !tall_plain(vec3<i32>(base.x, j, base.z)) {
            low = j;
            break;
        }
    }

    l_tall[tall_slot(col) + TALL_LOW] = f32(low);
}

@compute @workgroup_size(256)
fn tall_cap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() {
        return;
    }

    let base = tall_base(col);
    var low = l_tall[tall_slot(col) + TALL_LOW];

    for (var dir = 0u; dir < 4u; dir = dir + 1u) {
        let side = bump(base, lateral_axis(dir), lateral_sign(dir));

        if l_inside(side) {
            low = min(low, l_tall[tall_slot(l_column(side)) + TALL_LOW]);
        }
    }

    l_tall[tall_slot(col) + TALL_CAP] = low - 1.0 - f32(TALL_BAND);
}

fn tall_relax(n: u32, m: u32) {
    let p = tall_slot(n) + TALL_CAP;
    l_tall[p] = min(l_tall[p], l_tall[tall_slot(m) + TALL_CAP] + TALL_SLOPE);
}

@compute @workgroup_size(256)
fn tall_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let k = gid.x;
    let nx = lv.dims.x;

    if k >= lv.dims.z {
        return;
    }

    for (var i = 1u; i < nx; i = i + 1u) {
        tall_relax(i + nx * k, i - 1u + nx * k);
    }

    for (var i = nx - 1u; i > 0u; i = i - 1u) {
        tall_relax(i - 1u + nx * k, i + nx * k);
    }
}

@compute @workgroup_size(256)
fn tall_lines(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let nx = lv.dims.x;
    let nz = lv.dims.z;

    if i >= nx {
        return;
    }

    for (var k = 1u; k < nz; k = k + 1u) {
        tall_relax(i + nx * k, i + nx * (k - 1u));
    }

    for (var k = nz - 1u; k > 0u; k = k - 1u) {
        tall_relax(i + nx * (k - 1u), i + nx * k);
    }

    for (var k = 0u; k < nz; k = k + 1u) {
        let cap = l_tall[tall_slot(i + nx * k) + TALL_CAP];
        l_tall[i + nx * k] = select(0.0, cap, cap >= 2.0);
    }
}

@compute @workgroup_size(256)
fn tall_assemble(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() {
        return;
    }

    let t = tall_slot(col);
    let top = l_top(col);

    for (var i = 0u; i < 11u; i = i + 1u) {
        l_tall[t + i] = 0.0;
    }

    let base = tall_base(col);
    var end = -1;

    for (var j = i32(lv.dims.y) - 1; j >= 0; j = j - 1) {
        if l_active[tall_at(base, j)] != 0u {
            end = j;
            break;
        }
    }

    l_tall[t + TALL_END] = f32(end);

    if top < 2 {
        return;
    }

    let len = l_len();
    let height = f32(top);
    var vertical = 0.0;

    for (var j = 0; j < top; j = j + 1) {
        vertical = vertical + l_coef[len + tall_at(base, j)];
    }

    for (var j = 1; j < top; j = j + 1) {
        let n = tall_at(base, j);
        l_active[n] = l_active[n] | TALL_INSIDE;
    }

    vertical = vertical / (height * height);

    var aa = vertical + l_floor(base);
    var bb = vertical;
    var ab = -vertical;
    var cross = array<vec4<f32>, 2>(vec4<f32>(0.0), vec4<f32>(0.0));
    let upper = tall_at(base, top);
    let k_top = l_coef[len + upper];

    if top + 1 < i32(lv.dims.y) && k_top > 0.0 {
        let m = tall_at(base, top + 1);

        if l_active[m] != 0u {
            bb = bb + k_top;
        } else if l_phi[m] >= 0.0 {
            bb = bb + k_top / tall_theta(upper, m);
        }
    }

    for (var dir = 0u; dir < 4u; dir = dir + 1u) {
        let axis = lateral_axis(dir);
        let toward = lateral_sign(dir);
        let side = bump(base, axis, toward);

        if !l_inside(side) {
            continue;
        }

        let other = l_top(l_column(side));

        for (var j = 0; j <= top; j = j + 1) {
            let n = tall_at(base, j);
            let m = tall_at(side, j);
            let k = select(l_coef[axis * len + m], l_coef[axis * len + n], toward > 0);
            let live = l_active[m] != 0u;

            if k <= 0.0 || (!live && l_phi[m] < 0.0) {
                continue;
            }

            let g = select(k / tall_theta(n, m), k, live);
            let s = f32(j) / height;
            let hat = vec2<f32>(1.0 - s, s);

            aa = aa + g * hat.x * hat.x;
            bb = bb + g * hat.y * hat.y;
            ab = ab + g * hat.x * hat.y;

            if live && dir < 2u && other >= 2 && j <= other {
                let near = vec2<f32>(1.0 - f32(j) / f32(other), f32(j) / f32(other));
                cross[dir] = cross[dir] + k * vec4<f32>(hat.x * near.x, hat.x * near.y, hat.y * near.x, hat.y * near.y);
            }
        }
    }

    let share = lv.regularization / (1.0 + lv.regularization);

    for (var j = 0; j <= top; j = j + 1) {
        let d = l_diag[tall_at(base, j)] * share;
        let s = f32(j) / height;

        aa = aa + d * (1.0 - s) * (1.0 - s);
        bb = bb + d * s * s;
        ab = ab + d * (1.0 - s) * s;
    }

    l_tall[t] = aa;
    l_tall[t + 1u] = bb;
    l_tall[t + 2u] = ab;

    for (var i = 0u; i < 4u; i = i + 1u) {
        l_tall[t + 3u + i] = cross[0][i];
        l_tall[t + 7u + i] = cross[1][i];
    }
}

@compute @workgroup_size(256)
fn tall_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() || l_top(col) < 2 {
        return;
    }

    let base = tall_base(col);
    let top = l_top(col);
    var sum = vec2<f32>(0.0);
    var largest = 0.0;

    for (var j = 0; j <= top; j = j + 1) {
        let n = tall_at(base, j);
        let s = f32(j) / f32(top);
        sum = sum + l_b[n] * vec2<f32>(1.0 - s, s);
        largest = max(largest, abs(l_b[n]));
        l_b[n] = 0.0;
    }

    l_b[tall_at(base, 0)] = sum.x;
    l_b[tall_at(base, top)] = sum.y;
    l_tall[tall_slot(col) + TALL_BMAX] = largest;
}

fn tall_rhs_norm(n: u32) -> f32 {
    if TALL {
        let c = l_coords(n);

        if tall_kind(c) >= 2u {
            return l_tall[tall_slot(l_column(c)) + TALL_BMAX];
        }
    }

    return abs(l_b[n]);
}

@compute @workgroup_size(256)
fn tall_expand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;

    if col >= l_columns() || l_top(col) < 2 {
        return;
    }

    let base = tall_base(col);
    let top = l_top(col);
    let a = l_x[tall_at(base, 0)];
    let b = l_x[tall_at(base, top)];

    for (var j = 1; j < top; j = j + 1) {
        let s = f32(j) / f32(top);
        l_x[tall_at(base, j)] = (1.0 - s) * a + s * b;
    }
}

@compute @workgroup_size(256)
fn tall_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x < l_columns() {
        l_tall[gid.x] = 0.0;
    }
}
