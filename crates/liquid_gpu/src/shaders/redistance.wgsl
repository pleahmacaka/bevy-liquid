override FLIP: u32 = 0u;

const PROJECTION_BAND: f32 = 3.0;
const PROJECTION_BISECTIONS: u32 = 28u;

fn crosses(v: f32, w: f32) -> bool {
    return (w < 0.0) != (v < 0.0);
}

fn interface_distance(n: u32, c: vec3<i32>) -> f32 {
    let v = phi[n];
    var inverse_square = 0.0;
    var crossed = false;
    var touching = false;

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        var nearest = BIG;

        for (var side = -1; side <= 1; side = side + 2) {
            let p = bump(c, axis, side);

            if !in_cells(p) {
                continue;
            }

            let w = phi[cell_index(p)];

            if crosses(v, w) {
                crossed = true;
                nearest = min(nearest, v / (v - w) * params.dx);
            } else if abs(w) < abs(v) {
                nearest = min(nearest, abs(v) / (abs(v) - abs(w)) * params.dx);
            }
        }

        if nearest <= 0.0 {
            touching = true;
        } else if nearest < BIG {
            inverse_square = inverse_square + 1.0 / (nearest * nearest);
        }
    }

    if !crossed {
        return BIG;
    }

    if touching {
        return 0.0;
    }

    return 1.0 / sqrt(inverse_square);
}

fn sweep_read(n: u32) -> f32 {
    if FLIP == 0u {
        return sweep_a[n];
    }

    return sweep_b[n];
}

fn sweep_write(n: u32, v: f32) {
    if FLIP == 0u {
        sweep_b[n] = v;
    } else {
        sweep_a[n] = v;
    }
}

fn eikonal_update(c: vec3<i32>) -> f32 {
    var a = vec3<f32>(BIG);

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        for (var side = -1; side <= 1; side = side + 2) {
            let p = bump(c, axis, side);

            if in_cells(p) {
                a = with_axis(a, axis, min(a[axis], sweep_read(cell_index(p))));
            }
        }
    }

    if a.y < a.x {
        a = a.yxz;
    }

    if a.z < a.y {
        a = a.xzy;
    }

    if a.y < a.x {
        a = a.yxz;
    }

    let h = params.dx;
    let one = a.x + h;

    if one <= a.y {
        return one;
    }

    let two = 0.5 * (a.x + a.y + sqrt(2.0 * h * h - (a.x - a.y) * (a.x - a.y)));

    if two <= a.z {
        return two;
    }

    let s = a.x + a.y + a.z;
    let q = a.x * a.x + a.y * a.y + a.z * a.z;
    let disc = s * s - 3.0 * (q - h * h);

    if disc < 0.0 {
        return BIG;
    }

    return (s + sqrt(disc)) / 3.0;
}

@compute @workgroup_size(256)
fn redistance_seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    let d = interface_distance(n, cell_coords(n));
    sweep_a[n] = d;
    signed_distance[n] = select(0.0, 1.0, d < BIG);
}

@compute @workgroup_size(256)
fn redistance_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    let own = sweep_read(n);

    if signed_distance[n] > 0.5 {
        sweep_write(n, own);
        return;
    }

    sweep_write(n, min(own, eikonal_update(cell_coords(n))));
}

@compute @workgroup_size(256)
fn redistance_finish(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    let d = min(sweep_a[n], params.physics.w);
    sweep_b[n] = select(d, -d, phi[n] < 0.0);
}

fn swept_at(g: vec3<f32>) -> f32 {
    let k = hood_index_corners(g);
    var acc = 0.0;

    for (var i = 0; i < 8; i = i + 1) {
        acc = acc + k.weight[i] * sweep_b[k.index[i]];
    }

    return acc;
}

struct PhiCell {
    base: vec3<i32>,
    near: vec4<f32>,
    far: vec4<f32>,
}

fn corner_phi(h: Hood, b: vec3<i32>, hi: vec3<i32>, pick: vec3<bool>) -> f32 {
    return phi[hood_cell(h, select(b, hi, pick))];
}

fn input_inside(cell: ptr<function, PhiCell>, g_in: vec3<f32>) -> bool {
    let top = max(vec3<f32>(params.dims.xyz) - 1.0, vec3<f32>(0.0));
    let g = clamp(g_in, vec3<f32>(0.0), top);
    let base = min(floor(g), max(top - 1.0, vec3<f32>(0.0)));
    let t = g - base;
    let b = vec3<i32>(base);

    if any(b != (*cell).base) {
        let hi = min(b + vec3<i32>(1), vec3<i32>(top));
        let h = hood_span(b, hi);
        let o = false;
        let l = true;
        (*cell).base = b;
        (*cell).near = vec4<f32>(
            corner_phi(h, b, hi, vec3<bool>(o, o, o)),
            corner_phi(h, b, hi, vec3<bool>(l, o, o)),
            corner_phi(h, b, hi, vec3<bool>(o, l, o)),
            corner_phi(h, b, hi, vec3<bool>(l, l, o)),
        );
        (*cell).far = vec4<f32>(
            corner_phi(h, b, hi, vec3<bool>(o, o, l)),
            corner_phi(h, b, hi, vec3<bool>(l, o, l)),
            corner_phi(h, b, hi, vec3<bool>(o, l, l)),
            corner_phi(h, b, hi, vec3<bool>(l, l, l)),
        );
    }

    var acc = 0.0;

    for (var i = 0; i < 8; i = i + 1) {
        let pick = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1) == vec3<i32>(1);
        let w = select(1.0 - t, t, pick);
        acc = acc + w.x * w.y * w.z * select((*cell).near[i & 3], (*cell).far[i & 3], i >= 4);
    }

    return acc < 0.0;
}

override PROJECT_ADOPT: bool = false;

@compute @workgroup_size(256)
fn redistance_project(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    let d = projected(n);
    signed_distance[n] = d;

    if PROJECT_ADOPT {
        scratch[n] = adopted(n, d);
    }
}

fn projected(n: u32) -> f32 {
    let d = sweep_b[n];

    if abs(d) > PROJECTION_BAND * params.dx {
        return d;
    }

    let start = vec3<f32>(cell_coords(n));
    let ex = vec3<f32>(0.5, 0.0, 0.0);
    let ey = vec3<f32>(0.0, 0.5, 0.0);
    let ez = vec3<f32>(0.0, 0.0, 0.5);
    let g = vec3<f32>(
        swept_at(start + ex) - swept_at(start - ex),
        swept_at(start + ey) - swept_at(start - ey),
        swept_at(start + ez) - swept_at(start - ez),
    );
    let len = length(g);

    if !(len > 0.0) {
        return d;
    }

    let normal = g / len;
    var cell: PhiCell;
    cell.base = vec3<i32>(-1);
    let origin = input_inside(&cell, start);
    let reach = select(-2.0, 2.0, origin) * max(abs(d) * params.inv_dx, 0.5);

    if input_inside(&cell, start + normal * reach) == origin {
        return d;
    }

    var near = 0.0;
    var far = reach;

    for (var i = 0u; i < PROJECTION_BISECTIONS; i = i + 1u) {
        let mid = 0.5 * (near + far);

        if input_inside(&cell, start + normal * mid) == origin {
            near = mid;
        } else {
            far = mid;
        }
    }

    let s = abs(0.5 * (near + far)) * params.dx;

    return select(s, -s, phi[n] < 0.0);
}
