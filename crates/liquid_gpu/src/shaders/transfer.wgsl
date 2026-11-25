struct Gathered {
    mass: vec3<f32>,
    mom: vec3<f32>,
}

const P2G_ROWS: u32 = 16u;

var<workgroup> p2g_blocks: u32;
var<workgroup> p2g_mass: array<vec3<f32>, 256>;
var<workgroup> p2g_mom: array<vec3<f32>, 256>;

fn p2g_add(sum: ptr<function, Gathered>, p: Particle, node: vec3<f32>) {
    let x = p.x.xyz;
    let g = x * params.inv_dx;
    let full = g - node;
    let half = (g - 0.5) - node;
    let a = vec3<f32>(bspline2(full.x), bspline2(full.y), bspline2(full.z));
    let b = vec3<f32>(bspline2(half.x), bspline2(half.y), bspline2(half.z));
    let w = vec3<f32>(a.x * b.y * b.z, b.x * a.y * b.z, b.x * b.y * a.z);
    let m = vec3<f32>(
        p.v.x + dot(p.c0.xyz, (node + face_offset(0u)) * params.dx - x),
        p.v.y + dot(p.c1.xyz, (node + face_offset(1u)) * params.dx - x),
        p.v.z + dot(p.c2.xyz, (node + face_offset(2u)) * params.dx - x),
    );

    (*sum).mass = (*sum).mass + w;
    (*sum).mom = (*sum).mom + w * m;
}

fn p2g_row(f: vec3<i32>, cy: i32, cz: i32) -> Gathered {
    var sum: Gathered;
    sum.mass = vec3<f32>(0.0);
    sum.mom = vec3<f32>(0.0);

    if cy < 0 || cz < 0 || cy >= i32(params.dims.y) || cz >= i32(params.dims.z) {
        return sum;
    }

    let node = vec3<f32>(f);
    let lo = max(f.x - 2, 0);
    let hi = min(f.x + 1, i32(params.dims.x) - 1);
    let row = grid_index(vec3<i32>(lo, cy, cz));
    let begin = cell_start[row];
    let end = cell_start[row + u32(hi - lo) + 1u];

    if begin >= end {
        return sum;
    }

    var p = p2g_particle(begin);

    for (var i = begin + 1u; i < end; i = i + 1u) {
        let next = p2g_particle(i);
        p2g_add(&sum, p, node);
        p = next;
    }

    p2g_add(&sum, p, node);

    return sum;
}

fn p2g_particle(i: u32) -> Particle {
    if QUANTIZED {
        return unpack_particle(i);
    }

    return particles[i];
}

fn node_items() -> u32 {
    if SPARSE {
        return brick_list[0] * BRICK_NODES;
    }

    return node_count();
}

@compute @workgroup_size(256)
fn p2g(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    if lid == 0u {
        p2g_blocks = select(0u, (node_items() + P2G_ROWS - 1u) / P2G_ROWS, state.running != 0u);
    }

    let blocks = workgroupUniformLoad(&p2g_blocks);
    let r = i32(lid / P2G_ROWS);

    for (var block = wid.x; block < blocks; block = block + groups.x) {
        let n = brick_node(block * P2G_ROWS + lid % P2G_ROWS);
        let live = n < node_count();
        var sum: Gathered;
        sum.mass = vec3<f32>(0.0);
        sum.mom = vec3<f32>(0.0);
        var f = vec3<i32>(0);

        if live {
            f = node_coords(n);
            sum = p2g_row(f, f.y - 2 + r % 4, f.z - 2 + r / 4);
        }

        p2g_mass[lid] = sum.mass;
        p2g_mom[lid] = sum.mom;
        workgroupBarrier();

        if live && r == 0 {
            for (var k = 1u; k < P2G_ROWS; k = k + 1u) {
                sum.mass = sum.mass + p2g_mass[lid + P2G_ROWS * k];
                sum.mom = sum.mom + p2g_mom[lid + P2G_ROWS * k];
            }

            for (var axis = 0u; axis < 3u; axis = axis + 1u) {
                if in_faces(axis, f) {
                    let m = face_index(axis, f);
                    let mass = sum.mass[axis];
                    valid[m] = mass;
                    velocity[m] = select(0.0, sum.mom[axis] / mass, mass > 0.0);
                }
            }
        }

        workgroupBarrier();
    }
}

fn home_at(x: vec3<f32>) {
    if SPARSE {
        let c = cell_of(x);
        set_home(brick_list[brick_table() + brick_of(c)], c & vec3<i32>(~7));
    }
}

fn inverse3(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0];
    let b = m[1];
    let c = m[2];
    let r0 = cross(b, c);
    let r1 = cross(c, a);
    let r2 = cross(a, b);
    let det = dot(a, r0);

    return transpose(mat3x3<f32>(r0, r1, r2)) * (1.0 / det);
}

struct AxisSample {
    v: f32,
    c: vec3<f32>,
}

struct Hood {
    low: vec3<i32>,
    near: vec4<u32>,
    far: vec4<u32>,
}

fn brick_slot(b: vec3<i32>) -> u32 {
    let o = b - home_brick;

    if home_slot != 0u && all(o == vec3<i32>(0)) {
        return home_slot;
    }

    if home_slot != 0u && all(o >= vec3<i32>(-1)) && all(o <= vec3<i32>(1)) {
        return brick_list[brick_neighbors() + 27u * home_slot + u32(13 + o.x + 3 * o.y + 9 * o.z)];
    }

    let dims = vec3<i32>(brick_dims());

    return brick_list[brick_table() + u32(b.x + dims.x * (b.y + dims.y * b.z))];
}

fn hood_pick(low: vec3<i32>, cross: vec3<i32>, bits: vec3<i32>) -> u32 {
    if all(bits <= cross) {
        return brick_slot(low + bits);
    }

    return 0u;
}

fn hood_at(low: vec3<i32>, high: vec3<i32>) -> Hood {
    var h: Hood;
    h.low = low;
    h.near = vec4<u32>(0u);
    h.far = vec4<u32>(0u);

    if SPARSE {
        let c = vec3<i32>(high > low);
        h.near = vec4<u32>(hood_pick(low, c, vec3<i32>(0, 0, 0)), hood_pick(low, c, vec3<i32>(1, 0, 0)), hood_pick(low, c, vec3<i32>(0, 1, 0)), hood_pick(low, c, vec3<i32>(1, 1, 0)));
        h.far = vec4<u32>(hood_pick(low, c, vec3<i32>(0, 0, 1)), hood_pick(low, c, vec3<i32>(1, 0, 1)), hood_pick(low, c, vec3<i32>(0, 1, 1)), hood_pick(low, c, vec3<i32>(1, 1, 1)));
    }

    return h;
}

fn hood_span(low: vec3<i32>, high: vec3<i32>) -> Hood {
    let last = vec3<i32>(params.dims.xyz) - 1;

    return hood_at(clamp(low, vec3<i32>(0), last) >> vec3<u32>(3u), clamp(high, vec3<i32>(0), last) >> vec3<u32>(3u));
}

fn hood_slot(h: Hood, owner: vec3<i32>) -> u32 {
    let up = (owner >> vec3<u32>(3u)) != h.low;
    let y0 = select(select(h.near.x, h.near.y, up.x), select(h.near.z, h.near.w, up.x), up.y);
    let y1 = select(select(h.far.x, h.far.y, up.x), select(h.far.z, h.far.w, up.x), up.y);

    return select(y0, y1, up.z);
}

fn hood_face(h: Hood, axis: u32, f: vec3<i32>) -> u32 {
    if !SPARSE {
        return face_index(axis, f);
    }

    let owner = min(f, vec3<i32>(params.dims.xyz) - 1);
    let local = vec3<u32>(f - (owner & vec3<i32>(~7)));
    let size = face_box(axis);

    return hood_slot(h, owner) * BRICK_FACES + axis * BRICK_FACE_BOX + local.x + size.x * (local.y + size.y * local.z);
}

fn hood_cell(h: Hood, c: vec3<i32>) -> u32 {
    if !SPARSE {
        return grid_index(c);
    }

    let local = vec3<u32>(c & vec3<i32>(7));

    return hood_slot(h, c) * BRICK_CELLS + local.x + BRICK * (local.y + BRICK * local.z);
}

fn velocity_hood(x: vec3<f32>) -> Hood {
    let top = max(vec3<f32>(params.dims.xyz) - 1.0, vec3<f32>(0.0));
    let g = clamp(x * params.inv_dx - 0.5, vec3<f32>(0.0), top);
    let base = vec3<i32>(min(floor(g), max(top - 1.0, vec3<f32>(0.0))));

    return hood_span(base, base + 2);
}

fn hood_corners(h: Hood, g_in: vec3<f32>, dims: vec3<u32>, axis: u32) -> Corners {
    let top = max(vec3<f32>(dims) - 1.0, vec3<f32>(0.0));
    let g = clamp(g_in, vec3<f32>(0.0), top);
    let base = min(floor(g), max(top - 1.0, vec3<f32>(0.0)));
    let t = g - base;
    let b = vec3<i32>(base);
    let hi = min(b + vec3<i32>(1), vec3<i32>(top));
    var out: Corners;

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let pick = vec3<i32>(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1) == vec3<i32>(1);
        let p = select(b, hi, pick);
        let w = select(1.0 - t, t, pick);

        if axis < 3u {
            out.index[corner] = hood_face(h, axis, p);
        } else {
            out.index[corner] = hood_cell(h, p);
        }

        out.weight[corner] = w.x * w.y * w.z;
    }

    return out;
}

fn hood_face_corners(axis: u32, x: vec3<f32>) -> Corners {
    return hood_corners(velocity_hood(x), x * params.inv_dx - face_offset(axis), face_dims(axis), axis);
}

fn cell_base(g_in: vec3<f32>) -> vec3<i32> {
    let top = max(vec3<f32>(params.dims.xyz) - 1.0, vec3<f32>(0.0));
    let g = clamp(g_in, vec3<f32>(0.0), top);

    return vec3<i32>(min(floor(g), max(top - 1.0, vec3<f32>(0.0))));
}

fn hood_index_corners(g: vec3<f32>) -> Corners {
    let base = cell_base(g);

    return hood_corners(hood_span(base, base + 1), g, params.dims.xyz, 3u);
}

struct Lattice {
    inside: vec3<bool>,
    weight: vec3<f32>,
    offset: vec3<f32>,
    owner: vec3<i32>,
    local: vec3<u32>,
}

fn lattice_axis(g: f32, x: f32, first: i32, offset: f32, faces: u32, cells: u32) -> Lattice {
    let p = vec3<i32>(first) + vec3<i32>(0, 1, 2);
    let owner = min(p, vec3<i32>(i32(cells) - 1));
    var out: Lattice;
    out.inside = p >= vec3<i32>(0) & p < vec3<i32>(i32(faces));
    out.weight = vec3<f32>(bspline2(g - f32(p.x)), bspline2(g - f32(p.y)), bspline2(g - f32(p.z)));
    out.offset = (vec3<f32>(p) + offset) * params.dx - x;
    out.owner = owner;
    out.local = vec3<u32>(select(p, p - (owner & vec3<i32>(~7)), SPARSE));

    return out;
}

fn lattice_index(h: Hood, lx: Lattice, ly: Lattice, lz: Lattice, ox: i32, oy: i32, oz: i32, first: u32, size: vec3<u32>) -> u32 {
    let at = first + lx.local[ox] + size.x * (ly.local[oy] + size.y * lz.local[oz]);

    if SPARSE {
        return at + hood_slot(h, vec3<i32>(lx.owner[ox], ly.owner[oy], lz.owner[oz])) * BRICK_FACES;
    }

    return at;
}

fn particle_hood(x: vec3<f32>) -> Hood {
    var low = vec3<i32>(params.dims.xyz);

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        low = min(low, vec3<i32>(floor(x * params.inv_dx - face_offset(axis) - 0.5)));
    }

    return hood_span(low, low + 3);
}

fn interpolate_axis(h: Hood, axis: u32, x: vec3<f32>) -> AxisSample {
    let offset = face_offset(axis);
    let g = x * params.inv_dx - offset;
    let base = vec3<i32>(floor(g - 0.5));
    let faces = face_dims(axis);
    let lx = lattice_axis(g.x, x.x, base.x, offset.x, faces.x, params.dims.x);
    let ly = lattice_axis(g.y, x.y, base.y, offset.y, faces.y, params.dims.y);
    let lz = lattice_axis(g.z, x.z, base.z, offset.z, faces.z, params.dims.z);
    let size = select(faces, face_box(axis), SPARSE);
    let first = select(params.face_base[axis], axis * BRICK_FACE_BOX, SPARSE);

    var v = 0.0;
    var wsum = 0.0;
    var count = 0u;
    var b_full = vec3<f32>(0.0);

    for (var o = 0; o < 27; o = o + 1) {
        let ox = o % 3;
        let oy = (o / 3) % 3;
        let oz = o / 9;

        if !(lx.inside[ox] && ly.inside[oy] && lz.inside[oz]) {
            continue;
        }

        let w = lx.weight[ox] * ly.weight[oy] * lz.weight[oz];
        let r = vec3<f32>(lx.offset[ox], ly.offset[oy], lz.offset[oz]);
        let u = velocity[lattice_index(h, lx, ly, lz, ox, oy, oz, first, size)];

        wsum = wsum + w;
        v = v + w * u;
        b_full = b_full + w * u * r;
        count = count + 1u;
    }

    var out: AxisSample;

    if wsum <= 0.0 {
        out.v = 0.0;
        out.c = vec3<f32>(0.0);
        return out;
    }

    v = v / wsum;
    out.v = v;

    if count == 27u {
        out.c = b_full * (4.0 * params.inv_dx * params.inv_dx);
        return out;
    }

    var b = vec3<f32>(0.0);
    var d = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));

    for (var o = 0; o < 27; o = o + 1) {
        let ox = o % 3;
        let oy = (o / 3) % 3;
        let oz = o / 9;

        if !(lx.inside[ox] && ly.inside[oy] && lz.inside[oz]) {
            continue;
        }

        let w = lx.weight[ox] * ly.weight[oy] * lz.weight[oz];
        let r = vec3<f32>(lx.offset[ox], ly.offset[oy], lz.offset[oz]);
        let u = velocity[lattice_index(h, lx, ly, lz, ox, oy, oz, first, size)];

        b = b + w * (u - v) * r;
        d = d + mat3x3<f32>(r * r.x, r * r.y, r * r.z) * w;
    }

    let det = determinant(d);
    let dx2 = params.dx * params.dx;

    if abs(det) < 1.0e-9 * dx2 * dx2 * dx2 {
        out.c = vec3<f32>(0.0);
    } else {
        out.c = transpose(inverse3(d)) * b;
    }

    return out;
}

override G2P_DELETE: bool = false;
override ADVECT_BODIES: bool = false;

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= state.particle_count {
        return;
    }

    var p = particles[i];
    let x = p.x.xyz;
    home_at(x);
    let h = particle_hood(x);
    let s0 = interpolate_axis(h, 0u, x);
    let s1 = interpolate_axis(h, 1u, x);
    let s2 = interpolate_axis(h, 2u, x);

    p.v = vec4<f32>(s0.v, s1.v, s2.v, 0.0);
    p.c0 = vec4<f32>(s0.c, p.c0.w);
    p.c1 = vec4<f32>(s1.c, 0.0);
    p.c2 = vec4<f32>(s2.c, 0.0);

    if G2P_DELETE && phi_at(x) < -band_depth() {
        p.v.w = 1.0;
    }

    particles[i] = p;
}

fn sample_axis(h: Hood, axis: u32, x: vec3<f32>) -> f32 {
    let dims = vec3<f32>(face_dims(axis));
    let hi = max(dims - 1.0, vec3<f32>(0.0));
    let g = clamp(x * params.inv_dx - face_offset(axis), vec3<f32>(0.0), hi);
    let base = min(floor(g), max(hi - 1.0, vec3<f32>(0.0)));
    let t = g - base;
    let b = vec3<i32>(base);
    let top = min(b + vec3<i32>(1), vec3<i32>(hi));

    var acc = 0.0;

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let pick = vec3<i32>(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let node = select(b, top, pick == vec3<i32>(1));
        let w = select(1.0 - t, t, pick == vec3<i32>(1));
        acc = acc + w.x * w.y * w.z * velocity[hood_face(h, axis, node)];
    }

    return acc;
}

fn sample_velocity(x: vec3<f32>) -> vec3<f32> {
    let h = velocity_hood(x);

    return vec3<f32>(sample_axis(h, 0u, x), sample_axis(h, 1u, x), sample_axis(h, 2u, x));
}

const ADVECT_CYCLES: u32 = 8u;

@compute @workgroup_size(256)
fn advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if ADVECT_BODIES && i == 0u {
        integrate_bodies();
    }

    if i >= state.particle_count {
        return;
    }

    let dt = state.dt;
    var x = particles[i].x.xyz;
    home_at(x);
    var v = sample_velocity(x);
    let cycles = select(1u, clamp(u32(ceil(dt * length(v) * params.inv_dx)), 1u, ADVECT_CYCLES), params.cfl > 1.0);
    let h = dt / f32(cycles);

    for (var k = 0u; k < cycles; k = k + 1u) {
        if k > 0u {
            v = sample_velocity(x);
        }

        x = x + h * sample_velocity(x + 0.5 * h * v);
    }

    particles[i].x = vec4<f32>(x, particles[i].x.w);
}

fn sdf_gradient(b: Body, x: vec3<f32>) -> vec3<f32> {
    let h = params.dx * 1.0e-2;
    let g = vec3<f32>(
        body_sdf(b, x + vec3<f32>(h, 0.0, 0.0)) - body_sdf(b, x - vec3<f32>(h, 0.0, 0.0)),
        body_sdf(b, x + vec3<f32>(0.0, h, 0.0)) - body_sdf(b, x - vec3<f32>(0.0, h, 0.0)),
        body_sdf(b, x + vec3<f32>(0.0, 0.0, h)) - body_sdf(b, x - vec3<f32>(0.0, 0.0, h)),
    );
    let l = length(g);

    if l > 0.0 {
        return g / l;
    }

    return vec3<f32>(0.0);
}

fn free_point(c: vec3<f32>, low: vec3<f32>, high: vec3<f32>) -> bool {
    if any(c < low) || any(c > high) {
        return false;
    }

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        if body_sdf(bodies[k], c) < 0.99 * params.margin {
            return false;
        }
    }

    return true;
}

fn push_out(b: Body, x: vec3<f32>, s: f32, low: vec3<f32>, high: vec3<f32>) -> vec3<f32> {
    let margin = params.margin;
    let along = x + sdf_gradient(b, x) * (margin - s);
    var best = along;
    var distance = BIG;

    if free_point(along, low, high) {
        distance = length(along - x);
    }

    let p = quat_rotate(quat_conj(b.rotation), x - b.position.xyz);

    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var side = 0u; side < 2u; side = side + 1u) {
            var reach = b.shape[a] + margin;

            if b.shape.w < 0.5 {
                let r = b.shape.x + margin;
                reach = sqrt(max(r * r - (dot(p, p) - p[a] * p[a]), 0.0));
            }

            let q = with_axis(p, a, select(-reach, reach, side == 1u));

            let c = b.position.xyz + quat_rotate(b.rotation, q);
            let d = length(q - p);

            if d < distance && free_point(c, low, high) {
                best = c;
                distance = d;
            }
        }
    }

    return best;
}

@compute @workgroup_size(256)
fn keep_inside(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i < state.particle_count {
        keep_particle(i);
    }
}

fn keep_particle(i: u32) {
    var x = particles[i].x.xyz;
    let margin = params.margin;
    let size = vec3<f32>(params.dims.xyz) * params.dx;
    let high = size - margin;
    var low = vec3<f32>(margin);

    if OPEN_FLOOR {
        low.y = 0.0;
    }

    var v = particles[i].v.xyz;

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        let b = bodies[k];
        let s = body_sdf(b, x);

        if s < margin {
            let out = push_out(b, x, s, low, high);
            let gap = out - x;
            var n = vec3<f32>(0.0);

            if dot(gap, gap) > 0.0 {
                n = normalize(gap);
            }

            let inward = dot(v - body_velocity_at(b, out), n);

            if inward < 0.0 {
                v = v - inward * n;
            }

            x = out;
        }
    }

    let clamped = clamp(x, low, high);
    v = select(v, vec3<f32>(0.0), clamped != x);

    particles[i].x = vec4<f32>(clamped, particles[i].x.w);
    particles[i].v = vec4<f32>(v, particles[i].v.w);
}
