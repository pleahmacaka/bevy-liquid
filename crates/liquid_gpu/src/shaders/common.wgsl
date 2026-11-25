struct Params {
    dims: vec4<u32>,
    face_base: vec4<u32>,
    node_dims: vec4<u32>,
    gravity: vec4<f32>,
    dx: f32,
    inv_dx: f32,
    density: f32,
    cfl: f32,
    max_dt: f32,
    particle_capacity: u32,
    body_count: u32,
    kernel_radius: f32,
    particle_radius: f32,
    margin: f32,
    frame_dt: f32,
    reduction_groups: u32,
    pool: vec4<u32>,
    physics: vec4<f32>,
    thermal: vec4<f32>,
}

struct Particle {
    x: vec4<f32>,
    v: vec4<f32>,
    c0: vec4<f32>,
    c1: vec4<f32>,
    c2: vec4<f32>,
}

struct Body {
    position: vec4<f32>,
    rotation: vec4<f32>,
    velocity: vec4<f32>,
    angular: vec4<f32>,
    shape: vec4<f32>,
    inv_inertia0: vec4<f32>,
    inv_inertia1: vec4<f32>,
    inv_inertia2: vec4<f32>,
}

struct State {
    dt: f32,
    remaining: f32,
    time: f32,
    running: u32,
    particle_count: u32,
    vmax_bits: atomic<u32>,
    substeps: u32,
    converged: u32,
    brick_overflow: u32,
    ticks: u32,
}

const WG: u32 = 256u;

// indirect-call validation can hand num_workgroups another dispatch's count, so strides read the args this dispatch used
fn dispatched_threads(slot: u32) -> u32 {
    return dispatch_args[3u * slot] * WG;
}
const BIG: f32 = 3.0e38;
const THETA_MIN: f32 = 0.01;

fn grid_cells() -> u32 {
    return params.dims.w;
}

fn grid_index(c: vec3<i32>) -> u32 {
    let d = vec3<i32>(params.dims.xyz);
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn grid_coords(n: u32) -> vec3<i32> {
    let d = params.dims.xyz;
    return vec3<i32>(vec3<u32>(n % d.x, (n / d.x) % d.y, n / (d.x * d.y)));
}

fn cell_count() -> u32 {
    if SPARSE {
        return params.pool.y;
    }

    return params.dims.w;
}

fn cell_index(c: vec3<i32>) -> u32 {
    if SPARSE {
        let local = vec3<u32>(c & vec3<i32>(7));
        return slot_of(c) * BRICK_CELLS + local.x + BRICK * (local.y + BRICK * local.z);
    }

    return grid_index(c);
}

fn cell_coords(n: u32) -> vec3<i32> {
    if SPARSE {
        let r = n % BRICK_CELLS;
        return slot_origin(n / BRICK_CELLS) + vec3<i32>(vec3<u32>(r % BRICK, (r / BRICK) % BRICK, r / (BRICK * BRICK)));
    }

    return grid_coords(n);
}

fn in_cells(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(params.dims.xyz));
}

fn axis_mask(a: u32) -> vec3<bool> {
    return vec3<u32>(a) == vec3<u32>(0u, 1u, 2u);
}

fn unit_axis(a: u32) -> vec3<i32> {
    return vec3<i32>(axis_mask(a));
}

fn bump(v: vec3<i32>, a: u32, d: i32) -> vec3<i32> {
    return v + d * unit_axis(a);
}

fn with_axis(v: vec3<f32>, a: u32, x: f32) -> vec3<f32> {
    return select(v, vec3<f32>(x), axis_mask(a));
}

fn face_dims(axis: u32) -> vec3<u32> {
    return params.dims.xyz + vec3<u32>(axis_mask(axis));
}

fn face_total() -> u32 {
    if SPARSE {
        return params.pool.z;
    }

    return params.face_base.w;
}

fn face_box(axis: u32) -> vec3<u32> {
    return vec3<u32>(BRICK) + vec3<u32>(axis_mask(axis));
}

fn face_axis(n: u32) -> u32 {
    if SPARSE {
        return (n % BRICK_FACES) / BRICK_FACE_BOX;
    }

    if n < params.face_base.y {
        return 0u;
    }
    if n < params.face_base.z {
        return 1u;
    }
    return 2u;
}

fn face_index(axis: u32, f: vec3<i32>) -> u32 {
    if SPARSE {
        let owner = min(f, vec3<i32>(params.dims.xyz) - 1);
        let local = vec3<u32>(f - (owner & vec3<i32>(~7)));
        let size = face_box(axis);
        return slot_of(owner) * BRICK_FACES + axis * BRICK_FACE_BOX + local.x + size.x * (local.y + size.y * local.z);
    }

    let d = vec3<i32>(face_dims(axis));
    return params.face_base[axis] + u32(f.x + d.x * (f.y + d.y * f.z));
}

fn face_coords(axis: u32, n: u32) -> vec3<i32> {
    if SPARSE {
        let q = n % BRICK_FACES - axis * BRICK_FACE_BOX;
        let size = face_box(axis);
        let local = vec3<u32>(q % size.x, (q / size.x) % size.y, q / (size.x * size.y));
        return slot_origin(n / BRICK_FACES) + vec3<i32>(local);
    }

    let local = n - params.face_base[axis];
    let d = face_dims(axis);
    return vec3<i32>(vec3<u32>(local % d.x, (local / d.x) % d.y, local / (d.x * d.y)));
}

fn face_step(n: u32, axis: u32, f: vec3<i32>, a: u32, step: i32) -> u32 {
    let p = bump(f, a, step);

    if SPARSE {
        let top = i32(params.dims[axis]);
        let local = (f[a] & 7) + step;

        if local >= 0 && local < 8 && f[axis] < top && p[axis] < top {
            let size = face_box(axis);
            let stride = select(select(size.x * size.y, size.x, a == 1u), 1u, a == 0u);
            return u32(i32(n) + step * i32(stride));
        }
    }

    return face_index(axis, p);
}

fn in_faces(axis: u32, f: vec3<i32>) -> bool {
    return all(f >= vec3<i32>(0)) && all(f < vec3<i32>(face_dims(axis)));
}

fn face_offset(axis: u32) -> vec3<f32> {
    return with_axis(vec3<f32>(0.5), axis, 0.0);
}

fn face_position(axis: u32, f: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(f) + face_offset(axis)) * params.dx;
}

fn cell_center(c: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(c) + 0.5) * params.dx;
}

fn node_count() -> u32 {
    if SPARSE {
        return params.pool.w;
    }

    return params.node_dims.w;
}

fn node_index(p: vec3<i32>) -> u32 {
    if SPARSE {
        let owner = min(p, vec3<i32>(params.dims.xyz) - 1);
        let local = vec3<u32>(p - (owner & vec3<i32>(~7)));
        let side = BRICK + 1u;
        return slot_of(owner) * BRICK_NODES + local.x + side * (local.y + side * local.z);
    }

    let d = vec3<i32>(params.node_dims.xyz);
    return u32(p.x + d.x * (p.y + d.y * p.z));
}

fn node_coords(n: u32) -> vec3<i32> {
    if SPARSE {
        let r = n % BRICK_NODES;
        let side = BRICK + 1u;
        return slot_origin(n / BRICK_NODES) + vec3<i32>(vec3<u32>(r % side, (r / side) % side, r / (side * side)));
    }

    let d = params.node_dims.xyz;
    return vec3<i32>(vec3<u32>(n % d.x, (n / d.x) % d.y, n / (d.x * d.y)));
}

fn cell_of(x: vec3<f32>) -> vec3<i32> {
    let c = vec3<i32>(floor(x * params.inv_dx));
    return clamp(c, vec3<i32>(0), vec3<i32>(params.dims.xyz) - 1);
}

fn bspline2(d: f32) -> f32 {
    let a = abs(d);
    if a < 0.5 {
        return 0.75 - a * a;
    }
    if a < 1.5 {
        let t = 1.5 - a;
        return 0.5 * t * t;
    }
    return 0.0;
}

fn bspline2_weight(d: vec3<f32>) -> f32 {
    return bspline2(d.x) * bspline2(d.y) * bspline2(d.z);
}

fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.xyz, q.w);
}

fn body_sdf(b: Body, x: vec3<f32>) -> f32 {
    let p = quat_rotate(quat_conj(b.rotation), x - b.position.xyz);

    if b.shape.w < 0.5 {
        return length(p) - b.shape.x;
    }

    let q = abs(p) - b.shape.xyz;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn body_velocity_at(b: Body, x: vec3<f32>) -> vec3<f32> {
    return b.velocity.xyz + cross(b.angular.xyz, x - b.position.xyz);
}

const S_SCALE: u32 = 0u;
const S_RZ: u32 = 1u;
const S_DQ: u32 = 3u;
const S_ALPHA: u32 = 4u;
const S_BETA: u32 = 5u;
const S_RNORM: u32 = 6u;
const S_BNORM: u32 = 7u;
const S_TOLERANCE: u32 = 8u;
const S_ITERATION: u32 = 9u;
const S_COUPLING: u32 = 10u;
const S_GRADIENT: u32 = 11u;
const S_FLOOR: u32 = 12u;
const S_FLOOR_MEAN: u32 = 13u;
const S_FLOOR_LEVEL: u32 = 14u;
const S_FLOOR_SET: u32 = 15u;
const S_FLOOR_RATE: u32 = 120u;
const S_FLOOR_OFFSET: u32 = 121u;
const MAX_BODIES: u32 = 8u;
const S_BODY_RESPONSE: u32 = 16u;
const S_IMPULSE: u32 = 64u;
const S_VISCOUS_ITERATIONS: u32 = 112u;
const S_VOLUME_TARGET: u32 = 114u;
const S_VOLUME_INTEGRAL: u32 = 115u;
const S_VOLUME: u32 = 116u;
const S_VOLUME_DIVERGENCE: u32 = 117u;
const S_BAND_DT: u32 = 118u;
const S_BAND_SEEDED: u32 = 119u;
const S_COUNT: u32 = 128u;

const BRICK: u32 = 8u;
const BRICK_ROWS: u32 = 64u;
const BRICK_CELLS: u32 = 512u;
const BRICK_FACE_BOX: u32 = 576u;
const BRICK_FACES: u32 = 1728u;
const BRICK_NODES: u32 = 729u;
const BRICK_OCCUPIED: u32 = 1u;
const BRICK_ACTIVE: u32 = 2u;
const BRICK_LIST_HEAD: u32 = 3u;
const NONE: u32 = 0xffffffffu;

override SPARSE: bool = false;
override OPEN_FLOOR: bool = false;
override FLOOR_REACH: f32 = 0.0;
override PER_CELL: u32 = 8u;
override BACKTRACE_ORDER: u32 = 4u;
override QUANTIZED: bool = false;
override LATE_SURFACE: bool = false;
override CULL_CROWDED: bool = false;

fn seed_octant(s: u32) -> u32 {
    if PER_CELL < 8u {
        let x = s & 1u;
        let y = (s >> 1u) & 1u;
        return x | (y << 1u) | ((x ^ y) << 2u);
    }

    return s;
}

fn sub_octant(s: u32) -> vec3<f32> {
    let o = seed_octant(s);
    return vec3<f32>(f32(o & 1u), f32((o >> 1u) & 1u), f32((o >> 2u) & 1u));
}

fn per_cell_scale() -> f32 {
    return 8.0 / f32(PER_CELL);
}

fn domain_size() -> vec3<f32> {
    return vec3<f32>(params.dims.xyz) * params.dx;
}

fn pack_position(x: vec3<f32>, temperature: f32) -> vec2<u32> {
    let u = clamp(x / domain_size(), vec3<f32>(0.0), vec3<f32>(1.0));
    let z = pack2x16unorm(vec2<f32>(u.z, 0.0)) | pack2x16float(vec2<f32>(0.0, temperature));
    return vec2<u32>(pack2x16unorm(u.xy), z);
}

fn unpack_position(q: vec2<u32>) -> vec3<f32> {
    return vec3<f32>(unpack2x16unorm(q.x), unpack2x16unorm(q.y).x) * domain_size();
}

fn motion_index(i: u32) -> u32 {
    return params.particle_capacity + 3u * i;
}

fn pack_motion(i: u32, p: Particle) {
    let m = motion_index(i);
    packed[m] = vec2<u32>(pack2x16float(p.v.xy), pack2x16float(vec2<f32>(p.v.z, p.c0.x)));
    packed[m + 1u] = vec2<u32>(pack2x16float(p.c0.yz), pack2x16float(p.c1.xy));
    packed[m + 2u] = vec2<u32>(pack2x16float(vec2<f32>(p.c1.z, p.c2.x)), pack2x16float(p.c2.yz));
}

fn unpack_particle(i: u32) -> Particle {
    let m = motion_index(i);
    let a = packed[m];
    let b = packed[m + 1u];
    let c = packed[m + 2u];
    let v = unpack2x16float(a.x);
    let w = unpack2x16float(a.y);
    let c0 = unpack2x16float(b.x);
    let c1 = unpack2x16float(b.y);
    let c2 = unpack2x16float(c.x);
    let c3 = unpack2x16float(c.y);

    var p: Particle;
    p.x = vec4<f32>(unpack_position(packed[i]), 0.0);
    p.v = vec4<f32>(v, w.x, 0.0);
    p.c0 = vec4<f32>(w.y, c0, unpack2x16float(packed[i].y).y);
    p.c1 = vec4<f32>(c1, c2.x, 0.0);
    p.c2 = vec4<f32>(c2.y, c3, 0.0);
    return p;
}

fn sorted_position(i: u32) -> vec3<f32> {
    if QUANTIZED {
        return unpack_position(packed[i]);
    }

    return particles[i].x.xyz;
}

fn brick_dims() -> vec3<u32> {
    return (params.dims.xyz + BRICK - 1u) / BRICK;
}

fn brick_total() -> u32 {
    let d = brick_dims();
    return d.x * d.y * d.z;
}

fn brick_coords(b: u32) -> vec3<i32> {
    let d = brick_dims();
    return vec3<i32>(vec3<u32>(b % d.x, (b / d.x) % d.y, b / (d.x * d.y)));
}

fn brick_origin(b: u32) -> vec3<i32> {
    return brick_coords(b) * i32(BRICK);
}

fn brick_of(c: vec3<i32>) -> u32 {
    let d = brick_dims();
    let p = vec3<u32>(c) / BRICK;
    return p.x + d.x * (p.y + d.y * p.z);
}

fn brick_table() -> u32 {
    return BRICK_LIST_HEAD + 4u * brick_total();
}

fn brick_neighbors() -> u32 {
    return brick_table() + brick_total() + 2u * params.pool.x + 1u;
}

fn brick_listing() -> u32 {
    return brick_neighbors() + 27u * (params.pool.x + 1u);
}

fn level_table() -> u32 {
    return brick_listing() + 2u * brick_total();
}

var<private> home_slot: u32;
var<private> home_brick: vec3<i32>;

fn set_home(slot: u32, origin: vec3<i32>) {
    home_slot = slot;
    home_brick = origin >> vec3<u32>(3u);
}

fn slot_of(c: vec3<i32>) -> u32 {
    let o = (c >> vec3<u32>(3u)) - home_brick;

    if home_slot != 0u && all(o >= vec3<i32>(-1)) && all(o <= vec3<i32>(1)) {
        if all(o == vec3<i32>(0)) {
            return home_slot;
        }

        return brick_list[brick_neighbors() + 27u * home_slot + u32(13 + o.x + 3 * o.y + 9 * o.z)];
    }

    return brick_list[brick_table() + brick_of(c)];
}

fn unpack_origin(packed: u32) -> vec3<i32> {
    return vec3<i32>(vec3<u32>(packed & 1023u, (packed >> 10u) & 1023u, packed >> 20u) * BRICK);
}

fn slot_origin(slot: u32) -> vec3<i32> {
    if slot == home_slot && slot != 0u {
        return home_brick * i32(BRICK);
    }

    return unpack_origin(brick_list[brick_table() + brick_total() + slot]);
}

fn pack_brick(b: u32) -> u32 {
    let p = vec3<u32>(brick_coords(b));
    return p.x | (p.y << 10u) | (p.z << 20u);
}

fn brick_live(c: vec3<i32>) -> bool {
    return slot_of(c) != 0u;
}

struct BrickRun {
    origin: vec3<i32>,
    width: u32,
    index: u32,
}

fn brick_run(t: u32, span: u32) -> BrickRun {
    var run: BrickRun;
    run.width = 0u;

    let s = t / span;

    if s >= brick_list[0] {
        return run;
    }

    let total = brick_total();
    let first = brick_list[BRICK_LIST_HEAD + 2u * total + s];
    run.origin = brick_origin(brick_list[BRICK_LIST_HEAD + first]);
    run.width = brick_list[BRICK_LIST_HEAD + 3u * total + s] * BRICK;
    run.index = t - first * span;

    return run;
}

fn grid_face(t: u32) -> u32 {
    if !SPARSE {
        return t;
    }

    let run = brick_run(t, BRICK_FACES);

    if run.width == 0u {
        return NONE;
    }

    let per_axis = run.width * (BRICK_FACE_BOX / BRICK);
    let axis = run.index / per_axis;
    let q = run.index % per_axis;
    let size = vec3<u32>(run.width, BRICK, BRICK) + vec3<u32>(axis_mask(axis));

    let local = vec3<u32>(q % size.x, (q / size.x) % size.y, q / (size.x * size.y));
    let f = run.origin + vec3<i32>(local);

    if local.z >= size.z || !in_faces(axis, f) || (local[axis] + 1u == size[axis] && f[axis] != i32(params.dims[axis])) {
        return NONE;
    }

    return face_index(axis, f);
}


fn masked_cell(t: u32) -> u32 {
    if !SPARSE {
        return t;
    }

    if t >= grid_cells() {
        return NONE;
    }

    let c = grid_coords(t);

    if !brick_live(c) {
        return NONE;
    }

    return cell_index(c);
}

fn listed_slot(t: u32, span: u32) -> u32 {
    let s = t / span;

    if s >= brick_list[0] {
        return 0u;
    }

    let at = brick_listing() + 2u * s;
    set_home(brick_list[at], unpack_origin(brick_list[at + 1u]));

    return home_slot;
}

fn brick_cell(t: u32) -> u32 {
    if !SPARSE {
        return t;
    }

    let slot = listed_slot(t, BRICK_CELLS);
    let n = slot * BRICK_CELLS + t % BRICK_CELLS;

    if slot == 0u || !in_cells(cell_coords(n)) {
        return NONE;
    }

    return n;
}

fn brick_face(t: u32) -> u32 {
    if !SPARSE {
        return t;
    }

    let slot = listed_slot(t, BRICK_FACES);

    if slot == 0u {
        return NONE;
    }

    let n = slot * BRICK_FACES + t % BRICK_FACES;
    let axis = face_axis(n);
    let f = face_coords(axis, n);
    let edge = f[axis] - slot_origin(slot)[axis] == i32(BRICK);

    if !in_faces(axis, f) || (edge && f[axis] != i32(params.dims[axis])) {
        return NONE;
    }

    return n;
}

fn brick_node(t: u32) -> u32 {
    if !SPARSE {
        return t;
    }

    let slot = listed_slot(t, BRICK_NODES);

    if slot == 0u {
        return NONE;
    }

    let n = slot * BRICK_NODES + t % BRICK_NODES;
    let p = node_coords(n);
    let top = vec3<i32>(params.dims.xyz);
    let edge = p - slot_origin(slot) == vec3<i32>(i32(BRICK));

    if any(p > top) || any(edge & (p != top)) {
        return NONE;
    }

    return n;
}

struct Corners {
    index: array<u32, 8>,
    weight: array<f32, 8>,
}

fn trilinear_corners(g_in: vec3<f32>, dims: vec3<u32>, axis: u32) -> Corners {
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
            out.index[corner] = face_index(axis, p);
        } else {
            out.index[corner] = cell_index(p);
        }

        out.weight[corner] = w.x * w.y * w.z;
    }

    return out;
}

fn cell_corners(x: vec3<f32>) -> Corners {
    return trilinear_corners(x * params.inv_dx - 0.5, params.dims.xyz, 3u);
}

fn index_corners(g: vec3<f32>) -> Corners {
    return trilinear_corners(g, params.dims.xyz, 3u);
}

fn face_corners(axis: u32, x: vec3<f32>) -> Corners {
    return trilinear_corners(x * params.inv_dx - face_offset(axis), face_dims(axis), axis);
}

fn hash_u32(seed: u32) -> u32 {
    var s = seed ^ 2747636419u;
    s = s * 2654435769u;
    s = s ^ (s >> 16u);
    s = s * 2654435769u;
    s = s ^ (s >> 16u);
    return s * 2654435769u;
}
