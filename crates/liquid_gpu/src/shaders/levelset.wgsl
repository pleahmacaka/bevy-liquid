fn particle_phi(n: u32) -> f32 {
    let c = cell_coords(n);
    let x = cell_center(c);
    let radius = params.kernel_radius * params.dx;
    let lo = max(c - vec3<i32>(1), vec3<i32>(0));
    let hi = min(c + vec3<i32>(1), vec3<i32>(params.dims.xyz) - 1);

    var wsum = 0.0;
    var centroid = vec3<f32>(0.0);

    let span = u32(hi.x - lo.x) + 1u;

    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            let row = grid_index(vec3<i32>(lo.x, cy, cz));
            let end = cell_start[row + span];

            for (var i = cell_start[row]; i < end; i = i + 1u) {
                let p = sorted_position(i);
                let d = x - p;
                let s2 = dot(d, d) / (radius * radius);

                if s2 < 1.0 {
                    let k = (1.0 - s2) * (1.0 - s2) * (1.0 - s2);
                    wsum = wsum + k;
                    centroid = centroid + k * p;
                }
            }
        }
    }

    if wsum <= 0.0 {
        return 3.0 * params.dx;
    }

    return length(x - centroid / wsum) - params.particle_radius * params.dx;
}

@compute @workgroup_size(256)
fn liquid_phi(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = masked_cell(gid.x);

    if n < cell_count() {
        phi[n] = particle_phi(n);
    }
}

const PHI_ROWS: u32 = 28u;

var<workgroup> phi_blocks: u32;
var<workgroup> phi_sums: array<vec4<f32>, 256>;

fn kernel_row(c: vec3<i32>, cy: i32, cz: i32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);

    if cy < 0 || cz < 0 || cy >= i32(params.dims.y) || cz >= i32(params.dims.z) {
        return sum;
    }

    let x = cell_center(c);
    let radius = params.kernel_radius * params.dx;
    let lo = max(c.x - 1, 0);
    let hi = min(c.x + 1, i32(params.dims.x) - 1);
    let row = grid_index(vec3<i32>(lo, cy, cz));
    let end = cell_start[row + u32(hi - lo) + 1u];

    for (var i = cell_start[row]; i < end; i = i + 1u) {
        let p = sorted_position(i);
        let d = x - p;
        let s2 = dot(d, d) / (radius * radius);

        if s2 < 1.0 {
            let k = (1.0 - s2) * (1.0 - s2) * (1.0 - s2);
            sum = sum + vec4<f32>(k * p, k);
        }
    }

    return sum;
}

fn cell_items() -> u32 {
    if SPARSE {
        return brick_list[0] * BRICK_CELLS;
    }

    return cell_count();
}

@compute @workgroup_size(256)
fn band_liquid_phi(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    blend_particle_phi(wid, groups, lid);
}

fn blend_particle_phi(wid: vec3<u32>, groups: vec3<u32>, lid: u32) {
    if lid == 0u {
        phi_blocks = select(0u, (cell_items() + PHI_ROWS - 1u) / PHI_ROWS, state.running != 0u);
    }

    let blocks = workgroupUniformLoad(&phi_blocks);
    let r = i32(lid / PHI_ROWS);

    for (var block = wid.x; block < blocks; block = block + groups.x) {
        let n = brick_cell(block * PHI_ROWS + lid % PHI_ROWS);
        let live = n < cell_count() && lid < 9u * PHI_ROWS;
        var sum = vec4<f32>(0.0);
        var c = vec3<i32>(0);

        if live && r < 9 {
            c = cell_coords(n);
            sum = kernel_row(c, c.y - 1 + r % 3, c.z - 1 + r / 3);
        }

        phi_sums[lid] = sum;
        workgroupBarrier();

        if live && r == 0 {
            for (var k = 1u; k < 9u; k = k + 1u) {
                sum = sum + phi_sums[lid + PHI_ROWS * k];
            }

            var d = 3.0 * params.dx;

            if sum.w > 0.0 {
                d = length(cell_center(c) - sum.xyz / sum.w) - params.particle_radius * params.dx;
            }

            phi[n] = min(d, sweep_a[n] + params.dx);
        }

        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn solid_nodes(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_node(gid.x);

    if n < node_count() {
        node_solid(n);
    }
}

fn node_solid(n: u32) {
    let x = vec3<f32>(node_coords(n)) * params.dx;
    var s = BIG;

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        s = min(s, body_sdf(bodies[k], x));
    }

    nodes[n] = s;
}

override STATIC_SOLID: bool = false;

fn has_solids() -> bool {
    return params.body_count > 0u || STATIC_SOLID;
}

@compute @workgroup_size(256)
fn solid_nodes_static(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_node(gid.x);

    if n < node_count() {
        node_solid(n);
        let p = vec3<u32>(node_coords(n));
        let d = params.dims.xyz + 1u;
        nodes[n] = min(nodes[n], static_solid[p.x + d.x * (p.y + d.y * p.z)]);
    }
}

fn solid_at(x: vec3<f32>) -> f32 {
    let g = x * params.inv_dx;
    let c = clamp(vec3<i32>(floor(g)), vec3<i32>(0), vec3<i32>(params.dims.xyz) - 1);
    let t = clamp(g - vec3<f32>(c), vec3<f32>(0.0), vec3<f32>(1.0));
    var s = 0.0;

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let o = vec3<i32>(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let w = select(1.0 - t, t, o == vec3<i32>(1));
        s = s + w.x * w.y * w.z * nodes[node_index(c + o)];
    }

    return s;
}

fn solid_normal(x: vec3<f32>) -> vec3<f32> {
    let h = 0.5 * params.dx;
    let g = vec3<f32>(
        solid_at(x + vec3<f32>(h, 0.0, 0.0)) - solid_at(x - vec3<f32>(h, 0.0, 0.0)),
        solid_at(x + vec3<f32>(0.0, h, 0.0)) - solid_at(x - vec3<f32>(0.0, h, 0.0)),
        solid_at(x + vec3<f32>(0.0, 0.0, h)) - solid_at(x - vec3<f32>(0.0, 0.0, h)),
    );
    let l = length(g);

    if l > 0.0 {
        return g / l;
    }

    return vec3<f32>(0.0);
}

@compute @workgroup_size(256)
fn expel_static(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= state.particle_count {
        return;
    }

    let x = particles[i].x.xyz;
    let s = solid_at(x);

    if s >= params.margin {
        return;
    }

    let n = solid_normal(x);
    var v = particles[i].v.xyz;
    let inward = dot(v, n);

    if inward < 0.0 {
        v = v - inward * n;
    }

    let size = vec3<f32>(params.dims.xyz) * params.dx;
    var low = vec3<f32>(params.margin);

    if OPEN_FLOOR {
        low.y = 0.0;
    }

    let out = clamp(x + n * (params.margin - s), low, size - params.margin);
    particles[i].x = vec4<f32>(out, particles[i].x.w);
    particles[i].v = vec4<f32>(v, particles[i].v.w);
}

fn solid_phi_at_cell(c: vec3<i32>) -> f32 {
    var sum = 0.0;

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let o = vec3<i32>(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        sum = sum + nodes[node_index(c + o)];
    }

    return sum * 0.125;
}

@compute @workgroup_size(256)
fn mark_solid_cells(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n >= cell_count() {
        return;
    }

    if has_solids() && solid_phi_at_cell(cell_coords(n)) < 0.0 {
        phi[n] = BIG;
    }
}

fn extension_source(m: u32, from_scratch: bool) -> f32 {
    if from_scratch {
        return scratch[m];
    }

    return phi[m];
}

fn settled(v: f32) -> f32 {
    return select(v, 3.0 * params.dx, v >= BIG);
}

@compute @workgroup_size(256)
fn extend_into_solid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        scratch[n] = extended(n, false);
    }
}

@compute @workgroup_size(256)
fn extend_from_scratch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        phi[n] = extended(n, true);
    }
}

@compute @workgroup_size(256)
fn extend_settle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        phi[n] = settled(extended(n, true));
    }
}

@compute @workgroup_size(256)
fn settle_from_scratch(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_cell(gid.x);

    if n < cell_count() {
        phi[n] = settled(scratch[n]);
    }
}

fn extended(n: u32, from_scratch: bool) -> f32 {
    let own = extension_source(n, from_scratch);

    if own < BIG {
        return own;
    }

    let c = cell_coords(n);
    var sum = 0.0;
    var count = 0.0;

    for (var axis = 0u; axis < 3u; axis = axis + 1u) {
        for (var step = -1; step <= 1; step = step + 2) {
            let p = bump(c, axis, step);

            if !in_cells(p) {
                continue;
            }

            let v = extension_source(cell_index(p), from_scratch);

            if v < BIG {
                sum = sum + v;
                count = count + 1.0;
            }
        }
    }

    if count > 0.0 {
        return sum / count;
    }

    return own;
}

fn covered(v: f32) -> bool {
    return v <= 0.0;
}

fn fraction_edge(a: f32, b: f32) -> f32 {
    if covered(a) && covered(b) {
        return 1.0;
    }
    if covered(a) {
        return a / (a - b);
    }
    if covered(b) {
        return b / (b - a);
    }
    return 0.0;
}

fn rotate4(l: array<f32, 4>) -> array<f32, 4> {
    return array<f32, 4>(l[1], l[2], l[3], l[0]);
}

fn fraction_square(bl: f32, br: f32, tl: f32, tr: f32) -> f32 {
    var inside = 0u;

    if covered(bl) { inside = inside + 1u; }
    if covered(br) { inside = inside + 1u; }
    if covered(tl) { inside = inside + 1u; }
    if covered(tr) { inside = inside + 1u; }

    var l = array<f32, 4>(bl, br, tr, tl);

    if inside == 4u {
        return 1.0;
    }

    if inside == 3u {
        for (var k = 0; k < 4; k = k + 1) {
            if !covered(l[0]) { break; }
            l = rotate4(l);
        }

        let s0 = 1.0 - fraction_edge(l[0], l[3]);
        let s1 = 1.0 - fraction_edge(l[0], l[1]);
        return 1.0 - 0.5 * s0 * s1;
    }

    if inside == 1u {
        for (var k = 0; k < 4; k = k + 1) {
            if covered(l[0]) { break; }
            l = rotate4(l);
        }

        let s0 = fraction_edge(l[0], l[3]);
        let s1 = fraction_edge(l[0], l[1]);
        return 0.5 * s0 * s1;
    }

    if inside == 2u {
        for (var k = 0; k < 4; k = k + 1) {
            if covered(l[0]) && (covered(l[1]) || covered(l[2])) { break; }
            l = rotate4(l);
        }

        if covered(l[1]) {
            return 0.5 * (fraction_edge(l[0], l[3]) + fraction_edge(l[1], l[2]));
        }

        let middle = 0.25 * (l[0] + l[1] + l[2] + l[3]);

        if middle < 0.0 {
            let s1 = 1.0 - fraction_edge(l[0], l[3]);
            let s3 = 1.0 - fraction_edge(l[2], l[3]);
            let s2 = 1.0 - fraction_edge(l[2], l[1]);
            let s0 = 1.0 - fraction_edge(l[0], l[1]);
            return 1.0 - 0.5 * s1 * s3 - 0.5 * s0 * s2;
        }

        let s0 = fraction_edge(l[0], l[1]);
        let s1 = fraction_edge(l[0], l[3]);
        let s2 = fraction_edge(l[2], l[1]);
        let s3 = fraction_edge(l[2], l[3]);
        return 0.5 * s0 * s1 + 0.5 * s2 * s3;
    }

    return 0.0;
}

@compute @workgroup_size(256)
fn face_weights(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        settle_face(n, false);
    }
}

fn moving_velocity(b: Body, x: vec3<f32>, gravity: bool) -> vec3<f32> {
    var v = b.velocity.xyz;

    if gravity && b.velocity.w > 1.5 {
        v = v + params.gravity.xyz * state.dt;
    }

    return v + cross(b.angular.xyz, x - b.position.xyz);
}

fn settle_face(n: u32, gravity: bool) {
    let axis = face_axis(n);
    let f = face_coords(axis, n);
    let open = OPEN_FLOOR && axis == 1u && f.y == 0;

    if (f[axis] == 0 && !open) || f[axis] == i32(params.dims[axis]) {
        weights[n] = 0.0;
        owners[n] = -1;
        solid_velocity[n] = 0.0;
        return;
    }

    let e1 = unit_axis((axis + 1u) % 3u);
    let e2 = unit_axis((axis + 2u) % 3u);

    let bl = nodes[node_index(f)];
    let br = nodes[node_index(f + e1)];
    let tl = nodes[node_index(f + e2)];
    let tr = nodes[node_index(f + e1 + e2)];
    let w = 1.0 - fraction_square(bl, br, tl, tr);
    weights[n] = w;

    var owner = -1;
    var best = BIG;
    let x = face_position(axis, f);

    if w < 1.0 {
        let reach = 0.70710678 * params.dx;

        for (var k = 0u; k < params.body_count; k = k + 1u) {
            let s = body_sdf(bodies[k], x);

            if s < reach && s < best {
                best = s;
                owner = i32(k);
            }
        }
    }

    owners[n] = owner;

    if owner >= 0 {
        solid_velocity[n] = moving_velocity(bodies[u32(owner)], x, gravity)[axis];
    } else {
        solid_velocity[n] = 0.0;
    }
}

@compute @workgroup_size(256)
fn refresh_solid_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = brick_face(gid.x);

    if n < face_total() {
        refresh_face(n, false);
    }
}

fn refresh_face(n: u32, gravity: bool) {
    let owner = owners[n];

    if owner < 0 {
        solid_velocity[n] = 0.0;
        return;
    }

    let axis = face_axis(n);
    let x = face_position(axis, face_coords(axis, n));
    solid_velocity[n] = moving_velocity(bodies[u32(owner)], x, gravity)[axis];
}
