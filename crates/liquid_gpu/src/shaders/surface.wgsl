@group(0) @binding(21) var<storage, read_write> ellipsoids: array<vec4<f32>>;
@group(0) @binding(22) var<storage, read_write> surface: array<atomic<u32>>;

const ANISO_NEIGHBORS: f32 = 25.0;
const ANISO_KR: f32 = 4.0;
const ANISO_KN: f32 = 0.5;
// below 1.0 the kernel of a particle with few neighbours is narrower than the cell spacing, so it flickers as it crosses cells
const SPARSE_KN: f32 = 1.0;
// a kernel axis shorter than one cell is undersampled by the grid, and the surface shows a ledge at every cell layer
const ANISO_MIN_AXIS: f32 = 1.0;
const ANISO_LAMBDA: f32 = 0.9;
const SURFACE_BAND: f32 = 2.5;
const FIELD_SCALE: f32 = 16777216.0;
const WALL_OPEN: f32 = 0.1;

fn row_particles(cy: i32, cz: i32, x0: i32, x1: i32) -> vec2<u32> {
    return vec2<u32>(cell_start[grid_index(vec3<i32>(x0, cy, cz))], cell_start[grid_index(vec3<i32>(x1, cy, cz)) + 1u]);
}

fn axis_gap(lo: f32, x: f32) -> f32 {
    return max(max(lo - x, x - lo - params.dx), 0.0);
}

fn gap_squared(lo: f32, x: f32) -> f32 {
    let g = axis_gap(lo, x);
    return g * g;
}

fn rotation_column(i: i32, pp: i32, qq: i32, c: f32, s: f32) -> vec3<f32> {
    let rows = vec3<i32>(0, 1, 2);
    let identity = vec3<f32>(rows == vec3<i32>(i));
    let on_p = rows == vec3<i32>(pp);
    let on_q = rows == vec3<i32>(qq);

    if i == pp {
        return select(select(identity, vec3<f32>(-s), on_q), vec3<f32>(c), on_p);
    }

    if i == qq {
        return select(select(identity, vec3<f32>(s), on_p), vec3<f32>(c), on_q);
    }

    return identity;
}

fn jacobi_rotate(a: ptr<function, mat3x3<f32>>, v: ptr<function, mat3x3<f32>>, pp: i32, qq: i32) {
    let apq = (*a)[qq][pp];

    if abs(apq) < 1.0e-20 {
        return;
    }

    let theta = ((*a)[qq][qq] - (*a)[pp][pp]) / (2.0 * apq);
    let t = sign(theta) / (abs(theta) + sqrt(theta * theta + 1.0));
    let c = 1.0 / sqrt(t * t + 1.0);
    let s = t * c;

    let j = mat3x3<f32>(rotation_column(0, pp, qq, c, s), rotation_column(1, pp, qq, c, s), rotation_column(2, pp, qq, c, s));

    *a = transpose(j) * (*a) * j;
    *v = (*v) * j;
}

fn jacobi_eigen(a_in: mat3x3<f32>) -> array<vec3<f32>, 4> {
    var a = a_in;
    var v = mat3x3<f32>(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.0, 0.0, 1.0));

    for (var sweep = 0; sweep < 8; sweep = sweep + 1) {
        jacobi_rotate(&a, &v, 0, 1);
        jacobi_rotate(&a, &v, 0, 2);
        jacobi_rotate(&a, &v, 1, 2);
    }

    return array<vec3<f32>, 4>(vec3<f32>(a[0][0], a[1][1], a[2][2]), v[0], v[1], v[2]);
}

fn packed_position(i: u32) -> vec3<f32> {
    return ellipsoids[params.particle_capacity + i].xyz;
}

fn shape_index(i: u32) -> u32 {
    return 2u * params.particle_capacity + i;
}

fn shape_rows(i: u32) -> u32 {
    return 3u * params.particle_capacity + 3u * i;
}

fn write_shape(i: u32, center: vec3<f32>, reach: f32, g: mat3x3<f32>) {
    let base = shape_rows(i);
    ellipsoids[shape_index(i)] = vec4<f32>(center, reach_squared(reach));
    ellipsoids[base] = vec4<f32>(g[0], shape_scale(determinant(g)));
    ellipsoids[base + 1u] = vec4<f32>(g[1], 0.0);
    ellipsoids[base + 2u] = vec4<f32>(g[2], 0.0);
}

struct Moments {
    weight: f32,
    first: vec3<f32>,
    diagonal: vec3<f32>,
    cross: vec3<f32>,
    count: f32,
}

fn add_moment(m: Moments, e: vec3<f32>, inv_r2: f32) -> Moments {
    let t = dot(e, e) * inv_r2;

    if t >= 1.0 {
        return m;
    }

    let w = 1.0 - t * sqrt(t);
    let we = w * e;

    return Moments(m.weight + w, m.first + we, m.diagonal + we * e, m.cross + we.xxy * e.yzz, m.count + 1.0);
}

@compute @workgroup_size(256)
fn aniso_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= state.particle_count {
        return;
    }

    let xi = packed_position(i);
    let r = 2.0 * params.dx;
    let r2 = r * r;
    let c = cell_of(xi);

    if phi[cell_index(c)] < -surface_band() {
        let g = 1.0 / (ANISO_KN * params.dx);
        write_shape(i, xi, 2.0 / g, mat3x3<f32>(vec3<f32>(g, 0.0, 0.0), vec3<f32>(0.0, g, 0.0), vec3<f32>(0.0, 0.0, g)));
        return;
    }
    let lo = max(c - vec3<i32>(2), vec3<i32>(0));
    let hi = min(c + vec3<i32>(2), vec3<i32>(params.dims.xyz) - 1);

    let reach2 = 1.01 * r2;
    let inv_r2 = 1.0 / r2;
    var m = Moments(0.0, vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), 0.0);

    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        let gz = axis_gap(f32(cz) * params.dx, xi.z);

        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            let gy = axis_gap(f32(cy) * params.dx, xi.y);
            let rest = reach2 - gy * gy - gz * gz;

            if rest < 0.0 {
                continue;
            }

            var x0 = lo.x;
            var x1 = hi.x;

            while x0 < c.x && gap_squared(f32(x0) * params.dx, xi.x) > rest {
                x0 = x0 + 1;
            }

            while x1 > c.x && gap_squared(f32(x1) * params.dx, xi.x) > rest {
                x1 = x1 - 1;
            }

            let span = row_particles(cy, cz, x0, x1);

            var j = span.x;

            while j + 4u <= span.y {
                let a = packed_position(j);
                let b = packed_position(j + 1u);
                let c4 = packed_position(j + 2u);
                let d = packed_position(j + 3u);
                m = add_moment(m, a - xi, inv_r2);
                m = add_moment(m, b - xi, inv_r2);
                m = add_moment(m, c4 - xi, inv_r2);
                m = add_moment(m, d - xi, inv_r2);
                j = j + 4u;
            }

            while j < span.y {
                m = add_moment(m, packed_position(j) - xi, inv_r2);
                j = j + 1u;
            }
        }
    }

    let count = m.count;
    let inv_w = 1.0 / max(m.weight, 1.0e-12);
    let offset = m.first * inv_w;
    let mean = xi + offset;
    let dg = m.diagonal * inv_w - offset * offset;
    let o = m.cross * inv_w - offset.xxy * offset.yzz;
    let cov = mat3x3<f32>(vec3<f32>(dg.x, o.x, o.y), vec3<f32>(o.x, dg.y, o.z), vec3<f32>(o.y, o.z, dg.z));

    let h = params.dx;
    var g = mat3x3<f32>(vec3<f32>(1.0 / h, 0.0, 0.0), vec3<f32>(0.0, 1.0 / h, 0.0), vec3<f32>(0.0, 0.0, 1.0 / h)) * (1.0 / SPARSE_KN);
    var reach = 2.0 * SPARSE_KN * h;

    if count > ANISO_NEIGHBORS {
        let eig = jacobi_eigen(cov);
        let sigma = eig[0];
        let s1 = max(sigma.x, max(sigma.y, sigma.z));
        let floor_value = s1 / ANISO_KR;
        let s = max(sigma, vec3<f32>(max(floor_value, 1.0e-12)));
        let rot = mat3x3<f32>(eig[1], eig[2], eig[3]);
        let scale = pow(s.x * s.y * s.z, 1.0 / 3.0);
        let axes = max(s / scale, vec3<f32>(ANISO_MIN_AXIS));
        let inv = mat3x3<f32>(vec3<f32>(1.0 / axes.x, 0.0, 0.0), vec3<f32>(0.0, 1.0 / axes.y, 0.0), vec3<f32>(0.0, 0.0, 1.0 / axes.z));
        g = rot * inv * transpose(rot) * (1.0 / h);
        reach = 2.0 * max(axes.x, max(axes.y, axes.z)) * h;
    }

    write_shape(i, mix(xi, mean, ANISO_LAMBDA), reach, g);
}

fn surface_band() -> f32 {
    if params.physics.z > 0.0 {
        return min(SURFACE_BAND, params.physics.z - 0.5) * params.dx;
    }

    return SURFACE_BAND * params.dx;
}

fn shape_scale(det: f32) -> f32 {
    return params.dx * params.dx * params.dx / f32(PER_CELL) * det / 3.14159265;
}

fn reach_squared(reach: f32) -> f32 {
    return 1.0001 * reach * reach;
}

fn cubic_kernel(q: f32) -> f32 {
    if q < 1.0 {
        return 1.0 - 1.5 * q * q + 0.75 * q * q * q;
    }
    if q < 2.0 {
        let t = 2.0 - q;
        return 0.25 * t * t * t;
    }
    return 0.0;
}

// kernel mass past a domain face folds back in; clipped, a layer thinner than a cell at the floor stays under the iso level
fn folded(c: vec3<i32>, dims: vec3<i32>) -> vec3<i32> {
    let low = select(c, -1 - c, c < vec3<i32>(0));
    return select(low, 2 * dims - 1 - low, low >= dims);
}

@compute @workgroup_size(256)
fn surface_splat(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;

    if j >= state.particle_count {
        return;
    }

    let dims = vec3<i32>(params.dims.xyz);
    let home = cell_of(packed_position(j));
    let shape = ellipsoids[shape_index(j)];
    let center = shape.xyz;
    let reach = sqrt(shape.w);
    let base = shape_rows(j);
    let g0 = ellipsoids[base];
    let g = mat3x3<f32>(g0.xyz, ellipsoids[base + 1u].xyz, ellipsoids[base + 2u].xyz);
    let lo = max(home - 2, vec3<i32>(ceil((center - reach) * params.inv_dx - 0.5)));
    let hi = min(home + 2, vec3<i32>(floor((center + reach) * params.inv_dx - 0.5)));

    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
                let c = vec3<i32>(cx, cy, cz);
                let d = cell_center(c) - center;

                if dot(d, d) >= shape.w {
                    continue;
                }

                let w = g0.w * cubic_kernel(length(g * d));

                if w > 0.0 {
                    atomicAdd(&surface[grid_index(folded(c, dims))], u32(w * FIELD_SCALE + 0.5));
                }
            }
        }
    }
}

// a cell centred in a body has phi = BIG; without this share a film thinner than a cell has no field at the wall
fn open_share(c: vec3<i32>) -> f32 {
    if !has_solids() {
        return 1.0;
    }

    return clamp(0.5 + solid_phi_at_cell(c) * params.inv_dx, WALL_OPEN, 1.0);
}

@compute @workgroup_size(256)
fn surface_field(@builtin(global_invocation_id) gid: vec3<u32>) {
    let g = gid.x;

    if g >= grid_cells() {
        return;
    }

    let c = grid_coords(g);
    var value = 0.0;

    if !SPARSE || brick_live(c) {
        let here = phi[cell_index(c)];
        let band = surface_band();
        let open = open_share(c);

        if here < -band {
            value = 1.0;
        } else if open < 1.0 {
            value = min(f32(atomicLoad(&surface[g])) / (FIELD_SCALE * open), 1.0);
        } else if here <= band {
            value = f32(atomicLoad(&surface[g])) / FIELD_SCALE;
        }
    }

    atomicStore(&surface[g], bitcast<u32>(value));

    if SURFACE_WHITE {
        white_gradient(g);
    }
}

override SURFACE_WHITE: bool = false;
