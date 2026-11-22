struct White {
    x: vec4<f32>,
    v: vec4<f32>,
}

struct WhiteState {
    count: u32,
    next: atomic<u32>,
    spawned: atomic<u32>,
    pad: u32,
}

@group(0) @binding(23) var<storage, read_write> white: array<White>;
@group(0) @binding(24) var<storage, read_write> white_out: array<White>;
@group(0) @binding(25) var<storage, read_write> white_state: WhiteState;

const WHITE_SPRAY: f32 = 0.0;
const WHITE_FOAM: f32 = 1.0;
const WHITE_BUBBLE: f32 = 2.0;

const TA_MIN: f32 = 2.0;
const TA_MAX: f32 = 8.0;
const WC_MIN: f32 = 0.5;
const WC_MAX: f32 = 2.5;
const KE_MIN: f32 = 1.0;
const KE_MAX: f32 = 10.0;
const K_TA: f32 = 300.0;
const K_WC: f32 = 500.0;
const K_BUOYANCY: f32 = 2.0;
const K_DRAG: f32 = 0.5;
const LIFE_MIN: f32 = 2.0;
const LIFE_MAX: f32 = 6.0;
const MAX_SPAWN: f32 = 8.0;
const WHITE_INTERVAL: f32 = 1.0 / 120.0;
const SPRAY_NEIGHBORS: u32 = 6u;
const BUBBLE_NEIGHBORS: u32 = 20u;
const SPRAY_CHECK: f32 = 12.0;
const SPHERE_SHARE: f32 = 4.18879 / 8.0;
const NORMAL_VALID: f32 = 0.5;

fn white_hash(n: u32) -> u32 {
    let s = n * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

fn white_random(seed: ptr<function, u32>) -> f32 {
    *seed = white_hash(*seed);
    return f32(*seed >> 8u) / 16777216.0;
}

fn potential(value: f32, lo: f32, hi: f32) -> f32 {
    return (min(value, hi) - min(value, lo)) / (hi - lo);
}

fn white_dt() -> f32 {
    return max(params.frame_dt - state.remaining, 0.0);
}

fn phi_cell(c: vec3<i32>) -> f32 {
    return phi[cell_index(clamp(c, vec3<i32>(0), vec3<i32>(params.dims.xyz) - 1))];
}

fn phi_at(x: vec3<f32>) -> f32 {
    let hi = vec3<f32>(params.dims.xyz) - 1.0;
    let g = clamp(x * params.inv_dx - 0.5, vec3<f32>(0.0), hi);
    let base = min(floor(g), max(hi - 1.0, vec3<f32>(0.0)));
    let t = g - base;
    let b = vec3<i32>(base);
    var acc = 0.0;

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let pick = vec3<i32>(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let w = select(1.0 - t, t, pick == vec3<i32>(1));
        acc = acc + w.x * w.y * w.z * phi_cell(b + pick);
    }

    return acc;
}

fn unit_or_zero(g: vec3<f32>) -> vec3<f32> {
    let l = length(g);

    if l < NORMAL_VALID {
        return vec3<f32>(0.0);
    }

    return g / l;
}

fn phi_gradient(c: vec3<i32>) -> vec3<f32> {
    let k = clamp(c, vec3<i32>(0), vec3<i32>(params.dims.xyz) - 1);
    return ellipsoids[6u * params.particle_capacity + grid_index(k)].xyz;
}

fn surface_normal(x: vec3<f32>) -> vec3<f32> {
    let hi = vec3<f32>(params.dims.xyz) - 1.0;
    let g = clamp(x * params.inv_dx - 0.5, vec3<f32>(0.0), hi);
    let base = min(floor(g), max(hi - 1.0, vec3<f32>(0.0)));
    let t = g - base;
    let b = vec3<i32>(base);
    var acc = vec3<f32>(0.0);

    for (var corner = 0; corner < 8; corner = corner + 1) {
        let pick = vec3<i32>(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let w = select(1.0 - t, t, pick == vec3<i32>(1));
        acc = acc + w.x * w.y * w.z * phi_gradient(b + pick);
    }

    return unit_or_zero(acc);
}

fn cell_normal(c: vec3<i32>) -> vec3<f32> {
    return unit_or_zero(phi_gradient(c));
}

fn white_gradient(n: u32) {
    let c = grid_coords(n);
    let ex = vec3<i32>(1, 0, 0);
    let ey = vec3<i32>(0, 1, 0);
    let ez = vec3<i32>(0, 0, 1);
    let g = vec3<f32>(
        phi_cell(c + ex) - phi_cell(c - ex),
        phi_cell(c + ey) - phi_cell(c - ey),
        phi_cell(c + ez) - phi_cell(c - ez),
    );

    ellipsoids[6u * params.particle_capacity + n] = vec4<f32>(g * (0.5 * params.inv_dx), 0.0);
}

fn basis(n: vec3<f32>) -> mat2x3<f32> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;

    return mat2x3<f32>(
        vec3<f32>(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3<f32>(b, s + n.y * n.y * a, -n.y),
    );
}

fn push_out_of_bodies(x_in: vec3<f32>) -> vec3<f32> {
    var x = x_in;

    for (var k = 0u; k < params.body_count; k = k + 1u) {
        let b = bodies[k];
        let s = body_sdf(b, x);

        if s < params.margin {
            x = x + sdf_gradient(b, x) * (params.margin - s);
        }
    }

    return x;
}

fn reach_overlap(o: i32, f: vec3<f32>) -> vec3<f32> {
    if o < 0 {
        return 1.0 - f;
    }

    if o > 0 {
        return f;
    }

    return vec3<f32>(1.0);
}

fn is_close(p: vec3<f32>, x: vec3<f32>, h2: f32) -> u32 {
    let d = p - x;
    return u32(dot(d, d) < h2);
}

fn close_neighbors(x: vec3<f32>, c: vec3<i32>, lo: vec3<i32>, hi: vec3<i32>) -> u32 {
    let h2 = params.dx * params.dx;
    var count = 0u;

    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            let rest = h2 - gap_squared(f32(cy) * params.dx, x.y) - gap_squared(f32(cz) * params.dx, x.z);

            if rest <= 0.0 {
                continue;
            }

            var x0 = lo.x;
            var x1 = hi.x;

            if x0 < c.x && gap_squared(f32(x0) * params.dx, x.x) >= rest {
                x0 = x0 + 1;
            }

            if x1 > c.x && gap_squared(f32(x1) * params.dx, x.x) >= rest {
                x1 = x1 - 1;
            }

            let span = row_particles(cy, cz, x0, x1);
            var j = span.x;

            while j + 4u <= span.y {
                let a = packed_position(j);
                let b = packed_position(j + 1u);
                let e = packed_position(j + 2u);
                let g = packed_position(j + 3u);
                count = count + is_close(a, x, h2) + is_close(b, x, h2) + is_close(e, x, h2) + is_close(g, x, h2);
                j = j + 4u;
            }

            while j < span.y {
                count = count + is_close(packed_position(j), x, h2);
                j = j + 1u;
            }

            if count >= SPRAY_NEIGHBORS {
                return count;
            }
        }
    }

    return count;
}

fn white_kind(x: vec3<f32>) -> f32 {
    let h2 = params.dx * params.dx;
    let c = cell_of(x);
    let f = clamp(x * params.inv_dx - vec3<f32>(c), vec3<f32>(0.0), vec3<f32>(1.0));
    let lo = max(c - vec3<i32>(1), vec3<i32>(0));
    let hi = min(c + vec3<i32>(1), vec3<i32>(params.dims.xyz) - 1);
    var bound = 0u;
    var expected = 0.0;

    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            let rest = h2 - gap_squared(f32(cy) * params.dx, x.y) - gap_squared(f32(cz) * params.dx, x.z);
            let wyz = reach_overlap(cy - c.y, f).y * reach_overlap(cz - c.z, f).z;

            for (var cx = lo.x; cx <= hi.x; cx = cx + 1) {
                let cell = grid_index(vec3<i32>(cx, cy, cz));
                let n = cell_start[cell + 1u] - cell_start[cell];
                expected = expected + wyz * reach_overlap(cx - c.x, f).x * f32(n);

                if gap_squared(f32(cx) * params.dx, x.x) < rest {
                    bound = bound + n;
                }
            }
        }
    }

    expected = expected * SPHERE_SHARE;

    if bound < SPRAY_NEIGHBORS {
        return WHITE_SPRAY;
    }

    if expected > f32(BUBBLE_NEIGHBORS) {
        return WHITE_BUBBLE;
    }

    if expected < SPRAY_CHECK && close_neighbors(x, c, lo, hi) < SPRAY_NEIGHBORS {
        return WHITE_SPRAY;
    }

    return WHITE_FOAM;
}

fn advect_white(i: u32) {
    let dt = white_dt();
    let g = params.gravity.xyz;
    var p = white[i];
    var x = p.x.xyz;
    var v = p.v.xyz;
    var life = p.x.w;
    let kind = white_kind(x);

    if kind == WHITE_SPRAY {
        x = x + dt * v + 0.5 * dt * dt * g;
        v = v + dt * g;
    } else if kind == WHITE_FOAM {
        let mid = x + 0.5 * dt * sample_velocity(x);
        v = sample_velocity(mid);
        x = x + dt * v;
        life = life - dt;
    } else {
        let drag = 1.0 - pow(1.0 - K_DRAG, dt / WHITE_INTERVAL);
        v = v - dt * K_BUOYANCY * g + drag * (sample_velocity(x) - v);
        x = x + dt * v;
    }

    x = push_out_of_bodies(x);

    let size = vec3<f32>(params.dims.xyz) * params.dx;
    let inside = clamp(x, vec3<f32>(params.margin), size - params.margin);
    let hit = inside != x;

    if (kind == WHITE_SPRAY && any(hit)) || life <= 0.0 || any(x != x) {
        return;
    }

    v = select(v, vec3<f32>(0.0), hit);
    p.x = vec4<f32>(inside, life);
    p.v = vec4<f32>(v, kind);
    let slot = atomicAdd(&white_state.next, 1u);

    if slot < arrayLength(&white_out) {
        white_out[slot] = p;
    }
}

struct SpawnProbe {
    x: vec3<f32>,
    v: vec3<f32>,
    n: vec3<f32>,
    crest: bool,
}

struct FluidSample {
    x: vec3<f32>,
    v: vec3<f32>,
}

fn fluid_sample(j: u32) -> FluidSample {
    return FluidSample(packed_position(j), ellipsoids[j].xyz);
}

fn spawn_potentials(p: SpawnProbe, q: FluidSample) -> vec2<f32> {
    let h = params.dx;
    let xij = p.x - q.x;
    let r = length(xij);

    if r >= h || r <= 0.0 {
        return vec2<f32>(0.0);
    }

    let w = 1.0 - r / h;
    let vij = p.v - q.v;
    let vl = length(vij);
    var out = vec2<f32>(0.0);

    if vl > 0.0 {
        out.x = vl * (1.0 - dot(vij, xij) / (vl * r)) * w;
    }

    if p.crest && dot(xij, p.n) > 0.0 {
        let nj = cell_normal(cell_of(q.x));

        if dot(nj, nj) > 0.0 {
            out.y = (1.0 - dot(p.n, nj)) * w;
        }
    }

    return out;
}

override WHITE_ADVECT: bool = false;

@compute @workgroup_size(256)
fn white_spawn(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    if WHITE_ADVECT && white_dt() > 0.0 {
        for (var k = gid.x; k < white_state.count; k = k + groups.x * WG) {
            advect_white(k);
        }
    }

    spawn_white(gid.x);
}

fn spawn_white(i: u32) {
    let capacity = arrayLength(&white_out);

    if i >= state.particle_count || atomicLoad(&white_state.next) >= capacity {
        return;
    }

    let dt = white_dt();

    if dt <= 0.0 {
        return;
    }

    let xi = packed_position(i);
    let vi = ellipsoids[i].xyz;
    let speed = length(vi);
    let energy = potential(0.5 * speed * speed, KE_MIN, KE_MAX);

    if energy <= 0.0 {
        return;
    }

    let h = params.dx;
    let ni = surface_normal(xi);
    let crest = dot(vi, ni) >= 0.6 * speed;
    let c = cell_of(xi);
    let lo = max(c - vec3<i32>(1), vec3<i32>(0));
    let hi = min(c + vec3<i32>(1), vec3<i32>(params.dims.xyz) - 1);
    var sums = vec2<f32>(0.0);

    let probe = SpawnProbe(xi, vi, ni, crest);

    for (var cz = lo.z; cz <= hi.z; cz = cz + 1) {
        let gz = axis_gap(f32(cz) * h, xi.z);

        for (var cy = lo.y; cy <= hi.y; cy = cy + 1) {
            let gy = axis_gap(f32(cy) * h, xi.y);

            if gy * gy + gz * gz > 1.01 * h * h {
                continue;
            }

            let span = row_particles(cy, cz, lo.x, hi.x);
            var j = span.x;

            while j + 4u <= span.y {
                let a = fluid_sample(j);
                let b = fluid_sample(j + 1u);
                let c4 = fluid_sample(j + 2u);
                let d = fluid_sample(j + 3u);
                sums = sums + spawn_potentials(probe, a) + spawn_potentials(probe, b) + spawn_potentials(probe, c4) + spawn_potentials(probe, d);
                j = j + 4u;
            }

            while j < span.y {
                sums = sums + spawn_potentials(probe, fluid_sample(j));
                j = j + 1u;
            }
        }
    }

    let rate = energy * (K_TA * potential(sums.x, TA_MIN, TA_MAX) + K_WC * potential(sums.y, WC_MIN, WC_MAX));
    var seed = white_hash(bitcast<u32>(particles[i].x.w) ^ white_hash(bitcast<u32>(state.time)));
    let cap = u32(max(MAX_SPAWN, ceil(MAX_SPAWN * dt / WHITE_INTERVAL - 1.0e-3)));
    let n = min(u32(floor(rate * dt + white_random(&seed))), cap);

    if n == 0u {
        return;
    }

    atomicAdd(&white_state.spawned, n);

    let base = atomicAdd(&white_state.next, n);
    let axis = vi / speed;
    let frame = basis(axis);
    let radius = params.particle_radius * params.dx;
    let size = vec3<f32>(params.dims.xyz) * params.dx;

    for (var k = 0u; k < n; k = k + 1u) {
        if base + k >= capacity {
            return;
        }

        let r = radius * sqrt(white_random(&seed));
        let theta = 6.2831853 * white_random(&seed);
        let along = white_random(&seed) * dt * speed;
        let radial = r * (cos(theta) * frame[0] + sin(theta) * frame[1]);
        let x = clamp(xi + radial + along * axis, vec3<f32>(params.margin), size - params.margin);
        let life = mix(LIFE_MIN, LIFE_MAX, white_random(&seed));

        var p: White;
        p.x = vec4<f32>(x, life);
        p.v = vec4<f32>(vi + radial, WHITE_FOAM);
        white_out[base + k] = p;
    }
}

fn write_white_slot(run: bool) {
    let base = 4u * SLOT_COUNT;
    indirect[base] = select(0u, (white_state.count + WG - 1u) / WG, run);
    indirect[base + 1u] = 1u;
    indirect[base + 2u] = 1u;
}

@compute @workgroup_size(1)
fn white_close() {
    let run = white_dt() > 0.0;

    if run {
        white_state.count = min(atomicLoad(&white_state.next), arrayLength(&white_out));
        atomicStore(&white_state.next, 0u);
    }

    write_white_slot(run);
}

@compute @workgroup_size(256)
fn white_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i < white_state.count {
        white[i] = white_out[i];
    }
}
