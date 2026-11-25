struct LiquidSurfaceSettings {
    dims: vec4<u32>,
    origin_dx: vec4<f32>,
    options: vec4<f32>,
    iso: f32,
    band: f32,
    march: vec4<f32>,
}

struct SurfaceHit {
    hit: bool,
    t: f32,
    samples: u32,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<storage, read> field: array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<uniform> settings: LiquidSurfaceSettings;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var<storage, read> distances: array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var<storage, read> bricks: array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var<storage, read> motion: array<vec4<f32>>;

const BRICK: i32 = 8;
const MAX_TRACE: i32 = 512;
const STRIDE_SCALE: f32 = 0.9;
const MIN_STRIDE: f32 = 0.05;
const NEAR: f32 = 1.5;
const CORNER_SLACK: f32 = 0.87;
const BISECTIONS: i32 = 7;

struct Cube {
    base: vec3<i32>,
    t: vec3<f32>,
}

fn cube(x: vec3<f32>) -> Cube {
    let g = (x - settings.origin_dx.xyz) / settings.origin_dx.w - 0.5;
    let b = floor(g);
    return Cube(vec3<i32>(b), g - b);
}

fn cell_index(c: vec3<i32>) -> u32 {
    let d = vec3<i32>(settings.dims.xyz);
    let p = clamp(c, vec3<i32>(0), d - 1);
    return u32(p.x + d.x * (p.y + d.y * p.z));
}

fn trilinear(c000: f32, c100: f32, c010: f32, c110: f32, c001: f32, c101: f32, c011: f32, c111: f32, t: vec3<f32>) -> f32 {
    let x00 = mix(c000, c100, t.x);
    let x10 = mix(c010, c110, t.x);
    let x01 = mix(c001, c101, t.x);
    let x11 = mix(c011, c111, t.x);
    return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

fn sample_field(x: vec3<f32>) -> f32 {
    let k = cube(x);
    let c = k.base;
    return trilinear(
        field[cell_index(c)],
        field[cell_index(c + vec3<i32>(1, 0, 0))],
        field[cell_index(c + vec3<i32>(0, 1, 0))],
        field[cell_index(c + vec3<i32>(1, 1, 0))],
        field[cell_index(c + vec3<i32>(0, 0, 1))],
        field[cell_index(c + vec3<i32>(1, 0, 1))],
        field[cell_index(c + vec3<i32>(0, 1, 1))],
        field[cell_index(c + vec3<i32>(1, 1, 1))],
        k.t,
    );
}

fn sample_distance(x: vec3<f32>) -> f32 {
    let k = cube(x);
    let c = k.base;
    return trilinear(
        distances[cell_index(c)],
        distances[cell_index(c + vec3<i32>(1, 0, 0))],
        distances[cell_index(c + vec3<i32>(0, 1, 0))],
        distances[cell_index(c + vec3<i32>(1, 1, 0))],
        distances[cell_index(c + vec3<i32>(0, 0, 1))],
        distances[cell_index(c + vec3<i32>(1, 0, 1))],
        distances[cell_index(c + vec3<i32>(0, 1, 1))],
        distances[cell_index(c + vec3<i32>(1, 1, 1))],
        k.t,
    );
}

fn sample_displacement(x: vec3<f32>) -> vec3<f32> {
    let k = cube(x);
    let c = k.base;
    let t = k.t;
    let x00 = mix(motion[cell_index(c)], motion[cell_index(c + vec3<i32>(1, 0, 0))], t.x);
    let x10 = mix(motion[cell_index(c + vec3<i32>(0, 1, 0))], motion[cell_index(c + vec3<i32>(1, 1, 0))], t.x);
    let x01 = mix(motion[cell_index(c + vec3<i32>(0, 0, 1))], motion[cell_index(c + vec3<i32>(1, 0, 1))], t.x);
    let x11 = mix(motion[cell_index(c + vec3<i32>(0, 1, 1))], motion[cell_index(c + vec3<i32>(1, 1, 1))], t.x);
    let m = mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
    return select(vec3<f32>(0.0), m.xyz / m.w, m.w > 1.0e-6);
}

fn spline_weights(t: f32) -> vec4<f32> {
    let u = 1.0 - t;
    return vec4<f32>(u * u * u, 3.0 * t * t * t - 6.0 * t * t + 4.0, -3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0, t * t * t) / 6.0;
}

fn spline_slopes(t: f32) -> vec4<f32> {
    let u = 1.0 - t;
    return vec4<f32>(-u * u, 3.0 * t * t - 4.0 * t, -3.0 * t * t + 2.0 * t + 1.0, t * t) * 0.5;
}

fn field_gradient(x: vec3<f32>) -> vec3<f32> {
    let k = cube(x);
    let wx = spline_weights(k.t.x);
    let wy = spline_weights(k.t.y);
    let wz = spline_weights(k.t.z);
    let sx = spline_slopes(k.t.x);
    let sy = spline_slopes(k.t.y);
    let sz = spline_slopes(k.t.z);
    var sum = vec3<f32>(0.0);

    for (var l = 0; l < 4; l = l + 1) {
        for (var j = 0; j < 4; j = j + 1) {
            for (var i = 0; i < 4; i = i + 1) {
                let value = field[cell_index(k.base + vec3<i32>(i, j, l) - 1)];
                sum = sum + value * vec3<f32>(sx[i] * wy[j] * wz[l], wx[i] * sy[j] * wz[l], wx[i] * wy[j] * sz[l]);
            }
        }
    }

    return sum;
}

fn surface_normal(x: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let n = -normalize(field_gradient(x) + vec3<f32>(0.0, 1.0e-6, 0.0));
    return select(n, -n, dot(n, dir) > 0.0);
}

fn box_lo() -> vec3<f32> {
    return settings.origin_dx.xyz;
}

fn box_hi() -> vec3<f32> {
    return settings.origin_dx.xyz + vec3<f32>(settings.dims.xyz) * settings.origin_dx.w;
}

fn box_interval(eye: vec3<f32>, dir: vec3<f32>) -> vec2<f32> {
    let inv = 1.0 / dir;
    let a = (box_lo() - eye) * inv;
    let b = (box_hi() - eye) * inv;
    let t0 = max(max(min(a.x, b.x), min(a.y, b.y)), min(a.z, b.z));
    let t1 = min(min(max(a.x, b.x), max(a.y, b.y)), max(a.z, b.z));
    return vec2<f32>(max(t0, 0.0), t1);
}

fn brick_count() -> vec3<i32> {
    return (vec3<i32>(settings.dims.xyz) + BRICK - 1) / BRICK;
}

fn brick_at(x: vec3<f32>) -> vec3<i32> {
    let g = (x - settings.origin_dx.xyz) / (settings.origin_dx.w * f32(BRICK));
    return clamp(vec3<i32>(floor(g)), vec3<i32>(0), brick_count() - 1);
}

fn brick_distance(b: vec3<i32>) -> f32 {
    let n = brick_count();
    return bricks[u32(b.x + n.x * (b.y + n.y * b.z))];
}

fn brick_exit(eye: vec3<f32>, dir: vec3<f32>, b: vec3<i32>) -> f32 {
    let size = settings.origin_dx.w * f32(BRICK);
    let lo = settings.origin_dx.xyz + vec3<f32>(b) * size;
    let toward = select(vec3<f32>(-1.0e-9), vec3<f32>(1.0e-9), dir >= vec3<f32>(0.0));
    let safe = select(dir, toward, abs(dir) < vec3<f32>(1.0e-9));
    let exits = (select(lo, lo + size, safe > vec3<f32>(0.0)) - eye) / safe;
    return min(exits.x, min(exits.y, exits.z));
}

fn trace_surface(eye: vec3<f32>, dir: vec3<f32>, t_near: f32, t_far: f32) -> SurfaceHit {
    let dx = settings.origin_dx.w;
    var out = SurfaceHit(false, t_far, 0u);
    var lo = t_near;
    var t = t_near;
    var crossed = false;

    for (var i = 0; i < MAX_TRACE; i = i + 1) {
        if t > t_far {
            if lo >= t_far {
                break;
            }

            t = t_far;
        }

        let p = eye + dir * t;
        let b = brick_at(p);
        out.samples = out.samples + 1u;

        if brick_distance(b) > 0.0 {
            lo = max(brick_exit(eye, dir, b), t);
            t = lo + 1.0e-3 * dx;
            continue;
        }

        let d = sample_distance(p);
        out.samples = out.samples + 8u;

        if d < NEAR * dx {
            out.samples = out.samples + 8u;

            if sample_field(p) >= settings.iso {
                crossed = true;
                break;
            }
        }

        lo = t;
        let reach = select(d * STRIDE_SCALE, d - CORNER_SLACK * dx, d >= settings.band);
        t = t + max(reach, MIN_STRIDE * dx);
    }

    if !crossed {
        return out;
    }

    var hi = t;

    for (var i = 0; i < BISECTIONS; i = i + 1) {
        let mid = 0.5 * (lo + hi);
        out.samples = out.samples + 8u;

        if sample_field(eye + dir * mid) >= settings.iso {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    out.hit = true;
    out.t = hi;
    return out;
}

fn surface_hit(eye: vec3<f32>, dir: vec3<f32>, t_limit: f32) -> SurfaceHit {
    let span = box_interval(eye, dir);
    var found = trace_surface(eye, dir, span.x, min(span.y, t_limit));

    if found.hit && settings.options.x > 0.5 {
        let p = eye + dir * found.t;
        let edge = settings.origin_dx.w;
        let lo = box_lo() + edge;
        let hi = box_hi() - edge;

        if p.x < lo.x || p.x > hi.x || p.z < lo.z || p.z > hi.z {
            found.hit = false;
        }
    }

    return found;
}
