struct Params {
    dims: vec4<u32>,
    origin_dx: vec4<f32>,
    sun_iso: vec4<f32>,
    map: vec4<u32>,
}

struct CausticMap {
    header: array<vec4<f32>, 2>,
    texels: array<atomic<u32>>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> field: array<f32>;
@group(0) @binding(2) var<storage, read_write> map: CausticMap;

const IOR: f32 = 1.333;
const MIN_SLOPE_COS: f32 = 0.2;

fn cell_value(c: vec3<i32>) -> f32 {
    let d = vec3<i32>(params.dims.xyz);
    let p = clamp(c, vec3<i32>(0), d - 1);
    return field[u32(p.x + d.x * (p.y + d.y * p.z))];
}

fn sample_field(x: vec3<f32>) -> f32 {
    let g = (x - params.origin_dx.xyz) / params.origin_dx.w - 0.5;
    let b = floor(g);
    let t = g - b;
    let c = vec3<i32>(b);

    let x00 = mix(cell_value(c), cell_value(c + vec3<i32>(1, 0, 0)), t.x);
    let x10 = mix(cell_value(c + vec3<i32>(0, 1, 0)), cell_value(c + vec3<i32>(1, 1, 0)), t.x);
    let x01 = mix(cell_value(c + vec3<i32>(0, 0, 1)), cell_value(c + vec3<i32>(1, 0, 1)), t.x);
    let x11 = mix(cell_value(c + vec3<i32>(0, 1, 1)), cell_value(c + vec3<i32>(1, 1, 1)), t.x);

    return mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);
}

fn field_gradient(x: vec3<f32>) -> vec3<f32> {
    let h = params.origin_dx.w;
    return vec3<f32>(
        sample_field(x + vec3<f32>(h, 0.0, 0.0)) - sample_field(x - vec3<f32>(h, 0.0, 0.0)),
        sample_field(x + vec3<f32>(0.0, h, 0.0)) - sample_field(x - vec3<f32>(0.0, h, 0.0)),
        sample_field(x + vec3<f32>(0.0, 0.0, h)) - sample_field(x - vec3<f32>(0.0, 0.0, h)),
    );
}

fn transmission(cos_theta: f32) -> f32 {
    let r = (1.0 - IOR) / (1.0 + IOR);
    let r0 = r * r;
    return 1.0 - (r0 + (1.0 - r0) * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0));
}

fn splat(texel: vec2<f32>, weight: f32) {
    let size = vec2<i32>(params.map.xy);
    let g = texel - 0.5;
    let base = vec2<i32>(floor(g));
    let f = g - floor(g);

    for (var k = 0; k < 4; k = k + 1) {
        let corner = vec2<i32>(k & 1, k >> 1u);
        let q = base + corner;

        if any(q < vec2<i32>(0)) || any(q >= size) {
            continue;
        }

        let w = mix(1.0 - f, f, vec2<f32>(corner));
        let amount = u32(weight * w.x * w.y * f32(params.map.w) + 0.5);
        atomicAdd(&map.texels[u32(q.x + q.y * size.x)], amount);
    }
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) id: vec3<u32>) {
    let rays_per_texel = params.dims.w;
    let margin = params.map.z;
    let rays = (params.map.xy + 2u * margin) * rays_per_texel;

    if any(id.xy >= rays) {
        return;
    }

    let dx = params.origin_dx.w;
    let origin = params.origin_dx.xyz;
    let extent = vec3<f32>(params.dims.xyz) * dx;
    let texel_size = extent.xz / vec2<f32>(params.map.xy);
    let source = (vec2<f32>(id.xy) + 0.5) / f32(rays_per_texel) - f32(margin);
    let column = origin.xz + source * texel_size;
    let iso = params.sun_iso.w;

    var inside = origin.y + extent.y;
    var outside = inside;
    var wet = false;

    for (var j = i32(params.dims.y) - 1; j >= 0; j = j - 1) {
        let y = origin.y + (f32(j) + 0.5) * dx;

        if sample_field(vec3<f32>(column.x, y, column.y)) >= iso {
            inside = y;
            wet = true;
            break;
        }

        outside = y;
    }

    if !wet {
        splat(source, 1.0);
        return;
    }

    for (var i = 0; i < 6; i = i + 1) {
        let mid = 0.5 * (inside + outside);

        if sample_field(vec3<f32>(column.x, mid, column.y)) >= iso {
            inside = mid;
        } else {
            outside = mid;
        }
    }

    let surface = vec3<f32>(column.x, inside, column.y);
    let n = -normalize(field_gradient(surface) + vec3<f32>(0.0, -1.0e-6, 0.0));
    let sun = params.sun_iso.xyz;
    let cos_in = dot(n, sun);

    if cos_in <= 0.0 || n.y <= 0.0 {
        return;
    }

    let r = refract(-sun, n, 1.0 / IOR);

    if r.y > -1.0e-3 {
        return;
    }

    let floor_hit = surface + r * ((surface.y - origin.y) / -r.y);
    let flux = cos_in / (max(n.y, MIN_SLOPE_COS) * sun.y);
    let weight = flux * transmission(cos_in) / transmission(sun.y);

    splat((floor_hit.xz - origin.xz) / texel_size, weight);
}
