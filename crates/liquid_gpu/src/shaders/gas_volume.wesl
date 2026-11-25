import bevy_pbr::{
    render::{mesh_view_bindings::{view, lights}, forward_io::VertexOutput, view_transformations},
    prepass::utils,
};

struct GasVolumeSettings {
    dims: vec4<u32>,
    origin_dx: vec4<f32>,
    march: vec4<f32>,
    medium: vec4<f32>,
    smoke: vec4<f32>,
    glow: vec4<f32>,
    sun: vec4<f32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var volume: texture_3d<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var volume_sampler: sampler;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var<storage, read> density: array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(3) var<storage, read> temperature: array<f32>;
@group(constants::MATERIAL_BIND_GROUP) @binding(4) var<uniform> settings: GasVolumeSettings;
@group(constants::MATERIAL_BIND_GROUP) @binding(5) var occupancy: texture_3d<f32>;

const PI: f32 = 3.14159265;
const MIN_TRANSMITTANCE: f32 = 0.004;
const MIN_SAMPLE_DEPTH: f32 = 1.0e-3;
const MAX_SAMPLE_DEPTH: f32 = 0.35;
const MIN_SPACING_FRACTION: f32 = 0.25;
const EMPTY_STRIDE: f32 = 2.0;
const SKIP_REACH_CELLS: f32 = 0.5;
const FADE_CELLS: f32 = 4.0;
const BACK_LOBE: f32 = -0.3;
const BACK_WEIGHT: f32 = 0.25;

fn box_size() -> vec3<f32> {
    return vec3<f32>(settings.dims.xyz) * settings.origin_dx.w;
}

fn buffer_index(c: vec3<i32>) -> u32 {
    let d = vec3<i32>(settings.dims.xyz);
    let p = clamp(c, vec3<i32>(0), d - 1);
    return u32(p.x + d.x * (p.y + d.y * p.z));
}

fn buffer_corner(c: vec3<i32>) -> vec2<f32> {
    let n = buffer_index(c);
    return vec2<f32>(density[n], temperature[n]);
}

fn sample_buffer(x: vec3<f32>) -> vec2<f32> {
    let g = (x - settings.origin_dx.xyz) / settings.origin_dx.w - 0.5;
    let b = floor(g);
    let t = g - b;
    let c = vec3<i32>(b);
    let x00 = mix(buffer_corner(c), buffer_corner(c + vec3<i32>(1, 0, 0)), t.x);
    let x10 = mix(buffer_corner(c + vec3<i32>(0, 1, 0)), buffer_corner(c + vec3<i32>(1, 1, 0)), t.x);
    let x01 = mix(buffer_corner(c + vec3<i32>(0, 0, 1)), buffer_corner(c + vec3<i32>(1, 0, 1)), t.x);
    let x11 = mix(buffer_corner(c + vec3<i32>(0, 1, 1)), buffer_corner(c + vec3<i32>(1, 1, 1)), t.x);
    let gas = mix(mix(x00, x10, t.y), mix(x01, x11, t.y), t.z);

    return vec2<f32>(gas.x, gas.y - settings.smoke.w);
}

fn edge_fade(x: vec3<f32>) -> f32 {
    let g = (x - settings.origin_dx.xyz) / settings.origin_dx.w;
    let d = min(g, vec3<f32>(settings.dims.xyz) - g);
    return smoothstep(0.0, FADE_CELLS, min(d.x, min(d.y, d.z)));
}

fn sample_gas(x: vec3<f32>) -> vec4<f32> {
    if settings.dims.w == 1u {
        let gas = sample_buffer(x);

        return vec4<f32>(gas.x * edge_fade(x), gas.y, 0.0, 0.0);
    }

    return textureSampleLevel(volume, volume_sampler, (x - settings.origin_dx.xyz) / box_size(), 0.0);
}

fn extinction(x: vec3<f32>) -> f32 {
    return max(sample_gas(x).x, 0.0) * settings.medium.x;
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1.0e-4), 1.5));
}

fn lobes(cos_theta: f32) -> f32 {
    return mix(henyey_greenstein(cos_theta, settings.medium.z), henyey_greenstein(cos_theta, BACK_LOBE), BACK_WEIGHT);
}

fn optical_depth(x: vec3<f32>, dir: vec3<f32>, samples: i32) -> f32 {
    var span = settings.march.w * settings.origin_dx.w;
    var s = 0.0;
    var depth = 0.0;

    for (var i = 0; i < samples; i = i + 1) {
        depth = depth + extinction(x + dir * (s + 0.5 * span)) * span;
        s = s + span;
        span = 2.0 * span;
    }

    return depth;
}

fn light_reach(x: vec3<f32>, gas: vec4<f32>, to_sun: vec3<f32>, samples: i32) -> vec2<f32> {
    if settings.dims.w == 1u {
        return exp(-vec2<f32>(optical_depth(x, to_sun, samples), optical_depth(x, vec3<f32>(0.0, 1.0, 0.0), 1)));
    }

    return gas.zw;
}

fn glow_color(excess: f32) -> vec3<f32> {
    let heat = clamp(excess / settings.glow.y, 0.0, 1.0);
    let tint = mix(vec3<f32>(1.0, 0.18, 0.02), vec3<f32>(1.0, 0.62, 0.25), heat);
    return tint * heat * heat * heat * heat * settings.glow.x;
}

fn scene_distance(position: vec4<f32>, sample_index: u32, eye: vec3<f32>) -> f32 {
    @if(DEPTH_PREPASS) {
        let depth = utils::prepass_depth(position, sample_index);

        if depth > 0.0 {
            let ndc = view_transformations::frag_coord_to_ndc(vec4<f32>(position.xy, depth, 1.0));
            return distance(eye, view_transformations::position_ndc_to_world(ndc));
        }
    }

    return 3.0e38;
}

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    mesh: VertexOutput,
) -> @location(0) vec4<f32> {
    @if(!MULTISAMPLED) let sample_index = 0u;

    let eye = view.world_position;
    let dir = normalize(mesh.world_position.xyz - eye);
    let inv = 1.0 / dir;
    let lo = (settings.origin_dx.xyz - eye) * inv;
    let hi = (settings.origin_dx.xyz + box_size() - eye) * inv;
    let near = min(lo, hi);
    let far = max(lo, hi);
    let t_enter = max(max(near.x, max(near.y, near.z)), 0.0);
    let t_exit = min(min(far.x, min(far.y, far.z)), scene_distance(mesh.position, sample_index, eye));

    if t_exit <= t_enter {
        discard;
    }

    var to_sun = vec3<f32>(0.0, 1.0, 0.0);
    var sun = vec3<f32>(0.0);

    var brightest = -1.0;

    for (var k = 0u; k < lights.n_directional_lights; k = k + 1u) {
        let light = lights.directional_lights[k];
        let power = dot(light.color.rgb, vec3<f32>(0.2126, 0.7152, 0.0722));

        if power > brightest {
            brightest = power;
            to_sun = light.direction_to_light;
            sun = light.color.rgb;
        }
    }

    let spacing = settings.march.x * settings.origin_dx.w;
    let max_samples = i32(settings.march.y);
    let albedo = settings.medium.y;
    let scatter = lobes(dot(dir, to_sun));
    let ambient = lights.ambient_color.rgb * settings.medium.w;
    let light_samples = i32(settings.march.z);
    let textured = settings.dims.w == 0u;
    let block_cells = settings.glow.z;
    let last_block = vec3<i32>(textureDimensions(occupancy)) - 1;

    var t = t_enter;
    var span = spacing;
    var block_end = t_enter;
    var transmittance = 1.0;
    var radiance = vec3<f32>(0.0);
    var skipped = false;
    var cautious = false;

    for (var i = 0; i < max_samples; i = i + 1) {
        if t >= t_exit || transmittance < MIN_TRANSMITTANCE {
            break;
        }

        if textured && t >= block_end {
            let g = (eye + dir * t - settings.origin_dx.xyz) / settings.origin_dx.w;
            let block = floor(g / block_cells);
            let wall = (block + step(vec3<f32>(0.0), dir)) * block_cells;
            let exits = (wall - g) * inv * settings.origin_dx.w;
            let peak = textureLoad(occupancy, clamp(vec3<i32>(block), vec3<i32>(0), last_block), 0).x;

            block_end = t + min(exits.x, min(exits.y, exits.z));

            if peak * settings.medium.x * spacing < MIN_SAMPLE_DEPTH {
                let steps = (block_end - t_enter) / spacing;
                let next = t_enter + select(floor(steps), ceil(steps), settings.march.x <= SKIP_REACH_CELLS) * spacing;

                if next > t {
                    t = next;
                    span = spacing;
                    skipped = false;
                    continue;
                }
            }
        }

        let width = min(span, t_exit - t);
        let x = eye + dir * (t + 0.5 * width);
        let gas = sample_gas(x);
        let sigma = max(gas.x, 0.0) * settings.medium.x;
        let empty = sigma * spacing < MIN_SAMPLE_DEPTH;

        if skipped && !empty {
            t = t - spacing * (EMPTY_STRIDE - 1.0);
            span = spacing;
            skipped = false;
            cautious = true;
            continue;
        }

        if sigma * width > MIN_SAMPLE_DEPTH {
            let reach = light_reach(x, gas, to_sun, light_samples);
            let lit = sun * scatter * reach.x + ambient * reach.y;
            let source = albedo * sigma * lit * settings.smoke.rgb + gas.x * glow_color(gas.y);
            let through = exp(-sigma * width);

            radiance = radiance + transmittance * source * (1.0 - through) / sigma;
            transmittance = transmittance * through;
        }

        let stride = select(EMPTY_STRIDE, 1.0, cautious);
        skipped = empty && !cautious;
        cautious = cautious && empty;
        t = t + width + select(0.0, spacing * (stride - 1.0), empty);
        span = clamp(MAX_SAMPLE_DEPTH / max(sigma, 1.0e-6), spacing * MIN_SPACING_FRACTION, spacing);
    }

    let alpha = 1.0 - transmittance;

    if alpha <= 0.0 {
        discard;
    }

    return vec4<f32>(radiance * view.exposure, alpha);
}
