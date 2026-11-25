struct GasVolumeSettings {
    dims: vec4<u32>,
    origin_dx: vec4<f32>,
    march: vec4<f32>,
    medium: vec4<f32>,
    smoke: vec4<f32>,
    glow: vec4<f32>,
    sun: vec4<f32>,
}

@group(0) @binding(0) var source: texture_3d<f32>;
@group(0) @binding(1) var source_sampler: sampler;
@group(0) @binding(2) var<uniform> settings: GasVolumeSettings;
@group(0) @binding(3) var result: texture_storage_3d<rgba16float, write>;

const FADE_CELLS: f32 = 4.0;
const TAP_TEXELS: f32 = 8.0;

fn edge_fade(g: vec3<f32>, inset: f32) -> f32 {
    let d = min(g, vec3<f32>(settings.dims.xyz) - g);
    return smoothstep(0.0, FADE_CELLS, min(d.x, min(d.y, d.z)) - inset);
}

fn density(g: vec3<f32>) -> f32 {
    let gas = textureSampleLevel(source, source_sampler, g / vec3<f32>(settings.dims.xyz), 0.0).x;
    return max(gas, 0.0) * edge_fade(g, 0.0);
}

fn optical_depth(g: vec3<f32>, dir: vec3<f32>, samples: i32) -> f32 {
    var span = settings.march.w;
    var s = 0.0;
    var depth = 0.0;

    for (var i = 0; i < samples; i = i + 1) {
        depth = depth + density(g + dir * (s + 0.5 * span)) * span;
        s = s + span;
        span = 2.0 * span;
    }

    return depth * settings.medium.x * settings.origin_dx.w;
}

@compute @workgroup_size(4, 4, 4)
fn gas_light(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= settings.dims.xyz) {
        return;
    }

    let g = vec3<f32>(id) + 0.5;
    let gas = textureLoad(source, vec3<i32>(id), 0).xy;
    let sun = optical_depth(g, settings.sun.xyz, i32(settings.march.z));
    let sky = optical_depth(g, vec3<f32>(0.0, 1.0, 0.0), 1);

    // boundary texels must be exactly zero and no texel negative: box faces and gas_occupancy rely on it
    textureStore(result, vec3<i32>(id), vec4<f32>(max(gas.x, 0.0) * edge_fade(g, 0.5), gas.y, exp(-sun), exp(-sky)));
}

@compute @workgroup_size(4, 4, 4)
fn gas_occupancy(@builtin(global_invocation_id) id: vec3<u32>) {
    if any(id >= textureDimensions(result)) {
        return;
    }

    let first = vec3<f32>(id) * settings.glow.z;
    let taps = (i32(settings.glow.z) + 1) / 2;
    var peak = 0.0;

    for (var z = 0; z <= taps; z = z + 1) {
        for (var y = 0; y <= taps; y = y + 1) {
            for (var x = 0; x <= taps; x = x + 1) {
                let corner = first + 2.0 * vec3<f32>(vec3<i32>(x, y, z));
                peak = max(peak, textureSampleLevel(source, source_sampler, corner / vec3<f32>(settings.dims.xyz), 0.0).x);
            }
        }
    }

    textureStore(result, vec3<i32>(id), vec4<f32>(peak * TAP_TEXELS, 0.0, 0.0, 0.0));
}
