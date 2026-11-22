import bevy_pbr::render::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::{alpha_discard, apply_pbr_lighting, main_pass_post_lighting_processing},
};

struct CausticMap {
    bounds: vec4<f32>,
    info: vec4<f32>,
    texels: array<u32>,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(100) var<storage, read> caustics: CausticMap;

const EDGE_FADE: f32 = 0.2;
const PLANE_FADE: f32 = 0.05;

fn texel(c: vec2<i32>, size: vec2<i32>) -> f32 {
    let q = clamp(c, vec2<i32>(0), size - 1);
    return f32(caustics.texels[u32(q.x + q.y * size.x)]);
}

fn caustic_factor(world: vec3<f32>) -> f32 {
    let norm = caustics.info.z;

    if norm <= 0.0 {
        return 1.0;
    }

    let uv = (world.xz - caustics.bounds.xy) / caustics.bounds.zw;

    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 1.0;
    }

    let size = vec2<i32>(caustics.info.xy);
    let g = uv * caustics.info.xy - 0.5;
    let base = vec2<i32>(floor(g));
    let f = g - floor(g);

    let top = mix(texel(base, size), texel(base + vec2<i32>(1, 0), size), f.x);
    let bottom = mix(texel(base + vec2<i32>(0, 1), size), texel(base + vec2<i32>(1, 1), size), f.x);
    let value = mix(top, bottom, f.y) * norm;

    let edge = min(uv, 1.0 - uv) * caustics.bounds.zw;
    let fade = smoothstep(0.0, EDGE_FADE, min(edge.x, edge.y)) * (1.0 - smoothstep(0.0, PLANE_FADE, abs(world.y - caustics.info.w)));

    return mix(1.0, value, fade);
}

@fragment
fn fragment(in: VertexOutput, @builtin(front_facing) is_front: bool) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material.flags, pbr_input.material.alpha_cutoff, pbr_input.material.base_color);
    pbr_input.directional_shadow_factor = caustic_factor(in.world_position.xyz);

    var out: FragmentOutput;
    out.color = main_pass_post_lighting_processing(pbr_input, apply_pbr_lighting(pbr_input));

    return out;
}
