import bevy_pbr::render::mesh_functions;
import bevy_pbr::render::mesh_view_bindings::view;
import bevy_pbr::render::view_transformations::position_world_to_clip;
import liquid_ocean::water_shading::{WaterLook, shade_water};

struct OceanSettings {
    grid: vec4<u32>,
    patch_size: vec4<f32>,
    params: vec4<f32>,
    look: WaterLook,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> settings: OceanSettings;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<storage, read> surface: array<vec4<f32>>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) rest: vec2<f32>,
};

fn texel(cascade: u32, x: i32, z: i32, slot: u32) -> vec4<f32> {
    let n = i32(settings.grid.x);
    let wx = ((x % n) + n) % n;
    let wz = ((z % n) + n) % n;
    let index = cascade * settings.grid.x * settings.grid.x + u32(wz) * settings.grid.x + u32(wx);
    return surface[2u * index + slot];
}

fn sample_cascade(cascade: u32, position: vec2<f32>, slot: u32) -> vec4<f32> {
    let p = position / settings.patch_size[cascade] * f32(settings.grid.x);
    let base = floor(p);
    let t = p - base;
    let x = i32(base.x);
    let z = i32(base.y);
    let bottom = mix(texel(cascade, x, z, slot), texel(cascade, x + 1, z, slot), t.x);
    let top = mix(texel(cascade, x, z + 1, slot), texel(cascade, x + 1, z + 1, slot), t.x);
    return mix(bottom, top, t.y);
}

fn cascade_weight(cascade: u32, distance: f32) -> f32 {
    let fade = settings.params.y * settings.patch_size[cascade];
    return 1.0 - smoothstep(fade, 2.0 * fade, distance);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let anchor = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
    let rest = anchor.xz;
    let distance = length(rest - view.world_position.xz);

    var displacement = vec3<f32>(0.0);
    for (var c = 0u; c < settings.grid.y; c = c + 1u) {
        displacement = displacement + cascade_weight(c, distance) * sample_cascade(c, rest, 0u).xyz;
    }

    let world = vec3<f32>(rest.x + displacement.x, anchor.y + displacement.y, rest.y + displacement.z);

    var out: VertexOutput;
    out.world_position = vec4<f32>(world, 1.0);
    out.clip_position = position_world_to_clip(world);
    out.rest = rest;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let distance = length(in.rest - view.world_position.xz);
    var slope = vec2<f32>(0.0);
    var foam = 0.0;
    var folding = 0.0;

    for (var c = 0u; c < settings.grid.y; c = c + 1u) {
        let w = cascade_weight(c, distance);
        let detail = sample_cascade(c, in.rest, 1u);
        slope = slope + w * detail.xy;
        foam = foam + w * detail.z;
        folding = folding + w * (1.0 - sample_cascade(c, in.rest, 0u).w);
    }

    let crest = smoothstep(0.2, 0.6, folding) * 0.35;
    let normal = vec3<f32>(-slope.x, 1.0, -slope.y);
    let froth = clamp(foam * settings.params.z + crest, 0.0, 1.0);

    return shade_water(in.clip_position, in.world_position.xyz, normal, settings.params.x, froth, settings.look);
}
