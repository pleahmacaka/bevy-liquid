import bevy_pbr::render::mesh_functions;
import bevy_pbr::render::view_transformations::position_world_to_clip;
import liquid_ocean::water_shading::{WaterLook, shade_water};

struct HeightfieldSettings {
    size: vec4<u32>,
    cell: vec4<f32>,
    look: WaterLook,
}

@group(constants::MATERIAL_BIND_GROUP) @binding(0) var<uniform> settings: HeightfieldSettings;
@group(constants::MATERIAL_BIND_GROUP) @binding(1) var<storage, read> state: array<vec4<f32>>;
@group(constants::MATERIAL_BIND_GROUP) @binding(2) var<storage, read> bed: array<vec4<f32>>;

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) depth: f32,
    @location(3) speed: f32,
};

fn cell_of(i: i32, k: i32) -> u32 {
    let ci = clamp(i, 0, i32(settings.size.x) - 1);
    let ck = clamp(k, 0, i32(settings.size.y) - 1);
    return u32(ck) * settings.size.x + u32(ci);
}

fn surface_height(i: i32, k: i32) -> f32 {
    let c = cell_of(i, k);
    let h = state[c].x;
    let b = bed[c];
    if h <= settings.cell.z {
        return b.x - 0.05;
    }
    if h >= b.z - b.x {
        return h + b.x;
    }
    let range = b.z - b.y;
    let exponent = (b.x - b.y) / (b.z - b.x) + 1.0;
    return b.y + range * pow(h * exponent / range, 1.0 / exponent);
}

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    let i = i32(round(vertex.uv.x));
    let k = i32(round(vertex.uv.y));
    let c = cell_of(i, k);
    let u = state[c];

    let height = surface_height(i, k);
    let gx = (surface_height(i + 1, k) - surface_height(i - 1, k)) / (2.0 * settings.cell.x);
    let gz = (surface_height(i, k + 1) - surface_height(i, k - 1)) / (2.0 * settings.cell.y);
    let local = vec3<f32>(vertex.position.x, height, vertex.position.z);

    let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
    let world = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(local, 1.0));

    var out: VertexOutput;
    out.world_position = world;
    out.clip_position = position_world_to_clip(world.xyz);
    out.world_normal = normalize(vec3<f32>(-gx, 1.0, -gz));
    out.depth = u.x;

    var speed = 0.0;
    if u.x > settings.cell.z {
        speed = length(u.yz) / u.x;
    }
    out.speed = speed;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.depth <= settings.cell.z {
        discard;
    }
    let froth = smoothstep(2.5, 6.0, in.speed) * 0.6;
    return shade_water(in.clip_position, in.world_position.xyz, in.world_normal, in.depth, froth, settings.look);
}
