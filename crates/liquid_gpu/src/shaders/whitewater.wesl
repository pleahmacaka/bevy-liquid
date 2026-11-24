import bevy_render::view::View;

struct White {
    x: vec4<f32>,
    v: vec4<f32>,
}

struct Whitewater {
    count: u32,
    next: u32,
    spawned: u32,
    pad: u32,
    items: array<White>,
}

@group(0) @binding(0) var<uniform> view: View;
@group(1) @binding(0) var<storage, read> whitewater: Whitewater;
@group(1) @binding(1) var<uniform> origin_dx: vec4<f32>;

@if(MULTISAMPLED) @group(0) @binding(10) var accum_texture: texture_multisampled_2d<f32>;
@if(!MULTISAMPLED) @group(0) @binding(10) var accum_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) tint: vec3<f32>,
    @location(2) opacity: f32,
    @location(3) view_depth: f32,
}

const FADE: f32 = 0.6;
const SPRAY_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const FOAM_COLOR: vec3<f32> = vec3<f32>(0.92, 0.95, 1.0);
const BUBBLE_COLOR: vec3<f32> = vec3<f32>(0.75, 0.88, 0.95);

@vertex
fn vertex(@builtin(vertex_index) corner_index: u32, @builtin(instance_index) i: u32) -> VertexOutput {
    var out: VertexOutput;
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    out.corner = corners[corner_index];

    if i >= whitewater.count {
        out.clip_position = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.tint = vec3<f32>(0.0);
        out.opacity = 0.0;
        return out;
    }

    let dx = origin_dx.w;
    let p = whitewater.items[i];
    let kind = p.v.w;
    var radius = 0.45 * dx;
    var tint = vec3<f32>(1.0, 1.0, 0.0);
    var opacity = clamp(p.x.w / FADE, 0.0, 1.0) * 0.8;

    if kind < 0.5 {
        radius = 0.3 * dx;
        tint = vec3<f32>(1.0, 0.0, 0.0);
        opacity = 0.9;
    } else if kind > 1.5 {
        radius = 0.2 * dx;
        tint = vec3<f32>(1.0, 0.0, 1.0);
        opacity = 0.35;
    }

    let center = origin_dx.xyz + p.x.xyz;
    let right = view.world_from_view[0].xyz;
    let up = view.world_from_view[1].xyz;
    let world = vec4<f32>(center + (right * out.corner.x + up * out.corner.y) * radius, 1.0);

    out.clip_position = view.clip_from_world * world;
    out.view_depth = -(view.view_from_world * world).z;
    out.tint = tint;
    out.opacity = opacity;
    return out;
}

fn depth_weight(z: f32, alpha: f32) -> f32 {
    let near = z / 5.0;
    let far = z / 200.0;
    return alpha * clamp(10.0 / (1.0e-5 + near * near + far * far * far * far * far * far), 1.0e-2, 3.0e3);
}

@fragment
fn accumulate(input: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(input.corner, input.corner);

    if r2 >= 1.0 {
        discard;
    }

    let falloff = 1.0 - r2;
    let alpha = input.opacity * falloff * falloff;
    let w = depth_weight(input.view_depth, alpha);
    return vec4<f32>(input.tint * (alpha * w), alpha);
}

@fragment
fn composite(
    @builtin(position) position: vec4<f32>,
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);

    @if(MULTISAMPLED) let s = i32(sample_index);
    @else let s = 0;

    let accum = textureLoad(accum_texture, p, s);
    let reveal = accum.a;

    if reveal > 0.999 {
        discard;
    }

    let share = accum.yz / clamp(accum.x, 1.0e-4, 5.0e4);
    let color = SPRAY_COLOR + share.x * (FOAM_COLOR - SPRAY_COLOR) + share.y * (BUBBLE_COLOR - SPRAY_COLOR);
    return vec4<f32>(color, 1.0 - reveal);
}
