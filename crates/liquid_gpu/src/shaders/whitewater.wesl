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
@if(MULTISAMPLED) @group(0) @binding(11) var reveal_texture: texture_multisampled_2d<f32>;
@if(!MULTISAMPLED) @group(0) @binding(10) var accum_texture: texture_2d<f32>;
@if(!MULTISAMPLED) @group(0) @binding(11) var reveal_texture: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) corner: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) view_depth: f32,
}

struct Accumulation {
    @location(0) accum: vec4<f32>,
    @location(1) reveal: f32,
}

const FADE: f32 = 0.6;

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
        out.color = vec4<f32>(0.0);
        return out;
    }

    let dx = origin_dx.w;
    let p = whitewater.items[i];
    let kind = p.v.w;
    var radius = 0.45 * dx;
    var color = vec4<f32>(0.92, 0.95, 1.0, clamp(p.x.w / FADE, 0.0, 1.0) * 0.8);

    if kind < 0.5 {
        radius = 0.3 * dx;
        color = vec4<f32>(1.0, 1.0, 1.0, 0.9);
    } else if kind > 1.5 {
        radius = 0.2 * dx;
        color = vec4<f32>(0.75, 0.88, 0.95, 0.35);
    }

    let center = origin_dx.xyz + p.x.xyz;
    let right = view.world_from_view[0].xyz;
    let up = view.world_from_view[1].xyz;
    let world = vec4<f32>(center + (right * out.corner.x + up * out.corner.y) * radius, 1.0);

    out.clip_position = view.clip_from_world * world;
    out.view_depth = -(view.view_from_world * world).z;
    out.color = color;
    return out;
}

fn depth_weight(z: f32, alpha: f32) -> f32 {
    let near = z / 5.0;
    let far = z / 200.0;
    return alpha * clamp(10.0 / (1.0e-5 + near * near + far * far * far * far * far * far), 1.0e-2, 3.0e3);
}

@fragment
fn accumulate(input: VertexOutput) -> Accumulation {
    let r2 = dot(input.corner, input.corner);

    if r2 >= 1.0 {
        discard;
    }

    let falloff = 1.0 - r2;
    let alpha = input.color.a * falloff * falloff;
    let w = depth_weight(input.view_depth, alpha);
    return Accumulation(vec4<f32>(input.color.rgb * alpha, alpha) * w, alpha);
}

@fragment
fn composite(
    @builtin(position) position: vec4<f32>,
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
) -> @location(0) vec4<f32> {
    let p = vec2<i32>(position.xy);

    @if(MULTISAMPLED) let s = i32(sample_index);
    @else let s = 0;

    let reveal = textureLoad(reveal_texture, p, s).r;

    if reveal > 0.999 {
        discard;
    }

    let accum = textureLoad(accum_texture, p, s);
    return vec4<f32>(accum.rgb / clamp(accum.a, 1.0e-4, 5.0e4), 1.0 - reveal);
}
