import bevy_pbr::render::{mesh_functions, mesh_view_bindings::view, view_transformations};
@if(PREPASS_PIPELINE && MOTION_VECTOR_PREPASS) import bevy_pbr::prepass::bindings::previous_view_uniforms;
import liquid_gpu::shaders::brick_trace::{surface_hit, surface_normal, sample_displacement};

struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
}

struct MotionVertex {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec4<f32>,
}

@if(PREPASS_PIPELINE)
struct MotionOutput {
    @if(NORMAL_PREPASS) @location(0) normal: vec4<f32>,
    @if(MOTION_VECTOR_PREPASS) @location(1) motion_vector: vec2<f32>,
    @builtin(frag_depth) depth: f32,
}

@vertex
fn vertex(vertex: Vertex) -> MotionVertex {
    var out = MotionVertex(vec4<f32>(2.0, 2.0, 2.0, 1.0), vec4<f32>(0.0));

    @if(PREPASS_PIPELINE && MOTION_VECTOR_PREPASS) {
        let world_from_local = mesh_functions::get_world_from_local(vertex.instance_index);
        out.world_position = mesh_functions::mesh_position_local_to_world(world_from_local, vec4<f32>(vertex.position, 1.0));
        out.position = view_transformations::position_world_to_clip(out.world_position.xyz);
    }

    return out;
}

@if(PREPASS_PIPELINE)
@fragment
fn fragment(input: MotionVertex) -> MotionOutput {
    var out: MotionOutput;
    let eye = view.world_position;
    let dir = normalize(input.world_position.xyz - eye);
    let found = surface_hit(eye, dir, 3.0e38);

    if !found.hit {
        discard;
    }

    let surface = eye + dir * found.t;
    out.depth = view_transformations::position_world_to_ndc(surface).z;

    @if(NORMAL_PREPASS)
    out.normal = vec4<f32>(surface_normal(surface, dir) * 0.5 + vec3<f32>(0.5), 1.0);

    @if(MOTION_VECTOR_PREPASS) {
        let clip = view.unjittered_clip_from_world * vec4<f32>(surface, 1.0);
        let previous = previous_view_uniforms.unjittered_clip_from_world * vec4<f32>(surface - sample_displacement(surface), 1.0);
        out.motion_vector = (clip.xy / clip.w - previous.xy / previous.w) * vec2<f32>(0.5, -0.5);
    }

    return out;
}

@if(!PREPASS_PIPELINE)
@fragment
fn fragment(input: MotionVertex) -> @location(0) vec4<f32> {
    discard;
    return vec4<f32>(0.0);
}
