import bevy_pbr::{
    render::{mesh_view_bindings::view, forward_io::VertexOutput, view_transformations},
    prepass::utils,
};
@if(VIEW_TRANSMISSION_TEXTURE) import bevy_pbr::render::mesh_view_bindings::{view_transmission_texture, view_transmission_sampler};
@if(DEPTH_PREPASS) import bevy_pbr::render::mesh_view_bindings::depth_prepass_texture;
@if(DEPTH_PREPASS) import bevy_pbr::ssr::raymarch::{
    depth_ray_march_from_cs,
    depth_ray_march_march,
    depth_ray_march_new_from_depth,
    depth_ray_march_to_ws_dir,
};
import liquid_ocean::water_shading::{WaterLook, WATER_IOR, schlick_fresnel, sky_radiance};
import liquid_gpu::shaders::brick_trace::{
    settings,
    sample_field,
    surface_hit,
    surface_normal,
    box_lo,
    box_hi,
};

@group(constants::MATERIAL_BIND_GROUP) @binding(5) var<uniform> look: WaterLook;

const SSR_THICKNESS: f32 = 0.3;
const SSR_EDGE_FADE: f32 = 0.08;
const EXIT_REFINEMENTS: i32 = 4;

struct SceneHit {
    hit: bool,
    uv: vec2<f32>,
}

struct Refraction {
    background: vec3<f32>,
    path: f32,
}

struct LiquidOutput {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

fn scene_color(uv: vec2<f32>) -> vec3<f32> {
    @if(VIEW_TRANSMISSION_TEXTURE)
    return textureSampleLevel(view_transmission_texture, view_transmission_sampler, clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999)), 0.0).rgb;
    @else
    return vec3<f32>(0.0);
}

fn screen_uv(p: vec3<f32>) -> vec2<f32> {
    return view_transformations::ndc_to_uv(view_transformations::position_world_to_ndc(p).xy);
}

@if(DEPTH_PREPASS)
fn behind_scene(p: vec3<f32>) -> bool {
    let ndc = view_transformations::position_world_to_ndc(p);

    if any(abs(ndc.xy) > vec2<f32>(1.0)) {
        return false;
    }

    let frag = view_transformations::ndc_to_frag_coord(ndc.xy);
    let scene_z = view_transformations::depth_ndc_to_view_z(utils::prepass_depth(vec4<f32>(frag, 0.0, 0.0), 0u));
    let behind = scene_z - view_transformations::depth_ndc_to_view_z(ndc.z);
    return behind > 0.0 && behind < SSR_THICKNESS;
}

@if(!DEPTH_PREPASS)
fn behind_scene(p: vec3<f32>) -> bool {
    return false;
}

@if(DEPTH_PREPASS)
fn trace_scene(start: vec3<f32>, dir: vec3<f32>) -> SceneHit {
    let size = vec2<f32>(textureDimensions(depth_prepass_texture));
    var march = depth_ray_march_new_from_depth(size);
    depth_ray_march_from_cs(&march, view_transformations::position_world_to_ndc(start));
    depth_ray_march_to_ws_dir(&march, dir);
    march.linear_steps = u32(settings.march.z);
    march.bisection_steps = u32(settings.march.w);
    march.use_secant = true;
    march.depth_thickness_linear_z = SSR_THICKNESS;

    let result = depth_ray_march_march(&march);
    return SceneHit(result.hit, result.hit_uv);
}

@if(!DEPTH_PREPASS)
fn trace_scene(start: vec3<f32>, dir: vec3<f32>) -> SceneHit {
    return SceneHit(false, vec2<f32>(0.0));
}

fn reflection(surface: vec3<f32>, r: vec3<f32>) -> vec3<f32> {
    let sky = sky_radiance(r, look);

    if r.y <= 0.0 {
        return sky;
    }

    let hit = trace_scene(surface, r);

    if !hit.hit {
        return sky;
    }

    let edge = min(hit.uv, 1.0 - hit.uv);
    return mix(sky, scene_color(hit.uv), smoothstep(0.0, SSR_EDGE_FADE, min(edge.x, edge.y)));
}

fn leaves_liquid(p: vec3<f32>) -> bool {
    let left_box = any(p < box_lo()) || any(p > box_hi());
    let closed = settings.options.x < 0.5;
    return behind_scene(p) || sample_field(p) < settings.iso || p.y < box_lo().y || (left_box && closed);
}

fn refraction(surface: vec3<f32>, dir: vec3<f32>, straight_uv: vec2<f32>) -> Refraction {
    let stride = settings.march.x * settings.origin_dx.w;
    let count = i32(settings.march.y);
    var p = surface;
    var path = 0.0;

    for (var i = 0; i < count; i = i + 1) {
        if leaves_liquid(p + dir * stride) {
            var inside = 0.0;
            var outside = stride;

            for (var k = 0; k < EXIT_REFINEMENTS; k = k + 1) {
                let mid = 0.5 * (inside + outside);

                if leaves_liquid(p + dir * mid) {
                    outside = mid;
                } else {
                    inside = mid;
                }
            }

            let exit = p + dir * outside;

            if behind_scene(exit) {
                return Refraction(scene_color(screen_uv(exit)), path + outside);
            }

            let hit = trace_scene(exit, dir);
            return Refraction(scene_color(select(straight_uv, hit.uv, hit.hit)), path + outside);
        }

        p = p + dir * stride;
        path = path + stride;
    }

    return Refraction(scene_color(straight_uv), path);
}

@fragment
fn fragment(
    @if(MULTISAMPLED) @builtin(sample_index) sample_index: u32,
    mesh: VertexOutput,
) -> LiquidOutput {
    @if(!MULTISAMPLED) let sample_index = 0u;

    let eye = view.world_position;
    let dir = normalize(mesh.world_position.xyz - eye);
    var t_far = 3.0e38;

    @if(DEPTH_PREPASS) {
        let depth = utils::prepass_depth(mesh.position, sample_index);
        let ndc = view_transformations::frag_coord_to_ndc(vec4<f32>(mesh.position.xy, depth, 1.0));
        let scene = view_transformations::position_ndc_to_world(ndc);
        t_far = distance(eye, scene);
    }

    let found = surface_hit(eye, dir, t_far);

    if !found.hit {
        discard;
    }

    let surface = eye + dir * found.t;
    let n = surface_normal(surface, dir);
    let fresnel = schlick_fresnel(dot(-dir, n));
    let reflected = reflection(surface, reflect(dir, n));
    let straight_uv = view_transformations::frag_coord_to_uv(mesh.position.xy);
    let refracted = refraction(surface, refract(dir, n, 1.0 / WATER_IOR), straight_uv);
    let transmittance = exp(-look.absorption.rgb * refracted.path);

    var color = fresnel * reflected + (1.0 - fresnel) * look.scatter.rgb * (1.0 - transmittance);
    var alpha = 1.0 - (1.0 - fresnel) * dot(transmittance, vec3<f32>(1.0 / 3.0));

    @if(VIEW_TRANSMISSION_TEXTURE) {
        color = color + (1.0 - fresnel) * refracted.background * transmittance;
        alpha = 1.0;
    }

    return LiquidOutput(vec4<f32>(color, alpha), view_transformations::position_world_to_ndc(surface).z);
}
