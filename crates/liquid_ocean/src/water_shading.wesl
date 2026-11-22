import bevy_pbr::render::mesh_view_bindings::view;
@if(VIEW_TRANSMISSION_TEXTURE) import bevy_pbr::render::mesh_view_bindings::{view_transmission_texture, view_transmission_sampler};
import bevy_pbr::render::view_transformations::{frag_coord_to_uv, frag_coord_to_ndc, position_ndc_to_world};
import bevy_pbr::prepass::utils as prepass_utils;

struct WaterLook {
    absorption: vec4<f32>,
    scatter: vec4<f32>,
    sky_zenith: vec4<f32>,
    sky_horizon: vec4<f32>,
    sun: vec4<f32>,
    foam: vec4<f32>,
}

const WATER_IOR: f32 = 1.333;

fn sky_radiance(direction: vec3<f32>, look: WaterLook) -> vec3<f32> {
    let up = max(direction.y, 0.0);
    let base = mix(look.sky_horizon.rgb, look.sky_zenith.rgb, sqrt(up));
    let sun_direction = normalize(look.sun.xyz);
    let disk = pow(max(dot(direction, sun_direction), 0.0), 1200.0) * look.sun.w;
    let halo = pow(max(dot(direction, sun_direction), 0.0), 12.0) * 0.08 * look.sun.w;
    return base + vec3<f32>(disk + halo);
}

fn schlick_fresnel(cos_theta: f32) -> f32 {
    let r = (1.0 - WATER_IOR) / (1.0 + WATER_IOR);
    let r0 = r * r;
    return r0 + (1.0 - r0) * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
}

fn view_path_length(frag_coord: vec4<f32>, world: vec3<f32>, column_depth: f32, cos_view: f32) -> f32 {
    var path = column_depth / max(cos_view, 0.15);
    @if(DEPTH_PREPASS)
    {
        let scene_depth = prepass_utils::prepass_depth(frag_coord, 0u);
        if scene_depth > 0.0 {
            let scene = position_ndc_to_world(frag_coord_to_ndc(vec4<f32>(frag_coord.xy, scene_depth, 1.0)));
            path = distance(scene, world);
        } else {
            path = 1.0e4;
        }
    }
    return path;
}

fn shade_water(frag_coord: vec4<f32>, world: vec3<f32>, normal: vec3<f32>, column_depth: f32, foam: f32, look: WaterLook) -> vec4<f32> {
    let to_eye = normalize(view.world_position - world);
    var n = normalize(normal);
    if dot(n, to_eye) < 0.0 {
        n = -n;
    }

    let cos_view = max(dot(n, to_eye), 0.0);
    let fresnel = schlick_fresnel(cos_view);
    let reflection = sky_radiance(reflect(-to_eye, n), look);

    let path = view_path_length(frag_coord, world, column_depth, cos_view);
    let transmittance = exp(-look.absorption.rgb * path);
    var background = look.scatter.rgb;
    @if(VIEW_TRANSMISSION_TEXTURE)
    {
        let uv = clamp(frag_coord_to_uv(frag_coord.xy) + n.xz * 0.03 * (1.0 - fresnel), vec2<f32>(0.001), vec2<f32>(0.999));
        background = textureSampleLevel(view_transmission_texture, view_transmission_sampler, uv, 0.0).rgb;
    }
    let refracted = background * transmittance + look.scatter.rgb * (1.0 - transmittance);

    var color = fresnel * reflection + (1.0 - fresnel) * refracted;

    let sun_direction = normalize(look.sun.xyz);
    let lit_foam = look.foam.rgb * (0.35 + 0.65 * max(dot(n, sun_direction), 0.0));
    color = mix(color, lit_foam, clamp(foam, 0.0, 1.0) * look.foam.w);

    return vec4<f32>(color, 1.0);
}
