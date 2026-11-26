mod common;

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use common::Preset;
use liquid_gpu::prelude::*;

const ORIGIN: Vec3 = Vec3::new(-1.0, 0.0, -1.0);
const CELLS: UVec3 = UVec3::new(64, 96, 64);
const DX: f32 = 1.0 / 32.0;
const SOURCE: Vec3 = Vec3::new(0.9, 0.22, 1.02);
const SOURCE_RADIUS: f32 = 0.18;
const SOURCE_WANDER: f32 = 0.05;
const DENSITY_RATE: f32 = 6.0;
const HEAT_RATE: f32 = 650.0;
const PULSE: f32 = 0.3;
const SPHERE: Vec3 = Vec3::new(1.0, 1.3, 1.0);
const RADIUS: f32 = 0.3;
const SUN: Vec3 = Vec3::new(0.55, 0.7, 0.45);
const CAPTURES: [f32; 4] = [1.5, 3.0, 5.0, 8.0];

#[derive(Resource, Clone, Copy)]
struct Smoke {
    quality: Quality,
    sampling: GasSampling,
}

fn main() {
    let sampling = common::choice("--sampling", &[("texture", GasSampling::Texture), ("buffer", GasSampling::Buffer)]);
    let mut app = common::app(Window { title: "bevy fluid smoke".into(), ..default() });
    let quality = app.world().resource::<Preset>().0;

    app.insert_resource(ClearColor(Color::srgb(0.55, 0.66, 0.8)))
        .insert_resource(Smoke { quality, sampling })
        .insert_resource(GlobalAmbientLight {
            brightness: 500.0,
            ..default()
        })
        .add_plugins(GasPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (stir_source, reset_domain, capture_frames));

    app.run();
}

fn domain(preset: Smoke) -> GasDomain {
    let config = GasConfig {
        dims: CELLS,
        dx: DX,
        params: GasParams {
            vorticity_confinement: 4.0,
            density_decay: 0.25,
            cooling: 0.6,
            cfl: 2.0,
            open_low: [true, false, true],
            open_high: [true; 3],
            ..default()
        },
        ..default()
    }
    .with_quality(preset.quality);
    let look = GasLook {
        extinction: 24.0,
        glow: 30_000.0,
        glow_temperature: 260.0,
        sampling: preset.sampling,
        ..GasLook::default().with_quality(preset.quality)
    };

    GasDomain {
        emitters: vec![GasEmitter {
            region: GasRegion::sphere(ORIGIN + SOURCE, SOURCE_RADIUS),
            density_rate: DENSITY_RATE,
            heat_rate: HEAT_RATE,
        }],
        obstacles: vec![GasRegion::sphere(ORIGIN + SPHERE, RADIUS)],
        look,
        ..GasDomain::new(config, ORIGIN)
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    preset: Res<Smoke>,
) {
    let size = CELLS.as_vec3() * DX;
    let stone = materials.add(StandardMaterial {
        base_color: Color::srgb(0.45, 0.43, 0.41),
        perceptual_roughness: 0.9,
        ..default()
    });
    let wall = materials.add(StandardMaterial {
        base_color: Color::srgb(0.32, 0.34, 0.38),
        perceptual_roughness: 0.95,
        ..default()
    });
    let ceramic = materials.add(StandardMaterial {
        base_color: Color::srgb(0.75, 0.32, 0.18),
        perceptual_roughness: 0.4,
        ..default()
    });
    let iron = materials.add(StandardMaterial {
        base_color: Color::srgb(0.08, 0.08, 0.09),
        metallic: 0.8,
        perceptual_roughness: 0.5,
        ..default()
    });

    commands.spawn(domain(*preset));

    commands.spawn((
        Mesh3d(meshes.add(Sphere::new(RADIUS).mesh().uv(64, 32))),
        MeshMaterial3d(ceramic),
        Transform::from_translation(ORIGIN + SPHERE),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(SOURCE_RADIUS + 0.06, 0.12))),
        MeshMaterial3d(iron),
        Transform::from_translation(ORIGIN + Vec3::new(SOURCE.x, 0.06, SOURCE.z)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 0.1, 8.0))),
        MeshMaterial3d(stone),
        Transform::from_translation(ORIGIN + Vec3::new(size.x * 0.5, -0.05, size.z * 0.5)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 5.0, 0.1))),
        MeshMaterial3d(wall),
        Transform::from_translation(ORIGIN + Vec3::new(size.x * 0.5, 2.5, -0.6)),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 12_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_to(-SUN, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            first_cascade_far_bound: 5.0,
            maximum_distance: 14.0,
            ..default()
        }
        .build(),
    ));

    let target = ORIGIN + Vec3::new(size.x * 0.5, 1.3, size.z * 0.5);
    let eye = target + Vec3::new(0.35, -0.15, 4.4);

    commands.spawn((
        Camera3d::default(),
        Hdr,
        DepthPrepass,
        Msaa::Off,
        TemporalAntiAliasing::default(),
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
    ));
}

fn stir_source(time: Res<Time>, mut domains: Query<&mut GasDomain>) {
    let t = time.elapsed_secs();
    let wander = Vec3::new((1.7 * t).sin(), 0.0, (2.3 * t).cos()) * SOURCE_WANDER;
    let pulse = 1.0 + PULSE * (3.1 * t).sin() * (0.7 * t).cos();

    for mut domain in &mut domains {
        if let Some(source) = domain.emitters.first_mut() {
            source.region.center = ORIGIN + SOURCE + wander;
            source.density_rate = DENSITY_RATE * pulse;
            source.heat_rate = HEAT_RATE * pulse;
        }
    }
}

fn reset_domain(keys: Res<ButtonInput<KeyCode>>, preset: Res<Smoke>, mut commands: Commands, domains: Query<Entity, With<GasDomain>>) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }

    for entity in &domains {
        commands.entity(entity).despawn();
    }

    commands.spawn(domain(*preset));
}

fn capture_frames(time: Res<Time>, mut commands: Commands, mut taken: Local<usize>, mut exit: MessageWriter<AppExit>) {
    let Ok(path) = std::env::var("LIQUID_CAPTURE") else {
        return;
    };

    let t = time.elapsed_secs();

    if let Some(at) = CAPTURES.get(*taken).filter(|at| t > **at) {
        *taken += 1;
        let file = format!("{}-{at}s.png", std::path::Path::new(&path).with_extension("").display());
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(file));
    }

    if t > CAPTURES[CAPTURES.len() - 1] + 1.0 {
        exit.write(AppExit::Success);
    }
}

