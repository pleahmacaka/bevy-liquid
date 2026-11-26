mod common;

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight, NotShadowCaster};
use bevy::pbr::ScreenSpaceTransmission;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use common::Preset;
use liquid_gpu::prelude::*;

const POT: Vec3 = Vec3::new(0.5, 0.4375, 0.5);
const POT_ORIGIN: Vec3 = Vec3::new(-0.25, 0.06, -0.25);
const POT_CELLS: UVec3 = UVec3::new(32, 28, 32);
const POT_DX: f32 = 1.0 / 64.0;
const WALL: f32 = 0.02;
const FILL: f32 = 0.75;
const GAS_ORIGIN: Vec3 = Vec3::new(-1.0, 0.06, -1.0);
const GAS_CELLS: UVec3 = UVec3::new(64, 88, 64);
const GAS_DX: f32 = 1.0 / 32.0;
const PLATE_TEMPERATURE: f32 = 750.0;
const PLATE_CONDUCTANCE: f32 = 0.008;
const PLATE_DEPTH: f32 = 0.045;
const SUN: Vec3 = Vec3::new(-0.4, 0.75, 0.5);
const CAPTURES: [f32; 4] = [2.0, 5.0, 10.0, 20.0];

#[derive(Resource, Clone, Copy)]
struct SteamTarget(Entity);

fn main() {
    let mut app = common::app(Window { title: "bevy fluid steam".into(), ..default() });

    app.insert_resource(ClearColor(Color::srgb(0.16, 0.17, 0.19)))
        .insert_resource(GlobalAmbientLight {
            brightness: 350.0,
            ..default()
        })
        .add_plugins(LiquidPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (reset_domain, capture_frames));

    app.run();
}

fn domain(quality: Quality, steam: Entity) -> LiquidDomain {
    let config = SolverConfig {
        dims: POT_CELLS,
        dx: POT_DX,
        max_particles: 300_000,
        cfl: 2.5,
        tolerance: 1e-3,
        cooling: 0.002,
        ..default()
    }
    .with_quality(quality);
    let mut seed = seed_box(&config, Vec3::ZERO, POT * Vec3::new(1.0, FILL, 1.0));

    for (i, p) in seed.iter_mut().enumerate() {
        p.c[0][3] = config.boiling_point + config.latent_heat * boil_progress(i as u32);
    }

    let look = WaterLook {
        absorption: Vec4::new(5.0, 1.6, 1.0, 0.0),
        scatter: Vec4::new(0.02, 0.26, 0.4, 0.0),
        sky_zenith: Vec4::new(0.7, 0.75, 0.8, 0.0),
        sky_horizon: Vec4::new(0.12, 0.28, 0.34, 0.0),
        sun: SUN.extend(60.0),
        ..default()
    };

    LiquidDomain::new(config).with_seed(seed).with_look(look).boils_into(steam)
}

fn boil_progress(id: u32) -> f32 {
    (id.wrapping_mul(2_654_435_761) >> 8) as f32 / (1u32 << 24) as f32
}

fn steam(quality: Quality) -> GasDomain {
    let config = GasConfig {
        dims: GAS_CELLS,
        dx: GAS_DX,
        params: GasParams {
            density_buoyancy: 0.0,
            vorticity_confinement: 2.0,
            density_decay: 3.2,
            cooling: 0.5,
            cfl: 2.0,
            open_low: [true, false, true],
            open_high: [true; 3],
            ..default()
        },
        injection_capacity: 4096,
        ..default()
    }
    .with_quality(quality);
    let look = GasLook {
        color: Vec3::new(0.96, 0.97, 1.0),
        extinction: 9.0,
        albedo: 0.97,
        ambient: 0.9,
        ..GasLook::default().with_quality(quality)
    };

    GasDomain { obstacles: pot_walls(0.07), look, ..GasDomain::new(config, GAS_ORIGIN) }
}

fn pot_walls(thickness: f32) -> Vec<GasRegion> {
    let center = POT_ORIGIN + POT * 0.5;
    let height = POT.y + WALL;
    let y = POT_ORIGIN.y + height * 0.5;
    let x = POT.x * 0.5 + thickness * 0.5;
    let z = POT.z * 0.5 + thickness * 0.5;
    let reach = POT.x * 0.5 + thickness;

    vec![
        GasRegion::cuboid(Vec3::new(center.x - x, y, center.z), Vec3::new(thickness * 0.5, height * 0.5, reach)),
        GasRegion::cuboid(Vec3::new(center.x + x, y, center.z), Vec3::new(thickness * 0.5, height * 0.5, reach)),
        GasRegion::cuboid(Vec3::new(center.x, y, center.z - z), Vec3::new(reach, height * 0.5, thickness * 0.5)),
        GasRegion::cuboid(Vec3::new(center.x, y, center.z + z), Vec3::new(reach, height * 0.5, thickness * 0.5)),
    ]
}

fn hot_plate() -> HeatSource {
    let center = Vec3::new(POT_ORIGIN.x + POT.x * 0.5, POT_ORIGIN.y, POT_ORIGIN.z + POT.z * 0.5);

    HeatSource {
        region: GasRegion::cuboid(center, Vec3::new(POT.x * 0.5, PLATE_DEPTH, POT.z * 0.5)),
        temperature: PLATE_TEMPERATURE,
        conductance: PLATE_CONDUCTANCE,
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    preset: Res<Preset>,
) {
    let steel = materials.add(StandardMaterial {
        base_color: Color::srgb(0.62, 0.63, 0.66),
        metallic: 0.9,
        perceptual_roughness: 0.32,
        ..default()
    });
    let glass = materials.add(StandardMaterial {
        base_color: Color::srgba(0.7, 0.85, 0.9, 0.1),
        alpha_mode: AlphaMode::Blend,
        unlit: true,
        ..default()
    });
    let glowing = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.05, 0.02),
        emissive: LinearRgba::rgb(1.6, 0.22, 0.02),
        perceptual_roughness: 0.6,
        ..default()
    });
    let stove = materials.add(StandardMaterial {
        base_color: Color::srgb(0.28, 0.28, 0.3),
        perceptual_roughness: 0.6,
        ..default()
    });
    let tiles = materials.add(StandardMaterial {
        base_color: Color::srgb(0.12, 0.13, 0.15),
        perceptual_roughness: 0.8,
        ..default()
    });

    let target = commands.spawn(steam(preset.0)).id();
    commands.insert_resource(SteamTarget(target));
    commands.spawn((domain(preset.0, target), Transform::from_translation(POT_ORIGIN), Visibility::default()));
    commands.spawn(hot_plate());

    let center = POT_ORIGIN + POT * 0.5;
    let height = POT.y + 2.0 * WALL;
    let wall_y = POT_ORIGIN.y - WALL + height * 0.5;
    let side = Cuboid::new(WALL, height, POT.z + 2.0 * WALL);
    let front = Cuboid::new(POT.x, height, WALL);
    let offsets = [
        (side, Vec3::new(-(POT.x + WALL) * 0.5, 0.0, 0.0)),
        (side, Vec3::new((POT.x + WALL) * 0.5, 0.0, 0.0)),
        (front, Vec3::new(0.0, 0.0, -(POT.z + WALL) * 0.5)),
        (front, Vec3::new(0.0, 0.0, (POT.z + WALL) * 0.5)),
    ];

    for (shape, offset) in offsets {
        commands.spawn((
            Mesh3d(meshes.add(shape)),
            MeshMaterial3d(glass.clone()),
            Transform::from_translation(Vec3::new(center.x, wall_y, center.z) + offset),
            NotShadowCaster,
        ));
    }

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(POT.x, WALL, POT.z))),
        MeshMaterial3d(steel),
        Transform::from_translation(Vec3::new(center.x, POT_ORIGIN.y - WALL * 0.5, center.z)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cylinder::new(0.36, POT_ORIGIN.y - WALL))),
        MeshMaterial3d(glowing),
        Transform::from_translation(Vec3::new(center.x, (POT_ORIGIN.y - WALL) * 0.5, center.z)),
    ));

    commands.spawn((
        PointLight {
            color: Color::srgb(1.0, 0.45, 0.15),
            intensity: 40_000.0,
            range: 2.0,
            ..default()
        },
        Transform::from_translation(Vec3::new(center.x, 0.02, center.z + 0.45)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(12.0, 0.1, 6.0))),
        MeshMaterial3d(stove),
        Transform::from_translation(Vec3::new(0.0, -0.05, 0.0)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(12.0, 6.0, 0.1))),
        MeshMaterial3d(tiles),
        Transform::from_translation(Vec3::new(0.0, 2.0, -1.2)),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 7_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_to(-SUN, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            first_cascade_far_bound: 4.0,
            maximum_distance: 10.0,
            ..default()
        }
        .build(),
    ));

    let eye = Vec3::new(0.65, 0.95, 2.1);
    let target = Vec3::new(0.0, 0.75, 0.0);

    commands.spawn((
        Camera3d::default(),
        Hdr,
        DepthPrepass,
        ScreenSpaceTransmission::default(),
        Msaa::Off,
        TemporalAntiAliasing::default(),
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
    ));
}

fn reset_domain(keys: Res<ButtonInput<KeyCode>>, preset: Res<Preset>, target: Res<SteamTarget>, mut domains: Query<&mut LiquidDomain>) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }

    for mut current in &mut domains {
        *current = domain(preset.0, target.0);
    }
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
