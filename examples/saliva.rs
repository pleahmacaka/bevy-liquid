mod common;

use std::path::Path;

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::pbr::ScreenSpaceTransmission;
use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::window::WindowResolution;
use common::Preset;
use liquid_gpu::prelude::*;

const MM: f32 = 1e-3;
const CELLS: UVec3 = UVec3::new(100, 125, 60);
const DX: f32 = 0.4 * MM;
const SUBSTEPS_PER_FRAME: u32 = 6;
const ORIGIN: Vec3 = Vec3::new(-20.0 * MM, 0.0, -12.0 * MM);

const PLATE: Vec3 = Vec3::new(15.0 * MM, 37.0 * MM, 12.0 * MM);
const PLATE_HALF: Vec3 = Vec3::new(12.0 * MM, 1.0 * MM, 7.0 * MM);
const PLATE_SLOPE: f32 = 25.0;
const GOB: Vec3 = Vec3::new(-5.0 * MM, 2.5 * MM, 0.0);
const GOB_RADIUS: f32 = 5.5 * MM;
const SPHERE: Vec3 = Vec3::new(28.0 * MM, 19.0 * MM, 12.0 * MM);
const RADIUS: f32 = 5.0 * MM;

const SURFACE_TENSION: f32 = 0.059;
const DENSITY: f32 = 1000.0;
const ADHESION: f32 = 0.03;

const VIEW_PITCH: f32 = 25.0;
const VIEW_DISTANCE: f32 = 72.0 * MM;
const SUN: Vec3 = Vec3::new(0.3, 0.8, 0.5);
const CAPTURES: [f32; 4] = [0.5, 1.5, 3.0, 6.0];
const DURATION: f32 = 8.0;
const FLUSH: f32 = 0.1;

#[derive(Clone, Copy)]
struct Level {
    name: &'static str,
    viscosity: f32,
}

const THIN: Level = Level { name: "thin", viscosity: 0.0019 };
const MEDIUM: Level = Level { name: "medium", viscosity: 0.006 };
const THICK: Level = Level { name: "thick", viscosity: 0.1 };

#[derive(Resource, Clone, Copy)]
struct Scene {
    quality: Quality,
    level: Level,
}

fn main() {
    let level = common::choice("--saliva", &[("medium", MEDIUM), ("thin", THIN), ("thick", THICK)]);
    let window = Window {
        title: format!("bevy fluid saliva {}", level.name),
        resolution: WindowResolution::new(960, 720).with_scale_factor_override(1.0),
        ..default()
    };
    let mut app = common::app(window);
    let quality = app.world().resource::<Preset>().0;

    if let Some(dir) = std::env::var_os("LIQUID_RECORD") {
        std::fs::create_dir_all(dir).expect("record folder");
    }

    app.insert_resource(ClearColor(Color::srgb(0.16, 0.18, 0.22)))
        .insert_resource(Scene { quality, level })
        .insert_resource(GlobalAmbientLight {
            brightness: 400.0,
            ..default()
        })
        .add_plugins(LiquidPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (reset_domain, capture_frames, record_frames));

    let capillary_dt = (DENSITY * DX.powi(3) / (std::f32::consts::TAU * SURFACE_TENSION)).sqrt();
    info!("slow motion at 60 fps: {:.1} times", 1.0 / (60.0 * SUBSTEPS_PER_FRAME as f32 * capillary_dt));

    app.run();
}

fn saliva(level: Level) -> LiquidProperties {
    let water = LiquidProperties::water();

    LiquidProperties {
        density: DENSITY,
        viscosity: level.viscosity,
        surface_tension: SURFACE_TENSION,
        adhesion: ADHESION,
        look: WaterLook {
            absorption: Vec4::new(70.0, 62.0, 55.0, 0.0),
            scatter: Vec4::new(0.82, 0.85, 0.88, 0.0),
            sky_zenith: Vec4::new(0.3, 0.33, 0.38, 0.0),
            sky_horizon: Vec4::new(0.75, 0.76, 0.78, 0.0),
            sun: SUN.normalize().extend(60.0),
            ..water.look
        },
        ..water
    }
}

fn domain(scene: Scene) -> LiquidDomain {
    let config = SolverConfig {
        dims: CELLS,
        dx: DX,
        max_particles: 200_000,
        cfl: 2.0,
        max_substeps: SUBSTEPS_PER_FRAME,
        tolerance: 1e-3,
        viscosity_iterations: 4,
        ..default()
    }
    .with_quality(scene.quality);
    let config = SolverConfig { dims: CELLS, dx: DX, ..config };
    let center = PLATE + plate_rotation() * GOB;
    let inside = |x: Vec3| {
        let above = (plate_rotation().inverse() * (x - PLATE)).y - PLATE_HALF.y;

        x.distance(center) < GOB_RADIUS && above > 0.5 * DX
    };

    LiquidDomain::new(config).with_properties(&saliva(scene.level)).filled_where(inside)
}

fn plate_rotation() -> Quat {
    Quat::from_rotation_z(-PLATE_SLOPE.to_radians())
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    scene: Res<Scene>,
) {
    let center = ORIGIN + CELLS.as_vec3() * DX * 0.5;
    let slate = materials.add(StandardMaterial {
        base_color: Color::srgb(0.2, 0.22, 0.26),
        perceptual_roughness: 0.6,
        ..default()
    });
    let backdrop = materials.add(StandardMaterial {
        base_color: Color::srgb(0.3, 0.33, 0.38),
        perceptual_roughness: 0.9,
        ..default()
    });
    let enamel = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.88, 0.84),
        perceptual_roughness: 0.35,
        ..default()
    });

    commands.spawn((domain(*scene), Transform::from_translation(ORIGIN), Visibility::default()));

    commands.spawn((
        LiquidBody {
            shape: LiquidShape::Cuboid(PLATE_HALF),
            density: 2400.0,
            dynamic: false,
        },
        Mesh3d(meshes.add(Cuboid::from_size(PLATE_HALF * 2.0))),
        MeshMaterial3d(slate),
        Transform::from_translation(ORIGIN + PLATE).with_rotation(plate_rotation()),
    ));

    commands.spawn((
        LiquidBody {
            shape: LiquidShape::Sphere(RADIUS),
            density: 2400.0,
            dynamic: false,
        },
        Mesh3d(meshes.add(Sphere::new(RADIUS).mesh().uv(48, 24))),
        MeshMaterial3d(enamel),
        Transform::from_translation(ORIGIN + SPHERE),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.4, 0.01, 0.3))),
        MeshMaterial3d(backdrop.clone()),
        Transform::from_translation(Vec3::new(center.x, -0.005, center.z)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(0.4, 0.3, 0.01))),
        MeshMaterial3d(backdrop),
        Transform::from_translation(Vec3::new(center.x, 0.15, ORIGIN.z - 0.011)),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 9_000.0,
            shadow_maps_enabled: true,
            shadow_depth_bias: 0.0002,
            ..default()
        },
        Transform::default().looking_to(-SUN, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            minimum_distance: 0.01,
            first_cascade_far_bound: 0.12,
            maximum_distance: 0.4,
            ..default()
        }
        .build(),
    ));

    let target = Vec3::new(center.x, center.y, center.z);
    let eye = target + Quat::from_rotation_x(-VIEW_PITCH.to_radians()) * Vec3::Z * VIEW_DISTANCE;

    commands.spawn((
        Camera3d::default(),
        Projection::from(PerspectiveProjection {
            near: 2.0 * MM,
            far: 2.0,
            ..default()
        }),
        Hdr,
        DepthPrepass,
        ScreenSpaceTransmission::default(),
        TemporalAntiAliasing::default(),
        Msaa::Off,
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
    ));
}

fn reset_domain(keys: Res<ButtonInput<KeyCode>>, scene: Res<Scene>, mut domains: Query<&mut LiquidDomain>) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }

    for mut current in &mut domains {
        *current = domain(*scene);
    }
}

fn capture_frames(time: Res<Time>, mut commands: Commands, mut taken: Local<usize>, mut exit: MessageWriter<AppExit>) {
    let Ok(path) = std::env::var("LIQUID_CAPTURE") else {
        return;
    };

    let t = time.elapsed_secs();

    if let Some(at) = CAPTURES.get(*taken).filter(|at| t > **at) {
        *taken += 1;
        let file = format!("{}-{at}s.png", Path::new(&path).with_extension("").display());
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(file));
    }

    if t > CAPTURES[CAPTURES.len() - 1] + FLUSH {
        exit.write(AppExit::Success);
    }
}

fn record_frames(time: Res<Time>, mut commands: Commands, mut saved: Local<u32>, mut exit: MessageWriter<AppExit>) {
    let Ok(dir) = std::env::var("LIQUID_RECORD") else {
        return;
    };

    let t = time.elapsed_secs();

    if t <= DURATION {
        let file = Path::new(&dir).join(format!("{:05}.png", *saved));
        *saved += 1;
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(file));
    } else if t > DURATION + FLUSH {
        exit.write(AppExit::Success);
    }
}

