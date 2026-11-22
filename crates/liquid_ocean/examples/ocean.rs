use std::f64::consts::FRAC_PI_2;

use bevy::asset::RenderAssetUsages;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::light::GlobalAmbientLight;
use bevy::mesh::Indices;
use bevy::pbr::ScreenSpaceTransmission;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;
use liquid_ocean::{Heightfield, LiquidOceanPlugin, OceanSurface};
use liquid_wave::dispersive::{Dispersive, DispersiveParams};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::ocean::{Ocean, OceanParams};
use liquid_wave::spectrum::SeaState;
use liquid_wave::swe::{Boundary, Side, Swe};

const LAKE_LEVEL: f64 = 22.0;
const DAM: f64 = 240.0;
const DOMAIN_ORIGIN: [f64; 2] = [-64.0, 80.0];
const DOMAIN_CELLS: [usize; 2] = [128, 256];

fn main() {
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.62, 0.74, 0.86)))
        .insert_resource(GlobalAmbientLight {
            brightness: 150.0,
            ..default()
        })
        .add_plugins((
            DefaultPlugins.set(WindowPlugin {
                primary_window: Some(Window {
                    title: "liquid ocean".into(),
                    ..default()
                }),
                ..default()
            }),
            LiquidOceanPlugin,
        ))
        .add_systems(Startup, setup)
        .add_systems(Update, (fly_camera, reset_dam_break))
        .run();
}

fn terrain(x: f64, z: f64) -> f64 {
    let land = if z < 0.0 {
        -1.0 + 0.15 * z.max(-200.0)
    } else {
        -1.0 + 0.13 * z
    };
    let rolling = 1.5 * (0.03 * x).sin() * (0.025 * z).cos() + 0.8 * (0.07 * x + 0.05 * z).sin();
    let inland = ((z - 20.0) / 60.0).clamp(0.0, 1.0);
    let valley = 8.0 * (-(x / 18.0).powi(2)).exp() * inland * inland * (3.0 - 2.0 * inland);
    let basin = 12.0 * (-(x * x + (z - 280.0).powi(2)) / 1800.0).exp();

    land + rolling * inland - valley - basin
}

fn river_valley() -> Swe {
    let grid = Grid::new(DOMAIN_CELLS[0], DOMAIN_CELLS[1], 1.0, 1.0);
    let bed = Bathymetry::from_fn(&grid, |x, z| {
        terrain(x + DOMAIN_ORIGIN[0], z + DOMAIN_ORIGIN[1])
    });
    let mut state = State::lake(&grid, &bed, LAKE_LEVEL);

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            if grid.center(i, k)[1] + DOMAIN_ORIGIN[1] < DAM {
                state.h[grid.cell(i, k)] = 0.0;
            }
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.boundaries = swe.boundaries.with(Side::South, Boundary::Transmissive);
    swe.params.manning = 0.03;
    swe.params.dry_depth = 0.01;

    swe
}

fn terrain_mesh() -> Mesh {
    let (min, max, n) = ([-400.0f64, -300.0f64], [400.0f64, 500.0f64], 400usize);
    let step = [(max[0] - min[0]) / n as f64, (max[1] - min[1]) / n as f64];
    let mut positions = Vec::new();
    let mut colors = Vec::new();

    for k in 0..=n {
        for i in 0..=n {
            let x = min[0] + i as f64 * step[0];
            let z = min[1] + k as f64 * step[1];
            let y = terrain(x, z);
            let slope = (terrain(x + 1.0, z) - terrain(x - 1.0, z))
                .hypot(terrain(x, z + 1.0) - terrain(x, z - 1.0))
                / 2.0;

            let color = if y < 1.5 {
                [0.76, 0.70, 0.52, 1.0]
            } else if slope > 0.45 {
                [0.42, 0.40, 0.38, 1.0]
            } else {
                [0.30, 0.45, 0.20, 1.0]
            };

            positions.push([x as f32, y as f32, z as f32]);
            colors.push(color);
        }
    }

    let side = (n + 1) as u32;
    let mut indices = Vec::new();

    for k in 0..side - 1 {
        for i in 0..side - 1 {
            let a = k * side + i;
            indices.extend([a, a + side, a + 1, a + 1, a + side, a + side + 1]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, colors)
    .with_inserted_indices(Indices::U32(indices));
    mesh.compute_smooth_normals();

    mesh
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    commands.spawn((
        Mesh3d(meshes.add(terrain_mesh())),
        MeshMaterial3d(materials.add(StandardMaterial {
            perceptual_roughness: 0.95,
            ..default()
        })),
    ));

    let sea = SeaState {
        wind_speed: 9.0,
        wind_direction: FRAC_PI_2,
        fetch: 150_000.0,
        depth: 40.0,
        ..default()
    };
    let ocean = Ocean::new(OceanParams {
        sea,
        choppiness: 1.2,
        seed: 7,
        ..default()
    });
    commands.spawn(OceanSurface::new(ocean));

    let mut field =
        Heightfield::dispersive(Dispersive::new(river_valley(), DispersiveParams::default()));
    field.substeps = 4;
    field.dt_max = 0.02;
    commands.spawn((
        field,
        Transform::from_xyz(DOMAIN_ORIGIN[0] as f32, 0.0, DOMAIN_ORIGIN[1] as f32),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 7_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::from_xyz(0.0, 0.0, 0.0).looking_to(Vec3::new(-0.45, -0.55, -0.35), Vec3::Y),
    ));

    let eye = Vec3::new(70.0, 85.0, -120.0);
    let target = Vec3::new(0.0, 0.0, 140.0);
    let look = (target - eye).normalize();

    commands.spawn((
        Camera3d::default(),
        Hdr,
        DepthPrepass,
        ScreenSpaceTransmission::default(),
        Msaa::Off,
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
        FlyCamera {
            yaw: (-look.x).atan2(-look.z),
            pitch: look.y.asin(),
            speed: 25.0,
        },
    ));
}

#[derive(Component)]
struct FlyCamera {
    yaw: f32,
    pitch: f32,
    speed: f32,
}

fn fly_camera(
    time: Res<Time>,
    keys: Res<ButtonInput<KeyCode>>,
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    mut cameras: Query<(&mut Transform, &mut FlyCamera)>,
) {
    let dt = time.delta_secs();

    for (mut transform, mut fly) in &mut cameras {
        if buttons.pressed(MouseButton::Right) {
            fly.yaw -= motion.delta.x * 0.003;
            fly.pitch = (fly.pitch - motion.delta.y * 0.003).clamp(-1.54, 1.54);
        }

        transform.rotation = Quat::from_euler(EulerRot::YXZ, fly.yaw, fly.pitch, 0.0);

        let bindings = [
            (KeyCode::KeyW, *transform.forward()),
            (KeyCode::KeyS, *transform.back()),
            (KeyCode::KeyA, *transform.left()),
            (KeyCode::KeyD, *transform.right()),
            (KeyCode::KeyE, Vec3::Y),
            (KeyCode::KeyQ, Vec3::NEG_Y),
        ];

        let direction: Vec3 = bindings
            .iter()
            .filter(|(key, _)| keys.pressed(*key))
            .map(|(_, d)| *d)
            .sum();
        let boost = if keys.pressed(KeyCode::ShiftLeft) {
            4.0
        } else {
            1.0
        };

        transform.translation += direction.normalize_or_zero() * fly.speed * boost * dt;
    }
}

fn reset_dam_break(keys: Res<ButtonInput<KeyCode>>, mut fields: Query<&mut Heightfield>) {
    if keys.just_pressed(KeyCode::KeyR) {
        for mut field in &mut fields {
            field.generation += 1;
        }
    }
}
