mod common;

use std::f64::consts::FRAC_PI_2;

use avian3d::prelude::*;
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::input::mouse::AccumulatedMouseMotion;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::pbr::{ExtendedMaterial, ScreenSpaceTransmission};
use bevy::prelude::*;
use common::Preset;
use liquid_gpu::caustics::{CausticMap, CausticMaterial, CausticReceiver};
use liquid_gpu::coupling::FollowPromotion;
use liquid_gpu::emitter::MAX_LIQUID_EMITTERS;
use liquid_gpu::prelude::*;
use liquid_ocean::{Heightfield, HeightfieldLedger, OceanSurface, PromotionRegion, PromotionSettings};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::swe::Swe;
use liquid_wave::ocean::{Ocean, OceanParams};
use liquid_wave::spectrum::SeaState;

const DOMAIN_ORIGIN: Vec3 = Vec3::new(-2.0, 0.6, -1.5);
const DOMAIN_CELLS: UVec3 = UVec3::new(64, 32, 48);
const DX: f32 = 1.0 / 16.0;
const MAX_THROWN: usize = 8;
const LAKE_DEPTH: f32 = 0.45;
const POOL_ORIGIN: Vec3 = Vec3::new(-6.0, 0.6, -6.0);
const POOL_CELLS: usize = 120;
const POOL_DX: f64 = 0.1;
const SUN: Vec3 = Vec3::new(0.4, 0.7, 0.45);
const POUR_FLOW: f32 = 0.5;

#[derive(Resource, Clone, Copy)]
struct Pool(Entity);

fn main() {
    let window = Window {
        title: "bevy fluid".into(),
        desired_maximum_frame_latency: std::num::NonZeroU32::new(3),
        ..default()
    };
    let mut app = common::app(window);

    app.insert_resource(ClearColor(Color::srgb(0.62, 0.74, 0.86)))
        .insert_resource(GlobalAmbientLight {
            brightness: 300.0,
            ..default()
        })
        .add_plugins((PhysicsPlugins::default(), LiquidPlugin))
        .add_systems(Startup, setup)
        .add_systems(Update, (fly_camera, reset_domain, throw_ball, capture, toggle_follow, report_tiers));

    if let Some(flow) = pour_flow() {
        app.add_systems(Startup, move |commands: Commands| pour(commands, flow));
    }

    app.run();
}

fn pour_flow() -> Option<f32> {
    let value = std::env::var("LIQUID_POUR").ok()?;

    Some(value.parse().unwrap_or(POUR_FLOW))
}

fn pour(mut commands: Commands, flow: f32) {
    let size = DOMAIN_CELLS.as_vec3() * DX;
    let half = Vec3::new(0.15, 0.08, 0.15);
    let count = MAX_LIQUID_EMITTERS as f32;

    for k in 0..MAX_LIQUID_EMITTERS {
        let x = size.x * (k as f32 + 0.5) / count;
        let center = DOMAIN_ORIGIN + Vec3::new(x, size.y - 0.25, size.z * 0.5);

        commands.spawn(LiquidEmitter {
            velocity: Vec3::new(0.0, -2.0, 0.0),
            flow: Some(flow / count),
            ..LiquidEmitter::new(GasRegion::cuboid(center, half), 293.15)
        });
    }
}

fn toggle_follow(keys: Res<ButtonInput<KeyCode>>, mut commands: Commands, domains: Query<(Entity, Has<FollowPromotion>), With<LiquidDomain>>) {
    if !keys.just_pressed(KeyCode::KeyF) {
        return;
    }

    for (entity, following) in &domains {
        if following {
            commands.entity(entity).remove::<FollowPromotion>();
        } else {
            commands.entity(entity).insert(FollowPromotion::default());
        }
    }
}

fn report_tiers(time: Res<Time>, mut next: Local<f32>, fields: Query<(&HeightfieldLedger, &PromotionRegion)>) {
    if time.elapsed_secs() < *next {
        return;
    }

    *next = time.elapsed_secs() + 1.0;

    for (ledger, region) in &fields {
        let l = ledger.ledger;
        let window = region.promotion.map(|p| (p.flagged, p.window));

        info!(
            "tiers: 3D {:.3} m3, heightfield outside {:.3} m3, interface {:+.4} m3, conserved {:.4} m3, promotion {window:?}",
            l.tier,
            l.outside,
            l.interface,
            l.conserved()
        );
    }
}

fn domain(pool: Entity, quality: Quality) -> LiquidDomain {
    let tank = pour_flow().is_some();
    let config = SolverConfig {
        dims: DOMAIN_CELLS,
        dx: DX,
        max_particles: 600_000,
        cfl: 3.0,
        max_substeps: 6,
        whitewater_capacity: 1 << 17,
        narrow_band: (!tank).then_some(2.0),
        block_smoothing: true,
        smoothing: 4,
        pressure_iterations: 5,
        tolerance: 0.0,
        ..default()
    }
    .with_quality(quality);
    let size = config.dims.as_vec3() * config.dx;
    let column = Vec3::new(1.4, 1.3, size.z);
    let inside = |p: Vec3| p.cmplt(column).all() || p.y < LAKE_DEPTH;
    let domain = LiquidDomain::new(config).with_look(water_look()).filled_where(inside);

    if tank {
        return domain;
    }

    domain.coupled_to(HeightfieldCoupling {
        entity: pool,
        relax_cells: 4.0,
        blend: 0.5,
    })
}

fn pool() -> Heightfield {
    let grid = Grid::new(POOL_CELLS, POOL_CELLS, POOL_DX, POOL_DX);
    let bed = Bathymetry::from_fn(&grid, |_, _| 0.0);
    let mut state = State::lake(&grid, &bed, LAKE_DEPTH as f64);

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            if grid.center(i, k)[0] < 2.5 {
                state.h[grid.cell(i, k)] = 1.1;
            }
        }
    }

    let mut field = Heightfield::shallow_water(Swe::new(grid, bed, state));
    field.substeps = 4;
    field.dt_max = 0.01;
    field.look = water_look();
    field.promotion = Some(PromotionSettings::default());

    field
}

fn water_look() -> WaterLook {
    WaterLook {
        sun: SUN.normalize().extend(60.0),
        ..default()
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut floor_materials: ResMut<Assets<CausticMaterial>>,
    caustics: Res<CausticMap>,
    preset: Res<Preset>,
) {
    let size = DOMAIN_CELLS.as_vec3() * DX;
    let stone = floor_materials.add(ExtendedMaterial {
        base: StandardMaterial {
            base_color: Color::srgb(0.55, 0.53, 0.5),
            perceptual_roughness: 0.9,
            ..default()
        },
        extension: CausticReceiver { map: caustics.0.clone() },
    });

    let pool = commands
        .spawn((pool(), Transform::from_translation(POOL_ORIGIN)))
        .id();
    commands.insert_resource(Pool(pool));

    commands.spawn((
        domain(pool, preset.0),
        Transform::from_translation(DOMAIN_ORIGIN),
        Visibility::default(),
    ));

    let floor = (POOL_CELLS as f64 * POOL_DX) as f32 + 1.0;

    commands.spawn((
        RigidBody::Static,
        Collider::cuboid(floor, 0.2, floor),
        Mesh3d(meshes.add(Cuboid::new(floor, 0.2, floor))),
        MeshMaterial3d(stone),
        Transform::from_translation(POOL_ORIGIN + Vec3::new(floor * 0.5 - 0.5, -0.1, floor * 0.5 - 0.5)),
    ));

    let walls = [
        (Vec3::new(-0.05, size.y * 0.5, size.z * 0.5), Vec3::new(0.1, size.y, size.z)),
        (Vec3::new(size.x + 0.05, size.y * 0.5, size.z * 0.5), Vec3::new(0.1, size.y, size.z)),
        (Vec3::new(size.x * 0.5, size.y * 0.5, -0.05), Vec3::new(size.x, size.y, 0.1)),
        (Vec3::new(size.x * 0.5, size.y * 0.5, size.z + 0.05), Vec3::new(size.x, size.y, 0.1)),
    ];

    for (offset, extent) in walls {
        commands.spawn((
            RigidBody::Static,
            Collider::cuboid(extent.x, extent.y, extent.z),
            Transform::from_translation(DOMAIN_ORIGIN + offset),
        ));
    }

    let wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.42, 0.25),
        perceptual_roughness: 0.8,
        ..default()
    });
    let half = Vec3::new(0.2, 0.12, 0.3);

    for (i, x) in [2.2f32, 3.0].into_iter().enumerate() {
        commands.spawn((
            RigidBody::Dynamic,
            Collider::cuboid(half.x * 2.0, half.y * 2.0, half.z * 2.0),
            ColliderDensity(450.0),
            LiquidBody {
                shape: LiquidShape::Cuboid(half),
                density: 450.0,
                dynamic: true,
            },
            Mesh3d(meshes.add(Cuboid::from_size(half * 2.0))),
            MeshMaterial3d(wood.clone()),
            Transform::from_translation(DOMAIN_ORIGIN + Vec3::new(x, 0.4, 0.9 + 1.2 * i as f32)),
        ));
    }

    let sea = SeaState {
        wind_speed: 8.0,
        wind_direction: FRAC_PI_2,
        fetch: 120_000.0,
        depth: 40.0,
        ..default()
    };
    commands.spawn(OceanSurface {
        look: water_look(),
        ..OceanSurface::new(Ocean::new(OceanParams {
            sea,
            choppiness: 1.1,
            seed: 3,
            ..default()
        }))
    });

    commands.spawn((
        DirectionalLight {
            illuminance: 8_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_to(-SUN, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            first_cascade_far_bound: 10.0,
            maximum_distance: 20.0,
            ..default()
        }
        .build(),
    ));

    let eye = DOMAIN_ORIGIN + Vec3::new(size.x * 0.5, 2.6, -3.2);
    let target = DOMAIN_ORIGIN + Vec3::new(size.x * 0.5, 0.4, size.z * 0.5);
    let look = (target - eye).normalize();

    commands.spawn((
        Camera3d::default(),
        Hdr,
        DepthPrepass,
        ScreenSpaceTransmission::default(),
        Msaa::Off,
        TemporalAntiAliasing::default(),
        Transform::from_translation(eye).looking_at(target, Vec3::Y),
        FlyCamera {
            yaw: (-look.x).atan2(-look.z),
            pitch: look.y.asin(),
            speed: 3.0,
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
        let boost = if keys.pressed(KeyCode::ShiftLeft) { 4.0 } else { 1.0 };

        transform.translation += direction.normalize_or_zero() * fly.speed * boost * dt;
    }
}

fn capture(time: Res<Time>, mut commands: Commands, mut taken: Local<bool>, mut exit: MessageWriter<AppExit>) {
    let Ok(path) = std::env::var("LIQUID_CAPTURE") else {
        return;
    };

    let t = time.elapsed_secs();

    if !*taken && t > 5.0 {
        *taken = true;
        commands
            .spawn(bevy::render::view::screenshot::Screenshot::primary_window())
            .observe(bevy::render::view::screenshot::save_to_disk(path));
    }

    if t > 7.0 {
        exit.write(AppExit::Success);
    }
}

fn reset_domain(
    keys: Res<ButtonInput<KeyCode>>,
    pool: Res<Pool>,
    mut preset: ResMut<Preset>,
    mut domains: Query<&mut LiquidDomain>,
    mut fields: Query<&mut Heightfield>,
) {
    let toggle = keys.just_pressed(KeyCode::Tab);

    if !toggle && !keys.just_pressed(KeyCode::KeyR) {
        return;
    }

    if toggle {
        preset.0 = match preset.0 {
            Quality::Game => Quality::High,
            Quality::High => Quality::Game,
        };
        info!("liquid preset {}", preset.0.name());
    }

    for mut current in &mut domains {
        *current = domain(pool.0, preset.0);
    }

    for mut field in &mut fields {
        field.generation += 1;
    }
}

fn throw_ball(
    keys: Res<ButtonInput<KeyCode>>,
    cameras: Query<&Transform, With<FlyCamera>>,
    bodies: Query<(), With<LiquidBody>>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !keys.just_pressed(KeyCode::Space) || bodies.iter().count() >= MAX_THROWN {
        return;
    }

    let Ok(camera) = cameras.single() else {
        return;
    };

    let radius = 0.15;

    commands.spawn((
        RigidBody::Dynamic,
        Collider::sphere(radius),
        ColliderDensity(700.0),
        LinearVelocity(*camera.forward() * 6.0),
        LiquidBody {
            shape: LiquidShape::Sphere(radius),
            density: 700.0,
            dynamic: true,
        },
        Mesh3d(meshes.add(Sphere::new(radius))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.85, 0.3, 0.2),
            ..default()
        })),
        Transform::from_translation(camera.translation + *camera.forward() * 0.5),
    ));
}
