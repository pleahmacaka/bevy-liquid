mod common;

use std::io::Write;

use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::DiagnosticsStore;
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::pbr::ScreenSpaceTransmission;
use bevy::prelude::*;
use bevy::platform::collections::HashMap;
use bevy::render::render_resource::{BufferDescriptor, BufferUsages, CommandEncoderDescriptor, MapMode, PollType};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use common::Preset;
use liquid_gpu::diagnostics::PARTICLES;
use liquid_gpu::layout::{GpuParticle, GpuState};
use liquid_gpu::plugin::LiquidFrame;
use liquid_gpu::prelude::*;
use liquid_gpu::solver::GpuSolver;

const ORIGIN: Vec3 = Vec3::new(-0.6, 0.0, -0.24);
const CELLS: UVec3 = UVec3::new(60, 60, 24);
const DX: f32 = 0.02;
const CEILING: f32 = 1.05;
const SLAB: Vec3 = Vec3::new(0.8, 0.1, 0.5);
const FILM: f32 = 0.07;
const DRIPS: [f32; 3] = [0.22, 0.5, 0.86];
const DRIP_RADIUS: f32 = 0.12;
const SPHERE: Vec3 = Vec3::new(0.86, 0.48, 0.24);
const RADIUS: f32 = 0.13;
const COAT: f32 = 0.06;
const SUN: Vec3 = Vec3::new(0.3, 0.8, 0.5);
const CAPTURES: [f32; 4] = [2.0, 5.0, 10.0, 20.0];

const SPARSE_NEIGHBORS: usize = 25;
const AIR: Vec2 = Vec2::new(0.12, 0.88);
const PARTICLE_BYTES: usize = std::mem::size_of::<GpuParticle>();
const STATE_BYTES: usize = std::mem::size_of::<GpuState>().next_multiple_of(8);

#[derive(Resource)]
struct ParticleView {
    twin: Option<GpuSolver>,
    log: std::fs::File,
}

fn main() {
    let particles = std::env::var_os("LIQUID_PARTICLES").map(std::path::PathBuf::from);
    let mut app = common::app(Window { title: "bevy fluid honey".into(), ..default() });

    app.insert_resource(ClearColor(Color::srgb(0.8, 0.76, 0.7)))
        .insert_resource(GlobalAmbientLight {
            brightness: 400.0,
            ..default()
        })
        .add_plugins(LiquidPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, (reset_domain, capture_frames));

    if let Some(path) = particles {
        let mut log = std::fs::File::create(path).expect("particle log");
        writeln!(log, "time,shown,twin,dead,in_sphere,near_sphere,sparse,sparse_near_sphere,air,air_sparse,air_lateral_speed,air_fall_speed").expect("particle log");
        app.world_mut().resource_mut::<GizmoConfigStore>().config_mut::<DefaultGizmoConfigGroup>().0.depth_bias = -1.0;
        app.insert_resource(ParticleView { twin: None, log }).add_systems(PostUpdate, particle_view);
    }

    app.run();
}

#[derive(Clone, Copy)]
struct Recipe {
    viscosity: f32,
    surface_tension: f32,
    adhesion: f32,
}

const RUNNY: Recipe = Recipe { viscosity: 150.0, surface_tension: 2.0, adhesion: 1.2 };
const THICK: Recipe = Recipe { viscosity: 1500.0, surface_tension: 4.0, adhesion: 1.2 };

fn honey() -> LiquidProperties {
    let water = LiquidProperties::water();
    let recipe = common::choice("--honey", &[("runny", RUNNY), ("thick", THICK)]);

    LiquidProperties {
        density: 1420.0,
        viscosity: recipe.viscosity,
        surface_tension: recipe.surface_tension,
        adhesion: recipe.adhesion,
        look: WaterLook {
            absorption: Vec4::new(4.0, 14.0, 45.0, 0.0),
            scatter: Vec4::new(0.6, 0.12, 0.0, 0.0),
            sky_zenith: Vec4::new(0.25, 0.24, 0.22, 0.0),
            sky_horizon: Vec4::new(0.45, 0.42, 0.38, 0.0),
            sun: SUN.normalize().extend(60.0),
            ..water.look
        },
        ..water
    }
}

fn domain(quality: Quality) -> LiquidDomain {
    let config = SolverConfig {
        dims: CELLS,
        dx: DX,
        max_particles: 100_000,
        cfl: 2.5,
        tolerance: 1e-3,
        viscosity_iterations: 40,
        ..default()
    }
    .with_quality(quality);
    let config = SolverConfig { dims: CELLS, dx: DX, ..config };
    let size = CELLS.as_vec3() * DX;

    LiquidDomain::new(config).with_properties(&honey()).filled_where(|p| is_honey(p, size))
}

fn is_honey(p: Vec3, size: Vec3) -> bool {
    let inset = Vec3::new(0.08, 0.0, 0.06);
    let film = p.y > CEILING - FILM && p.xz().cmpgt(inset.xz()).all() && p.xz().cmplt((size - inset).xz()).all();
    let drip = DRIPS.iter().any(|x| p.distance(Vec3::new(*x, CEILING, size.z * 0.5)) < DRIP_RADIUS);
    let r = p.distance(SPHERE);
    let coat = r > RADIUS && r < RADIUS + COAT && p.y > SPHERE.y;

    p.y < CEILING && (film || drip || coat)
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    preset: Res<Preset>,
) {
    let size = CELLS.as_vec3() * DX;
    let wood = materials.add(StandardMaterial {
        base_color: Color::srgb(0.42, 0.3, 0.2),
        perceptual_roughness: 0.85,
        ..default()
    });
    let stone = materials.add(StandardMaterial {
        base_color: Color::srgb(0.6, 0.58, 0.55),
        perceptual_roughness: 0.9,
        ..default()
    });
    let ceramic = materials.add(StandardMaterial {
        base_color: Color::srgb(0.9, 0.88, 0.84),
        perceptual_roughness: 0.35,
        ..default()
    });

    commands.spawn((domain(preset.0), Transform::from_translation(ORIGIN), Visibility::default()));

    commands.spawn((
        LiquidBody {
            shape: LiquidShape::Cuboid(SLAB),
            density: 700.0,
            dynamic: false,
        },
        Mesh3d(meshes.add(Cuboid::from_size(SLAB * 2.0))),
        MeshMaterial3d(wood),
        Transform::from_translation(ORIGIN + Vec3::new(size.x * 0.5, CEILING + SLAB.y, size.z * 0.5)),
    ));

    commands.spawn((
        LiquidBody {
            shape: LiquidShape::Sphere(RADIUS),
            density: 2400.0,
            dynamic: false,
        },
        Mesh3d(meshes.add(Sphere::new(RADIUS).mesh().uv(48, 24))),
        MeshMaterial3d(ceramic),
        Transform::from_translation(ORIGIN + SPHERE),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(4.0, 0.1, 3.0))),
        MeshMaterial3d(stone.clone()),
        Transform::from_translation(ORIGIN + Vec3::new(size.x * 0.5, -0.05, size.z * 0.5)),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(4.0, 3.0, 0.1))),
        MeshMaterial3d(stone),
        Transform::from_translation(ORIGIN + Vec3::new(size.x * 0.5, 1.5, -0.3)),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 9_000.0,
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

    let eye = ORIGIN + Vec3::new(size.x * 0.5, 0.45, 2.1);
    let target = ORIGIN + Vec3::new(size.x * 0.5, 0.6, size.z * 0.5);

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

fn reset_domain(keys: Res<ButtonInput<KeyCode>>, preset: Res<Preset>, mut domains: Query<&mut LiquidDomain>) {
    if !keys.just_pressed(KeyCode::KeyR) {
        return;
    }

    for mut current in &mut domains {
        *current = domain(preset.0);
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

#[derive(bevy::ecs::system::SystemParam)]
struct Gpu<'w> {
    device: Res<'w, RenderDevice>,
    queue: Res<'w, RenderQueue>,
}

fn particle_view(
    mut view: ResMut<ParticleView>,
    frame: Res<LiquidFrame>,
    gpu: Gpu,
    time: Res<Time>,
    store: Res<DiagnosticsStore>,
    mut gizmos: Gizmos,
) {
    let (Some(config), Some(seed)) = (&frame.config, &frame.seed) else {
        return;
    };
    let Gpu { device, queue } = gpu;

    let view = &mut *view;
    let twin = view.twin.get_or_insert_with(|| {
        let twin = GpuSolver::new(device.wgpu_device(), config.clone(), None);
        twin.initialize(&queue);
        twin.upload_particles(&queue, seed);

        twin
    });

    twin.upload_bodies(&queue, &frame.bodies);
    twin.set_frame_dt(&queue, frame.frame_dt);

    let bytes = (STATE_BYTES + seed.len() * PARTICLE_BYTES) as u64;
    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("honey particle view"),
        size: bytes,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor::default());

    twin.encode_frame(&mut encoder, None);
    encoder.copy_buffer_to_buffer(&twin.buffers.state, 0, &staging, 0, std::mem::size_of::<GpuState>() as u64);
    encoder.copy_buffer_to_buffer(&twin.buffers.particles, 0, &staging, STATE_BYTES as u64, bytes - STATE_BYTES as u64);
    queue.submit([encoder.finish()]);
    staging.map_async(MapMode::Read, .., |_| ());

    if device.poll(PollType::wait_indefinitely()).is_err() {
        return;
    }

    let Ok(mapped) = staging.get_mapped_range(..) else {
        return;
    };

    let word = |at: usize| [mapped[at], mapped[at + 1], mapped[at + 2], mapped[at + 3]];
    let count = (u32::from_le_bytes(word(16)) as usize).min(seed.len());
    let particles: Vec<(Vec3, Vec3, bool)> = (0..count)
        .map(|i| {
            let at = STATE_BYTES + i * PARTICLE_BYTES;
            let value = |k: usize| f32::from_le_bytes(word(at + 4 * k));

            (Vec3::new(value(0), value(1), value(2)), Vec3::new(value(4), value(5), value(6)), value(7) > 0.5)
        })
        .collect();

    let reach = 2.0 * config.dx;
    let key = |p: Vec3| (p / reach).floor().as_ivec3();
    let mut buckets: HashMap<IVec3, Vec<Vec3>> = HashMap::default();

    for (p, ..) in &particles {
        buckets.entry(key(*p)).or_default().push(*p);
    }

    let neighbors = |p: Vec3| {
        let home = key(p);
        let around = (-1..=1).flat_map(|x| (-1..=1).flat_map(move |y| (-1..=1).map(move |z| IVec3::new(x, y, z))));

        around.filter_map(|o| buckets.get(&(home + o))).flatten().filter(|q| q.distance_squared(p) < reach * reach).count()
    };

    let mut tally = [0usize; 7];
    let mut lateral = 0.0;
    let mut fall = 0.0;

    for (p, v, dead) in &particles {
        let r = p.distance(SPHERE);
        let near = r < RADIUS + 3.0 * config.dx;
        let sparse = neighbors(*p) <= SPARSE_NEIGHBORS;
        let air = !near && p.y > AIR.x && p.y < AIR.y;

        if air {
            lateral += v.xz().length_squared();
            fall -= v.y;
        }

        let color = match (*dead, r < RADIUS, sparse) {
            (true, ..) => Color::srgb(1.0, 0.0, 0.0),
            (_, true, _) => Color::srgb(1.0, 0.0, 1.0),
            (_, _, true) => Color::srgb(0.0, 1.0, 1.0),
            _ => Color::BLACK,
        };
        let at = ORIGIN + *p;
        let tick = 0.15 * config.dx;

        gizmos.line(at - Vec3::X * tick, at + Vec3::X * tick, color);
        gizmos.line(at - Vec3::Y * tick, at + Vec3::Y * tick, color);

        for (slot, hit) in tally.iter_mut().zip([*dead, r < RADIUS, near, sparse, sparse && near, air, air && sparse]) {
            *slot += hit as usize;
        }
    }

    let shown = store.get(&PARTICLES).and_then(|d| d.value()).unwrap_or(0.0);
    let [dead, inside, near, sparse, sparse_near, air, air_sparse] = tally;
    let mean = 1.0 / air.max(1) as f32;
    let line = writeln!(
        view.log,
        "{:.4},{shown},{count},{dead},{inside},{near},{sparse},{sparse_near},{air},{air_sparse},{:.4},{:.4}",
        time.elapsed_secs(),
        (lateral * mean).sqrt(),
        fall * mean
    );

    if line.is_err() {
        warn_once!("the particle log is not writable");
    }
}
