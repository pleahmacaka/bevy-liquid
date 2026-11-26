use bevy::math::{UVec3, Vec3};
use glam::DVec3;
use liquid_gpu::layout::{GpuBody, GpuParticle, GpuState, GpuWhiteState};
use liquid_gpu::solver::{GUARD_BYTES, GpuSolver, Profiler, SolverConfig};
use liquid_ref::particles::Particle;
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{dam_break, with_body};
use liquid_ref::sim::Params;

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

fn gpu(features: wgpu::Features) -> (wgpu::Device, wgpu::Queue) {
    static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();
    let instance = INSTANCE.get_or_init(|| wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env()));
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env(instance, None))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
        })
    .expect("no adapter");

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("sparse test"),
        required_features: features & adapter.features(),
        required_limits: limits::portable(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device")
}

fn read<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let bytes = (count.max(1) * std::mem::size_of::<T>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);

    staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");

    let view = staging.get_mapped_range(..).expect("mapped range");
    let out: Vec<T> = bytemuck::cast_slice(&view).to_vec();

    out[..count].to_vec()
}

fn to_gpu(id: usize, p: &Particle) -> GpuParticle {
    let f = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    let mut x = f(p.x);
    x[3] = id as f32;

    GpuParticle { x, v: f(p.v), c: [f(p.c[0]), f(p.c[1]), f(p.c[2])] }
}

fn to_gpu_body(b: &RigidBody) -> GpuBody {
    let f = |v: DVec3, w: f64| [v.x as f32, v.y as f32, v.z as f32, w as f32];
    let inv = b.inverse_inertia_world();
    let (kind, shape) = match b.shape {
        Shape::Sphere { radius } => (0.0, DVec3::splat(radius)),
        Shape::Cuboid { half } => (1.0, half),
    };

    GpuBody {
        position: f(b.position, 1.0 / b.mass()),
        rotation: [b.rotation.x as f32, b.rotation.y as f32, b.rotation.z as f32, b.rotation.w as f32],
        velocity: f(b.velocity, 2.0),
        angular: f(b.angular_velocity, 0.0),
        shape: f(shape, kind),
        inv_inertia: [f(inv.row(0), 0.0), f(inv.row(1), 0.0), f(inv.row(2), 0.0)],
    }
}

fn column(cells: UVec3, dx: f32) -> Vec<GpuParticle> {
    let mut out = Vec::new();

    for k in 0..cells.z {
        for j in 0..cells.y {
            for i in 0..cells.x {
                for s in 0..8u32 {
                    let sub = Vec3::new((s & 1) as f32, ((s >> 1) & 1) as f32, ((s >> 2) & 1) as f32);
                    let x = (Vec3::new(i as f32, j as f32, k as f32) + (sub + 0.5) * 0.5) * dx;
                    let id = out.len() as f32;
                    out.push(GpuParticle { x: [x.x, x.y, x.z, id], ..Default::default() });
                }
            }
        }
    }

    out
}

struct Scene {
    config: SolverConfig,
    particles: Vec<GpuParticle>,
    bodies: Vec<GpuBody>,
}

fn dam_break_scene() -> Scene {
    let dx = 1.0 / 32.0;
    let particles = column(UVec3::new(16, 24, 32), dx);
    let config = SolverConfig {
        dims: UVec3::new(48, 32, 32),
        dx,
        max_particles: particles.len() as u32,
        max_substeps: 4,
        whitewater_capacity: 1 << 16,
        ..SolverConfig::default()
    };

    Scene { config, particles, bodies: Vec::new() }
}

fn body_scene() -> Scene {
    let params = Params { density_projection: true, ..Params::default() };
    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.06, 0.04, 0.05) }, DVec3::new(0.3, 0.08, 0.08), Motion::Dynamic, 400.0);
    let sim = with_body(dam_break(glam::UVec3::new(24, 16, 8), 0.02, DVec3::new(0.16, 0.24, 0.0), params), body);
    let config = SolverConfig {
        dims: UVec3::new(24, 16, 8),
        dx: 0.02,
        max_particles: sim.particles.len() as u32 + 1024,
        max_substeps: 2,
        ..SolverConfig::default()
    };

    Scene {
        config,
        particles: sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect(),
        bodies: sim.bodies.iter().map(to_gpu_body).collect(),
    }
}

fn bench_scene() -> Scene {
    let dx = 1.0 / 64.0;
    let particles = column(UVec3::new(32, 48, 64), dx);
    let config = SolverConfig {
        dims: UVec3::new(96, 64, 64),
        dx,
        max_particles: particles.len() as u32,
        max_substeps: 2,
        ..SolverConfig::default()
    };

    Scene { config, particles, bodies: Vec::new() }
}

fn start(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, sparse: bool) -> GpuSolver {
    launch(device, queue, scene, SolverConfig { sparse, ..scene.config.clone() })
}

fn launch(device: &wgpu::Device, queue: &wgpu::Queue, scene: &Scene, config: SolverConfig) -> GpuSolver {
    let mut solver = GpuSolver::new(device, config, None);
    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, &scene.particles);
    solver.upload_bodies(queue, &scene.bodies);
    solver
}

fn step(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver, profiler: Option<&Profiler>) {
    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, profiler);

    if let Some(p) = profiler {
        p.resolve(&mut encoder);
    }

    queue.submit([encoder.finish()]);
}

struct Snapshot {
    state: GpuState,
    white: GpuWhiteState,
    particles: Vec<GpuParticle>,
    bodies: Vec<GpuBody>,
    fields: Vec<(&'static str, Vec<f32>)>,
}

fn stored(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f32> {
    read(device, queue, buffer, ((buffer.size() - GUARD_BYTES) / 4) as usize)
}

fn grid_cells(solver: &GpuSolver, slots: &[u32], values: &[f32]) -> Vec<f32> {
    let d = solver.config.dims;
    let mut out = Vec::new();

    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                out.push(values[solver.pooled_cell(slots, UVec3::new(x, y, z))]);
            }
        }
    }

    out
}

fn grid_faces(solver: &GpuSolver, slots: &[u32], values: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();

    for axis in 0..3 {
        let mut d = solver.config.dims;
        d[axis] += 1;

        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    out.push(values[solver.pooled_face(slots, axis, UVec3::new(x, y, z))]);
                }
            }
        }
    }

    out
}

fn slots(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> Vec<u32> {
    let range = solver.slot_table();
    let words: Vec<u32> = read(device, queue, &solver.buffers.brick_list, (range.end / 4) as usize);

    words[(range.start / 4) as usize..].to_vec()
}

fn snapshot(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> Snapshot {
    let b = &solver.buffers;
    let state: GpuState = read(device, queue, &b.state, 1)[0];
    let cells = solver.params.dims[3] as usize;
    let slots = slots(device, queue, solver);

    Snapshot {
        state,
        white: read(device, queue, &b.white_state, 1)[0],
        particles: read(device, queue, &b.particles, state.particle_count as usize),
        bodies: read(device, queue, &b.bodies, solver.params.body_count as usize),
        fields: vec![
            ("velocity", grid_faces(solver, &slots, &stored(device, queue, &b.velocity))),
            ("phi", grid_cells(solver, &slots, &stored(device, queue, &b.phi))),
            ("pressure", grid_cells(solver, &slots, &stored(device, queue, &solver.levels[0].x))),
            ("surface", read(device, queue, &b.surface, cells)),
        ],
    }
}

fn by_id(particles: &[GpuParticle]) -> Vec<Vec3> {
    let mut out = vec![Vec3::ZERO; particles.len()];

    for p in particles {
        out[p.x[3] as usize] = Vec3::new(p.x[0], p.x[1], p.x[2]);
    }

    out
}

fn compare(name: &str, scene: &Scene, frames: u32) -> (f32, bool) {
    let (device, queue) = gpu(wgpu::Features::empty());
    let runs: Vec<Snapshot> = [true, false]
        .into_iter()
        .map(|sparse| {
            let solver = start(&device, &queue, scene, sparse);

            for _ in 0..frames {
                step(&device, &queue, &solver, None);
            }

            snapshot(&device, &queue, &solver)
        })
        .collect();

    let (sparse, dense) = (&runs[0], &runs[1]);

    assert_eq!(sparse.particles.len(), dense.particles.len());

    let worst = by_id(&sparse.particles)
        .iter()
        .zip(by_id(&dense.particles))
        .fold(0.0f32, |m, (a, b)| m.max((*a - b).length()));
    let mut identical = bytemuck::bytes_of(&sparse.state) == bytemuck::bytes_of(&dense.state)
        && bytemuck::bytes_of(&sparse.white) == bytemuck::bytes_of(&dense.white)
        && bytemuck::cast_slice::<GpuParticle, u8>(&sparse.particles) == bytemuck::cast_slice::<GpuParticle, u8>(&dense.particles)
        && bytemuck::cast_slice::<GpuBody, u8>(&sparse.bodies) == bytemuck::cast_slice::<GpuBody, u8>(&dense.bodies);

    for ((field, a), (_, b)) in sparse.fields.iter().zip(&dense.fields) {
        let same = a.iter().zip(b).filter(|(x, y)| x.to_bits() == y.to_bits()).count();
        let diff = a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()));
        identical &= same == a.len();

        println!("{name}: {field} bit-identical {same}/{} values, max |diff| {diff:.3e}", a.len());
    }

    println!(
        "{name}: {frames} frames, {} particles, max position difference {worst:.3e} m ({:.3e} cells), whitewater {} vs {}, everything bit-identical: {identical}",
        sparse.particles.len(),
        worst / scene.config.dx,
        sparse.white.count,
        dense.white.count
    );

    (worst / scene.config.dx, identical)
}

#[test]
fn sparse_dam_break_is_bit_identical_to_dense() {
    let (worst, identical) = compare("dam break", &dam_break_scene(), 30);

    assert_eq!(worst, 0.0);
    assert!(identical);
}

#[test]
fn sparse_body_scene_tracks_dense() {
    let (worst, _) = compare("dam break with body", &body_scene(), 20);

    assert!(worst < 0.05, "position difference {worst} cells");
}

fn fingerprint(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> Vec<(&'static str, Vec<u8>)> {
    let s = snapshot(device, queue, solver);
    let mut white: Vec<[u32; 8]> = read(device, queue, &solver.buffers.white, s.white.count as usize);
    white.sort_unstable();

    let mut parts = vec![
        ("state", bytemuck::bytes_of(&s.state).to_vec()),
        ("whitewater state", bytemuck::bytes_of(&s.white).to_vec()),
        ("whitewater", bytemuck::cast_slice(&white).to_vec()),
        ("particles", bytemuck::cast_slice(&s.particles).to_vec()),
        ("bodies", bytemuck::cast_slice(&s.bodies).to_vec()),
    ];
    parts.extend(s.fields.iter().map(|(name, values)| (*name, bytemuck::cast_slice(values).to_vec())));

    parts
}

fn crowded_body_scene() -> Scene {
    let mut scene = body_scene();
    let spot = scene.particles[0].x;
    let extra = 160;
    let stride = scene.particles.len() / extra;

    for k in 0..extra {
        let id = scene.particles.len() as f32;
        scene.particles.insert(k * stride, GpuParticle { x: [spot[0], spot[1], spot[2], id], ..Default::default() });
    }

    scene.config.max_particles += extra as u32;
    scene
}

#[test]
fn repeated_runs_are_bit_identical() {
    let (device, queue) = gpu(wgpu::Features::SUBGROUP);
    let available = device.features().contains(wgpu::Features::SUBGROUP);
    let scene = crowded_body_scene();
    let frames = 12;
    let mut differing = Vec::new();

    for band in [Some(3.0), None] {
        for sparse in [true, false] {
            for subgroups in [true, false] {
                let config = SolverConfig { sparse, subgroups, narrow_band: band, whitewater_capacity: 1 << 12, ..scene.config.clone() };
                let label = format!("band {band:?} sparse {sparse} subgroups {}", subgroups && available);
                let runs: Vec<_> = (0..2)
                    .map(|_| {
                        let solver = launch(&device, &queue, &scene, config.clone());

                        for _ in 0..frames {
                            step(&device, &queue, &solver, None);
                        }

                        fingerprint(&device, &queue, &solver)
                    })
                    .collect();
                let white = u32::from_le_bytes(runs[0][1].1[..4].try_into().expect("whitewater count"));
                let fields: Vec<&str> = runs[0].iter().zip(&runs[1]).filter(|(a, b)| a.1 != b.1).map(|(a, _)| a.0).collect();

                println!("{label}: {frames} frames, {white} whitewater particles, repeated run differs in {fields:?}");

                differing.extend(fields.iter().map(|name| format!("{label}: {name}")));

                assert!(white > 0, "{label}: no whitewater spawned, so its kernels went untested");
            }
        }
    }

    assert!(differing.is_empty(), "repeated runs differ: {differing:?}");
}

fn active_bricks(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> u32 {
    read::<u32>(device, queue, &solver.buffers.brick_list, 1)[0]
}

#[test]
fn bench_scene_brick_fraction_and_timing() {
    let (device, queue) = gpu(wgpu::Features::TIMESTAMP_QUERY);
    let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let scene = bench_scene();
    let capacity = scene.config.dims.x * scene.config.dims.y * scene.config.dims.z / 512 * 2 / 3;
    let solvers = [Some(capacity), None].map(|capacity| {
        let config = SolverConfig { sparse: capacity.is_some(), brick_capacity: capacity, ..scene.config.clone() };
        launch(&device, &queue, &scene, config)
    });
    let profilers = solvers.each_ref().map(|s| timestamps.then(|| Profiler::new(&device, &queue, s.config.max_substeps)));
    let frames = 45;
    let mut totals = [0.0f32; 2];
    let mut fractions = Vec::new();

    for frame in 0..frames {
        for (k, solver) in solvers.iter().enumerate() {
            step(&device, &queue, solver, profilers[k].as_ref());

            if let Some(p) = &profilers[k]
                && frame >= frames / 3
            {
                totals[k] += p.read_milliseconds(&device).iter().map(|t| t.1).sum::<f32>();
            }
        }

        if frame % 15 == 14 {
            fractions.push(active_bricks(&device, &queue, &solvers[0]) as f32 / solvers[0].brick_count() as f32);
        }
    }

    let shown: Vec<String> = fractions.iter().map(|f| format!("{:.1}%", f * 100.0)).collect();
    println!("bench scene active brick fraction every 15 frames: {}", shown.join(" "));
    assert!(fractions.iter().all(|f| *f > 0.0 && *f < 1.0));

    let state: GpuState = read(&device, &queue, &solvers[0].buffers.state, 1)[0];
    println!(
        "bench scene pool: capacity {capacity} of {} bricks, {} unallocated, allocated bytes sparse {} vs dense {}",
        solvers[0].brick_count(),
        state.brick_overflow,
        solvers[0].allocated_bytes(),
        solvers[1].allocated_bytes()
    );
    assert_eq!(state.brick_overflow, 0);
    assert!(solvers[0].allocated_bytes() < solvers[1].allocated_bytes());

    if timestamps {
        let measured = (frames - frames / 3) as f32;
        let [sparse, dense] = totals.map(|t| t / measured);
        println!("bench scene mean gpu frame time: sparse {sparse:.2} ms, dense {dense:.2} ms, ratio {:.3}", sparse / dense);
        assert!(sparse < 1.15 * dense, "sparse {sparse} ms vs dense {dense} ms");
    }
}

#[test]
fn bench_scene_memory_scales_with_brick_capacity() {
    let (device, _) = gpu(wgpu::Features::empty());
    let scene = bench_scene();
    let bricks = scene.config.dims.x * scene.config.dims.y * scene.config.dims.z / 512;
    let dense = GpuSolver::new(&device, SolverConfig { sparse: false, ..scene.config.clone() }, None).grid_bytes();
    let pools: Vec<(u32, u64)> = [bricks, bricks * 2 / 3, bricks / 2]
        .into_iter()
        .map(|capacity| {
            let config = SolverConfig { brick_capacity: Some(capacity), ..scene.config.clone() };
            (capacity, GpuSolver::new(&device, config, None).grid_bytes())
        })
        .collect();
    let shown: Vec<String> = pools.iter().map(|(c, b)| format!("{c}/{bricks} bricks {b} B ({:.2}x)", *b as f64 / dense as f64)).collect();

    println!("bench scene grid bytes: dense {dense} B; sparse {}", shown.join(", "));

    assert!(pools.windows(2).all(|w| w[1].1 < w[0].1));
    assert!(pools[1].1 < dense);
}

#[test]
fn overflowing_pool_is_flagged_and_stays_in_bounds() {
    let (device, queue) = gpu(wgpu::Features::empty());
    let scene = dam_break_scene();
    let solver = launch(&device, &queue, &scene, SolverConfig { brick_capacity: Some(4), ..scene.config.clone() });

    for _ in 0..3 {
        step(&device, &queue, &solver, None);
    }

    let state: GpuState = read(&device, &queue, &solver.buffers.state, 1)[0];
    let violations = solver.guard_violations(&device, &queue);

    println!("capacity 4 of {} bricks: {} bricks left unallocated", solver.brick_count(), state.brick_overflow);

    assert!(state.brick_overflow > 0);
    assert!(violations.is_empty(), "{violations:?}");
}
