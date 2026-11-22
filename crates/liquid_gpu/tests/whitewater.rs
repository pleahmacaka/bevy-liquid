use bevy::math::{UVec3, Vec3};
use liquid_gpu::layout::{GpuParticle, GpuState, GpuWhite, GpuWhiteState};
use liquid_gpu::solver::{GpuSolver, SolverConfig};

const SPRAY: f32 = 0.0;
const CAPACITY: u32 = 1 << 16;

fn gpu() -> (wgpu::Device, wgpu::Queue) {
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
        label: Some("whitewater test"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
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

fn validated<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> T {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = build();

    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation error: {error}");
    }

    out
}

fn column() -> Vec<GpuParticle> {
    let mut out = Vec::new();

    for k in 0..16u32 {
        for j in 0..24u32 {
            for i in 0..12u32 {
                for s in 0..8u32 {
                    let sub = Vec3::new((s & 1) as f32, ((s >> 1) & 1) as f32, ((s >> 2) & 1) as f32);
                    let x = (Vec3::new(i as f32, j as f32, k as f32) + (sub + 0.5) * 0.5) / 32.0;
                    let id = out.len() as f32;
                    out.push(GpuParticle { x: [x.x, x.y, x.z, id], v: [3.0, 0.0, 0.0, 0.0], ..Default::default() });
                }
            }
        }
    }

    out
}

fn config(particles: usize, whitewater_capacity: u32, max_substeps: u32) -> SolverConfig {
    SolverConfig {
        dims: UVec3::new(48, 32, 16),
        dx: 1.0 / 32.0,
        max_particles: particles as u32,
        max_substeps,
        whitewater_capacity,
        ..SolverConfig::default()
    }
}

fn solver(device: &wgpu::Device, queue: &wgpu::Queue, seed: &[GpuParticle], config: SolverConfig, frame_dt: f32) -> GpuSolver {
    let mut solver = validated(device, || GpuSolver::new(device, config, None));
    solver.initialize(queue);
    solver.set_frame_dt(queue, frame_dt);
    solver.upload_particles(queue, seed);
    solver
}

fn run(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver, frames: u32) {
    validated(device, || {
        for _ in 0..frames {
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode_frame(&mut encoder, None);
            queue.submit([encoder.finish()]);
        }
    });
}

fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ *b as u64).wrapping_mul(0x0100_0000_01b3))
}

struct Snapshot {
    state: GpuState,
    particles: Vec<GpuParticle>,
    velocity: Vec<f32>,
}

fn snapshot(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> Snapshot {
    let state: GpuState = read(device, queue, &solver.buffers.state, 1)[0];
    let particles = read(device, queue, &solver.buffers.particles, state.particle_count as usize);
    let velocity = read(device, queue, &solver.buffers.velocity, solver.params.face_base[3] as usize);

    Snapshot { state, particles, velocity }
}

#[test]
fn dam_break_generates_bounded_whitewater() {
    let (device, queue) = gpu();
    let seed = column();
    let solver = solver(&device, &queue, &seed, config(seed.len(), CAPACITY, 4), 1.0 / 60.0);

    run(&device, &queue, &solver, 30);

    let state: GpuWhiteState = read(&device, &queue, &solver.buffers.white_state, 1)[0];
    let white: Vec<GpuWhite> = read(&device, &queue, &solver.buffers.white, state.count as usize);
    let size = solver.config.dims.as_vec3() * solver.config.dx;
    let mut kinds = [0usize; 3];

    for p in &white {
        let x = Vec3::new(p.x[0], p.x[1], p.x[2]);

        assert!(p.x.iter().chain(&p.v).all(|f| f.is_finite()), "non-finite whitewater {p:?}");
        assert!(x.cmpge(Vec3::ZERO).all() && x.cmple(size).all(), "whitewater outside the domain {p:?}");

        kinds[p.v[3] as usize] += 1;
    }

    println!("whitewater count {} spawned {} (spray, foam, bubble) {kinds:?}", state.count, state.spawned);

    assert!(state.count > 0);
    assert!(state.count <= CAPACITY);
}

#[test]
fn spray_follows_ballistic_update() {
    let (device, queue) = gpu();
    let resting = [GpuParticle { x: [0.2, 0.05, 0.2, 0.0], ..Default::default() }];
    let solver = solver(&device, &queue, &resting, config(1, 64, 1), 1.0 / 120.0);
    let start = GpuWhite { x: [0.75, 0.6, 0.25, 1.0], v: [0.3, 0.5, -0.2, SPRAY] };

    queue.write_buffer(&solver.buffers.white, 0, bytemuck::bytes_of(&start));
    queue.write_buffer(&solver.buffers.white_state, 0, bytemuck::bytes_of(&GpuWhiteState { count: 1, ..Default::default() }));
    run(&device, &queue, &solver, 1);

    let dt = read::<GpuState>(&device, &queue, &solver.buffers.state, 1)[0].dt;
    let state: GpuWhiteState = read(&device, &queue, &solver.buffers.white_state, 1)[0];
    let end: GpuWhite = read(&device, &queue, &solver.buffers.white, 1)[0];

    let g = solver.config.gravity;
    let x0 = Vec3::new(start.x[0], start.x[1], start.x[2]);
    let v0 = Vec3::new(start.v[0], start.v[1], start.v[2]);
    let x1 = x0 + v0 * dt + 0.5 * g * dt * dt;
    let v1 = v0 + g * dt;

    println!("dt {dt:.5}, spray x {:?} expected {x1:?}, v {:?} expected {v1:?}", end.x, end.v);

    assert_eq!(state.count, 1);
    assert!(dt > 0.0);
    assert_eq!(end.v[3], SPRAY);
    assert!((Vec3::new(end.x[0], end.x[1], end.x[2]) - x1).length() < 1e-6);
    assert!((Vec3::new(end.v[0], end.v[1], end.v[2]) - v1).length() < 1e-5);
    assert!(end.x[1] < x0.y + v0.y * dt);
}

#[test]
fn disabled_whitewater_leaves_simulation_bit_identical() {
    let (device, queue) = gpu();
    let seed = column();
    let runs: Vec<Snapshot> = [0, CAPACITY]
        .into_iter()
        .map(|capacity| {
            let solver = solver(&device, &queue, &seed, config(seed.len(), capacity, 4), 1.0 / 60.0);
            run(&device, &queue, &solver, 20);
            snapshot(&device, &queue, &solver)
        })
        .collect();

    let (off, on) = (&runs[0], &runs[1]);

    println!(
        "checksum particles {:016x} velocity {:016x} count {}",
        fnv(bytemuck::cast_slice(&off.particles)),
        fnv(bytemuck::cast_slice(&off.velocity)),
        off.state.particle_count
    );

    let bytes = |s: &Snapshot| {
        [bytemuck::bytes_of(&s.state), bytemuck::cast_slice(&s.particles), bytemuck::cast_slice(&s.velocity)].concat()
    };

    assert!(bytes(off) == bytes(on), "main simulation differs with whitewater enabled");
}
