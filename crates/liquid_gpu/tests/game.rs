use bevy::math::{UVec3, Vec3};
use bevy::transform::components::GlobalTransform;
use liquid_gpu::layout::{GpuParticle, GpuState, S_FLOOR_LEVEL, S_FLOOR_MEAN, S_VOLUME, SCALAR_COUNT};
use liquid_gpu::plugin::{LiquidBody, LiquidShape, body_to_gpu, seed_box};
use liquid_gpu::solver::{Bind, GUARD_BYTES, GpuSolver, Quality, Slots, SolverConfig};

#[cfg(windows)]
mod common;

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

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
        label: Some("game preset test"),
        required_features: wgpu::Features::empty(),
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

fn half(bits: u32) -> f32 {
    let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 10) & 0x1f) as i32;
    let fraction = (bits & 0x3ff) as f32;

    match exponent {
        0 => sign * fraction * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        e => sign * (1.0 + fraction / 1024.0) * 2f32.powi(e - 15),
    }
}

fn halves(word: u32) -> [f32; 2] {
    [half(word & 0xffff), half(word >> 16)]
}

fn unorms(word: u32) -> [f32; 2] {
    [(word & 0xffff) as f32 / 65535.0, (word >> 16) as f32 / 65535.0]
}

fn base_config(dims: UVec3, dx: f32) -> SolverConfig {
    SolverConfig {
        dims,
        dx,
        max_particles: 200_000,
        cfl: 3.0,
        max_substeps: 6,
        whitewater_capacity: 1 << 14,
        narrow_band: Some(2.0),
        block_smoothing: true,
        smoothing: 4,
        pressure_iterations: 5,
        tolerance: 0.0,
        ..SolverConfig::default()
    }
}

fn swirling(config: &SolverConfig, hi: Vec3) -> Vec<GpuParticle> {
    let mut particles = seed_box(config, Vec3::ZERO, hi);

    for (n, p) in particles.iter_mut().enumerate() {
        let s = n as f32 * 0.37;
        p.v = [s.sin() * 1.5, s.cos() * 0.8, (2.0 * s).sin() * 2.5, 0.0];
        p.c = [[3.0 * s.cos(), -12.0, 0.7, 0.0], [-0.3, 25.0 * s.sin(), 4.0, 0.0], [9.0, 0.03, -40.0 * s.cos(), 0.0]];
    }

    particles
}

fn sorted_solver(device: &wgpu::Device, queue: &wgpu::Queue, config: SolverConfig, particles: &[GpuParticle]) -> GpuSolver {
    let mut solver = GpuSolver::new(device, config, None);
    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, particles);

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        solver.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
        solver.encode_sort(&mut pass);
        solver.dispatch(&mut pass, "p2g", Bind::Sim, Slots::FACES);
    }
    queue.submit([encoder.finish()]);

    solver
}

#[test]
fn quantized_particles_round_trip_within_half_precision() {
    let (device, queue) = gpu();
    let config = SolverConfig { quantized: true, sparse: false, ..base_config(UVec3::new(32, 16, 24), 1.0 / 16.0) };
    let size = config.dims.as_vec3() * config.dx;
    let particles = swirling(&config, Vec3::new(1.0, 0.6, size.z));
    let solver = sorted_solver(&device, &queue, config.clone(), &particles);

    let count = particles.len();
    let capacity = config.max_particles as usize;
    let sorted: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, count);
    let words: Vec<[u32; 2]> = read(&device, &queue, &solver.buffers.packed, capacity + 3 * count);
    let mut worst_position = Vec3::ZERO;
    let mut worst_relative = 0.0f32;

    for (i, p) in sorted.iter().enumerate() {
        let [x, y] = unorms(words[i][0]);
        let [z, _] = unorms(words[i][1]);
        let error = (Vec3::new(x, y, z) * size - Vec3::new(p.x[0], p.x[1], p.x[2])).abs();
        worst_position = worst_position.max(error);

        let motion = &words[capacity + 3 * i..capacity + 3 * i + 3];
        let unpacked: Vec<f32> = motion.iter().flat_map(|w| w.iter().flat_map(|h| halves(*h))).collect();
        let exact = [p.v[0], p.v[1], p.v[2], p.c[0][0], p.c[0][1], p.c[0][2], p.c[1][0], p.c[1][1], p.c[1][2], p.c[2][0], p.c[2][1], p.c[2][2]];

        for (q, e) in unpacked.iter().zip(exact) {
            let relative = (q - e).abs() / e.abs().max(1.0e-3);
            worst_relative = worst_relative.max(relative);
        }
    }

    let step = size / 65535.0;

    println!("{count} particles: worst position error {worst_position:?} m (step {step:?}), worst velocity/affine relative error {worst_relative:.3e}");

    assert!(worst_position.cmple(step * 0.5 + 1.0e-6).all());
    assert!(worst_relative <= 2f32.powi(-10) + 1.0e-6);
}

#[test]
fn quantized_p2g_matches_full_precision() {
    let (device, queue) = gpu();
    let config = SolverConfig { sparse: false, ..base_config(UVec3::new(32, 16, 24), 1.0 / 16.0) };
    let size = config.dims.as_vec3() * config.dx;
    let particles = swirling(&config, Vec3::new(1.0, 0.6, size.z));

    let full = sorted_solver(&device, &queue, config.clone(), &particles);
    let packed = sorted_solver(&device, &queue, SolverConfig { quantized: true, ..config.clone() }, &particles);

    let faces = |buffer: &wgpu::Buffer| -> Vec<f32> { read(&device, &queue, buffer, ((buffer.size() - GUARD_BYTES) / 4) as usize) };
    let (v_full, m_full) = (faces(&full.buffers.velocity), faces(&full.buffers.valid));
    let (v_packed, m_packed) = (faces(&packed.buffers.velocity), faces(&packed.buffers.valid));
    let vmax = particles.iter().fold(0.0f32, |m, p| m.max(Vec3::new(p.v[0], p.v[1], p.v[2]).length()));
    let cmax = particles.iter().flat_map(|p| p.c.iter().flatten()).fold(0.0f32, |m, c| m.max(c.abs()));
    let momentum = vmax + 3.0 * cmax * 1.5 * config.dx;
    let mmax = m_full.iter().fold(0.0f32, |m, v| m.max(*v));
    let mut worst_v = 0.0f32;
    let mut worst_m = 0.0f32;

    for n in 0..v_full.len() {
        worst_m = worst_m.max((m_full[n] - m_packed[n]).abs());

        if m_full[n] > 1.0e-3 {
            worst_v = worst_v.max((v_full[n] - v_packed[n]).abs());
        }
    }

    println!("quantized p2g: max |dv| {worst_v:.3e} m/s against momentum bound {momentum:.3}, max |dm| {worst_m:.3e} of max mass {mmax:.3}");

    assert!(worst_v < 2f32.powi(-10) * momentum);
    assert!(worst_m < 1.0e-3 * mmax);
}

#[test]
fn game_preset_runs_finite_with_bounded_volume() {
    let (device, queue) = gpu();
    let config = base_config(UVec3::new(48, 24, 32), 1.0 / 16.0).with_quality(Quality::Game);
    let size = config.dims.as_vec3() * config.dx;
    let mut particles = seed_box(&config, Vec3::ZERO, Vec3::new(0.9, 0.9, size.z));
    particles.extend(seed_box(&config, Vec3::new(0.9, 0.0, 0.0), Vec3::new(size.x, 0.35, size.z)));

    for (i, p) in particles.iter_mut().enumerate() {
        p.x[3] = i as f32;
    }

    let wood = LiquidBody { shape: LiquidShape::Cuboid(Vec3::new(0.2, 0.12, 0.3)), density: 450.0, dynamic: true };
    let bodies: Vec<_> = [Vec3::new(1.7, 0.4, 0.6), Vec3::new(2.3, 0.4, 1.4)]
        .into_iter()
        .map(|p| body_to_gpu(&wood, &GlobalTransform::from_translation(p), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO))
        .collect();

    let mut solver = GpuSolver::new(&device, config.clone(), None);
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, 1.0 / 60.0);
    solver.upload_particles(&queue, &particles);

    let mut first = None;
    let mut worst = 0.0f64;

    for _ in 0..180 {
        solver.upload_bodies(&queue, &bodies);
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        let volume = read::<f32>(&device, &queue, &solver.buffers.scalars, SCALAR_COUNT)[S_VOLUME] as f64;
        let reference = *first.get_or_insert(volume);
        worst = worst.max((volume / reference - 1.0).abs());
    }

    let state: GpuState = read(&device, &queue, &solver.buffers.state, 1)[0];
    let alive: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, state.particle_count as usize);
    let finite = alive.iter().all(|p| p.x.iter().chain(&p.v).chain(p.c.iter().flatten()).all(|f| f.is_finite()));
    let violations = solver.guard_violations(&device, &queue);

    println!(
        "game preset {:?} dx {:.4}: {} particles from {} seeded, worst volume deviation {worst:.4}, finite {finite}",
        config.dims,
        config.dx,
        alive.len(),
        particles.len()
    );

    assert_eq!(config.particles_per_cell, 4);
    assert!(config.late_surface && config.cull_crowded && !config.quantized);
    assert!(finite);
    assert!(!alive.is_empty());
    assert!(worst < 0.05);
    assert!(violations.is_empty(), "{violations:?}");
}

#[cfg(windows)]
#[test]
fn game_preset_pipelines_compile_under_fxc() {
    let base = base_config(UVec3::new(40, 24, 32), 1.0 / 16.0);

    common::assert_compiles_under_fxc(|device, _| {
        for sparse in [false, true] {
            GpuSolver::new(device, SolverConfig { sparse, ..base.clone() }.with_quality(Quality::Game), None);
        }
    });
}

struct Packing {
    particles: usize,
    moving: f64,
    settled: f64,
    crowded: f64,
}

fn dam_break_packing(device: &wgpu::Device, queue: &wgpu::Queue, config: &SolverConfig) -> Packing {
    let size = config.dims.as_vec3() * config.dx;
    let particles = seed_box(config, Vec3::ZERO, Vec3::new(0.9, 1.2, size.z));
    let seeded = particles.len() as f64 / config.particles_per_cell as f64;
    let mut solver = GpuSolver::new(device, config.clone(), None);
    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, &particles);

    let mut occupied = Vec::new();
    let mut crowded = Vec::new();
    let mut alive = particles.len();

    for frame in 1..=1800 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        if frame % 100 != 0 {
            continue;
        }

        let state: GpuState = read(device, queue, &solver.buffers.state, 1)[0];
        let now: Vec<GpuParticle> = read(device, queue, &solver.buffers.particles, state.particle_count as usize);
        let mut counts = std::collections::HashMap::new();

        for p in &now {
            let cell = (Vec3::new(p.x[0], p.x[1], p.x[2]) / config.dx).floor().as_ivec3();
            *counts.entry(cell).or_insert(0u32) += 1;
        }

        assert!(now.iter().all(|p| p.x.iter().chain(&p.v).all(|f| f.is_finite())));
        alive = now.len();
        occupied.push(counts.len() as f64 / seeded);
        crowded.push(counts.values().filter(|n| **n > 2 * config.particles_per_cell).sum::<u32>() as f64 / now.len() as f64);
    }

    assert!(solver.guard_violations(device, queue).is_empty());

    let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;

    Packing { particles: alive, moving: mean(&occupied[..6]), settled: mean(&occupied[8..]), crowded: mean(&crowded) }
}

#[test]
fn shared_density_grid_keeps_particles_spread() {
    let (device, queue) = gpu();
    let game = SolverConfig {
        dims: UVec3::new(48, 24, 32),
        dx: 1.0 / 16.0,
        max_particles: 200_000,
        cfl: 2.5,
        tolerance: 1e-3,
        ..SolverConfig::default()
    }
    .with_quality(Quality::Game);
    let separate = SolverConfig { shared_density_grid: false, density_iterations: 5, ..game.clone() };
    let off = SolverConfig { density_projection: false, ..game.clone() };

    let [shared, separate, off] = [&game, &separate, &off].map(|config| {
        let p = dam_break_packing(&device, &queue, config);

        println!(
            "shared grid {} projection {} iterations {}: {} particles, occupied volume over seeded: moving {:.3} settled {:.3}, share in cells over twice the seeded count {:.4}",
            config.shared_density_grid, config.density_projection, config.density_iterations, p.particles, p.moving, p.settled, p.crowded
        );

        p
    });

    assert!(game.shared_density_grid);
    assert_eq!(shared.particles, separate.particles);
    assert!(shared.moving > 0.98 * separate.moving);
    assert!(shared.settled > 0.98 * separate.settled);
    assert!(shared.crowded < separate.crowded + 0.01);
    assert!(shared.settled > 1.2 * off.settled);
}

struct FloorSample {
    particles: usize,
    inflow: usize,
    level: f32,
    reservoir: f32,
    speed: f32,
}

fn open_floor_run(device: &wgpu::Device, queue: &wgpu::Queue, config: &SolverConfig, particles: &[GpuParticle], frames: &[usize]) -> Vec<FloorSample> {
    let mut solver = GpuSolver::new(device, config.clone(), None);
    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, particles);

    let mut samples = Vec::new();

    for frame in 1..=*frames.last().expect("no sample frames") {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        if frames.contains(&frame) {
            let state: GpuState = read(device, queue, &solver.buffers.state, 1)[0];
            let stored: Vec<GpuParticle> = read(device, queue, &solver.buffers.particles, state.particle_count as usize);
            let alive: Vec<&GpuParticle> = stored.iter().filter(|p| p.v[3] <= 0.5).collect();

            assert!(alive.iter().all(|p| p.x.iter().chain(&p.v).all(|f| f.is_finite())));
            let scalars: Vec<f32> = read(device, queue, &solver.buffers.scalars, SCALAR_COUNT);

            samples.push(FloorSample {
                particles: alive.len(),
                inflow: alive.iter().filter(|p| p.x[3] < 0.0).count(),
                level: scalars[S_FLOOR_MEAN],
                reservoir: scalars[S_FLOOR_LEVEL],
                speed: alive.iter().map(|p| Vec3::new(p.v[0], p.v[1], p.v[2]).length()).fold(0.0, f32::max),
            });
        }
    }

    assert!(solver.guard_violations(device, queue).is_empty());

    samples
}

fn standing_wave(config: &SolverConfig, depth: f32, amplitude: f32) -> Vec<GpuParticle> {
    let size = config.dims.as_vec3() * config.dx;
    let deep = 1.0f32;
    let k = std::f32::consts::PI / size.x;
    let omega = (9.81 * k * (k * deep).tanh()).sqrt();
    let a = amplitude * 9.81 * k / (omega * (k * deep).cosh());
    let mut particles = seed_box(config, Vec3::ZERO, Vec3::new(size.x, depth, size.z));

    for p in &mut particles {
        let y = p.x[1] - depth + deep;
        p.v = [-a * (k * y).cosh() * (k * p.x[0]).sin(), a * (k * y).sinh() * (k * p.x[0]).cos(), 0.0, 0.0];
    }

    particles
}

#[test]
fn open_floor_inflow_follows_particles_per_cell() {
    let (device, queue) = gpu();
    let high = SolverConfig {
        dims: UVec3::new(32, 16, 8),
        dx: 1.0 / 32.0,
        max_particles: 200_000,
        open_floor: Some(3.0),
        density_projection: false,
        ..SolverConfig::default()
    };
    let game = high.clone().with_quality(Quality::Game);
    let depth = 0.3125;

    let [high_wave, game_wave] = [&high, &game].map(|config| {
        let particles = standing_wave(config, depth, 1.0 / 32.0);
        let samples = open_floor_run(&device, &queue, config, &particles, &[18, 180]);
        let volume = (config.dx as f64).powi(3) / config.particles_per_cell as f64;
        let inflow = samples[0].inflow as f64 * volume;
        let change = samples[1].particles as f64 / particles.len() as f64 - 1.0;

        println!(
            "standing wave {} {:?} at {} per cell: {} seeded, {} particles = {inflow:.3e} m3 through the floor after 0.3 s, count change {change:+.4} after 3 s",
            config.quality.name(),
            config.dims,
            config.particles_per_cell,
            particles.len(),
            samples[0].inflow
        );

        (inflow, change)
    });

    let still = standing_wave(&game, depth, 0.0);
    let rest = open_floor_run(&device, &queue, &game, &still, &[120]);

    println!("still pool game: {} of {} particles after 2 s, {} through the floor", rest[0].particles, still.len(), rest[0].inflow);

    assert_eq!(game.particles_per_cell, 4);
    assert!(game_wave.0 > 0.6 * high_wave.0 && game_wave.0 < 1.4 * high_wave.0);
    assert!((game_wave.1 - high_wave.1).abs() < 0.03);
    assert!(rest[0].particles.abs_diff(still.len()) * 200 <= still.len());
}

fn floor_tank() -> [SolverConfig; 2] {
    let high = SolverConfig {
        dims: UVec3::new(32, 16, 8),
        dx: 1.0 / 32.0,
        max_particles: 200_000,
        open_floor: Some(3.0),
        ..SolverConfig::default()
    };
    let game = high.clone().with_quality(Quality::Game);

    [high, game]
}

#[test]
fn open_floor_keeps_the_particle_count_over_a_slosh() {
    let (device, queue) = gpu();
    let depth = 0.3125;
    let frames: Vec<usize> = (1..=20).map(|k| 30 * k).collect();

    for config in &floor_tank() {
        let particles = standing_wave(config, depth, 1.0 / 32.0);
        let samples = open_floor_run(&device, &queue, config, &particles, &frames);
        let change: Vec<f64> = samples.iter().map(|s| s.particles as f64 / particles.len() as f64 - 1.0).collect();
        let (fewest, most) = change.iter().fold((f64::MAX, f64::MIN), |(l, h), c| (l.min(*c), h.max(*c)));
        let drift = samples.iter().map(|s| (s.level - s.reservoir).abs()).fold(0.0, f32::max);

        println!(
            "standing wave {} for 10 s: {} seeded, count change between {fewest:+.4} and {most:+.4}, reservoir level {:.4} m, mean level at most {drift:.4} m away",
            config.quality.name(),
            particles.len(),
            samples[0].reservoir
        );

        assert!(config.density_projection);
        assert!(fewest > -0.04 && most < 0.04);
        assert!((samples[0].reservoir - depth).abs() < 0.1 * config.dx);
        assert!(drift < 0.02 * depth);
    }
}

#[test]
fn open_floor_returns_to_the_reservoir_level() {
    let (device, queue) = gpu();
    let depth = 0.3125;
    let frames: Vec<usize> = (1..=180).collect();

    for config in &floor_tank() {
        let still = standing_wave(config, depth, 0.0);
        let rest = &open_floor_run(&device, &queue, config, &still, &[120])[0];
        let gained = rest.particles as i64 - still.len() as i64;
        let strip = (0.05 * (config.particles_per_cell * config.dims.x * config.dims.z) as f32).ceil() as i64;
        let mut kicked = still.clone();
        kicked.iter_mut().for_each(|p| p.v[1] = 0.2);

        let samples = open_floor_run(&device, &queue, config, &kicked, &frames);
        let rise: Vec<f32> = samples.iter().map(|s| s.level - rest.level).collect();
        let peak = rise.iter().copied().fold(0.0, f32::max);
        let settled = ((0..rise.len()).rev().take_while(|&f| rise[f].abs() < 0.01 * depth).last().unwrap_or(rise.len()) + 1) as f32 / 60.0;
        let change = samples[179].particles as f64 / still.len() as f64 - 1.0;

        println!(
            "{} still pool for 2 s: {gained:+} of {} particles against {strip} that fit in the empty seed strip at the floor, mean level {:.5} m against reservoir {:.5} m, max speed {:.2e} m/s",
            config.quality.name(),
            still.len(),
            rest.level,
            rest.reservoir,
            rest.speed
        );
        println!(
            "{} pool kicked up at 0.2 m/s: mean level peaks {peak:.4} m above rest and stays within 1 % of the depth from {settled:.2} s, count change {change:+.4} at 3 s",
            config.quality.name()
        );

        assert!(peak > 0.012 && peak < 0.025);
        assert!(settled < 1.0);
        assert!(change.abs() < 0.02);
        assert!(gained.abs() * 100 <= still.len() as i64);
        assert!((rest.level - rest.reservoir).abs() < 1e-4);
        assert!(rest.speed < 0.01 * (9.81 * depth).sqrt());
    }
}
