use bevy::math::{UVec3, Vec3};
use glam::DVec3;
use liquid_gpu::layout::{GpuParticle, GpuState, S_VISCOUS_ITERATIONS, SCALAR_COUNT};
use liquid_gpu::plugin::seed_box;
use liquid_gpu::solver::{Bind, Diffusion, GpuSolver, Quality, SolverConfig};
use liquid_ref::grid::{Grid, MacField};
use liquid_ref::levelset::{face_weights, solid_phi_nodes};
use liquid_ref::particles::Particle;
use liquid_ref::scene::dam_break;
use liquid_ref::sim::{Params, Simulation};
use liquid_ref::viscosity::{Fractions, ViscositySolve, flatten, unflatten};

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

const LAMB_ROOT: f64 = 5.763_459_196_894_55;

const RADIUS: f64 = 0.1;

const NU: f64 = 0.01;

const FALL_FRAMES: usize = 10;

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
        label: Some("viscosity test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits::portable(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device")
}

fn read<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let bytes = (count * std::mem::size_of::<T>()) as u64;
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
    bytemuck::cast_slice(&view).to_vec()
}

fn validated<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> T {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = build();

    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation error: {error}");
    }

    out
}

fn to_f32(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}

struct Ball {
    grid: Grid,

    fractions: Fractions,

    weights: MacField,

    center: DVec3,

    phi: Vec<f64>,
}

impl Ball {
    fn new(cells_per_radius: u32) -> Self {
        let n = 2 * cells_per_radius + 6;
        let grid = Grid::new(glam::UVec3::splat(n), RADIUS / cells_per_radius as f64);
        let center = grid.size() * 0.5;
        let mut phi = grid.cells();

        for i in 0..phi.data.len() {
            phi.data[i] = (grid.cell_center(phi.coords(i)) - center).length() - RADIUS;
        }

        let weights = face_weights(&grid, &solid_phi_nodes(&grid, &[]));
        let fractions = Fractions::new(&grid, &phi);

        Self { grid, fractions, weights, center, phi: phi.data }
    }

    fn field(&self, f: impl Fn(DVec3) -> DVec3) -> MacField {
        let mut u = MacField::new(&self.grid);

        for (axis, field) in u.axes.iter_mut().enumerate() {
            for n in 0..field.data.len() {
                if self.fractions.faces.axes[axis].data[n] > 0.0 {
                    let x = self.grid.face_position(axis, field.coords(n));
                    field.data[n] = f(x - self.center)[axis];
                }
            }
        }

        u
    }

    fn mode(&self) -> MacField {
        let k = LAMB_ROOT / RADIUS;

        self.field(|r| {
            let s = r.length();
            let swirl = DVec3::new(-r.y, r.x, 0.0);

            if s > 0.0 { swirl * spherical_j1(k * s) / s } else { DVec3::ZERO }
        })
    }

    fn inner(&self, a: &MacField, b: &MacField) -> f64 {
        (0..3)
            .map(|axis| {
                let w = &self.fractions.faces.axes[axis].data;
                let (a, b) = (&a.axes[axis].data, &b.axes[axis].data);

                (0..w.len()).map(|n| w[n] * a[n] * b[n]).sum::<f64>()
            })
            .sum()
    }

    fn cpu_diffuse(&self, u: &mut MacField, dt: f64) {
        ViscositySolve {
            grid: &self.grid,
            fractions: &self.fractions,
            weights: &self.weights,
            kinematic_viscosity: NU,
            dt,
            tolerance: 1e-10,
            max_iterations: 2000,
        }
        .apply(u);
    }
}

fn spherical_j1(x: f64) -> f64 {
    if x < 1e-6 { x / 3.0 } else { x.sin() / (x * x) - x.cos() / x }
}

struct GpuBall {
    grid: Grid,
    device: wgpu::Device,
    queue: wgpu::Queue,
    solver: GpuSolver,
    faces: usize,
}

impl GpuBall {
    fn new(ball: &Ball, dt: f64) -> Self {
        let (device, queue) = gpu();
        let config = SolverConfig {
            dims: UVec3::splat(ball.grid.dims.x),
            dx: ball.grid.dx as f32,
            max_particles: 64,
            viscosity: (NU * 1000.0) as f32,
            tolerance: 1e-6,
            viscosity_iterations: 800,
            sparse: false,
            ..SolverConfig::default()
        };
        let solver = validated(&device, || GpuSolver::new(&device, config, None));
        solver.initialize(&queue);

        let b = &solver.buffers;
        let state = GpuState { dt: dt as f32, ..GpuState::default() };
        queue.write_buffer(&b.state, 0, bytemuck::bytes_of(&state));
        queue.write_buffer(&b.phi, 0, bytemuck::cast_slice(&to_f32(&ball.phi)));
        queue.write_buffer(&b.weights, 0, bytemuck::cast_slice(&to_f32(&flatten(&ball.weights))));

        let faces = flatten(&ball.weights).len();

        Self { grid: ball.grid, device, queue, solver, faces }
    }

    fn diffuse(&self, u: &MacField, solves: usize) -> (MacField, f32) {
        let b = &self.solver.buffers;
        self.queue.write_buffer(&b.velocity, 0, bytemuck::cast_slice(&to_f32(&flatten(u))));
        let mut worst = 0.0f32;

        for _ in 0..solves {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                self.solver.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
                self.solver.encode_diffuse(&mut pass, Diffusion::Viscosity);
            }
            self.queue.submit([encoder.finish()]);

            let scalars: Vec<f32> = read(&self.device, &self.queue, &b.scalars, SCALAR_COUNT);
            worst = worst.max(scalars[S_VISCOUS_ITERATIONS]);
        }

        let values: Vec<f32> = read(&self.device, &self.queue, &b.velocity, self.faces);
        let solved: Vec<f64> = values.iter().map(|v| *v as f64).collect();

        (unflatten(&self.grid, &solved), worst)
    }
}

fn decay_rate(ball: &Ball, u: &MacField, mode: &MacField, solves: usize, dt: f64) -> f64 {
    let amplitude = ball.inner(u, mode) / ball.inner(mode, mode);

    ((1.0 / amplitude).powf(1.0 / solves as f64) - 1.0) / dt
}

#[test]
fn gpu_toroidal_mode_decays_like_reference_and_lamb() {
    let k = LAMB_ROOT / RADIUS;
    let exact = NU * k * k;
    let dt = 0.1 / exact;
    let solves = 8;
    let mut errors = Vec::new();

    for cells in [5, 10, 20] {
        let ball = Ball::new(cells);
        let mode = ball.mode();
        let gpu = GpuBall::new(&ball, dt);
        let (u, iterations) = gpu.diffuse(&mode, solves);
        let rate = decay_rate(&ball, &u, &mode, solves, dt);
        let error = (rate - exact).abs() / exact;

        if cells <= 10 {
            let mut cpu = mode.clone();

            for _ in 0..solves {
                ball.cpu_diffuse(&mut cpu, dt);
            }

            let reference = decay_rate(&ball, &cpu, &mode, solves, dt);
            let difference = (rate - reference).abs() / reference;

            println!(
                "radius {cells} cells: gpu rate {rate:.5} 1/s, cpu rate {reference:.5} 1/s (relative difference {difference:.2e}), Lamb {exact:.5}, error {:.2}%, max cg iterations {iterations}",
                100.0 * error
            );

            assert!(difference < 1e-4, "gpu {rate} vs cpu {reference}");
        } else {
            println!("radius {cells} cells: gpu rate {rate:.5} 1/s vs Lamb {exact:.5}, error {:.2}%, max cg iterations {iterations}", 100.0 * error);
        }

        errors.push(error);
    }

    assert!(errors[1] < errors[0] && errors[2] < errors[1]);
    assert!(errors[2] < 0.05);
}

#[test]
fn gpu_viscous_solve_matches_reference_when_bricks_outnumber_face_groups() {
    let k = LAMB_ROOT / RADIUS;
    let dt = 0.1 / (NU * k * k);
    let solves = 4;
    let ball = Ball::new(3);
    let mode = ball.mode();
    let gpu = GpuBall::new(&ball, dt);
    let (u, iterations) = gpu.diffuse(&mode, solves);
    let mut cpu = mode.clone();

    for _ in 0..solves {
        ball.cpu_diffuse(&mut cpu, dt);
    }

    let rate = decay_rate(&ball, &u, &mode, solves, dt);
    let reference = decay_rate(&ball, &cpu, &mode, solves, dt);
    let difference = (rate - reference).abs() / reference;

    println!("12 cell dense grid: gpu rate {rate:.5} 1/s, cpu rate {reference:.5} 1/s (relative difference {difference:.2e}), max cg iterations {iterations}");

    assert!(difference < 1e-4, "gpu {rate} vs cpu {reference}");
}

#[test]
fn gpu_rigid_rotation_survives_strong_viscosity() {
    let ball = Ball::new(10);
    let dt = 50.0 * ball.grid.dx * ball.grid.dx / NU;
    let rotation = ball.field(|r| DVec3::new(-r.y, r.x, 0.0));
    let gpu = GpuBall::new(&ball, dt);
    let (u, iterations) = gpu.diffuse(&rotation, 5);
    let ratio = ball.inner(&u, &rotation) / ball.inner(&rotation, &rotation);

    println!("angular momentum kept after 5 gpu solves at dt = 50 dx^2/nu: {ratio:.7} (max cg iterations {iterations})");

    assert!((ratio - 1.0).abs() < 1e-4);
}

fn to_gpu(id: usize, p: &Particle) -> GpuParticle {
    let f = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    let mut x = f(p.x);
    x[3] = id as f32;

    GpuParticle { x, v: f(p.v), c: [f(p.c[0]), f(p.c[1]), f(p.c[2])] }
}

fn column_config(sim: &Simulation) -> SolverConfig {
    SolverConfig {
        dims: UVec3::new(sim.grid.dims.x, sim.grid.dims.y, sim.grid.dims.z),
        dx: sim.grid.dx as f32,
        max_particles: sim.particles.len() as u32 + 1024,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        max_substeps: 1,
        pressure_iterations: 60,
        density_iterations: 30,
        tolerance: 1e-6,
        coarse_sweeps: 16,
        density_projection: sim.params.density_projection,
        viscosity: sim.params.viscosity as f32,
        viscosity_iterations: 300,
        ..SolverConfig::default()
    }
}

#[test]
fn gpu_viscous_substep_matches_reference() {
    let (device, queue) = gpu();
    let params = Params { viscosity: 2.0, density_projection: false, ..Params::default() };
    let mut sim = dam_break(glam::UVec3::new(24, 16, 8), 0.02, DVec3::new(0.16, 0.24, 0.0), params);

    for _ in 0..3 {
        let dt = sim.cfl_dt();
        sim.substep(dt);
    }

    let start = sim.particles.clone();
    let frame_dt = (0.999 * sim.cfl_dt()) as f32;
    let mut solver = validated(&device, || GpuSolver::new(&device, column_config(&sim), None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, frame_dt);
    solver.upload_particles(&queue, &start.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let gpu_particles: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, start.len());
    let scalars: Vec<f32> = read(&device, &queue, &solver.buffers.scalars, SCALAR_COUNT);
    sim.substep(frame_dt as f64);

    let mut dx = 0.0f64;
    let mut dv = 0.0f64;
    let vmax = sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length()));

    for p in &gpu_particles {
        let cpu = &sim.particles[p.x[3] as usize];
        let x = DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64);
        let v = DVec3::new(p.v[0] as f64, p.v[1] as f64, p.v[2] as f64);
        dx = dx.max((x - cpu.x).length());
        dv = dv.max((v - cpu.v).length());
    }

    let cpu_iterations = sim.report.viscosity.map_or(0, |s| s.iterations);

    println!(
        "viscous substep mu {} Pa s: max |dx| {dx:.3e} m ({:.3e} cells), max |dv| {dv:.3e} m/s of vmax {vmax:.3}, cg iterations gpu {} cpu {cpu_iterations}",
        params.viscosity,
        dx / sim.grid.dx,
        scalars[S_VISCOUS_ITERATIONS]
    );

    assert!(dx / sim.grid.dx < 1e-3);
    assert!(dv / vmax < 1e-3);
}

struct ColumnRun {
    front: f64,

    max_speed: f64,

    min_dt: f64,

    max_iterations: f32,

    finite: bool,
}

fn collapse_column(device: &wgpu::Device, queue: &wgpu::Queue, viscosity: f64) -> ColumnRun {
    let column = DVec3::new(0.06, 0.1, 0.0);
    let params = Params { viscosity, ..Params::default() };
    let sim = dam_break(glam::UVec3::new(40, 32, 4), 0.005, column, params);
    let config = SolverConfig { max_particles: sim.particles.len() as u32, tolerance: 1e-5, ..column_config(&sim) };
    let mut solver = validated(device, || GpuSolver::new(device, config, None));
    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    let mut run = ColumnRun { front: 0.0, max_speed: 0.0, min_dt: f64::INFINITY, max_iterations: 0.0, finite: true };
    let mut time = 0.0;
    let mut particles = Vec::new();

    while time < 0.4 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        let state: GpuState = read(device, queue, &solver.buffers.state, 1)[0];
        let scalars: Vec<f32> = read(device, queue, &solver.buffers.scalars, SCALAR_COUNT);
        particles = read::<GpuParticle>(device, queue, &solver.buffers.particles, state.particle_count as usize);

        time = state.time as f64;
        run.min_dt = run.min_dt.min(state.dt as f64);
        run.max_iterations = run.max_iterations.max(scalars[S_VISCOUS_ITERATIONS]);
        run.finite &= particles.iter().all(|p| p.x.iter().chain(&p.v).all(|v| v.is_finite()));
        run.max_speed = particles.iter().fold(run.max_speed, |m, p| m.max(Vec3::new(p.v[0], p.v[1], p.v[2]).length() as f64));
    }

    run.front = particles.iter().filter(|p| (p.x[1] as f64) < 0.25 * column.y).fold(0.0, |m, p| m.max(p.x[0] as f64));

    run
}

#[test]
fn gpu_stiff_viscosity_is_stable_far_above_explicit_limit() {
    let (device, queue) = gpu();
    let mu = 10.0;
    let dx: f64 = 0.005;
    let explicit_limit = dx * dx / (6.0 * mu / 1000.0);
    let viscous = collapse_column(&device, &queue, mu);
    let inviscid = collapse_column(&device, &queue, 0.0);
    let free_fall = (2.0 * 9.81 * 0.1f64).sqrt();

    println!(
        "gpu mu {mu} Pa s: min dt {:.2e} s = {:.1}x explicit limit {explicit_limit:.2e} s, max cg iterations {}, max speed {:.3} m/s (free fall {free_fall:.3}), front {:.4} m vs inviscid {:.4} m from 0.06 m",
        viscous.min_dt,
        viscous.min_dt / explicit_limit,
        viscous.max_iterations,
        viscous.max_speed,
        viscous.front,
        inviscid.front,
    );

    assert!(viscous.finite);
    assert!(viscous.min_dt > 10.0 * explicit_limit);
    assert!(viscous.max_iterations < 300.0);
    assert!(viscous.max_speed < free_fall);
    assert!(viscous.front - 0.06 < 0.5 * (inviscid.front - 0.06));
}

fn falling_thread(device: &wgpu::Device, queue: &wgpu::Queue, quality: Quality) -> (Vec3, f32) {
    let dims = UVec3::new(16, 40, 16);
    let dx = 0.02;
    let config = SolverConfig {
        dims,
        dx,
        max_particles: 8192,
        density: 1420.0,
        viscosity: 1500.0,
        cfl: 2.5,
        tolerance: 1e-3,
        viscosity_iterations: 40,
        ..SolverConfig::default()
    }
    .with_quality(quality);
    let config = SolverConfig { dims, dx, ..config };
    let size = dims.as_vec3() * dx;
    let center = Vec3::new(0.5 * size.x + 0.3 * dx, size.y - 14.0 * dx, 0.5 * size.z + 0.2 * dx);
    let mut seed = seed_box(&config, Vec3::ZERO, size);

    seed.retain(|p| {
        let r = Vec3::new(p.x[0], p.x[1], p.x[2]) - center;

        r.y.abs() < 10.0 * dx && r.x.hypot(r.z) < 1.5 * dx
    });

    for (i, p) in seed.iter_mut().enumerate() {
        p.x[3] = i as f32;
    }

    let mut solver = validated(device, || GpuSolver::new(device, config, None));
    solver.initialize(queue);
    solver.upload_particles(queue, &seed);

    for _ in 0..FALL_FRAMES {
        solver.set_frame_dt(queue, 1.0 / 60.0);

        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);
    }

    let particles: Vec<GpuParticle> = read(device, queue, &solver.buffers.particles, seed.len());
    let count = particles.len() as f32;
    let mean = particles.iter().map(|p| Vec3::new(p.v[0], p.v[1], p.v[2])).sum::<Vec3>() / count;
    let lateral = (particles.iter().map(|p| p.v[0] * p.v[0] + p.v[2] * p.v[2]).sum::<f32>() / count).sqrt();

    (mean, lateral)
}

#[test]
fn gpu_free_falling_viscous_thread_gains_no_lateral_momentum() {
    let (device, queue) = gpu();

    for quality in [Quality::High, Quality::Game] {
        let (mean, lateral) = falling_thread(&device, &queue, quality);
        let fall = 9.81 * FALL_FRAMES as f32 / 60.0;

        println!(
            "{} thread after {FALL_FRAMES} frames: fall {:.4} m/s, mean lateral ({:.2e}, {:.2e}) m/s, lateral rms {lateral:.2e} m/s",
            quality.name(),
            -mean.y,
            mean.x,
            mean.z
        );

        assert!((mean.y + fall).abs() < 1e-3 * fall);
        assert!(mean.x.abs() < 1e-6 && mean.z.abs() < 1e-6);
        assert!(lateral < 1e-5);
    }
}
