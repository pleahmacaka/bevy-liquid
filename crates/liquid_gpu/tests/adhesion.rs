use bevy::math::{UVec3, Vec3};
use glam::DVec3;
use liquid_gpu::layout::{GpuBody, GpuParticle, GpuState, S_VISCOUS_ITERATIONS, SCALAR_COUNT};
use liquid_gpu::plugin::LiquidProperties;
use liquid_gpu::solver::{Bind, GpuSolver, Slots, SolverConfig};
use liquid_ref::adhesion::adhere;
use liquid_ref::grid::{Grid, MacField};
use liquid_ref::levelset::{face_weights, solid_phi_nodes};
use liquid_ref::particles::{Particle, seed_box};
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::SEED_JITTER;
use liquid_ref::sim::{Params, Simulation};
use liquid_ref::viscosity::flatten;

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
        label: Some("adhesion test"),
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
        velocity: f(b.velocity, 0.0),
        angular: f(b.angular_velocity, 0.0),
        shape: f(shape, kind),
        inv_inertia: [f(inv.row(0), 0.0), f(inv.row(1), 0.0), f(inv.row(2), 0.0)],
    }
}

const CEILING: f64 = 0.3;

fn sticky_bodies(grid: &Grid) -> Vec<RigidBody> {
    let size = grid.size();
    let slab = RigidBody::new(
        Shape::Cuboid { half: DVec3::new(size.x, 0.05, size.z) },
        DVec3::new(0.5 * size.x, CEILING + 0.05, 0.5 * size.z),
        Motion::Static,
        1000.0,
    );
    let ball = RigidBody::new(Shape::Sphere { radius: 0.05 }, DVec3::new(0.25, 0.1, 0.5 * size.z), Motion::Static, 1000.0);

    vec![slab, ball]
}

fn sticky_scene(params: Params) -> Simulation {
    let grid = Grid::new(glam::UVec3::new(20, 20, 8), 0.02);
    let size = grid.size();
    let mut particles = seed_box(&grid, DVec3::new(0.06, 0.2, 0.0), DVec3::new(0.34, CEILING - 0.03, size.z), SEED_JITTER);
    particles.extend(seed_box(&grid, DVec3::new(0.06, 0.06, 0.0), DVec3::new(0.16, 0.14, size.z), SEED_JITTER));

    Simulation::new(grid, params, particles, sticky_bodies(&grid))
}

#[test]
fn gpu_adhesion_kernel_matches_reference() {
    let (device, queue) = gpu();
    let grid = Grid::new(glam::UVec3::new(20, 20, 8), 0.02);
    let bodies = sticky_bodies(&grid);
    let weights = face_weights(&grid, &solid_phi_nodes(&grid, &bodies));
    let (strength, dt) = (1.5, 1.0 / 60.0);
    let mut velocity = MacField::new(&grid);

    for (axis, field) in velocity.axes.iter_mut().enumerate() {
        for n in 0..field.data.len() {
            let x = grid.face_position(axis, field.coords(n));
            field.data[n] = (7.0 * x.x + 3.0 * axis as f64).sin() * (5.0 * x.y).cos() * 0.3;
        }
    }

    let config = SolverConfig {
        dims: UVec3::new(grid.dims.x, grid.dims.y, grid.dims.z),
        dx: grid.dx as f32,
        max_particles: 64,
        adhesion: strength as f32,
        sparse: false,
        ..SolverConfig::default()
    };
    let mut solver = GpuSolver::new(&device, config, None);
    solver.initialize(&queue);
    solver.upload_bodies(&queue, &bodies.iter().map(to_gpu_body).collect::<Vec<_>>());

    let b = &solver.buffers;
    let to_f32 = |values: Vec<f64>| values.iter().map(|v| *v as f32).collect::<Vec<f32>>();
    queue.write_buffer(&b.state, 0, bytemuck::bytes_of(&GpuState { dt: dt as f32, ..GpuState::default() }));
    queue.write_buffer(&b.weights, 0, bytemuck::cast_slice(&to_f32(flatten(&weights))));
    queue.write_buffer(&b.velocity, 0, bytemuck::cast_slice(&to_f32(flatten(&velocity))));

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        solver.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
        solver.dispatch(&mut pass, "adhere", Bind::Sim, Slots::FACES);
    }
    queue.submit([encoder.finish()]);

    let before = flatten(&velocity);
    adhere(&grid, &bodies, &weights, &mut velocity, strength, dt);
    let cpu = flatten(&velocity);
    let gpu: Vec<f32> = read(&device, &queue, &solver.buffers.velocity, cpu.len());

    let pulled = cpu.iter().zip(&before).filter(|(a, b)| a != b).count();
    let largest = cpu.iter().zip(&before).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    let error = cpu.iter().zip(&gpu).fold(0.0f64, |m, (c, g)| m.max((c - *g as f64).abs()));

    println!("adhesion kernel: {pulled} faces pulled, largest change {largest:.4e} m/s, max gpu-cpu difference {error:.3e} m/s");

    assert!(pulled > 100);
    assert!(error < 1e-3 * largest);
}

struct FrameDifference {
    cells: f64,

    speed: f64,
}

fn frame_difference(params: Params, viscosity_iterations: u32) -> FrameDifference {
    let (device, queue) = gpu();
    let mut sim = sticky_scene(params);

    for _ in 0..3 {
        let dt = sim.cfl_dt();
        sim.substep(dt);
    }

    let start = sim.particles.clone();
    let frame_dt = (0.999 * sim.cfl_dt()) as f32;
    let config = SolverConfig {
        dims: UVec3::new(sim.grid.dims.x, sim.grid.dims.y, sim.grid.dims.z),
        dx: sim.grid.dx as f32,
        max_particles: start.len() as u32 + 1024,
        density: params.density as f32,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        max_substeps: 1,
        pressure_iterations: 60,
        tolerance: 1e-6,
        coarse_sweeps: 16,
        density_projection: false,
        viscosity: params.viscosity as f32,
        viscosity_iterations,
        surface_tension: params.surface_tension as f32,
        adhesion: params.adhesion as f32,
        ..SolverConfig::default()
    };
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(to_gpu_body).collect();
    let mut solver = GpuSolver::new(&device, config, None);
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, frame_dt);
    solver.upload_particles(&queue, &start.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    solver.upload_bodies(&queue, &bodies);

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let gpu_particles: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, start.len());
    let state: GpuState = read(&device, &queue, &solver.buffers.state, 1)[0];
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

    let cpu_solve = sim.report.viscosity.map_or(0, |s| s.iterations);

    println!(
        "adhesive frame {} J/kg with mu {} Pa s, sigma {} N/m, density {} kg/m3: dt gpu {:.6e} cpu {frame_dt:.6e}; max |dx| {dx:.3e} m ({:.3e} cells), max |dv| {dv:.3e} m/s of vmax {vmax:.3} (relative {:.3e}); cg iterations gpu {} cpu {cpu_solve}",
        params.adhesion,
        params.viscosity,
        params.surface_tension,
        params.density,
        state.dt,
        dx / sim.grid.dx,
        dv / vmax,
        scalars[S_VISCOUS_ITERATIONS]
    );

    assert_eq!(state.substeps, 1);
    assert!(sim.report.capillary.is_none());

    FrameDifference { cells: dx / sim.grid.dx, speed: dv / vmax }
}

#[test]
fn gpu_adhesive_frame_matches_reference() {
    let params = Params { adhesion: 0.3, viscosity: 2.0, density_projection: false, ..Params::default() };
    let difference = frame_difference(params, 300);

    assert!(difference.cells < 1e-3);
    assert!(difference.speed < 1e-3);
}

#[test]
fn gpu_honey_frame_matches_reference() {
    let params = Params {
        density: 1420.0,
        viscosity: 1500.0,
        surface_tension: 4.0,
        adhesion: 1.2,
        max_iterations: 2000,
        density_projection: false,
        ..Params::default()
    };
    let difference = frame_difference(params, 1000);

    assert!(difference.cells < 1e-3);
    assert!(difference.speed < 1e-3);
}

#[test]
fn water_properties_keep_the_default_config() {
    let config = SolverConfig::default();

    assert_eq!(config.clone().with_properties(&LiquidProperties::water()), config);
}

#[cfg(windows)]
#[test]
fn adhesion_pipelines_compile_under_fxc() {
    let base = SolverConfig { dims: UVec3::new(40, 24, 32), max_particles: 4096, adhesion: 1.0, ..SolverConfig::default() };
    let band = SolverConfig { narrow_band: Some(3.0), ..base.clone() };

    common::assert_compiles_under_fxc(|device, _| {
        for config in [base, band] {
            for sparse in [false, true] {
                GpuSolver::new(device, SolverConfig { sparse, ..config.clone() }, None);
            }
        }
    });
}
