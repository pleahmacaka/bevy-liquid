use bevy::math::{UVec3, Vec3};
use glam::DVec3;
use liquid_gpu::layout::{GpuParticle, GpuState};
use liquid_gpu::solver::{GpuSolver, SolverConfig};
use liquid_ref::particles::{CellIndex, Particle};
use liquid_ref::scene::dam_break;
use liquid_ref::sim::{Params, Simulation};
use liquid_ref::transfer::p2g;

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

    println!("adapter: {:?}", adapter.get_info().name);

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("liquid test"),
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

fn grid_faces(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver, buffer: &wgpu::Buffer) -> Vec<f32> {
    let stored: Vec<f32> = read(device, queue, buffer, ((buffer.size() - liquid_gpu::solver::GUARD_BYTES) / 4) as usize);
    let table = solver.slot_table();
    let words: Vec<u32> = read(device, queue, &solver.buffers.brick_list, (table.end / 4) as usize);
    let slots = &words[(table.start / 4) as usize..];
    let mut out = Vec::new();

    for axis in 0..3 {
        let mut d = solver.config.dims;
        d[axis] += 1;

        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    out.push(stored[solver.pooled_face(slots, axis, UVec3::new(x, y, z))]);
                }
            }
        }
    }

    out
}

fn to_gpu(id: usize, p: &Particle) -> GpuParticle {
    let f = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    let mut x = f(p.x);
    x[3] = id as f32;

    GpuParticle { x, v: f(p.v), c: [f(p.c[0]), f(p.c[1]), f(p.c[2])] }
}

fn config_for(sim: &Simulation) -> SolverConfig {
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
        block_smoothing: sim.params.block_smoothing,
        ..SolverConfig::default()
    }
}

fn validated<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> T {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = build();

    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation error: {error}");
    }

    out
}

fn small_dam_break(density_projection: bool) -> Simulation {
    let params = Params { density_projection, ..Params::default() };

    dam_break(glam::UVec3::new(24, 16, 8), 0.02, DVec3::new(0.16, 0.24, 0.0), params)
}

#[test]
fn shaders_and_pipelines_validate() {
    let (device, _) = gpu();
    let sim = small_dam_break(true);

    validated(&device, || GpuSolver::new(&device, config_for(&sim), None));
}

#[cfg(windows)]
#[test]
fn every_solver_pipeline_compiles_under_fxc() {
    let base = SolverConfig { dims: UVec3::new(40, 24, 32), max_particles: 4096, whitewater_capacity: 1024, ..SolverConfig::default() };
    let band = SolverConfig { narrow_band: Some(3.0), ..base.clone() };
    let diffusion = SolverConfig { viscosity: 1.0, surface_tension: 0.07, boundary_width: 2, boundary_sweeps: 2, ..base.clone() };
    let floor = SolverConfig { open_floor: Some(3.0), ..base.clone() };
    let tall = SolverConfig { tall_cells: Some(4), ..base.clone() };
    let everything = SolverConfig { narrow_band: Some(3.0), open_floor: Some(3.0), tall_cells: Some(4), ..diffusion.clone() };

    common::assert_compiles_under_fxc(|device, _| {
        for config in [base, band, diffusion, floor, tall, everything] {
            for sparse in [false, true] {
                GpuSolver::new(device, SolverConfig { sparse, ..config.clone() }, None);
            }
        }
    });
}

#[test]
fn gpu_p2g_matches_reference() {
    let (device, queue) = gpu();
    let sim = small_dam_break(false);
    let mut particles = sim.particles.clone();

    for (n, p) in particles.iter_mut().enumerate() {
        let s = n as f64 * 0.37;
        p.v = DVec3::new(s.sin(), s.cos(), (2.0 * s).sin()) * 0.5;
        p.c = [DVec3::new(0.1, -0.2, 0.3), DVec3::new(-0.3, 0.1, 0.2), DVec3::new(0.2, 0.3, -0.1)];
    }

    let mut solver = validated(&device, || GpuSolver::new(&device, config_for(&sim), None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, 1.0 / 60.0);
    solver.upload_particles(&queue, &particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        solver.dispatch_direct(&mut pass, "begin_frame", liquid_gpu::solver::Bind::Plan, 1);
        solver.encode_sort(&mut pass);
        solver.dispatch(&mut pass, "p2g", liquid_gpu::solver::Bind::Sim, liquid_gpu::solver::Slots::FACES);
    }
    queue.submit([encoder.finish()]);

    let faces = solver.params.face_base[3] as usize;
    let gpu_velocity = grid_faces(&device, &queue, &solver, &solver.buffers.velocity);
    let gpu_mass = grid_faces(&device, &queue, &solver, &solver.buffers.valid);

    let index = CellIndex::build(&sim.grid, &particles);
    let reference = p2g(&sim.grid, &particles, &index);
    let cpu_velocity: Vec<f64> = reference.velocity.axes.iter().flat_map(|a| a.data.clone()).collect();
    let cpu_mass: Vec<f64> = reference.mass.axes.iter().flat_map(|a| a.data.clone()).collect();

    let mut worst_v = 0.0f64;
    let mut worst_m = 0.0f64;

    for n in 0..faces {
        worst_m = worst_m.max((gpu_mass[n] as f64 - cpu_mass[n]).abs());

        if cpu_mass[n] > 1e-3 {
            worst_v = worst_v.max((gpu_velocity[n] as f64 - cpu_velocity[n]).abs());
        }
    }

    println!("p2g max |dv| {worst_v:.3e}, max |dm| {worst_m:.3e}");

    assert!(worst_v < 1e-4);
    assert!(worst_m < 1e-4);
}

const BLOCK_ITERATIONS: usize = 3;

fn run_substep(density_projection: bool, block_smoothing: bool) -> (f64, f64, usize) {
    let (device, queue) = gpu();
    let mut sim = small_dam_break(density_projection);

    for _ in 0..3 {
        let dt = sim.cfl_dt();
        sim.substep(dt);
    }

    let start: Vec<Particle> = sim.particles.clone();
    let dt = sim.cfl_dt();
    let mut config = config_for(&sim);

    if block_smoothing {
        sim.params.block_smoothing = true;
        sim.params.tolerance = 0.0;
        sim.params.max_iterations = BLOCK_ITERATIONS;
        sim.pressure.iter_mut().for_each(|p| *p = 0.0);
        config = SolverConfig { block_smoothing: true, tolerance: 0.0, pressure_iterations: BLOCK_ITERATIONS as u32, ..config };
    }

    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, dt as f32);
    solver.upload_particles(&queue, &start.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let state: Vec<GpuState> = read(&device, &queue, &solver.buffers.state, 1);
    let gpu_particles: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, start.len());

    sim.substep(dt);

    let cpu: Vec<(DVec3, DVec3)> = sim.particles.iter().map(|p| (p.x, p.v)).collect();
    let mut gpu = vec![(DVec3::ZERO, DVec3::ZERO); cpu.len()];

    for p in &gpu_particles {
        gpu[p.x[3] as usize] = (
            DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64),
            DVec3::new(p.v[0] as f64, p.v[1] as f64, p.v[2] as f64),
        );
    }

    let vmax = cpu.iter().fold(0.0f64, |m, p| m.max(p.1.length()));
    let mut dx = 0.0f64;
    let mut dv = 0.0f64;

    for (c, g) in cpu.iter().zip(&gpu) {
        dx = dx.max((c.0 - g.0).length());
        dv = dv.max((c.1 - g.1).length());
    }

    println!(
        "substeps {} dt {:.4e} (cpu {dt:.4e}); max |dx| {dx:.3e} m, max |dv| {dv:.3e} m/s of vmax {vmax:.3}",
        state[0].substeps, state[0].dt
    );

    (dx / sim.grid.dx, dv / vmax, state[0].substeps as usize)
}

#[test]
fn gpu_substep_matches_reference_without_density_projection() {
    let (dx_cells, dv_rel, substeps) = run_substep(false, false);

    assert_eq!(substeps, 1);
    assert!(dx_cells < 1e-3, "position error {dx_cells} cells");
    assert!(dv_rel < 1e-3, "velocity error {dv_rel}");
}

#[test]
fn gpu_substep_matches_reference_with_density_projection() {
    let (dx_cells, dv_rel, substeps) = run_substep(true, false);

    assert_eq!(substeps, 1);
    assert!(dx_cells < 1e-3, "position error {dx_cells} cells");
    assert!(dv_rel < 1e-3, "velocity error {dv_rel}");
}

#[test]
fn gpu_substep_matches_reference_with_block_smoothing() {
    let (dx_cells, dv_rel, substeps) = run_substep(false, true);

    assert_eq!(substeps, 1);
    assert!(dx_cells < 1e-3, "position error {dx_cells} cells");
    assert!(dv_rel < 1e-3, "velocity error {dv_rel}");
}

fn to_gpu_body(b: &liquid_ref::rigid::RigidBody) -> liquid_gpu::layout::GpuBody {
    let f = |v: DVec3, w: f64| [v.x as f32, v.y as f32, v.z as f32, w as f32];
    let inv = b.inverse_inertia_world();
    let (kind, shape) = match b.shape {
        liquid_ref::rigid::Shape::Sphere { radius } => (0.0, DVec3::splat(radius)),
        liquid_ref::rigid::Shape::Cuboid { half } => (1.0, half),
    };
    let motion = match b.motion {
        liquid_ref::rigid::Motion::Static => 0.0,
        liquid_ref::rigid::Motion::Kinematic => 1.0,
        liquid_ref::rigid::Motion::Dynamic => 2.0,
    };

    liquid_gpu::layout::GpuBody {
        position: f(b.position, 1.0 / b.mass()),
        rotation: [b.rotation.x as f32, b.rotation.y as f32, b.rotation.z as f32, b.rotation.w as f32],
        velocity: f(b.velocity, motion),
        angular: f(b.angular_velocity, 0.0),
        shape: f(shape, kind),
        inv_inertia: [f(inv.row(0), 0.0), f(inv.row(1), 0.0), f(inv.row(2), 0.0)],
    }
}

#[test]
fn gpu_rigid_coupling_matches_reference() {
    use liquid_ref::rigid::{Motion, RigidBody, Shape};
    use liquid_ref::scene::{tank, with_body};

    let (device, queue) = gpu();
    let mut body = RigidBody::new(Shape::Sphere { radius: 0.1 }, DVec3::new(0.25, 0.22, 0.25), Motion::Dynamic, 300.0);
    body.lock_rotation = true;

    let params = Params { density_projection: false, ..Params::default() };
    let mut sim = with_body(tank(glam::UVec3::new(20, 20, 20), 0.025, 0.35, params), body);
    let dt = 0.004;

    let mut solver = validated(&device, || GpuSolver::new(&device, config_for(&sim), None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, dt as f32);
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    solver.upload_bodies(&queue, &[to_gpu_body(&sim.bodies[0])]);

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let bodies: Vec<liquid_gpu::layout::GpuBody> = read(&device, &queue, &solver.buffers.bodies, 1);
    let scalars: Vec<f32> = read(&device, &queue, &solver.buffers.scalars, liquid_gpu::layout::SCALAR_COUNT);

    sim.substep(dt);

    let cpu_v = sim.bodies[0].velocity;
    let gpu_v = DVec3::new(bodies[0].velocity[0] as f64, bodies[0].velocity[1] as f64, bodies[0].velocity[2] as f64);
    let impulse_y = scalars[liquid_gpu::layout::S_IMPULSE + 1] as f64;

    println!("body velocity cpu {cpu_v:?} gpu {gpu_v:?}, gpu impulse y {impulse_y:.4e}, pcg iterations {}", scalars[liquid_gpu::layout::S_ITERATION]);

    assert!((cpu_v - gpu_v).length() < 1e-4 * cpu_v.length().max(1e-3));
    assert!(impulse_y > 0.0);
}

struct CoupledRun {
    mean_u: f32,
    interior: [f32; 4],
    edge: [f32; 4],
}

fn coupled_tank(narrow_band: Option<f32>, fill: f64, flow: f32, frames: usize) -> CoupledRun {
    use liquid_gpu::solver::HeightfieldLink;
    use liquid_ref::scene::tank;
    use wgpu::util::DeviceExt;

    let (device, queue) = gpu();
    let depth = 0.3;
    let params = Params { density_projection: false, ..Params::default() };
    let sim = tank(glam::UVec3::new(16, 12, 16), 0.05, fill, params);

    let cells = [24u32, 24];
    let count = (cells[0] * cells[1]) as usize;
    let state: Vec<[f32; 4]> = vec![[depth as f32, depth as f32 * flow, 0.0, 0.0]; count];
    let bed: Vec<[f32; 4]> = vec![[0.0; 4]; count];
    let make = |label: &str, data: &[[f32; 4]]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    };
    let state_buffer = make("hf state", &state);
    let bed_buffer = make("hf bed", &bed);
    let coupled = make("hf coupled", &bed);
    let config = SolverConfig { narrow_band, max_particles: 4 * sim.particles.len() as u32, ..config_for(&sim) };

    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    validated(&device, || {
        solver.attach_heightfield(
            &device,
            HeightfieldLink {
                offset: Vec3::new(0.2, 0.0, 0.2),
                cells,
                cell_size: [0.05, 0.05],
                relax_cells: 4.0,
                blend: 1.0,
                state: &state_buffer,
                bed: &bed_buffer,
                coupled: &coupled,
            },
        )
    });
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, 1.0 / 60.0);
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    for _ in 0..frames {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);
    }

    let live: usize = read::<GpuState>(&device, &queue, &solver.buffers.state, 1)[0].particle_count as usize;
    let particles: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, live);
    let band = 4.0 * 0.05;
    let near: Vec<f32> = particles.iter().filter(|p| p.v[3] < 0.5 && p.x[0] < band && p.x[1] < 0.25).map(|p| p.v[0]).collect();
    let columns: Vec<[f32; 4]> = read(&device, &queue, &coupled, count);

    CoupledRun {
        mean_u: near.iter().sum::<f32>() / near.len().max(1) as f32,
        interior: columns[(12 * cells[0] + 12) as usize],
        edge: columns[0],
    }
}

#[test]
fn heightfield_coupling_exchanges_flow_and_depth() {
    let run = coupled_tank(None, 0.3, 0.2, 6);

    println!(
        "mean inflow-band u {:.3} (target 0.2), coupled column h {:.4} (3D depth 0.3), hu {:.4}",
        run.mean_u, run.interior[0], run.interior[1]
    );

    assert!((run.mean_u - 0.2).abs() < 0.3 * 0.2);
    assert!((run.interior[0] as f64 - 0.3).abs() < 0.5 * 0.05);
    assert!(run.edge[0] == 0.0);
}

#[test]
fn narrow_band_coupled_depth_matches_dense() {
    let band = coupled_tank(Some(3.0), 0.2, 0.2, 120);
    let dense = coupled_tank(None, 0.2, 0.2, 120);

    println!(
        "coupled column h after 2 s: narrow band {:.4}, dense {:.4}; inflow-band u band {:.3}, dense {:.3}",
        band.interior[0], dense.interior[0], band.mean_u, dense.mean_u
    );

    assert!(band.interior.iter().all(|v| v.is_finite()));
    assert!((band.interior[0] - dense.interior[0]).abs() < 0.25 * 0.05);
}

#[test]
fn kernels_never_write_past_buffer_ends() {
    use liquid_ref::rigid::{Motion, RigidBody, Shape};
    use liquid_ref::scene::with_body;

    let (device, queue) = gpu();
    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.06, 0.04, 0.05) }, DVec3::new(0.3, 0.08, 0.08), Motion::Dynamic, 400.0);
    let sim = with_body(small_dam_break(true), body);
    let config = SolverConfig { max_substeps: 2, ..config_for(&sim) };

    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, 1.0 / 60.0);
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    solver.upload_bodies(&queue, &[to_gpu_body(&sim.bodies[0])]);

    for _ in 0..20 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);
    }

    let violations = solver.guard_violations(&device, &queue);

    assert!(violations.is_empty(), "{violations:?}");

    let phi = &solver.buffers.phi;
    queue.write_buffer(phi, phi.size() - liquid_gpu::solver::GUARD_BYTES, bytemuck::bytes_of(&1.0f32));

    assert_eq!(solver.guard_violations(&device, &queue).len(), 1);
}

#[test]
fn extension_kernels_never_write_past_buffer_ends() {
    use liquid_ref::rigid::{Motion, RigidBody, Shape};
    use liquid_ref::scene::with_body;

    let (device, queue) = gpu();
    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.06, 0.04, 0.05) }, DVec3::new(0.3, 0.08, 0.08), Motion::Dynamic, 400.0);
    let sim = with_body(small_dam_break(false), body);

    for sparse in [true, false] {
        let config = SolverConfig {
            max_substeps: 2,
            narrow_band: Some(3.0),
            viscosity: 1.0,
            surface_tension: 0.07,
            viscosity_iterations: 20,
            whitewater_capacity: 1 << 14,
            sparse,
            ..config_for(&sim)
        };

        let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
        solver.initialize(&queue);
        solver.set_frame_dt(&queue, 1.0 / 60.0);
        solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
        solver.upload_bodies(&queue, &[to_gpu_body(&sim.bodies[0])]);

        for _ in 0..20 {
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode_frame(&mut encoder, None);
            queue.submit([encoder.finish()]);
        }

        let state: Vec<GpuState> = read(&device, &queue, &solver.buffers.state, 1);
        let violations = solver.guard_violations(&device, &queue);

        println!("sparse {sparse}: {} particles after 20 frames with every extension on", state[0].particle_count);

        assert!(violations.is_empty(), "{violations:?}");
        assert!(state[0].particle_count > 0);
    }
}

fn grid_cells(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver, buffer: &wgpu::Buffer) -> Vec<f32> {
    let stored: Vec<f32> = read(device, queue, buffer, ((buffer.size() - liquid_gpu::solver::GUARD_BYTES) / 4) as usize);
    let table = solver.slot_table();
    let words: Vec<u32> = read(device, queue, &solver.buffers.brick_list, (table.end / 4) as usize);
    let slots = &words[(table.start / 4) as usize..];
    let d = solver.config.dims;
    let mut out = Vec::with_capacity((d.x * d.y * d.z) as usize);

    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                out.push(stored[solver.pooled_cell(slots, UVec3::new(x, y, z))]);
            }
        }
    }

    out
}

#[derive(Clone, Copy, Debug)]
struct Gauges {
    time: f64,
    p1: f64,
    h2: f64,
    h4: f64,
}

fn kleefsman_experiment() -> Vec<Gauges> {
    include_str!("kleefsman.csv")
        .lines()
        .skip(1)
        .map(|line| {
            let v: Vec<f64> = line.split(',').map(|s| s.parse().expect("number in kleefsman.csv")).collect();

            Gauges { time: v[0], p1: v[1], h2: v[2], h4: v[3] }
        })
        .collect()
}

struct Kleefsman {
    gauges: Vec<Gauges>,
    peak_speed: f32,
}

fn kleefsman_simulation(dx: f64, duration: f64, frame: f64) -> Kleefsman {
    use liquid_ref::particles::seed_box;
    use liquid_ref::rigid::{Motion, RigidBody, Shape};
    use liquid_ref::scene::SEED_JITTER;

    let (device, queue) = gpu();
    let tank = DVec3::new(3.22, 1.0, 1.0);
    let dims = (tank / dx).round().as_uvec3();
    let grid = liquid_ref::grid::Grid::new(dims, dx);
    let particles = seed_box(&grid, DVec3::new(tank.x - 1.228, 0.0, 0.0), DVec3::new(tank.x, 0.55, tank.z), SEED_JITTER);
    let obstacle = RigidBody::new(
        Shape::Cuboid { half: DVec3::new(0.0805, 0.0805, 0.2015) },
        DVec3::new(0.6635 + 0.0805, 0.0805, 0.295 + 0.2015),
        Motion::Static,
        1000.0,
    );
    let config = SolverConfig {
        dims: UVec3::new(dims.x, dims.y, dims.z),
        dx: dx as f32,
        max_particles: particles.len() as u32 + 1024,
        max_substeps: 24,
        pressure_iterations: 60,
        ..SolverConfig::default()
    };

    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, frame as f32);
    solver.upload_particles(&queue, &particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    solver.upload_bodies(&queue, &[to_gpu_body(&obstacle)]);

    let index = |i: u32, j: u32, k: u32| (i + dims.x * (j + dims.y * k)) as usize;
    let split = |x: f64| {
        let s = x / dx - 0.5;

        (s.floor() as u32, s - s.floor())
    };
    let column = |phi: &[f32], i: u32, k: u32| {
        let at = |j: u32| phi[index(i, j, k)] as f64;

        if at(0) >= 0.0 {
            return 0.0;
        }

        (0..dims.y - 1).find(|&j| at(j + 1) >= 0.0).map_or(tank.y, |j| (j as f64 + 0.5 + at(j) / (at(j) - at(j + 1))) * dx)
    };
    let gauge = |phi: &[f32], x: f64, z: f64| {
        let ((i, a), (k, b)) = (split(x), split(z));

        (1.0 - a) * (1.0 - b) * column(phi, i, k) + a * (1.0 - b) * column(phi, i + 1, k) + (1.0 - a) * b * column(phi, i, k + 1) + a * b * column(phi, i + 1, k + 1)
    };
    let sensor = |pressure: &[f32], x: f64, y: f64, z: f64| {
        let (j, t) = split(y);
        let (i, k) = ((x / dx).floor() as u32, (z / dx).floor() as u32);

        (1.0 - t) * pressure[index(i, j, k)] as f64 + t * pressure[index(i, j + 1, k)] as f64
    };

    let mut gauges = Vec::new();
    let mut peak_speed = 0.0f32;

    for n in 1..=(duration / frame).round() as usize {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        let state: GpuState = read(&device, &queue, &solver.buffers.state, 1)[0];
        let speed = f32::from_bits(state.vmax_bits);
        assert!(state.remaining < 1e-3 * frame as f32, "frame {n} left {} s unsimulated at particle speed {speed} m/s", state.remaining);
        peak_speed = peak_speed.max(speed);

        let pressure = grid_cells(&device, &queue, &solver, &solver.levels[0].x);
        let phi = grid_cells(&device, &queue, &solver, &solver.buffers.phi);

        gauges.push(Gauges {
            time: n as f64 * frame,
            p1: sensor(&pressure, 0.8245, 0.021, 0.471),
            h2: gauge(&phi, 0.992, 0.5),
            h4: gauge(&phi, 2.638, 0.5),
        });
    }

    Kleefsman { gauges, peak_speed }
}

fn first_above(series: &[Gauges], threshold: f64) -> f64 {
    series.iter().find(|g| g.p1 > threshold).map_or(f64::INFINITY, |g| g.time)
}

#[test]
fn kleefsman_dam_break_matches_experiment() {
    let experiment = kleefsman_experiment();
    let frame = 0.01;
    let dx = 0.02;
    let duration = experiment.last().map_or(0.0, |g| g.time);
    let start = std::time::Instant::now();
    let run = kleefsman_simulation(dx, duration, frame);
    let elapsed = start.elapsed().as_secs_f64();
    let simulated = &run.gauges;

    for s in simulated.iter().step_by(10) {
        let e = experiment[(s.time / frame).round() as usize];
        println!("t {:.2} s: P1 {:6.0} Pa (experiment {:6.0}), H2 {:.3} m ({:.3}), H4 {:.3} m ({:.3})", s.time, s.p1, e.p1, s.h2, e.h2, s.h4, e.h4);
    }

    let matched: Vec<(Gauges, Gauges)> = simulated.iter().map(|s| (*s, experiment[(s.time / frame).round() as usize])).collect();
    let rms = |f: fn(&Gauges) -> f64| (matched.iter().map(|(s, e)| (f(s) - f(e)).powi(2)).sum::<f64>() / matched.len() as f64).sqrt();
    let peak = |series: &[Gauges]| series.iter().filter(|g| g.time < 1.0).fold(0.0f64, |m, g| m.max(g.p1));
    let arrival = (first_above(simulated, 500.0), first_above(&experiment, 500.0));
    let peaks = (peak(simulated), peak(&experiment));
    let (h2, h4) = (rms(|g| g.h2), rms(|g| g.h4));

    println!("{duration} s simulated at dx {dx} m in {elapsed:.1} s, peak particle speed {:.2} m/s", run.peak_speed);
    println!("P1 impact at {:.2} s (experiment {:.2} s), peak {:.0} Pa (experiment {:.0} Pa, ratio {:.2})", arrival.0, arrival.1, peaks.0, peaks.1, peaks.0 / peaks.1);
    println!("rms height error over 0..{duration} s: H2 {h2:.4} m, H4 {h4:.4} m");

    assert!((arrival.0 - arrival.1).abs() < 3.5 * frame, "P1 impact at {:.2} s, experiment {:.2} s", arrival.0, arrival.1);
    assert!((0.5..=2.0).contains(&(peaks.0 / peaks.1)), "P1 peak {:.0} Pa, experiment {:.0} Pa", peaks.0, peaks.1);
    assert!(h2 < 2.5 * dx, "H2 rms error {h2:.4} m");
    assert!(h4 < dx, "H4 rms error {h4:.4} m");
}

fn airy_omega(k: f64, depth: f64) -> f64 {
    (9.81 * k * (k * depth).tanh()).sqrt()
}

fn crossing_omega(samples: &[(f64, f64)]) -> f64 {
    let crossings: Vec<f64> = samples
        .windows(2)
        .filter(|w| w[0].1.signum() != w[1].1.signum())
        .map(|w| w[0].0 + (w[1].0 - w[0].0) * w[0].1 / (w[0].1 - w[1].1))
        .collect();
    let spans = crossings.len() - 1;

    std::f64::consts::PI * spans as f64 / (crossings[spans] - crossings[0])
}

struct Dispersion {
    kh: f64,
    measured: f64,
    exact: f64,
}

impl Dispersion {
    fn error(&self) -> f64 {
        self.measured / self.exact - 1.0
    }
}

fn sloshing_3d(device: &wgpu::Device, queue: &wgpu::Queue, kh: f64) -> Dispersion {
    use liquid_ref::particles::seed_box;
    use liquid_ref::scene::{SEED_JITTER, moving};

    let depth = 0.1;
    let depth_cells = 16;
    let dx = depth / depth_cells as f64;
    let length_cells = (std::f64::consts::PI * depth_cells as f64 / kh).round() as u32;
    let k = std::f64::consts::PI / (length_cells as f64 * dx);
    let exact = airy_omega(k, depth);
    let speed = 0.05 * depth * exact / (k * depth).sinh();
    let dims = glam::UVec3::new(length_cells, depth_cells + 4, 4);
    let grid = liquid_ref::grid::Grid::new(dims, dx);
    let mut water = grid.size();
    water.y = depth;

    let particles = moving(seed_box(&grid, DVec3::ZERO, water, SEED_JITTER), |x| {
        DVec3::new(-speed * (k * x.y).cosh() * (k * x.x).sin(), speed * (k * x.y).sinh() * (k * x.x).cos(), 0.0)
    });
    let frame = 2.0 * std::f64::consts::PI / exact / 40.0;
    let config = SolverConfig {
        dims: UVec3::new(dims.x, dims.y, dims.z),
        dx: dx as f32,
        max_particles: 4 * particles.len() as u32,
        max_substeps: 8,
        pressure_iterations: 60,
        tolerance: 1e-6,
        coarse_sweeps: 16,
        density_projection: false,
        narrow_band: Some(3.0),
        ..SolverConfig::default()
    };

    let mut solver = validated(device, || GpuSolver::new(device, config, None));
    solver.initialize(queue);
    solver.set_frame_dt(queue, frame as f32);
    let uploaded: Vec<GpuParticle> = particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect();
    solver.upload_particles(queue, &uploaded);

    let mode = |particles: &[GpuParticle]| particles.iter().filter(|p| p.v[3] < 0.5).map(|p| p.v[0] as f64 * (k * p.x[0] as f64).sin()).sum::<f64>();
    let mut samples = vec![(0.0, mode(&uploaded))];

    for n in 1..=128 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        let state: GpuState = read(device, queue, &solver.buffers.state, 1)[0];
        assert!(state.remaining < 1e-3 * frame as f32, "frame {n} left {} s unsimulated", state.remaining);

        let live: Vec<GpuParticle> = read(device, queue, &solver.buffers.particles, state.particle_count as usize);
        samples.push((n as f64 * frame, mode(&live)));
    }

    Dispersion { kh: k * depth, measured: crossing_omega(&samples), exact }
}

fn heightfield_waves(device: &wgpu::Device, queue: &wgpu::Queue, kh: f64) -> Dispersion {
    use liquid_ocean::dispersive::GpuDispersive;
    use liquid_wave::dispersive::{Dispersive, DispersiveParams};
    use liquid_wave::grid::{Bathymetry, Grid, State};
    use liquid_wave::swe::Swe;

    let depth = 4.0;
    let k = kh / depth;
    let cells = 64;
    let dx = 2.0 * std::f64::consts::PI / k / cells as f64;
    let grid = Grid::new(cells, 4, dx, dx);
    let mut state = State::zeros(grid.cells());

    for z in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, z);
            state.h[grid.cell(i, z)] = depth + 1e-3 * depth * (k * x).cos();
        }
    }

    let model = Dispersive::new(Swe::new(grid, Bathymetry::flat(&grid, 0.0), state), DispersiveParams::default());
    let exact = airy_omega(k, depth);
    let period = 2.0 * std::f64::consts::PI / exact;
    let dt = (period / 200.0).min(0.2 * dx / (9.81 * depth).sqrt());
    let mut solver = GpuDispersive::new(device, &model, dt as f32, liquid_ocean::swe::shared_buffers(device, &model.swe));
    solver.write_params(queue, dt as f32);

    let mode = |h: &[[f32; 4]]| (0..grid.nx).map(|i| (h[grid.cell(i, 0)][0] as f64 - depth) * (k * grid.center(i, 0)[0]).cos()).sum::<f64>();
    let mut samples = vec![(0.0, mode(&read(device, queue, &solver.total, grid.cells())))];

    while samples.len() as f64 * dt < 3.2 * period {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, 1);
        queue.submit([encoder.finish()]);

        samples.push((samples.len() as f64 * dt, mode(&read(device, queue, &solver.total, grid.cells()))));
    }

    let clock: Vec<f32> = read(device, queue, &solver.bulk.buffers.clock, 4);
    let elapsed = (samples.len() - 1) as f64 * dt;
    assert!((clock[2] as f64 - elapsed).abs() < 1e-4 * elapsed, "heightfield clock {} s, expected {elapsed} s", clock[2]);

    Dispersion { kh, measured: crossing_omega(&samples), exact }
}

fn ocean_waves(device: &wgpu::Device, queue: &wgpu::Queue, kh: f64) -> Dispersion {
    use liquid_ocean::ocean::{GpuOcean, surface_bytes};
    use liquid_wave::fft::Complex64;
    use liquid_wave::ocean::{Ocean, OceanParams};
    use liquid_wave::spectrum::SeaState;

    let patch = 100.0;
    let k = 2.0 * std::f64::consts::PI / patch;
    let depth = kh / k;
    let mut ocean = Ocean::new(OceanParams {
        sea: SeaState { depth, ..SeaState::default() },
        resolution: 16,
        patch_sizes: vec![patch],
        ..OceanParams::default()
    });
    ocean.cascades[0].h0.fill(Complex64::default());
    ocean.cascades[0].h0[1] = Complex64::new(0.5, 0.0);

    let exact = airy_omega(k, depth);
    let frame = 2.0 * std::f64::consts::PI / exact / 40.0;
    let surface = liquid_ocean::kernel::zeroed_storage(device, "ocean surface", surface_bytes(&ocean));
    let mut solver = GpuOcean::new(device, &ocean, surface);
    let mut samples = Vec::new();

    for n in 0..=128 {
        let t = n as f64 * frame;
        solver.write_time(queue, t as f32, frame as f32);

        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode(&mut encoder);
        queue.submit([encoder.finish()]);

        samples.push((t, read::<[f32; 4]>(device, queue, &solver.surface, 1)[0][1] as f64));
    }

    Dispersion { kh, measured: crossing_omega(&samples), exact }
}

#[test]
fn tiers_share_airy_dispersion() {
    let (device, queue) = gpu();

    for kh in [0.3, 1.0, 3.0] {
        let solid = sloshing_3d(&device, &queue, kh);
        let field = heightfield_waves(&device, &queue, solid.kh);
        let ocean = ocean_waves(&device, &queue, solid.kh);
        let errors = [solid.error(), field.error(), ocean.error()];
        let spread = errors.iter().copied().fold(f64::NEG_INFINITY, f64::max) - errors.iter().copied().fold(f64::INFINITY, f64::min);

        println!(
            "kh {:.4}: omega / Airy - 1 = 3D {:+.3}%, heightfield {:+.3}%, FFT ocean {:+.6}%, spread {:.3}%",
            solid.kh,
            100.0 * solid.error(),
            100.0 * field.error(),
            100.0 * ocean.error(),
            100.0 * spread
        );

        assert!(solid.error().abs() < 0.025, "3D tier at kh {:.3}: {:+.2}%", solid.kh, 100.0 * solid.error());
        assert!(field.error().abs() < 0.01, "heightfield tier at kh {:.3}: {:+.2}%", solid.kh, 100.0 * field.error());
        assert!(ocean.error().abs() < 1e-4, "FFT ocean tier at kh {:.3}: {:+.4}%", solid.kh, 100.0 * ocean.error());
        assert!(spread < 0.025, "tiers disagree by {:.2}% at kh {:.3}", 100.0 * spread, solid.kh);
    }
}

fn subgroup_gpu() -> (wgpu::Device, wgpu::Queue) {
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
        label: Some("liquid subgroup test"),
        required_features: adapter.features() & wgpu::Features::SUBGROUP,
        required_limits: limits::portable(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device")
}

fn has_subgroups(device: &wgpu::Device) -> bool {
    let on = device.features().contains(wgpu::Features::SUBGROUP);

    if !on {
        println!("adapter has no wgpu::Features::SUBGROUP: only the shared-memory reduction path runs here");
    }

    on
}

struct Snapshot {
    pressure: Vec<f32>,
    velocity: Vec<f32>,
    body: DVec3,
    iterations: f32,
    subgroups: bool,
}

fn snapshot(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> Snapshot {
    let bodies: Vec<liquid_gpu::layout::GpuBody> = read(device, queue, &solver.buffers.bodies, 1);
    let scalars: Vec<f32> = read(device, queue, &solver.buffers.scalars, liquid_gpu::layout::SCALAR_COUNT);
    let v = bodies[0].velocity;

    Snapshot {
        pressure: grid_cells(device, queue, solver, &solver.levels[0].x),
        velocity: grid_faces(device, queue, solver, &solver.buffers.velocity),
        body: DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64),
        iterations: scalars[liquid_gpu::layout::S_ITERATION],
        subgroups: solver.subgroups,
    }
}

fn run_frames(device: &wgpu::Device, queue: &wgpu::Queue, sim: &Simulation, config: SolverConfig, frames: usize) -> GpuSolver {
    let mut solver = validated(device, || GpuSolver::new(device, config, None));
    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    if !sim.bodies.is_empty() {
        solver.upload_bodies(queue, &sim.bodies.iter().map(to_gpu_body).collect::<Vec<_>>());
    }

    for _ in 0..frames {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);
    }

    solver
}

fn relative_gap(a: &[f32], b: &[f32]) -> f64 {
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
    let gap = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs() as f64));

    gap / scale.max(1e-30)
}

fn body_dam_break() -> Simulation {
    use liquid_ref::rigid::{Motion, RigidBody, Shape};

    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.04, 0.03, 0.04) }, DVec3::new(0.08, 0.12, 0.08), Motion::Dynamic, 600.0);

    liquid_ref::scene::with_body(small_dam_break(false), body)
}

#[test]
fn subgroup_reductions_match_shared_memory() {
    let (device, queue) = subgroup_gpu();
    let available = has_subgroups(&device);
    let sim = body_dam_break();
    let config = SolverConfig { max_substeps: 2, ..config_for(&sim) };
    let [shared, fast] = [false, true].map(|subgroups| {
        let solver = run_frames(&device, &queue, &sim, SolverConfig { subgroups, ..config.clone() }, 3);
        snapshot(&device, &queue, &solver)
    });

    let pressure = relative_gap(&shared.pressure, &fast.pressure);
    let velocity = relative_gap(&shared.velocity, &fast.velocity);
    let body = (shared.body - fast.body).length() / shared.body.length().max(1e-6);

    println!(
        "dam break with a submerged dynamic box, 3 frames x 2 substeps: subgroup path {} vs shared memory {}; max |dp|/max|p| {pressure:.3e}, max |du|/max|u| {velocity:.3e}, body velocity {:?} vs {:?} (|dv|/|v| {body:.3e}), last pcg iteration counter {} vs {}",
        fast.subgroups, shared.subgroups, fast.body, shared.body, fast.iterations, shared.iterations
    );

    assert!(!shared.subgroups);
    assert_eq!(fast.subgroups, available);
    assert!(pressure < 1e-4, "pressure gap {pressure}");
    assert!(velocity < 1e-4, "velocity gap {velocity}");
    assert!(body < 1e-4, "body gap {body}");
}

#[test]
fn subgroup_kernels_never_write_past_buffer_ends() {
    let (device, queue) = subgroup_gpu();
    let available = has_subgroups(&device);
    let sim = body_dam_break();
    let extensions = SolverConfig {
        max_substeps: 2,
        narrow_band: Some(3.0),
        viscosity: 1.0,
        surface_tension: 0.07,
        viscosity_iterations: 20,
        ..config_for(&sim)
    };
    let floor = SolverConfig {
        max_substeps: 2,
        boundary_width: 2,
        boundary_sweeps: 2,
        open_floor: Some(3.0),
        max_particles: 4 * sim.particles.len() as u32,
        ..config_for(&sim)
    };

    for subgroups in [true, false] {
        for (name, config) in [("narrow band + viscosity + tension", &extensions), ("open floor + boundary smoothing", &floor)] {
            for sparse in [true, false] {
                let config = SolverConfig { subgroups, sparse, ..config.clone() };
                let solver = run_frames(&device, &queue, &sim, config, 20);
                let state: Vec<GpuState> = read(&device, &queue, &solver.buffers.state, 1);
                let violations = solver.guard_violations(&device, &queue);

                println!("subgroups {} sparse {sparse} {name}: {} particles after 20 frames", solver.subgroups, state[0].particle_count);

                assert_eq!(solver.subgroups, subgroups && available);
                assert!(violations.is_empty(), "{violations:?}");
                assert!(state[0].particle_count > 0);
            }
        }
    }
}

fn words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    read(device, queue, buffer, ((buffer.size() - liquid_gpu::solver::GUARD_BYTES) / 4) as usize)
}

fn replay_scans(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    solver: &mut GpuSolver,
    particles: &[GpuParticle],
    bodies: &[liquid_gpu::layout::GpuBody],
) -> Vec<(&'static str, Vec<u32>)> {
    use liquid_gpu::solver::{Bind, Slots};

    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, particles);
    solver.upload_bodies(queue, bodies);

    let mut encoder = device.create_command_encoder(&Default::default());

    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        solver.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
        solver.dispatch(&mut pass, "measure_speed", Bind::Sim, Slots::PARTICLES);
        solver.dispatch_direct(&mut pass, "plan_substep", Bind::Plan, 1);
        solver.encode_sort(&mut pass);
        solver.encode_geometry(&mut pass);
        solver.encode_resample(&mut pass);
    }

    queue.submit([encoder.finish()]);

    let b = &solver.buffers;

    [
        ("state", &b.state),
        ("particles", &b.particles),
        ("sorted", &b.sorted),
        ("cell start", &b.cell_start),
        ("brick list", &b.brick_list),
        ("reseed offsets", &b.band_offsets),
        ("scalars", &b.scalars),
    ]
    .into_iter()
    .map(|(name, buffer)| (name, words(device, queue, buffer)))
    .collect()
}

#[test]
fn subgroup_scans_are_bit_identical_to_shared_memory() {
    use liquid_ref::rigid::{Motion, RigidBody, Shape};

    let (device, queue) = subgroup_gpu();
    let available = has_subgroups(&device);
    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.06, 0.05, 0.06) }, DVec3::new(0.7, 0.1, 0.35), Motion::Dynamic, 600.0);
    let params = Params { density_projection: false, ..Params::default() };
    let wide = liquid_ref::scene::with_body(dam_break(glam::UVec3::new(96, 64, 48), 1.0 / 64.0, DVec3::new(0.375, 0.5, 0.0), params), body);

    for sim in [body_dam_break(), wide] {
        let config = SolverConfig { max_substeps: 2, narrow_band: Some(3.0), ..config_for(&sim) };
        let seeded = replay_checkpoints(&device, &queue, &sim, &config, available);

        assert!(seeded > 0.0, "the narrow band never reseeded, so the reseed scan went untested");
    }
}

fn replay_checkpoints(device: &wgpu::Device, queue: &wgpu::Queue, sim: &Simulation, config: &SolverConfig, available: bool) -> f32 {
    let source = run_frames(device, queue, sim, SolverConfig { subgroups: true, ..config.clone() }, 0);
    let mut replays = [false, true].map(|subgroups| validated(device, || GpuSolver::new(device, SolverConfig { subgroups, ..config.clone() }, None)));
    let mut seeded = 0.0;

    assert!(!replays[0].subgroups);
    assert_eq!(replays[1].subgroups, available);

    for checkpoint in 0..4 {
        for _ in 0..3 {
            let mut encoder = device.create_command_encoder(&Default::default());
            source.encode_frame(&mut encoder, None);
            queue.submit([encoder.finish()]);
        }

        let state: GpuState = read(device, queue, &source.buffers.state, 1)[0];
        let particles: Vec<GpuParticle> = read(device, queue, &source.buffers.particles, state.particle_count as usize);
        let bodies: Vec<liquid_gpu::layout::GpuBody> = read(device, queue, &source.buffers.bodies, 1);
        let [shared, fast] = replays.each_mut().map(|solver| replay_scans(device, queue, solver, &particles, &bodies));
        let reseeded = f32::from_bits(shared[6].1[liquid_gpu::layout::S_BAND_SEEDED]);
        seeded += reseeded;

        println!(
            "{} grid after {} frames: {} particles, {reseeded} reseeded, live bricks {} of {}",
            sim.grid.dims,
            3 * (checkpoint + 1),
            state.particle_count,
            shared[4].1[0],
            source.brick_count()
        );

        for ((name, a), (_, b)) in shared.iter().zip(&fast) {
            let differ = a.iter().zip(b).filter(|(x, y)| x != y).count();
            assert_eq!(differ, 0, "{name}: {differ} of {} words differ between shared-memory and subgroup scans", a.len());
        }
    }

    seeded
}

#[test]
fn subgroup_heightfield_seed_matches_shared_memory() {
    use liquid_gpu::solver::HeightfieldLink;
    use wgpu::util::DeviceExt;

    let (device, queue) = subgroup_gpu();
    let available = has_subgroups(&device);
    let cells = [32u32, 32];
    let state: Vec<[f32; 4]> = (0..cells[0] * cells[1]).map(|i| [0.12 + 0.01 * ((i * 7) % 23) as f32, 0.0, 0.0, 0.0]).collect();
    let bed = vec![[0.0f32; 4]; state.len()];
    let make = |label: &str, data: &[[f32; 4]]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        })
    };
    let (state, bed, coupled) = (make("hf state", &state), make("hf bed", &bed), make("hf coupled", &bed));
    let config = SolverConfig { dims: UVec3::new(24, 12, 24), dx: 0.05, max_particles: 1 << 18, ..SolverConfig::default() };

    let runs = [false, true].map(|subgroups| {
        let mut solver = validated(&device, || GpuSolver::new(&device, SolverConfig { subgroups, ..config.clone() }, None));
        solver.attach_heightfield(
            &device,
            HeightfieldLink {
                offset: Vec3::new(0.2, 0.0, 0.2),
                cells,
                cell_size: [0.05, 0.05],
                relax_cells: 4.0,
                blend: 1.0,
                state: &state,
                bed: &bed,
                coupled: &coupled,
            },
        );
        solver.initialize(&queue);

        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_seed(&mut encoder);
        queue.submit([encoder.finish()]);

        let count = read::<GpuState>(&device, &queue, &solver.buffers.state, 1)[0].particle_count;

        (solver.subgroups, count, read::<u32>(&device, &queue, &solver.buffers.particles, 20 * count as usize))
    });

    println!("heightfield seed over 576 columns: {} particles with shared memory, {} with subgroups", runs[0].1, runs[1].1);

    assert!(!runs[0].0);
    assert_eq!(runs[1].0, available);
    assert!(runs[0].1 > 10_000);
    assert_eq!(runs[0].1, runs[1].1);
    assert!(runs[0].2 == runs[1].2, "seeded particles differ between shared-memory and subgroup scans");
}

fn iteration_counts(device: &wgpu::Device, queue: &wgpu::Queue, sim: &Simulation) -> [f32; 2] {
    [(0, 0), (2, 2)].map(|(boundary_width, boundary_sweeps)| {
        let config = SolverConfig { boundary_width, boundary_sweeps, tolerance: 1e-5, pressure_iterations: 200, ..config_for(sim) };
        let solver = run_frames(device, queue, sim, config, 1);

        snapshot(device, queue, &solver).iterations
    })
}

#[test]
fn boundary_smoothing_cuts_gpu_iterations_and_matches_reference() {
    use liquid_ref::rigid::{Motion, RigidBody, Shape};
    use liquid_ref::scene::{tank, with_body};

    let (device, queue) = gpu();
    let params = Params { density_projection: false, ..Params::default() };
    let mut dam = dam_break(glam::UVec3::new(48, 32, 16), 0.01, DVec3::new(0.16, 0.24, 0.0), params);

    for _ in 0..20 {
        let dt = dam.cfl_dt();
        dam.substep(dt);
    }

    let sphere = RigidBody::new(Shape::Sphere { radius: 0.1 }, DVec3::new(0.25, 0.15, 0.25), Motion::Static, 1000.0);
    let tank = with_body(tank(glam::UVec3::new(20, 24, 20), 0.025, 0.3, params), sphere);

    for (name, sim) in [("dam break", &dam), ("sphere in tank", &tank)] {
        let [plain, banded] = iteration_counts(&device, &queue, sim);

        println!("{name}: gpu pcg iterations to 1e-5 without boundary smoothing {plain}, with width 2 / 2 sweeps {banded}");

        assert!(banded <= plain, "{name}: {banded} > {plain}");
    }

    let mut sim = small_dam_break(false);
    sim.params.boundary_width = 2;
    sim.params.boundary_sweeps = 2;

    for _ in 0..3 {
        let dt = sim.cfl_dt();
        sim.substep(dt);
    }

    let start = sim.particles.clone();
    let dt = sim.cfl_dt();
    let config = SolverConfig { boundary_width: 2, boundary_sweeps: 2, max_substeps: 1, ..config_for(&sim) };
    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, dt as f32);
    solver.upload_particles(&queue, &start.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let gpu_particles: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, start.len());
    sim.substep(dt);

    let vmax = sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length()));
    let (mut dx, mut dv) = (0.0f64, 0.0f64);

    for p in &gpu_particles {
        let c = &sim.particles[p.x[3] as usize];
        dx = dx.max((c.x - DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64)).length());
        dv = dv.max((c.v - DVec3::new(p.v[0] as f64, p.v[1] as f64, p.v[2] as f64)).length());
    }

    println!("boundary smoothing on both sides, one frame: max |dx| {:.3e} cells, max |dv| {:.3e} of vmax {vmax:.3}", dx / sim.grid.dx, dv / vmax);

    assert!(dx / sim.grid.dx < 1e-3);
    assert!(dv / vmax < 1e-3);
}

fn deep_wave_sim(open_floor: Option<f64>) -> Simulation {
    let dx = 1.0 / 32.0;
    let depth = 10.0 * dx;
    let full_depth = 1.0;
    let grid = liquid_ref::grid::Grid::new(glam::UVec3::new(32, 16, 4), dx);
    let k = std::f64::consts::PI / grid.size().x;
    let omega = (9.81 * k * (k * full_depth).tanh()).sqrt();
    let a = 0.5 * dx * 9.81 * k / (omega * (k * full_depth).cosh());
    let mut water = grid.size();
    water.y = depth;

    let particles = liquid_ref::scene::moving(liquid_ref::particles::seed_box(&grid, DVec3::ZERO, water, liquid_ref::scene::SEED_JITTER), |x| {
        let y = x.y - depth + full_depth;

        DVec3::new(-a * (k * y).cosh() * (k * x.x).sin(), a * (k * y).sinh() * (k * x.x).cos(), 0.0)
    });
    let params = Params { open_floor, density_projection: false, ..Params::default() };

    Simulation::new(grid, params, particles, Vec::new())
}

#[test]
fn open_floor_matches_reference() {
    let (device, queue) = gpu();
    let mut sim = deep_wave_sim(Some(3.0));
    let dt = 1.0 / 120.0;
    let config = SolverConfig { open_floor: Some(3.0), max_substeps: 1, max_particles: 2 * sim.particles.len() as u32, ..config_for(&sim) };
    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, dt as f32);
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    for frame in 1..=4 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);
        sim.substep(dt);

        let state: Vec<GpuState> = read(&device, &queue, &solver.buffers.state, 1);
        let gpu = snapshot(&device, &queue, &solver);
        let cpu_velocity: Vec<f32> = sim.velocity.axes.iter().flat_map(|a| a.data.iter().map(|v| *v as f32)).collect();
        let cpu_pressure: Vec<f32> = sim.pressure.iter().map(|p| *p as f32).collect();
        let velocity = relative_gap(&cpu_velocity, &gpu.velocity);
        let pressure = relative_gap(&cpu_pressure, &gpu.pressure);
        let d = sim.grid.dims;
        let floor: Vec<f32> = (0..d.x * d.z).map(|c| gpu.velocity[((d.x + 1) * d.y * d.z + c % d.x + d.x * (d.y + 1) * (c / d.x)) as usize]).collect();
        let inflow = floor.iter().filter(|v| **v > 0.0).count();

        println!(
            "frame {frame}: particles cpu {} gpu {}, max |du|/max|u| {velocity:.3e}, max |dp|/max|p| {pressure:.3e}, inflow columns {inflow}/{}",
            sim.particles.len(),
            state[0].particle_count,
            floor.len()
        );

        assert_eq!(state[0].substeps, 1);
        assert!(inflow > 0 && inflow < floor.len());
        assert!((state[0].particle_count as i64 - sim.particles.len() as i64).abs() <= 2);
        assert!(velocity < 2e-3, "velocity gap {velocity}");
        assert!(pressure < 2e-3, "pressure gap {pressure}");
    }
}

fn floor_box_dam_break() -> Simulation {
    use liquid_ref::rigid::{Motion, RigidBody, Shape};

    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.0605, 0.0505, 0.0805) }, DVec3::new(0.703, 0.0505, 0.151), Motion::Static, 1000.0);
    let params = Params::default();

    liquid_ref::scene::with_body(dam_break(glam::UVec3::new(50, 20, 15), 0.02, DVec3::new(0.3, 0.3, 0.0), params), body)
}

#[test]
fn floor_box_push_out_matches_reference() {
    let (device, queue) = gpu();
    let mut sim = floor_box_dam_break();
    let margin = sim.params.solid_margin_cells * sim.grid.dx;

    for (x, z) in [(0.645, 0.1), (0.655, 0.15), (0.66, 0.2), (0.7, 0.15), (0.76, 0.08)] {
        sim.particles.push(Particle::at_rest(DVec3::new(x, margin, z)));
    }

    let dt = 0.004;
    let config = SolverConfig { max_substeps: 1, ..config_for(&sim) };
    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, dt as f32);
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    solver.upload_bodies(&queue, &[to_gpu_body(&sim.bodies[0])]);

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let count = sim.particles.len();
    let gpu_particles: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, count);
    sim.substep(dt);

    let body = sim.bodies[0];
    let vmax = sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length()));
    let mut gap = 0.0f64;
    let mut slip = 0.0f64;
    let mut clearance = f64::INFINITY;

    for p in &gpu_particles {
        let x = DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64);
        let cpu = &sim.particles[p.x[3] as usize];
        gap = gap.max((cpu.x - x).length());
        slip = slip.max((cpu.v - DVec3::new(p.v[0] as f64, p.v[1] as f64, p.v[2] as f64)).length());
        clearance = clearance.min(body.sdf(x));
    }

    println!(
        "one frame with five particles trapped in a floor box: gpu vs cpu max |dx| {:.3e} cells, max |dv| {:.3e} of vmax {vmax:.3}, gpu min box clearance {clearance:.4e} m (margin {margin:.1e} m)",
        gap / sim.grid.dx,
        slip / vmax
    );

    assert!(gap / sim.grid.dx < 1e-3);
    assert!(slip / vmax < 1e-3);
    assert!(clearance >= 0.98 * margin);

    let sim = floor_box_dam_break();
    let solver = run_frames(&device, &queue, &sim, SolverConfig { max_substeps: 8, ..config_for(&sim) }, 0);
    let bound = 4.0 * (2.0 * 9.81 * 0.3f64).sqrt();
    let mut fastest = 0.0f64;
    let mut closest = f64::INFINITY;

    for _ in 0..90 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        let live = read::<GpuState>(&device, &queue, &solver.buffers.state, 1)[0].particle_count as usize;

        for p in read::<GpuParticle>(&device, &queue, &solver.buffers.particles, live) {
            let x = DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64);
            fastest = fastest.max(DVec3::new(p.v[0] as f64, p.v[1] as f64, p.v[2] as f64).length());
            closest = closest.min(body.sdf(x));
        }
    }

    println!("gpu dam break onto a floor box for 1.5 s: max particle speed {fastest:.3} m/s (bound {bound:.3}), min box clearance {closest:.4e} m");

    assert!(fastest < bound);
    assert!(closest >= 0.98 * margin);
}

#[test]
fn grid_aligned_box_face_weights_match_reference() {
    use liquid_ref::levelset::{face_weights, solid_phi_nodes};
    use liquid_ref::rigid::{Motion, RigidBody, Shape};
    use liquid_ref::scene::{tank, with_body};

    let (device, queue) = gpu();
    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.125, 0.0625, 0.125) }, DVec3::new(0.25, 0.125, 0.25), Motion::Static, 1000.0);
    let sim = with_body(tank(glam::UVec3::splat(16), 1.0 / 32.0, 0.375, Params::default()), body);
    let config = SolverConfig { sparse: false, ..config_for(&sim) };

    let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, 1.0 / 60.0);
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    solver.upload_bodies(&queue, &[to_gpu_body(&sim.bodies[0])]);

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let reference = face_weights(&sim.grid, &solid_phi_nodes(&sim.grid, &sim.bodies));
    let expected: Vec<f64> = reference.axes.iter().flat_map(|a| a.data.iter().copied()).collect();
    let weights: Vec<f32> = read(&device, &queue, &solver.buffers.weights, expected.len());
    let worst = weights.iter().zip(&expected).fold(0.0f64, |m, (g, c)| m.max((*g as f64 - c).abs()));
    let closed = expected.iter().filter(|w| **w == 0.0).count();

    println!("grid-aligned box: {closed} closed faces, worst GPU vs CPU weight gap {worst:.2e}");

    assert!(worst < 1e-6);
}

fn sphere_drop(static_solid: bool) -> (f32, [Vec3; 2]) {
    let (device, queue) = gpu();
    let (center, radius) = (Vec3::new(0.24, 0.2, 0.24), 0.08);
    let config = SolverConfig {
        dims: UVec3::new(24, 32, 24),
        dx: 0.02,
        max_particles: 16_384,
        static_solid,
        ..SolverConfig::default()
    };
    let seed = liquid_gpu::plugin::seed_box(&config, Vec3::new(0.16, 0.4, 0.16), Vec3::new(0.32, 0.56, 0.32));
    let mut solver = validated(&device, || GpuSolver::new(&device, config.clone(), None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, 1.0 / 60.0);
    solver.upload_particles(&queue, &seed);

    if static_solid {
        let nodes = config.dims + 1;
        let distances: Vec<f32> = (0..nodes.element_product())
            .map(|i| UVec3::new(i % nodes.x, (i / nodes.x) % nodes.y, i / (nodes.x * nodes.y)))
            .map(|p| (p.as_vec3() * config.dx).distance(center) - radius)
            .collect();
        solver.upload_static_solid(&queue, &distances);
    } else {
        let mass = 1000.0 * 4.0 / 3.0 * std::f32::consts::PI * radius.powi(3);
        let sphere = liquid_gpu::layout::GpuBody {
            position: [center.x, center.y, center.z, 1.0 / mass],
            rotation: [0.0, 0.0, 0.0, 1.0],
            velocity: [0.0, 0.0, 0.0, 1.0],
            angular: [0.0; 4],
            shape: [radius, radius, radius, 0.0],
            inv_inertia: [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        };
        solver.upload_bodies(&queue, &[sphere]);
    }

    let mut deepest = f32::MAX;
    let mut means = [Vec3::ZERO; 2];

    for frame in 1..=60 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        queue.submit([encoder.finish()]);

        let count = read::<GpuState>(&device, &queue, &solver.buffers.state, 1)[0].particle_count as usize;
        let particles: Vec<GpuParticle> = read(&device, &queue, &solver.buffers.particles, count);
        let positions = || particles.iter().map(|p| Vec3::new(p.x[0], p.x[1], p.x[2]));
        deepest = positions().map(|x| x.distance(center) - radius).fold(deepest, f32::min);

        if let Some(slot) = [SPHERE_CONTACT_FRAME, 60].iter().position(|at| *at == frame) {
            means[slot] = positions().sum::<Vec3>() / count as f32;
        }
    }

    (deepest, means)
}

const SPHERE_CONTACT_FRAME: usize = 18;

#[test]
fn static_sdf_sphere_matches_sphere_body() {
    let (body_depth, body_means) = sphere_drop(false);
    let (solid_depth, solid_means) = sphere_drop(true);
    let dx = 0.02;

    println!("deepest particle over the run: body {body_depth:.4} m, static solid {solid_depth:.4} m");
    println!("mean position: body {body_means:?}, static solid {solid_means:?}");
    assert!(body_depth < 0.5 * dx, "the liquid never reached the sphere body");
    assert!(body_depth > -0.5 * dx, "liquid entered the sphere body by {body_depth} m");
    assert!(solid_depth > -0.5 * dx, "liquid entered the static solid by {solid_depth} m");

    for (body, solid) in body_means.iter().zip(solid_means) {
        assert!(body.distance(solid) < dx, "mean liquid position differs by {} m", body.distance(solid));
    }
}
