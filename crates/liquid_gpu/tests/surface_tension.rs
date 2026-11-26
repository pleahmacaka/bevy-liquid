use bevy::math::{UVec3, Vec3};
use glam::DVec3;
use liquid_gpu::layout::{GpuParticle, GpuState, S_CAPILLARY_ITERATIONS, S_VISCOUS_ITERATIONS, SCALAR_COUNT};
use liquid_gpu::solver::{Bind, GUARD_BYTES, GpuSolver, Quality, SolverConfig};
use liquid_ref::grid::{Field3, Grid};
use liquid_ref::levelset::redistance;
use liquid_ref::particles::{Particle, seed_box};
use liquid_ref::scene::SEED_JITTER;
use liquid_ref::sim::{Params, Simulation};
use liquid_ref::surface_tension::curvature;

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
        label: Some("surface tension test"),
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

fn distorted_sphere(n: u32) -> (Grid, Field3, DVec3, f64) {
    let grid = Grid::new(glam::UVec3::splat(n), 1.0 / n as f64);
    let center = DVec3::splat(0.5);
    let radius = 0.3;
    let mut phi = grid.cells();

    for i in 0..phi.data.len() {
        let r = (grid.cell_center(phi.coords(i)) - center).length();
        phi.data[i] = (r * r - radius * radius) * 7.0;
    }

    (grid, phi, center, radius)
}

#[test]
fn gpu_redistance_and_curvature_match_reference() {
    let (device, queue) = gpu();
    let (grid, phi, center, radius) = distorted_sphere(32);
    let config = SolverConfig {
        dims: UVec3::splat(32),
        dx: grid.dx as f32,
        max_particles: 64,
        surface_tension: 0.07,
        sparse: false,
        ..SolverConfig::default()
    };
    let solver = validated(&device, || GpuSolver::new(&device, config, None));
    solver.initialize(&queue);
    queue.write_buffer(&solver.buffers.phi, 0, bytemuck::cast_slice(&to_f32(&phi.data)));

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        solver.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
        solver.encode_redistance(&mut pass);
        solver.encode_curvature(&mut pass);
    }
    queue.submit([encoder.finish()]);

    let cells = phi.data.len();
    let gpu_distance: Vec<f32> = read(&device, &queue, &solver.buffers.signed_distance, cells);
    let gpu_kappa: Vec<f32> = read(&device, &queue, &solver.buffers.kappa, cells);
    let cpu_distance = redistance(&phi, grid.dx);
    let cpu_kappa = curvature(&grid, &cpu_distance);

    let mut to_cpu = 0.0f64;
    let mut to_exact = 0.0f64;
    let mut kappa_error = 0.0f64;

    for n in 0..cells {
        let exact = (grid.cell_center(phi.coords(n)) - center).length() - radius;

        if exact.abs() < 2.5 * grid.dx {
            to_cpu = to_cpu.max((gpu_distance[n] as f64 - cpu_distance.data[n]).abs());
            to_exact = to_exact.max((gpu_distance[n] as f64 - exact).abs());
        }

        if cpu_distance.data[n].abs() < 2.0 * grid.dx {
            kappa_error = kappa_error.max((gpu_kappa[n] as f64 - cpu_kappa.data[n]).abs());
        }
    }

    let sphere = 2.0 / radius;

    println!(
        "redistance within 2.5 cells: max |gpu - cpu| {:.3e} dx, max |gpu - exact| {:.3e} dx; curvature within 2 cells: max |gpu - cpu| {:.3e} of 2/R",
        to_cpu / grid.dx,
        to_exact / grid.dx,
        kappa_error / sphere
    );

    assert!(to_cpu < 1e-4 * grid.dx);
    assert!(to_exact < 0.05 * grid.dx);
    assert!(kappa_error < 1e-3 * sphere);
}

const RADIUS: f64 = 0.01;

const SIGMA: f64 = 0.07;

fn droplet(cells_per_radius: u32, stretch: f64) -> Simulation {
    let dx = RADIUS / cells_per_radius as f64;
    let grid = Grid::new(glam::UVec3::splat(2 * cells_per_radius + 8), dx);
    let center = grid.size() * 0.5;
    let axes = DVec3::new(stretch, 1.0 / stretch.sqrt(), 1.0 / stretch.sqrt()) * RADIUS;
    let particles: Vec<Particle> = seed_box(&grid, DVec3::ZERO, grid.size(), SEED_JITTER)
        .into_iter()
        .filter(|p| ((p.x - center) / axes).length_squared() < 1.0)
        .collect();
    let params = Params { gravity: DVec3::ZERO, surface_tension: SIGMA, ..Params::default() };

    Simulation::new(grid, params, particles, Vec::new())
}

fn equivalent_radius(count: usize, dx: f64) -> f64 {
    let volume = count as f64 * dx.powi(3) / 8.0;

    (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt()
}

fn to_gpu(id: usize, p: &Particle) -> GpuParticle {
    let f = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    let mut x = f(p.x);
    x[3] = id as f32;

    GpuParticle { x, v: f(p.v), c: [f(p.c[0]), f(p.c[1]), f(p.c[2])] }
}

fn core_average(grid: &Grid, pressure: &[f64], radius: f64) -> f64 {
    let center = grid.size() * 0.5;
    let cells = grid.cells();
    let mut sum = 0.0;
    let mut count = 0.0;

    for (n, p) in pressure.iter().enumerate() {
        if (grid.cell_center(cells.coords(n)) - center).length() < 0.5 * radius {
            sum += p;
            count += 1.0;
        }
    }

    sum / count
}

struct GpuDroplet {
    device: wgpu::Device,
    queue: wgpu::Queue,
    solver: GpuSolver,
    grid: Grid,
    count: usize,
}

impl GpuDroplet {
    fn new(sim: &Simulation, frame_dt: f32, max_substeps: u32) -> Self {
        let (device, queue) = gpu();
        let d = sim.grid.dims;
        let config = SolverConfig {
            dims: UVec3::new(d.x, d.y, d.z),
            dx: sim.grid.dx as f32,
            max_particles: sim.particles.len() as u32,
            gravity: Vec3::ZERO,
            max_substeps,
            pressure_iterations: 40,
            density_iterations: 20,
            tolerance: 1e-6,
            coarse_sweeps: 16,
            surface_tension: SIGMA as f32,
            viscosity: sim.params.viscosity as f32,
            viscosity_iterations: if sim.params.viscosity > 0.0 { 300 } else { 60 },
            ..SolverConfig::default()
        };
        let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
        solver.initialize(&queue);
        solver.set_frame_dt(&queue, frame_dt);
        solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

        Self { device, queue, solver, grid: sim.grid, count: sim.particles.len() }
    }

    fn frame(&self) -> GpuState {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.solver.encode_frame(&mut encoder, None);
        self.queue.submit([encoder.finish()]);

        read(&self.device, &self.queue, &self.solver.buffers.state, 1)[0]
    }

    fn particles(&self) -> Vec<GpuParticle> {
        read(&self.device, &self.queue, &self.solver.buffers.particles, self.count)
    }

    fn cells(&self, buffer: &wgpu::Buffer) -> Vec<f64> {
        let stored: Vec<f32> = read(&self.device, &self.queue, buffer, ((buffer.size() - GUARD_BYTES) / 4) as usize);
        let table = self.solver.slot_table();
        let words: Vec<u32> = read(&self.device, &self.queue, &self.solver.buffers.brick_list, (table.end / 4) as usize);
        let slots = &words[(table.start / 4) as usize..];
        let d = self.solver.config.dims;
        let mut out = Vec::new();

        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    out.push(stored[self.solver.pooled_cell(slots, UVec3::new(x, y, z))] as f64);
                }
            }
        }

        out
    }

    fn pressure(&self) -> Vec<f64> {
        self.cells(&self.solver.levels[0].x)
    }

    fn max_speed(&self) -> f64 {
        self.particles().iter().fold(0.0f64, |m, p| m.max(Vec3::new(p.v[0], p.v[1], p.v[2]).length() as f64))
    }
}

fn assert_substep_matches_reference(mut sim: Simulation, warmup: usize, speed_tolerance: f64) {
    for _ in 0..warmup {
        let dt = sim.cfl_dt();
        sim.substep(dt);
    }

    let frame_dt = (0.999 * sim.cfl_dt()) as f32;
    let gpu = GpuDroplet::new(&sim, frame_dt, 1);
    let state = gpu.frame();
    let particles = gpu.particles();
    let pressure = gpu.pressure();
    let scalars: Vec<f32> = read(&gpu.device, &gpu.queue, &gpu.solver.buffers.scalars, SCALAR_COUNT);
    sim.substep(frame_dt as f64);

    let radius = equivalent_radius(sim.particles.len(), sim.grid.dx);
    let vmax = sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length()));
    let mut dx = 0.0f64;
    let mut dv = 0.0f64;

    for p in &particles {
        let cpu = &sim.particles[p.x[3] as usize];
        dx = dx.max((DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64) - cpu.x).length());
        dv = dv.max((DVec3::new(p.v[0] as f64, p.v[1] as f64, p.v[2] as f64) - cpu.v).length());
    }

    let gpu_core = core_average(&sim.grid, &pressure, radius);
    let cpu_core = core_average(&sim.grid, &sim.pressure, radius);
    let laplace = 2.0 * SIGMA / radius;
    let worst = pressure.iter().zip(&sim.pressure).fold(0.0f64, |m, (g, c)| m.max((g - c).abs()));

    let cpu_solve = sim.report.viscosity.or(sim.report.capillary).map_or(0, |s| s.iterations);

    println!(
        "droplet substep (8 cells, mu {} Pa s): dt gpu {:.6e} cpu {frame_dt:.6e}; max |dx| {:.3e} cells, max |dv| {dv:.3e} m/s of vmax {vmax:.3e} (relative {:.3e}); core pressure gpu {gpu_core:.4} cpu {cpu_core:.4} Pa (2 sigma/R {laplace:.4}), max |dp| {worst:.3e} Pa ({:.3e} of 2 sigma/R); cg iterations gpu viscous {} capillary {} cpu {cpu_solve}",
        sim.params.viscosity,
        state.dt,
        dx / sim.grid.dx,
        dv / vmax,
        worst / laplace,
        scalars[S_VISCOUS_ITERATIONS],
        scalars[S_CAPILLARY_ITERATIONS]
    );

    assert_eq!(state.substeps, 1);
    assert_eq!(sim.report.capillary.is_some(), sim.params.viscosity == 0.0);
    assert!(dx / sim.grid.dx < 1e-3);
    assert!(dv < speed_tolerance * vmax.max(1e-3));
    assert!(worst < 1e-3 * laplace);
}

#[test]
fn gpu_droplet_substep_matches_reference() {
    assert_substep_matches_reference(droplet(8, 1.0), 3, 1e-2);
}

#[test]
fn gpu_viscous_droplet_substep_matches_reference() {
    let mut sim = droplet(8, 1.15);
    sim.params.viscosity = 1.0;

    assert_substep_matches_reference(sim, 10, 1e-3);
}

#[test]
fn gpu_static_droplet_pressure_matches_laplace() {
    let mut errors = Vec::new();

    for cells in [4, 8, 16] {
        let sim = droplet(cells, 1.0);
        let radius = equivalent_radius(sim.particles.len(), sim.grid.dx);
        let gpu = GpuDroplet::new(&sim, 1.0 / 60.0, 1);
        let capillary = liquid_ref::surface_tension::capillary_dt(&sim.grid, 1000.0, SIGMA);
        let mut pressure = 0.0;
        let mut speed = 0.0f64;
        let mut dt = 0.0f32;

        for n in 0..20 {
            dt = gpu.frame().dt;

            if n >= 10 {
                pressure += core_average(&gpu.grid, &gpu.pressure(), radius) / 10.0;
                speed = speed.max(gpu.max_speed());
            }
        }

        let laplace = 2.0 * SIGMA / radius;
        let error = (pressure - laplace).abs() / laplace;

        println!(
            "radius {cells} cells: gpu dt {dt:.6e} vs capillary bound {capillary:.6e}; core pressure {pressure:.3} Pa vs 2 sigma/R {laplace:.3} Pa, error {:.2}%, max spurious speed {speed:.2e} m/s",
            100.0 * error
        );

        assert!((dt as f64 - capillary).abs() < 1e-6 * capillary);

        if cells == 8 {
            let mut cpu = droplet(cells, 1.0);
            let mut reference = 0.0;

            for n in 0..20 {
                let dt = cpu.cfl_dt();
                cpu.substep(dt);

                if n >= 10 {
                    reference += core_average(&cpu.grid, &cpu.pressure, radius) / 10.0;
                }
            }

            let difference = (pressure - reference).abs() / reference;
            println!("radius {cells} cells: cpu core pressure {reference:.3} Pa, gpu - cpu {difference:.2e} relative");

            assert!(difference < 1e-2);
        }

        errors.push(error);
    }

    assert!(errors[2] < errors[0]);
    assert!(errors[2] < 0.05);
}

fn rayleigh_period(count: usize, dx: f64) -> f64 {
    let omega = (8.0 * SIGMA / (1000.0 * equivalent_radius(count, dx).powi(3))).sqrt();

    2.0 * std::f64::consts::PI / omega
}

fn axis_moment(positions: &[DVec3]) -> f64 {
    let n = positions.len() as f64;
    let center: DVec3 = positions.iter().sum::<DVec3>() / n;

    positions
        .iter()
        .map(|x| {
            let r = *x - center;
            r.x * r.x - 0.5 * (r.y * r.y + r.z * r.z)
        })
        .sum::<f64>()
        / n
}

fn mean_spacing(crossings: &[f64]) -> f64 {
    (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64
}

struct Crossings {
    times: Vec<f64>,
    previous: (f64, f64),
}

impl Crossings {
    fn new(moment: f64) -> Self {
        Self { times: Vec::new(), previous: (0.0, moment) }
    }

    fn push(&mut self, time: f64, moment: f64) {
        let (t0, m0) = self.previous;

        if (moment < 0.0) != (m0 < 0.0) {
            self.times.push(t0 + (time - t0) * m0 / (m0 - moment));
        }

        self.previous = (time, moment);
    }
}

fn gpu_period(cells_per_radius: u32) -> (f64, f64, usize) {
    let sim = droplet(cells_per_radius, 1.15);
    let period = rayleigh_period(sim.particles.len(), sim.grid.dx);
    let gpu = GpuDroplet::new(&sim, 1.0 / 240.0, 4);
    let mut crossings = Crossings::new(axis_moment(&sim.particles.iter().map(|p| p.x).collect::<Vec<_>>()));
    let mut time = 0.0;

    while time < 3.0 * period {
        time = gpu.frame().time as f64;
        let positions: Vec<DVec3> = gpu.particles().iter().map(|p| DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64)).collect();
        crossings.push(time, axis_moment(&positions));
    }

    (2.0 * mean_spacing(&crossings.times), period, crossings.times.len())
}

#[test]
fn gpu_oscillating_droplet_follows_rayleigh_period() {
    let runs: Vec<(u32, (f64, f64, usize))> = [8, 12].into_iter().map(|c| (c, gpu_period(c))).collect();
    let errors: Vec<f64> = runs.iter().map(|(_, (measured, period, _))| (measured - period).abs() / period).collect();

    for ((c, (measured, period, count)), e) in runs.iter().zip(&errors) {
        println!(
            "radius {c} cells: gpu period {measured:.4} s from {count} zero crossings vs Rayleigh {period:.4} s, error {:.2}%",
            100.0 * e
        );
    }

    assert!(runs.iter().all(|(_, (_, _, count))| *count >= 4));
    assert!(errors[1] < errors[0]);
    assert!(errors[1] < 0.05);
}

#[test]
fn gpu_static_droplet_stays_still() {
    let sim = droplet(12, 1.0);
    let period = rayleigh_period(sim.particles.len(), sim.grid.dx);
    let gpu = GpuDroplet::new(&sim, 1.0 / 240.0, 4);
    let mut startup = 0.0f64;
    let mut settled = 0.0f64;
    let mut drift = 0.0f64;
    let mut time = 0.0;
    let mut initial = None;
    let mut frame = 0;

    while time < 2.0 * period {
        time = gpu.frame().time as f64;
        let speed = gpu.max_speed();

        if time < 0.5 * period {
            startup = startup.max(speed);
        } else {
            settled = settled.max(speed);
        }

        frame += 1;

        if frame % 4 == 0 {
            let phi = gpu.cells(&gpu.solver.buffers.phi);
            let volume: f64 = phi.iter().map(|p| (0.5 - p / sim.grid.dx).clamp(0.0, 1.0)).sum();
            let reference = *initial.get_or_insert(volume);
            drift = drift.max((volume / reference - 1.0).abs());
        }
    }

    println!(
        "radius 12 cells, 2 Rayleigh periods: max speed {startup:.4} m/s in the first half period, {settled:.4} m/s afterwards, worst volume drift {drift:.4} (speed sampled every 4 substeps, volume every 16)"
    );

    assert!(settled < 0.006);
    assert!(drift < 0.01);
}

const HONEY_DX: f32 = 0.02;

const SUBCELL_OFFSETS: [Vec3; 3] = [Vec3::ZERO, Vec3::new(0.3, 0.25, 0.2), Vec3::new(-0.4, -0.15, 0.1)];

fn falling_lateral_speed(gpu: &(wgpu::Device, wgpu::Queue), quality: Quality, thread: bool, radius_cells: f32, offset: Vec3, viscosity: f32) -> f32 {
    let (device, queue) = gpu;
    let dims = UVec3::new(24, 72, 24);
    let config = SolverConfig {
        dims,
        dx: HONEY_DX,
        max_particles: 100_000,
        cfl: 2.5,
        tolerance: 1e-3,
        viscosity_iterations: 40,
        density: 1420.0,
        viscosity,
        surface_tension: 4.0,
        ..SolverConfig::default()
    }
    .with_quality(quality);
    let config = SolverConfig { dims, dx: HONEY_DX, ..config };
    let size = dims.as_vec3() * HONEY_DX;
    let radius = radius_cells * HONEY_DX;
    let half = 10.0 * HONEY_DX;
    let center = Vec3::new(0.5 * size.x, size.y - 6.0 * HONEY_DX - half, 0.5 * size.z) + offset * HONEY_DX;
    let mut seed = liquid_gpu::plugin::seed_box(&config, Vec3::ZERO, size);

    seed.retain(|p| {
        let d = Vec3::new(p.x[0], p.x[1], p.x[2]) - center;

        if thread { d.y.abs() < half && d.x.hypot(d.z) < radius } else { d.length() < radius }
    });

    let mut solver = validated(device, || GpuSolver::new(device, config, None));
    solver.initialize(queue);
    solver.upload_particles(queue, &seed);

    for _ in 0..20 {
        solver.set_frame_dt(queue, 1.0 / 60.0);

        let mut encoder = device.create_command_encoder(&Default::default());
        validated(device, || solver.encode_frame(&mut encoder, None));
        queue.submit([encoder.finish()]);
    }

    let particles: Vec<GpuParticle> = read(device, queue, &solver.buffers.particles, seed.len());
    let alive: Vec<Vec3> = particles.iter().filter(|p| p.v[3] < 0.5).map(|p| Vec3::new(p.v[0], p.v[1], p.v[2])).collect();
    let mean = alive.iter().sum::<Vec3>() / alive.len() as f32;

    mean.x.hypot(mean.z)
}

fn worst_lateral_speed(gpu: &(wgpu::Device, wgpu::Queue), quality: Quality, thread: bool, radius_cells: f32, viscosity: f32) -> f32 {
    SUBCELL_OFFSETS
        .into_iter()
        .map(|o| falling_lateral_speed(gpu, quality, thread, radius_cells, o, viscosity))
        .fold(0.0, f32::max)
}

#[test]
fn gpu_falling_thread_keeps_its_lateral_momentum() {
    let gpu = gpu();

    for quality in [Quality::Game, Quality::High] {
        for viscosity in [1500.0, 0.0] {
            for radius in [0.75, 1.0] {
                let speed = worst_lateral_speed(&gpu, quality, true, radius, viscosity);

                println!(
                    "{} thread of radius {radius} cells, {viscosity} Pa s, 20 frames of free fall: worst mean lateral speed {speed:.3e} m/s over 3 sub-cell offsets",
                    quality.name()
                );

                assert!(speed < 6e-3);
            }
        }
    }
}

#[test]
fn gpu_falling_ball_keeps_its_lateral_momentum() {
    let gpu = gpu();

    for quality in [Quality::Game, Quality::High] {
        let speed = worst_lateral_speed(&gpu, quality, false, 1.5, 0.0);

        println!(
            "{} ball of radius 1.5 cells, 0 Pa s, 20 frames of free fall: worst mean lateral speed {speed:.3e} m/s over 3 sub-cell offsets",
            quality.name()
        );

        assert!(speed < 1e-3);
    }
}
