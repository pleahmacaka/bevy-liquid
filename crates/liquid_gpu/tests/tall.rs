use bevy::math::{UVec3, Vec3};
use glam::DVec3;
use liquid_gpu::layout::{GpuBody, GpuParticle, GpuState, S_ITERATION, SCALAR_COUNT};
use liquid_gpu::solver::{GUARD_BYTES, GpuSolver, Profiler, Section, SolverConfig};
use liquid_ref::grid::Grid;
use liquid_ref::particles::{Particle, seed_box};
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{SEED_JITTER, moving, with_body};
use liquid_ref::sim::{Params, Simulation};

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

fn gpu(features: wgpu::Features, limits: impl FnOnce(wgpu::Limits) -> wgpu::Limits) -> (wgpu::Device, wgpu::Queue) {
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
        label: Some("tall test"),
        required_features: features & adapter.features(),
        required_limits: limits(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device")
}

fn desktop(supported: wgpu::Limits) -> wgpu::Limits {
    wgpu::Limits {
        max_storage_buffer_binding_size: supported.max_storage_buffer_binding_size,
        max_buffer_size: supported.max_buffer_size,
        ..limits::portable(supported)
    }
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

fn stored<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<T> {
    read(device, queue, buffer, ((buffer.size() - GUARD_BYTES) / 4) as usize)
}

fn validated<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> T {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = build();

    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation error: {error}");
    }

    out
}

fn slots(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> Vec<u32> {
    let range = solver.slot_table();
    let words: Vec<u32> = read(device, queue, &solver.buffers.brick_list, (range.end / 4) as usize);

    words[(range.start / 4) as usize..].to_vec()
}

fn grid_cells<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver, buffer: &wgpu::Buffer) -> Vec<T> {
    let values: Vec<T> = stored(device, queue, buffer);
    let slots = slots(device, queue, solver);
    let d = solver.config.dims;
    let mut out = Vec::new();

    for z in 0..d.z {
        for y in 0..d.y {
            for x in 0..d.x {
                out.push(values[solver.pooled_cell(&slots, UVec3::new(x, y, z))]);
            }
        }
    }

    out
}

fn grid_faces(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver, buffer: &wgpu::Buffer) -> Vec<f32> {
    let values: Vec<f32> = stored(device, queue, buffer);
    let slots = slots(device, queue, solver);
    let mut out = Vec::new();

    for axis in 0..3 {
        let mut d = solver.config.dims;
        d[axis] += 1;

        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    out.push(values[solver.pooled_face(&slots, axis, UVec3::new(x, y, z))]);
                }
            }
        }
    }

    out
}

fn tops(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> Vec<u32> {
    let d = solver.config.dims;
    let tall: Vec<f32> = stored(device, queue, &solver.levels[0].tall);

    (0..(d.x * d.z) as usize).map(|col| tall.get(col).map_or(0, |t| *t as u32)).collect()
}

fn unknowns(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> usize {
    let d = solver.config.dims;
    let active: Vec<u32> = grid_cells(device, queue, solver, &solver.levels[0].active);
    let tops = if solver.has_tall() { tops(device, queue, solver) } else { vec![0; (d.x * d.z) as usize] };

    (0..active.len())
        .filter(|&n| {
            let n = n as u32;
            let (x, y, z) = (n % d.x, (n / d.x) % d.y, n / (d.x * d.y));
            let top = tops[(x + d.x * z) as usize];

            active[n as usize] != 0 && !(y > 0 && y < top)
        })
        .count()
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

fn relative_gap(a: &[f32], b: &[f32]) -> f64 {
    let scale = a.iter().fold(0.0f64, |m, v| m.max(v.abs() as f64));
    let gap = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs() as f64));

    gap / scale.max(1e-30)
}

fn wave_tank() -> Simulation {
    let dx = 1.0 / 32.0;
    let grid = Grid::new(glam::UVec3::new(24, 40, 12), dx);
    let depth = 34.0 * dx;
    let k = std::f64::consts::PI / grid.size().x;
    let mut water = grid.size();
    water.y = depth;

    let particles = moving(seed_box(&grid, DVec3::ZERO, water, SEED_JITTER), |x| {
        let a = 0.05 * (k * (x.y - depth)).exp();

        DVec3::new(-a * (k * x.x).sin(), a * (k * x.x).cos(), 0.0)
    });
    let params = Params { density_projection: false, tall_cells: Some(4), ..Params::default() };
    let mut body = RigidBody::new(Shape::Sphere { radius: 0.08 }, DVec3::new(0.25, depth, 0.19), Motion::Dynamic, 500.0);
    body.lock_rotation = true;

    with_body(Simulation::new(grid, params, particles, Vec::new()), body)
}

fn open_floor_wave() -> Simulation {
    let dx = 1.0 / 32.0;
    let grid = Grid::new(glam::UVec3::new(32, 40, 4), dx);
    let depth = 34.0 * dx;
    let full_depth = 2.0;
    let k = std::f64::consts::PI / grid.size().x;
    let omega = (9.81 * k * (k * full_depth).tanh()).sqrt();
    let a = 0.5 * dx * 9.81 * k / (omega * (k * full_depth).cosh());
    let mut water = grid.size();
    water.y = depth;

    let particles = moving(seed_box(&grid, DVec3::ZERO, water, SEED_JITTER), |x| {
        let y = x.y - depth + full_depth;

        DVec3::new(-a * (k * y).cosh() * (k * x.x).sin(), a * (k * y).sinh() * (k * x.x).cos(), 0.0)
    });
    let params = Params { open_floor: Some(3.0), density_projection: false, tall_cells: Some(4), ..Params::default() };

    Simulation::new(grid, params, particles, Vec::new())
}

#[test]
fn gpu_tall_cells_match_reference_dense_and_sparse() {
    let (device, queue) = gpu(wgpu::Features::empty(), limits::portable);
    let dt = 1.0 / 120.0;

    for (name, scene) in [("sphere in a wave tank", wave_tank as fn() -> Simulation), ("open floor wave", open_floor_wave)] {
        for sparse in [false, true] {
            let mut sim = scene();
            let d = sim.grid.dims;
            let config = SolverConfig {
                dims: UVec3::new(d.x, d.y, d.z),
                dx: sim.grid.dx as f32,
                max_particles: 2 * sim.particles.len() as u32 + 1024,
                gravity: Vec3::new(0.0, -9.81, 0.0),
                max_substeps: 1,
                pressure_iterations: 60,
                tolerance: 1e-6,
                coarse_sweeps: 16,
                density_projection: false,
                sparse,
                open_floor: sim.params.open_floor.map(|r| r as f32),
                tall_cells: sim.params.tall_cells,
                ..SolverConfig::default()
            };
            let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
            solver.initialize(&queue);
            solver.set_frame_dt(&queue, dt as f32);
            solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

            if !sim.bodies.is_empty() {
                solver.upload_bodies(&queue, &sim.bodies.iter().map(to_gpu_body).collect::<Vec<_>>());
            }

            for frame in 1..=4 {
                let mut encoder = device.create_command_encoder(&Default::default());
                solver.encode_frame(&mut encoder, None);
                queue.submit([encoder.finish()]);
                sim.substep(dt);

                let state: GpuState = read(&device, &queue, &solver.buffers.state, 1)[0];
                let scalars: Vec<f32> = read(&device, &queue, &solver.buffers.scalars, SCALAR_COUNT);
                let gpu_pressure: Vec<f32> = grid_cells(&device, &queue, &solver, &solver.levels[0].x);
                let gpu_velocity = grid_faces(&device, &queue, &solver, &solver.buffers.velocity);
                let cpu_pressure: Vec<f32> = sim.pressure.iter().map(|p| *p as f32).collect();
                let cpu_velocity: Vec<f32> = sim.velocity.axes.iter().flat_map(|a| a.data.iter().map(|v| *v as f32)).collect();
                let pressure = relative_gap(&cpu_pressure, &gpu_pressure);
                let velocity = relative_gap(&cpu_velocity, &gpu_velocity);
                let body = sim.bodies.first().map_or(0.0, |b| {
                    let v = read::<GpuBody>(&device, &queue, &solver.buffers.bodies, 1)[0].velocity;
                    (b.velocity - DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).length() / b.velocity.length().max(1e-3)
                });
                let counted = unknowns(&device, &queue, &solver);
                let tall_columns = tops(&device, &queue, &solver).iter().filter(|t| **t > 0).count();

                println!(
                    "{name}, sparse {sparse}, frame {frame}: pressure unknowns cpu {} gpu {counted} of {} cells ({tall_columns} tall columns), pcg iterations gpu {}, cpu {}; max |dp|/max|p| {pressure:.3e}, max |du|/max|u| {velocity:.3e}, body |dv|/|v| {body:.3e}, particles cpu {} gpu {}",
                    sim.report.unknowns,
                    sim.grid.cell_count(),
                    scalars[S_ITERATION] - 1.0,
                    sim.report.pressure.iterations,
                    sim.particles.len(),
                    state.particle_count
                );

                assert_eq!(state.substeps, 1);
                assert!(tall_columns > 0);
                assert!(counted * 2 < sim.grid.cell_count());
                assert!(counted.abs_diff(sim.report.unknowns) <= 2, "gpu {counted} vs cpu {} unknowns", sim.report.unknowns);
                assert!(pressure < 2e-3, "pressure gap {pressure}");
                assert!(velocity < 2e-3, "velocity gap {velocity}");
                assert!(body < 2e-3, "body velocity gap {body}");
            }

            let violations = solver.guard_violations(&device, &queue);
            assert!(violations.is_empty(), "{violations:?}");
        }
    }
}

fn deep_particles(dims: UVec3, dx: f32, depth: u32) -> Vec<GpuParticle> {
    let k = std::f32::consts::TAU / (dims.x as f32 * dx);
    let surface = depth as f32 * dx;
    let mut out = Vec::new();
    let mut push = |x: Vec3| {
        let a = 0.1 * (k * (x.y - surface)).exp();
        let v = [-a * (k * x.x).sin(), a * (k * x.x).cos(), 0.0, 0.0];
        out.push(GpuParticle { x: [x.x, x.y, x.z, out.len() as f32], v, ..Default::default() });
    };

    for z in 0..dims.z {
        for y in 0..depth {
            for x in 0..dims.x {
                let cell = Vec3::new(x as f32, y as f32, z as f32);

                if y + 4 < depth {
                    push((cell + 0.5) * dx);
                    continue;
                }

                for s in 0..8u32 {
                    let sub = Vec3::new((s & 1) as f32, ((s >> 1) & 1) as f32, ((s >> 2) & 1) as f32);
                    push((cell + (sub + 0.5) * 0.5) * dx);
                }
            }
        }
    }

    out
}

struct Timing {
    unknowns: usize,
    cells: usize,
    pressure_ms: f32,
    iterations: f32,
    guards: usize,
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(f32::total_cmp);

    v[v.len() / 2]
}

fn deep_pair(device: &wgpu::Device, queue: &wgpu::Queue, particles: &[GpuParticle], configs: [SolverConfig; 2], timestamps: bool) -> [Timing; 2] {
    let solvers = configs.map(|config| {
        let solver = validated(device, || GpuSolver::new(device, config, None));
        solver.initialize(queue);
        solver.upload_particles(queue, particles);
        solver
    });
    let profilers = solvers.each_ref().map(|s| timestamps.then(|| Profiler::new(device, queue, s.config.max_substeps)));
    let mut times = [Vec::new(), Vec::new()];
    let mut iterations = [Vec::new(), Vec::new()];

    for frame in 0..36 {
        for (k, solver) in solvers.iter().enumerate() {
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode_frame(&mut encoder, profilers[k].as_ref());

            if let Some(p) = &profilers[k] {
                p.resolve(&mut encoder);
            }

            queue.submit([encoder.finish()]);

            if frame >= 6 {
                if let Some(p) = &profilers[k] {
                    times[k].extend(p.read_milliseconds(device).iter().filter(|(s, _)| *s == Section::Pressure).map(|(_, ms)| *ms));
                }

                iterations[k].push(read::<f32>(device, queue, &solver.buffers.scalars, SCALAR_COUNT)[S_ITERATION] - 1.0);
            }
        }
    }

    let mut k = 0;

    solvers.map(|solver| {
        let timing = Timing {
            unknowns: unknowns(device, queue, &solver),
            cells: solver.config.dims.element_product() as usize,
            pressure_ms: if times[k].is_empty() { f32::NAN } else { median(std::mem::take(&mut times[k])) },
            iterations: median(std::mem::take(&mut iterations[k])),
            guards: solver.guard_violations(device, queue).len(),
        };
        k += 1;

        timing
    })
}

#[test]
fn deep_scene_unknowns_and_pressure_time() {
    let (device, queue) = gpu(wgpu::Features::TIMESTAMP_QUERY, desktop);
    let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let dx = 1.0 / 64.0;

    for (dims, depth, modes) in [(UVec3::new(64, 128, 64), 120, 2), (UVec3::new(128, 256, 128), 248, 1)] {
        let particles = deep_particles(dims, dx, depth);
        let base = SolverConfig {
            dims,
            dx,
            max_particles: particles.len() as u32 + (1 << 16),
            max_substeps: 2,
            narrow_band: Some(3.0),
            ..SolverConfig::default()
        };

        for sparse in [false, true] {
            for (tolerance, iterations) in [(base.tolerance, base.pressure_iterations), (0.0, 30)].into_iter().take(modes) {
                let configs = [None, Some(8)].map(|tall| SolverConfig { sparse, tolerance, pressure_iterations: iterations, tall_cells: tall, ..base.clone() });
                let [regular, tall] = deep_pair(&device, &queue, &particles, configs, timestamps);

                for (name, run) in [("regular", &regular), ("tall band 8", &tall)] {
                    println!(
                        "{dims} surface at {depth}, {} layout, tolerance {tolerance:.0e} (cap {iterations}), {name}: pressure unknowns {} of {} cells, median pressure pass {:.3} ms per frame, median pcg iterations {}",
                        if sparse { "sparse" } else { "dense" },
                        run.unknowns,
                        run.cells,
                        run.pressure_ms,
                        run.iterations
                    );
                }

                assert_eq!(regular.guards + tall.guards, 0);
                assert!(tall.unknowns * 4 < regular.unknowns, "tall {} vs regular {} unknowns", tall.unknowns, regular.unknowns);
            }
        }
    }
}
