use std::time::Instant;

use bevy::math::{UVec3, Vec3};
use bevy::transform::components::GlobalTransform;
use liquid_gpu::layout::{GpuBody, GpuParticle, GpuState, S_CAPILLARY_ITERATIONS, S_ITERATION, S_VISCOUS_ITERATIONS, SCALAR_COUNT};
use liquid_gpu::plugin::{LiquidBody, LiquidShape, body_to_gpu, seed_box};
use liquid_gpu::solver::{GpuSolver, Profiler, Section, SolverConfig};

const FRAMES: usize = 90;
const SETTLE: usize = 30;

struct Run {
    label: String,
    solver: GpuSolver,
    profiler: Profiler,
    sections: Vec<Vec<f32>>,
    gpu: Vec<f32>,
    finish: Vec<f32>,
    bricks: Vec<f32>,
}

fn main() {
    let flags = wgpu::InstanceFlags::from_build_config().difference(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL).with_env();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { flags, ..wgpu::InstanceDescriptor::new_without_display_handle() });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("no adapter");
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: wgpu::Features::TIMESTAMP_QUERY | (adapter.features() & wgpu::Features::SUBGROUP),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no device");

    let only = std::env::args().nth(1).unwrap_or_default();
    let scenes = [
        ("bench scene", UVec3::new(96, 64, 64), 512),
        ("wide scene", UVec3::new(192, 64, 128), 1024),
        ("tall sparse scene", UVec3::new(96, 192, 64), 1024),
    ];

    for (name, dims, capacity) in scenes {
        if only.is_empty() || name.contains(only.as_str()) {
            scene(&device, &queue, name, dims, capacity);
        }
    }

    if only.is_empty() || "demo pressure".contains(only.as_str()) {
        demo_pressure(&device, &queue);
    }

    if only.is_empty() {
        extensions(&device, &queue);
        reductions(&device, &queue);
    }
}

fn median(values: &mut [f32]) -> f32 {
    values.sort_by(f32::total_cmp);

    values[values.len() / 2]
}

fn reductions(device: &wgpu::Device, queue: &wgpu::Queue) {
    let dx = 1.0 / 64.0;
    let particles = column(dx);
    let base = SolverConfig {
        dims: UVec3::new(96, 64, 64),
        dx,
        max_particles: particles.len() as u32,
        max_substeps: 2,
        brick_capacity: Some(512),
        density_projection: false,
        ..SolverConfig::default()
    };
    let configs = [
        ("shared-memory reductions", SolverConfig { subgroups: false, ..base.clone() }),
        ("subgroup reductions", SolverConfig { subgroups: true, ..base.clone() }),
        ("shared-memory reductions, boundary smoothing 2/2", SolverConfig { subgroups: false, boundary_width: 2, boundary_sweeps: 2, ..base.clone() }),
        ("subgroup reductions, boundary smoothing 2/2", SolverConfig { subgroups: true, boundary_width: 2, boundary_sweeps: 2, ..base.clone() }),
    ];
    let mut variants: Vec<Timed> = configs
        .into_iter()
        .map(|(name, config)| {
            let mut solver = GpuSolver::new(device, config, None);
            solver.initialize(queue);
            solver.set_frame_dt(queue, 1.0 / 60.0);
            solver.upload_particles(queue, &particles);

            Timed { name, solver, profiler: Profiler::new(device, queue, 2), times: Vec::new(), iterations: Vec::new() }
        })
        .collect();
    let pressure = Section::ALL.iter().position(|s| *s == Section::Pressure).unwrap_or(0);

    for frame in 0..FRAMES {
        for v in &mut variants {
            let mut encoder = device.create_command_encoder(&Default::default());
            v.solver.encode_frame(&mut encoder, Some(&v.profiler));
            v.profiler.resolve(&mut encoder);
            queue.submit([encoder.finish()]);

            let sections = v.profiler.read_milliseconds(device);

            if frame >= SETTLE {
                let bytes = read(device, queue, &v.solver.buffers.scalars, (SCALAR_COUNT * 4) as u64);
                let scalars: &[f32] = bytemuck::cast_slice(&bytes);
                v.times.push(sections[pressure].1);
                v.iterations.push(scalars[S_ITERATION]);
            }
        }
    }

    println!("pressure section on the 96x64x64 sparse scene without density projection, 2 solver updates per frame, frames {SETTLE}..{FRAMES} in lockstep:");

    for v in &mut variants {
        println!(
            "{} (subgroups active: {}): median pressure {:.3} ms, min {:.3} ms, median pcg iteration counter {}",
            v.name,
            v.solver.subgroups,
            median(&mut v.times),
            v.times[0],
            median(&mut v.iterations)
        );
    }
}

struct Timed {
    name: &'static str,
    solver: GpuSolver,
    profiler: Profiler,
    times: Vec<f32>,
    iterations: Vec<f32>,
}

fn column(dx: f32) -> Vec<GpuParticle> {
    let column = UVec3::new(32, 48, 64);
    let mut particles = Vec::new();

    for k in 0..column.z {
        for j in 0..column.y {
            for i in 0..column.x {
                for s in 0..8u32 {
                    let sub = Vec3::new((s & 1) as f32, ((s >> 1) & 1) as f32, ((s >> 2) & 1) as f32);
                    let x = (Vec3::new(i as f32, j as f32, k as f32) + (sub + 0.5) * 0.5) * dx;
                    let id = particles.len() as f32;
                    particles.push(GpuParticle { x: [x.x, x.y, x.z, id], ..Default::default() });
                }
            }
        }
    }

    particles
}

fn scene(device: &wgpu::Device, queue: &wgpu::Queue, name: &str, dims: UVec3, capacity: u32) {
    let dx = 1.0 / 64.0;
    let particles = column(dx);

    let mut runs: Vec<Run> = [(false, false), (false, true), (true, false), (true, true)]
        .into_iter()
        .map(|(sparse, subgroups)| {
            let config = SolverConfig {
                dims,
                dx,
                max_particles: particles.len() as u32,
                max_substeps: 2,
                sparse,
                subgroups,
                brick_capacity: Some(capacity),
                ..SolverConfig::default()
            };
            let mut solver = GpuSolver::new(device, config, None);
            solver.initialize(queue);
            solver.set_frame_dt(queue, 1.0 / 60.0);
            solver.upload_particles(queue, &particles);

            let mode = if sparse { "sparse" } else { "dense" };
            let lanes = if solver.subgroups { "subgroups" } else { "shared" };

            Run {
                label: format!("{mode} {lanes}"),
                profiler: Profiler::new(device, queue, 2),
                solver,
                sections: vec![Vec::new(); Section::ALL.len()],
                gpu: Vec::new(),
                finish: Vec::new(),
                bricks: Vec::new(),
            }
        })
        .collect();

    println!(
        "{name}: cells {} particles {} levels {} bricks {} pool capacity {capacity}, grid bytes dense {} sparse {}",
        dims.x * dims.y * dims.z,
        particles.len(),
        runs[0].solver.levels.len(),
        runs[0].solver.brick_count(),
        runs[0].solver.grid_bytes(),
        runs[2].solver.grid_bytes()
    );

    for frame in 0..FRAMES {
        for run in &mut runs {
            let mut encoder = device.create_command_encoder(&Default::default());
            run.solver.encode_frame(&mut encoder, Some(&run.profiler));
            run.profiler.resolve(&mut encoder);

            let finish = Instant::now();
            let commands = encoder.finish();
            let finish = finish.elapsed().as_secs_f32() * 1000.0;
            queue.submit([commands]);

            let times = run.profiler.read_milliseconds(device);

            if frame < SETTLE {
                continue;
            }

            for (samples, (_, t)) in run.sections.iter_mut().zip(&times) {
                samples.push(*t);
            }

            run.gpu.push(times.iter().map(|t| t.1).sum());
            run.finish.push(finish);

            if run.solver.config.sparse {
                let active: u32 = *bytemuck::from_bytes(&read(device, queue, &run.solver.buffers.brick_list, 4));
                run.bricks.push(active as f32 / run.solver.brick_count() as f32);
            }
        }
    }

    let shown: Vec<usize> = (0..Section::ALL.len()).filter(|s| runs.iter().any(|r| median(&mut r.sections[*s].clone()) > 0.005)).collect();
    let header: Vec<&str> = shown.iter().map(|s| Section::ALL[*s].label().trim_start_matches("liquid ")).collect();

    println!("{name}: per-section gpu medians in ms over frames {SETTLE}..{FRAMES}, 4 variants stepped in lockstep");
    println!("{:<18} {:>7} {}", "variant", "total", header.iter().map(|h| format!("{h:>9}")).collect::<String>());

    for run in &mut runs {
        let cells: String = shown.iter().map(|s| format!("{:>9.3}", median(&mut run.sections[*s]))).collect();
        println!("{:<18} {:>7.3} {cells}", run.label, median(&mut run.gpu));
    }

    let totals: Vec<f32> = runs.iter_mut().map(|r| median(&mut r.gpu)).collect();
    let sparse = &mut runs[2];
    let state: GpuState = *bytemuck::from_bytes(&read(device, queue, &sparse.solver.buffers.state, std::mem::size_of::<GpuState>() as u64));

    println!(
        "{name}: sparse/dense total ratio shared {:.3}, subgroups {:.3}; median active brick fraction {:.1}%, unallocated {}, median encoder.finish {:.2} ms",
        totals[2] / totals[0],
        totals[3] / totals[1],
        100.0 * median(&mut sparse.bricks),
        state.brick_overflow,
        median(&mut sparse.finish)
    );
}

fn demo_pressure(device: &wgpu::Device, queue: &wgpu::Queue) {
    let base = SolverConfig {
        dims: UVec3::new(64, 32, 48),
        dx: 1.0 / 16.0,
        max_particles: 600_000,
        cfl: 3.0,
        max_substeps: 6,
        narrow_band: Some(3.0),
        ..SolverConfig::default()
    };
    let variants = [
        ("cap 24, tolerance 1e-5", base.clone()),
        ("cap 12, tolerance 1e-5", SolverConfig { pressure_iterations: 12, ..base.clone() }),
        ("cap 24, forced", SolverConfig { tolerance: 0.0, ..base.clone() }),
        ("cap 12, forced", SolverConfig { pressure_iterations: 12, tolerance: 0.0, ..base.clone() }),
        ("cap 12, tolerance 1e-5, 2 coarse sweeps", SolverConfig { pressure_iterations: 12, coarse_sweeps: 2, ..base.clone() }),
    ];

    for (label, config) in variants {
        demo_variant(device, queue, label, config);
    }
}

fn demo_variant(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, config: SolverConfig) {
    let size = config.dims.as_vec3() * config.dx;
    let mut particles = seed_box(&config, Vec3::ZERO, Vec3::new(1.4, 1.3, size.z));
    particles.extend(seed_box(&config, Vec3::new(1.4, 0.0, 0.0), Vec3::new(size.x, 0.45, size.z)));

    for (i, p) in particles.iter_mut().enumerate() {
        p.x[3] = i as f32;
    }

    let wood = LiquidBody { shape: LiquidShape::Cuboid(Vec3::new(0.2, 0.12, 0.3)), density: 450.0, dynamic: true };
    let bodies: Vec<GpuBody> = [Vec3::new(2.2, 0.4, 0.9), Vec3::new(3.0, 0.4, 2.1)]
        .into_iter()
        .map(|p| body_to_gpu(&wood, &GlobalTransform::from_translation(p), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO))
        .collect();

    let mut solver = GpuSolver::new(device, config, None);
    solver.initialize(queue);
    solver.set_frame_dt(queue, 1.0 / 60.0);
    solver.upload_particles(queue, &particles);
    solver.upload_bodies(queue, &bodies);

    let profiler = Profiler::new(device, queue, solver.config.max_substeps);
    let pressure = Section::ALL.iter().position(|s| *s == Section::Pressure).unwrap_or(0);
    let (mut times, mut per_substep, mut dead, mut encode, mut iterations, mut substeps) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());

    for frame in 0..150 {
        let start = Instant::now();
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, Some(&profiler));
        profiler.resolve(&mut encoder);
        let commands = encoder.finish();
        let cpu = start.elapsed().as_secs_f32() * 1000.0;
        queue.submit([commands]);

        profiler.readback().map_async(wgpu::MapMode::Read, .., |r| r.expect("timestamp map failed"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let stamps: Vec<u64> = bytemuck::cast_slice(&profiler.readback().get_mapped_range(..).expect("timestamp range")).to_vec();
        profiler.readback().unmap();
        let sections = profiler.milliseconds(&stamps, solver.config.max_substeps);
        let state: GpuState = *bytemuck::from_bytes(&read(device, queue, &solver.buffers.state, std::mem::size_of::<GpuState>() as u64));
        let period = queue.get_timestamp_period() * 1e-6;
        let pass = |k: u32| {
            let i = 2 * (pressure as u32 * solver.config.max_substeps + k) as usize;
            stamps[i + 1].saturating_sub(stamps[i]) as f32 * period
        };
        let bytes = read(device, queue, &solver.buffers.scalars, (SCALAR_COUNT * 4) as u64);
        let scalars: &[f32] = bytemuck::cast_slice(&bytes);

        if frame >= SETTLE {
            times.push(sections[pressure].1);
            per_substep.push(pass(0));
            dead.push(pass(solver.config.max_substeps - 1));
            encode.push(cpu);
            iterations.push(scalars[S_ITERATION] - 1.0);
            substeps.push(state.substeps as f32);
        }
    }

    println!(
        "demo pressure scene 64x32x48, two dynamic boxes, {label}, frames {SETTLE}..150: pressure median {:.3} ms per frame, first substep {:.3} ms, last substep {:.3} ms, substeps {}, last-solve pcg iterations {}, encode+finish {:.2} ms",
        median(&mut times),
        median(&mut per_substep),
        median(&mut dead),
        median(&mut substeps),
        median(&mut iterations),
        median(&mut encode)
    );
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, bytes: u64) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);
    staging.map_async(wgpu::MapMode::Read, .., |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let view = staging.get_mapped_range(..).expect("range");

    view.to_vec()
}

struct Variant {
    name: &'static str,
    solver: GpuSolver,
    profiler: Profiler,
    sections: Vec<f32>,
    iterations: [f32; 2],
}

fn extensions(device: &wgpu::Device, queue: &wgpu::Queue) {
    let dx = 1.0 / 64.0;
    let particles = column(dx);
    let base = SolverConfig {
        dims: UVec3::new(96, 64, 64),
        dx,
        max_particles: particles.len() as u32,
        max_substeps: 2,
        brick_capacity: Some(512),
        ..SolverConfig::default()
    };
    let configs = [
        ("baseline", base.clone()),
        ("narrow band", SolverConfig { narrow_band: Some(3.0), ..base.clone() }),
        ("viscosity 1 Pa s", SolverConfig { viscosity: 1.0, ..base.clone() }),
        ("viscosity 1 Pa s, 20 cg iterations", SolverConfig { viscosity: 1.0, viscosity_iterations: 20, ..base.clone() }),
        ("surface tension 0.07 N/m", SolverConfig { surface_tension: 0.07, ..base.clone() }),
    ];
    let mut variants: Vec<Variant> = configs
        .into_iter()
        .map(|(name, config)| {
            let mut solver = GpuSolver::new(device, config, None);
            solver.initialize(queue);
            solver.set_frame_dt(queue, 1.0 / 60.0);
            solver.upload_particles(queue, &particles);

            Variant { name, solver, profiler: Profiler::new(device, queue, 2), sections: vec![0.0; Section::ALL.len()], iterations: [0.0; 2] }
        })
        .collect();

    for frame in 0..FRAMES {
        for v in &mut variants {
            let mut encoder = device.create_command_encoder(&Default::default());
            v.solver.encode_frame(&mut encoder, Some(&v.profiler));
            v.profiler.resolve(&mut encoder);
            queue.submit([encoder.finish()]);

            let times = v.profiler.read_milliseconds(device);

            if frame >= SETTLE {
                for (total, (_, t)) in v.sections.iter_mut().zip(&times) {
                    *total += t / (FRAMES - SETTLE) as f32;
                }

                let bytes = read(device, queue, &v.solver.buffers.scalars, (SCALAR_COUNT * 4) as u64);
                let scalars: &[f32] = bytemuck::cast_slice(&bytes);
                v.iterations[0] = v.iterations[0].max(scalars[S_VISCOUS_ITERATIONS]);
                v.iterations[1] = v.iterations[1].max(scalars[S_CAPILLARY_ITERATIONS]);
            }
        }
    }

    println!("extensions on the 96x64x64 sparse scene, frames stepped in lockstep, mean over frames {SETTLE}..{FRAMES}:");

    let baseline: f32 = variants[0].sections.iter().sum();

    for v in &variants {
        let total: f32 = v.sections.iter().sum();
        let state: GpuState = *bytemuck::from_bytes(&read(device, queue, &v.solver.buffers.state, std::mem::size_of::<GpuState>() as u64));
        let parts: Vec<String> = Section::ALL
            .iter()
            .zip(&v.sections)
            .filter(|(_, t)| **t > 0.005)
            .map(|(s, t)| format!("{}={t:.2}", s.label().trim_start_matches("liquid ")))
            .collect();

        println!(
            "{}: gpu {total:.2} ms ({:+.2} ms vs baseline), particles {}, max cg iterations viscous {} capillary {} | {}",
            v.name,
            total - baseline,
            state.particle_count,
            v.iterations[0],
            v.iterations[1],
            parts.join(" ")
        );
    }
}
