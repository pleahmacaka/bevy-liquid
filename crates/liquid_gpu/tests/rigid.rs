use std::collections::HashSet;

use bevy::math::{UVec3, Vec3};
use glam::DVec3;
use liquid_gpu::layout::{GpuBody, GpuParticle, GpuState, S_IMPULSE, SCALAR_COUNT};
use liquid_gpu::solver::{GUARD_BYTES, GpuSolver, Slots, SolverConfig};
use liquid_ref::particles::Particle;
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{tank, with_body};
use liquid_ref::sim::{Params, Simulation};

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
        label: Some("liquid rigid test"),
        required_features: adapter.features() & wgpu::Features::SUBGROUP,
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

fn stored<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<T> {
    read(device, queue, buffer, ((buffer.size() - GUARD_BYTES) as usize) / std::mem::size_of::<T>())
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
    let motion = match b.motion {
        Motion::Static => 0.0,
        Motion::Kinematic => 1.0,
        Motion::Dynamic => 2.0,
    };

    GpuBody {
        position: f(b.position, 1.0 / b.mass()),
        rotation: [b.rotation.x as f32, b.rotation.y as f32, b.rotation.z as f32, b.rotation.w as f32],
        velocity: f(b.velocity, motion),
        angular: f(b.angular_velocity, 0.0),
        shape: f(shape, kind),
        inv_inertia: [f(inv.row(0), 0.0), f(inv.row(1), 0.0), f(inv.row(2), 0.0)],
    }
}

fn floating_bodies() -> Simulation {
    let params = Params { density_projection: false, ..Params::default() };
    let sim = tank(glam::UVec3::new(40, 24, 40), 0.025, 0.35, params);
    let block = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.1, 0.05, 0.14) }, DVec3::new(0.3, 0.35, 0.35), Motion::Dynamic, 450.0);
    let ball = RigidBody::new(Shape::Sphere { radius: 0.08 }, DVec3::new(0.62, 0.33, 0.6), Motion::Dynamic, 700.0);
    let anchor = RigidBody::new(Shape::Sphere { radius: 0.06 }, DVec3::new(0.2, 0.15, 0.8), Motion::Kinematic, 1000.0);

    with_body(with_body(with_body(sim, block), ball), anchor)
}

struct Coupling {
    impulse: Vec<[f64; 6]>,
    expected: Vec<[f64; 6]>,
    magnitude: Vec<[f64; 6]>,
    listed: u32,
    swept: u32,
    tiles: usize,
}

fn couple_once(device: &wgpu::Device, queue: &wgpu::Queue, sim: &Simulation, sparse: bool, subgroups: bool) -> Coupling {
    let dims = UVec3::new(sim.grid.dims.x, sim.grid.dims.y, sim.grid.dims.z);
    let config = SolverConfig {
        dims,
        dx: sim.grid.dx as f32,
        max_particles: sim.particles.len() as u32 + 1024,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        max_substeps: 1,
        pressure_iterations: 60,
        tolerance: 1e-6,
        coarse_sweeps: 16,
        density_projection: false,
        sparse,
        subgroups,
        ..SolverConfig::default()
    };
    let bodies: Vec<GpuBody> = sim.bodies.iter().map(to_gpu_body).collect();

    let mut solver = GpuSolver::new(device, config, None);
    solver.initialize(queue);
    solver.set_frame_dt(queue, 0.004);
    solver.upload_particles(queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    solver.upload_bodies(queue, &bodies);

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    assert!(solver.guard_violations(device, queue).is_empty(), "guard overwritten");

    let state: Vec<GpuState> = read(device, queue, &solver.buffers.state, 1);
    assert_eq!(state[0].substeps, 1);

    let scalars: Vec<f32> = read(device, queue, &solver.buffers.scalars, SCALAR_COUNT);
    let indirect: Vec<u32> = read(device, queue, &solver.buffers.indirect, 3 * solver.slots.count() as usize);
    let table = solver.slot_table();
    let words: Vec<u32> = read(device, queue, &solver.buffers.brick_list, (table.end / 4) as usize);
    let slots = &words[(table.start / 4) as usize..];
    let pressure: Vec<f32> = stored(device, queue, &solver.levels[0].x);
    let active: Vec<u32> = stored(device, queue, &solver.levels[0].active);
    let owners: Vec<i32> = stored(device, queue, &solver.buffers.owners);
    let weights: Vec<f32> = stored(device, queue, &solver.buffers.weights);

    let dx = sim.grid.dx;
    let area = dx * dx;
    let mut expected = vec![[0.0; 6]; bodies.len()];
    let mut magnitude = vec![[0.0; 6]; bodies.len()];
    let mut tiles = HashSet::new();

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let c = UVec3::new(x, y, z);
                let n = solver.pooled_cell(slots, c);

                if active[n] == 0 {
                    continue;
                }

                for axis in 0..3 {
                    for high in 0..2 {
                        let mut f = c;
                        f[axis] += high;

                        let fi = solver.pooled_face(slots, axis, f);
                        let Ok(k) = usize::try_from(owners[fi]) else {
                            continue;
                        };
                        let open = 1.0f32 - weights[fi];

                        if bodies[k].velocity[3] < 1.5 || open <= 0.0 {
                            continue;
                        }

                        tiles.insert(n / 256);

                        let mut e = DVec3::ZERO;
                        e[axis] = 1.0;

                        let mut face = (DVec3::new(f.x as f64, f.y as f64, f.z as f64) + 0.5) * dx;
                        face[axis] = f[axis] as f64 * dx;

                        let p = bodies[k].position;
                        let r = face - DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
                        let sign = if high == 1 { 1.0 } else { -1.0 };
                        let s = sign * open as f64 * area;
                        let torque = r.cross(e);
                        let v = pressure[n] as f64;

                        for d in 0..3 {
                            let terms = [s * e[d] * v, s * torque[d] * v];

                            for (slot, term) in [d, d + 3].into_iter().zip(terms) {
                                expected[k][slot] += term;
                                magnitude[k][slot] += term.abs();
                            }
                        }
                    }
                }
            }
        }
    }

    let dt = state[0].dt as f64;
    let impulse = (0..bodies.len()).map(|k| std::array::from_fn(|d| scalars[S_IMPULSE + 6 * k + d] as f64 / dt)).collect();

    Coupling {
        impulse,
        expected,
        magnitude,
        listed: indirect[3 * Slots::BODIES as usize],
        swept: indirect[3 * Slots::CELLS as usize],
        tiles: tiles.len(),
    }
}

#[test]
fn compact_body_tiles_match_full_grid_coupling() {
    let (device, queue) = gpu();
    let sim = floating_bodies();

    for sparse in [false, true] {
        for subgroups in [false, true] {
            let run = couple_once(&device, &queue, &sim, sparse, subgroups);

            println!(
                "sparse {sparse} subgroups {subgroups}: {} of {} tiles listed, {} hold coupled cells, impulse {:?} expected {:?}",
                run.listed, run.swept, run.tiles, run.impulse, run.expected
            );

            assert_eq!(run.listed as usize, run.tiles, "listed tiles must be exactly the tiles holding coupled cells");
            assert!(run.listed > 0 && run.listed < run.swept, "{} of {} tiles listed", run.listed, run.swept);

            for k in 0..2 {
                assert!(run.expected[k][1] > 0.0, "body {k} feels no buoyancy");

                for d in 0..6 {
                    let gap = (run.impulse[k][d] - run.expected[k][d]).abs();
                    assert!(gap <= 2e-5 * run.magnitude[k][d], "body {k} component {d}: gpu {} cpu {} (scale {})", run.impulse[k][d], run.expected[k][d], run.magnitude[k][d]);
                }
            }

            assert_eq!(run.impulse[2], [0.0; 6], "kinematic body must not be coupled");
            assert_eq!(run.expected[2], [0.0; 6]);
        }
    }
}
