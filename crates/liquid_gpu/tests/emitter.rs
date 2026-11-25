use std::collections::HashMap;

use bevy::math::{IVec3, Quat, UVec3, Vec3};
use bevy::transform::components::{GlobalTransform, Transform};
use liquid_gpu::emitter::{GpuEmissionState, LiquidEmitter, particle_volume};
use liquid_gpu::gas::{GasRegion, GasShape};
use liquid_gpu::layout::{GpuBody, GpuParticle, GpuState, S_VOLUME, S_VOLUME_TARGET, SCALAR_COUNT};
use liquid_gpu::plugin::{LiquidBody, LiquidShape, body_to_gpu, seed_box};
use liquid_gpu::solver::{Bind, GpuSolver, Quality, SolverConfig};

#[cfg(windows)]
mod common;

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

const FRAME: f32 = 1.0 / 60.0;

const DX: f32 = 1.0 / 32.0;

const WARMTH: f32 = 311.0;

const BAND: f32 = 2.0;

const BAND_TOLERANCE: f32 = 0.1;

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
        label: Some("emitter test"),
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

fn config(quality: Quality) -> SolverConfig {
    SolverConfig { dims: UVec3::new(24, 32, 24), dx: DX, max_particles: 120_000, ..SolverConfig::default() }.with_quality(quality)
}

fn pool(config: &SolverConfig) -> Vec<GpuParticle> {
    seed_box(config, Vec3::ZERO, Vec3::new(0.4, 0.3, 0.75))
}

fn blob() -> LiquidEmitter {
    LiquidEmitter { velocity: Vec3::new(0.3, -1.0, 0.2), ..LiquidEmitter::new(GasRegion::sphere(Vec3::new(0.37, 0.71, 0.36), 0.171), WARMTH) }
}

fn slab() -> LiquidEmitter {
    LiquidEmitter::new(GasRegion::cuboid(Vec3::new(0.33, 0.21, 0.4), Vec3::new(0.21, 0.083, 0.17)), WARMTH)
}

fn position(p: &GpuParticle) -> Vec3 {
    Vec3::new(p.x[0], p.x[1], p.x[2])
}

fn emitted(p: &GpuParticle) -> bool {
    p.x[3] < 0.0
}

fn by_id(particles: &[GpuParticle]) -> Vec<GpuParticle> {
    let mut kept: Vec<GpuParticle> = particles.iter().copied().filter(|p| !emitted(p)).collect();
    kept.sort_by(|a, b| a.x[3].total_cmp(&b.x[3]));

    kept
}

fn bits(particles: &[GpuParticle]) -> &[u8] {
    bytemuck::cast_slice(particles)
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    solver: GpuSolver,
}

impl Scene {

    fn new(config: SolverConfig, particles: &[GpuParticle]) -> Self {
        let (device, queue) = gpu();
        let mut solver = GpuSolver::new(&device, config, None);
        solver.initialize(&queue);
        solver.set_frame_dt(&queue, FRAME);
        solver.upload_particles(&queue, particles);

        Self { device, queue, solver }
    }

    fn emit(&mut self, emitters: &[LiquidEmitter]) {
        self.solver.set_emitters(&self.queue, emitters);
    }

    fn frames(&self, count: usize) {
        for _ in 0..count {
            let mut encoder = self.device.create_command_encoder(&Default::default());
            self.solver.encode_frame(&mut encoder, None);
            self.queue.submit([encoder.finish()]);
        }
    }

    fn sort(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());

        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            self.solver.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
            self.solver.encode_emitting_sort(&mut pass);
        }

        self.queue.submit([encoder.finish()]);
    }

    fn state(&self) -> GpuState {
        read::<GpuState>(&self.device, &self.queue, &self.solver.buffers.state, 1)[0]
    }

    fn pace(&self, dt: f32) {
        let state = GpuState { dt, ..self.state() };
        self.queue.write_buffer(&self.solver.buffers.state, 0, bytemuck::bytes_of(&state));
    }

    fn emission(&self) -> GpuEmissionState {
        read::<GpuEmissionState>(&self.device, &self.queue, self.solver.emission_state(), 1)[0]
    }

    fn volume(&self) -> (f32, f32) {
        let scalars: Vec<f32> = read(&self.device, &self.queue, &self.solver.buffers.scalars, SCALAR_COUNT);

        (scalars[S_VOLUME], scalars[S_VOLUME_TARGET])
    }

    fn stored(&self) -> Vec<GpuParticle> {
        let count = self.state().particle_count as usize;

        read(&self.device, &self.queue, &self.solver.buffers.particles, count)
    }

    fn cell(&self, x: Vec3) -> IVec3 {
        let top = self.solver.config.dims.as_ivec3() - 1;

        (x * self.solver.params.inv_dx).floor().as_ivec3().clamp(IVec3::ZERO, top)
    }

    fn occupancy(&self, particles: &[GpuParticle]) -> HashMap<IVec3, usize> {
        let mut cells = HashMap::new();

        for p in particles {
            *cells.entry(self.cell(position(p))).or_default() += 1;
        }

        cells
    }

    fn region_cells(&self, emitter: &LiquidEmitter) -> Vec<IVec3> {
        let c = &self.solver.config;
        let mut cells = Vec::new();

        for z in 0..c.dims.z {
            for y in 0..c.dims.y {
                for x in 0..c.dims.x {
                    let cell = UVec3::new(x, y, z);
                    let p = (cell.as_vec3() + 0.5) * c.dx - emitter.region.center;
                    let inside = match emitter.region.shape {
                        GasShape::Sphere(r) => {
                            let gap = p.dot(p) - r * r;
                            assert!(gap.abs() > 1e-5, "cell {cell} sits on the region surface, move the region");
                            gap <= 0.0
                        }
                        GasShape::Cuboid(half) => {
                            let gap = p.abs() - half;
                            assert!(gap.abs().min_element() > 1e-5, "cell {cell} sits on the region surface, move the region");
                            gap.max_element() <= 0.0
                        }
                    };

                    if inside {
                        cells.push(cell.as_ivec3());
                    }
                }
            }
        }

        cells
    }

    fn guards(&self) -> Vec<String> {
        self.solver.guard_violations(&self.device, &self.queue)
    }

}

fn sane(scene: &Scene, particles: &[GpuParticle]) -> bool {
    let size = scene.solver.config.dims.as_vec3() * scene.solver.config.dx;

    particles.iter().all(|p| {
        let x = position(p);

        p.x.iter().chain(&p.v).chain(p.c.iter().flatten()).all(|v| v.is_finite())
            && x.cmpge(Vec3::ZERO).all()
            && x.cmple(size).all()
    })
}

#[test]
fn emission_fills_each_empty_cell_of_the_region_once() {
    for quality in [Quality::High, Quality::Game] {
        let mut scene = Scene::new(config(quality), &[]);
        let per_cell = scene.solver.config.particles_per_cell as usize;
        let emitter = blob();
        let cells = scene.region_cells(&emitter);
        scene.emit(&[emitter]);
        scene.sort();

        let stored = scene.stored();
        let filled = scene.occupancy(&stored);

        println!("{}: {} cells in the region, {} particles after one emission", quality.name(), cells.len(), stored.len());

        assert!(cells.len() > 100);
        assert_eq!(stored.len(), cells.len() * per_cell);
        assert_eq!(filled.len(), cells.len());
        assert!(cells.iter().all(|c| filled.get(c) == Some(&per_cell)));
        assert!(stored.iter().all(|p| p.v == [0.3, -1.0, 0.2, 0.0] && p.c == [[0.0, 0.0, 0.0, WARMTH], [0.0; 4], [0.0; 4]] && emitted(p)));

        scene.sort();
        assert_eq!(scene.state().particle_count as usize, stored.len());
        assert!(scene.guards().is_empty());
    }
}

#[test]
fn emission_leaves_existing_particles_bit_identical() {
    for quality in [Quality::High, Quality::Game] {
        let config = config(quality);
        let seed = pool(&config);
        let per_cell = config.particles_per_cell as usize;
        let plain = Scene::new(config.clone(), &seed);
        let mut fed = Scene::new(config, &seed);

        plain.frames(6);
        fed.frames(6);
        assert_eq!(bits(&plain.stored()), bits(&fed.stored()));

        let emitters = [blob(), slab()];
        fed.emit(&emitters);
        plain.sort();
        fed.sort();

        let before = plain.stored();
        let after = fed.stored();
        let taken = plain.occupancy(&before);
        let mut wanted: Vec<IVec3> = emitters.iter().flat_map(|e| fed.region_cells(e)).filter(|c| !taken.contains_key(c)).collect();
        wanted.sort_by_key(|c| c.to_array());
        wanted.dedup();

        let added: Vec<GpuParticle> = after.iter().copied().filter(emitted).collect();
        let filled = fed.occupancy(&added);
        let skipped = emitters.iter().flat_map(|e| fed.region_cells(e)).filter(|c| taken.contains_key(c)).count();

        println!(
            "{}: {} existing particles, {} added in {} empty cells, {skipped} occupied region cells left alone",
            quality.name(),
            before.len(),
            added.len(),
            wanted.len()
        );

        assert!(skipped > 50);
        assert_eq!(after.len(), before.len() + wanted.len() * per_cell);
        assert_eq!(bits(&by_id(&after)), bits(&by_id(&before)));
        assert_eq!(filled.len(), wanted.len());
        assert!(wanted.iter().all(|c| filled.get(c) == Some(&per_cell)));
    }
}

#[test]
fn drop_in_the_air_moves_the_pool_only_through_the_shared_solve() {
    for quality in [Quality::High, Quality::Game] {
        let config = config(quality);
        let seed = pool(&config);
        let plain = Scene::new(config.clone(), &seed);
        let mut fed = Scene::new(config, &seed);
        let mut drifts = Vec::new();

        fed.emit(&[blob()]);

        for frame in 0..5 {
            plain.frames(1);
            fed.frames(1);
            fed.emit(&[]);

            let (alone, joined) = (by_id(&plain.stored()), by_id(&fed.stored()));
            let lowest = fed.stored().iter().filter(|p| emitted(p)).map(|p| p.x[1]).fold(f32::MAX, f32::min);

            assert_eq!(alone.len(), joined.len());
            assert!(lowest > 0.4, "the drop reached the pool in frame {frame}");
            drifts.push(alone.iter().zip(&joined).map(|(a, b)| position(a).distance(position(b))).fold(0.0, f32::max) / DX);
        }

        println!("{}: largest pool particle offset against a run without the drop, in cells per frame {drifts:?}", quality.name());

        assert!(drifts[0] < 5e-3);
        assert!(drifts.iter().all(|d| *d < 0.5));
    }
}

#[test]
fn emission_stops_at_capacity() {
    for (quality, room) in [(Quality::High, 1000), (Quality::Game, 1000), (Quality::High, 0)] {
        let base = config(quality);
        let seed = pool(&base);
        let capacity = seed.len() as u32 + room;
        let mut scene = Scene::new(SolverConfig { max_particles: capacity, ..base }, &seed);
        let wanted = scene.region_cells(&blob()).len() * scene.solver.config.particles_per_cell as usize;
        scene.emit(&[blob()]);
        scene.sort();

        let after = scene.stored();

        println!(
            "{} with room for {room}: region wants {wanted}, {} of {capacity} slots used after emission",
            quality.name(),
            after.len()
        );

        assert!(wanted > room as usize);
        assert_eq!(after.len(), capacity as usize);
        assert_eq!(bits(&by_id(&after)), bits(&by_id(&seed)));
        assert!(scene.guards().is_empty());

        for _ in 0..6 {
            scene.frames(1);

            let stored = scene.stored();

            assert!(stored.len() <= capacity as usize);
            assert!(sane(&scene, &stored));
        }

        assert!(scene.guards().is_empty());
    }
}

fn plinth() -> (LiquidBody, Transform) {
    let body = LiquidBody { shape: LiquidShape::Cuboid(Vec3::new(0.12, 0.13, 0.14)), density: 1000.0, dynamic: false };

    (body, Transform::from_xyz(0.41, 0.7, 0.36).with_rotation(Quat::from_rotation_z(0.4)))
}

fn plinth_gpu() -> GpuBody {
    let (body, transform) = plinth();

    body_to_gpu(&body, &GlobalTransform::from(transform), Vec3::ZERO, Vec3::ZERO, Vec3::ZERO)
}

fn plinth_distance(x: Vec3) -> f32 {
    let (body, transform) = plinth();
    let LiquidShape::Cuboid(half) = body.shape else {
        return f32::MAX;
    };
    let q = (transform.rotation.inverse() * (x - transform.translation)).abs() - half;

    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

fn inner() -> LiquidEmitter {
    LiquidEmitter::new(GasRegion::sphere(Vec3::new(0.41, 0.7, 0.36), 0.07), WARMTH)
}

#[test]
fn nothing_is_emitted_inside_a_body() {
    for quality in [Quality::High, Quality::Game] {
        let config = config(quality);
        let margin = config.margin * config.dx;
        let mut scene = Scene::new(config, &[]);
        scene.solver.upload_bodies(&scene.queue, &[plinth_gpu()]);

        let full = scene.region_cells(&blob()).len() * scene.solver.config.particles_per_cell as usize;
        scene.emit(&[blob()]);
        scene.sort();

        let stored = scene.stored();
        let nearest = stored.iter().map(|p| plinth_distance(position(p))).fold(f32::MAX, f32::min);

        println!(
            "{}: {} of {full} region particles emitted around the body, nearest at {:.4} cells, margin {:.4} cells",
            quality.name(),
            stored.len(),
            nearest / DX,
            margin / DX
        );

        assert!(stored.len() > full / 10 && stored.len() < full * 9 / 10);
        assert!(nearest >= margin - 1e-6);

        let mut buried = Scene::new(scene.solver.config.clone(), &[]);
        buried.solver.upload_bodies(&buried.queue, &[plinth_gpu()]);
        buried.emit(&[inner()]);
        buried.sort();

        assert!(buried.region_cells(&inner()).len() > 10);
        assert_eq!(buried.state().particle_count, 0);
    }
}

fn ball_distance(x: Vec3) -> f32 {
    x.distance(Vec3::new(0.4, 0.66, 0.38)) - 0.12
}

fn node_distances(config: &SolverConfig) -> Vec<f32> {
    let nodes = config.dims + 1;

    (0..nodes.element_product())
        .map(|n| UVec3::new(n % nodes.x, (n / nodes.x) % nodes.y, n / (nodes.x * nodes.y)))
        .map(|p| ball_distance(p.as_vec3() * config.dx))
        .collect()
}

fn sampled(config: &SolverConfig, distances: &[f32], x: Vec3) -> f32 {
    let nodes = config.dims + 1;
    let g = x / config.dx;
    let c = g.floor().as_uvec3().min(config.dims - 1);
    let t = g - c.as_vec3();

    (0..8)
        .map(|k| {
            let o = UVec3::new(k & 1, (k >> 1) & 1, k >> 2);
            let w = Vec3::select(o.cmpeq(UVec3::ONE), t, 1.0 - t);
            let p = c + o;

            w.element_product() * distances[(p.x + nodes.x * (p.y + nodes.y * p.z)) as usize]
        })
        .sum()
}

#[test]
fn nothing_is_emitted_inside_a_static_solid() {
    for quality in [Quality::High, Quality::Game] {
        let config = SolverConfig { static_solid: true, ..config(quality) };
        let margin = config.margin * config.dx;
        let distances = node_distances(&config);
        let mut scene = Scene::new(config.clone(), &[]);
        scene.solver.upload_static_solid(&scene.queue, &distances);

        let full = scene.region_cells(&blob()).len() * config.particles_per_cell as usize;
        scene.emit(&[blob()]);
        scene.sort();

        let stored = scene.stored();
        let nearest = stored.iter().map(|p| sampled(&config, &distances, position(p))).fold(f32::MAX, f32::min);

        println!(
            "{}: {} of {full} region particles emitted around the static solid, nearest at {:.4} cells, margin {:.4} cells",
            quality.name(),
            stored.len(),
            nearest / DX,
            margin / DX
        );

        assert!(stored.len() > full / 10 && stored.len() < full * 9 / 10);
        assert!(nearest >= margin - 1e-5 * DX);
        assert!(stored.iter().all(|p| ball_distance(position(p)) > -0.1 * DX));
    }
}

#[test]
fn idle_emitters_leave_the_run_bit_identical() {
    for quality in [Quality::High, Quality::Game] {
        let config = config(quality);
        let seed = pool(&config);
        let plain = Scene::new(config.clone(), &seed);
        let mut cleared = Scene::new(config.clone(), &seed);
        let mut buried = Scene::new(config, &seed);

        cleared.emit(&[blob()]);
        cleared.emit(&[]);
        buried.emit(&[inner()]);


        let mut scenes = [plain, cleared, buried];

        for scene in &mut scenes {
            scene.solver.upload_bodies(&scene.queue, &[plinth_gpu()]);
            scene.frames(30);
        }

        let [plain, cleared, buried] = scenes.map(|s| (s.state(), s.stored()));

        println!("{}: {} particles at time {} after 30 frames", quality.name(), plain.1.len(), plain.0.time);

        assert_eq!(plain.1.len(), seed.len());
        assert_eq!(plain.0, cleared.0);
        assert_eq!(bits(&plain.1), bits(&cleared.1));
        assert_eq!(plain.0, buried.0);
        assert_eq!(bits(&plain.1), bits(&buried.1));
    }
}

#[test]
fn repeated_drops_accumulate() {
    for quality in [Quality::High, Quality::Game] {
        let mut scene = Scene::new(config(quality), &[]);
        let cells = scene.region_cells(&blob()).len() * scene.solver.config.particles_per_cell as usize;
        let mut counts = Vec::new();

        for _ in 0..3 {
            scene.emit(&[blob()]);
            scene.frames(1);
            scene.emit(&[]);
            scene.frames(45);
            counts.push(scene.stored().len());
        }

        println!("{}: a full region holds {cells} particles, counts after each drop {counts:?}", quality.name());

        assert!(counts[0] >= cells);
        assert!(counts[1] >= counts[0] + cells * 9 / 10);
        assert!(counts[2] >= counts[1] + cells * 9 / 10);
        assert!(sane(&scene, &scene.stored()));
        assert!(scene.guards().is_empty());
    }
}

#[test]
fn held_emitter_keeps_pouring() {
    for quality in [Quality::High, Quality::Game] {
        let mut scene = Scene::new(config(quality), &[]);
        let cells = scene.region_cells(&blob()).len() * scene.solver.config.particles_per_cell as usize;
        scene.emit(&[blob()]);

        let counts: Vec<usize> = (0..6)
            .map(|_| {
                scene.frames(10);
                scene.stored().len()
            })
            .collect();

        println!("{}: a full region holds {cells} particles, counts every 10 frames while held {counts:?}", quality.name());

        assert!(counts.windows(2).all(|w| w[1] > w[0]));
        assert!(counts[5] > 3 * cells);
        assert!(sane(&scene, &scene.stored()));
        assert!(scene.guards().is_empty());
    }
}

#[test]
fn metered_emitter_follows_the_flow_rate() {
    for quality in [Quality::High, Quality::Game] {
        let mut scene = Scene::new(config(quality), &[]);
        let per_cell = scene.solver.config.particles_per_cell as f32;
        let volume = particle_volume(&scene.solver.config);
        let flow = 2.37 * per_cell * volume / FRAME;
        let mut samples = vec![(0.0f32, 0u32)];
        scene.emit(&[LiquidEmitter { flow: Some(flow), ..blob() }]);

        for _ in 0..40 {
            scene.frames(1);

            let state = scene.state();
            let spent = scene.emission().slots[0];

            assert_eq!(state.particle_count, spent.particles);
            assert!(spent.budget >= 0.0 && spent.budget < per_cell);
            assert!((flow * state.time / volume - spent.particles as f32 - spent.budget).abs() < 0.01);
            samples.push((state.time, spent.particles));
        }

        let off = |a: &(f32, u32), b: &(f32, u32)| (b.1 as f32 - a.1 as f32 - flow * (b.0 - a.0) / volume).abs();
        let worst = samples.iter().flat_map(|a| samples.iter().map(move |b| off(a, b))).fold(0.0, f32::max);
        let (time, count) = samples[samples.len() - 1];

        println!(
            "{}: {flow:.6} m^3/s asks for {:.2} particles in {time:.4} s, {count} emitted, worst window of the 40 frames is off by {worst:.3} particles, a cell holds {per_cell}",
            quality.name(),
            flow * time / volume
        );

        assert!(count > 80 * per_cell as u32);
        assert!(worst < per_cell);
        assert_eq!(scene.emission().total, count);
        assert!(sane(&scene, &scene.stored()));
        assert!(scene.guards().is_empty());
    }
}

#[test]
fn amount_is_emitted_once() {
    for quality in [Quality::High, Quality::Game] {
        let mut scene = Scene::new(config(quality), &[]);
        let per_cell = scene.solver.config.particles_per_cell as usize;
        let volume = particle_volume(&scene.solver.config);
        let cells = scene.region_cells(&blob()).len();
        let amount = (0.4 * cells as f32 + 0.3) * per_cell as f32 * volume;
        let wanted = (amount / volume).round() as usize;
        let drop = LiquidEmitter { amount: Some(amount), ..blob() };
        let short = |count: usize, wanted: usize| count <= wanted && wanted - count < per_cell;

        scene.emit(&[drop]);
        scene.frames(1);

        let first = scene.stored().len();

        for _ in 0..30 {
            scene.emit(&[drop]);
            scene.frames(1);
        }

        let idle = scene.stored().len();
        let nearest = scene.stored().iter().map(|p| position(p).distance(drop.region.center)).fold(f32::MAX, f32::min);

        scene.emit(&[LiquidEmitter { amount: Some(2.0 * amount), ..drop }]);
        scene.frames(20);

        let raised = scene.stored().len();

        scene.emit(&[]);
        scene.frames(1);
        scene.emit(&[LiquidEmitter { amount: Some(1.5 * cells as f32 * per_cell as f32 * volume), ..drop }]);
        scene.frames(1);

        let partial = scene.stored().len() - raised;

        scene.frames(40);

        let large = scene.stored().len() - raised;

        println!(
            "{}: {wanted} particles asked from a region of {} particles, {first} after one frame, {idle} after 30 more with the nearest {:.1} cells from the centre, {raised} after the amount doubled, a new emitter with 1.5 regions gave {partial} in one frame and {large} in total",
            quality.name(),
            cells * per_cell,
            nearest / scene.solver.config.dx
        );

        assert!(short(first, wanted));
        assert_eq!(idle, first);
        assert!(nearest > 0.2);
        assert!(short(raised, 2 * wanted));
        assert!(partial >= cells * per_cell && partial < large);
        assert!(short(large, 3 * cells * per_cell / 2));
        assert_eq!(scene.emission().slots[0].particles as usize, large);
        assert_eq!(scene.emission().total as usize, raised + large);
        assert!(scene.guards().is_empty());
    }
}

fn keys(particles: &[GpuParticle]) -> Vec<[u32; 16]> {
    let mut keys: Vec<[u32; 16]> = particles
        .iter()
        .filter(|p| emitted(p))
        .map(|p| {
            let mut key: [u32; 16] = bytemuck::cast([p.x, p.v, p.c[0], p.c[1]]);
            key[3] = 0;
            key
        })
        .collect();
    keys.sort_unstable();

    keys
}

#[test]
fn blocked_metered_emitter_emits_less_and_never_overfills() {
    for quality in [Quality::High, Quality::Game] {
        let config = config(quality);
        let seed = pool(&config);
        let per_cell = config.particles_per_cell as f32;
        let flow = 40.3 * per_cell * particle_volume(&config) / FRAME;
        let pass = (40.3 * per_cell) as u32;
        let mut full = Scene::new(config.clone(), &seed);
        let mut fed = Scene::new(config, &seed);
        let open = [blob(), slab()];

        for scene in [&mut full, &mut fed] {
            scene.solver.upload_bodies(&scene.queue, &[plinth_gpu()]);
        }

        full.emit(&open);
        full.sort();
        fed.emit(&open.map(|e| LiquidEmitter { flow: Some(flow), ..e }));

        let limit = keys(&full.stored());
        let mut counts = Vec::new();

        for _ in 0..12 {
            fed.pace(FRAME);
            fed.sort();

            let stored = fed.stored();
            let spent = fed.emission();
            let added = keys(&stored);

            assert_eq!(bits(&by_id(&stored)), bits(&by_id(&seed)));
            assert!(added.iter().all(|k| limit.binary_search(k).is_ok()));
            assert!(added.windows(2).all(|w| w[0] != w[1]));
            assert!(spent.slots[..2].iter().all(|s| s.budget >= 0.0 && s.budget <= per_cell));
            counts.push([spent.slots[0].particles, spent.slots[1].particles]);
        }

        let asked = 12.0 * 40.3 * per_cell;

        println!(
            "{}: keep full gives {} particles around the body and the pool, the metered pair asked for {asked} each and emitted per pass {counts:?}",
            quality.name(),
            limit.len()
        );

        assert!(counts[0].iter().all(|c| *c <= pass && pass - c < per_cell as u32));
        assert!(counts.windows(2).all(|w| w[1][0] >= w[0][0] && w[1][1] >= w[0][1]));
        assert_eq!(counts[10], counts[11]);
        assert!(counts[11].iter().all(|c| (*c as f32) < 0.5 * asked));
        assert_eq!(keys(&fed.stored()), limit);
        assert!(fed.guards().is_empty());
    }
}

fn band_config(quality: Quality, sparse: bool) -> SolverConfig {
    SolverConfig { narrow_band: Some(BAND), sparse, ..config(quality) }
}

fn lake(config: &SolverConfig) -> Vec<GpuParticle> {
    let size = config.dims.as_vec3() * config.dx;

    seed_box(config, Vec3::ZERO, Vec3::new(size.x, 0.3, size.z))
}

fn bulb() -> LiquidEmitter {
    LiquidEmitter::new(GasRegion::sphere(Vec3::new(0.37, 0.67, 0.36), 0.243), WARMTH)
}

fn nozzle() -> LiquidEmitter {
    LiquidEmitter { velocity: Vec3::new(0.0, -1.5, 0.0), ..LiquidEmitter::new(GasRegion::sphere(Vec3::new(0.37, 0.61, 0.36), 0.11), WARMTH) }
}

fn sunk() -> LiquidEmitter {
    LiquidEmitter::new(GasRegion::cuboid(Vec3::new(0.37, 0.085, 0.36), Vec3::new(0.15, 0.04, 0.15)), WARMTH)
}

fn alive(p: &GpuParticle) -> bool {
    p.v[3] < 0.5
}

#[test]
fn drop_joins_the_narrow_band() {
    for (quality, sparse) in [(Quality::High, true), (Quality::Game, true), (Quality::High, false)] {
        let config = band_config(quality, sparse);
        let seed = lake(&config);
        let (dx, per_cell, volume) = (config.dx, config.particles_per_cell, particle_volume(&config));
        let plain = Scene::new(config.clone(), &seed);
        let mut fed = Scene::new(config, &seed);

        plain.frames(20);
        fed.frames(20);
        assert_eq!(bits(&plain.stored()), bits(&fed.stored()));

        let cells = fed.region_cells(&bulb()).len() as u32;
        let drop = LiquidEmitter { amount: Some((cells * per_cell) as f32 * volume), ..bulb() };
        let (_, target) = fed.volume();
        fed.emit(&[drop]);
        plain.frames(2);
        fed.frames(2);

        let spent = fed.emission().slots[0].particles;
        let added = spent as f32 * volume;
        let stored = fed.stored();
        let aloft: Vec<&GpuParticle> = stored.iter().filter(|p| alive(p) && p.x[1] > 0.4).collect();
        let core = aloft.iter().filter(|p| position(p).distance(drop.region.center) < 0.243 - (BAND + 2.5) * dx).count();
        let (early, raised) = fed.volume();
        let (control, _) = plain.volume();
        let early = (early - control) / added;

        if !sparse {
            let centre = fed.cell(drop.region.center).as_uvec3();
            let d = fed.solver.config.dims;
            let phi: Vec<f32> = read(&fed.device, &fed.queue, &fed.solver.buffers.band_phi, d.element_product() as usize);
            let depth = -phi[(centre.x + d.x * (centre.y + d.y * centre.z)) as usize] / dx;

            println!("{} dense: the band level set is {depth:.2} cells deep at the centre of the drop", quality.name());
            assert!(depth > BAND + 1.0);
        }

        plain.frames(240);
        fed.frames(240);

        let (late, held) = fed.volume();
        let (control, _) = plain.volume();
        let late = (late - control) / added;

        plain.frames(120);
        fed.frames(120);

        let (last, _) = fed.volume();
        let (control, _) = plain.volume();
        let last = (last - control) / added;

        println!(
            "{} sparse {sparse}: {spent} particles emitted, {} alive in the air two frames later and {core} of them in the core, target +{:.6} m^3 for {added:.6} m^3 emitted, measured gain over the run without the drop {early:.3} after 2 frames, {late:.3} after 4 s, {last:.3} after 6 s",
            quality.name(),
            aloft.len(),
            raised - target
        );

        assert_eq!(spent, cells * per_cell);
        assert!(!aloft.is_empty() && aloft.len() < spent as usize * 9 / 10);
        assert_eq!(core, 0);
        assert!(((raised - target) / added - 1.0).abs() < 1e-3);
        assert_eq!(held, raised);
        assert!((late - 1.0).abs() < BAND_TOLERANCE);
        assert!((last - 1.0).abs() < BAND_TOLERANCE);
        assert_eq!(fed.emission().total, spent);
        assert!(sane(&fed, &fed.stored()));
        assert!(fed.guards().is_empty());
    }
}

#[test]
fn metered_stream_feeds_a_narrow_band() {
    for (quality, sparse) in [(Quality::High, true), (Quality::Game, true), (Quality::High, false)] {
        let config = band_config(quality, sparse);
        let seed = lake(&config);
        let (per_cell, volume) = (config.particles_per_cell as f32, particle_volume(&config));
        let flow = 6.3 * per_cell * volume / FRAME;
        let plain = Scene::new(config.clone(), &seed);
        let mut fed = Scene::new(config, &seed);

        plain.frames(20);
        fed.frames(20);

        let (_, target) = fed.volume();
        let start = fed.state().time;
        fed.emit(&[LiquidEmitter { flow: Some(flow), ..nozzle() }]);
        plain.frames(60);
        fed.frames(60);

        let poured = fed.emission().slots[0].particles;
        let wanted = flow * (fed.state().time - start) / volume;
        let added = poured as f32 * volume;

        fed.emit(&[]);
        plain.frames(180);
        fed.frames(180);

        let (measured, raised) = fed.volume();
        let (control, _) = plain.volume();
        let gain = (measured - control) / added;

        println!(
            "{} sparse {sparse}: {flow:.6} m^3/s for 1 s asks for {wanted:.1} particles, {poured} emitted, target +{:.6} m^3 for {added:.6} m^3 emitted, measured gain over the run without the stream {gain:.3} three seconds later",
            quality.name(),
            raised - target
        );

        assert!(wanted - poured as f32 > -0.01 && wanted - (poured as f32) < per_cell);
        assert!(((raised - target) / added - 1.0).abs() < 1e-3);
        assert!((gain - 1.0).abs() < BAND_TOLERANCE);
        assert_eq!(fed.emission().total, poured);
        assert!(sane(&fed, &fed.stored()));
        assert!(fed.guards().is_empty());
    }
}

#[test]
fn submerged_emitter_adds_nothing_to_a_narrow_band() {
    for quality in [Quality::High, Quality::Game] {
        let config = band_config(quality, true);
        let seed = lake(&config);
        let plain = Scene::new(config.clone(), &seed);
        let mut fed = Scene::new(config, &seed);

        plain.frames(10);
        fed.frames(10);

        let stored = fed.stored();
        let live: Vec<GpuParticle> = stored.iter().copied().filter(alive).collect();
        let held = fed.occupancy(&live);
        let cells = fed.region_cells(&sunk());
        let bare = cells.iter().filter(|c| !held.contains_key(c)).count();

        fed.emit(&[sunk()]);
        plain.frames(20);
        fed.frames(20);

        println!(
            "{}: {bare} of {} region cells hold no particle under the band, {} particles emitted in 20 frames",
            quality.name(),
            cells.len(),
            fed.emission().total
        );

        assert!(bare > 50 && bare == cells.len());
        assert_eq!(fed.emission().total, 0);
        assert_eq!(plain.state(), fed.state());
        assert_eq!(bits(&plain.stored()), bits(&fed.stored()));
    }
}

#[cfg(windows)]
#[test]
fn emitter_pipelines_compile_under_fxc() {
    common::assert_compiles_under_fxc(|device, queue| {
        for quality in [Quality::High, Quality::Game] {
            for static_solid in [false, true] {
                for sparse in [false, true] {
                    for narrow_band in [None, Some(BAND)] {
                        let mut solver = GpuSolver::new(device, SolverConfig { static_solid, sparse, narrow_band, ..config(quality) }, None);
                        solver.set_emitters(queue, &[blob(), LiquidEmitter { flow: Some(1e-3), amount: Some(1e-2), ..slab() }]);
                    }
                }
            }
        }
    });
}

#[test]
fn thrown_drop_keeps_its_velocity_in_a_narrow_band() {
    for quality in [Quality::High, Quality::Game] {
        let config = band_config(quality, true);
        let mut scene = Scene::new(config.clone(), &lake(&config));
        scene.frames(5);
        let throw = Vec3::new(0.6, 0.0, 0.3);
        scene.emit(&[LiquidEmitter { velocity: throw, amount: Some(1.0), ..bulb() }]);
        scene.frames(1);
        scene.emit(&[]);

        let mut speeds = Vec::new();

        for _ in 0..3 {
            let stored = scene.stored();
            let live: Vec<&GpuParticle> = stored.iter().filter(|p| alive(p) && p.x[1] > 0.4).collect();
            let mean = live.iter().map(|p| Vec3::new(p.v[0], p.v[1], p.v[2])).sum::<Vec3>() / live.len() as f32;

            speeds.push(Vec3::new(mean.x, 0.0, mean.z).dot(throw) / throw.length_squared());
            scene.frames(1);
        }

        println!("{}: share of the thrown horizontal velocity kept by the live particles after 1, 2 and 3 frames {speeds:?}", quality.name());

        assert!(speeds.iter().all(|s| (s - 1.0).abs() < 0.03));
    }
}

#[test]
fn overlapping_emitters_fill_each_cell_once() {
    for quality in [Quality::High, Quality::Game] {
        let mut shared = Scene::new(config(quality), &[]);
        let mut spent = Scene::new(config(quality), &[]);
        let full = shared.region_cells(&blob()).len() * shared.solver.config.particles_per_cell as usize;

        shared.emit(&[blob(), blob()]);
        shared.sort();
        spent.emit(&[LiquidEmitter { amount: Some(0.0), ..blob() }, blob()]);
        spent.sort();

        let (first, second) = (shared.emission(), spent.emission());

        println!(
            "{}: a region of {full} particles shared by two emitters got {} and {}, behind a spent emitter {} and {}",
            quality.name(),
            first.slots[0].particles,
            first.slots[1].particles,
            second.slots[0].particles,
            second.slots[1].particles
        );

        assert_eq!([first.slots[0].particles, first.slots[1].particles], [full as u32, 0]);
        assert_eq!([second.slots[0].particles, second.slots[1].particles], [0, full as u32]);
        assert_eq!(keys(&shared.stored()), keys(&spent.stored()));
    }
}
