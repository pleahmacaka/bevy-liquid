use bevy::math::{UVec3, Vec3};
use liquid_gpu::gas::{GAS_MASS_QUANTUM, GasConfig, GasRegion, GasSolver, GasState, GasTargets};
use liquid_gpu::heat::HeatSource;
use liquid_gpu::layout::{GpuParticle, GpuState};
use liquid_gpu::plugin::seed_box;
use liquid_gpu::solver::{GpuSolver, SolverConfig};

#[cfg(windows)]
mod common;

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

const FRAME: f32 = 1.0 / 60.0;

const DX: f32 = 1.0 / 32.0;

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
        label: Some("heat test"),
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

fn liquid_config() -> SolverConfig {
    SolverConfig {
        dims: UVec3::new(16, 16, 16),
        dx: DX,
        max_particles: 40_000,
        latent_heat: 20.0,
        ..SolverConfig::default()
    }
}

fn pool(config: &SolverConfig) -> Vec<GpuParticle> {
    seed_box(config, Vec3::ZERO, Vec3::new(0.5, 0.25, 0.5))
}

fn temperature(p: &GpuParticle) -> f32 {
    p.c[0][3]
}

fn floor_plate(temperature: f32, conductance: f32) -> HeatSource {
    HeatSource { region: GasRegion::cuboid(Vec3::new(0.25, 0.0, 0.25), Vec3::new(0.3, 0.07, 0.3)), temperature, conductance }
}

struct Scene {
    device: wgpu::Device,
    queue: wgpu::Queue,
    liquid: GpuSolver,
    gas: GasSolver,
    initial: usize,
}

impl Scene {

    fn new(config: SolverConfig, particles: &[GpuParticle], capacity: u32) -> Self {
        let (device, queue) = gpu();
        let gas_config = GasConfig { dims: UVec3::new(16, 24, 16), dx: DX, injection_capacity: capacity, ..GasConfig::default() };
        let gas = GasSolver::new(&device, gas_config, &[], GasTargets::default());
        gas.initialize(&queue);

        let mut liquid = GpuSolver::new(&device, config, None);
        liquid.initialize(&queue);
        liquid.set_frame_dt(&queue, FRAME);
        liquid.upload_particles(&queue, particles);
        liquid.attach_gas(&device, &gas);

        Self { device, queue, liquid, gas, initial: particles.len() }
    }

    fn heat(&mut self, sources: &[HeatSource]) {
        self.liquid.set_heat(&self.queue, sources, Vec3::ZERO);
    }

    fn liquid_frame(&self) {
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.liquid.encode_frame(&mut encoder, None);
        self.queue.submit([encoder.finish()]);
    }

    fn gas_frame(&mut self) {
        self.gas.set_frame(&self.queue, FRAME, &[]);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.gas.encode_frame(&mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    fn run(&mut self, frames: usize) {
        for _ in 0..frames {
            self.liquid_frame();
            self.gas_frame();
        }
    }

    fn liquid_state(&self) -> GpuState {
        read::<GpuState>(&self.device, &self.queue, &self.liquid.buffers.state, 1)[0]
    }

    fn stored(&self) -> Vec<GpuParticle> {
        let count = self.liquid_state().particle_count as usize;

        read(&self.device, &self.queue, &self.liquid.buffers.particles, count)
    }

    fn alive(&self) -> Vec<GpuParticle> {
        self.stored().into_iter().filter(|p| p.v[3] <= 0.5).collect()
    }

    fn removed(&self) -> u64 {
        (self.initial - self.alive().len()) as u64
    }

    fn gas_state(&self) -> GasState {
        read::<GasState>(&self.device, &self.queue, &self.gas.buffers.state, 1)[0]
    }

    fn queued(&self) -> u32 {
        read::<u32>(&self.device, &self.queue, &self.gas.buffers.injections, 1)[0]
    }

    fn mass_units(&self) -> u64 {
        (self.liquid.particle_mass() as f64 / GAS_MASS_QUANTUM) as u64
    }

}

#[test]
fn heated_box_moves_exact_mass_into_gas() {
    let config = liquid_config();
    let mut scene = Scene::new(config.clone(), &pool(&config), 8192);
    scene.heat(&[floor_plate(3000.0, 50.0)]);
    scene.run(40);

    let gas = scene.gas_state();
    let removed = scene.removed();
    let units = scene.mass_units();
    let mass = scene.liquid.particle_mass();

    println!(
        "{removed} of {} particles boiled away, {} accepted, {} rejected, {} quanta injected, {units} quanta per particle of {mass} kg",
        scene.initial,
        gas.accepted,
        gas.rejected,
        gas.injected_units()
    );

    assert!(removed > 500);
    assert_eq!(units as f64 * GAS_MASS_QUANTUM, mass as f64);
    assert_eq!(gas.rejected, 0);
    assert_eq!(gas.accepted as u64, removed);
    assert_eq!(gas.injected_units(), removed * units);
}

#[test]
fn cold_box_injects_nothing_and_cools_to_ambient() {
    let cooling = 2.0;
    let start = 330.0;
    let config = SolverConfig { cooling, ..liquid_config() };
    let mut particles = pool(&config);

    for p in &mut particles {
        p.c[0][3] = start;
    }

    let mut scene = Scene::new(config, &particles, 1024);
    let ambient = scene.gas.config.params.ambient_temperature;
    let far = HeatSource { region: GasRegion::sphere(Vec3::splat(5.0), 0.1), temperature: 3000.0, conductance: 50.0 };
    scene.heat(&[far]);
    scene.run(200);

    let time = scene.liquid_state().time as f64;
    let expected = ambient as f64 + (start - ambient) as f64 * (-cooling as f64 * time).exp();
    let alive = scene.alive();
    let worst = alive.iter().fold(0.0f64, |m, p| m.max((temperature(p) as f64 - expected).abs()));
    let gas = scene.gas_state();

    println!(
        "after {time:.4} s: expected {expected:.4} K (ambient {ambient} K), worst deviation {worst:e} K, {} of {} particles alive, {} accepted",
        alive.len(),
        scene.initial,
        gas.accepted
    );

    assert_eq!(alive.len(), scene.initial);
    assert_eq!(gas.accepted, 0);
    assert_eq!(gas.injected_units(), 0);
    assert_eq!(scene.queued(), 0);
    assert!(worst < 1e-2);
    assert!((expected - ambient as f64) < 0.1);
}

#[test]
fn full_injection_buffer_keeps_particles() {
    let capacity = 64;
    let config = liquid_config();
    let threshold = config.boiling_point + config.latent_heat;
    let mut scene = Scene::new(config.clone(), &pool(&config), capacity);
    let everywhere = HeatSource { region: GasRegion::cuboid(Vec3::splat(0.25), Vec3::splat(1.0)), temperature: 5000.0, conductance: 200.0 };
    scene.heat(&[everywhere]);
    scene.liquid_frame();

    let alive = scene.alive();
    let overheated = alive.iter().filter(|p| temperature(p) > threshold).count();

    println!(
        "one liquid frame with a full queue: {} of {} alive, {overheated} still liquid above {threshold} K, queue counter {}",
        alive.len(),
        scene.initial,
        scene.queued()
    );

    assert_eq!(scene.removed(), capacity as u64);
    assert_eq!(overheated, alive.len());
    assert!(scene.queued() >= capacity);

    scene.gas_frame();

    for frame in 1..=3u64 {
        let gas = scene.gas_state();

        println!("after {frame} gas frames: {} removed, {} accepted, {} quanta", scene.removed(), gas.accepted, gas.injected_units());

        assert_eq!(scene.queued(), 0);
        assert_eq!(scene.removed(), frame * capacity as u64);
        assert_eq!(gas.accepted as u64, frame * capacity as u64);
        assert_eq!(gas.injected_units(), frame * capacity as u64 * scene.mass_units());

        scene.run(1);
    }
}

fn marked(config: &SolverConfig) -> Vec<GpuParticle> {
    let mut particles = seed_box(config, Vec3::ZERO, Vec3::new(0.2, 0.4, 0.5));

    for p in &mut particles {
        p.c[0][3] = marker(p.x[3]);
    }

    particles
}

fn marker(id: f32) -> f32 {
    280.0 + (id as u32 % 997) as f32 * 0.173
}

#[test]
fn temperature_survives_transfer_sort_and_quantized_packing() {
    for quantized in [false, true] {
        let config = SolverConfig { quantized, ..liquid_config() };
        let particles = marked(&config);
        let scene = Scene::new(config, &particles, 64);

        for _ in 0..30 {
            scene.liquid_frame();
        }

        let stored = scene.stored();
        let moved = stored.iter().zip(&particles).filter(|(a, b)| a.x[3] != b.x[3]).count();
        let wrong = stored.iter().filter(|p| temperature(p).to_bits() != marker(p.x[3]).to_bits()).count();
        let mut packed_error = 0.0f32;

        if quantized {
            let words: Vec<[u32; 2]> = read(&scene.device, &scene.queue, &scene.liquid.buffers.packed, stored.len());

            for (w, p) in words.iter().zip(&stored) {
                packed_error = packed_error.max((half(w[1] >> 16) - temperature(p)).abs() / temperature(p));
            }
        }

        println!(
            "quantized {quantized}: {} particles, {moved} reordered by sort, {wrong} with a changed temperature, packed relative error {packed_error:e}",
            stored.len()
        );

        assert_eq!(stored.len(), particles.len());
        assert!(moved > stored.len() / 2);
        assert_eq!(wrong, 0);
        assert!(packed_error <= 2f32.powi(-10));
    }
}

#[cfg(windows)]
#[test]
fn boil_pipelines_compile_under_fxc() {
    common::assert_compiles_under_fxc(|device, _| {
        let gas = GasSolver::new(device, GasConfig { injection_capacity: 64, ..GasConfig::default() }, &[], GasTargets::default());
        let mut liquid = GpuSolver::new(device, liquid_config(), None);
        liquid.attach_gas(device, &gas);
    });
}

fn everywhere(temperature: f32, conductance: f32) -> HeatSource {
    HeatSource { region: GasRegion::cuboid(Vec3::splat(0.25), Vec3::splat(1.0)), temperature, conductance }
}

fn relaxed(h: f64, goal: f32, rate: f32, dt: f64) -> f64 {
    goal as f64 + (h - goal as f64) * (-rate as f64 * dt).exp()
}

fn worst_deviation(device: &wgpu::Device, queue: &wgpu::Queue, liquid: &GpuSolver, seeded: usize, expected: f64) -> f64 {
    let state: GpuState = read(device, queue, &liquid.buffers.state, 1)[0];
    let stored: Vec<GpuParticle> = read(device, queue, &liquid.buffers.particles, seeded);

    assert_eq!(state.particle_count as usize, seeded);
    assert!(stored.iter().all(|p| p.v[3] <= 0.5));

    stored.iter().fold(0.0f64, |m, p| m.max((temperature(p) as f64 - expected).abs()))
}

fn heats_and_cools_without_gas(device: &wgpu::Device, queue: &wgpu::Queue) {
    let cooling = 2.0;
    let ambient = 285.0;
    let start = 300.0;
    let plate = everywhere(350.0, 8.0);
    let config = SolverConfig { cooling, ambient_temperature: ambient, ..liquid_config() };
    let boiling = config.boiling_point;
    let mut particles = pool(&config);

    for p in &mut particles {
        p.c[0][3] = start;
    }

    let seeded = particles.len();
    let mut liquid = GpuSolver::new(device, config, None);
    liquid.initialize(queue);
    liquid.set_frame_dt(queue, FRAME);
    liquid.upload_particles(queue, &particles);

    let run = |liquid: &GpuSolver, frames: usize, sources: &[HeatSource], from: f64| {
        let mut h = from;
        let mut time = read::<GpuState>(device, queue, &liquid.buffers.state, 1)[0].time as f64;

        for _ in 0..frames {
            let mut encoder = device.create_command_encoder(&Default::default());
            liquid.encode_frame(&mut encoder, None);
            queue.submit([encoder.finish()]);

            let state: GpuState = read(device, queue, &liquid.buffers.state, 1)[0];
            let dt = state.time as f64 - time;

            assert_eq!(state.substeps, 1);

            h = sources.iter().fold(h, |h, s| relaxed(h, s.temperature, s.conductance, dt));
            h = relaxed(h, ambient, cooling, dt);
            time = state.time as f64;
        }

        (h, time)
    };

    run(&liquid, 5, &[], start as f64);

    let untouched = worst_deviation(device, queue, &liquid, seeded, start as f64);
    let latched = liquid.heats();

    liquid.set_heat(queue, &[plate], Vec3::ZERO);

    let (warm, warm_time) = run(&liquid, 90, &[plate], start as f64);
    let worst_warm = worst_deviation(device, queue, &liquid, seeded, warm);

    liquid.set_heat(queue, &[], Vec3::ZERO);

    let (_, cool_time) = run(&liquid, 150, &[], warm);
    let cool = ambient as f64 + (warm - ambient as f64) * (-cooling as f64 * (cool_time - warm_time)).exp();
    let worst_cool = worst_deviation(device, queue, &liquid, seeded, cool);

    liquid.set_heat(queue, &[everywhere(5000.0, 200.0)], Vec3::ZERO);
    run(&liquid, 20, &[], 0.0);

    let above_boiling = worst_deviation(device, queue, &liquid, seeded, boiling as f64);

    println!(
        "no gas domain, {seeded} particles: idle deviation {untouched:e} K, warmed {start} K to {warm:.4} K under a {} K source (worst deviation {worst_warm:e} K), cooled to {cool:.4} K over {:.4} s toward ambient {ambient} K (worst deviation {worst_cool:e} K), deviation from the boiling point {boiling} K under a 5000 K source {above_boiling:e} K",
        plate.temperature,
        cool_time - warm_time
    );

    assert!(!latched && liquid.heats());
    assert_eq!(untouched, 0.0);
    assert!(warm > start as f64 + 30.0 && warm < plate.temperature as f64);
    assert!(worst_warm < 1e-2);
    assert!(cool - (ambient as f64) < 1.0);
    assert!(worst_cool < 1e-2);
    assert_eq!(above_boiling, 0.0);
}

#[test]
fn heated_box_without_gas_warms_and_cools_to_ambient() {
    let (device, queue) = gpu();

    heats_and_cools_without_gas(&device, &queue);
}

#[cfg(windows)]
#[test]
fn heat_without_gas_runs_under_fxc() {
    common::assert_compiles_under_fxc(heats_and_cools_without_gas);
}
