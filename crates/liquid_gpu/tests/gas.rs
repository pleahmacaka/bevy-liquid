use bevy::math::{UVec3 as GpuUVec3, Vec3};
use glam::{DVec3, UVec3};
use liquid_gpu::gas::{GAS_MASS_QUANTUM, GasConfig, GasEmitter, GasInjection, GasParams, GasRegion, GasSolver, GasState, GasTargets};
use liquid_gpu::solver::Quality;
use liquid_ref::gas::{Emitter, GasDomain, GasParams as RefParams, Region};
use liquid_ref::grid::Grid;
use liquid_ref::rigid::Shape;

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

const DX: f64 = 1.0 / 16.0;

const FRAME: f64 = 1.0 / 60.0;

const VAPOR: f64 = 373.15;

const TOLERANCE: f32 = 1e-6;

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
        label: Some("gas test"),
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

fn sphere(center: DVec3, radius: f64) -> Region {
    Region {
        shape: Shape::Sphere { radius },
        center,
    }
}

fn region(r: &Region) -> GasRegion {
    let center = Vec3::new(r.center.x as f32, r.center.y as f32, r.center.z as f32);

    match r.shape {
        Shape::Sphere { radius } => GasRegion::sphere(center, radius as f32),
        Shape::Cuboid { half } => GasRegion::cuboid(center, Vec3::new(half.x as f32, half.y as f32, half.z as f32)),
    }
}

fn gpu_params(p: &RefParams, tolerance: f32, iterations: u32) -> GasParams {
    GasParams {
        ambient_temperature: p.ambient_temperature as f32,
        density_buoyancy: p.density_buoyancy as f32,
        thermal_buoyancy: p.thermal_buoyancy as f32,
        vorticity_confinement: p.vorticity_confinement as f32,
        density_decay: p.density_decay as f32,
        cooling: p.cooling as f32,
        air_density: p.air_density as f32,
        cfl: p.cfl as f32,
        max_dt: p.max_dt as f32,
        tolerance,
        max_iterations: iterations,
        maccormack: p.maccormack,
        velocity_maccormack: p.maccormack,
        open_low: p.open_low,
        open_high: p.open_high,
        inflow_at_rest: false,
    }
}

struct Pair {
    device: wgpu::Device,
    queue: wgpu::Queue,
    cpu: GasDomain,
    gpu: GasSolver,
    emitters: Vec<GasEmitter>,
}

fn reference(dims: UVec3, params: RefParams, plume: bool, obstacles: &[Region]) -> GasDomain {
    let mut gas = GasDomain::new(Grid::new(dims, DX), DVec3::ZERO, params);

    if plume {
        let base = DVec3::new(0.5 * dims.x as f64 * DX, 0.25, 0.5 * dims.z as f64 * DX);
        gas.emitters.push(Emitter {
            region: sphere(base, 0.15),
            density_rate: 4.0,
            heat_rate: 400.0,
        });
    }

    gas.obstacles.extend_from_slice(obstacles);

    gas
}

fn pair(cpu: GasDomain, tolerance: f32, iterations: u32, capacity: u32) -> Pair {
    let (device, queue) = gpu();
    let config = GasConfig {
        dims: GpuUVec3::new(cpu.grid.dims.x, cpu.grid.dims.y, cpu.grid.dims.z),
        dx: DX as f32,
        params: gpu_params(&cpu.params, tolerance, iterations),
        max_substeps: if tolerance > 0.0 { 8 } else { 1 },
        smoothing: 2,
        injection_capacity: capacity,
        warm_start: true,
        quality: Quality::High,
    };
    let obstacles: Vec<GasRegion> = cpu.obstacles.iter().map(region).collect();
    let emitters = cpu
        .emitters
        .iter()
        .map(|e| GasEmitter {
            region: region(&e.region),
            density_rate: e.density_rate as f32,
            heat_rate: e.heat_rate as f32,
        })
        .collect();
    let gpu = GasSolver::new(&device, config, &obstacles, GasTargets::default());
    gpu.initialize(&queue);

    Pair { device, queue, cpu, gpu, emitters }
}

impl Pair {

    fn upload_reference(&self) {
        let to_f32 = |v: &[f64]| v.iter().map(|x| *x as f32).collect::<Vec<f32>>();
        let velocity: Vec<f32> = self.cpu.velocity.axes.iter().flat_map(|a| to_f32(&a.data)).collect();

        self.gpu.upload(&self.queue, &to_f32(&self.cpu.density.data), &to_f32(&self.cpu.temperature.data), &velocity);
    }

    fn advance(&mut self, dt: f64) {
        self.gpu.set_frame(&self.queue, dt as f32, &self.emitters);

        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.gpu.encode_frame(&mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    fn run(&mut self, frames: usize) {
        for _ in 0..frames {
            self.advance(FRAME);
        }
    }

    fn density(&self) -> Vec<f32> {
        read(&self.device, &self.queue, &self.gpu.buffers.density, self.gpu.config.cell_count() as usize)
    }

    fn temperature(&self) -> Vec<f32> {
        read(&self.device, &self.queue, &self.gpu.buffers.temperature, self.gpu.config.cell_count() as usize)
    }

    fn velocity(&self) -> Vec<f32> {
        read(&self.device, &self.queue, &self.gpu.buffers.velocity, self.gpu.config.face_count() as usize)
    }

    fn state(&self) -> GasState {
        read::<GasState>(&self.device, &self.queue, &self.gpu.buffers.state, 1)[0]
    }

    fn bits(&self) -> Vec<u32> {
        let floats = [self.density(), self.temperature(), self.velocity()].concat();
        let state = self.state();

        floats.iter().map(|v| v.to_bits()).chain([state.time.to_bits(), state.substeps]).collect()
    }

    fn mass(&self) -> f64 {
        self.density().iter().map(|r| *r as f64).sum::<f64>() * DX.powi(3)
    }

    fn report(&self) -> Errors {
        let ambient = self.cpu.params.ambient_temperature;
        let cpu_velocity: Vec<f64> = self.cpu.velocity.axes.iter().flat_map(|a| a.data.iter().copied()).collect();
        let excess: Vec<f64> = self.cpu.temperature.data.iter().map(|t| t - ambient).collect();
        let gpu_excess: Vec<f32> = self.temperature().iter().map(|t| (*t as f64 - ambient) as f32).collect();

        Errors {
            velocity: relative_error(&self.velocity(), &cpu_velocity),
            density: relative_error(&self.density(), &self.cpu.density.data),
            temperature: relative_error(&gpu_excess, &excess),
        }
    }

}

#[derive(Debug)]
struct Errors {
    velocity: f64,
    density: f64,
    temperature: f64,
}

impl Errors {

    fn max(&self) -> f64 {
        self.velocity.max(self.density).max(self.temperature)
    }

}

fn relative_error(gpu: &[f32], cpu: &[f64]) -> f64 {
    let scale = cpu.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-30);
    let worst = gpu.iter().zip(cpu).fold(0.0f64, |m, (g, c)| m.max((*g as f64 - c).abs()));

    worst / scale
}

fn divergence(pair: &Pair, velocity: &[f32]) -> Vec<f64> {
    let d = pair.cpu.grid.dims;
    let fx = (d.x + 1) * d.y * d.z;
    let fy = d.x * (d.y + 1) * d.z;
    let face = |a: usize, i: u32, j: u32, k: u32| {
        let (base, nx, ny) = match a {
            0 => (0, d.x + 1, d.y),
            1 => (fx, d.x, d.y + 1),
            _ => (fx + fy, d.x, d.y),
        };

        velocity[(base + i + nx * (j + ny * k)) as usize] as f64
    };
    let mut out = Vec::with_capacity((d.x * d.y * d.z) as usize);

    for k in 0..d.z {
        for j in 0..d.y {
            for i in 0..d.x {
                let flux = face(0, i + 1, j, k) - face(0, i, j, k) + face(1, i, j + 1, k) - face(1, i, j, k) + face(2, i, j, k + 1)
                    - face(2, i, j, k);
                out.push(flux / DX);
            }
        }
    }

    out
}

fn developed(dims: UVec3, obstacles: &[Region], frames: usize) -> GasDomain {
    let mut cpu = reference(dims, RefParams::default(), true, obstacles);

    for _ in 0..frames {
        cpu.advance(FRAME);
    }

    cpu
}

#[test]
fn single_substep_matches_reference() {
    let obstacle = sphere(DVec3::new(0.7, 1.0, 0.75), 0.15);
    let mut pair = pair(developed(UVec3::new(24, 40, 24), &[obstacle], 30), TOLERANCE, 200, 0);
    pair.upload_reference();

    let dt = 1.0 / 120.0;
    let cpu_substeps = pair.cpu.advance(dt);
    pair.advance(dt);

    let state = pair.state();
    let errors = pair.report();

    println!(
        "one substep of {dt:.5} s from a developed plume: cpu substeps {cpu_substeps}, gpu substeps {}, gpu pcg {} iterations residual {:e}, cpu pcg {} iterations, errors {errors:?}",
        state.substeps, state.iterations, state.residual, pair.cpu.report.pressure.iterations
    );

    assert_eq!(cpu_substeps, 1);
    assert_eq!(state.substeps, 1);
    assert!(errors.velocity < 2e-6);
    assert!(errors.density < 2e-6);
    assert!(errors.temperature < 2e-6);
}

#[test]
fn few_substeps_match_reference() {
    let obstacle = sphere(DVec3::new(0.7, 1.0, 0.75), 0.15);
    let mut pair = pair(developed(UVec3::new(24, 40, 24), &[obstacle], 30), TOLERANCE, 200, 0);
    pair.upload_reference();

    let mut history = Vec::new();

    for _ in 0..4 {
        let cpu_substeps = pair.cpu.advance(FRAME);
        pair.advance(FRAME);
        let gpu_substeps = pair.state().substeps as usize;
        let errors = pair.report();

        println!("frame: cpu substeps {cpu_substeps}, gpu substeps {gpu_substeps}, errors {errors:?}");

        assert_eq!(cpu_substeps, gpu_substeps);
        history.push(errors.max());
    }

    assert!(history.iter().all(|e| *e < 5e-6));
}

#[test]
fn ambient_gas_stays_exactly_at_rest() {
    let mut pair = pair(reference(UVec3::new(16, 24, 16), RefParams::default(), false, &[]), TOLERANCE, 200, 0);
    let ambient = pair.gpu.config.params.ambient_temperature;
    pair.run(120);

    let state = pair.state();

    println!("time {}, substeps in the last frame {}, max |u| {}", state.time, state.substeps, state.max_speed());

    assert_eq!(state.substeps, 1);
    assert!((state.time as f64 - 2.0).abs() < 1e-4);
    assert!(pair.velocity().iter().all(|v| v.to_bits() == 0));
    assert!(pair.density().iter().all(|r| r.to_bits() == 0));
    assert!(pair.temperature().iter().all(|t| t.to_bits() == ambient.to_bits()));
}

#[test]
fn still_gas_cools_and_fades_exponentially() {
    let params = RefParams {
        density_buoyancy: 0.0,
        thermal_buoyancy: 0.0,
        ..RefParams::default()
    };
    let mut pair = pair(reference(UVec3::new(12, 12, 12), params, false, &[]), TOLERANCE, 200, 0);
    let ambient = params.ambient_temperature;

    pair.cpu.density.fill(2.0);
    pair.cpu.temperature.fill(ambient + 50.0);
    pair.upload_reference();

    for _ in 0..90 {
        pair.cpu.advance(FRAME);
    }

    pair.run(90);

    let time = pair.cpu.time;
    let rho = 2.0 * (-params.density_decay * time).exp();
    let excess = 50.0 * (-params.cooling * time).exp();
    let rho_error = pair.density().iter().fold(0.0f64, |m, r| m.max((*r as f64 - rho).abs() / rho));
    let t_error = pair.temperature().iter().fold(0.0f64, |m, t| m.max((*t as f64 - ambient - excess).abs() / excess));
    let errors = pair.report();

    println!(
        "t {time:.4} (gpu {:.6}): density {rho:.9} rel err {rho_error:e}, excess T {excess:.9} rel err {t_error:e}, vs reference {errors:?}",
        pair.state().time
    );

    assert!(pair.velocity().iter().all(|v| v.to_bits() == 0));
    assert!(rho_error < 5e-6);
    assert!(t_error < 1e-5);
}

#[test]
fn projection_leaves_small_divergence() {
    let shelf = Region {
        shape: Shape::Cuboid { half: DVec3::new(0.2, 0.05, 0.2) },
        center: DVec3::new(0.9, 1.0, 0.75),
    };

    for (tolerance, iterations, bound) in [(TOLERANCE, 200, 1e-5), (0.0, 2, 2e-2), (0.0, 6, 1e-5)] {
        let mut pair = pair(reference(UVec3::new(24, 40, 24), RefParams::default(), true, &[shelf]), tolerance, iterations, 0);
        pair.run(40);

        let state = pair.state();
        let solid = pair.cpu.solid();
        let worst = divergence(&pair, &pair.velocity())
            .iter()
            .zip(&solid)
            .filter(|(_, s)| !**s)
            .fold(0.0f64, |m, (d, _)| m.max(d.abs()));
        let input = state.input_divergence() as f64;

        println!(
            "tolerance {tolerance:e}, {iterations} iterations budget: max |div| {worst:e}, before projection {input:e}, ratio {:e}, {} iterations used, residual {:e}",
            worst / input,
            state.iterations,
            state.residual
        );

        assert!(solid.iter().any(|s| *s));
        assert!(input > 1.0);
        assert!(worst <= bound * input);
    }
}

#[test]
fn inject_adds_exact_mass_inside_and_drops_outside() {
    let cpu = reference(UVec3::new(8, 8, 8), RefParams::default(), false, &[sphere(DVec3::splat(0.375), 0.08)]);
    let mut pair = pair(cpu, TOLERANCE, 200, 16);
    let m = 0.0123;
    let units = (m / GAS_MASS_QUANTUM).round() as u64;
    let inside = [
        DVec3::new(0.21, 0.33, 0.27),
        DVec3::new(0.01, 0.49, 0.001),
        DVec3::ZERO,
        DVec3::splat(0.5),
        DVec3::new(0.3, 0.375, 0.375),
    ];

    let inject = |pair: &mut Pair, p: DVec3| {
        let item = GasInjection::new(Vec3::new(p.x as f32, p.y as f32, p.z as f32), m as f32, VAPOR as f32, Vec3::Y);
        pair.gpu.inject(&pair.queue, &[item]);
        pair.advance(0.0);
    };

    for p in inside {
        let before = pair.state().injected_units();
        let mass_before = pair.mass();

        inject(&mut pair, p);
        assert!(pair.cpu.inject(p, m, VAPOR, DVec3::Y));

        let added = pair.state().injected_units() - before;
        let error = (pair.mass() - mass_before - m).abs();
        let errors = pair.report();

        println!(
            "inside {p}: {added} quanta (expected {units}), mass error {error:e} kg vs quantum {GAS_MASS_QUANTUM:e}, vs reference {errors:?}"
        );

        assert_eq!(added, units);
        assert!(error <= GAS_MASS_QUANTUM + 1e-6 * m);
        assert!(errors.max() < 1e-4);
    }

    let outside = [
        DVec3::new(-0.01, 0.2, 0.2),
        DVec3::new(0.2, 0.51, 0.2),
        DVec3::new(0.2, 0.2, 1.0),
        DVec3::splat(0.375),
    ];

    for p in outside {
        let before = pair.bits();
        let rejected = pair.state().rejected;

        inject(&mut pair, p);

        assert!(!pair.cpu.inject(p, m, VAPOR, DVec3::Y));
        assert_eq!(pair.state().rejected, rejected + 1);
        assert!(pair.bits() == before);
    }
}

#[test]
fn injected_volume_expands_out_of_the_open_top() {
    let params = RefParams {
        density_buoyancy: 0.0,
        thermal_buoyancy: 0.0,
        ..RefParams::default()
    };
    let mass = 0.01;
    let expanded = |volume: f32| {
        let mut pair = pair(reference(UVec3::new(16, 24, 16), params, false, &[]), TOLERANCE, 200, 4);
        let ambient = params.ambient_temperature as f32;
        let item = GasInjection::new(Vec3::splat(0.5), mass as f32, ambient, Vec3::ZERO).with_volume(volume);
        pair.gpu.inject(&pair.queue, &[item]);
        pair.advance(FRAME);

        pair
    };

    let still = expanded(0.0);
    assert!(still.velocity().iter().all(|v| *v == 0.0));

    let volume = 0.01;
    let pair = expanded(volume);
    let state = pair.state();
    let cell = DX.powi(3);
    let rate = volume as f64 / FRAME;
    let div = divergence(&pair, &pair.velocity());
    let outflow = div.iter().sum::<f64>() * cell;
    let mut sources: Vec<f64> = div.iter().map(|d| d * cell).filter(|q| q.abs() > 1e-4 * rate).collect();
    sources.sort_by(f64::total_cmp);
    let elsewhere = div.iter().map(|d| d * cell).filter(|q| q.abs() <= 1e-4 * rate).fold(0.0f64, |m, q| m.max(q.abs()));

    println!(
        "{volume} m3 injected over {FRAME:.5} s: net outflow {outflow:.6} m3/s (expected {rate:.6}), {} source cells from {:.6} to {:.6} m3/s, worst elsewhere {elsewhere:e}, {} substeps, {} quanta",
        sources.len(),
        sources[0],
        sources[sources.len() - 1],
        state.substeps,
        state.injected_units()
    );

    assert_eq!(state.injected_units(), (mass / GAS_MASS_QUANTUM).round() as u64);
    assert_eq!(sources.len(), 8);
    assert!((outflow - rate).abs() < 1e-4 * rate);
    assert!(sources.iter().all(|q| (q - rate / 8.0).abs() < 1e-3 * rate));
    assert!(pair.velocity().iter().any(|v| *v > 0.1));
}

#[test]
fn identical_runs_are_bit_identical() {
    let make = || {
        let cpu = reference(UVec3::new(16, 24, 16), RefParams::default(), true, &[sphere(DVec3::new(0.6, 0.8, 0.5), 0.12)]);
        let mut pair = pair(cpu, 0.0, 4, 4);
        pair.run(40);
        pair.gpu.inject(&pair.queue, &[GasInjection::new(Vec3::new(0.3, 0.4, 0.5), 0.01, VAPOR as f32, Vec3::Y)]);
        pair.run(20);

        pair.bits()
    };

    let a = make();
    let b = make();

    println!("{} words compared", a.len());

    assert!(a == b);
}

#[test]
fn fixed_budget_residual_falls_each_iteration() {
    let shelf = Region {
        shape: Shape::Cuboid { half: DVec3::new(0.2, 0.05, 0.2) },
        center: DVec3::new(0.9, 1.0, 0.75),
    };
    let ratios: Vec<f32> = (1..=6)
        .map(|iterations| {
            let mut pair = pair(reference(UVec3::new(24, 40, 24), RefParams::default(), true, &[shelf]), 1e-30, iterations, 0);
            pair.run(20);

            let state = pair.state();
            state.residual / state.input_divergence()
        })
        .collect();

    println!("max residual over max input divergence after 1..=6 iterations: {ratios:?}");

    assert!(ratios.windows(2).all(|w| w[1] < 0.2 * w[0]));
    assert!(ratios[5] < 1e-6);
}

#[test]
fn warm_start_tightens_a_fixed_budget() {
    let shelf = Region {
        shape: Shape::Cuboid { half: DVec3::new(0.2, 0.05, 0.2) },
        center: DVec3::new(0.9, 1.0, 0.75),
    };
    let worst = |iterations: u32, warm: bool| {
        let mut pair = pair(reference(UVec3::new(24, 40, 24), RefParams::default(), true, &[shelf]), 0.0, iterations, 0);
        let config = GasConfig { warm_start: warm, ..pair.gpu.config.clone() };
        pair.gpu = GasSolver::new(&pair.device, config, &[region(&shelf)], GasTargets::default());
        pair.gpu.initialize(&pair.queue);

        let solid = pair.cpu.solid();
        let ratios: Vec<f64> = (0..6)
            .map(|_| {
                pair.run(10);

                let input = pair.state().input_divergence() as f64;
                let worst = divergence(&pair, &pair.velocity())
                    .iter()
                    .zip(&solid)
                    .filter(|(_, s)| !**s)
                    .fold(0.0f64, |m, (d, _)| m.max(d.abs()));

                worst / input
            })
            .collect();

        ratios.iter().fold(0.0f64, |m, r| m.max(*r))
    };

    let cold = worst(1, false);
    let warm = worst(1, true);
    let warm_two = worst(2, true);

    println!("worst divergence ratio over 60 frames: one cold iteration {cold:e}, one warm iteration {warm:e}, two warm iterations {warm_two:e}");

    assert!(warm < 0.25 * cold);
    assert!(warm < 5e-2);
    assert!(warm_two < 4e-3);
}

const STEAM_MASS: f32 = 2.26e-3;

const STEAM_VOLUME: f32 = 3.6e-3;

const INJECTIONS: usize = 6;

const SETTLED: usize = 24;

fn still_box(quality: Quality, tolerance: f32, open_top: bool) -> GasConfig {
    let params = GasParams {
        density_buoyancy: 0.0,
        thermal_buoyancy: 0.0,
        density_decay: 0.0,
        tolerance,
        max_iterations: if tolerance > 0.0 { 200 } else { 3 },
        open_high: [false, open_top, false],
        ..GasParams::default()
    };

    GasConfig {
        dims: GpuUVec3::splat(48),
        dx: 1.0 / 32.0,
        params,
        max_substeps: if tolerance > 0.0 { 8 } else { 2 },
        injection_capacity: 4,
        ..GasConfig::default()
    }
    .with_quality(quality)
}

fn chamber(center: Vec3, half: f32, thickness: f32) -> Vec<GasRegion> {
    (0..3)
        .flat_map(|a| [-1.0, 1.0].map(|side| (a, side)))
        .map(|(a, side)| {
            let mut offset = Vec3::ZERO;
            let mut extent = Vec3::splat(half + thickness);
            offset[a] = side * (half + 0.5 * thickness);
            extent[a] = 0.5 * thickness;

            GasRegion::cuboid(center + offset, extent)
        })
        .collect()
}

struct Boiled {
    after: f64,
    settled: f64,
    edge: f64,
    speed: f32,
    finite: bool,
}

fn boiled(config: GasConfig, obstacles: &[GasRegion], volume: f32, jet: f32) -> Boiled {
    let (device, queue) = gpu();
    let mut solver = GasSolver::new(&device, config, obstacles, GasTargets::default());
    solver.initialize(&queue);

    let dims = solver.config.dims;
    let center = solver.config.size() * 0.5;
    let cells = solver.config.cell_count() as usize;
    let cell = (solver.config.dx as f64).powi(3);
    let density = |solver: &GasSolver| read::<f32>(&device, &queue, &solver.buffers.density, cells);
    let ratio = |solver: &GasSolver| {
        let state = read::<GasState>(&device, &queue, &solver.buffers.state, 1)[0];
        let mass = density(solver).iter().map(|r| *r as f64).sum::<f64>() * cell;

        mass / (state.injected_units() as f64 * GAS_MASS_QUANTUM)
    };
    let mut after = 0.0;

    for frame in 0..SETTLED {
        if frame < INJECTIONS {
            let position = center + Vec3::new(0.013, 0.007, -0.011) * (frame % 4) as f32;
            let velocity = Vec3::X * if frame % 2 == 0 { jet } else { -jet };
            let item = GasInjection::new(position, STEAM_MASS, VAPOR as f32, velocity).with_volume(volume);
            solver.inject(&queue, &[item]);
        }

        solver.set_frame(&queue, FRAME as f32, &[]);

        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder);
        queue.submit([encoder.finish()]);

        if frame + 1 == INJECTIONS {
            after = ratio(&solver);
        }
    }

    let rho = density(&solver);
    let temperature: Vec<f32> = read(&device, &queue, &solver.buffers.temperature, cells);
    let velocity: Vec<f32> = read(&device, &queue, &solver.buffers.velocity, solver.config.face_count() as usize);
    let state = read::<GasState>(&device, &queue, &solver.buffers.state, 1)[0];
    let on_edge = |n: usize| {
        let n = n as u32;
        let c = GpuUVec3::new(n % dims.x, (n / dims.x) % dims.y, n / (dims.x * dims.y));

        (0..3).any(|a| c[a] == 0 || c[a] + 1 == dims[a])
    };
    let injected = state.injected_units() as f64 * GAS_MASS_QUANTUM;

    assert!((injected - INJECTIONS as f64 * STEAM_MASS as f64).abs() < 1e-6);

    Boiled {
        after,
        settled: ratio(&solver),
        edge: rho.iter().enumerate().filter(|(n, _)| on_edge(*n)).map(|(_, r)| *r as f64).sum::<f64>() * cell / injected,
        speed: velocity.iter().fold(0.0, |m, v| m.max(v.abs())),
        finite: [rho, temperature, velocity].iter().flatten().all(|v| v.is_finite()),
    }
}

#[test]
fn expanding_steam_keeps_its_mass() {
    for (name, quality, tolerance) in [("tolerance", Quality::High, TOLERANCE), ("game", Quality::Game, 0.0)] {
        let config = still_box(quality, tolerance, true);
        let still = boiled(config.clone(), &[], 0.0, 0.0);
        let jets = boiled(config.clone(), &[], 0.0, 3.0);
        let steam = boiled(config, &[], STEAM_VOLUME, 0.0);

        for (case, run) in [("no volume, at rest", &still), ("no volume, 3 m/s jets", &jets), ("steam volume", &steam)] {
            println!(
                "{name} mode, {case}: grid mass over injected mass {:.4} after the last injection and {:.4} at the end, {:.2e} of it at the walls, max |u| {:.3}",
                run.after, run.settled, run.edge, run.speed
            );

            assert!(run.finite);
            assert!(run.edge < 1e-2);
        }

        assert!((still.settled - 1.0).abs() < 1e-4);
        assert!((steam.after - 1.0).abs() < 0.25);
        assert!(steam.settled > 0.7 && steam.settled < 1.4);
    }
}

#[test]
fn closed_spaces_drop_the_expansion() {
    for (name, quality, tolerance) in [("tolerance", Quality::High, TOLERANCE), ("game", Quality::Game, 0.0)] {
        let closed = boiled(still_box(quality, tolerance, false), &[], STEAM_VOLUME, 0.0);
        let vented = still_box(quality, tolerance, true);
        let walls = chamber(vented.size() * 0.5, 0.15, 0.12);
        let pocket = boiled(vented, &walls, STEAM_VOLUME, 0.0);

        for (case, run) in [("closed box", &closed), ("sealed pocket in an open box", &pocket)] {
            println!(
                "{name} mode, {case}: grid mass over injected mass {:.6} at the end, max |u| {:e}, finite {}",
                run.settled, run.speed, run.finite
            );

            assert!(run.finite);
            assert!(run.speed == 0.0);
            assert!((run.settled - 1.0).abs() < 1e-4);
        }
    }
}

fn open_room() -> GasConfig {
    GasConfig {
        dims: GpuUVec3::new(40, 52, 40),
        dx: 0.04,
        params: GasParams {
            vorticity_confinement: 4.0,
            cooling: 0.6,
            cfl: 2.0,
            open_low: [true, false, true],
            open_high: [true; 3],
            ..GasParams::default()
        },
        max_substeps: 1,
        ..GasConfig::default()
    }
}

#[test]
fn draught_through_open_sides_dies_down_after_the_forcing_stops() {
    let (device, queue) = gpu();
    let mut solver = GasSolver::new(&device, open_room(), &[], GasTargets::default());
    solver.initialize(&queue);

    let cold = GasEmitter { region: GasRegion::sphere(Vec3::new(0.8, 1.7, 0.8), 0.14), density_rate: 0.0, heat_rate: -500.0 };
    let speed_after = |solver: &mut GasSolver, emitters: &[GasEmitter], frames: usize| {
        for _ in 0..frames {
            solver.set_frame(&queue, FRAME as f32, emitters);

            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode_frame(&mut encoder);
            queue.submit([encoder.finish()]);
        }

        read::<GasState>(&device, &queue, &solver.buffers.state, 1)[0].max_speed()
    };

    let forced = speed_after(&mut solver, &[cold], 300);
    let free: Vec<f32> = (0..8).map(|_| speed_after(&mut solver, &[], 60)).collect();

    println!("max speed {forced} m/s when the cold source stops, then each second without a source: {free:?}");

    assert!(forced > 1.0);
    assert!(free.iter().all(|s| s.is_finite() && *s < 1.2 * forced));
    assert!(free[7] < 0.25 * forced);
}
