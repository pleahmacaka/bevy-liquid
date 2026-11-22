use std::collections::HashMap;

use bevy::math::UVec3;
use glam::DVec3;
use liquid_gpu::layout::{GpuBody, GpuParticle, GpuState, S_BAND_DT, S_TOLERANCE, S_VOLUME, S_VOLUME_INTEGRAL, S_VOLUME_TARGET, SCALAR_COUNT};
use liquid_gpu::solver::{Bind, GUARD_BYTES, GpuSolver, SolverConfig};
use liquid_ref::grid::{Grid, sample_cells};
use liquid_ref::levelset::redistance;
use liquid_ref::narrow_band::{NarrowBand, Resample};
use liquid_ref::particles::{Particle, hash_u32};
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{dam_break, tank, with_body};
use liquid_ref::sim::{Params, Simulation, level_set_volume};
use liquid_ref::transfer::Transfer;
use liquid_ref::viscosity::flatten;

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
        label: Some("narrow band test"),
        required_features: wgpu::Features::empty(),
        required_limits: adapter.limits(),
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

fn to_gpu(id: usize, p: &Particle) -> GpuParticle {
    let f = |v: DVec3| [v.x as f32, v.y as f32, v.z as f32, 0.0];
    let mut x = f(p.x);
    x[3] = id as f32;

    GpuParticle { x, v: f(p.v), c: [f(p.c[0]), f(p.c[1]), f(p.c[2])] }
}

fn position(p: &GpuParticle) -> DVec3 {
    DVec3::new(p.x[0] as f64, p.x[1] as f64, p.x[2] as f64)
}

fn velocity(p: &GpuParticle) -> DVec3 {
    DVec3::new(p.v[0] as f64, p.v[1] as f64, p.v[2] as f64)
}

fn band_params() -> Params {
    Params { narrow_band: Some(NarrowBand::default()), ..Params::default() }
}

fn config_for(sim: &Simulation, sparse: bool) -> SolverConfig {
    let d = sim.grid.dims;

    SolverConfig {
        dims: UVec3::new(d.x, d.y, d.z),
        dx: sim.grid.dx as f32,
        max_particles: 4 * sim.particles.len() as u32,
        max_substeps: 1,
        pressure_iterations: 60,
        tolerance: 1e-6,
        coarse_sweeps: 16,
        density_projection: false,
        narrow_band: Some(NarrowBand::default().band_cells as f32),
        sparse,
        ..SolverConfig::default()
    }
}

fn small_band_dam_break(substeps: usize) -> Simulation {
    let mut sim = dam_break(glam::UVec3::new(24, 16, 8), 0.02, DVec3::new(0.16, 0.24, 0.0), band_params());

    for _ in 0..substeps {
        let dt = sim.cfl_dt();
        sim.substep(dt);
    }

    sim
}

fn alive(particles: &[GpuParticle]) -> impl Iterator<Item = &GpuParticle> {
    particles.iter().filter(|p| p.v[3] < 0.5)
}

#[test]
fn gpu_resample_matches_reference_with_shared_salt() {
    let (device, queue) = gpu();
    let sim = small_band_dam_break(6);
    let time = 0.123f32;
    let solver = validated(&device, || GpuSolver::new(&device, config_for(&sim, false), None));
    solver.initialize(&queue);

    let b = &solver.buffers;
    let state = GpuState { time, dt: 0.01, ..GpuState::default() };
    queue.write_buffer(&b.state, 0, bytemuck::bytes_of(&state));
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());
    queue.write_buffer(&b.phi, 0, bytemuck::cast_slice(&to_f32(&sim.phi.data)));
    queue.write_buffer(&b.velocity, 0, bytemuck::cast_slice(&to_f32(&flatten(&sim.velocity))));

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        solver.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
        solver.encode_sort(&mut pass);
        solver.encode_resample(&mut pass);
    }
    queue.submit([encoder.finish()]);

    let count = read::<GpuState>(&device, &queue, &b.state, 1)[0].particle_count as usize;
    let particles: Vec<GpuParticle> = read(&device, &queue, &b.particles, count);

    let mut cpu = sim.particles.clone();
    Resample {
        grid: &sim.grid,
        phi: &sim.phi,
        velocity: &sim.velocity,
        transfer: Transfer::Apic,
        band: NarrowBand::default(),
        salt: hash_u32(time.to_bits()),
    }
    .apply(&mut cpu);

    let ids: HashMap<[u32; 3], usize> = sim
        .particles
        .iter()
        .enumerate()
        .map(|(i, p)| ([p.x.x as f32, p.x.y as f32, p.x.z as f32].map(f32::to_bits), i))
        .collect();
    let key = |x: DVec3| [x.x as f32, x.y as f32, x.z as f32].map(f32::to_bits);
    let mut cpu_kept: Vec<usize> = cpu.iter().filter_map(|p| ids.get(&key(p.x)).copied()).collect();
    let cpu_fresh: Vec<&Particle> = cpu.iter().filter(|p| !ids.contains_key(&key(p.x))).collect();
    let mut gpu_kept: Vec<usize> = alive(&particles).filter(|p| p.x[3] >= 0.0).map(|p| p.x[3] as usize).collect();
    let gpu_fresh: Vec<&GpuParticle> = alive(&particles).filter(|p| p.x[3] < 0.0).collect();
    cpu_kept.sort_unstable();
    gpu_kept.sort_unstable();

    let mut worst_x = 0.0f64;
    let mut worst_v = 0.0f64;
    let mut unmatched = 0;
    let vmax = sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length()));

    for g in &gpu_fresh {
        let x = position(g);
        let nearest = cpu_fresh.iter().min_by(|a, b| (a.x - x).length().total_cmp(&(b.x - x).length()));

        match nearest {
            Some(c) if (c.x - x).length() < 1e-4 * sim.grid.dx => {
                worst_x = worst_x.max((c.x - x).length());
                worst_v = worst_v.max((c.v - velocity(g)).length());
            }
            _ => unmatched += 1,
        }
    }

    println!(
        "resample of {} particles: kept gpu {} cpu {}, seeded gpu {} cpu {} ({unmatched} unmatched), seeded |dx| {:.3e} dx, |dv| {:.3e} of vmax {vmax:.3}",
        sim.particles.len(),
        gpu_kept.len(),
        cpu_kept.len(),
        gpu_fresh.len(),
        cpu_fresh.len(),
        worst_x / sim.grid.dx,
        worst_v / vmax
    );

    assert_eq!(gpu_kept, cpu_kept);
    assert_eq!(gpu_fresh.len(), cpu_fresh.len());
    assert_eq!(unmatched, 0);
    assert!(worst_x < 1e-5 * sim.grid.dx);
    assert!(worst_v < 1e-4 * vmax);
}

#[test]
fn gpu_band_substep_matches_reference() {
    let (device, queue) = gpu();
    let mut sim = small_band_dam_break(6);
    let frame_dt = (0.999 * sim.cfl_dt()) as f32;
    let mut solver = validated(&device, || GpuSolver::new(&device, config_for(&sim, false), None));
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, frame_dt);

    let control = sim.volume_control.expect("volume control starts with the first substep");
    let mut scalars = vec![0f32; SCALAR_COUNT];
    scalars[S_TOLERANCE] = solver.config.tolerance;
    scalars[S_BAND_DT] = sim.report.dt as f32;
    scalars[S_VOLUME_TARGET] = control.target as f32;
    scalars[S_VOLUME_INTEGRAL] = control.integral as f32;

    let b = &solver.buffers;
    queue.write_buffer(&b.scalars, 0, bytemuck::cast_slice(&scalars));
    queue.write_buffer(&b.band_phi, 0, bytemuck::cast_slice(&to_f32(&sim.phi.data)));
    queue.write_buffer(&b.velocity, 0, bytemuck::cast_slice(&to_f32(&flatten(&sim.velocity))));
    solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let state: GpuState = read(&device, &queue, &b.state, 1)[0];
    let particles: Vec<GpuParticle> = read(&device, &queue, &b.particles, state.particle_count as usize);
    let phi: Vec<f32> = read(&device, &queue, &b.band_phi, sim.phi.data.len());
    let faces = flatten(&sim.velocity).len();
    let grid_velocity: Vec<f32> = read(&device, &queue, &b.velocity, faces);
    let pressure: Vec<f32> = read(&device, &queue, &solver.levels[0].x, sim.phi.data.len());
    let after: Vec<f32> = read(&device, &queue, &b.scalars, SCALAR_COUNT);
    let before = sim.particles.len();
    sim.substep(frame_dt as f64);

    let dx = sim.grid.dx;
    let mut phi_error = 0.0f64;

    for (g, c) in phi.iter().zip(&sim.phi.data) {
        if c.abs() < 3.0 * dx {
            phi_error = phi_error.max((*g as f64 - c).abs());
        }
    }

    let cpu_u = flatten(&sim.velocity);
    let vmax = cpu_u.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let mut u_error = 0.0f64;

    for (axis, field) in sim.velocity.axes.iter().enumerate() {
        let start: usize = sim.velocity.axes[..axis].iter().map(|a| a.data.len()).sum();

        for n in 0..field.data.len() {
            let x = sim.grid.face_position(axis, field.coords(n));

            if sample_cells(&sim.phi, &sim.grid, x) < 0.0 {
                u_error = u_error.max((grid_velocity[start + n] as f64 - cpu_u[start + n]).abs());
            }
        }
    }

    let pmax = sim.pressure.iter().fold(0.0f64, |m, p| m.max(p.abs()));
    let p_error = pressure.iter().zip(&sim.pressure).fold(0.0f64, |m, (g, c)| m.max((*g as f64 - c).abs()));
    let cpu_by_cell = bins(&sim.grid, sim.particles.iter().map(|p| p.x));
    let mut x_error = 0.0f64;
    let mut v_error = 0.0f64;
    let gpu_kept: Vec<&GpuParticle> = alive(&particles).filter(|p| p.x[3] >= 0.0).collect();
    let gpu_alive = alive(&particles).count();

    for g in &gpu_kept {
        let k = nearest(&sim.grid, &cpu_by_cell, &sim.particles, position(g));
        x_error = x_error.max((sim.particles[k].x - position(g)).length());
        v_error = v_error.max((sim.particles[k].v - velocity(g)).length());
    }

    let control = sim.volume_control.unwrap_or_else(|| unreachable!());
    let volume = level_set_volume(&sim.grid, &sim.phi);

    println!(
        "band substep: dt gpu {:.6e} cpu {frame_dt:.6e}; |dphi| in band {:.3e} dx, liquid |du| {:.3e} of {vmax:.3}, |dp| {:.3e} of {pmax:.1} Pa; kept particles |dx| {:.3e} dx, |dv| {:.3e}; particles gpu {gpu_alive} cpu {} (from {before}); volume gpu {:.6e} cpu {volume:.6e}, integral gpu {:.4e} cpu {:.4e}",
        state.dt,
        phi_error / dx,
        u_error / vmax,
        p_error / pmax,
        x_error / dx,
        v_error / vmax,
        sim.particles.len(),
        after[S_VOLUME],
        after[S_VOLUME_INTEGRAL],
        control.integral
    );

    assert_eq!(state.substeps, 1);
    assert!(phi_error < 1e-3 * dx);
    assert!(u_error < 1e-3 * vmax);
    assert!(p_error < 1e-3 * pmax);
    assert!(x_error < 1e-3 * dx);
    assert!(v_error < 1e-3 * vmax);
    assert!((gpu_alive as f64 - sim.particles.len() as f64).abs() < 0.01 * sim.particles.len() as f64);
    assert!((after[S_VOLUME] as f64 - volume).abs() < 1e-4 * volume);
}

fn bins(grid: &Grid, points: impl Iterator<Item = DVec3>) -> HashMap<glam::IVec3, Vec<usize>> {
    let mut out: HashMap<glam::IVec3, Vec<usize>> = HashMap::new();

    for (i, x) in points.enumerate() {
        out.entry(grid.cell_of(x)).or_default().push(i);
    }

    out
}

fn nearest(grid: &Grid, bins: &HashMap<glam::IVec3, Vec<usize>>, particles: &[Particle], x: DVec3) -> usize {
    let c = grid.cell_of(x);
    let mut best = (f64::INFINITY, 0);

    for o in 0..27 {
        let cell = c + glam::IVec3::new(o % 3, (o / 3) % 3, o / 9) - 1;

        for &i in bins.get(&cell).map_or(&[][..], |v| v.as_slice()) {
            let d = (particles[i].x - x).length();

            if d < best.0 {
                best = (d, i);
            }
        }
    }

    best.1
}

struct Run {
    device: wgpu::Device,
    queue: wgpu::Queue,
    solver: GpuSolver,
}

impl Run {
    fn new(sim: &Simulation, band: bool, max_substeps: u32) -> Self {
        let (device, queue) = gpu();
        let config = SolverConfig {
            narrow_band: band.then_some(NarrowBand::default().band_cells as f32),
            density_projection: !band,
            max_substeps,
            pressure_iterations: 40,
            density_iterations: 20,
            tolerance: 1e-5,
            ..config_for(sim, true)
        };
        let mut solver = validated(&device, || GpuSolver::new(&device, config, None));
        solver.initialize(&queue);
        solver.set_frame_dt(&queue, 1.0 / 60.0);
        solver.upload_particles(&queue, &sim.particles.iter().enumerate().map(|(i, p)| to_gpu(i, p)).collect::<Vec<_>>());

        Self { device, queue, solver }
    }

    fn advance(&mut self, duration: f32) -> GpuState {
        self.solver.set_frame_dt(&self.queue, duration);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.solver.encode_frame(&mut encoder, None);
        self.queue.submit([encoder.finish()]);

        let state: GpuState = read(&self.device, &self.queue, &self.solver.buffers.state, 1)[0];
        assert!(state.remaining < 1e-6 * duration, "frame left {} s unsimulated", state.remaining);

        state
    }

    fn particles(&self) -> Vec<GpuParticle> {
        let count = read::<GpuState>(&self.device, &self.queue, &self.solver.buffers.state, 1)[0].particle_count as usize;

        read(&self.device, &self.queue, &self.solver.buffers.particles, count)
    }

    fn scalar(&self, index: usize) -> f32 {
        read::<f32>(&self.device, &self.queue, &self.solver.buffers.scalars, SCALAR_COUNT)[index]
    }

    fn slots(&self) -> Vec<u32> {
        let table = self.solver.slot_table();
        let words: Vec<u32> = read(&self.device, &self.queue, &self.solver.buffers.brick_list, (table.end / 4) as usize);

        words[(table.start / 4) as usize..].to_vec()
    }

    fn stored(&self, buffer: &wgpu::Buffer) -> Vec<f32> {
        read(&self.device, &self.queue, buffer, ((buffer.size() - GUARD_BYTES) / 4) as usize)
    }

    fn cells(&self, buffer: &wgpu::Buffer) -> Vec<f64> {
        let (stored, slots) = (self.stored(buffer), self.slots());
        let d = self.solver.config.dims;
        let mut out = Vec::new();

        for z in 0..d.z {
            for y in 0..d.y {
                for x in 0..d.x {
                    out.push(stored[self.solver.pooled_cell(&slots, UVec3::new(x, y, z))] as f64);
                }
            }
        }

        out
    }

    fn faces(&self, buffer: &wgpu::Buffer) -> Vec<f64> {
        let (stored, slots) = (self.stored(buffer), self.slots());
        let mut out = Vec::new();

        for axis in 0..3 {
            let mut d = self.solver.config.dims;
            d[axis] += 1;

            for z in 0..d.z {
                for y in 0..d.y {
                    for x in 0..d.x {
                        out.push(stored[self.solver.pooled_face(&slots, axis, UVec3::new(x, y, z))] as f64);
                    }
                }
            }
        }

        out
    }
}

fn front_position(particles: &[GpuParticle], a: f64) -> f64 {
    alive(particles).filter(|p| (p.x[1] as f64) < 0.25 * a).fold(0.0, |m, p| m.max(p.x[0] as f64))
}

fn fitted_slope(samples: &[(f64, f64)]) -> (f64, usize) {
    let window: Vec<(f64, f64)> = samples.iter().copied().filter(|(_, z)| (1.5..=3.5).contains(z)).collect();
    let n = window.len() as f64;
    let mt = window.iter().map(|(t, _)| t).sum::<f64>() / n;
    let mz = window.iter().map(|(_, z)| z).sum::<f64>() / n;
    let num: f64 = window.iter().map(|(t, z)| (t - mt) * (z - mz)).sum();
    let den: f64 = window.iter().map(|(t, _)| (t - mt).powi(2)).sum();

    (num / den, window.len())
}

struct FrontRun {
    slope: f64,
    points: usize,
    initial: usize,
    mean: f64,
}

fn martin_moyce(band: bool) -> FrontRun {
    let a = 0.057;
    let cells = 16;
    let dx = a / cells as f64;
    let sim = dam_break(glam::UVec3::new(5 * cells, 3 * cells, 4), dx, DVec3::new(a, 2.0 * a, 0.0), Params::default());
    let mut run = Run::new(&sim, band, 8);
    let t_scale = (a / (2.0 * 9.81)).sqrt();
    let mut samples = Vec::new();
    let mut mean = 0.0;

    for n in 1..=40 {
        run.advance((0.1 * t_scale) as f32);
        let particles = run.particles();
        samples.push((0.1 * n as f64, front_position(&particles, a) / a));
        mean += alive(&particles).count() as f64 / 40.0;
    }

    let (slope, points) = fitted_slope(&samples);

    FrontRun { slope, points, initial: sim.particles.len(), mean }
}

#[test]
fn gpu_band_dam_break_front_matches_dense() {
    let band = martin_moyce(true);
    let dense = martin_moyce(false);
    let ratio = band.mean / dense.mean;

    println!(
        "gpu front slope: narrow band {:.3} ({} points), dense {:.3} ({} points), Martin & Moyce 1.29; particles dense {:.0} mean of {} seeded, band mean {:.0}, ratio {ratio:.3}",
        band.slope, band.points, dense.slope, dense.points, dense.mean, dense.initial, band.mean
    );

    assert!(band.points >= 5);
    assert!((band.slope - dense.slope).abs() < 0.1 * dense.slope);
    assert!((band.slope - 1.29).abs() < 0.15 * 1.29);
    assert!(ratio < 0.6);
}

fn redistanced_volume(grid: &Grid, phi: &[f64]) -> f64 {
    let field = liquid_ref::grid::Field3 { dims: grid.dims, data: phi.to_vec() };

    level_set_volume(grid, &redistance(&field, grid.dx))
}

#[test]
fn gpu_band_dam_break_keeps_volume() {
    let sim = dam_break(glam::UVec3::new(48, 32, 4), 0.01, DVec3::new(0.16, 0.24, 0.0), band_params());
    let mut run = Run::new(&sim, true, 16);
    let mut v0 = None;
    let mut worst = 0.0f64;
    let mut last = 0.0;

    for frame in 0..90 {
        run.advance(1.0 / 60.0);

        if frame % 6 == 5 || frame == 0 {
            let volume = redistanced_volume(&sim.grid, &run.cells(&run.solver.buffers.band_phi));
            let reference = *v0.get_or_insert(volume);
            last = volume / reference;
            worst = worst.max((last - 1.0).abs());
        }
    }

    let particles = alive(&run.particles()).count();

    println!(
        "gpu narrow band dam break volume after 1.5 s: {last:.4} (worst deviation {worst:.4} over 0.1 s samples), {particles} particles from {} seeded",
        sim.particles.len()
    );

    assert!((last - 1.0).abs() < 0.03);
    assert!(worst < 0.03);
}

#[test]
fn gpu_band_hydrostatic_tank_stays_at_rest() {
    let dx = 1.0 / 32.0;
    let depth = 12.0 * dx;
    let sim = tank(glam::UVec3::new(16, 24, 16), dx, depth, band_params());
    let seeded = sim.particles.len();
    let mut run = Run::new(&sim, true, 4);

    run.advance(1.0 / 60.0);
    let v0 = run.scalar(S_VOLUME) as f64;

    for _ in 0..17 {
        run.advance(1.0 / 60.0);
    }
    let mut bottom = 0.0;
    let frames = 10;

    for _ in 0..frames {
        run.advance(0.02);
        let pressure = run.cells(&run.solver.levels[0].x);
        let d = sim.grid.dims;
        let floor: f64 = (0..d.z).flat_map(|k| (0..d.x).map(move |i| (i, k))).map(|(i, k)| pressure[(i + d.x * d.y * k) as usize]).sum();
        bottom += floor / (d.x * d.z) as f64 / frames as f64;
    }

    let wave_speed = (9.81 * depth).sqrt();
    let particles = run.particles();
    let particle_speed = alive(&particles).fold(0.0f64, |m, p| m.max(velocity(p).length()));
    let phi = run.cells(&run.solver.buffers.band_phi);
    let field = liquid_ref::grid::Field3 { dims: sim.grid.dims, data: phi };
    let u = run.faces(&run.solver.buffers.velocity);
    let mut grid_speed = 0.0f64;
    let mut start = 0;

    for axis in 0..3 {
        let face_count = sim.grid.faces(axis).data.len();

        for n in 0..face_count {
            let f = sim.grid.faces(axis).coords(n);

            if sample_cells(&field, &sim.grid, sim.grid.face_position(axis, f)) < 0.0 {
                grid_speed = grid_speed.max(u[start + n].abs());
            }
        }

        start += face_count;
    }

    let expected = 1000.0 * 9.81 * (depth - 0.5 * dx);
    let drift = (run.scalar(S_VOLUME) as f64 - v0).abs() / v0;
    let count = alive(&particles).count();
    let ratio = count as f64 / seeded as f64;

    println!(
        "gpu band tank: max particle speed {particle_speed:.3e} m/s, max liquid face speed {grid_speed:.3e} m/s ({:.3e} of sqrt(gh)), bottom p {bottom:.1} vs {expected:.1}, volume drift {drift:.3e}, particles {count} of {seeded} ({ratio:.3})",
        grid_speed / wave_speed
    );

    assert!(particle_speed < 0.01 * wave_speed);
    assert!(grid_speed < 0.01 * wave_speed);
    assert!((bottom - expected).abs() < 0.01 * expected);
    assert!(drift < 0.005);
    assert!(ratio < 0.35);
}

fn to_gpu_body(b: &RigidBody) -> GpuBody {
    let f = |v: DVec3, w: f64| [v.x as f32, v.y as f32, v.z as f32, w as f32];
    let inv = b.inverse_inertia_world();
    let half = match b.shape {
        Shape::Cuboid { half } => half,
        Shape::Sphere { radius } => DVec3::splat(radius),
    };

    GpuBody {
        position: f(b.position, 1.0 / b.mass()),
        rotation: [b.rotation.x as f32, b.rotation.y as f32, b.rotation.z as f32, b.rotation.w as f32],
        velocity: f(b.velocity, 2.0),
        angular: f(b.angular_velocity, 0.0),
        shape: f(half, 1.0),
        inv_inertia: [f(inv.row(0), 0.0), f(inv.row(1), 0.0), f(inv.row(2), 0.0)],
    }
}

struct FloatRun {
    rest: f64,
    peak_speed: f64,
}

fn floating_block(band: bool) -> FloatRun {
    let body = RigidBody::new(
        Shape::Cuboid { half: DVec3::new(0.08, 0.05, 0.1) },
        DVec3::new(0.25, 0.36, 0.25),
        Motion::Dynamic,
        450.0,
    );
    let sim = with_body(tank(glam::UVec3::new(20, 24, 20), 0.025, 0.3, band_params()), body);
    let mut run = Run::new(&sim, band, 8);
    run.solver.upload_bodies(&run.queue, &[to_gpu_body(&sim.bodies[0])]);

    let mut peak_speed = 0.0f64;
    let mut heights = Vec::new();

    for frame in 0..180 {
        run.advance(1.0 / 60.0);

        let particles = run.particles();
        let speed = alive(&particles).fold(0.0f64, |m, p| m.max(velocity(p).length()));
        let y = read::<GpuBody>(&run.device, &run.queue, &run.solver.buffers.bodies, 1)[0].position[1] as f64;

        assert!(speed.is_finite() && y.is_finite(), "band {band}: diverged at frame {frame}");
        peak_speed = peak_speed.max(speed);

        if frame >= 120 {
            heights.push(y);
        }
    }

    FloatRun { rest: heights.iter().sum::<f64>() / heights.len() as f64, peak_speed }
}

#[test]
fn gpu_band_floating_block_settles_like_dense() {
    let band = floating_block(true);
    let dense = floating_block(false);

    println!(
        "gpu block rest height over 2..3 s: narrow band {:.4}, dense {:.4}; peak particle speed band {:.3}, dense {:.3} m/s",
        band.rest, dense.rest, band.peak_speed, dense.peak_speed
    );

    assert!(band.peak_speed < 2.0 * dense.peak_speed.max(0.5));
    assert!((band.rest - dense.rest).abs() < 0.4 * 0.025);
}
