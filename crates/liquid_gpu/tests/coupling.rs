use bevy::math::{UVec3, Vec3};
use liquid_gpu::coupling::TierLink;
use liquid_gpu::emitter::{GpuEmissionState, LiquidEmitter, particle_volume};
use liquid_gpu::gas::GasRegion;
use liquid_gpu::layout::{GpuState, S_VOLUME, SCALAR_COUNT};
use liquid_gpu::solver::{GpuSolver, HeightfieldLink, SolverConfig};
use liquid_ocean::swe::{self, GpuSwe};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::swe::{CouplingRegion, Ledger, Swe};

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

const FRAME: f32 = 1.0 / 60.0;
const HF_DX: f64 = 0.05;
const DOMAIN: UVec3 = UVec3::new(16, 12, 16);

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
        label: Some("liquid coupling test"),
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

fn region_at(offset: Vec3) -> CouplingRegion {
    let min = [(offset.x as f64 / HF_DX).round() as usize, (offset.z as f64 / HF_DX).round() as usize];

    CouplingRegion {
        min,
        max: [min[0] + DOMAIN.x as usize, min[1] + DOMAIN.z as usize],
        blend: 0.5,
    }
}

fn pool(offset: Vec3) -> Swe {
    let grid = Grid::new(64, 64, HF_DX, HF_DX);
    let bed = Bathymetry::flat(&grid, 0.0);
    let mut state = State::lake(&grid, &bed, 0.2);

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            if grid.center(i, k)[0] < 0.6 {
                state.h[grid.cell(i, k)] = 0.32;
            }
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.coupling = Some(region_at(offset));

    swe
}

fn domain(device: &wgpu::Device, queue: &wgpu::Queue, field: &GpuSwe, offset: Vec3) -> GpuSolver {
    let config = SolverConfig {
        dims: DOMAIN,
        dx: HF_DX as f32,
        max_particles: 200_000,
        max_substeps: 2,
        narrow_band: Some(3.0),
        ..SolverConfig::default()
    };
    let mut solver = GpuSolver::new(device, config, None);
    let b = &field.buffers;

    solver.attach_heightfield(
        device,
        HeightfieldLink {
            offset,
            cells: [64, 64],
            cell_size: [HF_DX as f32; 2],
            relax_cells: 4.0,
            blend: 0.5,
            state: &b.state,
            bed: &b.bed,
            coupled: &b.coupled,
        },
    );
    solver.link_tiers(device, TierLink { profile: &b.profile, ledger: &b.ledger });
    solver.initialize(queue);
    solver.set_frame_dt(queue, FRAME);

    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_seed(&mut encoder);
    queue.submit([encoder.finish()]);

    solver
}

fn advance(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver, field: &GpuSwe, frames: usize) {
    for _ in 0..frames {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_frame(&mut encoder, None);
        field.encode_frame(&mut encoder, 4);
        queue.submit([encoder.finish()]);
    }
}

fn ledger(device: &wgpu::Device, queue: &wgpu::Queue, field: &GpuSwe) -> Ledger {
    field.read_ledger(device, queue).ledger()
}

fn band_volume(device: &wgpu::Device, queue: &wgpu::Queue, solver: &GpuSolver) -> f32 {
    read::<f32>(device, queue, &solver.buffers.scalars, SCALAR_COUNT)[S_VOLUME]
}

struct Run {
    device: wgpu::Device,
    queue: wgpu::Queue,
    field: GpuSwe,
    solver: GpuSolver,
    seeded: Ledger,
}

fn start(offset: Vec3) -> Run {
    let (device, queue) = gpu();
    let swe = pool(offset);
    let mut field = GpuSwe::new(&device, &swe, FRAME / 4.0, swe::shared_buffers(&device, &swe));
    field.write_params(&queue, FRAME);

    let seeded = swe.ledger();
    let solver = domain(&device, &queue, &field, offset);
    let live = read::<GpuState>(&device, &queue, &solver.buffers.state, 1)[0].particle_count;

    println!("seeded {live} particles from the heightfield columns under the domain");
    assert!(live > 7_000, "seeding produced only {live} particles");

    Run {
        device,
        queue,
        field,
        solver,
        seeded,
    }
}

#[test]
fn coupled_run_conserves_tier_mass() {
    let offset = Vec3::new(1.2, 0.0, 1.2);
    let run = start(offset);
    let (d, q) = (&run.device, &run.queue);

    advance(d, q, &run.solver, &run.field, 2);
    let first = ledger(d, q, &run.field);
    let reference = first.conserved();
    let mut worst: f64 = 0.0;

    for second in 1..=6 {
        advance(d, q, &run.solver, &run.field, 60);
        let now = ledger(d, q, &run.field);
        let drift = (now.conserved() - reference) / first.tier;
        worst = worst.max(drift.abs());

        println!(
            "t = {second} s: 3D tier {:.5} m^3 (band volume {:.5}), outside {:.5}, interface {:+.5}, conserved {:.6} (drift {:+.3}% of the 3D volume)",
            now.tier,
            band_volume(d, q, &run.solver),
            now.outside,
            now.interface,
            now.conserved(),
            100.0 * drift
        );
    }

    let under = run.seeded.mass - run.seeded.outside;

    println!(
        "heightfield volume under the domain at seeding {under:.5}, 3D tier after the first frames {:.5}, worst drift {:.3}% of the 3D volume",
        first.tier,
        100.0 * worst
    );

    assert!((first.tier / under - 1.0).abs() < 0.05, "seeded 3D volume {} vs heightfield {under}", first.tier);
    assert!(worst < 0.03, "tier mass drifted by {:.3}% of the 3D volume", 100.0 * worst);
}

#[test]
fn relocation_conserves_tier_mass() {
    let offset = Vec3::new(1.2, 0.0, 1.2);
    let mut run = start(offset);
    let (d, q) = (run.device.clone(), run.queue.clone());

    advance(&d, &q, &run.solver, &run.field, 60);
    let before = ledger(&d, &q, &run.field);
    let moved = Vec3::new(0.4, 0.0, 1.6);

    run.field.set_coupling(Some(region_at(moved)));
    run.field.write_params(&q, FRAME);
    run.solver = domain(&d, &q, &run.field, moved);
    advance(&d, &q, &run.solver, &run.field, 1);
    let after = ledger(&d, &q, &run.field);

    advance(&d, &q, &run.solver, &run.field, 120);
    let settled = ledger(&d, &q, &run.field);
    let jump = (after.conserved() - before.conserved()) / before.tier;
    let drift = (settled.conserved() - before.conserved()) / before.tier;

    println!(
        "before: tier {:.5}, outside {:.5}, interface {:+.5}; one frame after relocation: tier {:.5}, outside {:.5}, interface {:+.5}; 2 s later: tier {:.5}, outside {:.5}",
        before.tier, before.outside, before.interface, after.tier, after.outside, after.interface, settled.tier, settled.outside
    );
    println!(
        "conserved total {:.6} -> {:.6} -> {:.6}: relocation jump {:+.3}%, drift after 2 s {:+.3}% of the 3D volume",
        before.conserved(),
        after.conserved(),
        settled.conserved(),
        100.0 * jump,
        100.0 * drift
    );

    assert!(jump.abs() < 0.03, "relocation changed the total by {:.3}%", 100.0 * jump);
    assert!(drift.abs() < 0.03, "total drifted by {:.3}% after relocation", 100.0 * drift);
}

#[test]
fn boundary_velocity_follows_the_airy_profile() {
    use liquid_ocean::dispersive::GpuDispersive;
    use liquid_wave::airy;
    use liquid_wave::dispersive::{Dispersive, DispersiveParams};

    let (device, queue) = gpu();
    let depth = 2.0;
    let k = 1.0;
    let amplitude = 0.002;
    let omega = airy::omega(k, depth, liquid_wave::GRAVITY);
    let speed = amplitude * omega;
    let spacing = 2.0 * std::f64::consts::PI / k / 64.0;
    let grid = Grid::new(64, 8, spacing, spacing);
    let mut state = State::zeros(grid.cells());

    for kk in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, kk);
            let c = grid.cell(i, kk);
            state.h[c] = depth + amplitude * (k * x).cos();
            state.hu[c] = speed / k * (k * x).cos();
        }
    }

    let model = Dispersive::new(Swe::new(grid, Bathymetry::flat(&grid, 0.0), state), DispersiveParams::default());
    let field = GpuDispersive::new(&device, &model, FRAME, swe::shared_buffers(&device, &model.swe));
    let dims = UVec3::new(16, 24, 8);
    let dx = 0.1;
    let offset = Vec3::new(3.0, 0.0, 0.0);
    let config = SolverConfig {
        dims,
        dx,
        max_particles: 1024,
        max_substeps: 1,
        narrow_band: Some(3.0),
        sparse: false,
        ..SolverConfig::default()
    };
    let mut solver = GpuSolver::new(&device, config, None);
    let b = &field.bulk.buffers;

    solver.attach_heightfield(
        &device,
        HeightfieldLink {
            offset,
            cells: [64, 8],
            cell_size: [spacing as f32; 2],
            relax_cells: 4.0,
            blend: 0.5,
            state: &field.total,
            bed: &b.bed,
            coupled: &b.coupled,
        },
    );
    solver.link_tiers(&device, TierLink { profile: &b.profile, ledger: &b.ledger });
    solver.initialize(&queue);
    solver.set_frame_dt(&queue, FRAME);

    let mut encoder = device.create_command_encoder(&Default::default());
    field.encode_frame(&mut encoder, 0);
    solver.encode_frame(&mut encoder, None);
    queue.submit([encoder.finish()]);

    let faces = (dims.x + 1) * dims.y * dims.z;
    let solid: Vec<f32> = read(&device, &queue, &solver.buffers.solid_velocity, faces as usize);
    let x = offset.x as f64;
    let row = 4;
    let mut worst: f64 = 0.0;
    let mut checked = 0;

    for j in 0..dims.y {
        let y = (j as f64 + 0.5) * dx as f64;

        if y > depth - 0.1 {
            break;
        }

        let z = y - depth;
        let got = solid[(dims.x + 1) as usize * (j + dims.y * row) as usize] as f64;
        let want = speed * airy::horizontal_profile(k, z, depth) * (k * x).cos();
        let error = (got - want).abs() / speed;
        worst = worst.max(error);
        checked += 1;

        if j % 4 == 0 {
            println!("z = {z:6.3}: boundary u {got:.5e}, Airy {want:.5e}, error {:.2}% of a*omega", 100.0 * error);
        }
    }

    println!("{checked} boundary faces checked, worst error {:.2}% of a*omega", 100.0 * worst);

    assert!(checked >= 15);
    assert!(worst < 0.03, "worst boundary velocity error {:.2}%", 100.0 * worst);
}

#[test]
fn emitted_volume_enters_the_tier_ledger() {
    let offset = Vec3::new(1.2, 0.0, 1.2);
    let plain = start(offset);
    let mut fed = start(offset);
    let region = GasRegion::cuboid(Vec3::new(0.4, 0.45, 0.4), Vec3::new(0.2, 0.06, 0.2));
    let drop = LiquidEmitter { amount: Some(0.05), ..LiquidEmitter::new(region, 293.15) };

    for run in [&plain, &fed] {
        advance(&run.device, &run.queue, &run.solver, &run.field, 60);
    }

    fed.solver.set_emitters(&fed.queue, &[drop]);

    let mut gains = Vec::new();

    for _ in 0..6 {
        for run in [&plain, &fed] {
            advance(&run.device, &run.queue, &run.solver, &run.field, 60);
        }

        let spent = read::<GpuEmissionState>(&fed.device, &fed.queue, fed.solver.emission_state(), 1)[0].total;
        let added = spent as f64 * particle_volume(&fed.solver.config) as f64;
        let alone = ledger(&plain.device, &plain.queue, &plain.field);
        let joined = ledger(&fed.device, &fed.queue, &fed.field);

        println!(
            "{spent} particles emitted for {added:.5} m^3: conserved total {:.5} with the emitter and {:.5} without, 3D tier {:.5} and {:.5}, outside {:.5} and {:.5}",
            joined.conserved(),
            alone.conserved(),
            joined.tier,
            alone.tier,
            joined.outside,
            alone.outside
        );

        gains.push((joined.conserved() - alone.conserved()) / added);
    }

    println!("gain of the conserved total over the run without the emitter, as a share of the emitted volume, each second {gains:?}");

    assert!(gains.iter().all(|g| (g - 1.0).abs() < 0.1));
}
