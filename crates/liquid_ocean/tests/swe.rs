mod common;

use liquid_ocean::swe::{self, GpuSwe};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::sponge::Sponge;
use liquid_wave::swe::{Boundary, CouplingRegion, Side, Swe};

fn dam_break_over_island() -> Swe {
    let grid = Grid::new(40, 28, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, z| {
        0.03 * x
            + 1.4 * (-((x - 12.0).powi(2) + (z - 7.0).powi(2)) / 5.0).exp()
            + 0.2 * (0.6 * z).sin()
    });
    let mut state = State::lake(&grid, &bed, 1.0);

    for k in 0..grid.nz {
        for i in 0..8 {
            state.h[grid.cell(i, k)] += 1.2;
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.boundaries = swe
        .boundaries
        .with(Side::East, Boundary::Transmissive)
        .with(Side::North, Boundary::Dirichlet);

    for i in 0..grid.nx {
        let b = swe.bathymetry.face_z[grid.z_face(i, grid.nz)];
        swe.boundaries.dirichlet[Side::North.index()][i] = [(1.1 - b).max(0.0), 0.0, -0.2];
    }

    swe
}

fn compare(gpu: &common::Gpu, cpu: &Swe, buffer: &wgpu::Buffer, label: &str) -> f64 {
    let data: Vec<[f32; 4]> = gpu.read(buffer);
    let mut worst: f64 = 0.0;

    for (field, name) in ["h", "hu", "hv"].iter().enumerate() {
        let got: Vec<f64> = data.iter().map(|v| v[field] as f64).collect();
        let want = match field {
            0 => &cpu.state.h,
            1 => &cpu.state.hu,
            _ => &cpu.state.hv,
        };
        let error = common::relative_error(&got, want);
        println!("{label}: {name} relative error {error:.3e}");
        worst = worst.max(error);
    }

    worst
}

#[cfg(windows)]
#[test]
fn swe_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|gpu| {
        let cpu = dam_break_over_island();
        GpuSwe::new(&gpu.device, &cpu, 0.004, swe::shared_buffers(&gpu.device, &cpu));
    });
}

#[test]
fn gpu_kp07_matches_cpu_reference() {
    let gpu = common::gpu();
    let mut cpu = dam_break_over_island();
    let dt = 0.004;
    let mut solver = GpuSwe::new(
        &gpu.device,
        &cpu,
        dt as f32,
        swe::shared_buffers(&gpu.device, &cpu),
    );
    solver.write_params(&gpu.queue, dt as f32);

    let steps = 25;
    gpu.submit(|e| {
        for _ in 0..steps {
            solver.encode_frame(e, 1);
        }
    });

    for _ in 0..steps {
        let taken = cpu.step(dt);
        assert_eq!(taken, dt, "the CPU reference must take the same fixed step");
    }

    let error = compare(
        &gpu,
        &cpu,
        &solver.buffers.state,
        "KP07 dam break over island, 25 steps",
    );
    assert!(error < 1e-4, "GPU deviates from CPU by {error:e}");
}

#[test]
fn gpu_manning_friction_matches_cpu_reference() {
    let gpu = common::gpu();
    let mut cpu = dam_break_over_island();
    cpu.params.manning = 0.03;
    let dt = 0.004;
    let mut solver = GpuSwe::new(
        &gpu.device,
        &cpu,
        dt as f32,
        swe::shared_buffers(&gpu.device, &cpu),
    );
    solver.write_params(&gpu.queue, dt as f32);

    gpu.submit(|e| {
        for _ in 0..25 {
            solver.encode_frame(e, 1);
        }
    });

    for _ in 0..25 {
        cpu.step(dt);
    }

    let error = compare(
        &gpu,
        &cpu,
        &solver.buffers.state,
        "KP07 + Manning friction, 25 steps",
    );
    assert!(error < 1e-4, "GPU deviates from CPU by {error:e}");
}

#[test]
fn gpu_cfl_step_matches_cpu_reference() {
    let gpu = common::gpu();
    let mut cpu = dam_break_over_island();
    let mut solver = GpuSwe::new(
        &gpu.device,
        &cpu,
        1.0,
        swe::shared_buffers(&gpu.device, &cpu),
    );
    solver.write_params(&gpu.queue, 0.5);
    gpu.submit(|e| solver.encode_frame(e, 1));

    let dt = cpu.step(1.0);
    let clock: Vec<swe::Clock> = gpu.read(&solver.buffers.clock);

    println!("CFL step: CPU dt {dt:.6e}, GPU dt {:.6e}", clock[0].dt);
    assert!((clock[0].dt as f64 / dt - 1.0).abs() < 1e-5);
}

#[test]
fn gpu_finish_applies_sources_coupling_and_sponge() {
    let gpu = common::gpu();
    let grid = Grid::new(24, 20, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, _| 0.02 * x);
    let rest = State::lake(&grid, &bed, 1.0);
    let mut state = rest.clone();

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, z] = grid.center(i, k);
            state.h[grid.cell(i, k)] += 0.05 * (0.8 * x).sin() * (0.5 * z).cos();
        }
    }

    let mut cpu = Swe::new(grid, bed, state);
    cpu.sponge = Some(Sponge::new(&grid, [3.0, 0.0, 0.0, 2.0], 5.0, 2.0, rest));
    cpu.coupling = Some(CouplingRegion {
        min: [5, 5],
        max: [11, 9],
        blend: 0.5,
    });
    cpu.coupled.h.fill(1.3);
    cpu.coupled.hu.fill(0.2);

    let dt = 0.01;
    let mut solver = GpuSwe::new(
        &gpu.device,
        &cpu,
        dt as f32,
        swe::shared_buffers(&gpu.device, &cpu),
    );
    solver.write_params(&gpu.queue, dt as f32);

    let mut sources = vec![[0.0f32; 4]; grid.cells()];

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let c = grid.cell(i, k);
            let value = 0.01 * ((i + 2 * k) as f64 * 0.3).sin();
            cpu.source.h[c] = value;
            cpu.source.hv[c] = 0.5 * value;
            sources[c] = [value as f32, 0.0, 0.5 * value as f32, 0.0];
        }
    }

    gpu.queue
        .write_buffer(&solver.buffers.source, 0, bytemuck::cast_slice(&sources));
    gpu.submit(|e| solver.encode_frame(e, 1));
    cpu.step(dt);

    let error = compare(
        &gpu,
        &cpu,
        &solver.buffers.state,
        "finish (sources + coupling + sponge)",
    );
    let cleared: Vec<[f32; 4]> = gpu.read(&solver.buffers.source);

    assert!(error < 1e-4);
    assert!(
        cleared.iter().all(|v| v[0] == 0.0 && v[2] == 0.0),
        "source buffer must be consumed"
    );
}

fn ledger_basin() -> Swe {
    let grid = Grid::new(40, 32, 0.25, 0.25);
    let bed = Bathymetry::from_fn(&grid, |x, z| 0.02 * x + 0.1 * (0.7 * z).sin());
    let mut state = State::lake(&grid, &bed, 0.8);

    for k in 0..grid.nz {
        for i in 0..8 {
            state.h[grid.cell(i, k)] += 0.4;
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.coupling = Some(CouplingRegion {
        min: [14, 10],
        max: [26, 22],
        blend: 0.5,
    });

    for c in 0..grid.cells() {
        swe.coupled.h[c] = swe.state.h[c] + 0.03;
        swe.coupled.hu[c] = 0.05;
    }

    swe
}

fn outside_sources(swe: &Swe, frame: usize) -> Vec<[f32; 4]> {
    let g = swe.grid;
    let mut sources = vec![[0.0f32; 4]; g.cells()];

    for k in 0..g.nz {
        for i in 0..g.nx {
            if !swe.coupling.is_some_and(|r| r.contains(i, k)) {
                let value = 2e-4 * (0.3 * i as f64 + 0.5 * k as f64 + 0.07 * frame as f64).sin();
                sources[g.cell(i, k)] = [value as f32, 0.0, 0.0, 0.0];
            }
        }
    }

    sources
}

fn print_ledger(label: &str, cpu: &liquid_wave::swe::Ledger, gpu: &liquid_wave::swe::Ledger) {
    println!(
        "{label}: interface cpu {:.6e} gpu {:.6e} | injected cpu {:.6e} gpu {:.6e} | coupled cpu {:.6e} gpu {:.6e} | outside cpu {:.6} gpu {:.6} | tier cpu {:.6} gpu {:.6}",
        cpu.interface, gpu.interface, cpu.injected, gpu.injected, cpu.coupled, gpu.coupled, cpu.outside, gpu.outside, cpu.tier, gpu.tier
    );
}

fn ledger_error(cpu: &liquid_wave::swe::Ledger, gpu: &liquid_wave::swe::Ledger, scale: f64) -> f64 {
    [
        cpu.interface - gpu.interface,
        cpu.injected - gpu.injected,
        cpu.coupled - gpu.coupled,
        cpu.mass - gpu.mass,
        cpu.outside - gpu.outside,
        cpu.tier - gpu.tier,
    ]
    .iter()
    .map(|d| d.abs() / scale)
    .fold(0.0, f64::max)
}

#[test]
fn gpu_ledger_matches_cpu_ledger() {
    let gpu = common::gpu();
    let mut cpu = ledger_basin();
    let dt = 0.005;
    let mut solver = GpuSwe::new(&gpu.device, &cpu, dt as f32, swe::shared_buffers(&gpu.device, &cpu));
    solver.write_params(&gpu.queue, dt as f32);

    let start = cpu.ledger();
    let invariant = |l: &liquid_wave::swe::Ledger| l.outside + l.interface - l.injected;
    let mut worst_drift: f64 = 0.0;
    let frames = 240;

    for frame in 0..frames {
        let sources = outside_sources(&cpu, frame);

        for (c, s) in sources.iter().enumerate() {
            cpu.source.h[c] = s[0] as f64;
        }

        gpu.queue.write_buffer(&solver.buffers.source, 0, bytemuck::cast_slice(&sources));
        gpu.submit(|e| solver.encode_frame(e, 1));
        cpu.step(dt);

        if frame % 60 == 59 {
            let ledger = solver.read_ledger(&gpu.device, &gpu.queue).ledger();
            worst_drift = worst_drift.max((invariant(&ledger) - invariant(&start)).abs() / start.mass);
        }
    }

    let read = solver.read_ledger(&gpu.device, &gpu.queue);
    let (got, want) = (read.ledger(), cpu.ledger());
    let error = ledger_error(&want, &got, want.mass);

    print_ledger("ledger after 240 frames", &want, &got);
    println!(
        "GPU frames {}, CPU/GPU ledger error {error:.3e} (of total mass), GPU outside + interface - injected drift {worst_drift:.3e}",
        read.frames()
    );

    assert_eq!(read.frames(), frames as u32);
    assert!(want.interface.abs() > 1e-3);
    assert!(error < 1e-5, "CPU and GPU ledgers differ by {error:e}");
    assert!(worst_drift < 1e-5, "GPU flux ledger drifted by {worst_drift:e}");
}

#[test]
fn gpu_relocation_matches_cpu_ledger() {
    let gpu = common::gpu();
    let mut cpu = ledger_basin();
    let dt = 0.005;
    let mut solver = GpuSwe::new(&gpu.device, &cpu, dt as f32, swe::shared_buffers(&gpu.device, &cpu));
    solver.write_params(&gpu.queue, dt as f32);

    for _ in 0..40 {
        gpu.submit(|e| solver.encode_frame(e, 1));
        cpu.step(dt);
    }

    let before = cpu.ledger();
    let target = CouplingRegion {
        min: [4, 2],
        max: [16, 14],
        blend: 0.5,
    };

    cpu.relocate(Some(target));
    solver.set_coupling(Some(target));
    solver.write_params(&gpu.queue, dt as f32);

    for _ in 0..40 {
        gpu.submit(|e| solver.encode_frame(e, 1));
        cpu.step(dt);
    }

    let (got, want) = (solver.read_ledger(&gpu.device, &gpu.queue).ledger(), cpu.ledger());
    let error = ledger_error(&want, &got, want.mass);

    print_ledger("ledger 40 frames after relocation", &want, &got);
    println!(
        "interface before relocation {:.6e}, CPU/GPU ledger error {error:.3e}",
        before.interface
    );

    assert!(error < 1e-5, "CPU and GPU ledgers differ by {error:e}");
    assert!(
        compare(&gpu, &cpu, &solver.buffers.state, "state after relocation") < 1e-4
    );
}
