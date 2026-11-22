mod common;

use liquid_ocean::dispersive::GpuDispersive;
use liquid_ocean::swe;
use liquid_wave::dispersive::{Dispersive, DispersiveParams};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::swe::Swe;

fn wavy_basin() -> Dispersive {
    let grid = Grid::new(32, 16, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, z| {
        1.6 * (-((x - 11.0).powi(2) + (z - 4.0).powi(2)) / 3.0).exp() + 0.01 * x
    });
    let mut state = State::lake(&grid, &bed, 2.0);

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let c = grid.cell(i, k);
            if state.h[c] > 0.0 {
                let [x, z] = grid.center(i, k);
                let bump = 0.15 * (-((x - 4.0).powi(2) + (z - 5.0).powi(2)) / 2.0).exp();
                state.h[c] += bump;
                state.hu[c] = 0.2 * bump;
            }
        }
    }

    Dispersive::new(Swe::new(grid, bed, state), DispersiveParams::default())
}

#[cfg(windows)]
#[test]
fn dispersive_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|gpu| {
        let cpu = wavy_basin();
        GpuDispersive::new(&gpu.device, &cpu, 0.01, swe::shared_buffers(&gpu.device, &cpu.swe));
    });
}

#[test]
fn gpu_dispersive_step_matches_cpu_reference() {
    let gpu = common::gpu();
    let mut cpu = wavy_basin();
    let initial = cpu.swe.state.h.clone();
    let dt = 0.01;
    let mut solver = GpuDispersive::new(
        &gpu.device,
        &cpu,
        dt as f32,
        swe::shared_buffers(&gpu.device, &cpu.swe),
    );
    solver.write_params(&gpu.queue, dt as f32);

    let steps = 8;
    gpu.submit(|e| {
        for _ in 0..steps {
            solver.encode_frame(e, 1);
        }
    });

    for _ in 0..steps {
        let taken = cpu.step(dt);
        assert_eq!(taken, dt);
    }

    let data: Vec<[f32; 4]> = gpu.read(&solver.total);
    let mut worst: f64 = 0.0;

    for (field, name) in ["h", "hu", "hv"].iter().enumerate() {
        let got: Vec<f64> = data.iter().map(|v| v[field] as f64).collect();
        let want = match field {
            0 => &cpu.swe.state.h,
            1 => &cpu.swe.state.hu,
            _ => &cpu.swe.state.hv,
        };
        let error = common::relative_error(&got, want);
        println!("dispersive, {steps} steps: {name} relative error {error:.3e}");
        worst = worst.max(error);
    }

    let got_change: Vec<f64> = data
        .iter()
        .zip(&initial)
        .map(|(v, h)| v[0] as f64 - h)
        .collect();
    let want_change: Vec<f64> = cpu
        .swe
        .state
        .h
        .iter()
        .zip(&initial)
        .map(|(a, b)| a - b)
        .collect();
    let change = common::relative_error(&got_change, &want_change);
    println!("dispersive, {steps} steps: depth change relative error {change:.3e}");

    assert!(worst < 1e-4, "GPU deviates by {worst:e}");
}

#[test]
fn gpu_surface_profile_matches_cpu_reference() {
    use liquid_wave::airy;

    let gpu = common::gpu();
    let depth = 4.0;
    let k = 2.0 / depth;
    let grid = Grid::new(64, 4, 2.0 * std::f64::consts::PI / k / 64.0, 0.2);
    let bed = Bathymetry::flat(&grid, 0.0);
    let mut state = State::zeros(grid.cells());
    let amplitude = 0.004;
    let speed = amplitude * airy::omega(k, depth, liquid_wave::GRAVITY);

    for kk in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, kk);
            let c = grid.cell(i, kk);
            state.h[c] = depth + amplitude * (k * x).cos();
            state.hu[c] = speed / k * (k * x).cos();
        }
    }

    let cpu = Dispersive::new(Swe::new(grid, bed, state), DispersiveParams::default());
    let solver = GpuDispersive::new(&gpu.device, &cpu, 0.01, swe::shared_buffers(&gpu.device, &cpu.swe));
    gpu.submit(|e| solver.encode_frame(e, 0));

    let parts = cpu.decompose(&cpu.swe.state);
    let modes = cpu.surface_modes(&parts.surface);
    let profile: Vec<[f32; 4]> = gpu.read(&solver.bulk.buffers.profile);
    let interior: Vec<usize> = (0..grid.nz)
        .flat_map(|kk| (2..grid.nx - 2).map(move |i| grid.cell(i, kk)))
        .collect();
    let error = |field: usize, want: &dyn Fn(usize) -> f64| {
        let got: Vec<f64> = interior.iter().map(|&c| profile[c][field] as f64).collect();
        let want: Vec<f64> = interior.iter().map(|&c| want(c)).collect();
        common::relative_error(&got, &want)
    };
    let errors = [
        error(0, &|c| modes[c].k),
        error(1, &|c| modes[c].flux[0]),
        error(3, &|c| modes[c].divergence),
    ];

    println!(
        "surface profile: wavenumber error {:.3e}, flux error {:.3e}, divergence error {:.3e}, GPU k at cell 8: {} (exact {k})",
        errors[0], errors[1], errors[2], profile[8][0]
    );

    assert!(errors[0] < 1e-2, "f32 wavenumber error {:e}", errors[0]);
    assert!(errors[1] < 2e-3 && errors[2] < 2e-3);
    assert!((profile[8][0] as f64 / k - 1.0).abs() < 1e-2);
}

#[test]
fn gpu_dispersive_ledger_matches_cpu_reference() {
    let gpu = common::gpu();
    let mut cpu = wavy_basin();
    let dt = 0.01;

    cpu.swe.coupling = Some(liquid_wave::swe::CouplingRegion {
        min: [2, 2],
        max: [12, 12],
        blend: 0.5,
    });
    cpu.swe.coupled = cpu.swe.state.clone();

    let mut solver = GpuDispersive::new(&gpu.device, &cpu, dt as f32, swe::shared_buffers(&gpu.device, &cpu.swe));
    solver.write_params(&gpu.queue, dt as f32);

    let frames = 30;
    gpu.submit(|e| {
        for _ in 0..frames {
            solver.encode_frame(e, 1);
        }
    });

    for _ in 0..frames {
        assert_eq!(cpu.step(dt), dt);
    }

    let got = solver.bulk.read_ledger(&gpu.device, &gpu.queue).ledger();
    let want = cpu.swe.ledger();
    let scale = want.mass;
    let errors = [
        (got.interface - want.interface).abs() / scale,
        (got.coupled - want.coupled).abs() / scale,
        (got.outside - want.outside).abs() / scale,
        (got.tier - want.tier).abs() / scale,
    ];

    println!(
        "dispersive ledger after {frames} frames: interface cpu {:.6e} gpu {:.6e}, outside cpu {:.6} gpu {:.6}, errors {errors:?}",
        want.interface, got.interface, want.outside, got.outside
    );

    assert!(want.interface.abs() > 1e-4);
    assert!(errors.iter().all(|&e| e < 1e-5));
}

#[test]
fn gpu_dispersive_frames_are_bit_identical_across_runs() {
    let gpu = common::gpu();
    let mut cpu = wavy_basin();
    let dt = 0.01;

    cpu.swe.coupling = Some(liquid_wave::swe::CouplingRegion {
        min: [2, 2],
        max: [12, 12],
        blend: 0.5,
    });
    cpu.swe.coupled = cpu.swe.state.clone();

    let runs: Vec<Vec<u32>> = (0..2)
        .map(|_| {
            let mut solver = GpuDispersive::new(&gpu.device, &cpu, dt as f32, swe::shared_buffers(&gpu.device, &cpu.swe));
            solver.write_params(&gpu.queue, dt as f32);
            gpu.submit(|e| {
                for _ in 0..12 {
                    solver.encode_frame(e, 1);
                }
            });

            let b = &solver.bulk.buffers;

            [&solver.total, &solver.bulk_start, &solver.surface, &b.state, &b.accumulated, &b.ledger, &b.clock, &b.profile]
                .into_iter()
                .flat_map(|buffer| gpu.read::<u32>(buffer))
                .collect()
        })
        .collect();

    let differing = runs[0].iter().zip(&runs[1]).filter(|(a, b)| a != b).count();
    println!("dispersive determinism: {differing} of {} words differ between two identical runs", runs[0].len());

    assert_eq!(differing, 0);
}
