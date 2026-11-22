mod common;

use liquid_ocean::swe::{self, GpuSwe};
use liquid_wave::GRAVITY;
use liquid_wave::airy::{self, AmbientField, WaveComponent};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::sponge::{AmbientFrame, Sponge};
use liquid_wave::swe::Swe;

fn ambient_basin() -> Swe {
    let grid = Grid::new(64, 16, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, z| 0.01 * x + 0.05 * (0.4 * z).sin());
    let field = AmbientField::new(
        vec![
            WaveComponent {
                k: [0.6, 0.2],
                amplitude: 0.02,
                phase: 0.3,
                omega: airy::omega(0.632, 2.0, GRAVITY),
            },
            WaveComponent {
                k: [0.3, -0.4],
                amplitude: 0.01,
                phase: 1.1,
                omega: airy::omega(0.5, 2.0, GRAVITY),
            },
        ],
        2.0,
    );
    let frame = AmbientFrame {
        origin: [-10.0, 4.0],
        level: 2.0,
    };
    let state = State::lake(&grid, &bed, 2.0);
    let mut swe = Swe::new(grid, bed, state.clone());
    let mut sponge = Sponge::new(&grid, [4.0, 6.0, 2.0, 0.0], 3.0, 2.0, state);
    sponge.ambient = Some((frame, field));
    swe.sponge = Some(sponge);

    swe
}

#[cfg(windows)]
#[test]
fn sponge_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|gpu| {
        let cpu = ambient_basin();
        GpuSwe::new(&gpu.device, &cpu, 0.01, swe::shared_buffers(&gpu.device, &cpu));
    });
}

#[test]
fn gpu_sponge_tracks_the_ambient_field() {
    let gpu = common::gpu();
    let mut cpu = ambient_basin();
    let dt = 0.01;
    let mut solver = GpuSwe::new(
        &gpu.device,
        &cpu,
        dt as f32,
        swe::shared_buffers(&gpu.device, &cpu),
    );
    solver.write_params(&gpu.queue, dt as f32);

    gpu.submit(|e| {
        for _ in 0..40 {
            solver.encode_frame(e, 1);
        }
    });

    for _ in 0..40 {
        assert_eq!(cpu.step(dt), dt);
    }

    let target: Vec<[f32; 4]> = gpu.read(&solver.buffers.sponge);
    let sponge = cpu.sponge.as_ref().expect("sponge");
    let got: Vec<f64> = target
        .iter()
        .flat_map(|v| [v[0] as f64, v[1] as f64, v[2] as f64])
        .collect();
    let want: Vec<f64> = (0..cpu.grid.cells())
        .flat_map(|c| {
            if sponge.sigma[c] > 0.0 {
                [sponge.target.h[c], sponge.target.hu[c], sponge.target.hv[c]]
            } else {
                [
                    target[c][0] as f64,
                    target[c][1] as f64,
                    target[c][2] as f64,
                ]
            }
        })
        .collect();
    let target_error = common::relative_error(&got, &want);

    let state: Vec<[f32; 4]> = gpu.read(&solver.buffers.state);
    let state_error = ["h", "hu", "hv"]
        .iter()
        .enumerate()
        .map(|(field, name)| {
            let got: Vec<f64> = state.iter().map(|v| v[field] as f64).collect();
            let want = [&cpu.state.h, &cpu.state.hu, &cpu.state.hv][field];
            let error = common::relative_error(&got, want);
            println!("ambient sponge, 40 steps: {name} relative error {error:.3e}");
            error
        })
        .fold(0.0, f64::max);

    println!("ambient sponge target relative error {target_error:.3e}");
    assert!(target_error < 1e-4);
    assert!(state_error < 1e-4);
}
