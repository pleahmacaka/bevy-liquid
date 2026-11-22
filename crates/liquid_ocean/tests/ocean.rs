mod common;

use liquid_ocean::kernel;
use liquid_ocean::ocean::{self, GpuOcean};
use liquid_wave::ocean::{Foam, Ocean, OceanParams};

fn small_ocean() -> Ocean {
    Ocean::new(OceanParams {
        resolution: 64,
        patch_sizes: vec![600.0, 110.0, 23.0],
        choppiness: 2.5,
        ..OceanParams::default()
    })
}

#[cfg(windows)]
#[test]
fn ocean_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|gpu| {
        let cpu = small_ocean();
        let surface = kernel::zeroed_storage(&gpu.device, "ocean surface", ocean::surface_bytes(&cpu));
        GpuOcean::new(&gpu.device, &cpu, surface);
    });
}

#[test]
fn gpu_ocean_matches_cpu_cascades() {
    let gpu = common::gpu();
    let cpu = small_ocean();
    let surface = kernel::zeroed_storage(&gpu.device, "ocean surface", ocean::surface_bytes(&cpu));
    let mut solver = GpuOcean::new(&gpu.device, &cpu, surface);

    let (t, dt) = (7.25, 0.5);
    solver.write_time(&gpu.queue, t as f32, dt as f32);
    gpu.submit(|e| solver.encode(e));

    let data: Vec<[f32; 4]> = gpu.read(&solver.surface);
    let n = cpu.params.resolution;
    let mut foam_seen = false;

    for (c, cascade) in cpu.cascades.iter().enumerate() {
        let fields = cpu.fields(c, t);
        let mut foam = Foam::new(n * n);
        foam.update(&fields.jacobian, dt, &cpu.params.foam);

        let pick = |slot: usize, lane: usize| -> Vec<f64> {
            (0..n * n)
                .map(|i| data[2 * (c * n * n + i) + slot][lane] as f64)
                .collect()
        };

        let checks = [
            ("eta", pick(0, 1), &fields.eta),
            ("Dx", pick(0, 0), &fields.dx),
            ("Dz", pick(0, 2), &fields.dz),
            ("J", pick(0, 3), &fields.jacobian),
            ("slope x", pick(1, 0), &fields.slope_x),
            ("slope z", pick(1, 1), &fields.slope_z),
            ("foam", pick(1, 2), &foam.value),
        ];

        let foam_max = foam.value.iter().copied().fold(0.0, f64::max);
        foam_seen |= foam_max > 0.05;
        println!("cascade L={:6.1} max foam {foam_max:.3}", cascade.patch);

        for (name, got, want) in checks {
            let error = common::relative_error(&got, want);
            println!(
                "cascade L={:6.1} {name:8}: relative error {error:.3e}",
                cascade.patch
            );
            assert!(error < 2e-4, "{name}: {error:e}");
        }
    }

    assert!(
        foam_seen,
        "the test sea state must fold somewhere so foam is exercised"
    );
}
