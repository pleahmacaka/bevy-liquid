mod common;

use liquid_ocean::fft::GpuFft;
use liquid_ocean::kernel;
use liquid_wave::fft::{Complex64, Fft2};

#[cfg(windows)]
#[test]
fn fft_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|gpu| {
        let buffer = kernel::zeroed_storage(&gpu.device, "fft data", 64 * 32 * 2 * 8);
        GpuFft::new(&gpu.device, &buffer, 64, 32, 2);
    });
}

#[test]
fn gpu_fft_matches_rustfft() {
    let gpu = common::gpu();
    let (width, height, fields) = (64usize, 32usize, 2usize);
    let input: Vec<[f32; 2]> = (0..width * height * fields)
        .map(|i| {
            let x = i as f32;
            [
                (0.37 * x).sin() + 0.1 * (x * 0.011).cos(),
                (0.91 * x).cos() * 0.5,
            ]
        })
        .collect();

    let buffer = kernel::storage(&gpu.device, "fft data", bytemuck::cast_slice(&input));
    let fft = GpuFft::new(
        &gpu.device,
        &buffer,
        width as u32,
        height as u32,
        fields as u32,
    );
    let cpu = Fft2::new(width, height);

    for inverse in [false, true] {
        gpu.queue
            .write_buffer(&buffer, 0, bytemuck::cast_slice(&input));
        gpu.submit(|e| {
            let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            fft.encode(&mut pass, inverse);
        });

        let output: Vec<[f32; 2]> = gpu.read(&buffer);

        for field in 0..fields {
            let range = field * width * height..(field + 1) * width * height;
            let mut expected: Vec<Complex64> = input[range.clone()]
                .iter()
                .map(|v| Complex64::new(v[0] as f64, v[1] as f64))
                .collect();

            if inverse {
                cpu.inverse_unnormalized(&mut expected);
            } else {
                cpu.forward(&mut expected);
            }

            let got: Vec<f64> = output[range]
                .iter()
                .flat_map(|v| [v[0] as f64, v[1] as f64])
                .collect();
            let want: Vec<f64> = expected.iter().flat_map(|v| [v.re, v.im]).collect();
            let error = common::relative_error(&got, &want);

            println!(
                "FFT {width}x{height} field {field} inverse={inverse}: relative error {error:.3e}"
            );
            assert!(error < 1e-5);
        }
    }
}
