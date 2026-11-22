mod common;

use liquid_ocean::ambient::{self, GpuAmbient};
use liquid_ocean::kernel::{self, Binding, Kernel};
use liquid_wave::ocean::{Ocean, OceanParams};

fn small_ocean() -> Ocean {
    Ocean::new(OceanParams {
        resolution: 64,
        patch_sizes: vec![600.0, 110.0, 23.0],
        choppiness: 2.5,
        ..OceanParams::default()
    })
}

fn evaluator(device: &wgpu::Device) -> Kernel {
    let source = format!(
        "@group(0) @binding(2) var<storage, read> points: array<vec4<f32>>;\n\
         @group(0) @binding(3) var<storage, read_write> samples: array<vec4<f32>>;\n\
         @compute @workgroup_size(64)\n\
         fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>) {{\n\
             let s = ambient_sample(points[gid.x].xyz, ambient.time);\n\
             samples[gid.x] = vec4<f32>(s.velocity, s.elevation);\n\
         }}\n{}",
        ambient::source(0, 0)
    );

    Kernel::new(
        device,
        "ambient test",
        &source,
        &[
            Binding::Uniform,
            Binding::ReadOnly,
            Binding::ReadOnly,
            Binding::Storage,
        ],
        &["evaluate"],
    )
}

#[cfg(windows)]
#[test]
fn ambient_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|gpu| {
        evaluator(&gpu.device);
    });
}

#[test]
fn gpu_ambient_matches_cpu_airy_field() {
    let gpu = common::gpu();
    let field = small_ocean().ambient().strongest(96);
    let mut ambient = GpuAmbient::new(&gpu.device, &field);
    let t = 3.3;
    ambient.write_time(&gpu.queue, t as f32);

    let points: Vec<[f32; 4]> = (0..64)
        .map(|i| {
            let s = i as f32;
            [
                7.0 * s.sin() * 3.0,
                -0.1 - 0.4 * (s % 13.0),
                5.0 * (0.7 * s).cos() * 4.0,
                0.0,
            ]
        })
        .collect();

    let kernel = evaluator(&gpu.device);
    let points_buffer = kernel::storage(&gpu.device, "points", bytemuck::cast_slice(&points));
    let samples = kernel::zeroed_storage(&gpu.device, "samples", 16 * points.len() as u64);
    let group = kernel.bind(
        &gpu.device,
        "ambient",
        &[
            &ambient.params,
            &ambient.components,
            &points_buffer,
            &samples,
        ],
    );

    gpu.submit(|e| {
        let mut pass = e.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
        kernel.dispatch(&mut pass, "evaluate", &group, points.len() as u32);
    });

    let got: Vec<[f32; 4]> = gpu.read(&samples);
    let mut gpu_values = Vec::new();
    let mut cpu_values = Vec::new();

    for (p, g) in points.iter().zip(&got) {
        let s = field.sample(p[0] as f64, p[1] as f64, p[2] as f64, t);
        cpu_values.extend([s.velocity[0], s.velocity[1], s.velocity[2], s.elevation]);
        gpu_values.extend(g.iter().map(|&v| v as f64));
    }

    let error = common::relative_error(&gpu_values, &cpu_values);
    println!("ambient Airy field (96 components, 64 points): relative error {error:.3e}");
    assert!(error < 1e-4);
}
