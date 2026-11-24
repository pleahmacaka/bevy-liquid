use std::f64::consts::TAU;

use bytemuck::{Pod, Zeroable};

use crate::kernel::{self, Binding, Kernel};
use crate::shaders;

pub const MAX_LENGTH: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FftParams {
    pub n: u32,
    pub log2n: u32,
    pub stride: u32,
    pub line_stride: u32,
    pub field_stride: u32,
    pub inverse: u32,
    pub twiddle_stride: u32,
    pub pad: u32,
}

struct Pass {
    group: wgpu::BindGroup,
    workgroups: [u32; 2],
}

pub fn twiddles(longest: u32) -> Vec<[f32; 2]> {
    (0..longest / 2)
        .map(|j| {
            let angle = -TAU * j as f64 / longest as f64;
            [angle.cos() as f32, angle.sin() as f32]
        })
        .collect()
}

pub struct GpuFft {
    kernel: Kernel,
    rows: [Pass; 2],
    columns: [Pass; 2],
    pub width: u32,
    pub height: u32,
    pub fields: u32,
}

impl GpuFft {
    pub fn new(
        device: &wgpu::Device,
        data: &wgpu::Buffer,
        width: u32,
        height: u32,
        fields: u32,
    ) -> Self {
        assert!(
            width.is_power_of_two() && height.is_power_of_two(),
            "FFT sizes must be powers of two"
        );
        assert!(
            width <= MAX_LENGTH && height <= MAX_LENGTH,
            "FFT lines are limited to {MAX_LENGTH} points"
        );

        let kernel = Kernel::new(
            device,
            "fft",
            shaders::FFT,
            &[Binding::Uniform, Binding::Storage, Binding::ReadOnly],
            &["fft_lines"],
        );

        let longest = width.max(height).max(2);
        let twiddles = kernel::storage(device, "fft twiddles", bytemuck::cast_slice(&twiddles(longest)));

        let make = |n: u32, stride: u32, line_stride: u32, lines: u32, inverse: bool| {
            let params = FftParams {
                n,
                log2n: n.trailing_zeros(),
                stride,
                line_stride,
                field_stride: width * height,
                inverse: inverse as u32,
                twiddle_stride: longest / n,
                pad: 0,
            };
            let buffer = kernel::uniform(device, "fft params", bytemuck::bytes_of(&params));
            let group = kernel.bind(device, "fft", &[&buffer, data, &twiddles]);

            Pass {
                group,
                workgroups: [lines, fields],
            }
        };

        let rows = [
            make(width, 1, width, height, false),
            make(width, 1, width, height, true),
        ];
        let columns = [
            make(height, width, 1, width, false),
            make(height, width, 1, width, true),
        ];

        Self {
            kernel,
            rows,
            columns,
            width,
            height,
            fields,
        }
    }

    pub fn encode(&self, pass: &mut wgpu::ComputePass, inverse: bool) {
        let direction = inverse as usize;

        for stage in [&self.rows[direction], &self.columns[direction]] {
            self.kernel
                .dispatch_grid(pass, "fft_lines", &stage.group, stage.workgroups);
        }
    }
}
