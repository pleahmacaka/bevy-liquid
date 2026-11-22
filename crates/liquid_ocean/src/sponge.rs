use bytemuck::{Pod, Zeroable};
use liquid_wave::airy::AmbientField;
use liquid_wave::grid::Grid;
use liquid_wave::sponge::AmbientFrame;

use crate::ambient::{self, GpuAmbient};
use crate::kernel::{self, Binding, Kernel};
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct TrackParams {
    pub nx: u32,
    pub nz: u32,
    pub dx: f32,
    pub dz: f32,
    pub origin: [f32; 2],
    pub level: f32,
    pub pad: f32,
}

pub struct SpongeBuffers<'a> {
    pub sponge: &'a wgpu::Buffer,
    pub bed: &'a wgpu::Buffer,
    pub clock: &'a wgpu::Buffer,
}

pub struct SpongeTracker {
    kernel: Kernel,
    group: wgpu::BindGroup,
    pub ambient: GpuAmbient,
    pub params: TrackParams,
    cells: u32,
}

impl SpongeTracker {
    pub fn new(
        device: &wgpu::Device,
        grid: &Grid,
        frame: &AmbientFrame,
        field: &AmbientField,
        buffers: SpongeBuffers,
    ) -> Self {
        let source = format!(
            "{}{}{}",
            shaders::SPONGE,
            shaders::BED,
            ambient::source(0, 1)
        );
        let kernel = Kernel::new(
            device,
            "sponge tracker",
            &source,
            &[
                Binding::Uniform,
                Binding::Uniform,
                Binding::ReadOnly,
                Binding::Storage,
                Binding::ReadOnly,
                Binding::ReadOnly,
            ],
            &["track_ambient"],
        );

        let params = TrackParams {
            nx: grid.nx as u32,
            nz: grid.nz as u32,
            dx: grid.dx as f32,
            dz: grid.dz as f32,
            origin: [frame.origin[0] as f32, frame.origin[1] as f32],
            level: frame.level as f32,
            pad: 0.0,
        };
        let uniform = kernel::uniform(device, "sponge tracker", bytemuck::bytes_of(&params));
        let ambient = GpuAmbient::new(device, field);
        let group = kernel.bind(
            device,
            "sponge tracker",
            &[
                &uniform,
                &ambient.params,
                &ambient.components,
                buffers.sponge,
                buffers.bed,
                buffers.clock,
            ],
        );

        Self {
            kernel,
            group,
            ambient,
            params,
            cells: grid.cells() as u32,
        }
    }

    pub fn encode(&self, pass: &mut wgpu::ComputePass) {
        self.kernel
            .dispatch(pass, "track_ambient", &self.group, self.cells);
    }
}
