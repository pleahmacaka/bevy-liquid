use bytemuck::{Pod, Zeroable};
use liquid_wave::airy::AmbientField;

use crate::kernel;
use crate::shaders;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct AmbientParams {
    pub count: u32,
    pub depth: f32,
    pub time: f32,
    pub pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct AmbientComponent {
    pub k: [f32; 2],
    pub amplitude: f32,
    pub phase: f32,
    pub omega: f32,
    pub pad: [f32; 3],
}

pub fn source(group: u32, binding: u32) -> String {
    format!(
        "@group({group}) @binding({binding}) var<uniform> ambient: AmbientParams;\n\
         @group({group}) @binding({next}) var<storage, read> ambient_components: array<AmbientComponent>;\n{body}",
        next = binding + 1,
        body = shaders::AMBIENT,
    )
}

pub fn components(field: &AmbientField) -> Vec<AmbientComponent> {
    field
        .components
        .iter()
        .map(|c| AmbientComponent {
            k: [c.k[0] as f32, c.k[1] as f32],
            amplitude: c.amplitude as f32,
            phase: c.phase as f32,
            omega: c.omega as f32,
            pad: [0.0; 3],
        })
        .collect()
}

pub struct GpuAmbient {
    pub params: wgpu::Buffer,
    pub components: wgpu::Buffer,
    pub uniform: AmbientParams,
}

impl GpuAmbient {
    pub fn new(device: &wgpu::Device, field: &AmbientField) -> Self {
        let list = components(field);
        let uniform = AmbientParams {
            count: list.len() as u32,
            depth: field.depth as f32,
            time: 0.0,
            pad: 0.0,
        };
        let padded = if list.is_empty() {
            vec![AmbientComponent::default()]
        } else {
            list
        };

        Self {
            params: kernel::uniform(device, "ambient params", bytemuck::bytes_of(&uniform)),
            components: kernel::storage(
                device,
                "ambient components",
                bytemuck::cast_slice(&padded),
            ),
            uniform,
        }
    }

    pub fn write_time(&mut self, queue: &wgpu::Queue, time: f32) {
        self.uniform.time = time;
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.uniform));
    }
}
