use bytemuck::{Pod, Zeroable};
use liquid_wave::ocean::Ocean;

use crate::fft::GpuFft;
use crate::kernel::{self, Binding, Kernel};
use crate::shaders;

pub const MAX_CASCADES: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct OceanUniform {
    pub n: u32,
    pub cascades: u32,
    pub time: f32,
    pub dt: f32,
    pub choppiness: f32,
    pub foam_threshold: f32,
    pub foam_gain: f32,
    pub foam_decay: f32,
    pub patch: [f32; 4],
}

impl OceanUniform {
    pub fn from_ocean(ocean: &Ocean) -> Self {
        let mut patch = [1.0; 4];

        for (p, c) in patch.iter_mut().zip(&ocean.cascades) {
            *p = c.patch as f32;
        }

        Self {
            n: ocean.params.resolution as u32,
            cascades: ocean.cascades.len() as u32,
            time: 0.0,
            dt: 0.0,
            choppiness: ocean.params.choppiness as f32,
            foam_threshold: ocean.params.foam.threshold as f32,
            foam_gain: ocean.params.foam.gain as f32,
            foam_decay: ocean.params.foam.decay as f32,
            patch,
        }
    }
}

pub fn surface_bytes(ocean: &Ocean) -> u64 {
    let n = ocean.params.resolution as u64;

    32 * n * n * ocean.cascades.len() as u64
}

pub fn spectrum_data(ocean: &Ocean) -> (Vec<[f32; 2]>, Vec<f32>) {
    let h0 = ocean
        .cascades
        .iter()
        .flat_map(|c| c.h0.iter().map(|h| [h.re as f32, h.im as f32]))
        .collect();
    let omega = ocean
        .cascades
        .iter()
        .flat_map(|c| c.omega.iter().map(|&w| w as f32))
        .collect();

    (h0, omega)
}

pub struct GpuOcean {
    kernel: Kernel,
    fft: GpuFft,
    group: wgpu::BindGroup,
    pub uniform: OceanUniform,
    pub params: wgpu::Buffer,
    pub h0: wgpu::Buffer,
    pub omega: wgpu::Buffer,
    pub spectrum: wgpu::Buffer,
    pub surface: wgpu::Buffer,
}

impl GpuOcean {
    pub fn new(device: &wgpu::Device, ocean: &Ocean, surface: wgpu::Buffer) -> Self {
        assert!(
            ocean.cascades.len() <= MAX_CASCADES,
            "at most {MAX_CASCADES} cascades"
        );

        let kernel = Kernel::new(
            device,
            "ocean",
            shaders::OCEAN,
            &[
                Binding::Uniform,
                Binding::ReadOnly,
                Binding::ReadOnly,
                Binding::Storage,
                Binding::Storage,
            ],
            &["evolve", "assemble"],
        );

        let uniform = OceanUniform::from_ocean(ocean);
        let n = uniform.n;
        let fields = 4 * uniform.cascades;
        let (h0, omega) = spectrum_data(ocean);

        let params = kernel::uniform(device, "ocean params", bytemuck::bytes_of(&uniform));
        let h0 = kernel::storage(device, "ocean h0", bytemuck::cast_slice(&h0));
        let omega = kernel::storage(device, "ocean omega", bytemuck::cast_slice(&omega));
        let spectrum =
            kernel::zeroed_storage(device, "ocean spectrum", 8 * (n * n * fields) as u64);
        let group = kernel.bind(
            device,
            "ocean",
            &[&params, &h0, &omega, &spectrum, &surface],
        );
        let fft = GpuFft::new(device, &spectrum, n, n, fields);

        Self {
            kernel,
            fft,
            group,
            uniform,
            params,
            h0,
            omega,
            spectrum,
            surface,
        }
    }

    pub fn write_time(&mut self, queue: &wgpu::Queue, time: f32, dt: f32) {
        self.uniform.time = time;
        self.uniform.dt = dt;
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&self.uniform));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ocean"),
            timestamp_writes: None,
        });

        let groups = [
            (self.uniform.n * self.uniform.n).div_ceil(kernel::WORKGROUP),
            self.uniform.cascades,
        ];

        self.kernel
            .dispatch_grid(&mut pass, "evolve", &self.group, groups);
        self.fft.encode(&mut pass, true);
        self.kernel
            .dispatch_grid(&mut pass, "assemble", &self.group, groups);
    }
}
