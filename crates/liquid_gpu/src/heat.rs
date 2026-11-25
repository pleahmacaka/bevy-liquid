use bevy::math::Vec3;
use bevy::prelude::Component;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gas::{GAS_MASS_QUANTUM, GasInjection, GasRegion, GasSolver};
use crate::solver::{GpuSolver, SolverConfig};

pub const MAX_HEAT_SOURCES: usize = 8;

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct HeatSource {
    pub region: GasRegion,

    pub temperature: f32,

    pub conductance: f32,
}

impl HeatSource {

    pub fn shifted(self, offset: Vec3) -> Self {
        Self { region: self.region.shifted(offset), ..self }
    }

}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuHeatSource {
    center: [f32; 4],
    extent: [f32; 4],
    heat: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuHeating {
    shift: [f32; 4],
    cell: [f32; 4],
    air: [f32; 4],
    gas: [u32; 4],
    count: [u32; 4],
    sources: [GpuHeatSource; MAX_HEAT_SOURCES],
}

pub(crate) struct HeatLink {
    pub(crate) bind: wgpu::BindGroup,

    buffer: wgpu::Buffer,

    uniform: GpuHeating,

    warmed: bool,
}

impl HeatLink {

    pub(crate) fn alone(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, config: &SolverConfig, warmed: bool) -> Self {
        let stand_in = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })
        };
        let injections = stand_in("liquid heat no injections", 16 + std::mem::size_of::<GasInjection>() as u64);
        let flags = stand_in("liquid heat no gas flags", 4);
        let uniform = GpuHeating { air: [config.ambient_temperature, config.cooling, config.boiling_point, 0.0], ..Default::default() };

        Self::bound(device, layout, uniform, &injections, &flags, warmed)
    }

    fn bound(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform: GpuHeating,
        injections: &wgpu::Buffer,
        flags: &wgpu::Buffer,
        warmed: bool,
    ) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("liquid heat"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid heat bind"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 7, resource: buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 8, resource: injections.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: flags.as_entire_binding() },
            ],
        });

        Self { bind, buffer, uniform, warmed }
    }

}

impl GpuSolver {

    pub fn particle_mass(&self) -> f32 {
        let c = &self.config;
        let mass = c.density as f64 * (c.dx as f64).powi(3) / c.particles_per_cell as f64;

        ((mass / GAS_MASS_QUANTUM).round() * GAS_MASS_QUANTUM) as f32
    }

    pub fn steam_volume(&self) -> f32 {
        self.particle_mass() / self.config.density * self.config.expansion
    }

    pub fn attach_gas(&mut self, device: &wgpu::Device, gas: &GasSolver) {
        let g = &gas.config;
        let c = &self.config;
        let mass = self.particle_mass();

        assert!(g.injection_capacity > 0, "liquid boiling needs a gas domain with injection capacity");
        assert!(mass > 0.0 && (mass as f64 / GAS_MASS_QUANTUM) < u32::MAX as f64, "liquid particle mass {mass} kg does not fit the gas mass quantum");

        let uniform = GpuHeating {
            shift: [0.0, 0.0, 0.0, mass],
            cell: [g.dx, 1.0 / g.dx, self.steam_volume(), 0.0],
            air: [g.params.ambient_temperature, c.cooling, c.boiling_point + 2.0 * c.latent_heat, 0.0],
            gas: [g.dims.x, g.dims.y, g.dims.z, g.injection_capacity],
            ..Default::default()
        };

        self.heat = HeatLink::bound(device, &self.heat_layout, uniform, &gas.buffers.injections, &gas.buffers.flags, self.heat.warmed);
    }

    pub fn detach_gas(&mut self, device: &wgpu::Device) {
        self.heat = HeatLink::alone(device, &self.heat_layout, &self.config, self.heat.warmed);
    }

    pub fn set_heat(&mut self, queue: &wgpu::Queue, sources: &[HeatSource], offset: Vec3) {
        let link = &mut self.heat;
        let mut uniform = link.uniform;
        uniform.shift = (-offset).extend(uniform.shift[3]).to_array();
        uniform.count[0] = sources.len().min(MAX_HEAT_SOURCES) as u32;
        uniform.sources = [GpuHeatSource::default(); MAX_HEAT_SOURCES];

        for (slot, s) in uniform.sources.iter_mut().zip(sources) {
            let [center, extent] = s.region.packed();
            *slot = GpuHeatSource { center, extent, heat: [s.temperature, s.conductance, 0.0, 0.0] };
        }

        link.warmed |= uniform.count[0] > 0;

        if bytemuck::bytes_of(&uniform) != bytemuck::bytes_of(&link.uniform) {
            link.uniform = uniform;
            queue.write_buffer(&link.buffer, 0, bytemuck::bytes_of(&uniform));
        }
    }

    pub fn heats(&self) -> bool {
        self.heat.warmed
    }

}
