use bevy::math::{UVec3, Vec3};
use bevy::prelude::{Component, Entity, Query, ResMut};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::gas::{GasRegion, GasShape};
use crate::plugin::LiquidFrame;
use crate::solver::{Bind, GpuSolver, SolverConfig, Spec};

pub const MAX_LIQUID_EMITTERS: usize = 4;

pub(crate) const EMIT_BINDINGS: [(u32, wgpu::BufferBindingType); 2] =
    [(10, wgpu::BufferBindingType::Uniform), (11, wgpu::BufferBindingType::Storage { read_only: false })];

const MAX_CELLS: u32 = 1 << 18;

const UNLIMITED: u32 = u32::MAX;

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct LiquidEmitter {
    pub region: GasRegion,

    pub velocity: Vec3,

    pub temperature: f32,

    pub flow: Option<f32>,

    pub amount: Option<f32>,
}

impl LiquidEmitter {

    pub fn new(region: GasRegion, temperature: f32) -> Self {
        Self { region, velocity: Vec3::ZERO, temperature, flow: None, amount: None }
    }

    pub fn shifted(self, offset: Vec3) -> Self {
        Self { region: self.region.shifted(offset), ..self }
    }

}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuEmitterState {
    pub mark: u32,
    pub particles: u32,
    pub budget: f32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuEmissionState {
    pub slots: [GpuEmitterState; MAX_LIQUID_EMITTERS],
    pub total: u32,
    pub pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuEmitter {
    center: [f32; 4],
    extent: [f32; 4],
    velocity: [f32; 4],
    low: [u32; 4],
    size: [u32; 4],
    limit: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuEmission {
    count: [u32; 4],
    emitters: [GpuEmitter; MAX_LIQUID_EMITTERS],
}

pub(crate) struct Emission {
    pub(crate) bind: wgpu::BindGroup,

    state: wgpu::Buffer,

    buffer: wgpu::Buffer,

    uniform: GpuEmission,

    keys: [Option<u64>; MAX_LIQUID_EMITTERS],

    marks: [u32; MAX_LIQUID_EMITTERS],

    seated: u32,
}

impl Emission {

    pub(crate) fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let uniform = GpuEmission::default();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("liquid emission"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let state = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("liquid emission state"),
            contents: bytemuck::bytes_of(&GpuEmissionState::default()),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid emission bind"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: EMIT_BINDINGS[0].0, resource: buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: EMIT_BINDINGS[1].0, resource: state.as_entire_binding() },
            ],
        });

        Self { bind, state, buffer, uniform, keys: [None; MAX_LIQUID_EMITTERS], marks: [0; MAX_LIQUID_EMITTERS], seated: 0 }
    }

    fn cells(&self) -> u32 {
        self.uniform.count[1]
    }

    fn release(&mut self, kept: &[(u64, LiquidEmitter)]) {
        for held in &mut self.keys {
            if held.is_some_and(|k| kept.iter().all(|(key, _)| *key != k)) {
                *held = None;
            }
        }
    }

    fn seat(&mut self, key: u64) -> Option<usize> {
        if let Some(slot) = self.keys.iter().position(|k| *k == Some(key)) {
            return Some(slot);
        }

        let slot = self.keys.iter().position(Option::is_none)?;
        self.seated += 1;
        self.keys[slot] = Some(key);
        self.marks[slot] = self.seated;

        Some(slot)
    }

}

pub(crate) fn specs<'a>(plan: &[(&'a str, f64)], config: &SolverConfig) -> Vec<Spec<'a>> {
    let entry = if config.static_solid { "emit_static" } else { "emit" };

    vec![("emit".to_string(), entry, Vec::new()), ("emit_grow".to_string(), "emit_grow", plan.to_vec())]
}

fn cell_box(config: &SolverConfig, region: &GasRegion) -> (UVec3, UVec3) {
    let half = match region.shape {
        GasShape::Sphere(radius) => Vec3::splat(radius),
        GasShape::Cuboid(half) => half,
    };
    let top = config.dims.as_vec3();
    let low = ((region.center - half) / config.dx).floor().clamp(Vec3::ZERO, top).as_uvec3();
    let high = ((region.center + half) / config.dx).ceil().clamp(Vec3::ZERO, top).as_uvec3();

    (low, high.saturating_sub(low))
}

pub fn particle_volume(config: &SolverConfig) -> f32 {
    config.dx.powi(3) / config.particles_per_cell as f32
}

impl GpuSolver {

    pub fn set_emitters(&mut self, queue: &wgpu::Queue, emitters: &[LiquidEmitter]) {
        let keyed: Vec<(u64, LiquidEmitter)> = emitters.iter().enumerate().map(|(slot, e)| (slot as u64, *e)).collect();

        self.set_keyed_emitters(queue, &keyed);
    }

    pub fn set_keyed_emitters(&mut self, queue: &wgpu::Queue, emitters: &[(u64, LiquidEmitter)]) {
        let emitters = &emitters[..emitters.len().min(MAX_LIQUID_EMITTERS)];
        let volume = particle_volume(&self.config);
        let mut uniform = GpuEmission::default();
        let mut cells = 0;

        self.emission.release(emitters);

        for (key, e) in emitters {
            let Some(slot) = self.emission.seat(*key) else {
                continue;
            };
            let [center, extent] = e.region.packed();
            let (low, mut size) = cell_box(&self.config, &e.region);
            let room = (MAX_CELLS - cells).checked_div(size.x * size.y).unwrap_or(0);
            size.z = size.z.min(room);

            let rate = e.flow.map_or(-1.0, |flow| flow.max(0.0) / volume);
            let amount = e.amount.map_or(UNLIMITED, |amount| (amount.max(0.0) / volume).round() as u32);

            uniform.emitters[slot] = GpuEmitter {
                center,
                extent: [extent[0], extent[1], extent[2], e.temperature],
                velocity: e.velocity.extend(rate).to_array(),
                low: [low.x, low.y, low.z, 0],
                size: [size.x, size.y, size.z, size.element_product()],
                limit: [amount, self.emission.marks[slot], 0, 0],
            };
            uniform.count[0] = uniform.count[0].max(slot as u32 + 1);
            cells += size.element_product();
        }

        uniform.count[1] = cells;

        if bytemuck::bytes_of(&uniform) != bytemuck::bytes_of(&self.emission.uniform) {
            self.emission.uniform = uniform;
            queue.write_buffer(&self.emission.buffer, 0, bytemuck::bytes_of(&uniform));
        }
    }

    pub fn emits(&self) -> bool {
        self.emission.cells() > 0
    }

    pub fn emission_state(&self) -> &wgpu::Buffer {
        &self.emission.state
    }

    pub(crate) fn encode_emission(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.emits() {
            self.dispatch_direct(pass, "emit", Bind::Sim, 1);
            self.dispatch_direct(pass, "emit_grow", Bind::Plan, 1);
        }
    }

    pub fn encode_emitting_sort(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.encode_sorted(pass, "sort_copy", true);
    }

}

pub(crate) fn collect_emitters(emitters: Query<(Entity, &LiquidEmitter)>, mut frame: ResMut<LiquidFrame>) {
    let origin = frame.origin;
    let mut list: Vec<_> = emitters.iter().collect();
    list.sort_by_key(|(e, _)| *e);
    list.truncate(MAX_LIQUID_EMITTERS);

    frame.emitters = list.into_iter().map(|(entity, e)| (entity.to_bits(), e.shifted(-origin))).collect();
}
