use bytemuck::{Pod, Zeroable};
use liquid_wave::grid::Grid;
use liquid_wave::promotion::{Body, Criteria, Promotion};

use crate::kernel::{self, Binding, Kernel};
use crate::shaders;

pub const MAX_BODIES: usize = 8;
pub const RESULT_SLOTS: usize = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PromotionParams {
    pub nx: u32,
    pub nz: u32,
    pub window: [u32; 2],
    pub dx: f32,
    pub dz: f32,
    pub gravity: f32,
    pub dt: f32,
    pub breaking: f32,
    pub acceleration: f32,
    pub slope: f32,
    pub dry_depth: f32,
    pub body_count: u32,
    pub history: u32,
    pub pad: [u32; 2],
    pub bodies: [[f32; 4]; MAX_BODIES],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuPromotionResult {
    pub slots: [u32; RESULT_SLOTS],
}

impl GpuPromotionResult {
    pub fn promotion(&self) -> Option<Promotion> {
        let s = &self.slots;

        (s[10] != 0).then(|| Promotion {
            flagged: s[0],
            bounds: [s[1], s[2], s[3], s[4]],
            window: [s[5], s[6], s[7], s[8]],
            weight: s[9],
        })
    }
}

const BINDINGS: [Binding; 8] = [
    Binding::Uniform,
    Binding::ReadOnly,
    Binding::ReadOnly,
    Binding::ReadOnly,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
];

const ENTRIES: [&str; 5] = [
    "promote_vertical",
    "promote_cells",
    "promote_rows",
    "promote_columns",
    "promote_window",
];

pub struct PromotionInputs<'a> {
    pub state: &'a wgpu::Buffer,
    pub bed: &'a wgpu::Buffer,
    pub profile: &'a wgpu::Buffer,
}

pub struct GpuPromotion {
    kernel: Kernel,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    pub params: PromotionParams,
    pub indicators: wgpu::Buffer,
    pub result: wgpu::Buffer,
    cells: u32,
    frames: u32,
}

impl GpuPromotion {
    pub fn new(device: &wgpu::Device, grid: &Grid, gravity: f64, criteria: &Criteria, inputs: PromotionInputs) -> Self {
        let kernel = Kernel::new(device, "promotion", shaders::PROMOTION, &BINDINGS, &ENTRIES);
        let cells = grid.cells() as u32;
        let params = PromotionParams {
            nx: grid.nx as u32,
            nz: grid.nz as u32,
            window: [grid.nx as u32, grid.nz as u32],
            dx: grid.dx as f32,
            dz: grid.dz as f32,
            gravity: gravity as f32,
            dt: 1.0,
            breaking: criteria.breaking as f32,
            acceleration: criteria.acceleration as f32,
            slope: criteria.slope as f32,
            dry_depth: criteria.dry_depth as f32,
            body_count: 0,
            history: 0,
            pad: [0; 2],
            bodies: [[0.0; 4]; MAX_BODIES],
        };
        let uniform = kernel::uniform(device, "promotion params", bytemuck::bytes_of(&params));
        let history = kernel::zeroed_storage(device, "promotion history", 16 * cells as u64);
        let indicators = kernel::zeroed_storage(device, "promotion indicators", 16 * cells as u64);
        let table = kernel::zeroed_storage(device, "promotion table", 4 * ((grid.nx + 1) * (grid.nz + 1)) as u64);
        let result = kernel::zeroed_storage(device, "promotion result", (4 * RESULT_SLOTS) as u64);
        let group = kernel.bind(
            device,
            "promotion",
            &[&uniform, inputs.state, inputs.bed, inputs.profile, &history, &indicators, &table, &result],
        );

        Self {
            kernel,
            group,
            uniform,
            params,
            indicators,
            result,
            cells,
            frames: 0,
        }
    }

    pub fn write(&mut self, queue: &wgpu::Queue, dt: f32, window: [u32; 2], bodies: &[Body]) {
        let count = bodies.len().min(MAX_BODIES);

        for (slot, body) in self.params.bodies.iter_mut().zip(&bodies[..count]) {
            let c = body.center;
            *slot = [c[0] as f32, c[1] as f32, c[2] as f32, body.radius as f32];
        }

        self.params.body_count = count as u32;
        self.params.window = window;
        self.params.dt = dt.max(1e-6);
        self.params.history = (self.frames > 0) as u32;
        self.frames += 1;
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("promotion"),
            timestamp_writes: None,
        });
        let k = &self.kernel;

        k.dispatch(&mut pass, "promote_vertical", &self.group, self.cells);
        k.dispatch(&mut pass, "promote_cells", &self.group, self.cells);
        k.dispatch_grid(&mut pass, "promote_rows", &self.group, [self.params.nz.max(1), 1]);
        k.dispatch_grid(&mut pass, "promote_columns", &self.group, [self.params.nx + 1, 1]);
        k.dispatch_grid(&mut pass, "promote_window", &self.group, [1, 1]);
    }

    pub fn read(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> GpuPromotionResult {
        kernel::read_pod(device, queue, &self.result)
    }
}
