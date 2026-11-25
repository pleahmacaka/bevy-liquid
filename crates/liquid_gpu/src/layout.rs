use bytemuck::{Pod, Zeroable};

pub const WG: u32 = 256;
pub const SCAN_BLOCK: u32 = 1024;
pub const REDUCE_SPAN: u32 = WG * 4;
pub const MAX_BODIES: usize = 8;
pub const BODY_STRIDE: u32 = 48;
pub const SCALAR_COUNT: usize = 128;
pub const BRICK: u32 = 8;
pub const BRICK_ROWS: u32 = BRICK * BRICK;
pub const BRICK_CELLS: u32 = BRICK * BRICK * BRICK;
pub const BRICK_LIST_HEAD: u32 = 3;
pub const BRICK_FACES: u32 = 1728;
pub const BRICK_NODES: u32 = 729;

pub const S_RNORM: usize = 6;
pub const S_BNORM: usize = 7;
pub const S_TOLERANCE: usize = 8;
pub const S_ITERATION: usize = 9;
pub const S_FLOOR: usize = 12;
pub const S_FLOOR_MEAN: usize = 13;
pub const S_FLOOR_LEVEL: usize = 14;
pub const S_FLOOR_SET: usize = 15;
pub const S_IMPULSE: usize = 64;
pub const S_VISCOUS_ITERATIONS: usize = 112;
pub const S_CAPILLARY_ITERATIONS: usize = 113;
pub const S_VOLUME_TARGET: usize = 114;
pub const S_VOLUME_INTEGRAL: usize = 115;
pub const S_VOLUME: usize = 116;
pub const S_VOLUME_DIVERGENCE: usize = 117;
pub const S_BAND_DT: usize = 118;
pub const S_BAND_SEEDED: usize = 119;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuParams {
    pub dims: [u32; 4],
    pub face_base: [u32; 4],
    pub node_dims: [u32; 4],
    pub gravity: [f32; 4],
    pub dx: f32,
    pub inv_dx: f32,
    pub density: f32,
    pub cfl: f32,
    pub max_dt: f32,
    pub particle_capacity: u32,
    pub body_count: u32,
    pub kernel_radius: f32,
    pub particle_radius: f32,
    pub margin: f32,
    pub frame_dt: f32,
    pub reduction_groups: u32,
    pub pool: [u32; 4],
    pub physics: [f32; 4],
    pub thermal: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuParticle {
    pub x: [f32; 4],
    pub v: [f32; 4],
    pub c: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuBody {
    pub position: [f32; 4],
    pub rotation: [f32; 4],
    pub velocity: [f32; 4],
    pub angular: [f32; 4],
    pub shape: [f32; 4],
    pub inv_inertia: [[f32; 4]; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuState {
    pub dt: f32,
    pub remaining: f32,
    pub time: f32,
    pub running: u32,
    pub particle_count: u32,
    pub vmax_bits: u32,
    pub substeps: u32,
    pub converged: u32,
    pub brick_overflow: u32,
    pub ticks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct GpuLevelParams {
    pub dims: [u32; 4],
    pub regularization: f32,
    pub groups: u32,
    pub body_groups: u32,
    pub sparse: u32,
    pub bricks: u32,
    pub tall: u32,
    pub pack: u32,
    pub body_base: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GpuWhite {
    pub x: [f32; 4],
    pub v: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct GpuWhiteState {
    pub count: u32,
    pub next: u32,
    pub spawned: u32,
    pub pad: u32,
}

pub fn groups(count: u32, span: u32) -> u32 {
    count.div_ceil(span).max(1)
}
