use std::collections::HashMap;

use bevy::math::{DVec3, IVec3, UVec3, Vec3};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::solver::Quality;

pub const MAX_GAS_EMITTERS: usize = 8;
pub const GAS_MASS_QUANTUM: f64 = 1.0 / 16_777_216.0;
pub const GAS_VOLUME_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

const WG: u32 = 256;
const COARSE_ROWS: u32 = 16;
const COARSE_LIMIT: u32 = 1024;
const LEVEL_STRIDE: u64 = 256;
const SOLVE_FIRST: u32 = 2;
const NODES: u32 = 0;
const CELLS: u32 = 1;
const COARSE: u32 = 2;
const COARSE_REGULARIZATION: f64 = 1e-6;
const SINGULAR_SHIFT: f64 = 1e-3;
const GAME_COARSENING: f32 = 0.625;
const SOLID: u32 = 1;
const SEALED: u32 = 128;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GasShape {
    Sphere(f32),
    Cuboid(Vec3),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasRegion {
    pub shape: GasShape,

    pub center: Vec3,
}

impl GasRegion {

    pub fn sphere(center: Vec3, radius: f32) -> Self {
        Self { shape: GasShape::Sphere(radius), center }
    }

    pub fn cuboid(center: Vec3, half: Vec3) -> Self {
        Self { shape: GasShape::Cuboid(half), center }
    }

    pub fn contains(&self, x: DVec3) -> bool {
        let p = x - self.center.as_dvec3();
        let distance = match self.shape {
            GasShape::Sphere(radius) => p.length() - radius as f64,
            GasShape::Cuboid(half) => {
                let q = p.abs() - half.as_dvec3();
                q.max(DVec3::ZERO).length() + q.max_element().min(0.0)
            }
        };

        distance <= 0.0
    }

    pub fn shifted(self, offset: Vec3) -> Self {
        Self { center: self.center + offset, ..self }
    }

    pub(crate) fn packed(&self) -> [[f32; 4]; 2] {
        let (kind, extent) = match self.shape {
            GasShape::Sphere(r) => (0.0, Vec3::splat(r)),
            GasShape::Cuboid(half) => (1.0, half),
        };

        [self.center.extend(kind).to_array(), extent.extend(0.0).to_array()]
    }

}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasEmitter {
    pub region: GasRegion,

    pub density_rate: f32,

    pub heat_rate: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasParams {
    pub ambient_temperature: f32,

    pub density_buoyancy: f32,

    pub thermal_buoyancy: f32,

    pub vorticity_confinement: f32,

    pub density_decay: f32,

    pub cooling: f32,

    pub air_density: f32,

    pub cfl: f32,

    pub max_dt: f32,

    pub tolerance: f32,

    pub max_iterations: u32,

    pub maccormack: bool,

    pub velocity_maccormack: bool,

    pub open_low: [bool; 3],

    pub open_high: [bool; 3],

    pub inflow_at_rest: bool,
}

impl Default for GasParams {
    fn default() -> Self {
        Self {
            ambient_temperature: 293.15,
            density_buoyancy: 0.05,
            thermal_buoyancy: 9.81 / 293.15,
            vorticity_confinement: 0.5,
            density_decay: 0.1,
            cooling: 0.5,
            air_density: 1.2,
            cfl: 1.0,
            max_dt: 1.0 / 60.0,
            tolerance: 0.0,
            max_iterations: 3,
            maccormack: true,
            // clamped MacCormack on velocity diverges after 5-10 s here and in the f64 reference; scalars stay stable
            velocity_maccormack: false,
            open_low: [false; 3],
            open_high: [false, true, false],
            inflow_at_rest: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GasConfig {
    pub dims: UVec3,

    pub dx: f32,

    pub params: GasParams,

    pub max_substeps: u32,

    pub smoothing: u32,

    pub injection_capacity: u32,

    pub warm_start: bool,

    pub quality: Quality,
}

impl Default for GasConfig {
    fn default() -> Self {
        Self {
            dims: UVec3::new(64, 96, 64),
            dx: 1.0 / 32.0,
            params: GasParams::default(),
            max_substeps: 2,
            smoothing: 2,
            injection_capacity: 0,
            warm_start: true,
            quality: Quality::High,
        }
    }
}

impl GasConfig {

    pub fn with_quality(self, quality: Quality) -> Self {
        if quality == Quality::High || self.quality == Quality::Game {
            return Self { quality, ..self };
        }

        let dims = (self.dims.as_vec3() * GAME_COARSENING).ceil().as_uvec3();
        let scale = (0..3).map(|k| self.dims[k] as f32 / dims[k] as f32).fold(f32::INFINITY, f32::min);

        Self {
            quality,
            dims,
            dx: self.dx * scale,
            params: GasParams { max_iterations: 1, ..self.params },
            max_substeps: 1,
            smoothing: 1,
            ..self
        }
    }

    pub fn size(&self) -> Vec3 {
        self.dims.as_vec3() * self.dx
    }

    pub fn cell_count(&self) -> u32 {
        self.dims.element_product()
    }

    pub fn face_count(&self) -> u32 {
        (0..3).map(|a| face_dims(self.dims, a).element_product()).sum()
    }

}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GasInjection {
    pub position_mass: [f32; 4],

    pub velocity_temperature: [f32; 4],

    pub volume: f32,

    pub pad: [f32; 3],
}

impl GasInjection {

    pub fn new(position: Vec3, mass: f32, temperature: f32, velocity: Vec3) -> Self {
        Self {
            position_mass: position.extend(mass).to_array(),
            velocity_temperature: velocity.extend(temperature).to_array(),
            ..Default::default()
        }
    }

    pub fn with_volume(self, volume: f32) -> Self {
        Self { volume, ..self }
    }

}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct GasState {
    pub dt: f32,
    pub remaining: f32,
    pub time: f32,
    pub running: u32,
    pub vmax_bits: u32,
    pub divergence_bits: u32,
    pub substeps: u32,
    pub iterations: u32,
    pub converged: u32,
    pub residual: f32,
    pub rz: f32,
    pub injected_lo: u32,
    pub injected_hi: u32,
    pub accepted: u32,
    pub rejected: u32,
    pub pad: u32,
}

impl GasState {

    pub fn injected_units(&self) -> u64 {
        self.injected_lo as u64 | ((self.injected_hi as u64) << 32)
    }

    pub fn input_divergence(&self) -> f32 {
        f32::from_bits(self.divergence_bits)
    }

    pub fn max_speed(&self) -> f32 {
        f32::from_bits(self.vmax_bits)
    }

}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuEmitter {
    center: [f32; 4],
    extent: [f32; 4],
    rates: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuParams {
    dims: [u32; 4],
    nodes: [u32; 4],
    faces: [u32; 4],
    counts: [u32; 4],
    groups: [u32; 4],
    grid: [f32; 4],
    flow: [f32; 4],
    loss: [f32; 4],
    solve: [f32; 4],
    emitters: [GpuEmitter; MAX_GAS_EMITTERS],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct GpuLevel {
    dims: [u32; 4],
    coarse: [u32; 4],
    span: [u32; 4],
    pad: [[u32; 4]; 13],
}

#[derive(Default)]
pub struct GasTargets {
    pub density: Option<wgpu::Buffer>,

    pub temperature: Option<wgpu::Buffer>,
}

pub struct GasBuffers {
    pub params: wgpu::Buffer,
    pub levels: wgpu::Buffer,
    pub state: wgpu::Buffer,
    pub density: wgpu::Buffer,
    pub temperature: wgpu::Buffer,
    pub velocity: wgpu::Buffer,
    pub density_next: wgpu::Buffer,
    pub temperature_next: wgpu::Buffer,
    pub velocity_next: wgpu::Buffer,
    pub scratch: wgpu::Buffer,
    pub flags: wgpu::Buffer,
    pub coef: wgpu::Buffer,
    pub mg_x: wgpu::Buffer,
    pub mg_b: wgpu::Buffer,
    pub krylov: wgpu::Buffer,
    pub partials: wgpu::Buffer,
    pub inverse: wgpu::Buffer,
    pub injections: wgpu::Buffer,
    pub accum: wgpu::Buffer,
    pub source: wgpu::Buffer,
    pub indirect: wgpu::Buffer,
}

impl GasBuffers {

    fn storage(&self) -> [&wgpu::Buffer; 18] {
        [
            &self.state,
            &self.density,
            &self.temperature,
            &self.velocity,
            &self.density_next,
            &self.temperature_next,
            &self.velocity_next,
            &self.scratch,
            &self.flags,
            &self.coef,
            &self.mg_x,
            &self.mg_b,
            &self.krylov,
            &self.partials,
            &self.inverse,
            &self.injections,
            &self.accum,
            &self.source,
        ]
    }

}

pub struct GasSolver {
    pub config: GasConfig,

    pub buffers: GasBuffers,

    levels: Vec<UVec3>,

    pipelines: HashMap<&'static str, wgpu::ComputePipeline>,

    bind: wgpu::BindGroup,

    level_bind: wgpu::BindGroup,

    control_bind: wgpu::BindGroup,

    volume_layout: wgpu::BindGroupLayout,

    params: GpuParams,

    obstacles: Vec<GasRegion>,

    full: Vec<u32>,
}

fn face_dims(dims: UVec3, a: usize) -> UVec3 {
    let mut d = dims;
    d[a] += 1;

    d
}

fn open_sides(p: &GasParams) -> u32 {
    (0..3).fold(0, |bits, a| bits | (p.open_low[a] as u32) << a | (p.open_high[a] as u32) << (a + 3))
}

fn groups(count: u32, span: u32) -> u32 {
    count.div_ceil(span).max(1)
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16).next_multiple_of(16),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn floats(device: &wgpu::Device, label: &str, count: u32) -> wgpu::Buffer {
    storage(device, label, count as u64 * 4)
}

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

fn buffer_type(ty: wgpu::BufferBindingType, dynamic: Option<u64>) -> wgpu::BindingType {
    wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset: dynamic.is_some(),
        min_binding_size: dynamic.and_then(wgpu::BufferSize::new),
    }
}

fn level_dims(dims: UVec3) -> Vec<UVec3> {
    let mut levels = vec![dims];

    while levels.last().is_some_and(|d| d.element_product() > COARSE_LIMIT) {
        let next = (levels[levels.len() - 1] + 1) / 2;
        levels.push(next);
    }

    levels
}

fn cell_coords(dims: UVec3, n: usize) -> UVec3 {
    let n = n as u32;

    UVec3::new(n % dims.x, (n / dims.x) % dims.y, n / (dims.x * dims.y))
}

fn cell_index(dims: UVec3, c: UVec3) -> usize {
    (c.x + dims.x * (c.y + dims.y * c.z)) as usize
}

fn low_bit(a: usize) -> u32 {
    2 << (2 * a)
}

fn high_bit(a: usize) -> u32 {
    4 << (2 * a)
}

fn cell_flags(config: &GasConfig, obstacles: &[GasRegion]) -> Vec<u32> {
    let dims = config.dims;
    let dx = config.dx as f64;
    let cells = config.cell_count() as usize;
    let solid: Vec<bool> = (0..cells)
        .map(|n| {
            let x = (cell_coords(dims, n).as_dvec3() + 0.5) * dx;
            obstacles.iter().any(|o| o.contains(x))
        })
        .collect();

    let fluid = |c: UVec3, offset: i32, a: usize, open: bool| {
        let mut p = c.as_ivec3();
        p[a] += offset;

        if p[a] < 0 || p[a] >= dims[a] as i32 {
            return open;
        }

        !solid[cell_index(dims, p.as_uvec3())]
    };

    let mut flags: Vec<u32> = (0..cells)
        .map(|n| {
            if solid[n] {
                return SOLID;
            }

            let c = cell_coords(dims, n);
            let p = config.params;

            (0..3).fold(0, |bits, a| {
                let low = fluid(c, -1, a, p.open_low[a]);
                let high = fluid(c, 1, a, p.open_high[a]);

                bits | if low { low_bit(a) } else { 0 } | if high { high_bit(a) } else { 0 }
            })
        })
        .collect();

    seal(dims, &mut flags);

    flags
}

fn seal(dims: UVec3, flags: &mut [u32]) {
    let vented = |n: usize| {
        let c = cell_coords(dims, n);

        (0..3).any(|a| (c[a] == 0 && flags[n] & low_bit(a) != 0) || (c[a] + 1 == dims[a] && flags[n] & high_bit(a) != 0))
    };
    let mut reached: Vec<bool> = (0..flags.len()).map(vented).collect();
    let mut frontier: Vec<usize> = (0..flags.len()).filter(|&n| reached[n]).collect();

    while let Some(n) = frontier.pop() {
        let c = cell_coords(dims, n).as_ivec3();

        for a in 0..3 {
            for (bit, offset) in [(low_bit(a), -1), (high_bit(a), 1)] {
                let p = c + IVec3::AXES[a] * offset;

                if flags[n] & bit == 0 || p[a] < 0 || p[a] >= dims[a] as i32 {
                    continue;
                }

                let m = cell_index(dims, p.as_uvec3());

                if !reached[m] {
                    reached[m] = true;
                    frontier.push(m);
                }
            }
        }
    }

    for (bits, vented) in flags.iter_mut().zip(reached) {
        if !vented && *bits & SOLID == 0 {
            *bits |= SEALED;
        }
    }
}

#[derive(Clone)]
struct Operator {
    dims: UVec3,
    coef: Vec<[f64; 4]>,
    boundary: Vec<[f64; 6]>,
}

impl Operator {

    fn fine(dims: UVec3, flags: &[u32]) -> Self {
        let cells = dims.element_product() as usize;
        let mut op = Self { dims, coef: vec![[0.0; 4]; cells], boundary: vec![[0.0; 6]; cells] };

        for (n, &bits) in flags.iter().enumerate() {
            let c = cell_coords(dims, n);

            for a in 0..3 {
                let high = bits & high_bit(a) != 0;

                if c[a] + 1 < dims[a] && high {
                    op.coef[n][a + 1] = 1.0;
                }

                if c[a] == 0 && bits & low_bit(a) != 0 {
                    op.boundary[n][2 * a] = 1.0;
                }

                if c[a] + 1 == dims[a] && high {
                    op.boundary[n][2 * a + 1] = 1.0;
                }
            }
        }

        op.finalize(0.0);

        op
    }

    fn finalize(&mut self, regularization: f64) {
        let dims = self.dims;

        for n in 0..self.coef.len() {
            let c = cell_coords(dims, n);
            let mut diag = 2.0 * self.boundary[n].iter().sum::<f64>();

            for a in 0..3 {
                diag += self.coef[n][a + 1];

                if c[a] > 0 {
                    let mut lo = c;
                    lo[a] -= 1;
                    diag += self.coef[cell_index(dims, lo)][a + 1];
                }
            }

            self.coef[n][0] = diag * (1.0 + regularization);
        }
    }

    fn coarsen(&self) -> Self {
        let fine = self.dims;
        let dims = (fine + 1) / 2;
        let cells = dims.element_product() as usize;
        let mut op = Self { dims, coef: vec![[0.0; 4]; cells], boundary: vec![[0.0; 6]; cells] };

        for n in 0..cells {
            let cc = cell_coords(dims, n);

            for a in 0..3 {
                let (t1, t2) = ((a + 1) % 3, (a + 2) % 3);
                let footprint = |layer: u32| {
                    (0..4u32).filter_map(move |o| {
                        let mut f = cc * 2;
                        f[a] = layer;
                        f[t1] += o & 1;
                        f[t2] += o >> 1;

                        f.cmplt(fine).all().then_some(f)
                    })
                };

                if cc[a] + 1 < dims[a] {
                    let sum: f64 = footprint(2 * cc[a] + 1)
                        .filter(|f| f[a] + 1 < fine[a])
                        .map(|f| self.coef[cell_index(fine, f)][a + 1])
                        .sum();
                    op.coef[n][a + 1] = sum / 16.0;
                }

                if cc[a] == 0 {
                    let sum: f64 = footprint(0).map(|f| self.boundary[cell_index(fine, f)][2 * a]).sum();
                    op.boundary[n][2 * a] = sum / 16.0;
                }

                if cc[a] + 1 == dims[a] {
                    let sum: f64 = footprint(fine[a] - 1).map(|f| self.boundary[cell_index(fine, f)][2 * a + 1]).sum();
                    op.boundary[n][2 * a + 1] = sum / 16.0;
                }
            }
        }

        op.finalize(COARSE_REGULARIZATION);

        op
    }

    fn inverse(&self) -> Vec<f32> {
        let dims = self.dims;
        let cells = self.coef.len();
        let active: Vec<usize> = (0..cells).filter(|&n| self.coef[n][0] > 0.0).collect();
        let mut slot = vec![usize::MAX; cells];

        for (i, &n) in active.iter().enumerate() {
            slot[n] = i;
        }

        let k = active.len();
        let mut m = vec![0.0f64; k * k];
        let mut parent: Vec<usize> = (0..k).collect();

        fn root(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }

            i
        }

        for (i, &n) in active.iter().enumerate() {
            m[i * k + i] = self.coef[n][0];
            let c = cell_coords(dims, n);

            for a in 0..3 {
                if c[a] + 1 >= dims[a] || self.coef[n][a + 1] <= 0.0 {
                    continue;
                }

                let mut hi = c;
                hi[a] += 1;
                let j = slot[cell_index(dims, hi)];

                if j == usize::MAX {
                    continue;
                }

                m[i * k + j] = -self.coef[n][a + 1];
                m[j * k + i] = -self.coef[n][a + 1];

                let (ri, rj) = (root(&mut parent, i), root(&mut parent, j));
                parent[ri] = rj;
            }
        }

        let mut anchored = vec![false; k];

        for (i, &n) in active.iter().enumerate() {
            if self.boundary[n].iter().any(|b| *b > 0.0) {
                let r = root(&mut parent, i);
                anchored[r] = true;
            }
        }

        for i in 0..k {
            if !anchored[root(&mut parent, i)] {
                m[i * k + i] *= 1.0 + SINGULAR_SHIFT;
            }
        }

        let inv = cholesky_inverse(&m, k);
        let mut out = vec![0.0f32; cells * cells];

        for (i, &n) in active.iter().enumerate() {
            for (j, &p) in active.iter().enumerate() {
                out[p * cells + n] = inv[j * k + i] as f32;
            }
        }

        out
    }

    fn packed(&self) -> Vec<[f32; 4]> {
        self.coef.iter().map(|c| c.map(|v| v as f32)).collect()
    }

}

fn cholesky_inverse(m: &[f64], k: usize) -> Vec<f64> {
    let band = (0..k)
        .flat_map(|i| (0..i).filter(move |&j| m[i * k + j] != 0.0).map(move |j| i - j))
        .max()
        .unwrap_or(0);
    let mut l = vec![0.0f64; k * k];

    for j in 0..k {
        let start = j.saturating_sub(band);
        let mut d = m[j * k + j];

        for p in start..j {
            d -= l[j * k + p] * l[j * k + p];
        }

        let d = d.max(f64::MIN_POSITIVE).sqrt();
        l[j * k + j] = d;

        for i in j + 1..(j + band + 1).min(k) {
            let mut s = m[i * k + j];

            for p in i.saturating_sub(band)..j {
                s -= l[i * k + p] * l[j * k + p];
            }

            l[i * k + j] = s / d;
        }
    }

    let mut inv = vec![0.0f64; k * k];
    let mut y = vec![0.0f64; k];

    for col in 0..k {
        y[..col].fill(0.0);

        for i in col..k {
            let mut s = if i == col { 1.0 } else { 0.0 };

            for p in i.saturating_sub(band).max(col)..i {
                s -= l[i * k + p] * y[p];
            }

            y[i] = s / l[i * k + i];
        }

        for i in (0..k).rev() {
            let mut s = y[i];

            for p in i + 1..(i + band + 1).min(k) {
                s -= l[p * k + i] * inv[p * k + col];
            }

            inv[i * k + col] = s / l[i * k + i];
        }
    }

    inv
}

struct Hierarchy {
    dims: Vec<UVec3>,
    offsets: Vec<u32>,
    coef: Vec<[f32; 4]>,
    inverse: Vec<f32>,
}

fn hierarchy(config: &GasConfig, flags: &[u32]) -> Hierarchy {
    let dims = level_dims(config.dims);
    let mut ops = vec![Operator::fine(config.dims, flags)];

    while ops.len() < dims.len() {
        let next = ops[ops.len() - 1].coarsen();
        ops.push(next);
    }

    let mut offsets = Vec::new();
    let mut coef = Vec::new();

    for op in &ops {
        offsets.push(coef.len() as u32);
        coef.extend(op.packed());
    }

    let inverse = ops[ops.len() - 1].inverse();

    Hierarchy { dims, offsets, coef, inverse }
}

fn level_entries(h: &Hierarchy) -> Vec<GpuLevel> {
    (0..h.dims.len())
        .map(|l| {
            let d = h.dims[l];
            let hx = d.x.div_ceil(2);
            let coarse = h.dims.get(l + 1).map_or([0; 4], |c| [c.x, c.y, c.z, h.offsets[l + 1]]);

            GpuLevel {
                dims: [d.x, d.y, d.z, h.offsets[l]],
                coarse,
                span: [hx, hx * d.y * d.z, d.element_product(), 0],
                ..Default::default()
            }
        })
        .collect()
}

fn solve_cells(l: usize) -> u32 {
    3 + 3 * l as u32
}

fn solve_half(l: usize) -> u32 {
    4 + 3 * l as u32
}

fn solve_blocks(l: usize) -> u32 {
    5 + 3 * l as u32
}

fn slot_table(config: &GasConfig, levels: &[UVec3]) -> Vec<u32> {
    let nodes = (config.dims + 1).element_product();
    let coarse = levels[levels.len() - 1].element_product();
    let mut full = vec![groups(nodes, WG), groups(config.cell_count(), WG), groups(coarse, COARSE_ROWS)];

    for d in levels {
        full.push(groups(d.element_product(), WG));
        full.push(groups(d.x.div_ceil(2) * d.y * d.z, WG));
        full.push(((*d + 3) / 4).element_product());
    }

    let mut table: Vec<u32> = full.iter().flat_map(|_| [0, 1, 1]).collect();
    table.extend(full.iter().flat_map(|g| [*g, 1, 1]));

    table
}

impl GasSolver {

    pub fn new(device: &wgpu::Device, config: GasConfig, obstacles: &[GasRegion], targets: GasTargets) -> Self {
        let dims = config.dims;
        let cells = config.cell_count();
        let faces = config.face_count();
        let nodes = (dims + 1).element_product();
        let levels = level_dims(dims);
        let coarse = levels[levels.len() - 1].element_product();
        let level_cells: u32 = levels.iter().map(|d| d.element_product()).sum();
        let rz_groups = if levels.len() > 1 {
            groups(dims.x.div_ceil(2) * dims.y * dims.z, WG)
        } else {
            groups(coarse, COARSE_ROWS)
        };
        let cell_groups = groups(cells, WG);
        let partial_count = cell_groups.max(rz_groups);
        let capacity = config.injection_capacity;
        let accum_words = if capacity > 0 { 3 * (cells + faces) + 3 * cells } else { 4 };
        let source_words = if capacity > 0 { 2 * cells } else { 4 };

        let params = GpuParams {
            dims: [dims.x, dims.y, dims.z, cells],
            nodes: [dims.x + 1, dims.y + 1, dims.z + 1, nodes],
            faces: [
                face_dims(dims, 0).element_product(),
                face_dims(dims, 0).element_product() + face_dims(dims, 1).element_product(),
                faces,
                0,
            ],
            counts: [0, capacity, levels.len() as u32, open_sides(&config.params)],
            groups: [cell_groups, rz_groups, partial_count, 0],
            grid: [config.dx, 1.0 / config.dx, 1.0 / 60.0, config.params.max_dt],
            flow: [
                config.params.cfl,
                config.params.ambient_temperature,
                config.params.density_buoyancy,
                config.params.thermal_buoyancy,
            ],
            loss: [
                config.params.vorticity_confinement,
                config.params.density_decay,
                config.params.cooling,
                config.params.air_density,
            ],
            solve: [config.params.tolerance, 0.0, 0.0, 0.0],
            emitters: [GpuEmitter::default(); MAX_GAS_EMITTERS],
        };

        let level_count = levels.len() as u64;
        let buffers = GasBuffers {
            params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gas params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }),
            levels: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gas levels"),
                size: level_count * LEVEL_STRIDE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: storage(device, "gas state", std::mem::size_of::<GasState>() as u64),
            density: targets.density.unwrap_or_else(|| floats(device, "gas density", cells)),
            temperature: targets.temperature.unwrap_or_else(|| floats(device, "gas temperature", cells)),
            velocity: floats(device, "gas velocity", faces),
            density_next: floats(device, "gas density next", cells),
            temperature_next: floats(device, "gas temperature next", cells),
            velocity_next: floats(device, "gas velocity next", faces),
            scratch: floats(device, "gas scratch", (8 * cells).max(6 * cells + 3 * faces)),
            flags: floats(device, "gas flags", cells),
            coef: floats(device, "gas coefficients", 4 * level_cells),
            mg_x: floats(device, "gas multigrid x", level_cells),
            mg_b: floats(device, "gas multigrid b", level_cells),
            krylov: floats(device, "gas krylov", 3 * cells),
            partials: floats(device, "gas partials", 3 * partial_count),
            inverse: floats(device, "gas coarse inverse", coarse * coarse),
            injections: storage(device, "gas injections", 16 + capacity.max(1) as u64 * std::mem::size_of::<GasInjection>() as u64),
            accum: floats(device, "gas injection accumulators", accum_words),
            source: floats(device, "gas expansion source", source_words),
            indirect: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gas indirect"),
                size: (3 + 3 * level_count) * 24,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        let read_write = wgpu::BufferBindingType::Storage { read_only: false };
        let main_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gas"),
            entries: &std::iter::once(entry(0, buffer_type(wgpu::BufferBindingType::Uniform, None)))
                .chain((1..19).map(|b| entry(b, buffer_type(read_write, None))))
                .collect::<Vec<_>>(),
        });
        let level_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gas level"),
            entries: &[entry(0, buffer_type(wgpu::BufferBindingType::Uniform, Some(48)))],
        });
        let control_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gas control"),
            entries: &[entry(0, buffer_type(read_write, None))],
        });
        let volume_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gas volume"),
            entries: &[entry(
                0,
                wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: GAS_VOLUME_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D3,
                },
            )],
        });

        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gas"),
            layout: &main_layout,
            entries: &std::iter::once(wgpu::BindGroupEntry { binding: 0, resource: buffers.params.as_entire_binding() })
                .chain(
                    buffers
                        .storage()
                        .into_iter()
                        .enumerate()
                        .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }),
                )
                .collect::<Vec<_>>(),
        });
        let level_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gas level"),
            layout: &level_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffers.levels,
                    offset: 0,
                    size: wgpu::BufferSize::new(48),
                }),
            }],
        });
        let control_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gas control"),
            layout: &control_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffers.indirect.as_entire_binding() }],
        });

        let layout = |groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("gas"),
                bind_group_layouts: groups,
                immediate_size: 0,
            })
        };
        let main = layout(&[Some(&main_layout), Some(&level_layout)]);
        let control = layout(&[Some(&main_layout), None, Some(&control_layout)]);
        let pack = layout(&[Some(&main_layout), None, None, Some(&volume_layout)]);

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gas"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gas.wgsl").into()),
        });
        let slot_count = 3.0 + 3.0 * level_count as f64;
        let fixed = config.params.tolerance <= 0.0;
        let share = 1.0 / config.max_substeps.max(1) as f64;
        let warm = if fixed && config.warm_start { 1.0 } else { 0.0 };
        let maccormack = if config.params.maccormack { 1.0 } else { 0.0 };
        let velocity_maccormack = if config.params.velocity_maccormack { 1.0 } else { 0.0 };
        let expansion = if capacity > 0 { 1.0 } else { 0.0 };
        let inflow_at_rest = if config.params.inflow_at_rest { 1.0 } else { 0.0 };
        let controlled = |constants: &[(&'static str, f64)]| {
            constants.iter().copied().chain([("SLOT_COUNT", slot_count), ("SOLVE_FIRST", SOLVE_FIRST as f64)]).collect::<Vec<_>>()
        };
        let specs = [
            ("plan#begin", "gas_plan", &control, controlled(&[("BEGIN", 1.0)])),
            ("plan", "gas_plan", &control, controlled(&[])),
            ("plan#last", "gas_plan", &control, controlled(&[("LAST", 1.0)])),
            ("plan#begin_last", "gas_plan", &control, controlled(&[("BEGIN", 1.0), ("LAST", 1.0)])),
            ("plan#fixed_begin", "gas_plan", &control, controlled(&[("BEGIN", 1.0), ("SHARE", share)])),
            ("plan#fixed", "gas_plan", &control, controlled(&[("SHARE", share)])),
            ("check#initial", "gas_check", &control, controlled(&[("INITIAL", 1.0)])),
            ("check", "gas_check", &control, controlled(&[])),
            ("weigh", "gas_weigh", &main, vec![("EXPANSION", expansion)]),
            ("advect", "gas_advect", &main, vec![("EXPANSION", expansion), ("INFLOW_AT_REST", inflow_at_rest)]),
            (
                "correct",
                "gas_correct",
                &main,
                vec![("MACCORMACK", maccormack), ("VELOCITY_MACCORMACK", velocity_maccormack), ("EXPANSION", expansion)],
            ),
            ("vorticity", "gas_vorticity", &main, vec![]),
            ("confine", "gas_confine", &main, vec![]),
            ("force", "gas_force", &main, vec![("WARM", warm), ("EXPANSION", expansion)]),
            ("gradient", "gas_gradient", &main, vec![]),
            ("smooth#0", "mg_smooth", &main, vec![("COLOR", 0.0)]),
            ("smooth#1", "mg_smooth", &main, vec![("COLOR", 1.0)]),
            ("smooth_rz", "mg_smooth_rz", &main, vec![]),
            ("restrict", "mg_restrict", &main, vec![]),
            ("prolong", "mg_prolong", &main, vec![]),
            ("coarse", "mg_coarse", &main, vec![]),
            ("coarse#rz", "mg_coarse", &main, vec![("RZ", 1.0)]),
            ("direction", "pcg_direction", &main, vec![]),
            ("update", "pcg_update", &main, vec![]),
            ("inject", "gas_inject", &main, vec![]),
            ("apply", "gas_apply", &main, vec![]),
            ("pack", "gas_pack", &pack, vec![]),
        ];

        let pipelines = crate::solver::parallel_map(&specs, |(key, entry_point, layout, constants)| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

            (*key, pipeline)
        })
        .into_iter()
        .collect();

        Self {
            config,
            buffers,
            levels,
            pipelines,
            bind,
            level_bind,
            control_bind,
            volume_layout,
            params,
            obstacles: obstacles.to_vec(),
            full: Vec::new(),
        }
        .with_slots()
    }

    fn with_slots(mut self) -> Self {
        let table = slot_table(&self.config, &self.levels);
        self.full = table[table.len() / 2..].iter().step_by(3).copied().collect();

        self
    }

    fn fixed(&self) -> bool {
        self.config.params.tolerance <= 0.0
    }

    pub fn initialize(&self, queue: &wgpu::Queue) {
        let flags = cell_flags(&self.config, &self.obstacles);
        let h = hierarchy(&self.config, &flags);

        for (l, level) in level_entries(&h).iter().enumerate() {
            queue.write_buffer(&self.buffers.levels, l as u64 * LEVEL_STRIDE, &bytemuck::bytes_of(level)[..48]);
        }

        queue.write_buffer(&self.buffers.indirect, 0, bytemuck::cast_slice(&slot_table(&self.config, &self.levels)));
        queue.write_buffer(&self.buffers.flags, 0, bytemuck::cast_slice(&flags));
        queue.write_buffer(&self.buffers.coef, 0, bytemuck::cast_slice(&h.coef));
        queue.write_buffer(&self.buffers.inverse, 0, bytemuck::cast_slice(&h.inverse));
        self.reset(queue);
    }

    pub fn reset(&self, queue: &wgpu::Queue) {
        let cells = self.config.cell_count() as usize;
        let faces = self.config.face_count() as usize;
        let ambient = vec![self.config.params.ambient_temperature; cells];
        let state = GasState::default();

        queue.write_buffer(&self.buffers.density, 0, bytemuck::cast_slice(&vec![0.0f32; cells]));
        queue.write_buffer(&self.buffers.temperature, 0, bytemuck::cast_slice(&ambient));
        queue.write_buffer(&self.buffers.velocity, 0, bytemuck::cast_slice(&vec![0.0f32; faces]));
        queue.write_buffer(&self.buffers.krylov, 0, bytemuck::cast_slice(&vec![0.0f32; 3 * cells]));
        queue.write_buffer(&self.buffers.state, 0, bytemuck::bytes_of(&state));
    }

    pub fn upload(&self, queue: &wgpu::Queue, density: &[f32], temperature: &[f32], velocity: &[f32]) {
        let vmax = velocity.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        queue.write_buffer(&self.buffers.density, 0, bytemuck::cast_slice(density));
        queue.write_buffer(&self.buffers.temperature, 0, bytemuck::cast_slice(temperature));
        queue.write_buffer(&self.buffers.velocity, 0, bytemuck::cast_slice(velocity));
        queue.write_buffer(&self.buffers.state, 16, bytemuck::bytes_of(&vmax.to_bits()));
    }

    pub fn set_obstacles(&mut self, queue: &wgpu::Queue, obstacles: &[GasRegion]) {
        let flags = cell_flags(&self.config, obstacles);
        let h = hierarchy(&self.config, &flags);

        self.obstacles = obstacles.to_vec();
        queue.write_buffer(&self.buffers.flags, 0, bytemuck::cast_slice(&flags));
        queue.write_buffer(&self.buffers.coef, 0, bytemuck::cast_slice(&h.coef));
        queue.write_buffer(&self.buffers.inverse, 0, bytemuck::cast_slice(&h.inverse));
        queue.write_buffer(&self.buffers.krylov, 0, bytemuck::cast_slice(&vec![0.0f32; 3 * self.config.cell_count() as usize]));
    }

    pub fn obstacles(&self) -> &[GasRegion] {
        &self.obstacles
    }

    pub fn set_frame(&mut self, queue: &wgpu::Queue, frame_dt: f32, emitters: &[GasEmitter]) {
        let mut params = self.params;
        params.grid[2] = frame_dt;
        params.counts[0] = emitters.len().min(MAX_GAS_EMITTERS) as u32;
        params.emitters = [GpuEmitter::default(); MAX_GAS_EMITTERS];

        for (slot, e) in params.emitters.iter_mut().zip(emitters) {
            let [center, extent] = e.region.packed();

            *slot = GpuEmitter { center, extent, rates: [e.density_rate, e.heat_rate, 0.0, 0.0] };
        }

        if bytemuck::bytes_of(&params) != bytemuck::bytes_of(&self.params) {
            self.params = params;
            queue.write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(&self.params));
        }
    }

    pub fn inject(&self, queue: &wgpu::Queue, items: &[GasInjection]) {
        let count = items.len().min(self.config.injection_capacity as usize);

        if count > 0 {
            queue.write_buffer(&self.buffers.injections, 16, bytemuck::cast_slice(&items[..count]));
        }

        queue.write_buffer(&self.buffers.injections, 0, bytemuck::bytes_of(&(count as u32)));
    }

    pub fn levels(&self) -> &[UVec3] {
        &self.levels
    }

    pub fn volume_bind_group(&self, device: &wgpu::Device, view: &wgpu::TextureView) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gas volume"),
            layout: &self.volume_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) }],
        })
    }

    pub fn encode_frame(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gas"), timestamp_writes: None });

        self.record_frame(&mut pass);
    }

    pub fn record_frame(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.config.injection_capacity > 0 {
            self.direct(pass, "inject", groups(self.config.injection_capacity, WG));
            self.direct(pass, "apply", self.params.nodes[3].div_ceil(WG));
        }

        let budget = self.config.max_substeps.max(1);

        if self.fixed() && self.params.grid[2] <= 0.0 {
            return;
        }

        for k in 0..budget {
            self.record_substep(pass, k == 0, k + 1 == budget);
        }
    }

    pub fn record_pack(&self, pass: &mut wgpu::ComputePass<'_>, volume: &wgpu::BindGroup) {
        pass.set_pipeline(&self.pipelines["pack"]);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_bind_group(3, volume, &[]);
        pass.dispatch_workgroups(groups(self.config.cell_count(), WG), 1, 1);
    }

    fn record_substep(&self, pass: &mut wgpu::ComputePass<'_>, first: bool, last: bool) {
        let p = self.config.params;
        let tolerance = p.tolerance > 0.0;
        let iterations = p.max_iterations.max(1);

        let plan = match (self.fixed(), first, last) {
            (true, true, _) => "plan#fixed_begin",
            (true, false, _) => "plan#fixed",
            (false, true, true) => "plan#begin_last",
            (false, true, false) => "plan#begin",
            (false, false, true) => "plan#last",
            (false, false, false) => "plan",
        };

        self.control(pass, plan);

        if self.config.injection_capacity > 0 {
            self.run(pass, "weigh", 0, CELLS);
        }

        self.run(pass, "advect", 0, NODES);
        self.run(pass, "correct", 0, NODES);

        if p.vorticity_confinement > 0.0 {
            self.run(pass, "vorticity", 0, CELLS);
            self.run(pass, "confine", 0, CELLS);
        }

        self.run(pass, "force", 0, NODES);

        if tolerance {
            self.control(pass, "check#initial");
        }

        self.record_vcycle(pass);

        for it in 0..iterations {
            self.run(pass, "direction", 0, solve_cells(0));
            self.run(pass, "update", 0, solve_cells(0));

            if tolerance {
                self.control(pass, "check");
            }

            if it + 1 < iterations {
                self.record_vcycle(pass);
            }
        }

        self.run(pass, "gradient", 0, NODES);
    }

    fn record_vcycle(&self, pass: &mut wgpu::ComputePass<'_>) {
        let last = self.levels.len() - 1;

        if last == 0 {
            self.run(pass, "coarse#rz", 0, COARSE);
            return;
        }

        let sweeps = self.config.smoothing.max(1);

        for l in 0..last {
            self.run(pass, "smooth#1", l, solve_half(l));

            for _ in 1..sweeps {
                self.run(pass, "smooth#0", l, solve_half(l));
                self.run(pass, "smooth#1", l, solve_half(l));
            }

            self.run(pass, "restrict", l, solve_blocks(l + 1));
        }

        self.run(pass, "coarse", last, COARSE);

        for l in (0..last).rev() {
            self.run(pass, "prolong", l, solve_half(l));

            for _ in 1..sweeps {
                self.run(pass, "smooth#0", l, solve_half(l));
                self.run(pass, "smooth#1", l, solve_half(l));
            }

            self.run(pass, if l == 0 { "smooth_rz" } else { "smooth#0" }, l, solve_half(l));
        }
    }

    fn control(&self, pass: &mut wgpu::ComputePass<'_>, key: &str) {
        pass.set_pipeline(&self.pipelines[key]);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_bind_group(2, &self.control_bind, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn run(&self, pass: &mut wgpu::ComputePass<'_>, key: &str, level: usize, slot: u32) {
        pass.set_pipeline(&self.pipelines[key]);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_bind_group(1, &self.level_bind, &[(level as u64 * LEVEL_STRIDE) as u32]);
        if self.fixed() {
            pass.dispatch_workgroups(self.full[slot as usize], 1, 1);
        } else {
            pass.dispatch_workgroups_indirect(&self.buffers.indirect, slot as u64 * 12);
        }
    }

    fn direct(&self, pass: &mut wgpu::ComputePass<'_>, key: &str, count: u32) {
        pass.set_pipeline(&self.pipelines[key]);
        pass.set_bind_group(0, &self.bind, &[]);
        pass.set_bind_group(1, &self.level_bind, &[0]);
        pass.dispatch_workgroups(count, 1, 1);
    }

}
