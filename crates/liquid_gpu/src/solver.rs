use std::collections::HashMap;

use bevy::math::{UVec3, Vec3};
use wgpu::util::DeviceExt;

use crate::layout::{
    BODY_STRIDE, BRICK, BRICK_CELLS, BRICK_FACES, BRICK_LIST_HEAD, BRICK_NODES, BRICK_ROWS, FLOOR_SPAWN, GpuBody, GpuLevelParams, GpuParams,
    GpuParticle, GpuState, GpuWhite, GpuWhiteState, MAX_BODIES, REDUCE_SPAN, S_TOLERANCE, SCALAR_COUNT, SCAN_BLOCK, WG, groups,
};
use crate::tall::{TALL_LEVELS, TALL_STRIDE};

pub const SHADER_SOURCES: [&str; 22] = [
    include_str!("shaders/common.wgsl"),
    include_str!("shaders/bindings.wgsl"),
    include_str!("shaders/level.wgsl"),
    include_str!("shaders/plan.wgsl"),
    include_str!("shaders/sort.wgsl"),
    include_str!("shaders/brick.wgsl"),
    include_str!("shaders/transfer.wgsl"),
    include_str!("shaders/levelset.wgsl"),
    include_str!("shaders/velocity.wgsl"),
    include_str!("shaders/mg.wgsl"),
    include_str!("shaders/pcg.wgsl"),
    include_str!("shaders/rigid.wgsl"),
    include_str!("shaders/idp.wgsl"),
    include_str!("shaders/surface.wgsl"),
    include_str!("shaders/coupling.wgsl"),
    include_str!("shaders/whitewater.wgsl"),
    include_str!("shaders/redistance.wgsl"),
    include_str!("shaders/band.wgsl"),
    include_str!("shaders/tension.wgsl"),
    include_str!("shaders/viscosity.wgsl"),
    include_str!("shaders/floor.wgsl"),
    include_str!("shaders/tall.wgsl"),
];

pub fn shader_source(subgroups: bool, tail_words: u32) -> String {
    let reduce = if subgroups {
        include_str!("shaders/reduce_subgroup.wgsl")
    } else {
        include_str!("shaders/reduce_shared.wgsl")
    };

    [SHADER_SOURCES.join("\n"), reduce.to_string(), format!("const TAIL_WORDS: u32 = {tail_words}u;")].join("\n")
}

#[derive(Clone, Debug, PartialEq)]
pub struct SolverConfig {
    pub dims: UVec3,
    pub dx: f32,
    pub max_particles: u32,
    pub density: f32,
    pub gravity: Vec3,
    pub cfl: f32,
    pub max_dt: f32,
    pub max_substeps: u32,
    pub pressure_iterations: u32,
    pub density_iterations: u32,
    pub tolerance: f32,
    pub extrapolation_layers: u32,
    pub multigrid_min_dim: u32,
    pub smoothing: u32,
    pub coarse_sweeps: u32,
    pub block_smoothing: bool,
    pub density_projection: bool,
    pub kernel_radius: f32,
    pub particle_radius: f32,
    pub margin: f32,
    pub whitewater_capacity: u32,
    pub sparse: bool,
    pub brick_capacity: Option<u32>,
    pub narrow_band: Option<f32>,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub viscosity_iterations: u32,
    pub redistance_iterations: u32,
    pub subgroups: bool,
    pub boundary_width: u32,
    pub boundary_sweeps: u32,
    pub open_floor: Option<f32>,
    pub tall_cells: Option<u32>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            dims: UVec3::new(64, 48, 64),
            dx: 1.0 / 32.0,
            max_particles: 1 << 20,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            cfl: 1.0,
            max_dt: 1.0 / 60.0,
            max_substeps: 4,
            pressure_iterations: 24,
            density_iterations: 8,
            tolerance: 1e-5,
            extrapolation_layers: 4,
            multigrid_min_dim: 4,
            smoothing: 2,
            coarse_sweeps: 8,
            block_smoothing: false,
            density_projection: true,
            kernel_radius: 1.0,
            particle_radius: 0.4,
            margin: 0.1,
            whitewater_capacity: 0,
            sparse: true,
            brick_capacity: None,
            narrow_band: None,
            viscosity: 0.0,
            surface_tension: 0.0,
            viscosity_iterations: 50,
            redistance_iterations: 16,
            subgroups: true,
            boundary_width: 0,
            boundary_sweeps: 0,
            open_floor: None,
            tall_cells: None,
        }
    }
}

impl SolverConfig {
    pub fn has_band(&self) -> bool {
        self.narrow_band.is_some()
    }

    pub fn has_tension(&self) -> bool {
        self.surface_tension > 0.0
    }

    pub fn has_viscosity(&self) -> bool {
        self.viscosity > 0.0
    }

    pub fn redistances(&self) -> bool {
        self.has_band() || self.has_tension()
    }

    pub fn diffuses(&self) -> bool {
        self.has_viscosity() || self.has_tension()
    }

    pub fn smooths_boundary(&self) -> bool {
        self.boundary_width > 0 && self.boundary_sweeps > 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slots {
    pub levels: u32,
}

impl Slots {
    pub const PARTICLES: u32 = 0;
    pub const CELLS: u32 = 1;
    pub const FACES: u32 = 2;
    pub const NODES: u32 = 3;
    pub const SINGLE: u32 = 4;
    pub const SCAN: u32 = 5;
    pub const REDUCE: u32 = 6;
    pub const GRID: u32 = 7;
    pub const BRICKS: u32 = 8;
    pub const CLEAR: u32 = 9;
    pub const BODIES: u32 = 10;
    const FIRST_LEVEL: u32 = 11;

    pub fn level(&self, l: usize) -> u32 {
        Self::FIRST_LEVEL + l as u32
    }

    pub fn solve_first(&self) -> u32 {
        Self::FIRST_LEVEL + self.levels
    }

    pub fn solve_single(&self) -> u32 {
        self.solve_first()
    }

    pub fn solve_reduce(&self) -> u32 {
        self.solve_first() + 1
    }

    pub fn solve_level(&self, l: usize) -> u32 {
        self.solve_first() + 2 + l as u32
    }

    pub fn solve_faces(&self) -> u32 {
        self.solve_first() + 2 + self.levels
    }

    pub fn solve_bodies(&self) -> u32 {
        self.solve_first() + 3 + self.levels
    }

    pub fn solve_grid(&self) -> u32 {
        self.solve_first() + 4 + self.levels
    }

    pub fn solve_block(&self, l: usize) -> u32 {
        self.solve_first() + 5 + self.levels + l as u32
    }

    pub fn count(&self) -> u32 {
        self.solve_first() + 5 + 2 * self.levels
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Layout {
    Sim,
    Level,
    Pair,
    Plan,
    Coupling,
    Solve,
    SolveLevel,
    SolvePair,
    SolvePlan,
    SolveLevelPlan,
    Extra,
    LevelExtra,
    SolveExtra,
    Link,
    LevelCoupling,
}

impl Layout {
    fn solve(self) -> bool {
        matches!(self, Layout::Solve | Layout::SolveLevel | Layout::SolvePair | Layout::SolvePlan | Layout::SolveLevelPlan | Layout::SolveExtra)
    }

    fn extra(self) -> bool {
        matches!(self, Layout::Extra | Layout::LevelExtra | Layout::SolveExtra | Layout::Link)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bind {
    Sim,
    Level(usize),
    Multigrid(usize),
    Swapped(usize),
    Pair(usize),
    Plan,
    LevelPlan(usize),
    Coupling,
    LevelCoupling(usize),
}

pub struct SimBuffers {
    pub params: wgpu::Buffer,
    pub particles: wgpu::Buffer,
    pub particles_out: wgpu::Buffer,
    pub cell_counter: wgpu::Buffer,
    pub cell_start: wgpu::Buffer,
    pub sorted: wgpu::Buffer,
    pub velocity: wgpu::Buffer,
    pub valid: wgpu::Buffer,
    pub scratch: wgpu::Buffer,
    pub weights: wgpu::Buffer,
    pub owners: wgpu::Buffer,
    pub solid_velocity: wgpu::Buffer,
    pub phi: wgpu::Buffer,
    pub nodes: wgpu::Buffer,
    pub bodies: wgpu::Buffer,
    pub state: wgpu::Buffer,
    pub scalars: wgpu::Buffer,
    pub partials: wgpu::Buffer,
    pub scratch_valid: wgpu::Buffer,
    pub pressure_keep: wgpu::Buffer,
    pub ellipsoids: wgpu::Buffer,
    pub surface: wgpu::Buffer,
    pub indirect: wgpu::Buffer,
    pub white: wgpu::Buffer,
    pub white_out: wgpu::Buffer,
    pub white_state: wgpu::Buffer,
    pub brick_flags: wgpu::Buffer,
    pub brick_list: wgpu::Buffer,
    pub band_phi: wgpu::Buffer,
    pub signed_distance: wgpu::Buffer,
    pub sweep_a: wgpu::Buffer,
    pub sweep_b: wgpu::Buffer,
    pub kappa: wgpu::Buffer,
    pub band_offsets: wgpu::Buffer,
    pub viscous: ViscousBuffers,
}

pub struct ViscousBuffers {
    pub cells: wgpu::Buffer,
    pub edges: wgpu::Buffer,
    pub mass: wgpu::Buffer,
    pub diagonal: wgpu::Buffer,
    pub x: wgpu::Buffer,
    pub r: wgpu::Buffer,
    pub d: wgpu::Buffer,
    pub q: wgpu::Buffer,
}

impl SimBuffers {
    fn extra(&self) -> [(u32, &wgpu::Buffer); 14] {
        let v = &self.viscous;

        [
            (12, &self.band_phi),
            (13, &self.signed_distance),
            (14, &self.sweep_a),
            (15, &self.sweep_b),
            (16, &self.kappa),
            (17, &v.cells),
            (18, &v.edges),
            (19, &v.mass),
            (20, &v.diagonal),
            (21, &v.x),
            (22, &v.r),
            (23, &v.d),
            (24, &v.q),
            (25, &self.band_offsets),
        ]
    }
}

pub struct LevelBuffers {
    pub dims: UVec3,
    pub len: u32,
    pub params: wgpu::Buffer,
    pub phi: wgpu::Buffer,
    pub coef: wgpu::Buffer,
    pub active: wgpu::Buffer,
    pub diag: wgpu::Buffer,
    pub off: wgpu::Buffer,
    pub x: wgpu::Buffer,
    pub b: wgpu::Buffer,
    pub r: wgpu::Buffer,
    pub z: wgpu::Buffer,
    pub d: wgpu::Buffer,
    pub q: wgpu::Buffer,
    pub t: wgpu::Buffer,
    pub tall: wgpu::Buffer,
    pub e: wgpu::Buffer,
    bind: wgpu::BindGroup,
    multigrid_bind: wgpu::BindGroup,
    swapped_bind: wgpu::BindGroup,
}

pub struct GpuSolver {
    pub config: SolverConfig,
    pub subgroups: bool,
    pub params: GpuParams,
    pub slots: Slots,
    pub buffers: SimBuffers,
    pub levels: Vec<LevelBuffers>,
    pub(crate) pack: wgpu::Buffer,
    pub(crate) tail: usize,
    sim_bind: wgpu::BindGroup,
    solve_bind: wgpu::BindGroup,
    plan_bind: wgpu::BindGroup,
    extra_bind: wgpu::BindGroup,
    pub(crate) coupling_layout: wgpu::BindGroupLayout,
    pub(crate) coupling: Option<Coupling>,
    pipelines: HashMap<String, (wgpu::ComputePipeline, Layout)>,
}

const SOLVE_BINDINGS: [u32; 6] = [0, 15, 17, 18, 26, 27];

const STORAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_DST)
    .union(wgpu::BufferUsages::COPY_SRC);

pub const GUARD_BYTES: u64 = 256;

pub const GUARD_WORD: u32 = 0x7FBA_DCAF;

const MAX_GROUPS: u32 = 65535;

fn storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    let data = bytes.max(16).next_multiple_of(16);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: data + GUARD_BYTES,
        usage: STORAGE,
        mapped_at_creation: true,
    });

    if let Ok(mut view) = buffer.slice(data..).get_mapped_range_mut() {
        view.copy_from_slice(bytemuck::cast_slice(&[GUARD_WORD; (GUARD_BYTES / 4) as usize]));
    }

    buffer.unmap();

    buffer
}

fn floats(device: &wgpu::Device, label: &str, count: u32) -> wgpu::Buffer {
    storage(device, label, count as u64 * 4)
}

fn uniform<T: bytemuck::Pod>(device: &wgpu::Device, label: &str, value: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::bytes_of(value),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

fn layout_entries(bindings: impl IntoIterator<Item = u32>) -> Vec<wgpu::BindGroupLayoutEntry> {
    bindings
        .into_iter()
        .map(|binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: if binding == 0 {
                    wgpu::BufferBindingType::Uniform
                } else {
                    wgpu::BufferBindingType::Storage { read_only: false }
                },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}

fn bind_entries<'a>(buffers: &[(u32, &'a wgpu::Buffer)]) -> Vec<wgpu::BindGroupEntry<'a>> {
    buffers
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry { binding: *binding, resource: buffer.as_entire_binding() })
        .collect()
}

pub fn tail_start(levels: &[UVec3], tall: usize, words: u32) -> usize {
    let last = levels.len() - 1;

    if last == 0 {
        return levels.len();
    }

    let lens: Vec<u32> = levels.iter().map(|d| d.element_product()).collect();

    (tall.max(1)..last).find(|&l| tail_words(&lens[l..]) <= words).unwrap_or(last)
}

fn tail_words(lens: &[u32]) -> u32 {
    6 * lens.iter().sum::<u32>() + lens[0]
}

pub fn level_dims(dims: UVec3, min_dim: u32) -> Vec<UVec3> {
    let mut out = vec![dims];

    while out.last().is_some_and(|d| d.max_element() > min_dim) {
        let next = out.last().map(|d| (*d + 1) / 2).unwrap_or(dims);
        out.push(next);
    }

    out
}

pub fn brick_dims(dims: UVec3) -> UVec3 {
    (dims + BRICK - 1) / BRICK
}

fn level_table(bricks: u32, capacity: u32) -> u32 {
    BRICK_LIST_HEAD + 7 * bricks + 2 * capacity + 1 + 27 * (capacity + 1)
}

fn level_sets(table: u32, levels: &[UVec3]) -> (Vec<u32>, u32) {
    let coarse = &levels[1..];
    let mut words = vec![coarse.len() as u32];
    let mut next = table + 1 + 4 * coarse.len() as u32;

    for d in coarse {
        words.extend([d.x, d.y, d.z, next]);
        next += 1 + 2 * brick_dims(*d).element_product();
    }

    (words, next)
}

impl GpuSolver {
    pub fn new(device: &wgpu::Device, config: SolverConfig, surface: Option<wgpu::Buffer>) -> Self {
        let dims = config.dims;
        let cells = dims.x * dims.y * dims.z;
        let face_counts = [0, 1, 2].map(|a| {
            let mut d = dims;
            d[a] += 1;
            d.x * d.y * d.z
        });
        let faces: u32 = face_counts.iter().sum();
        let node_dims = dims + 1;
        let node_count = node_dims.x * node_dims.y * node_dims.z;
        let reduce_groups = groups(cells, REDUCE_SPAN);
        let body_groups = groups(cells, WG);
        let bricks = brick_dims(dims);
        let brick_count = bricks.x * bricks.y * bricks.z;
        let capacity = config.brick_capacity.unwrap_or(brick_count).clamp(1, brick_count);
        assert!(!config.sparse || bricks.max_element() < 1024, "liquid sparse grid supports at most 1023 bricks per axis");
        let pool = if config.sparse {
            [capacity, (capacity + 1) * BRICK_CELLS, (capacity + 1) * BRICK_FACES, (capacity + 1) * BRICK_NODES]
        } else {
            [0; 4]
        };
        let (cell_slots, face_slots, node_slots) = if config.sparse { (pool[1], pool[2], pool[3]) } else { (cells, faces, node_count) };

        let params = GpuParams {
            dims: [dims.x, dims.y, dims.z, cells],
            face_base: [0, face_counts[0], face_counts[0] + face_counts[1], faces],
            node_dims: [node_dims.x, node_dims.y, node_dims.z, node_count],
            gravity: [config.gravity.x, config.gravity.y, config.gravity.z, 0.0],
            dx: config.dx,
            inv_dx: 1.0 / config.dx,
            density: config.density,
            cfl: config.cfl,
            max_dt: config.max_dt,
            particle_capacity: config.max_particles,
            body_count: 0,
            kernel_radius: config.kernel_radius,
            particle_radius: config.particle_radius,
            margin: config.margin * config.dx,
            frame_dt: 1.0 / 60.0,
            reduction_groups: reduce_groups,
            pool,
            physics: [
                config.surface_tension,
                config.viscosity,
                config.narrow_band.unwrap_or(0.0),
                (dims.as_dvec3().length() * config.dx as f64) as f32,
            ],
        };

        let level_dims = level_dims(dims, config.multigrid_min_dim);
        let (brick_sets, brick_words) = level_sets(level_table(brick_count, capacity), &level_dims);
        let slots = Slots { levels: level_dims.len() as u32 };
        let brick_groups = brick_count * BRICK_CELLS / WG;
        let tiles = body_groups.max(brick_groups);
        let body_base = (tiles * BODY_STRIDE)
            .max(groups(cells, SCAN_BLOCK))
            .max(reduce_groups)
            .max(groups(face_slots, WG));
        let partial_count = body_base + 2 * tiles + 1 + 4 * (tiles + 1);
        let when = |on: bool, count: u32| if on { count } else { 0 };
        let level_sets = config.redistances();
        let diffusion = config.diffuses();
        let particle_bytes = config.max_particles as u64 * std::mem::size_of::<GpuParticle>() as u64;
        let white_bytes = config.whitewater_capacity as u64 * std::mem::size_of::<GpuWhite>() as u64;

        let buffers = SimBuffers {
            params: uniform(device, "liquid params", &params),
            particles: storage(device, "liquid particles", particle_bytes),
            particles_out: storage(device, "liquid particles out", particle_bytes),
            cell_counter: floats(device, "liquid cell counter", cells),
            cell_start: floats(device, "liquid cell start", cells + 1),
            sorted: floats(device, "liquid sorted", 2 * config.max_particles),
            velocity: floats(device, "liquid velocity", face_slots),
            valid: floats(device, "liquid valid", face_slots),
            scratch: floats(device, "liquid scratch", face_slots.max(cell_slots)),
            weights: floats(device, "liquid weights", face_slots),
            owners: floats(device, "liquid owners", face_slots),
            solid_velocity: floats(device, "liquid solid velocity", face_slots),
            phi: floats(device, "liquid phi", cell_slots),
            nodes: floats(device, "liquid nodes", node_slots),
            bodies: storage(device, "liquid bodies", (MAX_BODIES * std::mem::size_of::<GpuBody>()) as u64),
            state: storage(device, "liquid state", std::mem::size_of::<GpuState>() as u64),
            scalars: floats(device, "liquid scalars", SCALAR_COUNT as u32),
            partials: floats(device, "liquid partials", partial_count),
            scratch_valid: floats(device, "liquid scratch valid", face_slots),
            pressure_keep: floats(device, "liquid pressure keep", cell_slots),
            ellipsoids: storage(device, "liquid ellipsoids", (6 * config.max_particles as u64 + cells as u64) * 16),
            surface: surface.unwrap_or_else(|| floats(device, "liquid surface", cells)),
            indirect: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquid indirect"),
                size: (slots.count() as u64 + 1) * 16,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            white: storage(device, "liquid whitewater", white_bytes),
            white_out: storage(device, "liquid whitewater out", white_bytes),
            white_state: storage(device, "liquid whitewater state", std::mem::size_of::<GpuWhiteState>() as u64),
            brick_flags: floats(device, "liquid brick flags", brick_count),
            brick_list: floats(device, "liquid brick list", brick_words),
            band_phi: floats(device, "liquid band phi", when(config.has_band(), cell_slots)),
            signed_distance: floats(device, "liquid signed distance", when(level_sets, cell_slots)),
            sweep_a: floats(device, "liquid sweep a", when(level_sets, cell_slots)),
            sweep_b: floats(device, "liquid sweep b", when(level_sets, cell_slots)),
            kappa: floats(device, "liquid curvature", when(config.has_tension(), cell_slots)),
            band_offsets: floats(device, "liquid band offsets", when(config.has_band(), cells)),
            viscous: ViscousBuffers {
                cells: floats(device, "liquid viscous cells", when(diffusion, cell_slots)),
                edges: floats(device, "liquid viscous edges", when(diffusion, 3 * node_slots)),
                mass: floats(device, "liquid viscous mass", when(diffusion, face_slots)),
                diagonal: floats(device, "liquid viscous diagonal", when(diffusion, face_slots)),
                x: floats(device, "liquid viscous x", when(diffusion, face_slots)),
                r: floats(device, "liquid viscous r", when(diffusion, face_slots)),
                d: floats(device, "liquid viscous d", when(diffusion, face_slots)),
                q: floats(device, "liquid viscous q", when(diffusion, face_slots)),
            },
        };

        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid sim layout"),
            entries: &layout_entries((0..28).filter(|b| *b != 16)),
        });
        let solve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid solve layout"),
            entries: &layout_entries(SOLVE_BINDINGS),
        });
        let level_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid level layout"),
            entries: &layout_entries((0..14).chain([26])),
        });
        let plan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid plan layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let extra_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid extra layout"),
            entries: &layout_entries(buffers.extra().map(|(binding, _)| binding)),
        });

        let coupling_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid coupling layout"),
            entries: &[
                coupling_entry(1, wgpu::BufferBindingType::Uniform),
                coupling_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                coupling_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                coupling_entry(4, wgpu::BufferBindingType::Storage { read_only: false }),
                coupling_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                coupling_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let b = &buffers;
        let sim_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid sim bind"),
            layout: &sim_layout,
            entries: &bind_entries(&[
                (0, &b.params),
                (1, &b.particles),
                (2, &b.particles_out),
                (3, &b.cell_counter),
                (4, &b.cell_start),
                (5, &b.sorted),
                (6, &b.velocity),
                (7, &b.valid),
                (8, &b.scratch),
                (9, &b.weights),
                (10, &b.owners),
                (11, &b.solid_velocity),
                (12, &b.phi),
                (13, &b.nodes),
                (14, &b.bodies),
                (15, &b.state),
                (17, &b.scalars),
                (18, &b.partials),
                (19, &b.scratch_valid),
                (20, &b.pressure_keep),
                (21, &b.ellipsoids),
                (22, &b.surface),
                (23, &b.white),
                (24, &b.white_out),
                (25, &b.white_state),
                (26, &b.brick_flags),
                (27, &b.brick_list),
            ]),
        });
        let solve_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid solve bind"),
            layout: &solve_layout,
            entries: &bind_entries(&[
                (0, &b.params),
                (15, &b.state),
                (17, &b.scalars),
                (18, &b.partials),
                (26, &b.brick_flags),
                (27, &b.brick_list),
            ]),
        });
        let plan_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid plan bind"),
            layout: &plan_layout,
            entries: &bind_entries(&[(0, &b.indirect)]),
        });
        let extra_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid extra bind"),
            layout: &extra_layout,
            entries: &bind_entries(&b.extra()),
        });

        let tall_levels = if config.tall_cells.is_some() { TALL_LEVELS.min(level_dims.len() - 1) } else { 0 };
        let words = device.limits().max_compute_workgroup_storage_size / 4;
        let tail = tail_start(&level_dims, tall_levels, words);
        let tail_size = if tail < level_dims.len() { tail_words(&level_dims[tail..].iter().map(|d| d.element_product()).collect::<Vec<_>>()) } else { 1 };
        let tail_offset = |l: usize| 4 * level_dims[tail..l.max(tail)].iter().map(|d| d.element_product()).sum::<u32>();
        let pack = floats(device, "liquid tail pack", tail_offset(level_dims.len()));

        let levels = level_dims
            .iter()
            .enumerate()
            .map(|(l, d)| {
                let cells = d.x * d.y * d.z;
                let len = if l == 0 { cell_slots } else { cells };
                let level_params = GpuLevelParams {
                    dims: [d.x, d.y, d.z, len],
                    regularization: if l == 0 { 0.0 } else { 1e-4 },
                    groups: groups(cells, REDUCE_SPAN),
                    body_groups: tiles,
                    sparse: u32::from(l == 0 && config.sparse),
                    bricks: if l > 0 && config.sparse { brick_sets[4 * l] } else { 0 },
                    tall: u32::from(l < tall_levels),
                    pack: tail_offset(l),
                    body_base,
                };
                let params = uniform(device, "liquid level params", &level_params);
                let phi = floats(device, "level phi", len);
                let columns = if l == 0 { 3 * d.x * d.z } else { d.x * d.z };
                let coef = floats(device, "level coef", 3 * len + when(config.open_floor.is_some(), columns));
                let active = floats(device, "level active", len);
                let diag = floats(device, "level diag", len);
                let off = floats(device, "level off", 3 * len);
                let x = floats(device, "level x", len);
                let bb = floats(device, "level b", len);
                let r = floats(device, "level r", len);
                let z = floats(device, "level z", len);
                let dd = floats(device, "level d", len);
                let q = floats(device, "level q", len);
                let t = floats(device, "level t", len);
                let stride = if l < TALL_LEVELS { 1 + TALL_STRIDE } else { 1 };
                let tall = floats(device, "level tall", when(config.tall_cells.is_some(), stride * d.x * d.z));

                let e = floats(device, "level e", when(l == 0, len));
                let group = |label: &str, vectors: [&wgpu::Buffer; 7]| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(label),
                        layout: &level_layout,
                        entries: &bind_entries(&[
                            (0, &params),
                            (1, &phi),
                            (2, &coef),
                            (3, &active),
                            (4, &diag),
                            (5, &off),
                            (6, vectors[0]),
                            (7, vectors[1]),
                            (8, vectors[2]),
                            (9, vectors[3]),
                            (10, vectors[4]),
                            (11, vectors[5]),
                            (12, &pack),
                            (13, vectors[6]),
                            (26, &tall),
                        ]),
                    })
                };
                let bind = group("liquid level bind", [&x, &bb, &r, &z, &dd, &q, &e]);
                let (multigrid_bind, swapped_bind) = if l == 0 {
                    (group("liquid level multigrid bind", [&z, &r, &t, &x, &dd, &q, &e]), group("liquid level swapped bind", [&x, &bb, &r, &z, &e, &q, &dd]))
                } else {
                    (bind.clone(), bind.clone())
                };

                LevelBuffers {
                    dims: *d,
                    len,
                    params,
                    phi,
                    coef,
                    active,
                    diag,
                    off,
                    x,
                    b: bb,
                    r,
                    z,
                    d: dd,
                    q,
                    t,
                    tall,
                    e,
                    bind,
                    multigrid_bind,
                    swapped_bind,
                }
            })
            .collect();

        let subgroups = config.subgroups && device.features().contains(wgpu::Features::SUBGROUP);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("liquid solver"),
            source: wgpu::ShaderSource::Wgsl(shader_source(subgroups, tail_size).into()),
        });

        let layouts = HashMap::from([
            (Layout::Sim, pipeline_layout(device, &[Some(&sim_layout)])),
            (Layout::Level, pipeline_layout(device, &[Some(&sim_layout), Some(&level_layout)])),
            (
                Layout::Pair,
                pipeline_layout(device, &[Some(&sim_layout), Some(&level_layout), Some(&level_layout)]),
            ),
            (Layout::Plan, pipeline_layout(device, &[Some(&sim_layout), None, None, Some(&plan_layout)])),
            (
                Layout::Coupling,
                pipeline_layout(device, &[Some(&sim_layout), None, None, Some(&coupling_layout)]),
            ),
            (Layout::Solve, pipeline_layout(device, &[Some(&solve_layout)])),
            (Layout::SolveLevel, pipeline_layout(device, &[Some(&solve_layout), Some(&level_layout)])),
            (
                Layout::SolvePair,
                pipeline_layout(device, &[Some(&solve_layout), Some(&level_layout), Some(&level_layout)]),
            ),
            (Layout::SolvePlan, pipeline_layout(device, &[Some(&solve_layout), None, None, Some(&plan_layout)])),
            (
                Layout::SolveLevelPlan,
                pipeline_layout(device, &[Some(&solve_layout), Some(&level_layout), None, Some(&plan_layout)]),
            ),
            (Layout::Extra, pipeline_layout(device, &[Some(&sim_layout), None, Some(&extra_layout)])),
            (
                Layout::LevelExtra,
                pipeline_layout(device, &[Some(&sim_layout), Some(&level_layout), Some(&extra_layout)]),
            ),
            (Layout::SolveExtra, pipeline_layout(device, &[Some(&solve_layout), None, Some(&extra_layout)])),
            (
                Layout::LevelCoupling,
                pipeline_layout(device, &[Some(&sim_layout), Some(&level_layout), None, Some(&coupling_layout)]),
            ),
            (
                Layout::Link,
                pipeline_layout(device, &[Some(&sim_layout), Some(&level_layout), Some(&extra_layout), Some(&plan_layout)]),
            ),
        ]);

        let plan_constants = [
            ("SLOT_COUNT", slots.count() as f64),
            ("SOLVE_FIRST", slots.solve_first() as f64),
            ("PARTICLE_SLOT", Slots::PARTICLES as f64),
            ("GRID_SLOT", Slots::GRID as f64),
            ("CELL_SLOT", Slots::CELLS as f64),
            ("FACE_SLOT", Slots::FACES as f64),
            ("NODE_SLOT", Slots::NODES as f64),
            ("CLEAR_SLOT", Slots::CLEAR as f64),
            ("LEVEL_SLOT", slots.level(0) as f64),
            ("SOLVE_LEVEL_SLOT", slots.solve_level(0) as f64),
            ("SOLVE_FACE_SLOT", slots.solve_faces() as f64),
            ("BODY_SLOT", Slots::BODIES as f64),
            ("SOLVE_BODY_SLOT", slots.solve_bodies() as f64),
        ];

        let mut pipelines = HashMap::new();

        for (key, entry, layout, constants) in pipeline_specs(&plan_constants, &config, words) {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&key),
                layout: Some(&layouts[&layout]),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
            pipelines.insert(key, (pipeline, layout));
        }

        Self {
            config,
            subgroups,
            params,
            slots,
            buffers,
            levels,
            pack,
            tail,
            sim_bind,
            solve_bind,
            plan_bind,
            extra_bind,
            coupling_layout,
            coupling: None,
            pipelines,
        }
    }

    pub fn initialize(&self, queue: &wgpu::Queue) {
        let slots = self.slots;
        let count = slots.count() as usize;
        let mut indirect = vec![1u32; count * 4];
        let cells = self.params.dims[3];

        let mut full = vec![0u32; count];
        full[Slots::PARTICLES as usize] = groups(self.config.max_particles, WG);
        full[Slots::CELLS as usize] = groups(cells, WG);
        full[Slots::FACES as usize] = groups(self.params.face_base[3], WG);
        full[Slots::NODES as usize] = groups(self.params.node_dims[3], WG);
        full[Slots::SINGLE as usize] = 1;
        full[Slots::SCAN as usize] = groups(cells, SCAN_BLOCK);
        full[Slots::REDUCE as usize] = groups(cells, REDUCE_SPAN);
        full[slots.solve_single() as usize] = 1;
        full[slots.solve_reduce() as usize] = groups(cells, REDUCE_SPAN);
        full[slots.solve_faces() as usize] = groups(self.params.face_base[3], WG);
        full[slots.solve_grid() as usize] = groups(cells, WG);

        full[Slots::GRID as usize] = groups(cells, WG);
        full[Slots::BRICKS as usize] = groups(self.brick_count() * BRICK_ROWS, WG);

        for (l, level) in self.levels.iter().enumerate() {
            full[slots.level(l) as usize] = groups(level.len, WG);
            full[slots.solve_level(l) as usize] = groups(level.len, WG);

            if !self.config.sparse {
                full[slots.solve_block(l) as usize] = brick_dims(level.dims).element_product();
            }
        }

        if self.config.sparse {
            for s in [Slots::CELLS, Slots::FACES, Slots::NODES, slots.level(0), slots.solve_level(0), slots.solve_faces()] {
                full[s as usize] = 0;
            }
        }

        if self.config.sparse {
            self.initialize_pool(queue);
        }

        for (s, f) in full.iter().enumerate() {
            indirect[3 * s] = 0;
            indirect[3 * count + s] = *f;
        }

        queue.write_buffer(&self.buffers.indirect, 0, bytemuck::cast_slice(&indirect));

        let mut scalars = vec![0f32; SCALAR_COUNT];
        scalars[S_TOLERANCE] = self.config.tolerance;
        queue.write_buffer(&self.buffers.scalars, 0, bytemuck::cast_slice(&scalars));
        queue.write_buffer(&self.buffers.state, 0, bytemuck::bytes_of(&GpuState::default()));

        if self.config.has_band() {
            let band = &self.buffers.band_phi;
            let count = ((band.size() - GUARD_BYTES) / 4) as usize;
            queue.write_buffer(band, 0, bytemuck::cast_slice(&vec![3.0e38f32; count]));
        }
    }

    pub fn upload_particles(&self, queue: &wgpu::Queue, particles: &[GpuParticle]) {
        let count = particles.len().min(self.config.max_particles as usize);
        queue.write_buffer(&self.buffers.particles, 0, bytemuck::cast_slice(&particles[..count]));
        queue.write_buffer(&self.buffers.state, 16, bytemuck::bytes_of(&(count as u32)));
    }

    pub fn upload_bodies(&mut self, queue: &wgpu::Queue, bodies: &[GpuBody]) {
        let count = bodies.len().min(MAX_BODIES);
        let mut slots = [GpuBody::default(); MAX_BODIES];
        slots[..count].copy_from_slice(&bodies[..count]);
        queue.write_buffer(&self.buffers.bodies, 0, bytemuck::cast_slice(&slots));
        self.params.body_count = count as u32;
        queue.write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn set_frame_dt(&mut self, queue: &wgpu::Queue, frame_dt: f32) {
        self.params.frame_dt = frame_dt;
        queue.write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(&self.params));
    }

    fn initialize_pool(&self, queue: &wgpu::Queue) {
        let bricks = self.brick_count();
        let capacity = self.params.pool[0];
        let free = (BRICK_LIST_HEAD + 5 * bricks + capacity + 1) as usize;
        let table = level_table(bricks, capacity);
        let level_dims: Vec<UVec3> = self.levels.iter().map(|l| l.dims).collect();
        let (sets, words) = level_sets(table, &level_dims);
        let mut list = vec![0u32; words as usize];
        list[2] = capacity;

        for (i, slot) in list[free..free + capacity as usize].iter_mut().enumerate() {
            *slot = i as u32 + 1;
        }

        list[table as usize..table as usize + sets.len()].copy_from_slice(&sets);

        let b = &self.buffers;
        let empty = 3.0 * self.config.dx;
        queue.write_buffer(&b.brick_list, 0, bytemuck::cast_slice(&list));
        queue.write_buffer(&b.brick_flags, 0, bytemuck::cast_slice(&vec![0u32; bricks as usize]));
        queue.write_buffer(&b.phi, 0, bytemuck::cast_slice(&[empty; BRICK_CELLS as usize]));
        queue.write_buffer(&self.levels[0].phi, 0, bytemuck::cast_slice(&[empty; BRICK_CELLS as usize]));
        queue.write_buffer(&b.weights, 0, bytemuck::cast_slice(&[1.0f32; BRICK_FACES as usize]));
        queue.write_buffer(&b.owners, 0, bytemuck::cast_slice(&[-1i32; BRICK_FACES as usize]));
        queue.write_buffer(&b.nodes, 0, bytemuck::cast_slice(&[3.0e38f32; BRICK_NODES as usize]));

        if self.config.redistances() {
            let far = self.params.physics[3];

            for (buffer, value) in [(&b.signed_distance, far), (&b.sweep_a, 3.0e38), (&b.sweep_b, 3.0e38)] {
                queue.write_buffer(buffer, 0, bytemuck::cast_slice(&[value; BRICK_CELLS as usize]));
            }
        }
    }

    pub fn slot_table(&self) -> std::ops::Range<u64> {
        let bricks = self.brick_count() as u64;
        let start = (BRICK_LIST_HEAD as u64 + 4 * bricks) * 4;

        start..start + bricks * 4
    }

    pub fn pooled_cell(&self, slots: &[u32], c: UVec3) -> usize {
        let d = self.config.dims;

        if !self.config.sparse {
            return (c.x + d.x * (c.y + d.y * c.z)) as usize;
        }

        let local = c % BRICK;
        let slot = slots[self.brick_index(c)];

        (slot * BRICK_CELLS + local.x + BRICK * (local.y + BRICK * local.z)) as usize
    }

    pub fn pooled_face(&self, slots: &[u32], axis: usize, f: UVec3) -> usize {
        if !self.config.sparse {
            let mut d = self.config.dims;
            d[axis] += 1;

            return (self.params.face_base[axis] + f.x + d.x * (f.y + d.y * f.z)) as usize;
        }

        let owner = f.min(self.config.dims - 1);
        let local = f - owner / BRICK * BRICK;
        let mut size = UVec3::splat(BRICK);
        size[axis] += 1;
        let slot = slots[self.brick_index(owner)];

        (slot * BRICK_FACES + axis as u32 * (BRICK_FACES / 3) + local.x + size.x * (local.y + size.y * local.z)) as usize
    }

    fn brick_index(&self, c: UVec3) -> usize {
        let d = brick_dims(self.config.dims);
        let b = c / BRICK;

        (b.x + d.x * (b.y + d.y * b.z)) as usize
    }

    pub fn brick_count(&self) -> u32 {
        let b = brick_dims(self.config.dims);
        b.x * b.y * b.z
    }

    pub(crate) fn prepare(&self, pass: &mut wgpu::ComputePass<'_>, key: &str, bind: Bind) {
        let (pipeline, layout) = self.pipelines.get(key).unwrap_or_else(|| panic!("liquid pipeline {key} missing"));
        let core = if layout.solve() { &self.solve_bind } else { &self.sim_bind };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, core, &[]);

        if layout.extra() {
            pass.set_bind_group(2, &self.extra_bind, &[]);
        }

        match bind {
            Bind::Sim => {}
            Bind::Level(l) => pass.set_bind_group(1, &self.levels[l].bind, &[]),
            Bind::Multigrid(l) => pass.set_bind_group(1, &self.levels[l].multigrid_bind, &[]),
            Bind::Swapped(l) => pass.set_bind_group(1, &self.levels[l].swapped_bind, &[]),
            Bind::Pair(l) => {
                pass.set_bind_group(1, &self.levels[l].multigrid_bind, &[]);
                pass.set_bind_group(2, &self.levels[l + 1].bind, &[]);
            }
            Bind::Plan => pass.set_bind_group(3, &self.plan_bind, &[]),
            Bind::LevelPlan(l) => {
                pass.set_bind_group(1, &self.levels[l].bind, &[]);
                pass.set_bind_group(3, &self.plan_bind, &[]);
            }
            Bind::Coupling => {
                if let Some(coupling) = &self.coupling {
                    pass.set_bind_group(3, &coupling.bind, &[]);
                }
            }
            Bind::LevelCoupling(l) => {
                pass.set_bind_group(1, &self.levels[l].bind, &[]);

                if let Some(coupling) = &self.coupling {
                    pass.set_bind_group(3, &coupling.bind, &[]);
                }
            }
        }
    }

    pub fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, key: &str, bind: Bind, slot: u32) {
        self.prepare(pass, key, bind);
        pass.dispatch_workgroups_indirect(&self.buffers.indirect, slot as u64 * 12);
    }

    pub fn dispatch_direct(&self, pass: &mut wgpu::ComputePass<'_>, key: &str, bind: Bind, groups: u32) {
        self.prepare(pass, key, bind);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    pub fn encode_frame(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>) {
        self.encode_frame_within(encoder, profiler, self.config.max_substeps);
    }

    pub fn encode_frame_within(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>, budget: u32) {
        {
            let mut pass = self.section(encoder, profiler, Section::Plan, 0);
            self.dispatch_direct(&mut pass, "begin_frame", Bind::Plan, 1);
        }

        for substep in 0..budget.clamp(1, self.config.max_substeps) {
            {
                let mut pass = self.section(encoder, profiler, Section::Sort, substep);
                if substep > 0 {
                    self.dispatch(&mut pass, "reset_speed", Bind::Sim, Slots::SINGLE);
                }

                if self.is_coupled() {
                    self.dispatch(&mut pass, "measure_inflow", Bind::Coupling, Slots::PARTICLES);
                    self.dispatch_direct(&mut pass, "plan_substep#grow", Bind::Plan, 1);
                } else {
                    self.dispatch(&mut pass, "measure_flow", Bind::Sim, Slots::PARTICLES);
                    self.dispatch_direct(&mut pass, "plan_substep", Bind::Plan, 1);
                }
                self.encode_sort(&mut pass);
            }

            self.encode_substep(encoder, profiler, substep);
        }

        let mut pass = self.section(encoder, profiler, Section::Surface, 0);
        self.dispatch_direct(&mut pass, "open_all", Bind::Plan, 1);
        self.encode_sorted(&mut pass, "sort_copy#surface");

        if !self.config.has_band() {
            self.dispatch(&mut pass, "liquid_phi", Bind::Sim, Slots::GRID);
        }

        self.encode_surface_field(&mut pass);
        self.encode_columns(&mut pass);
        drop(pass);
        self.encode_whitewater(encoder, profiler);
    }

    pub(crate) fn section<'e>(
        &self,
        encoder: &'e mut wgpu::CommandEncoder,
        profiler: Option<&'e Profiler>,
        section: Section,
        substep: u32,
    ) -> wgpu::ComputePass<'e> {
        let timestamp_writes = profiler.map(|p| p.writes(section, substep));

        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(section.label()), timestamp_writes })
    }

    pub fn encode_sort(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.encode_sorted(pass, "sort_copy");
    }

    fn encode_sorted(&self, pass: &mut wgpu::ComputePass<'_>, copy: &str) {
        self.dispatch(pass, "sort_count", Bind::Sim, Slots::PARTICLES);
        self.dispatch_direct(pass, "sort_scan", Bind::Plan, groups(self.params.dims[3], SCAN_BLOCK));
        self.dispatch_direct(pass, "sort_link", Bind::LevelPlan(0), groups(27 * self.brick_count(), WG).max(64));
        self.dispatch(pass, "scatter_particles", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, "rank_particles", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, copy, Bind::Sim, Slots::PARTICLES);
    }

    pub(crate) fn phi_groups(&self) -> u32 {
        let cells = if self.config.sparse { self.params.pool[0] * BRICK_CELLS } else { self.params.dims[3] };

        groups(cells, 28).min(MAX_GROUPS)
    }

    fn p2g_groups(&self) -> u32 {
        let nodes = if self.config.sparse { self.params.pool[0] * BRICK_NODES } else { self.params.node_dims[3] };

        groups(16 * nodes, WG).min(MAX_GROUPS)
    }

    pub fn encode_extrapolate(&self, pass: &mut wgpu::ComputePass<'_>, layers: u32) {
        for _ in 0..layers {
            self.dispatch(pass, "extrapolate_pass", Bind::Sim, Slots::FACES);
            self.dispatch(pass, "copy_scratch", Bind::Sim, Slots::FACES);
        }
    }

    fn extrapolated(&self, pass: &mut wgpu::ComputePass<'_>, layers: u32, last: &str) -> bool {
        if layers == 0 || layers % 2 == 1 {
            self.encode_extrapolate(pass, layers);
            return false;
        }

        for k in 0..layers - 1 {
            self.dispatch(pass, if k % 2 == 0 { "extrapolate_pass" } else { "extrapolate_back" }, Bind::Sim, Slots::FACES);
        }

        self.dispatch(pass, last, Bind::Sim, Slots::FACES);

        true
    }

    pub fn encode_geometry(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.encode_phi(pass);
        self.dispatch(pass, "face_weights", Bind::Sim, Slots::FACES);
        self.encode_boundary(pass);
    }

    fn encode_phi(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.config.has_band() {
            self.dispatch_direct(pass, "band_liquid_phi", Bind::Sim, self.phi_groups());
            self.dispatch(pass, "solid_nodes", Bind::Sim, Slots::NODES);
            self.encode_redistance(pass);
            self.dispatch(pass, "band_adopt", Bind::Sim, Slots::CELLS);
        } else {
            self.dispatch(pass, "liquid_phi", Bind::Sim, Slots::GRID);
            self.dispatch(pass, "solid_nodes", Bind::Sim, Slots::NODES);
            self.dispatch(pass, "mark_solid_cells", Bind::Sim, Slots::CELLS);
        }

        self.dispatch(pass, "extend_into_solid", Bind::Sim, Slots::CELLS);
        self.dispatch(pass, "extend_from_scratch", Bind::Sim, Slots::CELLS);
        self.dispatch(pass, "extend_into_solid", Bind::Sim, Slots::CELLS);
        self.dispatch(pass, "settle_from_scratch", Bind::Sim, Slots::CELLS);
    }

    pub fn encode_hierarchy(&self, pass: &mut wgpu::ComputePass<'_>) {
        let band = self.config.smooths_boundary();

        for l in 0..self.levels.len() {
            let slot = self.slots.level(l);

            if l == 0 {
                self.dispatch(pass, "level_from_fine", Bind::Level(0), slot);
            } else {
                self.dispatch(pass, "coarsen", Bind::Pair(l - 1), slot);
            }

            self.dispatch(pass, "level_classify", Bind::Level(l), slot);
            self.dispatch(pass, "level_assemble", Bind::Level(l), slot);

            if band {
                self.dispatch(pass, "level_band", Bind::Level(l), slot);
            }
        }

        if self.config.open_floor.is_some() {
            let columns = groups(self.config.dims.x * self.config.dims.z, WG);
            self.dispatch_direct(pass, "floor_surface", Bind::Level(0), columns);
            self.dispatch_direct(pass, "floor_mean", Bind::Level(0), 1);
            self.dispatch_direct(pass, "floor_ghost", Bind::Level(0), columns);
        }

        if self.levels.len() > 1 {
            for l in self.tail..self.levels.len() {
                self.dispatch(pass, "tail_pack", Bind::Level(l), self.slots.level(l));
            }
        }
    }


    fn smooth(&self, pass: &mut wgpu::ComputePass<'_>, l: usize, colors: [u32; 2], lines: bool) {
        if lines && l < self.tall_levels() {
            self.relax_lines(pass, l, colors);
        } else {
            self.relax(pass, "mg_smooth", l, colors);
        }
    }

    fn relax(&self, pass: &mut wgpu::ComputePass<'_>, kind: &str, l: usize, colors: [u32; 2]) {
        let bind = Bind::Multigrid(l);
        let slot = self.slots.solve_level(l);

        for color in colors {
            self.dispatch(pass, &self.variant(&format!("{kind}#{color}"), l), bind, slot);
        }
    }

    fn variant(&self, key: &str, l: usize) -> String {
        if self.has_tall() && l >= self.tall_levels() { format!("{key}~plain") } else { key.to_string() }
    }

    fn boundary_sweeps(&self) -> u32 {
        if self.config.smooths_boundary() { self.config.boundary_sweeps } else { 0 }
    }

    fn plain(&self, l: usize) -> bool {
        self.config.smoothing > 0 && !(self.has_tall() && l < self.tall_levels())
    }

    pub fn encode_vcycle(&self, pass: &mut wgpu::ComputePass<'_>, lines: bool) {
        let last = self.levels.len() - 1;

        if last == 0 || !self.plain(0) {
            self.dispatch(pass, "mg_zero", Bind::Multigrid(0), self.slots.solve_level(0));
        }

        if last == 0 {
            for _ in 0..self.config.coarse_sweeps {
                self.smooth(pass, last, [0, 1], lines);
            }

            for _ in 0..self.config.coarse_sweeps {
                self.smooth(pass, last, [1, 0], lines);
            }

            return;
        }

        for l in 0..self.tail {
            for s in 0..self.config.smoothing {
                if s == 0 && self.plain(l) {
                    self.dispatch(pass, &self.variant("mg_presmooth", l), Bind::Multigrid(l), self.slots.solve_level(l));
                } else {
                    self.smooth(pass, l, [0, 1], lines);
                }
            }

            for _ in 0..self.boundary_sweeps() {
                self.relax(pass, "mg_band", l, [0, 1]);
            }

            self.dispatch(pass, &self.variant("mg_residual", l), Bind::Multigrid(l), self.slots.solve_level(l));
            self.dispatch(pass, &self.variant("mg_restrict", l + 1), Bind::Pair(l), self.slots.solve_level(l + 1));
            self.gather_tall(pass, l + 1);
        }

        self.dispatch(pass, "mg_tail", Bind::Level(self.tail), self.slots.solve_level(last));

        for l in (0..self.tail).rev() {
            let correct = self.plain(l) && self.boundary_sweeps() == 0;
            let entry = if correct { "mg_correct" } else { "mg_prolong" };
            self.dispatch(pass, &self.variant(entry, l), Bind::Pair(l), self.slots.solve_level(l));

            for _ in 0..self.boundary_sweeps() {
                self.relax(pass, "mg_band", l, [1, 0]);
            }

            for s in 0..self.config.smoothing {
                let black = !(s == 0 && correct);
                let finish = l == 0 && s + 1 == self.config.smoothing && self.fuses_dot(lines);

                if black && !finish {
                    self.smooth(pass, l, [1, 0], lines);
                    continue;
                }

                if black {
                    self.dispatch(pass, &self.variant("mg_smooth#1", l), Bind::Multigrid(l), self.slots.solve_level(l));
                }

                if finish {
                    self.dispatch(pass, "mg_finish", Bind::Multigrid(0), self.slots.solve_grid());
                } else {
                    self.dispatch(pass, &self.variant("mg_smooth#0", l), Bind::Multigrid(l), self.slots.solve_level(l));
                }
            }
        }
    }

    fn fuses_dot(&self, lines: bool) -> bool {
        self.config.smoothing > 0 && self.levels.len() > 1 && !(lines && self.has_tall())
    }

    fn encode_precondition(&self, pass: &mut wgpu::ComputePass<'_>, lines: bool) {
        if self.blocks() {
            self.encode_blocks(pass, None, false);
            return;
        }

        self.encode_vcycle(pass, lines);

        if !self.fuses_dot(lines) {
            self.dispatch(pass, "pcg_dot", Bind::Level(0), self.slots.solve_grid());
        }
    }

    pub fn blocks(&self) -> bool {
        self.config.block_smoothing && self.levels.len() > 1 && !self.has_tall() && !self.config.smooths_boundary()
    }

    fn encode_blocks(&self, pass: &mut wgpu::ComputePass<'_>, update: Option<(bool, bool)>, check: bool) {
        let down = match update {
            Some((swap, coupled)) => format!("mg_block_down#update{}{}", if swap { "#swap" } else { "" }, if coupled { "#coupled" } else { "" }),
            None => "mg_block_down".to_string(),
        };
        let last = self.levels.len() - 1;
        self.dispatch(pass, &down, Bind::Multigrid(0), self.slots.solve_block(0));

        if check {
            self.dispatch_direct(pass, "pcg_check", Bind::LevelPlan(0), 1);
        }

        self.dispatch(pass, "mg_block_complete", Bind::Multigrid(0), self.slots.solve_level(0));

        for l in 1..self.tail {
            self.dispatch(pass, "mg_block_coarse", Bind::Pair(l - 1), self.slots.solve_block(l));
            self.dispatch(pass, "mg_block_complete", Bind::Multigrid(l), self.slots.solve_level(l));
        }

        self.dispatch(pass, "mg_block_restrict", Bind::Pair(self.tail - 1), self.slots.solve_block(self.tail));
        self.dispatch(pass, "mg_tail", Bind::Level(self.tail), self.slots.solve_level(last));

        for l in (1..self.tail).rev() {
            self.dispatch(pass, "mg_block_up", Bind::Pair(l), self.slots.solve_block(l));
        }

        self.dispatch(pass, "mg_block_finish", Bind::Pair(0), self.slots.solve_block(0));
    }

    pub fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, coupling: bool, iterations: u32, lines: bool) {
        let cells = self.slots.solve_level(0);
        let grid = self.slots.solve_grid();
        let tiles = self.slots.solve_bodies();
        let coupled = |key: &str| if coupling { format!("{key}#coupled") } else { key.to_string() };

        if coupling {
            self.dispatch(pass, "body_mark", Bind::Level(0), Slots::CELLS);
        }

        // wgpu skips rebinding group 3 when group 1 becomes a layout hole, so plan dispatches here keep group 1 bound
        self.dispatch_direct(pass, &coupled("pcg_open"), Bind::LevelPlan(0), 1);
        self.dispatch(pass, &coupled("pcg_init_residual"), Bind::Level(0), cells);

        if coupling {
            self.dispatch(pass, "body_gather#0", Bind::Level(0), tiles);
            self.dispatch_direct(pass, "pcg_scalar#lead#coupled", Bind::Level(0), 1);
            self.dispatch(pass, "pcg_update#lead#coupled", Bind::Level(0), cells);
        }

        let checked = self.config.tolerance > 0.0;

        if checked {
            self.dispatch_direct(pass, "pcg_check#lead", Bind::LevelPlan(0), 1);
        }

        if iterations > 0 {
            self.encode_precondition(pass, lines);
        }

        for i in 0..iterations {
            let bind = if i % 2 == 0 { Bind::Level(0) } else { Bind::Swapped(0) };
            let last = i + 1 == iterations;

            self.dispatch(pass, if i == 0 { "pcg_apply#lead" } else { "pcg_apply" }, bind, grid);

            if coupling {
                self.dispatch(pass, "body_gather#1", bind, tiles);
            }

            if self.blocks() && !last {
                self.encode_blocks(pass, Some((i % 2 == 1, coupling)), checked);
                continue;
            }

            self.dispatch_direct(pass, &coupled("pcg_scalar"), bind, 1);
            self.dispatch(pass, &coupled("pcg_update"), bind, cells);

            if checked {
                self.dispatch_direct(pass, "pcg_check", Bind::LevelPlan(0), 1);
            }

            if !last {
                self.encode_precondition(pass, lines);
            }
        }

        if !checked {
            self.dispatch_direct(pass, &format!("pcg_check#tally{iterations}"), Bind::LevelPlan(0), 1);
        }
    }

    pub fn encode_substep(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>, substep: u32) {
        let has_bodies = self.params.body_count > 0;
        let band = self.config.has_band();
        let tension = self.config.has_tension();

        {
            let pass = &mut self.section(encoder, profiler, Section::Transfer, substep);

            if band {
                self.encode_band_advection(pass);
            }

            if band {
                self.dispatch_direct(pass, "p2g", Bind::Sim, self.p2g_groups());
                self.dispatch(pass, "redistance_seed", Bind::Sim, Slots::CELLS);
                self.encode_seeded_redistance(pass, "redistance_project#adopt");
                self.dispatch(pass, "extend_from_scratch", Bind::Sim, Slots::CELLS);
                self.dispatch(pass, "extend_into_solid", Bind::Sim, Slots::CELLS);
                self.dispatch(pass, "extend_settle", Bind::Sim, Slots::CELLS);
                self.dispatch(pass, "band_pick", Bind::Sim, Slots::FACES);
            } else {
                self.dispatch_direct(pass, "p2g", Bind::Sim, self.p2g_groups());
            }

            let finish = if band { "faces" } else { "bodies" };
            self.dispatch(pass, "extrapolate_pass", Bind::Sim, Slots::FACES);

            if self.is_coupled() {
                self.dispatch(pass, &format!("extrapolate_inflow#{finish}"), Bind::Coupling, Slots::FACES);
            } else {
                self.dispatch(pass, &format!("extrapolate_back#{finish}"), Bind::Sim, Slots::FACES);
            }
        }

        {
            let pass = &mut self.section(encoder, profiler, Section::Geometry, substep);

            if !band {
                self.encode_geometry(pass);
            }

            self.dispatch(pass, if band { "set_pressure_scale_gravity" } else { "set_pressure_scale" }, Bind::Sim, Slots::SINGLE);
            self.encode_hierarchy(pass);
            self.encode_tall(pass);
            let floor = self.config.open_floor.is_some();
            let volume = band && !floor && (!self.is_coupled() || self.tiers_linked());

            if volume {
                self.dispatch(pass, "divergence_volume", Bind::Level(0), Slots::CELLS);

                if self.tiers_linked() {
                    self.dispatch(pass, "band_volume_target", Bind::Coupling, Slots::SINGLE);
                } else {
                    self.dispatch(pass, "band_volume_final", Bind::Sim, Slots::SINGLE);
                }

                self.dispatch(pass, "band_volume_rhs", Bind::Level(0), Slots::CELLS);
            } else {
                self.dispatch(pass, "divergence", Bind::Level(0), Slots::CELLS);
            }

            if floor {
                let columns = groups(self.config.dims.x * self.config.dims.z, WG);
                self.dispatch_direct(pass, "floor_rhs", Bind::Level(0), columns);
            }

            if tension {
                self.encode_redistance(pass);
                self.encode_curvature(pass);
                self.dispatch(pass, "jump_rhs", Bind::Level(0), Slots::CELLS);
            }

            self.encode_tall_rhs(pass);
        }

        let mut held = self.section(encoder, profiler, Section::Pressure, substep);
        let pass = &mut held;
        self.encode_solve(pass, has_bodies, self.config.pressure_iterations, self.has_tall());

        let fused = !tension && !self.has_tall();

        if has_bodies {
            self.dispatch(pass, "body_gather#0", Bind::Level(0), Slots::BODIES);
            self.dispatch(pass, "body_impulse", Bind::Level(0), Slots::SINGLE);

            if !fused {
                self.dispatch(pass, "refresh_solid_velocity", Bind::Sim, Slots::FACES);
                self.encode_boundary(pass);
            }
        }

        if fused && has_bodies && self.is_coupled() {
            self.dispatch(pass, &format!("gradient_inflow#{}", u8::from(band)), Bind::LevelCoupling(0), Slots::FACES);
        } else if fused {
            self.dispatch(pass, &format!("gradient_flow#{}{}", u8::from(has_bodies), u8::from(band)), Bind::Level(0), Slots::FACES);
        } else {
            self.encode_tall_expand(pass);
            self.dispatch(pass, "apply_gradient", Bind::Level(0), Slots::FACES);

            if tension {
                self.dispatch(pass, "jump_gradient", Bind::Level(0), Slots::FACES);
            }

            if band {
                self.dispatch(pass, "blend_partial", Bind::Sim, Slots::FACES);
            }
        }

        if !self.extrapolated(pass, self.config.extrapolation_layers, "extrapolate_back#constrain") {
            self.dispatch(pass, "constrain_solid", Bind::Sim, Slots::FACES);
        }

        drop(held);

        if self.config.diffuses() {
            let pass = &mut self.section(encoder, profiler, Section::Viscosity, substep);

            if tension {
                self.encode_diffuse(pass, Diffusion::Capillary);
            }

            if self.config.has_viscosity() {
                self.encode_diffuse(pass, Diffusion::Viscosity);
            }
        }

        let mut held = self.section(encoder, profiler, Section::Advect, substep);
        let pass = &mut held;
        let floor = self.config.open_floor.is_some();

        if floor && !band {
            let spawned = FLOOR_SPAWN * self.config.dims.x * self.config.dims.z;
            self.dispatch_direct(pass, "floor_spawn", Bind::Sim, groups(spawned, WG));
            self.dispatch_direct(pass, "floor_grow", Bind::Plan, 1);
        }

        if band {
            self.dispatch(pass, "g2p#delete", Bind::Sim, Slots::PARTICLES);
            self.encode_reseed(pass);
        } else {
            self.dispatch(pass, "g2p", Bind::Sim, Slots::PARTICLES);
        }

        self.dispatch(pass, "advect#bodies", Bind::Sim, Slots::PARTICLES);

        if floor {
            self.dispatch(pass, "floor_cull", Bind::Sim, Slots::PARTICLES);
        }

        if self.is_coupled() {
            self.dispatch(pass, "keep_inflow", Bind::Coupling, Slots::PARTICLES);
        } else {
            self.dispatch(pass, "keep_inside", Bind::Sim, Slots::PARTICLES);
        }

        if band {
            self.dispatch(pass, "band_blend#store", Bind::Sim, Slots::FACES);
        }

        drop(held);

        if self.config.density_projection && !band {
            let pass = &mut self.section(encoder, profiler, Section::Density, substep);
            self.encode_density_projection(pass);
        }
    }

    pub fn encode_density_projection(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.dispatch(pass, "save_pressure", Bind::Level(0), Slots::CELLS);
        self.encode_sort(pass);
        self.encode_geometry(pass);
        self.dispatch(pass, "set_density_scale", Bind::Sim, Slots::SINGLE);
        self.encode_hierarchy(pass);
        self.encode_tall_reset(pass);
        self.dispatch(pass, "density_error", Bind::Level(0), Slots::GRID);
        self.encode_solve(pass, false, self.config.density_iterations, false);
        self.dispatch(pass, "zero_velocity", Bind::Sim, Slots::FACES);
        self.dispatch(pass, "apply_gradient", Bind::Level(0), Slots::FACES);

        if !self.extrapolated(pass, 2, "extrapolate_back#zero") {
            self.dispatch(pass, "constrain_zero", Bind::Sim, Slots::FACES);
        }
        self.dispatch(pass, "displace", Bind::Sim, Slots::PARTICLES);

        if self.config.open_floor.is_some() {
            self.dispatch(pass, "floor_cull", Bind::Sim, Slots::PARTICLES);
        }

        self.dispatch(pass, "keep_inside", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, "restore_pressure", Bind::Level(0), Slots::CELLS);
    }
}

fn pipeline_layout(device: &wgpu::Device, layouts: &[Option<&wgpu::BindGroupLayout>]) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("liquid pipeline layout"),
        bind_group_layouts: layouts,
        immediate_size: 0,
    })
}

type Spec<'a> = (String, &'static str, Layout, Vec<(&'a str, f64)>);

const PLAIN: [&str; 7] = ["mg_smooth#", "mg_band#", "mg_residual", "mg_restrict", "mg_prolong", "mg_presmooth", "mg_correct"];

fn pipeline_specs<'a>(plan: &[(&'a str, f64)], config: &SolverConfig, words: u32) -> Vec<Spec<'a>> {
    let sim = [
        "scatter_particles",
        "rank_particles",
        "measure_speed",
        "measure_flow",
        "set_pressure_scale",
        "set_density_scale",
        "p2g",
        "g2p",
        "advect",
        "keep_inside",
        "liquid_phi",
        "solid_nodes",
        "mark_solid_cells",
        "extend_into_solid",
        "extend_from_scratch",
        "extend_settle",
        "settle_from_scratch",
        "face_weights",
        "refresh_solid_velocity",
        "extrapolate_pass",
        "extrapolate_back",
        "set_pressure_scale_gravity",
        "copy_scratch",
        "constrain_solid",
        "blend_partial",
        "constrain_zero",
        "zero_velocity",
        "displace",
        "aniso_prepare",
        "surface_splat",
        "surface_field",
        "white_spawn",
        "white_copy",
    ];
    let level = [
        "divergence",
        "divergence_volume",
        "apply_gradient",
        "level_from_fine",
        "body_mark",
        "body_impulse",
        "density_error",
        "save_pressure",
        "restore_pressure",
    ];
    let pair = ["coarsen", "mg_restrict", "mg_prolong", "mg_correct"];
    let solve = ["reset_speed", "pcg_start", "pcg_alpha", "pcg_beta"];
    let solve_level = [
        "level_classify",
        "level_assemble",
        "mg_zero",
        "mg_residual",
        "mg_presmooth",
        "tail_pack",
    ];
    let solve_plan = ["begin_frame", "open_all", "plan_substep"];
    let plan_entries = ["white_close"];

    let mut specs: Vec<Spec<'a>> = Vec::new();

    specs.extend(sim.iter().map(|e| (e.to_string(), *e, Layout::Sim, Vec::new())));
    specs.extend(level.iter().map(|e| (e.to_string(), *e, Layout::Level, Vec::new())));
    specs.extend(pair.iter().map(|e| (e.to_string(), *e, Layout::SolvePair, Vec::new())));
    specs.extend(solve.iter().map(|e| (e.to_string(), *e, Layout::Solve, Vec::new())));
    specs.extend(solve_level.iter().map(|e| (e.to_string(), *e, Layout::SolveLevel, Vec::new())));
    specs.extend(plan_entries.iter().map(|e| (e.to_string(), *e, Layout::Plan, plan.to_vec())));
    specs.extend(solve_plan.iter().map(|e| (e.to_string(), *e, Layout::SolvePlan, plan.to_vec())));
    specs.push(("plan_substep#grow".to_string(), "plan_substep", Layout::SolvePlan, [plan, &[("PLAN_GROW", 1.0)]].concat()));
    specs.push(("sort_count".to_string(), "sort_count", Layout::Extra, Vec::new()));
    specs.extend(face_finish_specs());
    specs.push(("advect#bodies".to_string(), "advect", Layout::Sim, vec![("ADVECT_BODIES", 1.0)]));
    specs.push(("surface_field#white".to_string(), "surface_field", Layout::Sim, vec![("SURFACE_WHITE", 1.0)]));
    specs.push(("white_spawn#advect".to_string(), "white_spawn", Layout::Sim, vec![("WHITE_ADVECT", 1.0)]));

    if config.has_band() {
        specs.push(("g2p#delete".to_string(), "g2p", Layout::Sim, vec![("G2P_DELETE", 1.0)]));
        specs.push(("band_blend#store".to_string(), "band_blend", Layout::Extra, vec![("BAND_STORE", 1.0)]));
    }
    specs.push(("sort_scan".to_string(), "sort_scan", Layout::Plan, plan.to_vec()));
    specs.push(("sort_link".to_string(), "sort_link", Layout::Link, plan.to_vec()));
    specs.push(("sort_copy".to_string(), "sort_copy", Layout::Extra, Vec::new()));
    specs.push(("sort_copy#surface".to_string(), "sort_copy", Layout::Extra, vec![("SORT_PACK", 1.0)]));
    specs.extend(pcg_specs(plan, &block_open(config, words)));
    specs.extend(
        ["couple_boundary", "couple_columns", "measure_inflow", "keep_inflow", "band_volume_target", "couple_seed"]
            .iter()
            .map(|e| (e.to_string(), *e, Layout::Coupling, Vec::new())),
    );

    for color in 0..2u32 {
        specs.push((format!("mg_smooth#{color}"), "mg_smooth", Layout::SolveLevel, vec![("COLOR", color as f64)]));
    }

    if config.smooths_boundary() {
        let reach = config.boundary_width as f64 - 1.0;
        specs.push(("level_band".to_string(), "level_band", Layout::SolveLevel, vec![("BOUNDARY_REACH", reach)]));

        for color in 0..2u32 {
            specs.push((format!("mg_band#{color}"), "mg_smooth", Layout::SolveLevel, vec![("COLOR", color as f64), ("MASK", 4.0)]));
        }
    }

    if config.open_floor.is_some() {
        specs.extend(
            ["floor_surface", "floor_mean", "floor_ghost", "floor_rhs"]
                .iter()
                .map(|e| (e.to_string(), *e, Layout::Level, Vec::new())),
        );
        specs.extend(["floor_spawn", "floor_cull"].iter().map(|e| (e.to_string(), *e, Layout::Sim, Vec::new())));
        specs.push(("floor_grow".to_string(), "floor_grow", Layout::Plan, plan.to_vec()));
    }

    specs.extend(tail_specs(config, words));
    specs.extend(block_specs(plan, config));
    specs.extend(feature_specs(plan, config));

    if config.has_band() {
        specs.push(("redistance_project#adopt".to_string(), "redistance_project", Layout::Extra, vec![("PROJECT_ADOPT", 1.0)]));
    }

    if config.tall_cells.is_some() {
        for entry in ["tall_low", "tall_cap", "tall_rows", "tall_lines", "tall_assemble", "tall_rhs", "tall_expand", "tall_reset"] {
            specs.push((entry.to_string(), entry, Layout::Level, Vec::new()));
        }

        specs.push(("tall_coarse".to_string(), "tall_coarse", Layout::SolvePair, Vec::new()));
        specs.push(("tall_gather".to_string(), "tall_gather", Layout::SolveLevel, Vec::new()));
        for mut spec in variants("mg_line", "mg_line", Layout::SolveLevel, "COLOR", &[0, 1]) {
            spec.3.push(("LINE_ROWS", (config.dims.y + 2) as f64));
            specs.push(spec);
        }
    }

    let plain: Vec<Spec<'a>> = if config.tall_cells.is_some() {
        specs
            .iter()
            .filter(|s| PLAIN.iter().any(|p| s.0.starts_with(p)))
            .map(|s| (format!("{}~plain", s.0), s.1, s.2, s.3.clone()))
            .collect()
    } else {
        Vec::new()
    };

    for spec in &mut specs {
        if config.sparse {
            spec.3.push(("SPARSE", 1.0));
        }

        if let Some(reach) = config.open_floor {
            spec.3.extend([("OPEN_FLOOR", 1.0), ("FLOOR_REACH", reach as f64)]);
        }

        if let Some(band) = config.tall_cells {
            spec.3.extend([("TALL", 1.0), ("TALL_BAND", band as f64)]);
        }
    }

    for mut spec in plain {
        if config.sparse {
            spec.3.push(("SPARSE", 1.0));
        }

        if let Some(reach) = config.open_floor {
            spec.3.extend([("OPEN_FLOOR", 1.0), ("FLOOR_REACH", reach as f64)]);
        }

        specs.push(spec);
    }

    specs
}

fn face_finish_specs<'a>() -> Vec<Spec<'a>> {
    let on = |name: &'a str| (name, 1.0);
    let mut specs: Vec<Spec<'a>> = vec![
        ("extrapolate_back#faces".to_string(), "extrapolate_back", Layout::Sim, vec![on("EXTRAP_GRAVITY"), on("EXTRAP_FACES")]),
        ("extrapolate_back#bodies".to_string(), "extrapolate_back", Layout::Sim, vec![on("EXTRAP_GRAVITY"), on("EXTRAP_BODIES")]),
        ("extrapolate_back#constrain".to_string(), "extrapolate_back", Layout::Sim, vec![("EXTRAP_CONSTRAIN", 1.0)]),
        ("extrapolate_back#zero".to_string(), "extrapolate_back", Layout::Sim, vec![("EXTRAP_CONSTRAIN", 2.0)]),
        ("extrapolate_inflow#faces".to_string(), "extrapolate_inflow", Layout::Coupling, vec![on("EXTRAP_GRAVITY"), on("EXTRAP_FACES")]),
        ("extrapolate_inflow#bodies".to_string(), "extrapolate_inflow", Layout::Coupling, vec![on("EXTRAP_GRAVITY"), on("EXTRAP_BODIES")]),
    ];

    for blend in 0..2u8 {
        for refresh in 0..2u8 {
            let constants = vec![("GRADIENT_REFRESH", f64::from(refresh)), ("GRADIENT_BLEND", f64::from(blend))];
            specs.push((format!("gradient_flow#{refresh}{blend}"), "gradient_flow", Layout::Level, constants));
        }

        specs.push((format!("gradient_inflow#{blend}"), "gradient_inflow", Layout::LevelCoupling, vec![("GRADIENT_BLEND", f64::from(blend))]));
    }

    specs
}

fn pcg_specs<'a>(plan: &[(&'a str, f64)], open: &[(&'a str, f64)]) -> Vec<Spec<'a>> {
    let flags = |coupled: bool, lead: bool| vec![("COUPLED", f64::from(u8::from(coupled))), ("LEAD", f64::from(u8::from(lead)))];
    let mut specs: Vec<Spec<'a>> = vec![
        ("pcg_apply#lead".to_string(), "pcg_apply", Layout::Level, flags(false, true)),
        ("pcg_apply".to_string(), "pcg_apply", Layout::Level, flags(false, false)),
        ("pcg_dot".to_string(), "pcg_dot", Layout::Level, Vec::new()),
        ("mg_finish".to_string(), "mg_finish", Layout::SolveLevel, vec![("COLOR", 0.0)]),
        ("pcg_scalar#lead#coupled".to_string(), "pcg_scalar", Layout::Level, flags(true, true)),
        ("pcg_update#lead#coupled".to_string(), "pcg_update", Layout::Level, flags(true, true)),
        ("pcg_check#lead".to_string(), "pcg_check", Layout::SolveLevelPlan, [plan, &flags(false, true)].concat()),
        ("pcg_check".to_string(), "pcg_check", Layout::SolveLevelPlan, [plan, &flags(false, false)].concat()),
    ];

    for coupled in [false, true] {
        let suffix = if coupled { "#coupled" } else { "" };

        for entry in ["pcg_init_residual", "pcg_scalar", "pcg_update"] {
            specs.push((format!("{entry}{suffix}"), entry, Layout::Level, flags(coupled, false)));
        }

        specs.push((format!("pcg_open{suffix}"), "pcg_open", Layout::SolveLevelPlan, [plan, &flags(coupled, false), open].concat()));
    }

    for vector in 0..2u32 {
        specs.push((format!("body_gather#{vector}"), "body_gather", Layout::Level, vec![("VECTOR", vector as f64)]));
    }

    specs
}

fn tail_specs<'a>(config: &SolverConfig, words: u32) -> Vec<Spec<'a>> {
    let dims = level_dims(config.dims, config.multigrid_min_dim);
    let tall = if config.tall_cells.is_some() { TALL_LEVELS.min(dims.len() - 1) } else { 0 };
    let tail = tail_start(&dims, tall, words);

    if tail >= dims.len() {
        return Vec::new();
    }

    let band = if config.smooths_boundary() { config.boundary_sweeps } else { 0 };
    let constants = vec![
        ("TAIL_COUNT", (dims.len() - tail) as f64),
        ("TAIL_CELLS", dims[tail..].iter().map(|d| d.element_product()).sum::<u32>() as f64),
        ("SWEEPS", config.coarse_sweeps as f64),
        ("SMOOTHING", config.smoothing as f64),
        ("BAND_SWEEPS", band as f64),
    ];

    vec![("mg_tail".to_string(), "mg_tail", Layout::SolveLevel, constants)]
}

fn block_open(config: &SolverConfig, words: u32) -> Vec<(&'static str, f64)> {
    if !config.block_smoothing {
        return Vec::new();
    }

    let dims = level_dims(config.dims, config.multigrid_min_dim);
    let first = Slots { levels: dims.len() as u32 }.solve_block(0);

    vec![("BLOCK_LEVELS", tail_start(&dims, 0, words) as f64), ("SOLVE_BLOCK_SLOT", first as f64)]
}

fn block_specs<'a>(plan: &[(&'a str, f64)], config: &SolverConfig) -> Vec<Spec<'a>> {
    let mut counts = vec![config.pressure_iterations, config.density_iterations];
    counts.sort_unstable();
    counts.dedup();

    let mut specs: Vec<Spec<'a>> = counts
        .into_iter()
        .map(|n| (format!("pcg_check#tally{n}"), "pcg_check", Layout::SolveLevelPlan, [plan, &[("LEAD", 1.0), ("TALLY", n as f64 + 1.0)]].concat()))
        .collect();

    if !config.block_smoothing {
        return specs;
    }

    let sweeps = ("BLOCK_SWEEPS", config.smoothing as f64);

    specs.extend([
        ("mg_block_down".to_string(), "mg_block_down", Layout::Level, vec![sweeps]),
        ("mg_block_coarse".to_string(), "mg_block_coarse", Layout::SolvePair, vec![sweeps]),
        ("mg_block_restrict".to_string(), "mg_block_coarse", Layout::SolvePair, vec![sweeps, ("RESTRICT_ONLY", 1.0)]),
        ("mg_block_up".to_string(), "mg_block_up", Layout::SolvePair, vec![sweeps]),
        ("mg_block_finish".to_string(), "mg_block_up", Layout::SolvePair, vec![sweeps, ("FINISH", 1.0)]),
        ("mg_block_complete".to_string(), "mg_block_complete", Layout::SolveLevel, Vec::new()),
    ]);

    for swap in [false, true] {
        for coupled in [false, true] {
            let key = format!("mg_block_down#update{}{}", if swap { "#swap" } else { "" }, if coupled { "#coupled" } else { "" });
            let bits = [("UPDATE", 1.0), ("SWAP", f64::from(u8::from(swap))), ("COUPLED", f64::from(u8::from(coupled)))];
            specs.push((key, "mg_block_down", Layout::Level, [&[sweeps][..], &bits].concat()));
        }
    }

    specs
}

fn variants<'a>(name: &str, entry: &'static str, layout: Layout, constant: &'a str, values: &[u32]) -> Vec<Spec<'a>> {
    values.iter().map(|v| (format!("{name}#{v}"), entry, layout, vec![(constant, *v as f64)])).collect()
}

fn feature_specs<'a>(plan: &[(&'a str, f64)], config: &SolverConfig) -> Vec<Spec<'a>> {
    let mut specs: Vec<Spec<'a>> = Vec::new();
    let mut plain = |names: &[&'static str], layout: Layout| {
        specs.extend(names.iter().map(|e| (e.to_string(), *e, layout, Vec::new())));
    };

    if config.redistances() {
        plain(&["redistance_seed", "redistance_finish", "redistance_project"], Layout::Extra);
    }

    if config.has_band() {
        plain(
            &[
                "band_backward",
                "band_pick",
                "band_volume_partial",
                "band_volume_final",
                "band_delete",
                "band_count",
                "band_scan_sums",
                "band_grow",
                "band_seed",
            ],
            Layout::Sim,
        );
        plain(
            &[
                "band_forward",
                "band_liquid_phi",
                "band_adopt",
                "band_scan_blocks",
                "band_spawn",
                "band_blend",
            ],
            Layout::Extra,
        );
        plain(&["band_volume_rhs"], Layout::Level);
    }

    if config.has_tension() {
        plain(&["curvature"], Layout::Extra);
        plain(&["jump_rhs", "jump_gradient"], Layout::LevelExtra);
    }

    if config.diffuses() {
        plain(&["viscous_diagonal", "viscous_load", "viscous_finish"], Layout::Extra);
        plain(&["viscous_residual", "viscous_apply", "viscous_update"], Layout::SolveExtra);
    }

    if config.redistances() {
        specs.extend(variants("redistance_sweep", "redistance_sweep", Layout::Extra, "FLIP", &[0, 1]));
    }

    if config.has_tension() {
        specs.extend(variants("blur", "blur", Layout::Extra, "BLUR", &[0, 1, 2]));
    }

    if config.diffuses() {
        let kinds: Vec<u32> = [(Diffusion::Viscosity, config.has_viscosity()), (Diffusion::Capillary, config.has_tension())]
            .iter()
            .filter(|(_, on)| *on)
            .map(|(kind, _)| *kind as u32)
            .collect();

        for entry in ["viscous_scale", "viscous_cell_fractions", "viscous_edge_fractions", "viscous_face_fractions"] {
            specs.extend(variants(entry, entry, Layout::Extra, "KIND", &kinds));
        }

        for entry in ["viscous_open", "viscous_check"] {
            for spec in variants(entry, entry, Layout::SolvePlan, "KIND", &kinds) {
                let mut constants = plan.to_vec();
                constants.extend(spec.3);
                specs.push((spec.0, spec.1, spec.2, constants));
            }
        }

        specs.extend(variants("viscous_direction", "viscous_direction", Layout::SolveExtra, "FIRST", &[0, 1]));

        for (pair, target) in [(0u32, 2u32), (1, 3), (2, 6), (3, 7)] {
            specs.push((format!("viscous_reduce_partial#{pair}"), "viscous_reduce_partial", Layout::SolveExtra, vec![("PAIR", pair as f64)]));
            specs.push((
                format!("viscous_reduce_final#{pair}"),
                "viscous_reduce_final",
                Layout::SolveExtra,
                vec![("PAIR", pair as f64), ("TARGET", target as f64)],
            ));
        }
    }

    specs
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diffusion {
    Viscosity = 0,
    Capillary = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Plan,
    Sort,
    Transfer,
    Geometry,
    Pressure,
    Advect,
    Density,
    Surface,
    Whitewater,
    Viscosity,
}

impl Section {
    pub const ALL: [Section; 10] = [
        Section::Plan,
        Section::Sort,
        Section::Transfer,
        Section::Geometry,
        Section::Pressure,
        Section::Advect,
        Section::Density,
        Section::Surface,
        Section::Whitewater,
        Section::Viscosity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Section::Plan => "liquid plan",
            Section::Sort => "liquid sort",
            Section::Transfer => "liquid transfer",
            Section::Geometry => "liquid geometry",
            Section::Pressure => "liquid pressure",
            Section::Advect => "liquid advect",
            Section::Density => "liquid density",
            Section::Surface => "liquid surface",
            Section::Whitewater => "liquid whitewater",
            Section::Viscosity => "liquid viscosity",
        }
    }

    fn index(self) -> u32 {
        Section::ALL.iter().position(|s| *s == self).unwrap_or(0) as u32
    }
}

pub struct Profiler {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    substeps: u32,
    period_ns: f32,
}

impl Profiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, substeps: u32) -> Self {
        let count = Section::ALL.len() as u32 * substeps * 2;
        let bytes = count as u64 * 8;

        Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("liquid timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquid timestamp resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquid timestamp readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            substeps,
            period_ns: queue.get_timestamp_period(),
        }
    }

    fn slot(&self, section: Section, substep: u32) -> u32 {
        (section.index() * self.substeps + substep) * 2
    }

    fn writes(&self, section: Section, substep: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        let slot = self.slot(section, substep);

        wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(slot),
            end_of_pass_write_index: Some(slot + 1),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = Section::ALL.len() as u32 * self.substeps * 2;
        encoder.clear_buffer(&self.resolve, 0, None);
        encoder.resolve_query_set(&self.set, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, count as u64 * 8);
    }

    pub fn read_milliseconds(&self, device: &wgpu::Device) -> Vec<(Section, f32)> {
        self.readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("timestamp map failed"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let stamps: Vec<u64> = {
            let view = self.readback.get_mapped_range(..).expect("timestamp range");
            bytemuck::cast_slice(&view).to_vec()
        };
        self.readback.unmap();

        self.milliseconds(&stamps, self.substeps)
    }

    pub fn readback(&self) -> &wgpu::Buffer {
        &self.readback
    }

    pub fn milliseconds(&self, stamps: &[u64], encoded: u32) -> Vec<(Section, f32)> {
        Section::ALL
            .iter()
            .map(|s| {
                let total: u64 = (0..encoded.clamp(1, self.substeps))
                    .map(|k| {
                        let i = self.slot(*s, k) as usize;
                        stamps[i + 1].saturating_sub(stamps[i])
                    })
                    .sum();

                (*s, total as f32 * self.period_ns * 1e-6)
            })
            .collect()
    }
}

fn coupling_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CouplingParams {
    pub offset: [f32; 4],
    pub grid: [u32; 4],
    pub cell: [f32; 4],
}

pub struct HeightfieldLink<'a> {
    pub offset: Vec3,
    pub cells: [u32; 2],
    pub cell_size: [f32; 2],
    pub relax_cells: f32,
    pub blend: f32,
    pub state: &'a wgpu::Buffer,
    pub bed: &'a wgpu::Buffer,
    pub coupled: &'a wgpu::Buffer,
}

pub struct Coupling {
    pub(crate) bind: wgpu::BindGroup,
    pub(crate) params: wgpu::Buffer,
    pub(crate) window: [u32; 2],
    pub(crate) sources: [wgpu::Buffer; 3],
    pub(crate) tiers: bool,
}

impl GpuSolver {
    pub fn attach_heightfield(&mut self, device: &wgpu::Device, link: HeightfieldLink<'_>) {
        let params = CouplingParams {
            offset: [link.offset.x, link.offset.y, link.offset.z, link.relax_cells],
            grid: [link.cells[0], link.cells[1], 0, 0],
            cell: [link.cell_size[0], link.cell_size[1], link.blend, 1.0],
        };
        let buffer = uniform(device, "liquid coupling params", &params);
        let sources = crate::coupling::CouplingSources { params: &buffer, state: link.state, bed: link.bed, coupled: link.coupled };
        let bind = crate::coupling::coupling_bind(device, &self.coupling_layout, sources, None);
        let sources = [link.state.clone(), link.bed.clone(), link.coupled.clone()];

        let size = self.config.dims.as_vec3() * self.config.dx;
        let window = [0, 1].map(|a| (size[2 * a] / link.cell_size[a]).ceil() as u32 + 3);

        self.coupling = Some(Coupling { bind, params: buffer, window, sources, tiers: false });
    }

    pub fn update_heightfield_offset(&self, queue: &wgpu::Queue, offset: Vec3) {
        if let Some(c) = &self.coupling {
            queue.write_buffer(&c.params, 0, bytemuck::cast_slice(&[offset.x, offset.y, offset.z]));
        }
    }

    pub fn is_coupled(&self) -> bool {
        self.coupling.is_some()
    }

    fn encode_boundary(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.is_coupled() {
            self.dispatch(pass, "couple_boundary", Bind::Coupling, Slots::FACES);
        }
    }

    fn encode_columns(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(c) = &self.coupling {
            self.prepare(pass, "couple_columns", Bind::Coupling);
            pass.dispatch_workgroups(c.window[0], c.window[1], 1);
        }
    }
}

impl GpuSolver {
    fn guarded(&self) -> Vec<&wgpu::Buffer> {
        let b = &self.buffers;
        let mut out = vec![
            &b.particles,
            &b.particles_out,
            &b.cell_counter,
            &b.cell_start,
            &b.sorted,
            &b.velocity,
            &b.valid,
            &b.scratch,
            &b.weights,
            &b.owners,
            &b.solid_velocity,
            &b.phi,
            &b.nodes,
            &b.bodies,
            &b.state,
            &b.scalars,
            &b.partials,
            &b.scratch_valid,
            &b.pressure_keep,
            &b.ellipsoids,
            &b.brick_flags,
            &b.brick_list,
        ];

        out.extend(b.extra().map(|(_, buffer)| buffer));

        out.push(&self.pack);

        for l in &self.levels {
            out.extend([&l.phi, &l.coef, &l.active, &l.diag, &l.off, &l.x, &l.b, &l.r, &l.z, &l.d, &l.q, &l.t, &l.tall, &l.e]);
        }

        out
    }

    pub fn allocated_bytes(&self) -> u64 {
        let b = &self.buffers;
        let rest = [&b.params, &b.surface, &b.indirect, &b.white, &b.white_out, &b.white_state];

        self.guarded().into_iter().chain(rest).map(|buffer| buffer.size()).sum()
    }

    pub fn grid_bytes(&self) -> u64 {
        let b = &self.buffers;
        let fine = [
            &b.cell_counter,
            &b.cell_start,
            &b.velocity,
            &b.valid,
            &b.scratch,
            &b.weights,
            &b.owners,
            &b.solid_velocity,
            &b.phi,
            &b.nodes,
            &b.scratch_valid,
            &b.pressure_keep,
            &b.surface,
            &b.brick_flags,
            &b.brick_list,
        ];
        let levels = self.levels.iter().flat_map(|l| [&l.phi, &l.coef, &l.active, &l.diag, &l.off, &l.x, &l.b, &l.r, &l.z, &l.d, &l.q, &l.t, &l.tall, &l.e]);
        let extra = b.extra().map(|(_, buffer)| buffer);

        fine.into_iter().chain(extra).chain(levels).map(|buffer| buffer.size()).sum()
    }

    pub fn guard_violations(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<String> {
        let buffers = self.guarded();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid guard readback"),
            size: buffers.len() as u64 * GUARD_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());

        for (i, buffer) in buffers.iter().enumerate() {
            let guard = buffer.size() - GUARD_BYTES;
            encoder.copy_buffer_to_buffer(buffer, guard, &staging, i as u64 * GUARD_BYTES, GUARD_BYTES);
        }

        queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("guard map failed"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let words: Vec<u32> = {
            let view = staging.get_mapped_range(..).expect("guard range");
            bytemuck::cast_slice(&view).to_vec()
        };

        let per = (GUARD_BYTES / 4) as usize;

        buffers
            .iter()
            .enumerate()
            .filter(|(i, _)| words[i * per..(i + 1) * per].iter().any(|w| *w != GUARD_WORD))
            .map(|(i, b)| format!("buffer #{i} ({} bytes) guard overwritten", b.size()))
            .collect()
    }
}
