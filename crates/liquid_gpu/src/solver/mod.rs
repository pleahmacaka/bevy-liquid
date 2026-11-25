use std::collections::HashMap;

use bevy::math::{UVec3, Vec3};
use wgpu::util::DeviceExt;

use crate::layout::{
    BODY_STRIDE, BRICK, BRICK_CELLS, BRICK_FACES, BRICK_LIST_HEAD, BRICK_NODES, BRICK_ROWS, GpuBody, GpuLevelParams, GpuParams,
    GpuParticle, GpuState, GpuWhite, GpuWhiteState, MAX_BODIES, REDUCE_SPAN, S_TOLERANCE, SCALAR_COUNT, SCAN_BLOCK, WG, groups,
};
use crate::emitter::{EMIT_BINDINGS, Emission};
use crate::heat::HeatLink;
use crate::render::SurfaceSdf;
use crate::tall::{TALL_LEVELS, TALL_STRIDE};

mod config;
mod coupling;
mod encode;
mod guards;
mod pipelines;
mod profile;

pub use config::*;
pub use coupling::*;
pub use pipelines::*;
pub use profile::*;


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Side {
    Plan,
    Coupling,
    Heat,
    Emit,
}

struct Kernel {
    pipeline: wgpu::ComputePipeline,
    groups: [Option<usize>; 3],
    side: Option<Side>,
}

pub(crate) fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    let workers = std::thread::available_parallelism().map_or(1, |n| n.get());
    let chunk = items.len().div_ceil(workers).max(1);
    let f = &f;

    std::thread::scope(|scope| {
        let parts: Vec<_> = items.chunks(chunk).map(|part| scope.spawn(move || part.iter().map(f).collect::<Vec<_>>())).collect();

        parts.into_iter().flat_map(|part| part.join().expect("pipeline compile thread")).collect()
    })
}

type Pending<'a> =(String, &'static str, Vec<(&'a str, f64)>, [Option<usize>; 3], Option<Side>);

type Bindings = [Vec<(u32, wgpu::BufferBindingType)>; 4];

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
    pub static_solid: Option<wgpu::Buffer>,
    pub bodies: wgpu::Buffer,
    pub state: wgpu::Buffer,
    pub packed: wgpu::Buffer,
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
    pub z: wgpu::Buffer,
    pub e: wgpu::Buffer,
    pub w: wgpu::Buffer,
    pub k: wgpu::Buffer,
    pub samples: wgpu::Buffer,
}

impl SimBuffers {
    fn extra(&self) -> [(u32, &wgpu::Buffer); 19] {
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
            (28, &v.z),
            (29, &v.e),
            (30, &v.w),
            (31, &v.k),
            (32, &v.samples),
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
}

impl LevelBuffers {
    fn table<'a>(&'a self, pack: &'a wgpu::Buffer, vectors: [&'a wgpu::Buffer; 7]) -> Vec<(u32, &'a wgpu::Buffer)> {
        let [x, b, r, z, d, q, e] = vectors;

        vec![
            (0, &self.params),
            (1, &self.phi),
            (2, &self.coef),
            (3, &self.active),
            (4, &self.diag),
            (5, &self.off),
            (6, x),
            (7, b),
            (8, r),
            (9, z),
            (10, d),
            (11, q),
            (12, pack),
            (13, e),
            (26, &self.tall),
        ]
    }
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
    plan_bind: wgpu::BindGroup,
    pub(crate) coupling_layout: wgpu::BindGroupLayout,
    pub(crate) coupling: Option<Coupling>,
    free_bodies: bool,
    pub(crate) heat_layout: wgpu::BindGroupLayout,
    pub(crate) heat: HeatLink,
    pub(crate) emission: Emission,
    families: Vec<Vec<wgpu::BindGroup>>,
    pipelines: HashMap<String, Kernel>,
}

const READ_WRITE: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: false };

const READ_ONLY: wgpu::BufferBindingType = wgpu::BufferBindingType::Storage { read_only: true };

const PLAN_BINDINGS: [(u32, wgpu::BufferBindingType); 1] = [(0, READ_WRITE)];

const COUPLING_BINDINGS: [(u32, wgpu::BufferBindingType); 6] = [
    (1, wgpu::BufferBindingType::Uniform),
    (2, READ_ONLY),
    (3, READ_ONLY),
    (4, READ_WRITE),
    (5, READ_ONLY),
    (6, READ_WRITE),
];

const HEAT_BINDINGS: [(u32, wgpu::BufferBindingType); 3] = [(7, wgpu::BufferBindingType::Uniform), (8, READ_WRITE), (9, READ_ONLY)];

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

fn layout_entries(bindings: impl IntoIterator<Item = (u32, wgpu::BufferBindingType)>) -> Vec<wgpu::BindGroupLayoutEntry> {
    bindings
        .into_iter()
        .map(|(binding, ty)| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
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

fn entry_bindings(source: &str) -> HashMap<String, Bindings> {
    use wgpu::naga;

    let module = naga::front::wgsl::parse_str(source).unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));

    module
        .entry_points
        .iter()
        .enumerate()
        .map(|(i, entry)| {
            let uses = info.get_entry_point(i);
            let mut groups = Bindings::default();

            for (handle, var) in module.global_variables.iter() {
                let Some(binding) = &var.binding else { continue };

                let ty = match var.space {
                    naga::AddressSpace::Uniform => wgpu::BufferBindingType::Uniform,
                    naga::AddressSpace::Storage { access } => {
                        wgpu::BufferBindingType::Storage { read_only: !access.contains(naga::StorageAccess::STORE) }
                    }
                    _ => continue,
                };

                if !uses[handle].is_empty() {
                    groups[binding.group as usize].push((binding.binding, ty));
                }
            }

            for group in &mut groups {
                group.sort_by_key(|(binding, _)| *binding);
            }

            (entry.name.clone(), groups)
        })
        .collect()
}

fn family(
    device: &wgpu::Device,
    bindings: &[(u32, wgpu::BufferBindingType)],
    tables: &[Vec<(u32, &wgpu::Buffer)>],
) -> (wgpu::BindGroupLayout, Vec<wgpu::BindGroup>) {
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("liquid kernel layout"),
        entries: &layout_entries(bindings.iter().copied()),
    });

    let binds = tables
        .iter()
        .map(|table| {
            let entries: Vec<(u32, &wgpu::Buffer)> = bindings
                .iter()
                .map(|(binding, _)| {
                    *table.iter().find(|(b, _)| b == binding).unwrap_or_else(|| panic!("liquid binding {binding} has no buffer"))
                })
                .collect();

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("liquid kernel bind"),
                layout: &layout,
                entries: &bind_entries(&entries),
            })
        })
        .collect();

    (layout, binds)
}

fn check_limits(limits: &wgpu::Limits, key: &str, types: impl Iterator<Item = wgpu::BufferBindingType>) {
    let (uniform, storage) = types.fold((0, 0), |(u, s), ty| if ty == wgpu::BufferBindingType::Uniform { (u + 1, s) } else { (u, s + 1) });

    assert!(
        storage <= limits.max_storage_buffers_per_shader_stage
            && uniform <= limits.max_uniform_buffers_per_shader_stage
            && storage + uniform <= limits.max_buffers_and_acceleration_structures_per_shader_stage,
        "liquid kernel {key} binds {storage} storage and {uniform} uniform buffers, but this device allows only {} storage, {} uniform and {} combined buffers per shader stage",
        limits.max_storage_buffers_per_shader_stage,
        limits.max_uniform_buffers_per_shader_stage,
        limits.max_buffers_and_acceleration_structures_per_shader_stage,
    );
}

fn level_slot(l: usize, variant: usize) -> usize {
    if l == 0 { variant } else { 2 + l }
}

fn variant(group: usize, bind: Bind) -> usize {
    match (group, bind) {
        (1, Bind::Level(l) | Bind::LevelPlan(l) | Bind::LevelCoupling(l)) => level_slot(l, 0),
        (1, Bind::Multigrid(l) | Bind::Pair(l)) => level_slot(l, 1),
        (1, Bind::Swapped(l)) => level_slot(l, 2),
        (2, Bind::Pair(l)) => level_slot(l + 1, 0),
        _ => 0,
    }
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
        assert!(
            (1..=4).contains(&config.particles_per_cell) || config.particles_per_cell == 8,
            "liquid seeding supports 1 to 4 or 8 particles per cell"
        );
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
            thermal: [config.temperature, config.boiling_point, config.latent_heat, config.expansion],
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
            static_solid: config.static_solid.then(|| floats(device, "liquid static solid", node_count)),
            bodies: storage(device, "liquid bodies", (MAX_BODIES * std::mem::size_of::<GpuBody>()) as u64),
            state: storage(device, "liquid state", std::mem::size_of::<GpuState>() as u64),
            packed: floats(device, "liquid packed particles", when(config.quantized, 8 * config.max_particles)),
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
                z: floats(device, "liquid viscous z", when(diffusion, face_slots)),
                e: floats(device, "liquid viscous e", when(diffusion, face_slots)),
                w: floats(device, "liquid viscous warm start", when(config.has_viscosity(), face_slots)),
                k: floats(device, "liquid viscous couplings", when(diffusion, 6 * face_slots)),
                samples: floats(device, "liquid viscous samples", when(diffusion, 8 * cell_slots)),
            },
        };

        let plan_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid plan layout"),
            entries: &layout_entries(PLAN_BINDINGS),
        });
        let coupling_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid coupling layout"),
            entries: &layout_entries(COUPLING_BINDINGS),
        });
        let heat_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid heat layout"),
            entries: &layout_entries(HEAT_BINDINGS),
        });
        let heat = HeatLink::alone(device, &heat_layout, &config, false);
        let emit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid emission layout"),
            entries: &layout_entries(EMIT_BINDINGS),
        });
        let emission = Emission::new(device, &emit_layout);
        let plan_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid plan bind"),
            layout: &plan_layout,
            entries: &bind_entries(&[(0, &buffers.indirect)]),
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
                }
            })
            .collect::<Vec<_>>();

        let subgroups = config.subgroups && device.features().contains(wgpu::Features::SUBGROUP);
        let source = shader_source(subgroups, tail_size);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("liquid solver"),
            source: wgpu::ShaderSource::Wgsl(source.as_str().into()),
        });
        let used = entry_bindings(&source);

        let b = &buffers;
        let mut sim = vec![vec![
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
            (16, &b.packed),
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
            (28, &b.indirect),
        ]];
        sim[0].extend(b.static_solid.as_ref().map(|buffer| (29, buffer)));
        let extra = vec![b.extra().to_vec()];
        let mut level_tables = Vec::new();

        for (l, level) in levels.iter().enumerate() {
            let (x, bb, r, z, dd, q, t, e) = (&level.x, &level.b, &level.r, &level.z, &level.d, &level.q, &level.t, &level.e);
            level_tables.push(level.table(&pack, [x, bb, r, z, dd, q, e]));

            if l == 0 {
                level_tables.push(level.table(&pack, [z, r, t, x, dd, q, e]));
                level_tables.push(level.table(&pack, [x, bb, r, z, e, q, dd]));
            }
        }

        let tables = |group: usize, bindings: &[(u32, wgpu::BufferBindingType)]| match group {
            0 => &sim,
            2 if bindings.iter().all(|(binding, _)| extra[0].iter().any(|(e, _)| e == binding)) => &extra,
            _ => &level_tables,
        };

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

        let limits = device.limits();
        let mut family_ids: HashMap<(usize, Vec<u32>), usize> = HashMap::new();
        let mut family_layouts = Vec::new();
        let mut families = Vec::new();
        let mut layouts = HashMap::new();
        let mut pending: Vec<Pending<'_>> = Vec::new();

        for (key, entry, mut constants) in pipeline_specs(&plan_constants, &config, words) {
            for slot in plan_constants {
                if !constants.iter().any(|(name, _)| *name == slot.0) {
                    constants.push(slot);
                }
            }

            let bindings = used.get(entry).unwrap_or_else(|| panic!("liquid entry point {entry} missing"));
            let mut groups = [None; 3];

            for (group, slot) in groups.iter_mut().enumerate() {
                if bindings[group].is_empty() {
                    continue;
                }

                let next = families.len();
                let id = *family_ids.entry((group, bindings[group].iter().map(|(binding, _)| *binding).collect())).or_insert(next);

                if id == next {
                    let (layout, binds) = family(device, &bindings[group], tables(group, &bindings[group]));
                    family_layouts.push(layout);
                    families.push(binds);
                }

                *slot = Some(id);
            }

            let (side, side_bindings) = match bindings[3].as_slice() {
                [] => (None, &[][..]),
                [(0, _)] => (Some(Side::Plan), &PLAN_BINDINGS[..]),
                used if used.iter().all(|(b, _)| EMIT_BINDINGS.iter().any(|(e, _)| e == b)) => (Some(Side::Emit), &EMIT_BINDINGS[..]),
                used if used.iter().all(|(b, _)| HEAT_BINDINGS.iter().any(|(h, _)| h == b)) => (Some(Side::Heat), &HEAT_BINDINGS[..]),
                _ => (Some(Side::Coupling), &COUPLING_BINDINGS[..]),
            };

            check_limits(&limits, &key, bindings[..3].iter().flatten().chain(side_bindings).map(|(_, ty)| *ty));

            layouts.entry((groups, side)).or_insert_with(|| {
                let mut slots: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().map(|g| g.map(|id| &family_layouts[id])).collect();

                match side {
                    Some(Side::Plan) => slots.push(Some(&plan_layout)),
                    Some(Side::Coupling) => slots.push(Some(&coupling_layout)),
                    Some(Side::Heat) => slots.push(Some(&heat_layout)),
                    Some(Side::Emit) => slots.push(Some(&emit_layout)),
                    None => {}
                }

                while matches!(slots.last(), Some(None)) {
                    slots.pop();
                }

                pipeline_layout(device, &slots)
            });

            pending.push((key, entry, constants, groups, side));
        }

        let build =|(key, entry, constants, groups, side): &Pending<'_>| {
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout: Some(&layouts[&(*groups, *side)]),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

            (key.clone(), Kernel { pipeline, groups: *groups, side: *side })
        };
        let pipelines: HashMap<String, Kernel> = parallel_map(&pending, build).into_iter().collect();

        Self {
            config,
            subgroups,
            params,
            slots,
            buffers,
            levels,
            pack,
            tail,
            plan_bind,
            coupling_layout,
            coupling: None,
            free_bodies: false,
            heat_layout,
            heat,
            emission,
            families,
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
        queue.write_buffer(&self.buffers.scalars, 4 * crate::layout::S_FLOOR_SET as u64, bytemuck::bytes_of(&0.0f32));
    }

    pub fn upload_static_solid(&self, queue: &wgpu::Queue, distances: &[f32]) {
        let Some(buffer) = &self.buffers.static_solid else {
            return;
        };

        assert_eq!(distances.len(), self.params.node_dims[3] as usize, "static solid needs one distance per grid node");
        queue.write_buffer(buffer, 0, bytemuck::cast_slice(distances));
    }

    pub fn upload_bodies(&mut self, queue: &wgpu::Queue, bodies: &[GpuBody]) {
        let count = bodies.len().min(MAX_BODIES);
        let mut slots = [GpuBody::default(); MAX_BODIES];
        slots[..count].copy_from_slice(&bodies[..count]);
        queue.write_buffer(&self.buffers.bodies, 0, bytemuck::cast_slice(&slots));
        self.params.body_count = count as u32;
        self.free_bodies = slots[..count].iter().any(|b| b.velocity[3] > 1.5);
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
        let kernel = self.pipelines.get(key).unwrap_or_else(|| panic!("liquid pipeline {key} missing"));
        pass.set_pipeline(&kernel.pipeline);

        for (group, family) in kernel.groups.iter().enumerate() {
            if let Some(family) = family {
                pass.set_bind_group(group as u32, &self.families[*family][variant(group, bind)], &[]);
            }
        }

        match (kernel.side, &self.coupling) {
            (Some(Side::Plan), ..) => pass.set_bind_group(3, &self.plan_bind, &[]),
            (Some(Side::Coupling), Some(coupling)) => pass.set_bind_group(3, &coupling.bind, &[]),
            (Some(Side::Heat), _) => pass.set_bind_group(3, &self.heat.bind, &[]),
            (Some(Side::Emit), _) => pass.set_bind_group(3, &self.emission.bind, &[]),
            _ => {}
        }
    }
}
