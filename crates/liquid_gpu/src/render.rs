use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey};
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    AsBindGroup, CompareFunction, Face, RenderPipelineDescriptor, ShaderType, SpecializedMeshPipelineError,
};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::shader::{ShaderRef, load_shader_library};
use bytemuck::{Pod, Zeroable};
use liquid_ocean::{Heightfield, WaterLook};
use wgpu::util::DeviceExt;

use crate::caustics::CausticsPlugin;
use crate::plugin::LiquidFrame;
use crate::solver::{Bind, GpuSolver, Slots};

const SHADER_PATH: &str = "embedded://liquid_gpu/shaders/liquid_surface.wesl";
const MOTION_SHADER_PATH: &str = "embedded://liquid_gpu/shaders/liquid_motion.wesl";
const SPREAD_PASSES: usize = 5;

pub const SURFACE_ISO: f32 = 0.5;
pub const SDF_BAND_CELLS: f32 = 4.0;
pub const SDF_BRICK: u32 = 8;

pub struct LiquidMaterialPlugin;

impl Plugin for LiquidMaterialPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "shaders/brick_trace.wesl");
        embedded_asset!(app, "shaders/liquid_surface.wesl");
        embedded_asset!(app, "shaders/liquid_motion.wesl");
        app.add_plugins((
            MaterialPlugin::<LiquidMaterial>::default(),
            MaterialPlugin::<LiquidMotionMaterial>::default(),
            CausticsPlugin,
        ))
        .add_systems(Update, follow_heightfield_look);
    }
}

#[derive(Clone, Copy, Debug, ShaderType)]
pub struct LiquidSurfaceSettings {
    pub dims: UVec4,
    pub origin_dx: Vec4,
    pub options: Vec4,
    pub iso: f32,
    pub band: f32,
}

impl LiquidSurfaceSettings {
    pub fn new(dims: UVec3, dx: f32, origin: Vec3) -> Self {
        Self {
            dims: dims.extend(0),
            origin_dx: origin.extend(dx),
            options: Vec4::ZERO,
            iso: SURFACE_ISO,
            band: SDF_BAND_CELLS * dx,
        }
    }
}

pub fn brick_grid(dims: UVec3) -> UVec3 {
    (dims + SDF_BRICK - 1) / SDF_BRICK
}

#[derive(Clone, Debug, Default)]
pub struct SurfaceBuffers {
    pub field: Handle<ShaderBuffer>,
    pub distances: Handle<ShaderBuffer>,
    pub bricks: Handle<ShaderBuffer>,
    pub motion: Handle<ShaderBuffer>,
}

impl SurfaceBuffers {
    pub fn new(dims: UVec3, buffers: &mut Assets<ShaderBuffer>) -> Self {
        let cells = dims.element_product() as u64;
        let bricks = brick_grid(dims).element_product() as u64;
        let mut add = |bytes| buffers.add(ShaderBuffer::with_size(bytes, RenderAssetUsages::RENDER_WORLD));

        Self {
            field: add(cells * 4),
            distances: add(cells * 4),
            bricks: add(bricks * 4),
            motion: add(cells * 16),
        }
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct LiquidMaterial {
    #[storage(0, read_only)]
    pub field: Handle<ShaderBuffer>,
    #[uniform(1)]
    pub settings: LiquidSurfaceSettings,
    #[storage(2, read_only)]
    pub distances: Handle<ShaderBuffer>,
    #[storage(3, read_only)]
    pub bricks: Handle<ShaderBuffer>,
    #[storage(4, read_only)]
    pub motion: Handle<ShaderBuffer>,
    #[uniform(5)]
    pub look: WaterLook,
}

impl LiquidMaterial {
    pub fn new(buffers: &SurfaceBuffers, settings: LiquidSurfaceSettings) -> Self {
        Self {
            field: buffers.field.clone(),
            settings,
            distances: buffers.distances.clone(),
            bricks: buffers.bricks.clone(),
            motion: buffers.motion.clone(),
            look: WaterLook::default(),
        }
    }
}

impl Material for LiquidMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn reads_view_transmission_texture(&self) -> bool {
        true
    }

    fn enable_shadows() -> bool {
        false
    }

    // must sort after the heightfield and ocean in the transmissive phase, or they draw over the domain
    fn depth_bias(&self) -> f32 {
        1.0e6
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = Some(Face::Front);

        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            // heightfield water also covers the coupled domain, so the liquid must win; prepass depth handles occlusion
            depth.depth_compare = Some(CompareFunction::Always);
            depth.depth_write_enabled = Some(true);
        }

        Ok(())
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct LiquidMotionMaterial {
    #[storage(0, read_only)]
    pub field: Handle<ShaderBuffer>,
    #[uniform(1)]
    pub settings: LiquidSurfaceSettings,
    #[storage(2, read_only)]
    pub distances: Handle<ShaderBuffer>,
    #[storage(3, read_only)]
    pub bricks: Handle<ShaderBuffer>,
    #[storage(4, read_only)]
    pub motion: Handle<ShaderBuffer>,
}

impl LiquidMotionMaterial {
    pub fn new(buffers: &SurfaceBuffers, settings: LiquidSurfaceSettings) -> Self {
        Self {
            field: buffers.field.clone(),
            settings,
            distances: buffers.distances.clone(),
            bricks: buffers.bricks.clone(),
            motion: buffers.motion.clone(),
        }
    }
}

impl Material for LiquidMotionMaterial {
    fn vertex_shader() -> ShaderRef {
        MOTION_SHADER_PATH.into()
    }

    fn fragment_shader() -> ShaderRef {
        MOTION_SHADER_PATH.into()
    }

    fn prepass_vertex_shader() -> ShaderRef {
        MOTION_SHADER_PATH.into()
    }

    fn prepass_fragment_shader() -> ShaderRef {
        MOTION_SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Mask(SURFACE_ISO)
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        _key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = Some(Face::Front);

        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);
        }

        Ok(())
    }
}

fn follow_heightfield_look(
    frame: Res<LiquidFrame>,
    fields: Query<&Heightfield>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
) {
    let Some(field) = frame.link.and_then(|link| fields.get(link.coupling.entity).ok()) else {
        return;
    };

    let stale: Vec<_> = materials
        .iter()
        .filter(|(_, m)| look_values(&m.look) != look_values(&field.look))
        .map(|(id, _)| id)
        .collect();

    for id in stale {
        if let Some(mut material) = materials.get_mut(id) {
            material.look = field.look;
        }
    }
}

fn look_values(look: &WaterLook) -> [Vec4; 6] {
    [look.absorption, look.scatter, look.sky_zenith, look.sky_horizon, look.sun, look.foam]
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SdfParams {
    dims: [u32; 4],
    bricks: [u32; 4],
    dx: f32,
    iso: f32,
    band: f32,
    frame_dt: f32,
}

pub struct SdfSources<'a> {
    pub density: &'a wgpu::Buffer,
    pub particles: &'a wgpu::Buffer,
    pub cell_start: &'a wgpu::Buffer,
}

pub struct SdfTargets<'a> {
    pub distances: &'a wgpu::Buffer,
    pub bricks: &'a wgpu::Buffer,
    pub motion: &'a wgpu::Buffer,
}

pub struct SurfaceSdf {
    seed: wgpu::ComputePipeline,
    spread: wgpu::ComputePipeline,
    write: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    binds: [wgpu::BindGroup; 2],
    params_buffer: wgpu::Buffer,
    params: SdfParams,
    cell_groups: u32,
    tile_groups: UVec3,
    brick_groups: UVec3,
}

impl SurfaceSdf {
    pub fn new(device: &wgpu::Device, dims: UVec3, dx: f32, sources: SdfSources, targets: SdfTargets) -> Self {
        let cells = dims.element_product();
        let bricks = brick_grid(dims);
        let params = SdfParams {
            dims: dims.extend(cells).to_array(),
            bricks: bricks.extend(bricks.element_product()).to_array(),
            dx,
            iso: SURFACE_ISO,
            band: SDF_BAND_CELLS * dx,
            frame_dt: 1.0 / 60.0,
        };

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let read = wgpu::BufferBindingType::Storage { read_only: true };
        let write = wgpu::BufferBindingType::Storage { read_only: false };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid surface sdf"),
            entries: &[
                entry(0, wgpu::BufferBindingType::Uniform),
                entry(1, read),
                entry(2, read),
                entry(3, read),
                entry(4, read),
                entry(5, write),
                entry(6, write),
                entry(7, write),
                entry(8, write),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("liquid surface sdf"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("liquid surface sdf"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/surface_sdf.wgsl").into()),
        });
        let pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("liquid surface sdf params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let closest = [0, 1].map(|_| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquid surface closest"),
                size: cells as u64 * 32,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        let binds = [0, 1].map(|k| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("liquid surface sdf"),
                layout: &layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: sources.density.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: sources.particles.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: sources.cell_start.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: closest[1 - k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: closest[k].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: targets.distances.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: targets.bricks.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: targets.motion.as_entire_binding() },
                ],
            })
        });

        Self {
            seed: pipeline("seed_closest"),
            spread: pipeline("spread_closest"),
            write: pipeline("write_distance"),
            reduce: pipeline("reduce_bricks"),
            binds,
            params_buffer,
            params,
            cell_groups: cells.div_ceil(256),
            tile_groups: (dims + UVec3::new(7, 7, 3)) / UVec3::new(8, 8, 4),
            brick_groups: bricks,
        }
    }

    pub fn for_solver(
        device: &wgpu::Device,
        solver: &GpuSolver,
        buffers: &SurfaceBuffers,
        assets: &RenderAssets<GpuShaderBuffer>,
    ) -> Option<Self> {
        let distances = assets.get(&buffers.distances)?;
        let bricks = assets.get(&buffers.bricks)?;
        let motion = assets.get(&buffers.motion)?;
        let sim = &solver.buffers;

        Some(Self::new(
            device,
            solver.config.dims,
            solver.config.dx,
            SdfSources { density: &sim.surface, particles: &sim.ellipsoids, cell_start: &sim.cell_start },
            SdfTargets { distances: &distances.buffer, bricks: &bricks.buffer, motion: &motion.buffer },
        ))
    }

    pub fn set_frame_dt(&mut self, queue: &wgpu::Queue, frame_dt: f32) {
        if self.params.frame_dt != frame_dt {
            self.params.frame_dt = frame_dt;
            queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(&self.params));
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("liquid surface sdf"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&self.seed);
        pass.set_bind_group(0, &self.binds[0], &[]);
        pass.dispatch_workgroups(self.cell_groups, 1, 1);

        for k in 0..SPREAD_PASSES {
            pass.set_pipeline(&self.spread);
            pass.set_bind_group(0, &self.binds[(k + 1) % 2], &[]);
            pass.dispatch_workgroups(self.tile_groups.x, self.tile_groups.y, self.tile_groups.z);
        }

        pass.set_pipeline(&self.write);
        pass.set_bind_group(0, &self.binds[(SPREAD_PASSES + 1) % 2], &[]);
        pass.dispatch_workgroups(self.cell_groups, 1, 1);

        pass.set_pipeline(&self.reduce);
        pass.dispatch_workgroups(self.brick_groups.x, self.brick_groups.y, self.brick_groups.z);
    }
}

impl GpuSolver {
    pub(crate) fn encode_surface_field(&self, pass: &mut wgpu::ComputePass<'_>) {
        let field = if self.config.whitewater_capacity > 0 { "surface_field#white" } else { "surface_field" };
        self.dispatch(pass, "aniso_prepare", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, "surface_splat", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, field, Bind::Sim, Slots::GRID);
    }
}
