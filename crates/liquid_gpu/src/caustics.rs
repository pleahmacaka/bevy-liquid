use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::core_pipeline::schedule::camera_driver;
use bevy::pbr::{ExtendedMaterial, ExtractedDirectionalLight, MaterialExtension};
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{AsBindGroup, BufferId};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{GpuResourceAppExt, Render, RenderApp, RenderSystems};
use bevy::shader::ShaderRef;
use bytemuck::{Pod, Zeroable};
use liquid_ocean::LiquidSimulation;

use crate::plugin::LiquidFrame;
use crate::render::SURFACE_ISO;

const RECEIVER_SHADER: &str = "embedded://liquid_gpu/shaders/caustic_receiver.wesl";
const MAP_SIDE: u32 = 512;
const HEADER_BYTES: u64 = 32;
const TEXELS_PER_CELL: u32 = 4;
const RAYS_PER_TEXEL: u32 = 2;
const MARGIN_TEXELS: u32 = 24;
const WEIGHT_SCALE: u32 = 256;

pub type CausticMaterial = ExtendedMaterial<StandardMaterial, CausticReceiver>;

pub struct CausticsPlugin;

impl Plugin for CausticsPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/caustic_receiver.wesl");
        app.add_plugins((ExtractResourcePlugin::<CausticMap>::default(), MaterialPlugin::<CausticMaterial>::default()))
            .init_resource::<CausticMap>();

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_gpu_resource::<CausticsPipeline>()
            .init_resource::<CausticsJob>()
            .add_systems(Render, prepare_caustics.in_set(RenderSystems::PrepareResources))
            .add_systems(RenderGraph, trace_caustics.after(LiquidSimulation).before(camera_driver));
    }
}

#[derive(Resource, Clone, ExtractResource)]
#[extract_app(RenderApp)]
pub struct CausticMap(pub Handle<ShaderBuffer>);

impl FromWorld for CausticMap {
    fn from_world(world: &mut World) -> Self {
        let bytes = HEADER_BYTES + (MAP_SIDE * MAP_SIDE * 4) as u64;
        let buffer = ShaderBuffer::with_size(bytes, RenderAssetUsages::RENDER_WORLD);

        Self(world.resource_mut::<Assets<ShaderBuffer>>().add(buffer))
    }
}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct CausticReceiver {
    #[storage(100, read_only)]
    pub map: Handle<ShaderBuffer>,
}

impl MaterialExtension for CausticReceiver {
    fn fragment_shader() -> ShaderRef {
        RECEIVER_SHADER.into()
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CausticsParams {
    dims: [u32; 4],
    origin_dx: [f32; 4],
    sun_iso: [f32; 4],
    map: [u32; 4],
}

#[derive(Resource)]
struct CausticsPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    params: wgpu::Buffer,
}

impl FromWorld for CausticsPipeline {
    fn from_world(world: &mut World) -> Self {
        let device = world.resource::<RenderDevice>().wgpu_device();
        let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("liquid caustics"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1, true),
                storage(2, false),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("liquid caustics"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/caustics.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("liquid caustics"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("liquid caustics"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid caustics params"),
            size: size_of::<CausticsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { layout, pipeline, params }
    }
}

struct Job {
    sources: (BufferId, BufferId),
    bind_group: wgpu::BindGroup,
    map: wgpu::Buffer,
    texel_bytes: u64,
    rays: UVec2,
}

#[derive(Resource, Default)]
struct CausticsJob(Option<Job>);

fn prepare_caustics(
    pipeline: Res<CausticsPipeline>,
    mut job: ResMut<CausticsJob>,
    frame: Res<LiquidFrame>,
    map: Res<CausticMap>,
    lights: Query<&ExtractedDirectionalLight>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
) {
    let (Some(config), Some(field), Some(target), Some(light)) =
        (&frame.config, buffers.get(&frame.surface), buffers.get(&map.0), lights.iter().next())
    else {
        job.0 = None;
        return;
    };

    let size = (config.dims.xz() * TEXELS_PER_CELL).min(UVec2::splat(MAP_SIDE));
    let rays = (size + 2 * MARGIN_TEXELS) * RAYS_PER_TEXEL;
    let extent = config.dims.as_vec3() * config.dx;
    let sun = light.transform.back();
    let origin = frame.origin;

    let params = CausticsParams {
        dims: config.dims.extend(RAYS_PER_TEXEL).to_array(),
        origin_dx: origin.extend(config.dx).to_array(),
        sun_iso: sun.extend(SURFACE_ISO).to_array(),
        map: [size.x, size.y, MARGIN_TEXELS, WEIGHT_SCALE],
    };
    let norm = 1.0 / (RAYS_PER_TEXEL * RAYS_PER_TEXEL * WEIGHT_SCALE) as f32;
    let header: [f32; 8] = [origin.x, origin.z, extent.x, extent.z, size.x as f32, size.y as f32, norm, origin.y];

    queue.write_buffer(&pipeline.params, 0, bytemuck::bytes_of(&params));
    queue.write_buffer(&target.buffer, 0, bytemuck::cast_slice(&header));

    let sources = (field.buffer.id(), target.buffer.id());
    let bind_group = match job.0.take() {
        Some(previous) if previous.sources == sources => previous.bind_group,
        _ => device.wgpu_device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("liquid caustics"),
            layout: &pipeline.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pipeline.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: field.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: target.buffer.as_entire_binding() },
            ],
        }),
    };

    job.0 = Some(Job {
        sources,
        bind_group,
        map: (*target.buffer).clone(),
        texel_bytes: (size.x * size.y * 4) as u64,
        rays,
    });
}

fn trace_caustics(mut render_context: RenderContext, pipeline: Res<CausticsPipeline>, job: Res<CausticsJob>) {
    let Some(job) = job.0.as_ref() else {
        return;
    };

    let encoder = render_context.command_encoder();
    encoder.clear_buffer(&job.map, HEADER_BYTES, Some(job.texel_bytes));

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("liquid caustics"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipeline.pipeline);
    pass.set_bind_group(0, &job.bind_group, &[]);
    pass.dispatch_workgroups(job.rays.x.div_ceil(8), job.rays.y.div_ceil(8), 1);
}
