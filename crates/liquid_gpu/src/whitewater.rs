use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::core_pipeline::core_3d::{CORE_3D_DEPTH_FORMAT, main_opaque_pass_3d, main_transparent_pass_3d};
use bevy::core_pipeline::{Core3d, Core3dSystems, FullscreenShader};
use bevy::pbr::main_transmissive_pass_3d;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::binding_types::{
    storage_buffer_read_only_sized, texture_2d, texture_2d_multisampled, uniform_buffer,
};
use bevy::render::render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, BlendComponent, BlendFactor,
    BlendOperation, BlendState, Buffer, BufferDescriptor, BufferId, BufferUsages, CachedRenderPipelineId,
    ColorTargetState, ColorWrites, CompareFunction, DepthStencilState, Extent3d, FragmentState, LoadOp,
    MultisampleState, Operations, PipelineCache, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipelineDescriptor, ShaderStages, SpecializedRenderPipeline, SpecializedRenderPipelines, StoreOp,
    TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType, TextureUsages, VertexState,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderQueue, ViewQuery};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::texture::{CachedTexture, TextureCache};
use bevy::render::view::{
    ExtractedView, Msaa, ViewDepthStencilTexture, ViewTarget, ViewUniform, ViewUniformOffset, ViewUniforms,
};
use bevy::render::{Render, RenderApp, RenderStartup, RenderSystems};
use bevy::shader::{Shader, ShaderDefVal};

use crate::layout::{GpuWhite, GpuWhiteState};
use crate::plugin::{LiquidDomain, LiquidFrame};
use crate::solver::{Bind, GpuSolver, Passes, Profiler, Section, Slots};

const SHADER_PATH: &str = "embedded://liquid_gpu/shaders/whitewater.wesl";
const STATE_BYTES: u64 = std::mem::size_of::<GpuWhiteState>() as u64;
const ACCUM_FORMAT: TextureFormat = TextureFormat::Rgba16Float;

pub struct WhitewaterPlugin;

impl Plugin for WhitewaterPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/whitewater.wesl");
        app.add_systems(Update, spawn_whitewater_view);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<SpecializedRenderPipelines<WhitewaterPipeline>>()
            .init_resource::<WhitewaterSource>()
            .add_systems(RenderStartup, init_whitewater_pipeline)
            .add_systems(
                Render,
                (
                    prepare_whitewater_targets.in_set(RenderSystems::PrepareResources),
                    prepare_whitewater_bind_groups.in_set(RenderSystems::PrepareBindGroups),
                ),
            )
            .add_systems(
                Core3d,
                (
                    accumulate_whitewater
                        .after(main_opaque_pass_3d)
                        .before(main_transmissive_pass_3d)
                        .in_set(Core3dSystems::MainPass),
                    composite_whitewater.after(main_transparent_pass_3d).in_set(Core3dSystems::MainPass),
                ),
            );
    }
}

fn spawn_whitewater_view(
    domains: Query<&LiquidDomain, Added<LiquidDomain>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut frame: ResMut<LiquidFrame>,
) {
    let Some(domain) = domains.iter().next() else {
        return;
    };

    let capacity = domain.config.whitewater_capacity;

    if capacity == 0 {
        return;
    }

    let bytes = STATE_BYTES + capacity as u64 * std::mem::size_of::<GpuWhite>() as u64;
    let mut buffer = ShaderBuffer::with_size(bytes, RenderAssetUsages::RENDER_WORLD);
    buffer.buffer_usage |= BufferUsages::COPY_DST;

    frame.whitewater = buffers.add(buffer);
}

#[derive(Resource)]
struct WhitewaterPipeline {
    view_layout: BindGroupLayoutDescriptor,
    particles_layout: BindGroupLayoutDescriptor,
    composite_layouts: [BindGroupLayoutDescriptor; 2],
    shader: Handle<Shader>,
    fullscreen: FullscreenShader,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct WhitewaterKey {
    composite: bool,
    samples: u32,
    target: TextureFormat,
}

fn init_whitewater_pipeline(mut commands: Commands, asset_server: Res<AssetServer>, fullscreen: Res<FullscreenShader>) {
    let float = TextureSampleType::Float { filterable: false };
    let textures = |multisampled: bool| {
        let entry = || if multisampled { texture_2d_multisampled(float) } else { texture_2d(float) };

        BindGroupLayoutDescriptor::new(
            "whitewater composite",
            &BindGroupLayoutEntries::with_indices(ShaderStages::FRAGMENT, ((10, entry()),)),
        )
    };

    commands.insert_resource(WhitewaterPipeline {
        view_layout: BindGroupLayoutDescriptor::new(
            "whitewater view",
            &BindGroupLayoutEntries::single(ShaderStages::VERTEX, uniform_buffer::<ViewUniform>(true)),
        ),
        particles_layout: BindGroupLayoutDescriptor::new(
            "whitewater particles",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::VERTEX,
                (storage_buffer_read_only_sized(false, None), uniform_buffer::<Vec4>(false)),
            ),
        ),
        composite_layouts: [textures(false), textures(true)],
        shader: asset_server.load(SHADER_PATH),
        fullscreen: fullscreen.clone(),
    });
}

impl SpecializedRenderPipeline for WhitewaterPipeline {
    type Key = WhitewaterKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        let multisampled = key.samples > 1;
        let shader_defs: Vec<ShaderDefVal> = if multisampled { vec!["MULTISAMPLED".into()] } else { Vec::new() };
        let multisample = MultisampleState { count: key.samples, ..default() };

        if key.composite {
            return RenderPipelineDescriptor {
                label: Some("whitewater composite".into()),
                layout: vec![self.composite_layouts[usize::from(multisampled)].clone()],
                vertex: self.fullscreen.to_vertex_state(),
                fragment: Some(FragmentState {
                    shader: self.shader.clone(),
                    shader_defs,
                    entry_point: Some("composite".into()),
                    targets: vec![Some(ColorTargetState {
                        format: key.target,
                        blend: Some(BlendState::ALPHA_BLENDING),
                        write_mask: ColorWrites::ALL,
                    })],
                    ..default()
                }),
                multisample,
                ..default()
            };
        }

        let sum = BlendComponent {
            src_factor: BlendFactor::One,
            dst_factor: BlendFactor::One,
            operation: BlendOperation::Add,
        };
        let product = BlendComponent {
            src_factor: BlendFactor::Zero,
            dst_factor: BlendFactor::OneMinusSrcAlpha,
            operation: BlendOperation::Add,
        };

        RenderPipelineDescriptor {
            label: Some("whitewater accumulate".into()),
            layout: vec![self.view_layout.clone(), self.particles_layout.clone()],
            vertex: VertexState {
                shader: self.shader.clone(),
                shader_defs: shader_defs.clone(),
                entry_point: Some("vertex".into()),
                ..default()
            },
            fragment: Some(FragmentState {
                shader: self.shader.clone(),
                shader_defs,
                entry_point: Some("accumulate".into()),
                targets: vec![Some(ColorTargetState {
                    format: ACCUM_FORMAT,
                    blend: Some(BlendState { color: sum, alpha: product }),
                    write_mask: ColorWrites::ALL,
                })],
                ..default()
            }),
            depth_stencil: Some(DepthStencilState {
                format: CORE_3D_DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(CompareFunction::GreaterEqual),
                stencil: default(),
                bias: default(),
            }),
            multisample,
            ..default()
        }
    }
}

#[derive(Component)]
struct WhitewaterTargets {
    accum: CachedTexture,
    accumulate: CachedRenderPipelineId,
    composite: CachedRenderPipelineId,
}

#[derive(Component)]
struct WhitewaterComposite(BindGroup);

#[derive(Resource, Default)]
struct WhitewaterSource {
    particles: Option<(BufferId, BindGroup)>,
    view: Option<BindGroup>,
    params: Option<Buffer>,
    instances: u32,
}

impl WhitewaterSource {
    fn ready(&self) -> Option<(&BindGroup, &BindGroup)> {
        Some((self.view.as_ref()?, &self.particles.as_ref()?.1))
    }
}

type PipelineAccess<'w> = (Res<'w, PipelineCache>, ResMut<'w, SpecializedRenderPipelines<WhitewaterPipeline>>);

fn prepare_whitewater_targets(
    mut commands: Commands,
    views: Query<(Entity, &ExtractedCamera, &ExtractedView, &Msaa)>,
    frame: Res<LiquidFrame>,
    pipeline: Option<Res<WhitewaterPipeline>>,
    (cache, mut pipelines): PipelineAccess,
    mut textures: ResMut<TextureCache>,
    device: Res<RenderDevice>,
) {
    let active = frame.config.as_ref().is_some_and(|c| c.whitewater_capacity > 0);

    for (entity, camera, view, msaa) in &views {
        let (Some(pipeline), Some(size), true) = (pipeline.as_deref(), camera.physical_target_size, active) else {
            commands.entity(entity).remove::<WhitewaterTargets>();
            continue;
        };

        let samples = msaa.samples();
        let mut target = |label: &'static str, format: TextureFormat| {
            textures.get(
                &device,
                TextureDescriptor {
                    label: Some(label),
                    size: Extent3d { width: size.x, height: size.y, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: TextureDimension::D2,
                    format,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
            )
        };
        let key = |composite| WhitewaterKey { composite, samples, target: view.target_format };

        commands.entity(entity).insert(WhitewaterTargets {
            accum: target("whitewater accum", ACCUM_FORMAT),
            accumulate: pipelines.specialize(&cache, pipeline, key(false)),
            composite: pipelines.specialize(&cache, pipeline, key(true)),
        });
    }
}

type BindingSources<'w> = (Res<'w, PipelineCache>, Res<'w, RenderAssets<GpuShaderBuffer>>, Res<'w, ViewUniforms>);

fn prepare_whitewater_bind_groups(
    mut commands: Commands,
    views: Query<(Entity, &WhitewaterTargets, &Msaa)>,
    (frame, pipeline): (Res<LiquidFrame>, Option<Res<WhitewaterPipeline>>),
    (cache, buffers, view_uniforms): BindingSources,
    mut source: ResMut<WhitewaterSource>,
    (device, queue): (Res<RenderDevice>, Res<RenderQueue>),
) {
    let (Some(pipeline), Some(config), Some(particles), Some(view_binding)) =
        (pipeline, frame.config.as_ref(), buffers.get(&frame.whitewater), view_uniforms.uniforms.binding())
    else {
        source.view = None;
        return;
    };

    let params = source
        .params
        .get_or_insert_with(|| {
            device.create_buffer(&BufferDescriptor {
                label: Some("whitewater params"),
                size: 16,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        })
        .clone();
    queue.write_buffer(&params, 0, bytemuck::cast_slice(&frame.origin.extend(config.dx).to_array()));

    let id = particles.buffer.id();

    if source.particles.as_ref().is_none_or(|(current, _)| *current != id) {
        let group = device.create_bind_group(
            "whitewater particles",
            &cache.get_bind_group_layout(&pipeline.particles_layout),
            &BindGroupEntries::sequential((particles.buffer.as_entire_binding(), params.as_entire_binding())),
        );
        source.particles = Some((id, group));
    }

    source.instances = config.whitewater_capacity;
    source.view = Some(device.create_bind_group(
        "whitewater view",
        &cache.get_bind_group_layout(&pipeline.view_layout),
        &BindGroupEntries::single(view_binding),
    ));

    for (entity, targets, msaa) in &views {
        let layout = &pipeline.composite_layouts[usize::from(msaa.samples() > 1)];
        let group = device.create_bind_group(
            "whitewater composite",
            &cache.get_bind_group_layout(layout),
            &BindGroupEntries::with_indices(((10, &targets.accum.default_view),)),
        );
        commands.entity(entity).insert(WhitewaterComposite(group));
    }
}

fn cleared(texture: &CachedTexture, value: wgpu::Color) -> Option<RenderPassColorAttachment<'_>> {
    Some(RenderPassColorAttachment {
        view: &texture.default_view,
        depth_slice: None,
        resolve_target: None,
        ops: Operations { load: LoadOp::Clear(value), store: StoreOp::Store },
    })
}

fn accumulate_whitewater(
    view: ViewQuery<(&ExtractedCamera, &ViewDepthStencilTexture, &ViewUniformOffset, &WhitewaterTargets)>,
    source: Res<WhitewaterSource>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (camera, depth, offset, targets) = view.into_inner();
    let (Some((view_group, particles)), Some(pipeline)) =
        (source.ready(), cache.get_render_pipeline(targets.accumulate))
    else {
        return;
    };

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("whitewater_accumulate"),
        color_attachments: &[cleared(&targets.accum, wgpu::Color::BLACK)],
        depth_stencil_attachment: Some(depth.get_attachment(StoreOp::Store)),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, view_group, &[offset.offset]);
    pass.set_bind_group(1, particles, &[]);
    pass.draw(0..6, 0..source.instances);
}

fn composite_whitewater(
    view: ViewQuery<(&ExtractedCamera, &ViewTarget, &WhitewaterTargets, &WhitewaterComposite)>,
    source: Res<WhitewaterSource>,
    cache: Res<PipelineCache>,
    mut ctx: RenderContext,
) {
    let (camera, target, targets, composite) = view.into_inner();
    let (Some(_), Some(_), Some(pipeline)) = (
        source.ready(),
        cache.get_render_pipeline(targets.accumulate),
        cache.get_render_pipeline(targets.composite),
    ) else {
        return;
    };

    let mut pass = ctx.begin_tracked_render_pass(RenderPassDescriptor {
        label: Some("whitewater_composite"),
        color_attachments: &[Some(target.get_color_attachment())],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    if let Some(viewport) = camera.viewport.as_ref() {
        pass.set_camera_viewport(viewport);
    }

    pass.set_render_pipeline(pipeline);
    pass.set_bind_group(0, &composite.0, &[]);
    pass.draw(0..3, 0..1);
}

impl GpuSolver {
    pub fn encode_whitewater(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>) {
        self.encode_whitewater_into(&mut Passes::new(encoder, profiler));
    }

    pub(crate) fn encode_whitewater_into(&self, passes: &mut Passes<'_>) {
        if self.config.whitewater_capacity == 0 {
            return;
        }

        let args = self.slots.count() as u64 * 16;
        let white = |pass: &mut wgpu::ComputePass<'_>, key: &str| {
            self.prepare(pass, key, Bind::Sim);
            pass.dispatch_workgroups_indirect(&self.buffers.indirect, args);
        };

        let pass = passes.section(Section::Whitewater, 0);
        self.dispatch(pass, "white_spawn#advect", Bind::Sim, Slots::PARTICLES);
        self.dispatch_direct(pass, "white_close", Bind::Plan, 1);
        white(pass, "white_copy");
    }

    pub fn copy_whitewater(&self, encoder: &mut wgpu::CommandEncoder, target: &wgpu::Buffer) {
        let particles = &self.buffers.white;
        let bytes = particles.size().min(target.size().saturating_sub(STATE_BYTES));

        encoder.copy_buffer_to_buffer(&self.buffers.white_state, 0, target, 0, STATE_BYTES);
        encoder.copy_buffer_to_buffer(particles, 0, target, STATE_BYTES, bytes);
    }
}
