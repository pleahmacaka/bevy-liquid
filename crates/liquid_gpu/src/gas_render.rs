use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::image::ImageSampler;
use bevy::mesh::MeshVertexBufferLayoutRef;
use bevy::pbr::{MaterialPipeline, MaterialPipelineKey, MeshPipelineKey};
use bevy::prelude::*;
use bevy::render::diagnostic::RecordDiagnostics;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    AsBindGroup, BufferUsages, CompareFunction, Extent3d, Face, RenderPipelineDescriptor, ShaderType,
    SpecializedMeshPipelineError, TextureDimension, TextureUsages, UniformBuffer,
};
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};
use bevy::shader::ShaderRef;

use crate::gas::{GAS_VOLUME_FORMAT, GasConfig, GasEmitter, GasRegion, GasSolver, GasTargets};
use crate::plugin::MAX_FRAME_DT;
use crate::solver::Quality;

const SHADER_PATH: &str = "embedded://liquid_gpu/shaders/gas_volume.wesl";
const OCCUPANCY_CELLS: u32 = 4;
const LIGHT_GROUP: u32 = 4;

pub struct GasPlugin;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct GasSimulation;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GasSampling {
    #[default]
    Texture,
    Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasLook {
    pub color: Vec3,

    pub extinction: f32,

    pub albedo: f32,

    pub anisotropy: f32,

    pub ambient: f32,

    pub glow: f32,

    pub glow_temperature: f32,

    pub spacing_cells: f32,

    pub max_samples: u32,

    pub light_samples: u32,

    pub light_spacing_cells: f32,

    pub sampling: GasSampling,
}

impl Default for GasLook {
    fn default() -> Self {
        Self {
            color: Vec3::new(0.95, 0.93, 0.9),
            extinction: 30.0,
            albedo: 0.9,
            anisotropy: 0.6,
            ambient: 0.3,
            glow: 0.0,
            glow_temperature: 300.0,
            spacing_cells: 0.6,
            max_samples: 256,
            light_samples: 6,
            light_spacing_cells: 1.0,
            sampling: GasSampling::Texture,
        }
    }
}

impl GasLook {

    pub fn with_quality(self, quality: Quality) -> Self {
        match quality {
            Quality::High => self,
            Quality::Game => Self {
                spacing_cells: self.spacing_cells.max(0.6),
                max_samples: self.max_samples.min(160),
                light_samples: self.light_samples.min(3),
                light_spacing_cells: self.light_spacing_cells.max(1.5),
                ..self
            },
        }
    }

}

#[derive(Component, Clone, Debug, PartialEq)]
pub struct GasDomain {
    pub config: GasConfig,

    pub origin: Vec3,

    pub emitters: Vec<GasEmitter>,

    pub obstacles: Vec<GasRegion>,

    pub look: GasLook,
}

impl GasDomain {

    pub fn new(config: GasConfig, origin: Vec3) -> Self {
        let look = GasLook::default().with_quality(config.quality);

        Self { config, origin, emitters: Vec::new(), obstacles: Vec::new(), look }
    }

}

#[derive(Clone, Copy, Debug, Default, ShaderType)]
pub struct GasVolumeSettings {
    pub dims: UVec4,
    pub origin_dx: Vec4,
    pub march: Vec4,
    pub medium: Vec4,
    pub smoke: Vec4,
    pub glow: Vec4,
    pub sun: Vec4,
}

impl GasVolumeSettings {

    fn new(domain: &GasDomain) -> Self {
        let look = domain.look;
        let config = &domain.config;
        let mode = match look.sampling {
            GasSampling::Texture => 0,
            GasSampling::Buffer => 1,
        };

        Self {
            dims: config.dims.extend(mode),
            origin_dx: domain.origin.extend(config.dx),
            march: Vec4::new(look.spacing_cells, look.max_samples as f32, look.light_samples as f32, look.light_spacing_cells),
            medium: Vec4::new(look.extinction, look.albedo, look.anisotropy, look.ambient),
            smoke: look.color.extend(config.params.ambient_temperature),
            glow: Vec4::new(look.glow, look.glow_temperature.max(1.0), OCCUPANCY_CELLS as f32, 0.0),
            sun: Vec4::Y,
        }
    }

}

#[derive(Asset, TypePath, AsBindGroup, Debug, Clone)]
pub struct GasMaterial {
    #[texture(0, dimension = "3d")]
    #[sampler(1)]
    pub volume: Handle<Image>,
    #[storage(2, read_only)]
    pub density: Handle<ShaderBuffer>,
    #[storage(3, read_only)]
    pub temperature: Handle<ShaderBuffer>,
    #[uniform(4)]
    pub settings: GasVolumeSettings,
    #[texture(5, dimension = "3d")]
    pub occupancy: Handle<Image>,
}

impl Material for GasMaterial {
    fn fragment_shader() -> ShaderRef {
        SHADER_PATH.into()
    }

    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Premultiplied
    }

    fn enable_shadows() -> bool {
        false
    }

    fn specialize(
        _pipeline: &MaterialPipeline,
        descriptor: &mut RenderPipelineDescriptor,
        _layout: &MeshVertexBufferLayoutRef,
        key: MaterialPipelineKey<Self>,
    ) -> Result<(), SpecializedMeshPipelineError> {
        descriptor.primitive.cull_mode = Some(Face::Front);

        if let Some(depth) = descriptor.depth_stencil.as_mut() {
            depth.depth_write_enabled = Some(false);

            if key.mesh_key.contains(MeshPipelineKey::DEPTH_PREPASS) {
                depth.depth_compare = Some(CompareFunction::Always);
            }
        }

        Ok(())
    }
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct GasFrame {
    pub config: Option<GasConfig>,
    pub generation: u32,
    pub emitters: Vec<GasEmitter>,
    pub obstacles: Vec<GasRegion>,
    pub frame_dt: f32,
    pub density: Handle<ShaderBuffer>,
    pub temperature: Handle<ShaderBuffer>,
    pub volume: Handle<Image>,
    pub lit: Handle<Image>,
    pub occupancy: Handle<Image>,
    pub settings: GasVolumeSettings,
    pub sampling: GasSampling,
    pub idle: bool,
}

#[derive(Component)]
struct GasReady {
    view: Entity,
    material: Handle<GasMaterial>,
    origin: Vec3,
    look: GasLook,
}

#[derive(Component)]
struct GasView(Entity);

struct GasLight {
    settings: UniformBuffer<GasVolumeSettings>,
    passes: [(wgpu::ComputePipeline, wgpu::BindGroup, UVec3); 2],
}

impl GasLight {

    fn new(device: &RenderDevice, queue: &RenderQueue, settings: GasVolumeSettings, [volume, lit, occupancy]: [&GpuImage; 3]) -> Option<Self> {
        let mut settings = UniformBuffer::from(settings);
        settings.write_buffer(device, queue);

        let wgpu_device = device.wgpu_device();
        let module = wgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gas light"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/gas_light.wgsl").into()),
        });
        let uniform = settings.binding()?;
        let pass = |entry_point: &str, source: &GpuImage, target: &GpuImage| {
            let pipeline = wgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
            let resources = [
                wgpu::BindingResource::TextureView(&source.texture_view),
                wgpu::BindingResource::Sampler(&source.sampler),
                uniform.clone(),
                wgpu::BindingResource::TextureView(&target.texture_view),
            ];
            let entries: Vec<_> = resources
                .into_iter()
                .zip(0..)
                .map(|(resource, binding)| wgpu::BindGroupEntry { binding, resource })
                .collect();
            let bind = wgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(entry_point),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            let size = target.texture_descriptor.size;
            let groups = UVec3::new(size.width, size.height, size.depth_or_array_layers).map(|n| n.div_ceil(LIGHT_GROUP));

            (pipeline, bind, groups)
        };
        let passes = [pass("gas_light", volume, lit), pass("gas_occupancy", lit, occupancy)];

        Some(Self { settings, passes })
    }

    fn record(&self, pass: &mut wgpu::ComputePass<'_>) {
        for (pipeline, bind, groups) in &self.passes {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind, &[]);
            pass.dispatch_workgroups(groups.x, groups.y, groups.z);
        }
    }

}

#[derive(Resource, Default)]
pub struct GasRenderState {
    solver: Option<GasSolver>,
    generation: u32,
    volume: Option<wgpu::BindGroup>,
    light: Option<GasLight>,
}

impl GasRenderState {

    pub fn solver(&self) -> Option<&GasSolver> {
        self.solver.as_ref()
    }

    pub fn generation(&self) -> u32 {
        self.generation
    }

}

impl Plugin for GasPlugin {
    fn build(&self, app: &mut App) {
        embedded_asset!(app, "shaders/gas_volume.wesl");
        crate::plugin::cap_frame_time(app);

        app.add_plugins((ExtractResourcePlugin::<GasFrame>::default(), MaterialPlugin::<GasMaterial>::default()))
            .init_resource::<GasFrame>()
            .add_systems(Update, (spawn_gas_view, follow_gas_domain, update_gas_frame, settle_gas).chain());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<GasRenderState>()
            .add_systems(Render, prepare_gas.in_set(RenderSystems::Prepare))
            .add_systems(RenderGraph, run_gas.in_set(GasSimulation).before(camera_driver));
    }
}

fn gas_buffer(cells: u32, buffers: &mut Assets<ShaderBuffer>) -> Handle<ShaderBuffer> {
    let mut buffer = ShaderBuffer::with_size(cells as u64 * 4, RenderAssetUsages::RENDER_WORLD);
    buffer.buffer_usage |= BufferUsages::COPY_DST | BufferUsages::COPY_SRC;

    buffers.add(buffer)
}

fn gas_volume(dims: UVec3, images: &mut Assets<Image>) -> Handle<Image> {
    let size = Extent3d { width: dims.x, height: dims.y, depth_or_array_layers: dims.z };
    let mut image = Image::new_fill(size, TextureDimension::D3, &[0; 8], GAS_VOLUME_FORMAT, RenderAssetUsages::RENDER_WORLD);
    image.texture_descriptor.usage |= TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING;
    image.sampler = ImageSampler::linear();

    images.add(image)
}

type NewDomains<'w, 's> = Query<'w, 's, (Entity, &'static GasDomain), Without<GasReady>>;

fn spawn_gas_view(
    mut commands: Commands,
    domains: NewDomains,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut images: ResMut<Assets<Image>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<GasMaterial>>,
    mut frame: ResMut<GasFrame>,
) {
    let Some((entity, domain)) = domains.iter().next() else {
        return;
    };

    let config = &domain.config;
    let size = config.size();
    let density = gas_buffer(config.cell_count(), &mut buffers);
    let temperature = gas_buffer(config.cell_count(), &mut buffers);
    let volume = gas_volume(config.dims, &mut images);
    let lit = gas_volume(config.dims, &mut images);
    let occupancy = gas_volume(config.dims.map(|n| n.div_ceil(OCCUPANCY_CELLS)), &mut images);
    let material = materials.add(GasMaterial {
        volume: lit.clone(),
        density: density.clone(),
        temperature: temperature.clone(),
        settings: GasVolumeSettings::new(domain),
        occupancy: occupancy.clone(),
    });
    let view = commands
        .spawn((
            Mesh3d(meshes.add(Cuboid::from_size(size))),
            MeshMaterial3d(material.clone()),
            Transform::from_translation(domain.origin + size * 0.5),
            GasView(entity),
        ))
        .id();

    commands.entity(entity).insert(GasReady { view, material, origin: domain.origin, look: domain.look });

    *frame = GasFrame {
        config: Some(config.clone()),
        generation: frame.generation + 1,
        density,
        temperature,
        volume,
        lit,
        occupancy,
        sampling: domain.look.sampling,
        ..default()
    };
}

type ChangedDomains<'w, 's> = Query<'w, 's, (Entity, Ref<'static, GasDomain>, &'static mut GasReady)>;

fn follow_gas_domain(
    mut commands: Commands,
    mut domains: ChangedDomains,
    views: Query<(Entity, &GasView)>,
    owners: Query<(), With<GasDomain>>,
    mut boxes: Query<&mut Transform, With<GasView>>,
    mut materials: ResMut<Assets<GasMaterial>>,
    frame: Res<GasFrame>,
) {
    for (view, owner) in &views {
        if owners.get(owner.0).is_err() {
            commands.entity(view).despawn();
        }
    }

    for (entity, domain, mut ready) in &mut domains {
        if !domain.is_changed() || domain.is_added() {
            continue;
        }

        let rebuilt = frame.config.as_ref() != Some(&domain.config) || frame.sampling != domain.look.sampling;

        if rebuilt {
            commands.entity(ready.view).despawn();
            commands.entity(entity).remove::<GasReady>();
            continue;
        }

        if ready.look == domain.look && ready.origin == domain.origin {
            continue;
        }

        if let Ok(mut transform) = boxes.get_mut(ready.view) {
            transform.translation = domain.origin + domain.config.size() * 0.5;
        }

        ready.look = domain.look;
        ready.origin = domain.origin;

        if let Some(mut material) = materials.get_mut(&ready.material) {
            material.settings = GasVolumeSettings::new(&domain);
        }
    }
}

fn update_gas_frame(
    domains: Query<&GasDomain, With<GasReady>>,
    suns: Query<(&DirectionalLight, &GlobalTransform, &InheritedVisibility)>,
    shallow: Query<(), (With<Camera3d>, Without<DepthPrepass>)>,
    mut frame: ResMut<GasFrame>,
    time: Res<Time>,
) {
    let Some(domain) = domains.iter().next() else {
        if frame.config.is_some() {
            frame.config = None;
        }

        return;
    };

    if !shallow.is_empty() {
        warn_once!("a camera without DepthPrepass draws gas wrongly around objects inside the gas domain");
    }

    if domains.iter().nth(1).is_some() {
        warn_once!("only the first gas domain is simulated; the others are ignored");
    }

    let offset = -domain.origin;

    frame.emitters = domain
        .emitters
        .iter()
        .map(|e| GasEmitter { region: e.region.shifted(offset), ..*e })
        .collect();
    frame.obstacles = domain.obstacles.iter().map(|o| o.shifted(offset)).collect();
    frame.frame_dt = time.delta_secs().min(MAX_FRAME_DT);
    frame.settings = GasVolumeSettings::new(domain);

    let power = |light: &DirectionalLight| light.illuminance * light.color.to_linear().luminance();
    let lit = suns.iter().filter(|(_, _, shown)| shown.get());

    if let Some((_, sun, _)) = lit.max_by(|a, b| power(a.0).total_cmp(&power(b.0))) {
        frame.settings.sun = sun.back().extend(0.0);
    }
}

pub(crate) fn prepare_gas(
    mut state: ResMut<GasRenderState>,
    frame: Res<GasFrame>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    images: Res<RenderAssets<GpuImage>>,
) {
    let Some(config) = &frame.config else {
        state.solver = None;
        state.light = None;
        return;
    };

    if state.solver.is_none() || state.generation != frame.generation {
        let (Some(density), Some(temperature), Some(volume), Some(lit), Some(occupancy)) = (
            buffers.get(&frame.density),
            buffers.get(&frame.temperature),
            images.get(&frame.volume),
            images.get(&frame.lit),
            images.get(&frame.occupancy),
        ) else {
            return;
        };

        let wgpu_device = device.wgpu_device();
        let targets = GasTargets {
            density: Some((*density.buffer).clone()),
            temperature: Some((*temperature.buffer).clone()),
        };
        let solver = GasSolver::new(wgpu_device, config.clone(), &frame.obstacles, targets);
        solver.initialize(&queue);

        state.volume = Some(solver.volume_bind_group(wgpu_device, &volume.texture_view));
        state.light = GasLight::new(&device, &queue, frame.settings, [volume, lit, occupancy]);
        state.solver = Some(solver);
        state.generation = frame.generation;
    }

    let Some(solver) = state.solver.as_mut() else {
        return;
    };

    if solver.obstacles() != frame.obstacles.as_slice() {
        solver.set_obstacles(&queue, &frame.obstacles);
    }

    solver.set_frame(&queue, frame.frame_dt, &frame.emitters);

    if let Some(light) = state.light.as_mut() {
        light.settings.set(frame.settings);
        light.settings.write_buffer(&device, &queue);
    }
}

fn run_gas(mut render_context: RenderContext, state: Res<GasRenderState>, frame: Res<GasFrame>) {
    let Some(solver) = state.solver.as_ref() else {
        return;
    };

    let diagnostics = render_context.diagnostic_recorder();
    let diagnostics = diagnostics.as_deref();
    let encoder = render_context.command_encoder();
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gas"), timestamp_writes: None });
    let span = diagnostics.pass_span(&mut pass, "gas");

    if frame.idle {
        span.end(&mut pass);
        return;
    }

    solver.record_frame(&mut pass);

    if let (Some(volume), Some(light), GasSampling::Texture) = (&state.volume, &state.light, frame.sampling) {
        solver.record_pack(&mut pass, volume);
        light.record(&mut pass);
    }

    span.end(&mut pass);
}

const SETTLED_DENSITY: f32 = 1e-3;

#[derive(Default)]
struct Quiet {
    generation: u32,
    fed: bool,
    frames: u32,
    seconds: f32,
}

impl Quiet {
    fn settled(&mut self, domain: &GasDomain, entity: Entity, generation: u32, liquids: &Query<&crate::plugin::LiquidDomain>, dt: f32) -> bool {
        let fed = domain.emitters.iter().any(|e| e.density_rate > 0.0 || e.heat_rate != 0.0)
            || liquids.iter().any(|l| l.boil_into == Some(entity));

        if generation != self.generation {
            *self = Self { generation, ..default() };
        }

        if fed {
            *self = Self { generation, fed: true, ..default() };
            return false;
        }

        self.frames += 1;
        self.seconds += dt;
        let decay = domain.config.params.density_decay;
        let fade = match (self.fed, decay > 0.0) {
            (false, _) => 0.0,
            (true, true) => -SETTLED_DENSITY.ln() / decay,
            (true, false) => f32::INFINITY,
        };

        self.frames > 2 && self.seconds >= fade
    }
}

fn settle_gas(
    domains: Query<(Entity, &GasDomain, &GasReady)>,
    liquids: Query<&crate::plugin::LiquidDomain>,
    mut views: Query<&mut Visibility, With<GasView>>,
    mut quiet: Local<Quiet>,
    mut frame: ResMut<GasFrame>,
) {
    let Some((entity, domain, ready)) = domains.iter().next() else {
        return;
    };

    let idle = quiet.settled(domain, entity, frame.generation, &liquids, frame.frame_dt);
    frame.idle = idle;

    if let Ok(mut shown) = views.get_mut(ready.view) {
        shown.set_if_neq(if idle { Visibility::Hidden } else { Visibility::Inherited });
    }
}
