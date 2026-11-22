use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::Arc;

use bevy::asset::{RenderAssetUsages, embedded_asset};
use bevy::camera::visibility::NoFrustumCulling;
use bevy::core_pipeline::schedule::camera_driver;
use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;
use bevy::render::gpu_readback::{Readback, ReadbackComplete};
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::BufferUsages;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Extract, ExtractSchedule, Render, RenderApp, RenderSystems};
use bevy::shader::load_shader_library;
use liquid_wave::dispersive::Dispersive;
use liquid_wave::ocean::Ocean;
use liquid_wave::promotion::{Body, Criteria, Promotion};
use liquid_wave::swe::{CouplingRegion, Ledger, Swe};

use crate::ambient::GpuAmbient;
use crate::dispersive::GpuDispersive;
use crate::material::{
    self, HeightfieldMaterial, HeightfieldSettings, OceanMaterial, OceanSettings, WaterLook,
};
use crate::ocean::{self, GpuOcean};
use crate::promotion::{GpuPromotion, GpuPromotionResult, PromotionInputs};
use crate::swe::{self, GpuLedger, GpuSwe, SharedBuffers};

const MAX_FRAME_TIME: f32 = 1.0 / 20.0;
const AMBIENT_COMPONENTS: usize = 512;

pub struct LiquidOceanPlugin;

#[derive(SystemSet, Debug, Clone, PartialEq, Eq, Hash)]
pub struct LiquidSimulation;

impl Plugin for LiquidOceanPlugin {
    fn build(&self, app: &mut App) {
        load_shader_library!(app, "water_shading.wesl");
        embedded_asset!(app, "heightfield_surface.wesl");
        embedded_asset!(app, "ocean_surface.wesl");

        app.add_plugins((
            MaterialPlugin::<HeightfieldMaterial>::default(),
            MaterialPlugin::<OceanMaterial>::default(),
        ))
        .add_systems(Update, (setup_heightfields, setup_oceans, follow_camera));

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .init_resource::<LiquidGpu>()
            .init_resource::<ExtractedLiquids>()
            .add_systems(ExtractSchedule, extract_liquids)
            .add_systems(
                Render,
                prepare_liquids.in_set(RenderSystems::PrepareResources),
            )
            .add_systems(
                RenderGraph,
                simulate_liquids
                    .in_set(LiquidSimulation)
                    .before(camera_driver),
            );
    }
}

#[derive(Clone)]
pub enum HeightfieldModel {
    ShallowWater(Arc<Swe>),
    Dispersive(Arc<Dispersive>),
}

impl HeightfieldModel {
    pub fn swe(&self) -> &Swe {
        match self {
            HeightfieldModel::ShallowWater(swe) => swe,
            HeightfieldModel::Dispersive(model) => &model.swe,
        }
    }
}

#[derive(Component, Clone)]
#[require(Transform, Visibility)]
pub struct Heightfield {
    pub model: HeightfieldModel,
    pub substeps: u32,
    pub dt_max: f32,
    pub coupling: Option<CouplingRegion>,
    pub look: WaterLook,
    pub generation: u32,
    pub promotion: Option<PromotionSettings>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PromotionSettings {
    pub criteria: Criteria,
    pub window: [u32; 2],
    pub bodies: Vec<Body>,
}

impl Default for PromotionSettings {
    fn default() -> Self {
        Self {
            criteria: Criteria::default(),
            window: [32, 32],
            bodies: Vec::new(),
        }
    }
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct HeightfieldLedger {
    pub ledger: Ledger,
    pub frames: u32,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct PromotionRegion {
    pub promotion: Option<Promotion>,
    pub frames: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Telemetry {
    pub ledger: GpuLedger,
    pub promotion: GpuPromotionResult,
}

const TELEMETRY_BYTES: u64 = std::mem::size_of::<Telemetry>() as u64;
const LEDGER_BYTES: u64 = std::mem::size_of::<GpuLedger>() as u64;

impl Heightfield {
    pub fn shallow_water(swe: Swe) -> Self {
        Self::with_model(HeightfieldModel::ShallowWater(Arc::new(swe)))
    }

    pub fn dispersive(model: Dispersive) -> Self {
        Self::with_model(HeightfieldModel::Dispersive(Arc::new(model)))
    }

    fn with_model(model: HeightfieldModel) -> Self {
        Self {
            coupling: model.swe().coupling,
            model,
            substeps: 4,
            dt_max: 0.05,
            look: WaterLook::default(),
            generation: 0,
            promotion: None,
        }
    }
}

#[derive(Component, Clone)]
pub struct HeightfieldHandles {
    pub state: Handle<ShaderBuffer>,
    pub bed: Handle<ShaderBuffer>,
    pub material: Handle<HeightfieldMaterial>,
    pub telemetry: Handle<ShaderBuffer>,
}

#[derive(Component, Clone)]
#[require(Transform, Visibility)]
pub struct OceanSurface {
    pub ocean: Arc<Ocean>,
    pub radius: f32,
    pub resolution: u32,
    pub fade: f32,
    pub foam: f32,
    pub look: WaterLook,
    pub follow_camera: bool,
}

impl OceanSurface {
    pub fn new(ocean: Ocean) -> Self {
        Self {
            ocean: Arc::new(ocean),
            radius: 4000.0,
            resolution: 384,
            fade: 12.0,
            foam: 1.0,
            look: WaterLook::default(),
            follow_camera: true,
        }
    }
}

#[derive(Component, Clone)]
pub struct OceanHandles {
    pub surface: Handle<ShaderBuffer>,
    pub material: Handle<OceanMaterial>,
}

#[derive(Clone)]
pub struct CouplingBuffers {
    pub state: wgpu::Buffer,
    pub bed: wgpu::Buffer,
    pub source: wgpu::Buffer,
    pub coupled: wgpu::Buffer,
    pub boundary: wgpu::Buffer,
    pub clock: wgpu::Buffer,
    pub params: wgpu::Buffer,
    pub ledger: wgpu::Buffer,
    pub profile: wgpu::Buffer,
}

pub enum HeightfieldSim {
    ShallowWater(Box<GpuSwe>),
    Dispersive(Box<GpuDispersive>),
}

pub struct HeightfieldGpu {
    pub sim: HeightfieldSim,
    pub substeps: u32,
    pub promotion: Option<GpuPromotion>,
    telemetry: Option<wgpu::Buffer>,
    generation: u32,
}

impl HeightfieldGpu {
    fn new(device: &wgpu::Device, field: &Heightfield, shared: SharedBuffers) -> Self {
        let sim = match &field.model {
            HeightfieldModel::ShallowWater(swe) => HeightfieldSim::ShallowWater(Box::new(
                GpuSwe::new(device, swe, field.dt_max, shared),
            )),
            HeightfieldModel::Dispersive(model) => HeightfieldSim::Dispersive(Box::new(
                GpuDispersive::new(device, model, field.dt_max, shared),
            )),
        };

        Self {
            sim,
            substeps: field.substeps,
            promotion: None,
            telemetry: None,
            generation: field.generation,
        }
    }

    fn solver_mut(&mut self) -> &mut GpuSwe {
        match &mut self.sim {
            HeightfieldSim::ShallowWater(sim) => sim,
            HeightfieldSim::Dispersive(sim) => &mut sim.bulk,
        }
    }

    pub fn solver(&self) -> &GpuSwe {
        match &self.sim {
            HeightfieldSim::ShallowWater(sim) => sim,
            HeightfieldSim::Dispersive(sim) => &sim.bulk,
        }
    }

    pub fn state(&self) -> &wgpu::Buffer {
        match &self.sim {
            HeightfieldSim::ShallowWater(sim) => &sim.buffers.state,
            HeightfieldSim::Dispersive(sim) => &sim.total,
        }
    }

    pub fn coupling(&self) -> CouplingBuffers {
        let b = &self.solver().buffers;

        CouplingBuffers {
            state: self.state().clone(),
            bed: b.bed.clone(),
            source: b.source.clone(),
            coupled: b.coupled.clone(),
            boundary: b.boundary.clone(),
            clock: b.clock.clone(),
            params: b.params.clone(),
            ledger: b.ledger.clone(),
            profile: b.profile.clone(),
        }
    }

    pub fn read_ledger(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Ledger {
        self.solver().read_ledger(device, queue).ledger()
    }

    fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, field: &Heightfield, frame_time: f32) {
        if field.generation != self.generation {
            let swe = field.model.swe();
            queue.write_buffer(
                self.state(),
                0,
                bytemuck::cast_slice(&swe::state_data(&swe.state)),
            );
            self.solver().reset_ledger(queue, field.coupling);
            self.generation = field.generation;
        }

        self.substeps = field.substeps;
        let solver = self.solver_mut();
        solver.set_coupling(field.coupling);
        solver.write_params(queue, frame_time);

        let Some(settings) = &field.promotion else {
            self.promotion = None;
            return;
        };

        if self.promotion.is_none() {
            let swe = field.model.swe();
            let b = &self.solver().buffers;
            let detector = GpuPromotion::new(
                device,
                &swe.grid,
                swe.params.gravity,
                &settings.criteria,
                PromotionInputs {
                    state: self.state(),
                    bed: &b.bed,
                    profile: &b.profile,
                },
            );
            self.promotion = Some(detector);
        }

        if let Some(detector) = &mut self.promotion {
            detector.write(queue, frame_time, settings.window, &settings.bodies);
        }
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match &self.sim {
            HeightfieldSim::ShallowWater(sim) => sim.encode_frame(encoder, self.substeps),
            HeightfieldSim::Dispersive(sim) => sim.encode_frame(encoder, self.substeps),
        }

        if let Some(detector) = &self.promotion {
            detector.encode(encoder);
        }

        let Some(telemetry) = &self.telemetry else {
            return;
        };

        encoder.copy_buffer_to_buffer(&self.solver().buffers.ledger, 0, telemetry, 0, LEDGER_BYTES);

        if let Some(detector) = &self.promotion {
            encoder.copy_buffer_to_buffer(
                &detector.result,
                0,
                telemetry,
                LEDGER_BYTES,
                TELEMETRY_BYTES - LEDGER_BYTES,
            );
        }
    }
}

pub struct OceanGpu {
    pub sim: GpuOcean,
    pub ambient: GpuAmbient,
}

#[derive(Resource, Default)]
pub struct LiquidGpu {
    pub heightfields: HashMap<Entity, HeightfieldGpu>,
    pub oceans: HashMap<Entity, OceanGpu>,
}

impl LiquidGpu {
    pub fn heightfield(&self, main_entity: Entity) -> Option<&HeightfieldGpu> {
        self.heightfields.get(&main_entity)
    }

    pub fn ocean(&self, main_entity: Entity) -> Option<&OceanGpu> {
        self.oceans.get(&main_entity)
    }
}

#[derive(Resource, Default)]
struct ExtractedLiquids {
    heightfields: Vec<(Entity, Heightfield, HeightfieldHandles)>,
    oceans: Vec<(Entity, OceanSurface, OceanHandles)>,
    delta: f32,
    elapsed: f32,
}

fn setup_heightfields(
    mut commands: Commands,
    fields: Query<(Entity, &Heightfield), Added<Heightfield>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<HeightfieldMaterial>>,
) {
    for (entity, field) in &fields {
        let swe = field.model.swe();
        let grid = swe.grid;

        let state = buffers.add(ShaderBuffer::new(
            swe::state_data(&swe.state),
            RenderAssetUsages::RENDER_WORLD,
        ));
        let bed = buffers.add(ShaderBuffer::new(
            swe::bed_data(&swe.bathymetry),
            RenderAssetUsages::RENDER_WORLD,
        ));
        let material = materials.add(HeightfieldMaterial {
            settings: HeightfieldSettings {
                size: UVec4::new(grid.nx as u32, grid.nz as u32, 0, 0),
                cell: Vec4::new(grid.dx as f32, grid.dz as f32, 1e-3, 0.0),
                look: field.look,
            },
            state: state.clone(),
            bed: bed.clone(),
        });
        let mesh = meshes.add(material::heightfield_mesh(
            &grid,
            &swe.state,
            &swe.bathymetry,
        ));
        let mut telemetry = ShaderBuffer::with_size(TELEMETRY_BYTES, RenderAssetUsages::RENDER_WORLD);
        telemetry.buffer_usage |= BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let telemetry = buffers.add(telemetry);

        commands
            .entity(entity)
            .insert((
                HeightfieldHandles {
                    state,
                    bed,
                    material: material.clone(),
                    telemetry: telemetry.clone(),
                },
                HeightfieldLedger::default(),
                PromotionRegion::default(),
                Mesh3d(mesh),
                MeshMaterial3d(material),
                NoFrustumCulling,
            ))
            .with_children(|parent| {
                parent.spawn(Readback::buffer(telemetry)).observe(
                    move |event: On<ReadbackComplete>,
                          mut fields: Query<(&mut HeightfieldLedger, &mut PromotionRegion)>| {
                        if let Ok((ledger, region)) = fields.get_mut(entity) {
                            read_telemetry(&event.data, ledger, region);
                        }
                    },
                );
            });
    }
}

fn read_telemetry(data: &[u8], mut ledger: Mut<HeightfieldLedger>, mut region: Mut<PromotionRegion>) {
    let Some(bytes) = data.get(..TELEMETRY_BYTES as usize) else {
        return;
    };

    let telemetry: Telemetry = bytemuck::pod_read_unaligned(bytes);
    let frames = telemetry.ledger.frames();

    *ledger = HeightfieldLedger {
        ledger: telemetry.ledger.ledger(),
        frames,
    };
    *region = PromotionRegion {
        promotion: telemetry.promotion.promotion(),
        frames,
    };
}

fn setup_oceans(
    mut commands: Commands,
    oceans: Query<(Entity, &OceanSurface), Added<OceanSurface>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<OceanMaterial>>,
) {
    for (entity, surface) in &oceans {
        let o = &surface.ocean;
        let mut patch = Vec4::ONE;

        for (i, c) in o.cascades.iter().enumerate() {
            patch[i] = c.patch as f32;
        }

        let buffer = buffers.add(ShaderBuffer::with_size(
            ocean::surface_bytes(o),
            RenderAssetUsages::RENDER_WORLD,
        ));
        let material = materials.add(OceanMaterial {
            settings: OceanSettings {
                grid: UVec4::new(o.params.resolution as u32, o.cascades.len() as u32, 0, 0),
                patch_size: patch,
                params: Vec4::new(o.params.sea.depth as f32, surface.fade, surface.foam, 0.0),
                look: surface.look,
            },
            surface: buffer.clone(),
        });
        let mesh = meshes.add(material::ocean_mesh(surface.radius, surface.resolution));

        commands.entity(entity).insert((
            OceanHandles {
                surface: buffer,
                material: material.clone(),
            },
            Mesh3d(mesh),
            MeshMaterial3d(material),
            NoFrustumCulling,
        ));
    }
}

fn follow_camera(
    cameras: Query<&GlobalTransform, With<Camera3d>>,
    mut oceans: Query<(&OceanSurface, &mut Transform)>,
) {
    let Some(camera) = cameras.iter().next() else {
        return;
    };

    let snap = 16.0;
    let at = camera.translation();

    for (surface, mut transform) in &mut oceans {
        if surface.follow_camera {
            transform.translation.x = (at.x / snap).round() * snap;
            transform.translation.z = (at.z / snap).round() * snap;
        }
    }
}

fn extract_liquids(
    mut extracted: ResMut<ExtractedLiquids>,
    fields: Extract<Query<(Entity, &Heightfield, &HeightfieldHandles)>>,
    oceans: Extract<Query<(Entity, &OceanSurface, &OceanHandles)>>,
    time: Extract<Res<Time>>,
) {
    extracted.heightfields = fields
        .iter()
        .map(|(e, f, h)| (e, f.clone(), h.clone()))
        .collect();
    extracted.oceans = oceans
        .iter()
        .map(|(e, o, h)| (e, o.clone(), h.clone()))
        .collect();
    extracted.delta = time.delta_secs();
    extracted.elapsed = time.elapsed_secs();
}

fn shader_buffer(
    buffers: &RenderAssets<GpuShaderBuffer>,
    handle: &Handle<ShaderBuffer>,
) -> Option<wgpu::Buffer> {
    buffers.get(handle).map(|b| wgpu::Buffer::clone(&b.buffer))
}

fn prepare_liquids(
    mut gpu: ResMut<LiquidGpu>,
    extracted: Res<ExtractedLiquids>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
) {
    let device = device.wgpu_device();
    let frame_time = extracted.delta.clamp(0.0, MAX_FRAME_TIME);
    let gpu = gpu.as_mut();

    gpu.heightfields
        .retain(|e, _| extracted.heightfields.iter().any(|(x, ..)| x == e));
    gpu.oceans
        .retain(|e, _| extracted.oceans.iter().any(|(x, ..)| x == e));

    for (entity, field, handles) in &extracted.heightfields {
        let sim = match gpu.heightfields.entry(*entity) {
            Entry::Occupied(slot) => slot.into_mut(),
            Entry::Vacant(slot) => {
                let (Some(state), Some(bed)) = (
                    shader_buffer(&buffers, &handles.state),
                    shader_buffer(&buffers, &handles.bed),
                ) else {
                    continue;
                };
                slot.insert(HeightfieldGpu::new(
                    device,
                    field,
                    SharedBuffers { state, bed },
                ))
            }
        };

        if sim.telemetry.is_none() {
            sim.telemetry = shader_buffer(&buffers, &handles.telemetry);
        }

        sim.update(device, &queue, field, frame_time);
    }

    for (entity, surface, handles) in &extracted.oceans {
        let sim = match gpu.oceans.entry(*entity) {
            Entry::Occupied(slot) => slot.into_mut(),
            Entry::Vacant(slot) => {
                let Some(buffer) = shader_buffer(&buffers, &handles.surface) else {
                    continue;
                };
                let ambient = surface.ocean.ambient().strongest(AMBIENT_COMPONENTS);
                slot.insert(OceanGpu {
                    sim: GpuOcean::new(device, &surface.ocean, buffer),
                    ambient: GpuAmbient::new(device, &ambient),
                })
            }
        };

        sim.sim.write_time(&queue, extracted.elapsed, frame_time);
        sim.ambient.write_time(&queue, extracted.elapsed);
    }
}

fn simulate_liquids(mut context: RenderContext, gpu: Res<LiquidGpu>) {
    if gpu.heightfields.is_empty() && gpu.oceans.is_empty() {
        return;
    }

    let encoder = context.command_encoder();

    for field in gpu.heightfields.values() {
        field.encode(encoder);
    }

    for ocean in gpu.oceans.values() {
        ocean.sim.encode(encoder);
    }
}
