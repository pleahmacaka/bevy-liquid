use std::sync::Arc;

use bevy::core_pipeline::schedule::camera_driver;
use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
use bevy::render::render_asset::RenderAssets;
use bevy::render::renderer::{RenderContext, RenderDevice, RenderGraph, RenderQueue};
use bevy::render::settings::{Backends, InstanceFlags, WgpuSettings};
use bevy::render::storage::{GpuShaderBuffer, ShaderBuffer};
use bevy::render::{Render, RenderApp, RenderSystems};

use crate::gas_render::{GasDomain, GasPlugin, GasRenderState, GasSimulation, prepare_gas};
use crate::emitter::LiquidEmitter;
use crate::heat::{HeatSource, MAX_HEAT_SOURCES};
use crate::layout::{GpuBody, GpuParticle, MAX_BODIES, S_IMPULSE};
use crate::render::{
    LiquidMaterial, LiquidMaterialPlugin, LiquidMotionMaterial, LiquidSurfaceSettings, SurfaceBuffers, SurfaceSdf,
};
use crate::coupling::TierLink;
use crate::solver::{GpuSolver, HeightfieldLink, SolverConfig};
use crate::whitewater::WhitewaterPlugin;
use liquid_ocean::{Heightfield, LiquidGpu, LiquidOceanPlugin, LiquidSimulation, WaterLook};
use liquid_wave::swe::CouplingRegion;

pub struct LiquidPlugin;

pub const MAX_FRAME_DT: f32 = 1.0 / 30.0;

pub fn wgpu_settings() -> WgpuSettings {
    // DX12 needs wgpu's indirect-dispatch validation pass to feed num_workgroups, so it stays opt-in through WGPU_BACKEND
    let backends = Backends::from_env().unwrap_or_else(default_backends);
    let mut settings = WgpuSettings {
        backends: Some(backends),
        ..default()
    };

    if !cfg!(debug_assertions) && !backends.contains(Backends::DX12) {
        settings.instance_flags = settings.instance_flags.difference(InstanceFlags::VALIDATION_INDIRECT_CALL).with_env();
    }

    settings
}

fn default_backends() -> Backends {
    let vulkan = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = bevy::tasks::block_on(vulkan.enumerate_adapters(Backends::VULKAN));

    if adapters.iter().any(|a| a.get_info().device_type != wgpu::DeviceType::Cpu) {
        Backends::all().difference(Backends::DX12)
    } else {
        Backends::all()
    }
}

#[derive(Component, Clone)]
pub struct LiquidDomain {
    pub config: SolverConfig,
    pub seed: Arc<Vec<GpuParticle>>,
    pub heightfield: Option<HeightfieldCoupling>,
    pub boil_into: Option<Entity>,
    pub look: WaterLook,
}

impl LiquidDomain {
    pub fn new(config: SolverConfig) -> Self {
        Self {
            config,
            seed: Arc::default(),
            heightfield: None,
            boil_into: None,
            look: WaterLook::default(),
        }
    }

    pub fn with_properties(self, properties: &LiquidProperties) -> Self {
        Self {
            config: self.config.with_properties(properties),
            look: properties.look,
            ..self
        }
    }

    pub fn with_look(self, look: WaterLook) -> Self {
        Self { look, ..self }
    }

    pub fn with_seed(self, seed: Vec<GpuParticle>) -> Self {
        Self { seed: Arc::new(seed), ..self }
    }

    pub fn filled_where(self, inside: impl Fn(Vec3) -> bool) -> Self {
        let seed = seed_where(&self.config, inside);

        self.with_seed(seed)
    }

    pub fn coupled_to(self, heightfield: HeightfieldCoupling) -> Self {
        Self { heightfield: Some(heightfield), ..self }
    }

    pub fn boils_into(self, gas: Entity) -> Self {
        Self { boil_into: Some(gas), ..self }
    }
}

#[derive(Component, Clone)]
pub struct LiquidSolid(pub Arc<Vec<f32>>);

fn solver_config(domain: &LiquidDomain, solid: Option<&LiquidSolid>) -> SolverConfig {
    SolverConfig { static_solid: solid.is_some(), ..domain.config.clone() }
}

#[derive(Clone, Copy, Debug)]
pub struct LiquidProperties {
    pub density: f32,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub adhesion: f32,
    pub temperature: f32,
    pub boiling_point: f32,
    pub latent_heat: f32,
    pub expansion: f32,
    pub look: WaterLook,
}

impl LiquidProperties {
    pub fn water() -> Self {
        let config = SolverConfig::default();

        Self {
            density: config.density,
            viscosity: config.viscosity,
            surface_tension: config.surface_tension,
            adhesion: config.adhesion,
            temperature: config.temperature,
            boiling_point: config.boiling_point,
            latent_heat: config.latent_heat,
            expansion: config.expansion,
            look: WaterLook::default(),
        }
    }
}

impl SolverConfig {
    pub fn with_properties(self, properties: &LiquidProperties) -> Self {
        Self {
            density: properties.density,
            viscosity: properties.viscosity,
            surface_tension: properties.surface_tension,
            adhesion: properties.adhesion,
            temperature: properties.temperature,
            boiling_point: properties.boiling_point,
            latent_heat: properties.latent_heat,
            expansion: properties.expansion,
            ..self
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightfieldCoupling {
    pub entity: Entity,
    pub relax_cells: f32,
    pub blend: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameLink {
    pub coupling: HeightfieldCoupling,
    pub offset: Vec3,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub enum LiquidShape {
    Sphere(f32),
    Cuboid(Vec3),
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct LiquidBody {
    pub shape: LiquidShape,
    pub density: f32,
    pub dynamic: bool,
}

#[derive(Component, Default, Clone, Copy, Debug, PartialEq)]
pub struct LiquidImpulse {
    pub linear: Vec3,
    pub angular: Vec3,
}

#[derive(Resource, Clone, Default, ExtractResource)]
#[extract_app(RenderApp)]
pub struct LiquidFrame {
    pub config: Option<SolverConfig>,
    pub seed: Option<Arc<Vec<GpuParticle>>>,
    pub solid: Option<Arc<Vec<f32>>>,
    pub generation: u32,
    pub origin: Vec3,
    pub bodies: Vec<GpuBody>,
    pub body_entities: Vec<Entity>,
    pub frame_dt: f32,
    pub index: u64,
    pub surface: Handle<ShaderBuffer>,
    pub link: Option<FrameLink>,
    pub whitewater: Handle<ShaderBuffer>,
    pub surface_view: SurfaceBuffers,
    pub seed_from_heightfield: bool,
    pub heat_sources: Vec<HeatSource>,
    pub emitters: Vec<(u64, LiquidEmitter)>,
    pub boil_offset: Option<Vec3>,
}

#[derive(Resource, Default)]
struct LiquidRenderState {
    solver: Option<GpuSolver>,
    generation: u32,
    gas_generation: Option<u32>,
    sdf: Option<SurfaceSdf>,
    pending_seed: bool,
}

#[derive(Resource, Default)]
struct ImpulseQueue {
    filtered: std::collections::HashMap<Entity, [Vec3; 2]>,
}

const BODY_FLOATS: usize = std::mem::size_of::<GpuBody>() / 4;
const READBACK_FLOATS: usize = MAX_BODIES * (6 + BODY_FLOATS);
const TAG_FLOAT: usize = 23;
const IMPULSE_BLEND: f32 = 0.5;
const IMPULSE_G_LIMIT: f32 = 4.0;

fn body_tag(entity: Entity) -> f32 {
    entity.index_u32() as f32 + 1.0
}

fn body_extent(shape: LiquidShape) -> (f32, f32) {
    match shape {
        LiquidShape::Sphere(r) => (4.0 / 3.0 * std::f32::consts::PI * r.powi(3), r),
        LiquidShape::Cuboid(h) => (8.0 * h.x * h.y * h.z, h.length()),
    }
}

// physics must not advance more time per frame than the liquid solver, or bodies outrun the fluid and the coupling blows up
pub(crate) fn cap_frame_time(app: &mut App) {
    if let Some(mut time) = app.world_mut().get_resource_mut::<Time<Virtual>>() {
        time.set_max_delta(std::time::Duration::from_secs_f32(MAX_FRAME_DT));
    }
}

impl Plugin for LiquidPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<LiquidOceanPlugin>() {
            app.add_plugins(LiquidOceanPlugin);
        }

        if !app.is_plugin_added::<GasPlugin>() {
            app.add_plugins(GasPlugin);
        }

        cap_frame_time(app);

        app.add_plugins((ExtractResourcePlugin::<LiquidFrame>::default(), LiquidMaterialPlugin, WhitewaterPlugin))
            .init_resource::<LiquidFrame>()
            .init_resource::<ImpulseQueue>()
            .add_systems(
                Update,
                (
                    spawn_domain_view,
                    reseed_domain,
                    crate::coupling::follow_promotion,
                    update_frame,
                    crate::emitter::collect_emitters,
                    link_heightfield,
                    link_gas,
                    crate::coupling::sync_promotion,
                )
                    .chain(),
            )
            .add_systems(Last, apply_impulses);

        let feedback = crate::feedback::Feedback::new((READBACK_FLOATS * 4) as u64);
        app.insert_resource(feedback.clone());

        let ring = crate::diagnostics::build(app);

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app
            .insert_resource(ring)
            .insert_resource(feedback)
            .init_resource::<LiquidRenderState>()
            .add_systems(Render, prepare_solver.in_set(RenderSystems::Prepare).after(prepare_gas))
            .add_systems(RenderGraph, run_solver.before(camera_driver).before(LiquidSimulation).before(GasSimulation));
    }
}

fn spawn_domain_view(
    mut commands: Commands,
    domains: Query<(Entity, &LiquidDomain, Option<&LiquidSolid>, &GlobalTransform), Without<LiquidReady>>,
    mut buffers: ResMut<Assets<ShaderBuffer>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<LiquidMaterial>>,
    mut motion_materials: ResMut<Assets<LiquidMotionMaterial>>,
    mut frame: ResMut<LiquidFrame>,
) {
    let Some((entity, domain, solid, transform)) = domains.iter().next() else {
        return;
    };

    let dims = domain.config.dims;
    let surface_view = SurfaceBuffers::new(dims, &mut buffers);
    let surface = surface_view.field.clone();

    let size = dims.as_vec3() * domain.config.dx;
    let origin = transform.translation();

    let settings = LiquidSurfaceSettings {
        options: Vec4::new(if domain.heightfield.is_some() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0),
        ..LiquidSurfaceSettings::new(dims, domain.config.dx, origin).with_quality(domain.config.quality)
    };
    let material = materials.add(LiquidMaterial { look: domain.look, ..LiquidMaterial::new(&surface_view, settings) });
    let motion = motion_materials.add(LiquidMotionMaterial::new(&surface_view, settings));
    let mesh = meshes.add(Cuboid::from_size(size));

    commands.entity(entity).insert(LiquidReady).with_children(|parent| {
        parent.spawn((
            LiquidView,
            Mesh3d(mesh.clone()),
            MeshMaterial3d(material),
            Transform::from_translation(size * 0.5),
            bevy::camera::visibility::NoFrustumCulling,
        ));
        parent.spawn((
            LiquidView,
            Mesh3d(mesh),
            MeshMaterial3d(motion),
            Transform::from_translation(size * 0.5),
            bevy::camera::visibility::NoFrustumCulling,
        ));
    });

    frame.whitewater = crate::whitewater::whitewater_buffer(domain.config.whitewater_capacity, &mut buffers);
    frame.config = Some(solver_config(domain, solid));
    frame.seed = Some(domain.seed.clone());
    frame.solid = solid.map(|s| s.0.clone());
    frame.seed_from_heightfield = false;
    frame.generation += 1;
    frame.origin = origin;
    frame.surface = surface;
    frame.surface_view = surface_view;
}

pub fn body_to_gpu(
    body: &LiquidBody,
    transform: &GlobalTransform,
    linear: Vec3,
    angular: Vec3,
    origin: Vec3,
) -> GpuBody {
    let (_, rotation, translation) = transform.to_scale_rotation_translation();
    let (kind, half, volume, unit_inertia) = match body.shape {
        LiquidShape::Sphere(r) => (0.0, Vec3::splat(r), 4.0 / 3.0 * std::f32::consts::PI * r.powi(3), Vec3::splat(0.4 * r * r)),
        LiquidShape::Cuboid(h) => {
            let s = h * 2.0;
            let inertia = Vec3::new(s.y * s.y + s.z * s.z, s.x * s.x + s.z * s.z, s.x * s.x + s.y * s.y) / 12.0;
            (1.0, h, 8.0 * h.x * h.y * h.z, inertia)
        }
    };

    let mass = body.density * volume;
    let r = Mat3::from_quat(rotation);
    let inv_inertia = r * Mat3::from_diagonal(1.0 / (unit_inertia * mass)) * r.transpose();
    let row = |i: usize| {
        let v = inv_inertia.row(i);
        [v.x, v.y, v.z, 0.0]
    };
    let p = translation - origin;
    let motion = if body.dynamic { 2.0 } else { 1.0 };

    GpuBody {
        position: [p.x, p.y, p.z, 1.0 / mass],
        rotation: rotation.to_array(),
        velocity: [linear.x, linear.y, linear.z, motion],
        angular: [angular.x, angular.y, angular.z, 0.0],
        shape: [half.x, half.y, half.z, kind],
        inv_inertia: [row(0), row(1), row(2)],
    }
}

type BodyQuery<'w, 's> = Query<
    'w,
    's,
    (
        Entity,
        &'static LiquidBody,
        &'static GlobalTransform,
        Option<&'static avian3d::prelude::LinearVelocity>,
        Option<&'static avian3d::prelude::AngularVelocity>,
    ),
>;

#[derive(Component)]
struct LiquidReady;

#[derive(Component)]
struct LiquidView;

type ChangedDomains<'w, 's> =
    Query<'w, 's, (Entity, &'static LiquidDomain, Option<&'static LiquidSolid>), (Changed<LiquidDomain>, With<LiquidReady>)>;

fn reseed_domain(
    mut commands: Commands,
    domains: ChangedDomains,
    views: Query<(Entity, &ChildOf), With<LiquidView>>,
    mut frame: ResMut<LiquidFrame>,
) {
    let Some((entity, domain, solid)) = domains.iter().next() else {
        return;
    };

    if frame.config.as_ref() == Some(&solver_config(domain, solid)) {
        frame.seed = Some(domain.seed.clone());
        frame.solid = solid.map(|s| s.0.clone());
        frame.generation += 1;
        frame.seed_from_heightfield = false;
        return;
    }

    commands.entity(entity).remove::<LiquidReady>();

    for (view, parent) in &views {
        if parent.parent() == entity {
            commands.entity(view).despawn();
        }
    }
}

fn update_frame(mut frame: ResMut<LiquidFrame>, bodies: BodyQuery, time: Res<Time>) {
    let origin = frame.origin;
    let mut list: Vec<_> = bodies.iter().collect();
    list.sort_by_key(|(e, ..)| *e);
    list.truncate(MAX_BODIES);

    frame.body_entities = list.iter().map(|(e, ..)| *e).collect();
    frame.bodies = list
        .iter()
        .map(|(entity, body, transform, lin, ang)| {
            let linear = lin.map_or(Vec3::ZERO, |v| v.0);
            let angular = ang.map_or(Vec3::ZERO, |v| v.0);
            let mut gpu = body_to_gpu(body, transform, linear, angular, origin);
            gpu.inv_inertia[0][3] = body_tag(*entity);
            gpu
        })
        .collect();
    frame.frame_dt = time.delta_secs().clamp(1.0 / 240.0, MAX_FRAME_DT);
    frame.index += 1;
}

fn apply_impulses(
    mut queue: ResMut<ImpulseQueue>,
    frame: Res<LiquidFrame>,
    feedback: Res<crate::feedback::Feedback>,
    device: Option<Res<RenderDevice>>,
    shapes: Query<&LiquidBody>,
    mut impulses: Query<&mut LiquidImpulse>,
    mut forces: Query<avian3d::prelude::Forces>,
) {
    let Some(config) = frame.config.as_ref() else {
        return;
    };
    let pushed = |e: &Entity| shapes.get(*e).is_ok_and(|b| b.dynamic) || impulses.contains(*e);

    if !frame.body_entities.iter().any(pushed) {
        return;
    }

    let Some(values) = device.and_then(|d| feedback.take(d.wgpu_device(), frame.index)) else {
        return;
    };

    if values.len() < READBACK_FLOATS {
        return;
    }

    let entities = &frame.body_entities;
    queue.filtered.retain(|e, _| entities.contains(e));

    for k in 0..MAX_BODIES {
        let tag = values[MAX_BODIES * 6 + k * BODY_FLOATS + TAG_FLOAT];
        let Some(entity) = entities.iter().copied().find(|e| body_tag(*e) == tag) else {
            continue;
        };
        let Ok(body) = shapes.get(entity) else {
            continue;
        };

        let (volume, reach) = body_extent(body.shape);
        let limit = config.density * volume * config.gravity.length() * IMPULSE_G_LIMIT * frame.frame_dt;
        let v = &values[k * 6..k * 6 + 6];
        let measured = [
            Vec3::new(v[0], v[1], v[2]).clamp_length_max(limit),
            Vec3::new(v[3], v[4], v[5]).clamp_length_max(limit * reach),
        ];

        if !measured.iter().all(|m| m.is_finite()) {
            continue;
        }

        // lagged readback plus added mass above the body's own mass oscillates in sign each frame; the blend damps that mode and keeps the long-run sum
        let previous = queue.filtered.get(&entity).copied().unwrap_or(measured);
        let blended = [0, 1].map(|i| previous[i].lerp(measured[i], IMPULSE_BLEND));
        queue.filtered.insert(entity, blended);

        if let Ok(mut out) = impulses.get_mut(entity) {
            *out = LiquidImpulse { linear: blended[0], angular: blended[1] };
        }

        if let Ok(mut f) = forces.get_mut(entity) {
            use avian3d::prelude::WriteRigidBodyForces;
            f.apply_linear_impulse(blended[0]);
            f.apply_angular_impulse(blended[1]);
        }
    }
}

fn prepare_solver(
    mut state: ResMut<LiquidRenderState>,
    frame: Res<LiquidFrame>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    heightfields: Option<Res<LiquidGpu>>,
    gas: Option<Res<GasRenderState>>,
) {
    let (Some(config), Some(seed)) = (&frame.config, &frame.seed) else {
        return;
    };

    if state.solver.is_none() || state.generation != frame.generation {
        let Some(surface) = buffers.get(&frame.surface) else {
            return;
        };

        let wgpu_device = device.wgpu_device();
        let solver = GpuSolver::new(wgpu_device, config.clone(), Some((*surface.buffer).clone()));
        solver.initialize(&queue);
        solver.upload_particles(&queue, seed);

        if let Some(solid) = &frame.solid {
            solver.upload_static_solid(&queue, solid);
        }

        state.solver = Some(solver);
        state.generation = frame.generation;
        state.gas_generation = None;
        state.sdf = None;
        state.pending_seed = frame.seed_from_heightfield;
    }

    let state = &mut *state;

    if state.sdf.is_none() {
        state.sdf = state
            .solver
            .as_ref()
            .and_then(|solver| SurfaceSdf::for_solver(device.wgpu_device(), solver, &frame.surface_view, &buffers));
    }

    if let Some(sdf) = state.sdf.as_mut() {
        sdf.set_frame_dt(&queue, frame.frame_dt);
    }

    let Some(solver) = state.solver.as_mut() else {
        return;
    };

    solver.upload_bodies(&queue, &frame.bodies);
    solver.set_frame_dt(&queue, frame.frame_dt);
    solver.set_keyed_emitters(&queue, &frame.emitters);
    link_boiling(solver, &mut state.gas_generation, &frame, gas.as_deref(), device.wgpu_device(), &queue);

    let (Some(link), Some(heightfields)) = (frame.link, heightfields) else {
        return;
    };

    let Some(field) = heightfields.heightfield(link.coupling.entity) else {
        return;
    };

    if solver.is_coupled() {
        solver.update_heightfield_offset(&queue, link.offset);
        return;
    }

    let buffers = field.coupling();
    let grid = &field.solver().params;

    solver.attach_heightfield(
        device.wgpu_device(),
        HeightfieldLink {
            offset: link.offset,
            cells: [grid.nx, grid.nz],
            cell_size: [grid.dx, grid.dz],
            relax_cells: link.coupling.relax_cells,
            blend: link.coupling.blend,
            state: &buffers.state,
            bed: &buffers.bed,
            coupled: &buffers.coupled,
        },
    );
    solver.link_tiers(device.wgpu_device(), TierLink { profile: &buffers.profile, ledger: &buffers.ledger });

    if std::mem::take(&mut state.pending_seed) {
        let mut encoder = device.wgpu_device().create_command_encoder(&Default::default());
        solver.encode_seed(&mut encoder);
        queue.submit([encoder.finish()]);
    }
}

fn link_boiling(
    solver: &mut GpuSolver,
    attached: &mut Option<u32>,
    frame: &LiquidFrame,
    gas: Option<&GasRenderState>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) {
    let target = gas.and_then(|g| Some((g.solver()?, g.generation())));
    let (Some(offset), Some((gas, generation))) = (frame.boil_offset, target) else {
        heat_without_gas(solver, attached, frame, device, queue);
        return;
    };

    if gas.config.injection_capacity == 0 {
        warn_once!("the liquid boils into a gas domain without injection capacity, so nothing boils");
        heat_without_gas(solver, attached, frame, device, queue);
        return;
    }

    if *attached != Some(generation) {
        solver.attach_gas(device, gas);
        *attached = Some(generation);
    }

    solver.set_heat(queue, &frame.heat_sources, offset);
}

fn heat_without_gas(solver: &mut GpuSolver, attached: &mut Option<u32>, frame: &LiquidFrame, device: &wgpu::Device, queue: &wgpu::Queue) {
    if attached.take().is_some() {
        solver.detach_gas(device);
    }

    solver.set_heat(queue, &frame.heat_sources, Vec3::ZERO);
}

fn link_heightfield(
    domains: Query<(&LiquidDomain, &GlobalTransform)>,
    mut fields: Query<(&mut Heightfield, &GlobalTransform)>,
    mut frame: ResMut<LiquidFrame>,
) {
    let Some((domain, transform)) = domains.iter().next() else {
        return;
    };

    let Some(coupling) = domain.heightfield else {
        frame.link = None;
        return;
    };

    let Ok((mut field, field_transform)) = fields.get_mut(coupling.entity) else {
        return;
    };

    let offset = transform.translation() - field_transform.translation();
    let grid = field.model.swe().grid;
    let size = domain.config.dims.as_vec3() * domain.config.dx;
    let cell = Vec2::new(grid.dx as f32, grid.dz as f32);
    let lo = (Vec2::new(offset.x, offset.z) / cell).floor().max(Vec2::ZERO).as_uvec2();
    let hi = (Vec2::new(offset.x + size.x, offset.z + size.z) / cell)
        .ceil()
        .min(Vec2::new(grid.nx as f32, grid.nz as f32))
        .as_uvec2();
    let region = CouplingRegion {
        min: [lo.x as usize, lo.y as usize],
        max: [hi.x as usize, hi.y as usize],
        blend: coupling.blend as f64,
    };

    if field.coupling != Some(region) {
        field.coupling = Some(region);
    }

    frame.link = Some(FrameLink { coupling, offset });
}

fn link_gas(domains: Query<&LiquidDomain>, gases: Query<&GasDomain>, sources: Query<(Entity, &HeatSource)>, mut frame: ResMut<LiquidFrame>) {
    let origin = frame.origin;
    let target = domains.iter().next().and_then(|d| d.boil_into).and_then(|e| gases.get(e).ok());
    let mut list: Vec<_> = sources.iter().collect();
    list.sort_by_key(|(e, _)| *e);
    list.truncate(MAX_HEAT_SOURCES);

    frame.boil_offset = target.map(|gas| gas.origin - origin);
    frame.heat_sources = list.into_iter().map(|(_, s)| s.shifted(-origin)).collect();
}

#[derive(bevy::ecs::system::SystemParam)]
struct Readbacks<'w> {
    ring: Res<'w, crate::diagnostics::DiagnosticsRing>,
    feedback: Res<'w, crate::feedback::Feedback>,
    device: Res<'w, RenderDevice>,
    queue: Res<'w, RenderQueue>,
}

fn run_solver(
    mut render_context: RenderContext,
    state: Res<LiquidRenderState>,
    frame: Res<LiquidFrame>,
    buffers: Res<RenderAssets<GpuShaderBuffer>>,
    readbacks: Readbacks,
) {
    let Some(solver) = state.solver.as_ref() else {
        return;
    };

    let Readbacks { ring, feedback, device, queue } = readbacks;
    let encoder = render_context.command_encoder();
    let late = solver.config.late_surface;
    ring.encode_frame(device.wgpu_device(), &queue, encoder, solver, state.sdf.as_ref().filter(|_| late));

    if let Some(sdf) = state.sdf.as_ref().filter(|_| !late) {
        sdf.encode(encoder);
    }

    feedback.record(device.wgpu_device(), encoder, frame.index, |encoder, target| {
        let offset = (S_IMPULSE * 4) as u64;
        encoder.copy_buffer_to_buffer(&solver.buffers.scalars, offset, target, 0, (MAX_BODIES * 6 * 4) as u64);
        encoder.copy_buffer_to_buffer(
            &solver.buffers.bodies,
            0,
            target,
            (MAX_BODIES * 6 * 4) as u64,
            (MAX_BODIES * BODY_FLOATS * 4) as u64,
        );
    });

    if let Some(target) = buffers.get(&frame.whitewater) {
        solver.copy_whitewater(encoder, &target.buffer);
    }
}

pub fn seed_box(config: &SolverConfig, lo: Vec3, hi: Vec3) -> Vec<GpuParticle> {
    seed_where(config, |x| x.cmpge(lo).all() && x.cmplt(hi).all())
}

pub fn seed_where(config: &SolverConfig, inside: impl Fn(Vec3) -> bool) -> Vec<GpuParticle> {
    let dims = config.dims;
    let mut out = Vec::new();

    for k in 0..dims.z {
        for j in 0..dims.y {
            for i in 0..dims.x {
                for s in 0..config.particles_per_cell {
                    let cell = UVec3::new(i, j, k);
                    let sub = sub_octant(config.particles_per_cell, s).as_vec3();
                    let jitter = seed_jitter(cell, s) - 0.5;
                    let x = (cell.as_vec3() + (sub + 0.5 + 0.8 * jitter) * 0.5) * config.dx;

                    if inside(x) {
                        let id = out.len() as f32;
                        out.push(GpuParticle {
                            x: [x.x, x.y, x.z, id],
                            c: [[0.0, 0.0, 0.0, config.temperature], [0.0; 4], [0.0; 4]],
                            ..Default::default()
                        });
                    }
                }
            }
        }
    }

    out
}

fn sub_octant(per_cell: u32, s: u32) -> UVec3 {
    let (x, y) = (s & 1, (s >> 1) & 1);
    let z = if per_cell < 8 { x ^ y } else { (s >> 2) & 1 };

    UVec3::new(x, y, z)
}

fn seed_jitter(cell: UVec3, sub: u32) -> Vec3 {
    let hash = |s: u32| {
        let mut v = s ^ 2_747_636_419;
        v = v.wrapping_mul(2_654_435_769);
        v ^= v >> 16;
        v = v.wrapping_mul(2_654_435_769);
        v ^= v >> 16;
        v.wrapping_mul(2_654_435_769) as f32 / u32::MAX as f32
    };
    let seed = cell.x.wrapping_mul(73_856_093)
        ^ cell.y.wrapping_mul(19_349_663)
        ^ cell.z.wrapping_mul(83_492_791)
        ^ sub.wrapping_mul(2_654_435_761);

    Vec3::new(hash(seed), hash(seed ^ 0x9E37_79B9), hash(seed ^ 0x7F4A_7C15))
}
