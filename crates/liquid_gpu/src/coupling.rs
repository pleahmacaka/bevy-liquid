use std::sync::Arc;

use bevy::prelude::*;
use liquid_ocean::{Heightfield, PromotionRegion};
use liquid_wave::promotion::Body;
use wgpu::util::DeviceExt;

use crate::plugin::{LiquidBody, LiquidDomain, LiquidFrame, LiquidShape};
use crate::render::{LiquidMaterial, LiquidMotionMaterial};
use crate::solver::{Bind, GpuSolver};

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct FollowPromotion {
    pub min_flagged: u32,
    pub min_shift: u32,
    pub cooldown: f32,
    pub since: f32,
}

impl Default for FollowPromotion {
    fn default() -> Self {
        Self {
            min_flagged: 4,
            min_shift: 4,
            cooldown: 1.5,
            since: 0.0,
        }
    }
}

type FollowQuery<'w, 's> = Query<
    'w,
    's,
    (
        &'static LiquidDomain,
        &'static mut FollowPromotion,
        &'static mut Transform,
        &'static mut GlobalTransform,
        &'static Children,
    ),
>;

type ViewQuery<'w, 's> = Query<
    'w,
    's,
    (
        Option<&'static MeshMaterial3d<LiquidMaterial>>,
        Option<&'static MeshMaterial3d<LiquidMotionMaterial>>,
    ),
>;

pub fn follow_promotion(
    time: Res<Time>,
    mut domains: FollowQuery,
    fields: Query<(&Heightfield, &PromotionRegion, &GlobalTransform), Without<LiquidDomain>>,
    views: ViewQuery,
    mut materials: ResMut<Assets<LiquidMaterial>>,
    mut motion: ResMut<Assets<LiquidMotionMaterial>>,
    mut frame: ResMut<LiquidFrame>,
) {
    let Some((domain, mut follow, mut transform, mut global, children)) = domains.iter_mut().next() else {
        return;
    };

    follow.since += time.delta_secs();

    let Some(coupling) = domain.heightfield else {
        return;
    };

    let Ok((field, region, field_transform)) = fields.get(coupling.entity) else {
        return;
    };

    let (Some(promotion), Some(current)) = (region.promotion, field.coupling) else {
        return;
    };

    let target = [promotion.window[0] as usize, promotion.window[1] as usize];
    let shift = target[0].abs_diff(current.min[0]).max(target[1].abs_diff(current.min[1]));

    if promotion.flagged < follow.min_flagged || follow.since < follow.cooldown || shift < follow.min_shift as usize {
        return;
    }

    let grid = field.model.swe().grid;
    let base = field_transform.translation();
    let origin = Vec3::new(
        base.x + target[0] as f32 * grid.dx as f32,
        transform.translation.y,
        base.z + target[1] as f32 * grid.dz as f32,
    );

    transform.translation = origin;
    *global = GlobalTransform::from(*transform);
    follow.since = 0.0;
    frame.seed = Some(Arc::new(Vec::new()));
    frame.seed_from_heightfield = true;
    frame.generation += 1;
    frame.origin = origin;

    for (surface, moving) in views.iter_many(children).flatten() {
        if let Some(mut m) = surface.and_then(|h| materials.get_mut(&h.0)) {
            m.settings.origin_dx = origin.extend(m.settings.origin_dx.w);
        }

        if let Some(mut m) = moving.and_then(|h| motion.get_mut(&h.0)) {
            m.settings.origin_dx = origin.extend(m.settings.origin_dx.w);
        }
    }
}

pub fn sync_promotion(
    domains: Query<&LiquidDomain>,
    mut fields: Query<(&mut Heightfield, &GlobalTransform)>,
    bodies: Query<(&LiquidBody, &GlobalTransform)>,
) {
    let Some(domain) = domains.iter().next() else {
        return;
    };

    let Some(coupling) = domain.heightfield else {
        return;
    };

    let Ok((mut field, transform)) = fields.get_mut(coupling.entity) else {
        return;
    };

    if field.promotion.is_none() {
        return;
    }

    let grid = field.model.swe().grid;
    let size = domain.config.dims.as_vec3() * domain.config.dx;
    let window = [(size.x / grid.dx as f32).round() as u32, (size.z / grid.dz as f32).round() as u32];
    let origin = transform.translation();
    let list: Vec<Body> = bodies
        .iter()
        .map(|(body, t)| {
            let c = t.translation() - origin;
            let radius = match body.shape {
                LiquidShape::Sphere(r) => r,
                LiquidShape::Cuboid(half) => half.length(),
            };

            Body {
                center: [c.x as f64, c.y as f64, c.z as f64],
                radius: radius as f64,
            }
        })
        .collect();

    let fresh = field
        .promotion
        .as_ref()
        .is_some_and(|p| p.window == window && p.bodies == list);

    if !fresh && let Some(settings) = field.promotion.as_mut() {
        settings.window = window;
        settings.bodies = list;
    }
}

pub struct TierLink<'a> {
    pub profile: &'a wgpu::Buffer,
    pub ledger: &'a wgpu::Buffer,
}

pub(crate) struct CouplingSources<'a> {
    pub params: &'a wgpu::Buffer,
    pub state: &'a wgpu::Buffer,
    pub bed: &'a wgpu::Buffer,
    pub coupled: &'a wgpu::Buffer,
}

pub(crate) fn coupling_bind(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    sources: CouplingSources<'_>,
    tiers: Option<TierLink<'_>>,
) -> wgpu::BindGroup {
    let zeros = |label| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: &[0u8; 256],
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let placeholders;
    let (profile, ledger) = match &tiers {
        Some(t) => (t.profile, t.ledger),
        None => {
            placeholders = [zeros("liquid profile placeholder"), zeros("liquid ledger placeholder")];
            (&placeholders[0], &placeholders[1])
        }
    };
    let entries: Vec<wgpu::BindGroupEntry> = [
        (1, sources.params),
        (2, sources.state),
        (3, sources.bed),
        (4, sources.coupled),
        (5, profile),
        (6, ledger),
    ]
    .into_iter()
    .map(|(binding, buffer)| wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    })
    .collect();

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("liquid coupling bind"),
        layout,
        entries: &entries,
    })
}

impl GpuSolver {
    pub fn link_tiers(&mut self, device: &wgpu::Device, link: TierLink<'_>) {
        let Some(coupling) = self.coupling.as_mut() else {
            return;
        };

        let [state, bed, coupled] = &coupling.sources;
        coupling.bind = coupling_bind(
            device,
            &self.coupling_layout,
            CouplingSources {
                params: &coupling.params,
                state,
                bed,
                coupled,
            },
            Some(link),
        );
        coupling.tiers = true;
    }

    pub fn tiers_linked(&self) -> bool {
        self.coupling.as_ref().is_some_and(|c| c.tiers)
    }

    pub fn encode_seed(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.is_coupled() {
            return;
        }

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("liquid seed from heightfield"),
            timestamp_writes: None,
        });

        self.dispatch_direct(&mut pass, "couple_seed", Bind::Coupling, 1);
    }
}
