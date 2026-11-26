use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use avian3d::prelude::*;
use bevy::anti_alias::taa::TemporalAntiAliasing;
use bevy::asset::io::file::FileAssetReader;
use bevy::camera::Hdr;
use bevy::core_pipeline::prepass::DepthPrepass;
use bevy::diagnostic::{DiagnosticPath, DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::ecs::system::SystemParam;
use bevy::gltf::{Gltf, GltfLoaderSettings};
use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::light::{CascadeShadowConfigBuilder, GlobalAmbientLight};
use bevy::mesh::VertexAttributeValues;
use bevy::mesh::skinning::{SkinnedMesh, SkinnedMeshInverseBindposes};
use bevy::pbr::ScreenSpaceTransmission;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::render::view::screenshot::{Screenshot, save_to_disk};
use bevy::time::TimeUpdateStrategy;
use bevy::window::PresentMode;
use bevy::world_serialization::WorldInstanceReady;
use liquid_gpu::diagnostics::{ENCODE_TIME, GPU_TIME, PARTICLES, SECTIONS, SUBSTEPS};
use liquid_gpu::prelude::*;

const MODEL: &str = "character/CamomeCamome.vrm";
const BONES: [&str; 12] = [
    "head",
    "leftEye",
    "neck",
    "hips",
    "leftUpperArm",
    "leftHand",
    "rightUpperArm",
    "rightHand",
    "leftUpperLeg",
    "leftFoot",
    "rightUpperLeg",
    "rightFoot",
];
const STAND_IN: [Vec3; 12] = [
    Vec3::new(0.0, 1.05, 0.0),
    Vec3::new(0.03, 1.11, 0.08),
    Vec3::new(0.0, 1.0, 0.0),
    Vec3::new(0.0, 0.68, 0.0),
    Vec3::new(0.1, 0.96, 0.0),
    Vec3::new(0.26, 0.62, 0.0),
    Vec3::new(-0.1, 0.96, 0.0),
    Vec3::new(-0.26, 0.62, 0.0),
    Vec3::new(0.06, 0.64, 0.0),
    Vec3::new(0.07, 0.08, 0.0),
    Vec3::new(-0.06, 0.64, 0.0),
    Vec3::new(-0.07, 0.08, 0.0),
];
const ARM_DROP: f32 = 1.1;
const HEAD_SCALE: f32 = 2.7;
const HEAD_LIFT: f32 = 0.65;
const SHOULDER_DEPTH: f32 = 0.4;
const SHELL_CELLS: f32 = 0.75;
const EXACT_CELLS: f32 = 2.0;
const MARK_SCALE: f32 = 0.3;
const LIQUID_DX: f32 = 0.0175;
const LIQUID_MARGIN: Vec3 = Vec3::new(0.12, 0.38, 0.15);
const BLOB_RADIUS: f32 = 0.12;
const STREAM_RADIUS: f32 = 0.05;
const BLOB_VOLUME: f32 = 0.007;
const STREAM_FLOW: f32 = 0.0025;
const GAS_DX: f32 = 0.025;
const GAS_HALF_WIDTH: f32 = 0.8;
const GAS_HEADROOM: f32 = 0.75;
const SUN: Vec3 = Vec3::new(0.35, 0.8, 0.5);
const HELP: &str = "1  Honey: tap to drop, hold to pour\n2  Water: tap to drop, hold to pour\n    Changing the liquid clears the other one\n3  Pour smoke\n4  Smoke from the floor\nC  Show colliders\nR  Clear everything\nTab  Toggle quality\nDrag  Orbit\nWheel  Zoom";
const SCRIPT: [(f32, Action); 4] = [(0.3, Action::Honey), (3.3, Action::Honey), (13.1, Action::Pour), (19.1, Action::Rise)];
const SCRIPT_HOLD: (f32, f32) = (7.0, 8.5);
const CAPTURES: [f32; 11] = [2.0, 3.2, 5.0, 8.4, 11.0, 16.0, 19.0, 19.6, 21.0, 23.0, 26.0];
const BENCH_WARMUP: usize = 180;
const BENCH_RUNS: [(&str, Action); 2] = [("honey", Action::Honey), ("honey and smoke", Action::Pour)];
const GAS_TIME: DiagnosticPath = DiagnosticPath::const_new("render/gas/elapsed_gpu");
const VOLUME_TIME: DiagnosticPath = DiagnosticPath::const_new("render/main_transparent_pass_3d/elapsed_gpu");

#[derive(Message, Clone, Copy, PartialEq)]
enum Action {
    Honey,
    Water,
    Pour,
    Rise,
    Reset,
    Quality,
}

#[derive(Clone, Copy, PartialEq)]
enum Liquid {
    Honey,
    Water,
}

#[derive(Clone, Copy, PartialEq)]
enum Smoke {
    Pour,
    Rise,
}

#[derive(Component, Clone, Copy, PartialEq)]
enum Faucet {
    Drop,
    Stream,
}

#[derive(Resource)]
struct Lab {
    quality: Quality,
    liquid: Liquid,
    smoke: Option<Smoke>,
    colliders: bool,
    clock: f32,
    pouring: u32,
}

#[derive(Resource)]
struct Model(Handle<Gltf>);

#[derive(Component)]
struct Skeleton(Vec<(&'static str, String)>);

#[derive(Component)]
struct Rig(HashMap<&'static str, Entity>);

#[derive(Clone, Copy)]
struct Part {
    center: Vec3,
    rotation: Quat,
    shape: LiquidShape,
}

type Triangle = [Vec3; 3];

#[derive(Resource)]
struct Figure {
    parts: Vec<Part>,
    triangles: Vec<Triangle>,
    crown: Vec3,
    lo: Vec3,
    hi: Vec3,
    visible: bool,
}

#[derive(Resource)]
struct Solids {
    liquid: Option<Arc<Vec<f32>>>,
    gas: Vec<GasRegion>,
    marks: Vec<Vec3>,
    cell: f32,
}

#[derive(Component)]
struct Orbit {
    target: Vec3,
    yaw: f32,
    pitch: f32,
    distance: f32,
}

#[derive(Resource)]
struct Bench {
    frames: usize,
    run: usize,
    seen: usize,
    last: Option<Instant>,
    wall: Vec<f64>,
    gpu: Vec<Vec<f64>>,
}

fn main() {
    let quality = quality_flag();
    let capture = std::env::var_os("LIQUID_CAPTURE").is_some();
    let bench = std::env::var("LIQUID_BENCH").ok().and_then(|v| v.parse::<usize>().ok());
    let diagnostics = bench.is_some() || std::env::var_os("LIQUID_DIAGNOSTICS").is_some();
    let mut app = App::new();

    app.insert_resource(ClearColor(Color::srgb(0.2, 0.22, 0.26)))
        .insert_resource(Lab {
            quality,
            liquid: Liquid::Honey,
            smoke: None,
            colliders: std::env::var_os("LIQUID_COLLIDERS").is_some(),
            clock: 0.0,
            pouring: 0,
        })
        .insert_resource(GlobalAmbientLight {
            brightness: 400.0,
            ..default()
        })
        .add_plugins((
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "bevy fluid".into(),
                        present_mode: if diagnostics { PresentMode::AutoNoVsync } else { PresentMode::AutoVsync },
                        ..default()
                    }),
                    ..default()
                })
                .set(RenderPlugin {
                    render_creation: wgpu_settings().into(),
                    ..default()
                }),
            PhysicsPlugins::default(),
            LiquidPlugin,
        ))
        .add_message::<Action>()
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                spawn_character.run_if(resource_exists::<Model>),
                fit_character,
                build_scene.run_if(resource_added::<Figure>),
                read_keys,
                apply_actions.run_if(resource_exists::<Solids>),
                run_faucets.run_if(resource_exists::<Figure>),
                orbit_camera,
                draw_colliders.run_if(resource_exists::<Solids>),
            )
                .chain(),
        );

    if capture || bench.is_some() {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(1.0 / 60.0)));
    }

    if capture {
        app.add_systems(Update, run_script.run_if(resource_exists::<Solids>).before(apply_actions));
    }

    if let Some(frames) = bench {
        app.add_plugins(RenderDiagnosticsPlugin)
            .insert_resource(Bench {
                frames,
                run: 0,
                seen: 0,
                last: None,
                wall: Vec::new(),
                gpu: vec![Vec::new(); bench_paths().len()],
            })
            .add_systems(Update, record_bench.run_if(resource_exists::<Solids>).before(apply_actions));
    } else if diagnostics {
        app.add_plugins((FrameTimeDiagnosticsPlugin::default(), LogDiagnosticsPlugin::default(), RenderDiagnosticsPlugin));
    }

    app.run();
}

fn quality_flag() -> Quality {
    let args: Vec<String> = std::env::args().collect();
    let value = args.iter().position(|a| a == "--quality").map(|i| args.get(i + 1).map(String::as_str));

    match value {
        None | Some(Some("game")) => Quality::Game,
        Some(Some("high")) => Quality::High,
        Some(other) => {
            eprintln!("unknown --quality {}, expected game or high", other.unwrap_or(""));
            std::process::exit(2);
        }
    }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut gizmos: ResMut<GizmoConfigStore>,
    assets: Res<AssetServer>,
) {
    gizmos.config_mut::<DefaultGizmoConfigGroup>().0.depth_bias = -1.0;

    if FileAssetReader::get_base_path().join("assets").join(MODEL).exists() {
        let model = assets
            .load_builder()
            .with_settings(|s: &mut GltfLoaderSettings| s.include_source = true)
            .load::<Gltf>(MODEL);
        commands.insert_resource(Model(model));
    } else {
        info!("character model not found at assets/{MODEL}, showing the stand-in figure");
        commands.insert_resource(figure(&STAND_IN, true));
    }

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 0.1, 8.0))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.45, 0.43, 0.41),
            perceptual_roughness: 0.9,
            ..default()
        })),
        Transform::from_xyz(0.0, -0.05, 0.0),
    ));

    commands.spawn((
        Mesh3d(meshes.add(Cuboid::new(8.0, 5.0, 0.1))),
        MeshMaterial3d(materials.add(StandardMaterial {
            base_color: Color::srgb(0.3, 0.33, 0.38),
            perceptual_roughness: 0.95,
            ..default()
        })),
        Transform::from_xyz(0.0, 2.5, -1.6),
    ));

    commands.spawn((
        DirectionalLight {
            illuminance: 10_000.0,
            shadow_maps_enabled: true,
            ..default()
        },
        Transform::default().looking_to(-SUN, Vec3::Y),
        CascadeShadowConfigBuilder {
            num_cascades: 2,
            first_cascade_far_bound: 4.0,
            maximum_distance: 12.0,
            ..default()
        }
        .build(),
    ));

    commands.spawn((
        Camera3d::default(),
        Hdr,
        DepthPrepass,
        ScreenSpaceTransmission::default(),
        Msaa::Off,
        TemporalAntiAliasing::default(),
        Orbit {
            target: Vec3::new(0.0, 0.75, 0.0),
            yaw: 0.35,
            pitch: 0.12,
            distance: 2.0,
        },
    ));

    commands.spawn((
        Text::new(HELP),
        TextFont::from_font_size(17.0),
        TextColor(Color::WHITE),
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
        Node {
            position_type: PositionType::Absolute,
            top: px(12),
            left: px(12),
            padding: UiRect::all(px(10)),
            ..default()
        },
    ));
}

fn spawn_character(
    mut commands: Commands,
    model: Res<Model>,
    gltfs: Res<Assets<Gltf>>,
    assets: Res<AssetServer>,
) {
    if assets.load_state(&model.0).is_failed() {
        warn!("assets/{MODEL} could not be loaded, showing the stand-in figure");
        commands.remove_resource::<Model>();
        commands.insert_resource(figure(&STAND_IN, true));
        return;
    }

    let Some(gltf) = gltfs.get(&model.0) else {
        return;
    };

    commands.remove_resource::<Model>();

    let scene = gltf.default_scene.clone().or_else(|| gltf.scenes.first().cloned());
    let skeleton = humanoid(gltf);

    let (Some(scene), Some(skeleton)) = (scene, skeleton) else {
        warn!("assets/{MODEL} has no VRM humanoid skeleton, showing the stand-in figure");
        commands.insert_resource(figure(&STAND_IN, true));
        return;
    };

    // VRM 0.x models face -Z, the camera looks down -Z
    commands
        .spawn((WorldAssetRoot(scene), Transform::from_rotation(Quat::from_rotation_y(std::f32::consts::PI)), skeleton))
        .observe(pose_character);
}

fn humanoid(gltf: &Gltf) -> Option<Skeleton> {
    let source = gltf.source.as_ref()?;
    let bones = source.extension_value("VRM")?.get("humanoid")?.get("humanBones")?.as_array()?;
    let node_name = |bone: &str| {
        let entry = bones.iter().find(|b| b.get("bone").and_then(|v| v.as_str()) == Some(bone))?;
        let index = entry.get("node")?.as_u64()? as usize;

        source.nodes().nth(index)?.name().map(str::to_owned)
    };

    BONES.iter().map(|bone| Some((*bone, node_name(bone)?))).collect::<Option<Vec<_>>>().map(Skeleton)
}

fn pose_character(
    ready: On<WorldInstanceReady>,
    mut commands: Commands,
    skeletons: Query<&Skeleton>,
    children: Query<&Children>,
    names: Query<&Name>,
    mut transforms: Query<&mut Transform>,
) {
    let Ok(skeleton) = skeletons.get(ready.entity) else {
        return;
    };

    let rig: HashMap<_, _> = children
        .iter_descendants(ready.entity)
        .filter_map(|e| {
            let name = names.get(e).ok()?;
            let (bone, _) = skeleton.0.iter().find(|(_, node)| node.as_str() == name.as_str())?;

            Some((*bone, e))
        })
        .collect();

    for (bone, angle) in [("leftUpperArm", ARM_DROP), ("rightUpperArm", -ARM_DROP)] {
        if let Some(mut arm) = rig.get(bone).and_then(|e| transforms.get_mut(*e).ok()) {
            arm.rotation = Quat::from_rotation_z(angle) * arm.rotation;
        }
    }

    commands.entity(ready.entity).remove::<Skeleton>().insert(Rig(rig));
}

#[derive(SystemParam)]
struct Drawn<'w, 's> {
    children: Query<'w, 's, &'static Children>,
    skinned: Query<'w, 's, (&'static Mesh3d, Option<&'static SkinnedMesh>)>,
    transforms: Query<'w, 's, &'static GlobalTransform>,
    meshes: Res<'w, Assets<Mesh>>,
    poses: Res<'w, Assets<SkinnedMeshInverseBindposes>>,
}

impl Drawn<'_, '_> {

    fn triangles(&self, root: Entity) -> Vec<Triangle> {
        self.children.iter_descendants(root).filter_map(|e| self.mesh_triangles(e)).flatten().collect()
    }

    fn mesh_triangles(&self, entity: Entity) -> Option<Vec<Triangle>> {
        let (mesh, skin) = self.skinned.get(entity).ok()?;
        let mesh = self.meshes.get(&mesh.0)?;
        let VertexAttributeValues::Float32x3(positions) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)? else {
            return None;
        };
        let placed: Vec<Vec3> = match skin {
            Some(skin) => {
                let VertexAttributeValues::Uint16x4(joints) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_INDEX)? else {
                    return None;
                };
                let VertexAttributeValues::Float32x4(weights) = mesh.attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT)? else {
                    return None;
                };
                let inverse = self.poses.get(&skin.inverse_bindposes)?;
                let bones: Vec<Mat4> = skin
                    .joints
                    .iter()
                    .zip(inverse.iter())
                    .map(|(joint, inverse)| Some(self.transforms.get(*joint).ok()?.to_matrix() * *inverse))
                    .collect::<Option<_>>()?;

                positions
                    .iter()
                    .zip(joints.iter().zip(weights))
                    .map(|(p, (joint, weight))| {
                        let total: f32 = weight.iter().sum();
                        let blended: Vec3 = (0..4)
                            .map(|k| weight[k] * bones[joint[k] as usize].transform_point3(Vec3::from_array(*p)))
                            .sum();

                        blended / total.max(f32::EPSILON)
                    })
                    .collect()
            }
            None => {
                let transform = self.transforms.get(entity).ok()?;

                positions.iter().map(|p| transform.transform_point(Vec3::from_array(*p))).collect()
            }
        };
        let indices: Vec<usize> = mesh.indices()?.iter().collect();

        Some(indices.as_chunks::<3>().0.iter().map(|t| t.map(|i| placed[i])).collect())
    }

}

fn fit_character(mut commands: Commands, rigs: Query<(Entity, &Rig)>, drawn: Drawn) {
    let Ok((entity, rig)) = rigs.single() else {
        return;
    };

    commands.entity(entity).remove::<Rig>();

    let found: Vec<Vec3> = BONES
        .iter()
        .filter_map(|bone| Some(drawn.transforms.get(*rig.0.get(bone)?).ok()?.translation()))
        .collect();
    let joints = <[Vec3; 12]>::try_from(found).unwrap_or_else(|_| {
        warn!("assets/{MODEL} is missing humanoid bones, the colliders use the stand-in proportions");
        STAND_IN
    });
    let mut figure = figure(&joints, false);
    let triangles = drawn.triangles(entity);

    if triangles.is_empty() {
        warn!("assets/{MODEL} has no readable meshes, the colliders use shapes fitted to the bones");
    } else {
        figure.adopt(triangles, joints[0], joints[1].y - joints[0].y);
    }

    commands.insert_resource(figure);
}

fn figure(joints: &[Vec3; 12], visible: bool) -> Figure {
    let [head, eye, neck, hips, arm_l, hand_l, arm_r, hand_r, leg_l, foot_l, leg_r, foot_r] = *joints;
    let shoulder = arm_l.distance(arm_r) * 0.5;
    let hip = leg_l.distance(leg_r) * 0.5;
    let waist = (leg_l.y + leg_r.y) * 0.5;
    let limb = |from: Vec3, to: Vec3, radius: f32| Part {
        center: (from + to) * 0.5,
        rotation: Quat::from_rotation_arc(Vec3::Y, (to - from).normalize()),
        shape: LiquidShape::Cuboid(Vec3::new(radius, from.distance(to) * 0.5, radius)),
    };
    let skull = (eye.y - head.y) * HEAD_SCALE;
    let arm = shoulder * 0.4;
    let chest = neck.y - (neck.y - waist) * SHOULDER_DEPTH;
    let block = |top: f32, bottom: f32, width: f32, depth: f32| Part {
        center: Vec3::new(hips.x, (top + bottom) * 0.5, hips.z),
        rotation: Quat::IDENTITY,
        shape: LiquidShape::Cuboid(Vec3::new(width, (top - bottom) * 0.5, depth)),
    };
    let center = Vec3::new(head.x, eye.y + (eye.y - head.y) * HEAD_LIFT, head.z);
    let parts = vec![
        Part {
            center,
            rotation: Quat::IDENTITY,
            shape: LiquidShape::Sphere(skull),
        },
        // the shoulders reach past the head so liquid that drips off a large head lands on them
        block(neck.y, chest, (shoulder + arm * 2.0).max(skull * 1.15), (shoulder * 0.8).max(skull * 0.75)),
        block(chest, waist, shoulder, shoulder * 0.8),
        limb(arm_l, hand_l + (hand_l - arm_l) * 0.2, arm),
        limb(arm_r, hand_r + (hand_r - arm_r) * 0.2, arm),
        limb(leg_l, foot_l.with_y(0.0), hip * 0.75),
        limb(leg_r, foot_r.with_y(0.0), hip * 0.75),
    ];
    let reach = |p: &Part| match p.shape {
        LiquidShape::Sphere(r) => Vec3::splat(r),
        LiquidShape::Cuboid(half) => Mat3::from_quat(p.rotation).abs() * half,
    };
    let lo = parts.iter().map(|p| p.center - reach(p)).reduce(Vec3::min).unwrap_or_default();
    let hi = parts.iter().map(|p| p.center + reach(p)).reduce(Vec3::max).unwrap_or_default();

    Figure { parts, triangles: Vec::new(), crown: center + Vec3::Y * skull, lo, hi, visible }
}

impl Figure {

    fn adopt(&mut self, triangles: Vec<Triangle>, head: Vec3, reach: f32) {
        let points = || triangles.iter().flatten().copied();

        self.lo = points().reduce(Vec3::min).unwrap_or(self.lo);
        self.hi = points().reduce(Vec3::max).unwrap_or(self.hi);
        self.crown = points()
            .filter(|p| p.xz().distance(head.xz()) < reach)
            .map(|p| p.y)
            .reduce(f32::max)
            .map_or(self.crown, |top| head.with_y(top));
        self.parts.clear();
        self.triangles = triangles;
    }

    fn liquid_origin(&self) -> Vec3 {
        (self.lo - LIQUID_MARGIN).with_y(0.0)
    }

    fn gas_origin(&self) -> Vec3 {
        let center = (self.lo + self.hi) * 0.5;

        Vec3::new(center.x - GAS_HALF_WIDTH, 0.0, center.z - GAS_HALF_WIDTH)
    }

}

fn closest_on_triangle(p: Vec3, [a, b, c]: Triangle) -> Vec3 {
    let (ab, ac, ap) = (b - a, c - a, p - a);
    let (d1, d2) = (ab.dot(ap), ac.dot(ap));

    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }

    let bp = p - b;
    let (d3, d4) = (ab.dot(bp), ac.dot(bp));

    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }

    let vc = d1 * d4 - d3 * d2;

    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + ab * (d1 / (d1 - d3));
    }

    let cp = p - c;
    let (d5, d6) = (ab.dot(cp), ac.dot(cp));

    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }

    let vb = d5 * d2 - d1 * d6;

    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + ac * (d2 / (d2 - d6));
    }

    let va = d3 * d6 - d5 * d4;

    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + (c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6)));
    }

    let sum = va + vb + vc;

    if sum.abs() < f32::MIN_POSITIVE {
        return a;
    }

    a + ab * (vb / sum) + ac * (vc / sum)
}

fn signed_distances(triangles: &[Triangle], origin: Vec3, dims: UVec3, dx: f32) -> Vec<f32> {
    let index = |p: UVec3| (p.x + dims.x * (p.y + dims.y * p.z)) as usize;
    let last = (dims - 1).as_vec3();
    let band = EXACT_CELLS * dx;
    let shell = SHELL_CELLS * dx;
    let mut near = vec![f32::MAX; dims.element_product() as usize];

    for triangle in triangles {
        let [a, b, c] = *triangle;
        let lo = ((a.min(b).min(c) - band - origin) / dx).floor().clamp(Vec3::ZERO, last).as_uvec3();
        let hi = ((a.max(b).max(c) + band - origin) / dx).ceil().clamp(Vec3::ZERO, last).as_uvec3();

        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let p = UVec3::new(x, y, z);
                    let point = origin + p.as_vec3() * dx;
                    let slot = &mut near[index(p)];
                    *slot = slot.min(point.distance(closest_on_triangle(point, *triangle)));
                }
            }
        }
    }

    let mut outside = vec![false; near.len()];
    let mut open: Vec<UVec3> = Vec::new();
    let visit = |p: UVec3, outside: &mut Vec<bool>, open: &mut Vec<UVec3>| {
        let i = index(p);

        if !outside[i] && near[i] >= shell {
            outside[i] = true;
            open.push(p);
        }
    };

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let p = UVec3::new(x, y, z);

                if p.cmpeq(UVec3::ZERO).any() || p.cmpeq(dims - 1).any() {
                    visit(p, &mut outside, &mut open);
                }
            }
        }
    }

    while let Some(p) = open.pop() {
        for axis in 0..3 {
            if p[axis] > 0 {
                let mut q = p;
                q[axis] -= 1;
                visit(q, &mut outside, &mut open);
            }

            if p[axis] + 1 < dims[axis] {
                let mut q = p;
                q[axis] += 1;
                visit(q, &mut outside, &mut open);
            }
        }
    }

    let exposed = |p: UVec3| {
        (0..3).any(|axis| {
            let below = p[axis] > 0 && outside[index(p - UVec3::AXES[axis])];
            let above = p[axis] + 1 < dims[axis] && outside[index(p + UVec3::AXES[axis])];

            below || above
        })
    };
    let mut distance = vec![f32::MIN; near.len()];

    for z in 0..dims.z {
        for y in 0..dims.y {
            for x in 0..dims.x {
                let p = UVec3::new(x, y, z);
                let i = index(p);

                if outside[i] {
                    distance[i] = if near[i] < f32::MAX { near[i] - shell } else { f32::MAX };
                } else if exposed(p) {
                    distance[i] = (near[i] - shell).min(-1.0e-4 * dx);
                }
            }
        }
    }

    spread(&mut distance, dims, dx);

    distance
}

fn spread(distance: &mut [f32], dims: UVec3, dx: f32) {
    let size = dims.as_ivec3();
    let index = |p: IVec3| (p.x + size.x * (p.y + size.y * p.z)) as usize;
    let offsets: Vec<IVec3> = (0..27)
        .map(|k| IVec3::new(k % 3, (k / 3) % 3, k / 9) - 1)
        .filter(|o| (o.z, o.y, o.x) < (0, 0, 0))
        .collect();
    let count = distance.len() as i32;

    for (order, direction) in [(0, 1), (count - 1, -1), (0, 1), (count - 1, -1)] {
        for step in 0..count {
            let i = order + direction * step;
            let p = IVec3::new(i % size.x, (i / size.x) % size.y, i / (size.x * size.y));
            let sign = distance[i as usize].signum();
            let mut best = distance[i as usize].abs();

            for offset in &offsets {
                let q = p + *offset * direction;

                if q.cmpge(IVec3::ZERO).all() && q.cmplt(size).all() {
                    let other = distance[index(q)];

                    if other.signum() == sign {
                        best = best.min(other.abs() + offset.as_vec3().length() * dx);
                    }
                }
            }

            distance[i as usize] = sign * best.min(1.0e3);
        }
    }
}

fn boxes(distance: &[f32], origin: Vec3, dims: UVec3, dx: f32) -> Vec<GasRegion> {
    let solid = |x: u32, y: u32, z: u32| distance[(x + dims.x * (y + dims.y * z)) as usize] < 0.0;
    let mut spans: Vec<(UVec3, UVec3)> = Vec::new();

    for z in 0..dims.z {
        let mut open: Vec<usize> = Vec::new();

        for x in 0..dims.x {
            let mut next = Vec::new();
            let mut y = 0;

            while y < dims.y {
                if !solid(x, y, z) {
                    y += 1;
                    continue;
                }

                let start = y;

                while y < dims.y && solid(x, y, z) {
                    y += 1;
                }

                let grown = open.iter().copied().find(|i| spans[*i].0.y == start && spans[*i].1.y == y);
                let span = grown.unwrap_or_else(|| {
                    spans.push((UVec3::new(x, start, z), UVec3::new(x, y, z + 1)));
                    spans.len() - 1
                });

                spans[span].1.x = x + 1;
                next.push(span);
            }

            open = next;
        }
    }

    spans
        .into_iter()
        .map(|(lo, hi)| GasRegion::cuboid(origin + (lo + hi).as_vec3() * 0.5 * dx, (hi - lo).as_vec3() * 0.5 * dx))
        .collect()
}

fn solids(figure: &Figure, quality: Quality) -> Solids {
    let gas = gas_config(figure, quality);

    if figure.triangles.is_empty() {
        let regions = figure.parts.iter().flat_map(|p| gas_obstacles(p, gas.dx * 0.9)).collect();

        return Solids { liquid: None, gas: regions, marks: Vec::new(), cell: 0.0 };
    }

    let liquid = liquid_config(figure, quality, Liquid::Honey);
    let nodes = liquid.dims + 1;
    let origin = figure.liquid_origin();
    let distance = signed_distances(&figure.triangles, origin, nodes, liquid.dx);
    let side = nodes.as_ivec3();
    let at = |p: IVec3| distance[(p.x + side.x * (p.y + side.y * p.z)) as usize];
    let marks = (0..distance.len() as i32)
        .map(|i| IVec3::new(i % side.x, (i / side.x) % side.y, i / (side.x * side.y)))
        .filter(|p| {
            at(*p) < 0.0
                && IVec3::AXES.iter().flat_map(|a| [*p + *a, *p - *a]).any(|q| {
                    q.cmpge(IVec3::ZERO).all() && q.cmplt(side).all() && at(q) >= 0.0
                })
        })
        .map(|p| origin + p.as_vec3() * liquid.dx)
        .collect();
    let centers = figure.gas_origin() + gas.dx * 0.5;
    let cells = signed_distances(&figure.triangles, centers, gas.dims, gas.dx);
    let regions = boxes(&cells, figure.gas_origin(), gas.dims, gas.dx);

    info!("character solid: liquid cell {:.4} m, gas cell {:.4} m, {} gas boxes", liquid.dx, gas.dx, regions.len());

    Solids { liquid: Some(Arc::new(distance)), gas: regions, marks, cell: liquid.dx }
}

fn build_scene(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut cameras: Query<&mut Orbit>,
    figure: Res<Figure>,
    lab: Res<Lab>,
) {
    let clay = materials.add(StandardMaterial {
        base_color: Color::srgb(0.82, 0.78, 0.72),
        perceptual_roughness: 0.6,
        ..default()
    });

    for part in &figure.parts {
        let (collider, mesh) = match part.shape {
            LiquidShape::Sphere(r) => (Collider::sphere(r), meshes.add(Sphere::new(r).mesh().uv(48, 24))),
            LiquidShape::Cuboid(half) => {
                (Collider::cuboid(half.x * 2.0, half.y * 2.0, half.z * 2.0), meshes.add(Cuboid::from_size(half * 2.0)))
            }
        };
        let mut body = commands.spawn((
            RigidBody::Static,
            collider,
            LiquidBody {
                shape: part.shape,
                density: 1000.0,
                dynamic: false,
            },
            Transform::from_translation(part.center).with_rotation(part.rotation),
        ));

        if figure.visible {
            body.insert((Mesh3d(mesh), MeshMaterial3d(clay.clone())));
        }
    }

    let solids = solids(&figure, lab.quality);
    let mut liquid = commands.spawn((
        liquid_domain(&figure, lab.quality, lab.liquid, false),
        Transform::from_translation(figure.liquid_origin()),
        Visibility::default(),
    ));

    if let Some(distance) = &solids.liquid {
        liquid.insert(LiquidSolid(distance.clone()));
    }

    commands.spawn(gas_domain(&figure, &solids, lab.quality, lab.smoke));
    commands.insert_resource(solids);

    for mut orbit in &mut cameras {
        orbit.target.y = figure.hi.y * 0.55;
    }
}

fn properties(liquid: Liquid) -> LiquidProperties {
    let water = LiquidProperties::water();
    let look = WaterLook {
        sun: SUN.normalize().extend(60.0),
        ..water.look
    };

    match liquid {
        Liquid::Water => LiquidProperties { look, ..water },
        Liquid::Honey => LiquidProperties {
            density: 1420.0,
            viscosity: 150.0,
            surface_tension: 2.0,
            adhesion: 1.2,
            look: WaterLook {
                absorption: Vec4::new(4.0, 14.0, 45.0, 0.0),
                scatter: Vec4::new(0.6, 0.12, 0.0, 0.0),
                sky_zenith: Vec4::new(0.25, 0.24, 0.22, 0.0),
                sky_horizon: Vec4::new(0.45, 0.42, 0.38, 0.0),
                ..look
            },
            ..water
        },
    }
}

fn liquid_config(figure: &Figure, quality: Quality, liquid: Liquid) -> SolverConfig {
    let size = figure.hi + LIQUID_MARGIN - figure.liquid_origin();

    SolverConfig {
        dims: (size / LIQUID_DX).ceil().as_uvec3(),
        dx: LIQUID_DX,
        max_particles: 200_000,
        cfl: 2.5,
        tolerance: 1e-3,
        viscosity_iterations: 40,
        ..default()
    }
    .with_properties(&properties(liquid))
    .with_quality(quality)
}

fn spout(figure: &Figure) -> Vec3 {
    figure.crown + Vec3::Y * (0.06 + BLOB_RADIUS)
}

fn faucet(figure: &Figure, liquid: Liquid, kind: Faucet) -> (Faucet, LiquidEmitter) {
    let radius = if kind == Faucet::Drop { BLOB_RADIUS } else { STREAM_RADIUS };
    let open = LiquidEmitter::new(GasRegion::sphere(spout(figure), radius), properties(liquid).temperature);

    let emitter = match kind {
        Faucet::Drop => LiquidEmitter { amount: Some(BLOB_VOLUME), ..open },
        Faucet::Stream => LiquidEmitter { flow: Some(STREAM_FLOW), ..open },
    };

    (kind, emitter)
}

fn liquid_domain(figure: &Figure, quality: Quality, liquid: Liquid, filled: bool) -> LiquidDomain {
    let center = spout(figure) - figure.liquid_origin();
    let domain = LiquidDomain::new(liquid_config(figure, quality, liquid)).with_look(properties(liquid).look);

    if !filled {
        return domain;
    }

    domain.filled_where(|p| p.distance(center) < BLOB_RADIUS)
}

fn gas_config(figure: &Figure, quality: Quality) -> GasConfig {
    let size = Vec3::new(GAS_HALF_WIDTH * 2.0, figure.hi.y + GAS_HEADROOM, GAS_HALF_WIDTH * 2.0);

    GasConfig {
        dims: (size / GAS_DX).ceil().as_uvec3(),
        dx: GAS_DX,
        params: GasParams {
            vorticity_confinement: 4.0,
            density_decay: 0.5,
            cooling: 0.6,
            cfl: 2.0,
            open_low: [true, false, true],
            open_high: [true; 3],
            ..default()
        },
        ..default()
    }
    .with_quality(quality)
}

fn gas_domain(figure: &Figure, solids: &Solids, quality: Quality, smoke: Option<Smoke>) -> GasDomain {
    GasDomain {
        emitters: smoke.map(|s| emitter(figure, s)).into_iter().collect(),
        obstacles: solids.gas.clone(),
        ..GasDomain::new(gas_config(figure, quality), figure.gas_origin())
    }
}

fn gas_obstacles(part: &Part, thinnest: f32) -> Vec<GasRegion> {
    match part.shape {
        LiquidShape::Sphere(r) => vec![GasRegion::sphere(part.center, r.max(thinnest))],
        LiquidShape::Cuboid(half) if part.rotation.angle_between(Quat::IDENTITY) < 0.05 => {
            vec![GasRegion::cuboid(part.center, half.max(Vec3::splat(thinnest)))]
        }
        LiquidShape::Cuboid(half) => {
            // gas obstacles are axis aligned, so a tilted limb becomes a chain of spheres along its length
            let radius = half.x.max(thinnest);
            let count = (half.y * 2.0 / radius).ceil().max(1.0) as usize;
            let axis = part.rotation * Vec3::Y;

            (0..=count)
                .map(|i| GasRegion::sphere(part.center + axis * half.y * (2.0 * i as f32 / count as f32 - 1.0), radius))
                .collect()
        }
    }
}

fn emitter(figure: &Figure, smoke: Smoke) -> GasEmitter {
    match smoke {
        Smoke::Pour => GasEmitter {
            region: GasRegion::sphere(figure.crown.with_y(figure.hi.y + 0.45), 0.14),
            density_rate: 3.5,
            heat_rate: -500.0,
        },
        Smoke::Rise => GasEmitter {
            region: GasRegion::cuboid(figure.crown.with_y(0.06), Vec3::new(0.25, 0.05, 0.18)),
            density_rate: 5.0,
            heat_rate: 40.0,
        },
    }
}

fn read_keys(keys: Res<ButtonInput<KeyCode>>, mut lab: ResMut<Lab>, mut actions: MessageWriter<Action>) {
    let drops = [(KeyCode::Digit1, Action::Honey), (KeyCode::Digit2, Action::Water)];

    if let Some((_, action)) = drops.into_iter().find(|(key, _)| keys.pressed(*key)) {
        actions.write(action);
    }

    let bindings = [
        (KeyCode::Digit3, Action::Pour),
        (KeyCode::Digit4, Action::Rise),
        (KeyCode::KeyR, Action::Reset),
        (KeyCode::Tab, Action::Quality),
    ];

    for (key, action) in bindings {
        if keys.just_pressed(key) {
            actions.write(action);
        }
    }

    if keys.just_pressed(KeyCode::KeyC) {
        lab.colliders = !lab.colliders;
    }
}

fn apply_actions(
    mut commands: Commands,
    mut actions: MessageReader<Action>,
    mut lab: ResMut<Lab>,
    mut solids: ResMut<Solids>,
    figure: Res<Figure>,
    mut liquids: Query<(&mut LiquidDomain, Option<&mut LiquidSolid>)>,
    mut gases: Query<(Entity, &mut GasDomain)>,
) {
    let mut dropped = false;

    for action in actions.read() {
        let mut filled = None;
        let mut cleared = false;

        match action {
            Action::Honey | Action::Water => {
                let liquid = if *action == Action::Honey { Liquid::Honey } else { Liquid::Water };
                dropped = true;

                if lab.liquid != liquid {
                    (lab.liquid, filled) = (liquid, Some(true));
                }
            }
            Action::Pour | Action::Rise => {
                let mode = if *action == Action::Pour { Smoke::Pour } else { Smoke::Rise };
                lab.smoke = (lab.smoke != Some(mode)).then_some(mode);
            }
            Action::Reset => (lab.smoke, filled, cleared) = (None, Some(false), true),
            Action::Quality => {
                lab.quality = match lab.quality {
                    Quality::Game => Quality::High,
                    Quality::High => Quality::Game,
                };
                (filled, cleared) = (Some(false), true);
                *solids = self::solids(&figure, lab.quality);
                info!("liquid preset {}", lab.quality.name());
            }
        }

        if let Some(filled) = filled {
            for (mut domain, solid) in &mut liquids {
                if let (Some(mut solid), Some(distance)) = (solid, &solids.liquid) {
                    solid.0 = distance.clone();
                }

                *domain = liquid_domain(&figure, lab.quality, lab.liquid, filled);
            }
        }

        for (entity, mut domain) in &mut gases {
            if cleared {
                commands.entity(entity).despawn();
                commands.spawn(gas_domain(&figure, &solids, lab.quality, lab.smoke));
            } else {
                domain.emitters = lab.smoke.map(|s| emitter(&figure, s)).into_iter().collect();
            }
        }
    }

    lab.pouring = if dropped { lab.pouring + 1 } else { 0 };
}

fn run_faucets(
    mut commands: Commands,
    lab: Res<Lab>,
    figure: Res<Figure>,
    rebuilt: Query<(), Changed<LiquidDomain>>,
    faucets: Query<(Entity, &Faucet)>,
) {
    let rebuilt = !rebuilt.is_empty();

    for (entity, kind) in &faucets {
        let spent = match kind {
            Faucet::Drop => rebuilt || lab.pouring == 1,
            Faucet::Stream => lab.pouring < 2,
        };

        if spent {
            commands.entity(entity).despawn();
        }
    }

    let kind = match lab.pouring {
        1 if !rebuilt => Faucet::Drop,
        2 => Faucet::Stream,
        _ => return,
    };

    commands.spawn(faucet(&figure, lab.liquid, kind));
}

fn orbit_camera(
    buttons: Res<ButtonInput<MouseButton>>,
    motion: Res<AccumulatedMouseMotion>,
    scroll: Res<AccumulatedMouseScroll>,
    mut cameras: Query<(&mut Transform, &mut Orbit)>,
) {
    for (mut transform, mut orbit) in &mut cameras {
        if buttons.pressed(MouseButton::Left) {
            orbit.yaw -= motion.delta.x * 0.005;
            orbit.pitch = (orbit.pitch + motion.delta.y * 0.005).clamp(-0.3, 1.5);
        }

        let zoom = (1.0 - scroll.delta.y.clamp(-3.0, 3.0) * 0.1).clamp(0.5, 2.0);
        orbit.distance = (orbit.distance * zoom).clamp(0.5, 8.0);

        let rotation = Quat::from_euler(EulerRot::YXZ, orbit.yaw, -orbit.pitch, 0.0);
        transform.translation = orbit.target + rotation * Vec3::Z * orbit.distance;
        transform.look_at(orbit.target, Vec3::Y);
    }
}

fn draw_colliders(lab: Res<Lab>, figure: Res<Figure>, solids: Res<Solids>, mut gizmos: Gizmos) {
    if !lab.colliders {
        return;
    }

    let color = Color::srgb(0.2, 1.0, 0.4);

    for part in &figure.parts {
        match part.shape {
            LiquidShape::Sphere(r) => {
                gizmos.sphere(Isometry3d::from_translation(part.center), r, color);
            }
            LiquidShape::Cuboid(half) => gizmos.cube(
                Transform::from_translation(part.center).with_rotation(part.rotation).with_scale(half * 2.0),
                color,
            ),
        }
    }

    for mark in &solids.marks {
        gizmos.cube(Transform::from_translation(*mark).with_scale(Vec3::splat(solids.cell * MARK_SCALE)), color);
    }
}

fn run_script(
    time: Res<Time>,
    mut lab: ResMut<Lab>,
    mut commands: Commands,
    mut actions: MessageWriter<Action>,
    mut exit: MessageWriter<AppExit>,
) {
    let Ok(path) = std::env::var("LIQUID_CAPTURE") else {
        return;
    };

    let before = lab.clock;
    lab.clock += time.delta_secs();
    let due = |at: f32| before <= at && at < lab.clock;

    for (_, action) in SCRIPT.iter().filter(|(at, _)| due(*at)) {
        actions.write(*action);
    }

    if (SCRIPT_HOLD.0..SCRIPT_HOLD.1).contains(&lab.clock) {
        actions.write(Action::Honey);
    }

    for at in CAPTURES.iter().filter(|at| due(**at)) {
        let file = format!("{}-{at}s.png", std::path::Path::new(&path).with_extension("").display());
        commands.spawn(Screenshot::primary_window()).observe(save_to_disk(file));
    }

    if lab.clock > CAPTURES[CAPTURES.len() - 1] + 1.0 {
        exit.write(AppExit::Success);
    }
}

fn bench_paths() -> Vec<DiagnosticPath> {
    [GPU_TIME, ENCODE_TIME].into_iter().chain(SECTIONS).chain([SUBSTEPS, PARTICLES, GAS_TIME, VOLUME_TIME]).collect()
}

fn record_bench(
    lab: Res<Lab>,
    store: Res<DiagnosticsStore>,
    mut bench: ResMut<Bench>,
    mut actions: MessageWriter<Action>,
    mut exit: MessageWriter<AppExit>,
) {
    let Some((label, action)) = BENCH_RUNS.get(bench.run) else {
        exit.write(AppExit::Success);
        return;
    };

    if bench.seen == 0 {
        actions.write(*action);
    }

    bench.seen += 1;
    let now = Instant::now();
    let previous = bench.last.replace(now);

    if bench.seen <= BENCH_WARMUP {
        return;
    }

    if let Some(previous) = previous {
        bench.wall.push((now - previous).as_secs_f64() * 1000.0);
    }

    let bench = &mut *bench;

    for (samples, path) in bench.gpu.iter_mut().zip(bench_paths()) {
        if let Some(value) = store.get(&path).and_then(|d| d.value()) {
            samples.push(value);
        }
    }

    if bench.seen < BENCH_WARMUP + bench.frames {
        return;
    }

    let stats = |values: &[f64]| {
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let at = |q: f64| sorted.get(((sorted.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0.0);
        (at(0.5), at(0.95))
    };
    let (median, p95) = stats(&bench.wall);

    info!(
        "bench {label} frames {} preset {}: wall median {median:.2} ms p95 {p95:.2} ms ({:.1} fps)",
        bench.wall.len(),
        lab.quality.name(),
        1000.0 / median
    );

    for (samples, path) in bench.gpu.iter_mut().zip(bench_paths()) {
        let (median, p95) = stats(samples);

        if median > 0.0 {
            info!("bench {label} {path}: median {median:.2} p95 {p95:.2}");
        }

        samples.clear();
    }

    bench.wall.clear();
    bench.seen = 0;
    bench.run += 1;
}
