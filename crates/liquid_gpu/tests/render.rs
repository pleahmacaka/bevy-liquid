use bevy::math::{UVec3, Vec3};
use bevy::render::render_resource::encase::UniformBuffer;
use liquid_gpu::render::{
    LiquidSurfaceSettings, SDF_BAND_CELLS, SDF_BRICK, SURFACE_ISO, SdfSources, SdfTargets, SurfaceSdf, brick_grid,
};
use wgpu::util::DeviceExt;

#[cfg(windows)]
mod common;

#[path = "../../liquid_ocean/tests/common/limits.rs"]
mod limits;

const DIMS: UVec3 = UVec3::new(40, 32, 36);
const DX: f32 = 0.0625;
const ORIGIN: Vec3 = Vec3::new(-1.0, 0.25, 0.5);
const TRACE_SHADER: &str = include_str!("../src/shaders/brick_trace.wesl");

fn gpu() -> (wgpu::Device, wgpu::Queue) {
    static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();
    let instance = INSTANCE.get_or_init(|| wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env()));
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env(instance, None))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
        })
    .expect("no adapter");

    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("render test"),
        required_features: wgpu::Features::empty(),
        required_limits: limits::portable(adapter.limits()),
        ..Default::default()
    }))
    .expect("no device")
}

fn read<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let bytes = (count.max(1) * std::mem::size_of::<T>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);

    staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");

    let view = staging.get_mapped_range(..).expect("mapped range");
    let out: Vec<T> = bytemuck::cast_slice(&view).to_vec();

    out[..count].to_vec()
}

fn validated<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> T {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = build();

    if let Some(error) = pollster::block_on(scope.pop()) {
        panic!("wgpu validation error: {error}");
    }

    out
}

fn storage<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    })
}

fn cells() -> usize {
    DIMS.element_product() as usize
}

fn center(i: usize) -> Vec3 {
    let d = DIMS.as_ivec3();
    let i = i as i32;
    let c = bevy::math::IVec3::new(i % d.x, (i / d.x) % d.y, i / (d.x * d.y));

    (c.as_vec3() + 0.5) * DX
}

fn field(density: impl Fn(Vec3) -> f32) -> Vec<f32> {
    (0..cells()).map(|i| density(center(i))).collect()
}

fn plane_normal() -> Vec3 {
    Vec3::new(0.3, 1.0, 0.2).normalize()
}

fn plane_point() -> Vec3 {
    DIMS.as_vec3() * DX * 0.5
}

fn plane_distance(x: Vec3) -> f32 {
    plane_normal().dot(x - plane_point())
}

fn sphere_center() -> Vec3 {
    Vec3::new(19.3, 15.6, 17.2) * DX
}

const SPHERE_RADIUS: f32 = 9.0 * DX;

fn sphere_distance(x: Vec3) -> f32 {
    (x - sphere_center()).length() - SPHERE_RADIUS
}

fn ramp(signed: f32) -> f32 {
    (SURFACE_ISO - signed / (6.0 * DX)).clamp(0.0, 1.0)
}

fn blobs(x: Vec3) -> f32 {
    let blobs = [
        (Vec3::new(10.0, 9.0, 12.0), 3.5),
        (Vec3::new(14.0, 11.0, 13.0), 2.5),
        (Vec3::new(25.0, 20.0, 18.0), 5.0),
        (Vec3::new(30.0, 8.0, 25.0), 2.0),
        (Vec3::new(31.5, 10.5, 25.0), 1.6),
        (Vec3::new(8.0, 25.0, 28.0), 4.0),
        (Vec3::new(20.0, 5.0, 6.0), 3.0),
        (Vec3::new(22.0, 26.0, 9.0), 1.2),
        (Vec3::new(35.0, 28.0, 5.0), 2.8),
    ];

    blobs
        .iter()
        .map(|(c, s)| {
            let r = (x / DX - *c).length() / s;
            (-0.5 * r * r).exp()
        })
        .sum::<f32>()
        .min(1.5)
}

struct Built {
    distances: Vec<f32>,
    bricks: Vec<f32>,
}

fn build(device: &wgpu::Device, queue: &wgpu::Queue, density: &[f32]) -> (Built, [wgpu::Buffer; 4]) {
    let density_buffer = storage(device, density);
    let particles = storage(device, &[0f32; 20]);
    let cell_start = storage(device, &vec![0u32; cells() + 1]);
    let brick_count = brick_grid(DIMS).element_product() as usize;
    let distances = storage(device, &vec![0f32; cells()]);
    let bricks = storage(device, &vec![0f32; brick_count]);
    let motion = storage(device, &vec![[0f32; 4]; cells()]);

    let sdf = validated(device, || {
        SurfaceSdf::new(
            device,
            DIMS,
            DX,
            SdfSources { density: &density_buffer, particles: &particles, cell_start: &cell_start },
            SdfTargets { distances: &distances, bricks: &bricks, motion: &motion },
        )
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    sdf.encode(&mut encoder);
    queue.submit([encoder.finish()]);

    let built = Built {
        distances: read(device, queue, &distances, cells()),
        bricks: read(device, queue, &bricks, brick_count),
    };

    (built, [density_buffer, distances, bricks, motion])
}

fn band_error(distances: &[f32], truth: impl Fn(Vec3) -> f32, density: &[f32]) -> (f32, usize) {
    let band = SDF_BAND_CELLS * DX;
    let mut worst = 0.0f32;
    let mut checked = 0;

    for (i, d) in distances.iter().enumerate() {
        let inside = density[i] >= SURFACE_ISO;
        assert_eq!(*d <= 0.0, inside, "cell {i} distance {d} has the wrong sign");

        let expected = truth(center(i));
        let x = center(i);
        let margin = (x.min(DIMS.as_vec3() * DX - x) / DX).min_element();

        if margin < SDF_BAND_CELLS {
            continue;
        }

        if expected.abs() < band - DX {
            worst = worst.max((d - expected).abs() / DX);
            checked += 1;
        } else if expected.abs() > band + DX {
            assert!((d.abs() - band).abs() < 1.0e-5, "cell {i} beyond the band holds {d}");
        }
    }

    (worst, checked)
}

#[cfg(windows)]
#[test]
fn sdf_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|device, queue| {
        build(device, queue, &field(|x| ramp(plane_distance(x))));
    });
}

#[test]
fn sdf_matches_plane_distance() {
    let (device, queue) = gpu();
    let density = field(|x| ramp(plane_distance(x)));
    let (built, _) = build(&device, &queue, &density);
    let (worst, checked) = band_error(&built.distances, plane_distance, &density);

    println!("plane: {checked} band cells, max error {worst:.4} dx");
    assert!(checked > 1000);
    assert!(worst < 0.01, "plane distance error {worst} dx");
}

#[test]
fn sdf_matches_sphere_distance() {
    let (device, queue) = gpu();
    let density = field(|x| ramp(sphere_distance(x)));
    let (built, _) = build(&device, &queue, &density);
    let (worst, checked) = band_error(&built.distances, sphere_distance, &density);

    println!("sphere: {checked} band cells, max error {worst:.4} dx");
    assert!(checked > 1000);
    assert!(worst < 0.12, "sphere distance error {worst} dx");
}

fn trilinear(density: &[f32], x: Vec3) -> f32 {
    let d = DIMS.as_ivec3();
    let g = x / DX - 0.5;
    let b = g.floor();
    let t = g - b;
    let value = |o: bevy::math::IVec3| {
        let c = (b.as_ivec3() + o).clamp(bevy::math::IVec3::ZERO, d - 1);
        density[(c.x + d.x * (c.y + d.y * c.z)) as usize]
    };
    let mut sum = 0.0;

    for k in 0..8 {
        let o = bevy::math::IVec3::new(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let w = Vec3::select(o.cmpeq(bevy::math::IVec3::ONE), t, 1.0 - t);
        sum += w.x * w.y * w.z * value(o);
    }

    sum
}

#[test]
fn empty_bricks_contain_no_liquid() {
    let (device, queue) = gpu();
    let grid = brick_grid(DIMS);

    for (name, density) in [
        ("sphere", field(|x| ramp(sphere_distance(x)))),
        ("blobs", field(blobs)),
        ("plane", field(|x| ramp(plane_distance(x)))),
    ] {
        let (built, _) = build(&device, &queue, &density);
        let mut empty = 0;
        let mut probes = 0;

        for (b, value) in built.bricks.iter().enumerate() {
            if *value <= 0.0 {
                continue;
            }

            empty += 1;
            let b = b as u32;
            let brick = UVec3::new(b % grid.x, (b / grid.x) % grid.y, b / (grid.x * grid.y));
            let lo = brick * SDF_BRICK;
            let hi = (lo + SDF_BRICK).min(DIMS);

            for z in lo.z * 4..hi.z * 4 {
                for y in lo.y * 4..hi.y * 4 {
                    for x in lo.x * 4..hi.x * 4 {
                        let p = (UVec3::new(x, y, z).as_vec3() + 0.5) * (DX / 4.0);
                        let rho = trilinear(&density, p);
                        probes += 1;
                        assert!(rho < SURFACE_ISO, "{name}: brick {b} marked empty but holds liquid ({rho}) at {p}");
                    }
                }
            }
        }

        println!("{name}: {empty} of {} bricks empty, {probes} probes outside the liquid", built.bricks.len());
        assert!(empty > 0, "{name}: no brick was skipped");
    }
}

const TRACE_TEST: &str = r#"
struct Ray {
    origin: vec4<f32>,
    dir: vec4<f32>,
}

struct Outcome {
    traced: f32,
    brute: f32,
    traced_hit: u32,
    brute_hit: u32,
    traced_samples: u32,
    brute_samples: u32,
    fixed_samples: u32,
    pad: u32,
}

@group(1) @binding(0) var<storage, read> rays: array<Ray>;
@group(1) @binding(1) var<storage, read_write> outcomes: array<Outcome>;

const FINE: f32 = 0.02;

fn first_crossing(eye: vec3<f32>, dir: vec3<f32>, t0: f32, t1: f32, stride: f32, refine: i32) -> vec3<f32> {
    var t = t0;
    var prev = t0;
    var samples = 0.0;

    loop {
        if t > t1 {
            if prev >= t1 {
                return vec3<f32>(t1, 0.0, samples);
            }

            t = t1;
        }

        samples = samples + 8.0;

        if sample_field(eye + dir * t) >= settings.iso {
            break;
        }

        prev = t;
        t = t + stride;
    }

    var lo = prev;
    var hi = t;

    for (var i = 0; i < refine; i = i + 1) {
        let mid = 0.5 * (lo + hi);
        samples = samples + 8.0;

        if sample_field(eye + dir * mid) >= settings.iso {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    return vec3<f32>(hi, 1.0, samples);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= arrayLength(&rays) {
        return;
    }

    let eye = rays[i].origin.xyz;
    let dir = rays[i].dir.xyz;
    let span = box_interval(eye, dir);
    let traced = surface_hit(eye, dir, 3.0e38);
    let dx = settings.origin_dx.w;
    let brute = first_crossing(eye, dir, span.x, span.y, FINE * dx, 24);
    let fixed = first_crossing(eye, dir, span.x, span.y, 0.5 * dx, 6);

    var out: Outcome;
    out.traced = traced.t;
    out.brute = brute.x;
    out.traced_hit = u32(traced.hit);
    out.brute_hit = u32(brute.y > 0.5);
    out.traced_samples = traced.samples;
    out.brute_samples = u32(brute.z);
    out.fixed_samples = u32(fixed.z);
    outcomes[i] = out;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Outcome {
    traced: f32,
    brute: f32,
    traced_hit: u32,
    brute_hit: u32,
    traced_samples: u32,
    brute_samples: u32,
    fixed_samples: u32,
    pad: u32,
}

fn rays() -> Vec<[[f32; 4]; 2]> {
    let size = DIMS.as_vec3() * DX;
    let middle = ORIGIN + size * 0.5;
    let eyes = [
        middle + Vec3::new(0.3, 2.2, -3.1),
        middle + Vec3::new(-3.4, 0.9, 0.4),
        middle + Vec3::new(2.0, -1.8, 2.6),
        middle + Vec3::new(0.05, 3.5, 0.1),
    ];
    let mut out = Vec::new();

    for eye in eyes {
        let forward = (middle - eye).normalize();
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);

        for v in 0..64 {
            for u in 0..64 {
                let s = (u as f32 + 0.5) / 64.0 * 2.0 - 1.0;
                let t = (v as f32 + 0.5) / 64.0 * 2.0 - 1.0;
                let dir = (forward + (right * s + up * t) * 0.45).normalize();
                out.push([eye.extend(0.0).to_array(), dir.extend(0.0).to_array()]);
            }
        }
    }

    out
}

fn entries<'a>(pairs: &[(u32, &'a wgpu::Buffer)]) -> Vec<wgpu::BindGroupEntry<'a>> {
    pairs
        .iter()
        .map(|(binding, buffer)| wgpu::BindGroupEntry { binding: *binding, resource: buffer.as_entire_binding() })
        .collect()
}

#[test]
fn trace_matches_fine_march() {
    let (device, queue) = gpu();
    let source = format!("{}\n{}", TRACE_SHADER.replace("constants::MATERIAL_BIND_GROUP", "0"), TRACE_TEST);
    let module = validated(&device, || {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("trace test"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    });
    let pipeline = validated(&device, || {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("trace test"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        })
    });

    let mut settings_bytes = UniformBuffer::new(Vec::<u8>::new());
    settings_bytes.write(&LiquidSurfaceSettings::new(DIMS, DX, ORIGIN)).expect("settings");
    let settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &settings_bytes.into_inner(),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let rays = rays();
    let ray_buffer = storage(&device, &rays);
    let outcome_buffer = storage(&device, &vec![[0u32; 8]; rays.len()]);

    for (name, density) in [
        ("sphere", field(|x| ramp(sphere_distance(x)))),
        ("blobs", field(blobs)),
        ("plane", field(|x| ramp(plane_distance(x)))),
    ] {
        let (_, [field_buffer, distances, bricks, _]) = build(&device, &queue, &density);
        let surface = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries(&[(0, &field_buffer), (1, &settings), (2, &distances), (3, &bricks)]),
        });
        let io = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(1),
            entries: &entries(&[(0, &ray_buffer), (1, &outcome_buffer)]),
        });

        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &surface, &[]);
            pass.set_bind_group(1, &io, &[]);
            pass.dispatch_workgroups((rays.len() as u32).div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);

        let outcomes: Vec<Outcome> = read(&device, &queue, &outcome_buffer, rays.len());
        let both = outcomes.iter().filter(|o| o.traced_hit == 1 && o.brute_hit == 1).count();
        let disagree = outcomes.iter().filter(|o| o.traced_hit != o.brute_hit).count();
        let worst = outcomes
            .iter()
            .filter(|o| o.traced_hit == 1 && o.brute_hit == 1)
            .map(|o| (o.traced - o.brute).abs() / DX)
            .fold(0.0f32, f32::max);
        let mean = |f: fn(&Outcome) -> u32| outcomes.iter().map(|o| f(o) as f64).sum::<f64>() / outcomes.len() as f64;

        println!(
            "{name}: {both} hits of {} rays, {disagree} hit/miss disagreements, max |t - t_fine| {worst:.4} dx, \
             loads per ray: traced {:.1}, fixed 0.5 dx march {:.1}, fine march {:.1}",
            rays.len(),
            mean(|o| o.traced_samples),
            mean(|o| o.fixed_samples),
            mean(|o| o.brute_samples),
        );

        assert!(both > rays.len() / 20, "{name}: too few rays hit the surface");
        assert_eq!(disagree, 0, "{name}: traced and fine march disagree on hits");
        assert!(worst < 0.01, "{name}: hit distance differs by {worst} dx");
        assert!(mean(|o| o.traced_samples) < 0.5 * mean(|o| o.fixed_samples));
    }
}
