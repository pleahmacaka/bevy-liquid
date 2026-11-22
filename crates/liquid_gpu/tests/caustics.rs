use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[cfg(windows)]
mod common;

const SHADER: &str = include_str!("../src/shaders/caustics.wgsl");
const DIMS: [u32; 3] = [64, 32, 48];
const DX: f32 = 1.0 / 16.0;
const TEXELS_PER_CELL: u32 = 4;
const RAYS_PER_TEXEL: u32 = 2;
const MARGIN_TEXELS: u32 = 24;
const WEIGHT_SCALE: u32 = 256;
const HEADER_BYTES: u64 = 32;
const BORDER_TEXELS: u32 = 8;
const LAKE_DEPTH: f32 = 0.45;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    dims: [u32; 4],
    origin_dx: [f32; 4],
    sun_iso: [f32; 4],
    map: [u32; 4],
}

struct Stats {
    mean: f32,
    min: f32,
    max: f32,
}

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
        label: Some("caustics test"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no device")
}

fn field(height: impl Fn(f32, f32) -> f32) -> Vec<f32> {
    let [nx, ny, nz] = DIMS;
    let mut out = Vec::with_capacity((nx * ny * nz) as usize);

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let x = (i as f32 + 0.5) * DX;
                let y = (j as f32 + 0.5) * DX;
                let z = (k as f32 + 0.5) * DX;
                out.push((0.5 + (height(x, z) - y) / (4.0 * DX)).clamp(0.0, 1.0));
            }
        }
    }

    out
}

fn trace_pipeline(device: &wgpu::Device) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("caustics"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });

    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("caustics"),
        layout: None,
        module: &module,
        entry_point: Some("trace"),
        compilation_options: Default::default(),
        cache: None,
    })
}

#[cfg(windows)]
#[test]
fn caustics_compile_under_fxc() {
    common::assert_compiles_under_fxc(|device, _| {
        trace_pipeline(device);
    });
}

fn trace(values: &[f32]) -> (Vec<f32>, [u32; 2]) {
    let (device, queue) = gpu();
    let size = [DIMS[0] * TEXELS_PER_CELL, DIMS[2] * TEXELS_PER_CELL];
    let rays = size.map(|s| (s + 2 * MARGIN_TEXELS) * RAYS_PER_TEXEL);
    let sun = glam::Vec3::new(0.4, 0.7, 0.45).normalize();
    let params = Params {
        dims: [DIMS[0], DIMS[1], DIMS[2], RAYS_PER_TEXEL],
        origin_dx: [0.0, 0.0, 0.0, DX],
        sun_iso: sun.extend(0.5).to_array(),
        map: [size[0], size[1], MARGIN_TEXELS, WEIGHT_SCALE],
    };

    let pipeline = trace_pipeline(&device);

    let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("params"),
        contents: bytemuck::bytes_of(&params),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let field_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("field"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let texel_bytes = (size[0] * size[1] * 4) as u64;
    let map = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("map"),
        size: HEADER_BYTES + texel_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: texel_bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("caustics"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: params_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: field_buffer.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: map.as_entire_binding() },
        ],
    });

    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(rays[0].div_ceil(8), rays[1].div_ceil(8), 1);
    }
    encoder.copy_buffer_to_buffer(&map, HEADER_BYTES, &staging, 0, texel_bytes);
    queue.submit([encoder.finish()]);

    staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("map failed"));
    device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");

    let view = staging.get_mapped_range(..).expect("mapped range");
    let raw: &[u32] = bytemuck::cast_slice(&view);
    let norm = 1.0 / (RAYS_PER_TEXEL * RAYS_PER_TEXEL * WEIGHT_SCALE) as f32;

    (raw.iter().map(|&v| v as f32 * norm).collect(), size)
}

fn interior_stats(name: &str, values: &[f32], size: [u32; 2]) -> Stats {
    let interior: Vec<f32> = (BORDER_TEXELS..size[1] - BORDER_TEXELS)
        .flat_map(|y| (BORDER_TEXELS..size[0] - BORDER_TEXELS).map(move |x| values[(x + y * size[0]) as usize]))
        .collect();
    let stats = Stats {
        mean: (interior.iter().map(|&v| v as f64).sum::<f64>() / interior.len() as f64) as f32,
        min: interior.iter().copied().fold(f32::MAX, f32::min),
        max: interior.iter().copied().fold(f32::MIN, f32::max),
    };

    println!("{name}: mean {:.4} min {:.4} max {:.4}", stats.mean, stats.min, stats.max);

    stats
}

fn assert_uniform(stats: &Stats) {
    assert!((stats.mean - 1.0).abs() < 0.01, "mean {} not within 0.01 of 1", stats.mean);
    assert!(stats.min > 0.97 && stats.max < 1.03, "range {}..{} not within 0.97..1.03", stats.min, stats.max);
}

#[test]
fn flat_water_lights_the_floor_uniformly() {
    let (map, size) = trace(&field(|_, _| LAKE_DEPTH));

    assert_uniform(&interior_stats("flat", &map, size));
}

#[test]
fn dry_floor_stays_at_full_light() {
    let (map, size) = trace(&field(|_, _| -1.0));

    assert_uniform(&interior_stats("dry", &map, size));
}

#[test]
fn sine_wave_conserves_flux_while_focusing() {
    let wave = |x: f32, _: f32| LAKE_DEPTH + 0.02 * (std::f32::consts::TAU * x / 0.6).sin();
    let (map, size) = trace(&field(wave));
    let stats = interior_stats("wave", &map, size);

    assert!((stats.mean - 1.0).abs() < 0.03, "mean {} not within 0.03 of 1", stats.mean);
    assert!(stats.max > 1.15, "max {} shows no focusing above 1.15", stats.max);
    assert!(stats.min < 0.9, "min {} shows no dimming below 0.9", stats.min);
}
