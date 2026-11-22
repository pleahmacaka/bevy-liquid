pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub fn gpu() -> Gpu {
    open(wgpu::InstanceDescriptor::new_without_display_handle_from_env())
}

#[cfg(windows)]
pub fn assert_compiles_under_fxc(build: impl FnOnce(&Gpu)) {
    use std::sync::{Arc, Mutex};

    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    descriptor.backends = wgpu::Backends::DX12;
    descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::Fxc;

    let gpu = open(descriptor);
    let errors = Arc::new(Mutex::new(Vec::new()));
    let sink = errors.clone();
    gpu.device.on_uncaptured_error(Arc::new(move |e| {
        sink.lock().expect("error sink").push(e.to_string())
    }));

    build(&gpu);

    let errors = errors.lock().expect("error sink");

    assert!(
        errors.is_empty(),
        "{} pipelines failed under FXC:\n{}",
        errors.len(),
        errors.join("\n")
    );
}

fn open(descriptor: wgpu::InstanceDescriptor) -> Gpu {
    static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();
    let instance = INSTANCE.get_or_init(|| wgpu::Instance::new(descriptor));
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env(instance, None))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
        })
    .expect("a GPU adapter is required for these tests");

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("liquid_ocean tests"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("device request failed");

    Gpu { device, queue }
}

impl Gpu {
    pub fn submit(&self, encode: impl FnOnce(&mut wgpu::CommandEncoder)) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encode(&mut encoder);
        self.queue.submit([encoder.finish()]);
    }

    pub fn read<T: bytemuck::Pod>(&self, buffer: &wgpu::Buffer) -> Vec<T> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.submit(|e| e.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size()));

        staging.map_async(wgpu::MapMode::Read, .., |result| {
            result.expect("map failed")
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("poll failed");

        let data =
            bytemuck::cast_slice(&staging.slice(..).get_mapped_range().expect("mapped range"))
                .to_vec();
        staging.unmap();

        data
    }
}

pub fn relative_error(gpu: &[f64], cpu: &[f64]) -> f64 {
    let scale = cpu.iter().map(|v| v.abs()).fold(0.0, f64::max).max(1e-30);
    let worst = gpu
        .iter()
        .zip(cpu)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    worst / scale
}
