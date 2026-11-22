use std::sync::{Arc, Mutex};

pub fn assert_compiles_under_fxc(build: impl FnOnce(&wgpu::Device, &wgpu::Queue)) {
    let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
    descriptor.backends = wgpu::Backends::DX12;
    descriptor.backend_options.dx12.shader_compiler = wgpu::Dx12Compiler::Fxc;

    static INSTANCE: std::sync::OnceLock<wgpu::Instance> = std::sync::OnceLock::new();
    let instance = INSTANCE.get_or_init(|| wgpu::Instance::new(descriptor));
    let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env(instance, None))
        .or_else(|_| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }))
        })
        .expect("no DX12 adapter");

    println!("adapter: {:?}", adapter.get_info().name);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("liquid fxc test"),
        required_limits: adapter.limits(),
        ..Default::default()
    }))
    .expect("no device");

    let errors = Arc::new(Mutex::new(Vec::new()));
    let sink = errors.clone();
    device.on_uncaptured_error(Arc::new(move |e| sink.lock().expect("error sink").push(e.to_string())));

    build(&device, &queue);

    let errors = errors.lock().expect("error sink");

    assert!(errors.is_empty(), "{} pipelines failed under FXC:\n{}", errors.len(), errors.join("\n"));
}
