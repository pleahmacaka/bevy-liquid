use wgpu::util::DeviceExt;

pub const WORKGROUP: u32 = 64;
const MAX_GROUPS: u32 = 65_535;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binding {
    Uniform,
    Storage,
    ReadOnly,
}

pub struct Kernel {
    layout: wgpu::BindGroupLayout,
    pipelines: Vec<(&'static str, wgpu::ComputePipeline)>,
}

impl Kernel {
    pub fn new(
        device: &wgpu::Device,
        label: &str,
        source: &str,
        bindings: &[Binding],
        entries: &[&'static str],
    ) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        let layout_entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .enumerate()
            .map(|(index, binding)| wgpu::BindGroupLayoutEntry {
                binding: index as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: match binding {
                        Binding::Uniform => wgpu::BufferBindingType::Uniform,
                        Binding::Storage => wgpu::BufferBindingType::Storage { read_only: false },
                        Binding::ReadOnly => wgpu::BufferBindingType::Storage { read_only: true },
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &layout_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipelines = entries
            .iter()
            .map(|&entry| {
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some(entry),
                    // every kernel writes its workgroup memory first; FXC takes over a minute to compile the FFT zero fill
                    compilation_options: wgpu::PipelineCompilationOptions {
                        zero_initialize_workgroup_memory: false,
                        ..Default::default()
                    },
                    cache: None,
                });

                (entry, pipeline)
            })
            .collect();

        Self { layout, pipelines }
    }

    pub fn pipeline(&self, entry: &str) -> &wgpu::ComputePipeline {
        self.pipelines
            .iter()
            .find(|(name, _)| *name == entry)
            .map(|(_, pipeline)| pipeline)
            .unwrap_or_else(|| panic!("kernel has no entry point {entry}"))
    }

    pub fn bind(
        &self,
        device: &wgpu::Device,
        label: &str,
        buffers: &[&wgpu::Buffer],
    ) -> wgpu::BindGroup {
        let entries: Vec<wgpu::BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(index, buffer)| wgpu::BindGroupEntry {
                binding: index as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &self.layout,
            entries: &entries,
        })
    }

    pub fn dispatch(
        &self,
        pass: &mut wgpu::ComputePass,
        entry: &str,
        group: &wgpu::BindGroup,
        threads: u32,
    ) {
        let groups = threads.div_ceil(WORKGROUP).max(1);

        pass.set_pipeline(self.pipeline(entry));
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups.min(MAX_GROUPS), groups.div_ceil(MAX_GROUPS), 1);
    }

    pub fn dispatch_grid(
        &self,
        pass: &mut wgpu::ComputePass,
        entry: &str,
        group: &wgpu::BindGroup,
        groups: [u32; 2],
    ) {
        pass.set_pipeline(self.pipeline(entry));
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(groups[0], groups[1], 1);
    }
}

pub const STORAGE_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

pub fn storage(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: STORAGE_USAGE,
    })
}

pub fn zeroed_storage(device: &wgpu::Device, label: &str, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: STORAGE_USAGE,
        mapped_at_creation: false,
    })
}

pub fn uniform(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

pub fn vec4_bytes(values: &[[f32; 4]]) -> &[u8] {
    bytemuck::cast_slice(values)
}

pub fn read_pod<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> T {
    let size = std::mem::size_of::<T>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    staging.map_async(wgpu::MapMode::Read, .., |result| result.expect("map failed"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll failed");

    let value = bytemuck::pod_read_unaligned(&staging.slice(..).get_mapped_range().expect("mapped range"));
    staging.unmap();

    value
}
