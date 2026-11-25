#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Diffusion {
    Viscosity = 0,
    Capillary = 1,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section {
    Plan,
    Sort,
    Transfer,
    Geometry,
    Pressure,
    Advect,
    Density,
    Surface,
    Whitewater,
    Viscosity,
}

impl Section {
    pub const ALL: [Section; 10] = [
        Section::Plan,
        Section::Sort,
        Section::Transfer,
        Section::Geometry,
        Section::Pressure,
        Section::Advect,
        Section::Density,
        Section::Surface,
        Section::Whitewater,
        Section::Viscosity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Section::Plan => "liquid plan",
            Section::Sort => "liquid sort",
            Section::Transfer => "liquid transfer",
            Section::Geometry => "liquid geometry",
            Section::Pressure => "liquid pressure",
            Section::Advect => "liquid advect",
            Section::Density => "liquid density",
            Section::Surface => "liquid surface",
            Section::Whitewater => "liquid whitewater",
            Section::Viscosity => "liquid viscosity",
        }
    }

    pub(super) fn index(self) -> u32 {
        Section::ALL.iter().position(|s| *s == self).unwrap_or(0) as u32
    }
}

pub(crate) struct Passes<'e> {
    encoder: &'e mut wgpu::CommandEncoder,
    profiler: Option<&'e Profiler>,
    pass: Option<wgpu::ComputePass<'static>>,
}

impl<'e> Passes<'e> {
    pub(crate) fn new(encoder: &'e mut wgpu::CommandEncoder, profiler: Option<&'e Profiler>) -> Self {
        Self { encoder, profiler, pass: None }
    }

    pub(crate) fn current(&mut self) -> &mut wgpu::ComputePass<'static> {
        let encoder = &mut *self.encoder;

        self.pass.get_or_insert_with(|| {
            let descriptor = wgpu::ComputePassDescriptor { label: Some("liquid coupling"), timestamp_writes: None };

            encoder.begin_compute_pass(&descriptor).forget_lifetime()
        })
    }

    pub(crate) fn section(&mut self, section: Section, substep: u32) -> &mut wgpu::ComputePass<'static> {
        if self.profiler.is_some() {
            self.pass = None;
        }

        let (encoder, profiler) = (&mut *self.encoder, self.profiler);

        self.pass.get_or_insert_with(|| {
            let timestamp_writes = profiler.map(|p| p.writes(section, substep));
            let descriptor = wgpu::ComputePassDescriptor { label: Some(section.label()), timestamp_writes };

            encoder.begin_compute_pass(&descriptor).forget_lifetime()
        })
    }
}

pub struct Profiler {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    substeps: u32,
    period_ns: f32,
}

impl Profiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, substeps: u32) -> Self {
        let count = Section::ALL.len() as u32 * substeps * 2;
        let bytes = count as u64 * 8;

        Self {
            set: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("liquid timestamps"),
                ty: wgpu::QueryType::Timestamp,
                count,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquid timestamp resolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("liquid timestamp readback"),
                size: bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            substeps,
            period_ns: queue.get_timestamp_period(),
        }
    }

    pub(super) fn slot(&self, section: Section, substep: u32) -> u32 {
        (section.index() * self.substeps + substep) * 2
    }

    pub(super) fn writes(&self, section: Section, substep: u32) -> wgpu::ComputePassTimestampWrites<'_> {
        let slot = self.slot(section, substep);

        wgpu::ComputePassTimestampWrites {
            query_set: &self.set,
            beginning_of_pass_write_index: Some(slot),
            end_of_pass_write_index: Some(slot + 1),
        }
    }

    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        let count = Section::ALL.len() as u32 * self.substeps * 2;
        encoder.clear_buffer(&self.resolve, 0, None);
        encoder.resolve_query_set(&self.set, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.readback, 0, count as u64 * 8);
    }

    pub fn read_milliseconds(&self, device: &wgpu::Device) -> Vec<(Section, f32)> {
        self.readback.map_async(wgpu::MapMode::Read, .., |r| r.expect("timestamp map failed"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let stamps: Vec<u64> = {
            let view = self.readback.get_mapped_range(..).expect("timestamp range");
            bytemuck::cast_slice(&view).to_vec()
        };
        self.readback.unmap();

        self.milliseconds(&stamps, self.substeps)
    }

    pub fn readback(&self) -> &wgpu::Buffer {
        &self.readback
    }

    pub fn milliseconds(&self, stamps: &[u64], encoded: u32) -> Vec<(Section, f32)> {
        Section::ALL
            .iter()
            .map(|s| {
                let total: u64 = (0..encoded.clamp(1, self.substeps))
                    .map(|k| {
                        let i = self.slot(*s, k) as usize;
                        stamps[i + 1].saturating_sub(stamps[i])
                    })
                    .sum();

                (*s, total as f32 * self.period_ns * 1e-6)
            })
            .collect()
    }
}
