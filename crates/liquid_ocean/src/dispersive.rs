use bytemuck::{Pod, Zeroable};
use liquid_wave::dispersive::Dispersive;

use crate::fft::GpuFft;
use crate::kernel::{self, Binding, Kernel};
use crate::shaders;
use crate::swe::{GpuSwe, SharedBuffers};

pub const MAX_DEPTHS: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DispersiveParams {
    pub nx: u32,
    pub nz: u32,
    pub ex: u32,
    pub ez: u32,
    pub dx: f32,
    pub dz: f32,
    pub gravity: f32,
    pub epsilon: f32,
    pub dry_depth: f32,
    pub filter_width: f32,
    pub gradient_penalty: f32,
    pub damping: f32,
    pub depth_count: u32,
    pub pad: [u32; 3],
    pub depths: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct NodeParams {
    index: u32,
    pad: [u32; 3],
}

const BINDINGS: [Binding; 15] = [
    Binding::Uniform,
    Binding::Uniform,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::ReadOnly,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::ReadOnly,
    Binding::Storage,
];

const ENTRIES: [&str; 20] = [
    "decompose_pack",
    "decompose_filter",
    "decompose_begin",
    "decompose_accumulate",
    "decompose_finish",
    "airy_pack",
    "airy_spectra",
    "airy_begin",
    "airy_accumulate_cells",
    "airy_accumulate_faces",
    "transport_faces",
    "transport_amplify",
    "transport_advect",
    "combine_faces",
    "combine_limit",
    "combine_apply",
    "combine_record",
    "profile_flow",
    "profile_curvature",
    "profile_modes",
];

pub struct GpuDispersive {
    kernel: Kernel,
    pub bulk: GpuSwe,
    total_group: wgpu::BindGroup,
    nodes: Vec<wgpu::BindGroup>,
    spectrum_fft: GpuFft,
    work_fft: GpuFft,
    pub params: DispersiveParams,
    pub total: wgpu::Buffer,
    pub bulk_start: wgpu::Buffer,
    pub surface: wgpu::Buffer,
    cells: u32,
    faces: u32,
    ext: u32,
}

impl GpuDispersive {
    pub fn new(
        device: &wgpu::Device,
        model: &Dispersive,
        dt_max: f32,
        shared: SharedBuffers,
    ) -> Self {
        let swe = &model.swe;
        let grid = swe.grid;
        let depths = &model.params.reference_depths;

        assert!(
            depths.len() <= MAX_DEPTHS,
            "at most {MAX_DEPTHS} reference depths"
        );
        assert!(
            grid.nx.is_power_of_two() && grid.nz.is_power_of_two(),
            "dispersive grids need power-of-two sizes"
        );

        let kernel = Kernel::new(
            device,
            "dispersive",
            shaders::DISPERSIVE,
            &BINDINGS,
            &ENTRIES,
        );
        let cells = grid.cells() as u32;
        let faces = (grid.x_faces() + grid.z_faces()) as u32;
        let (ex, ez) = (2 * grid.nx as u32, 2 * grid.nz as u32);
        let ext = ex * ez;

        let mut padded = [0.0f32; 4];
        for (p, &d) in padded.iter_mut().zip(depths) {
            *p = d as f32;
        }

        let params = DispersiveParams {
            nx: grid.nx as u32,
            nz: grid.nz as u32,
            ex,
            ez,
            dx: grid.dx as f32,
            dz: grid.dz as f32,
            gravity: swe.params.gravity as f32,
            epsilon: swe.params.epsilon() as f32,
            dry_depth: swe.params.dry_depth as f32,
            filter_width: model.params.filter_width as f32,
            gradient_penalty: model.params.gradient_penalty as f32,
            damping: model.params.amplification_damping as f32,
            depth_count: depths.len() as u32,
            pad: [0; 3],
            depths: padded,
        };

        let bulk_state = kernel::zeroed_storage(device, "bulk state", 16 * cells as u64);
        let bulk = GpuSwe::new(
            device,
            swe,
            dt_max,
            SharedBuffers {
                state: bulk_state,
                bed: shared.bed.clone(),
            },
        );
        let total_group = bulk.bind_state(device, &shared.state);

        let uniform = kernel::uniform(device, "dispersive params", bytemuck::bytes_of(&params));
        let bulk_start = kernel::zeroed_storage(device, "bulk start", 16 * cells as u64);
        let surface = kernel::zeroed_storage(device, "surface", 16 * cells as u64);
        let spectrum = kernel::zeroed_storage(device, "dispersive spectrum", 16 * ext as u64);
        let work = kernel::zeroed_storage(device, "dispersive work", 16 * ext as u64);
        let cell_slots = kernel::zeroed_storage(device, "dispersive cells", 64 * cells as u64);
        let face_slots = kernel::zeroed_storage(device, "dispersive faces", 12 * faces as u64);
        let limits = kernel::zeroed_storage(device, "dispersive limits", 4 * cells as u64);

        let nodes = (0..=depths.len() as u32)
            .map(|index| {
                let node = kernel::uniform(
                    device,
                    "dispersive node",
                    bytemuck::bytes_of(&NodeParams { index, pad: [0; 3] }),
                );
                kernel.bind(
                    device,
                    "dispersive",
                    &[
                        &uniform,
                        &node,
                        &shared.state,
                        &bulk.buffers.state,
                        &bulk_start,
                        &surface,
                        &shared.bed,
                        &spectrum,
                        &work,
                        &cell_slots,
                        &face_slots,
                        &limits,
                        &bulk.buffers.accumulated,
                        &bulk.buffers.clock,
                        &bulk.buffers.profile,
                    ],
                )
            })
            .collect();

        let spectrum_fft = GpuFft::new(device, &spectrum, ex, ez, 2);
        let work_fft = GpuFft::new(device, &work, ex, ez, 2);

        Self {
            kernel,
            bulk,
            total_group,
            nodes,
            spectrum_fft,
            work_fft,
            params,
            total: shared.state,
            bulk_start,
            surface,
            cells,
            faces,
            ext,
        }
    }

    pub fn write_params(&mut self, queue: &wgpu::Queue, frame_time: f32) {
        self.bulk.write_params(queue, frame_time);
    }

    pub fn encode_frame(&self, encoder: &mut wgpu::CommandEncoder, substeps: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("dispersive frame"),
            timestamp_writes: None,
        });

        self.bulk.encode_begin(&mut pass, &self.total_group);

        for _ in 0..substeps {
            self.encode_step(&mut pass);
        }

        self.bulk.encode_finish(&mut pass, &self.total_group);
        self.encode_profile(&mut pass);
    }

    fn encode_profile(&self, pass: &mut wgpu::ComputePass) {
        let k = &self.kernel;
        let base = &self.nodes[0];

        self.encode_decompose(pass);
        k.dispatch(pass, "profile_flow", base, self.cells);
        k.dispatch(pass, "profile_curvature", base, self.cells);
        k.dispatch(pass, "profile_modes", base, self.cells);
    }

    fn encode_decompose(&self, pass: &mut wgpu::ComputePass) {
        let k = &self.kernel;
        let base = &self.nodes[0];
        let depths = self.params.depth_count as usize;

        k.dispatch(pass, "decompose_pack", base, self.ext);
        self.spectrum_fft.encode(pass, false);
        k.dispatch(pass, "decompose_begin", base, self.cells);

        for node in &self.nodes[1..=depths] {
            k.dispatch(pass, "decompose_filter", node, self.ext);
            self.work_fft.encode(pass, true);
            k.dispatch(pass, "decompose_accumulate", node, self.cells);
        }

        k.dispatch(pass, "decompose_finish", base, self.cells);
    }

    fn encode_step(&self, pass: &mut wgpu::ComputePass) {
        let k = &self.kernel;
        let base = &self.nodes[0];
        let depths = self.params.depth_count as usize;

        self.encode_decompose(pass);
        self.bulk.encode_clear_accumulated(pass, self.bulk.group());
        self.bulk.encode_substep(pass, self.bulk.group(), true);

        k.dispatch(pass, "airy_pack", base, self.ext);
        self.spectrum_fft.encode(pass, false);
        k.dispatch(pass, "airy_begin", base, self.cells.max(self.faces));

        for node in &self.nodes[..depths] {
            k.dispatch(pass, "airy_spectra", node, self.ext);
            self.work_fft.encode(pass, true);
            k.dispatch(pass, "airy_accumulate_cells", node, self.cells);
            k.dispatch(pass, "airy_accumulate_faces", node, self.faces);
        }

        k.dispatch(pass, "transport_faces", base, self.faces);
        k.dispatch(pass, "transport_amplify", base, self.cells);
        k.dispatch(pass, "transport_advect", base, self.cells);

        k.dispatch(pass, "combine_faces", base, self.faces);
        k.dispatch(pass, "combine_limit", base, self.cells);
        k.dispatch(pass, "combine_apply", base, self.cells);
        k.dispatch(pass, "combine_record", base, self.faces);
        self.bulk.encode_interface_accumulated(pass);
    }
}
