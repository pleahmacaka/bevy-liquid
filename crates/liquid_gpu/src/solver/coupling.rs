use super::*;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CouplingParams {
    pub offset: [f32; 4],
    pub grid: [u32; 4],
    pub cell: [f32; 4],
}

pub struct HeightfieldLink<'a> {
    pub offset: Vec3,
    pub cells: [u32; 2],
    pub cell_size: [f32; 2],
    pub relax_cells: f32,
    pub blend: f32,
    pub state: &'a wgpu::Buffer,
    pub bed: &'a wgpu::Buffer,
    pub coupled: &'a wgpu::Buffer,
}

pub struct Coupling {
    pub(crate) bind: wgpu::BindGroup,
    pub(crate) params: wgpu::Buffer,
    pub(crate) window: [u32; 2],
    pub(crate) sources: [wgpu::Buffer; 3],
    pub(crate) tiers: bool,
}

impl GpuSolver {
    pub fn attach_heightfield(&mut self, device: &wgpu::Device, link: HeightfieldLink<'_>) {
        let params = CouplingParams {
            offset: [link.offset.x, link.offset.y, link.offset.z, link.relax_cells],
            grid: [link.cells[0], link.cells[1], 0, 0],
            cell: [link.cell_size[0], link.cell_size[1], link.blend, 1.0],
        };
        let buffer = uniform(device, "liquid coupling params", &params);
        let sources = crate::coupling::CouplingSources { params: &buffer, state: link.state, bed: link.bed, coupled: link.coupled };
        let bind = crate::coupling::coupling_bind(device, &self.coupling_layout, sources, None);
        let sources = [link.state.clone(), link.bed.clone(), link.coupled.clone()];

        let size = self.config.dims.as_vec3() * self.config.dx;
        let window = [0, 1].map(|a| (size[2 * a] / link.cell_size[a]).ceil() as u32 + 3);

        self.coupling = Some(Coupling { bind, params: buffer, window, sources, tiers: false });
    }

    pub fn update_heightfield_offset(&self, queue: &wgpu::Queue, offset: Vec3) {
        if let Some(c) = &self.coupling {
            queue.write_buffer(&c.params, 0, bytemuck::cast_slice(&[offset.x, offset.y, offset.z]));
        }
    }

    pub fn is_coupled(&self) -> bool {
        self.coupling.is_some()
    }

    pub(super) fn encode_boundary(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.is_coupled() {
            self.dispatch(pass, "couple_boundary", Bind::Coupling, Slots::FACES);
        }
    }

    pub(super) fn encode_columns(&self, pass: &mut wgpu::ComputePass<'_>) {
        if let Some(c) = &self.coupling {
            self.prepare(pass, "couple_columns", Bind::Coupling);
            pass.dispatch_workgroups(c.window[0], c.window[1], 1);
        }
    }
}
