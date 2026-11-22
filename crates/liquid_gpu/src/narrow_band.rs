use crate::solver::{Bind, GpuSolver, Slots};

impl GpuSolver {
    pub(crate) fn encode_band_advection(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.dispatch(pass, "band_forward", Bind::Sim, Slots::FACES);
        self.dispatch(pass, "band_backward", Bind::Sim, Slots::FACES);
        self.dispatch_direct(pass, "band_liquid_phi", Bind::Sim, self.phi_groups());
    }

    pub fn encode_resample(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.dispatch(pass, "band_delete", Bind::Sim, Slots::PARTICLES);
        self.encode_reseed(pass);
    }

    pub(crate) fn encode_reseed(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.dispatch(pass, "band_count", Bind::Sim, Slots::GRID);
        self.dispatch(pass, "band_scan_blocks", Bind::Sim, Slots::SCAN);
        self.dispatch(pass, "band_scan_sums", Bind::Sim, Slots::SINGLE);
        self.dispatch(pass, "band_spawn", Bind::Sim, Slots::GRID);
        self.dispatch(pass, "band_seed", Bind::Sim, Slots::GRID);
        self.dispatch(pass, "band_grow", Bind::Sim, Slots::SINGLE);
    }
}
