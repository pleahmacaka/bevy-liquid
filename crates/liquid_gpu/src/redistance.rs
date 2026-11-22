use crate::solver::{Bind, GpuSolver, Slots};

impl GpuSolver {
    pub fn encode_redistance(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.dispatch(pass, "redistance_seed", Bind::Sim, Slots::CELLS);
        self.encode_seeded_redistance(pass, "redistance_project");
    }

    pub(crate) fn encode_seeded_redistance(&self, pass: &mut wgpu::ComputePass<'_>, project: &str) {
        for k in 0..self.config.redistance_iterations.next_multiple_of(2) {
            self.dispatch(pass, &format!("redistance_sweep#{}", k % 2), Bind::Sim, Slots::CELLS);
        }

        self.dispatch(pass, "redistance_finish", Bind::Sim, Slots::CELLS);
        self.dispatch(pass, project, Bind::Sim, Slots::CELLS);
    }
}
