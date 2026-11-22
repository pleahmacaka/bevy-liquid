use crate::solver::{Bind, GpuSolver, Slots};

const BLUR_PASSES: [u32; 8] = [0, 1, 2, 1, 2, 1, 2, 1];

impl GpuSolver {
    pub fn encode_curvature(&self, pass: &mut wgpu::ComputePass<'_>) {
        for b in BLUR_PASSES {
            self.dispatch(pass, &format!("blur#{b}"), Bind::Sim, Slots::CELLS);
        }

        self.dispatch(pass, "curvature", Bind::Sim, Slots::CELLS);
    }
}
