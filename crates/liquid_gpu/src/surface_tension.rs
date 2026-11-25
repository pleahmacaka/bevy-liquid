use crate::solver::{Bind, GpuSolver, Slots};

const BLUR_PASSES: [u32; 8] = [0, 1, 2, 1, 2, 1, 2, 1];

const JUMP_SMOOTHING_PASSES: [u32; 2] = [0, 1];

impl GpuSolver {
    pub fn encode_curvature(&self, pass: &mut wgpu::ComputePass<'_>) {
        for b in BLUR_PASSES {
            self.dispatch(pass, &format!("blur#{b}"), Bind::Sim, Slots::CELLS);
        }

        self.dispatch(pass, "curvature", Bind::Sim, Slots::CELLS);
    }

    pub fn encode_jump(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.dispatch(pass, "jump_seed", Bind::Level(0), Slots::CELLS);

        for s in JUMP_SMOOTHING_PASSES {
            self.dispatch(pass, &format!("jump_smooth#{s}"), Bind::Level(0), Slots::CELLS);
        }

        // jump_gradient reads sweep_a after the pressure solve; nothing in between may write it
        self.dispatch(pass, "jump_rhs", Bind::Level(0), Slots::CELLS);
    }
}
