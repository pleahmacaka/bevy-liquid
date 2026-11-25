use crate::solver::{Bind, Diffusion, GpuSolver, Slots};

impl GpuSolver {
    pub fn encode_diffuse(&self, pass: &mut wgpu::ComputePass<'_>, kind: Diffusion) {
        self.encode_diffusion(pass, &[kind]);
    }

    pub fn encode_diffusion(&self, pass: &mut wgpu::ComputePass<'_>, kinds: &[Diffusion]) {
        let Some(&kind) = kinds.last() else {
            return;
        };
        let k = kind as u32;
        let faces = self.slots.solve_faces();
        let blocks = self.slots.solve_viscous();

        for (i, term) in kinds.iter().enumerate() {
            let t = *term as u32;
            let mode = match (kinds.len(), i) {
                (1, _) => "",
                (_, 0) => "#only",
                _ => "#add",
            };

            self.dispatch(pass, &format!("viscous_samples#{t}"), Bind::Sim, Slots::CELLS);
            self.dispatch(pass, &format!("viscous_edge_fractions#{t}"), Bind::Sim, Slots::NODES);
            self.dispatch(pass, &format!("viscous_couplings#{t}{mode}"), Bind::Sim, Slots::FACES);
        }

        self.dispatch_direct(pass, &format!("viscous_open#{k}"), Bind::Plan, 1);
        self.dispatch(pass, "viscous_residual", Bind::Sim, faces);
        self.dispatch(pass, "viscous_block#lead", Bind::Sim, blocks);
        self.dispatch_direct(pass, &format!("viscous_check#lead#{k}"), Bind::Plan, 1);

        for i in 0..self.config.viscosity_iterations {
            let swap = if i % 2 == 1 { "#swap" } else { "" };
            let first = if i == 0 { "#first" } else { swap };

            self.dispatch(pass, &format!("viscous_apply{first}"), Bind::Sim, faces);
            self.dispatch(pass, &format!("viscous_block{swap}"), Bind::Sim, blocks);
            self.dispatch_direct(pass, &format!("viscous_check#{k}"), Bind::Plan, 1);
        }

        self.dispatch(pass, &format!("viscous_finish#{k}"), Bind::Sim, Slots::FACES);

        if !self.extrapolated(pass, self.config.extrapolation_layers, "extrapolate_back#constrain") {
            self.dispatch(pass, "constrain_solid", Bind::Sim, Slots::FACES);
        }
    }
}
