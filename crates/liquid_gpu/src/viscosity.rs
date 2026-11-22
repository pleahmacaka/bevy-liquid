use crate::solver::{Bind, Diffusion, GpuSolver, Slots};

impl GpuSolver {
    fn viscous_reduce(&self, pass: &mut wgpu::ComputePass<'_>, pair: u32) {
        self.dispatch(pass, &format!("viscous_reduce_partial#{pair}"), Bind::Sim, self.slots.solve_faces());
        self.dispatch(pass, &format!("viscous_reduce_final#{pair}"), Bind::Sim, self.slots.solve_single());
    }

    pub fn encode_diffuse(&self, pass: &mut wgpu::ComputePass<'_>, kind: Diffusion) {
        let k = kind as u32;
        let faces = self.slots.solve_faces();
        let single = self.slots.solve_single();

        self.dispatch(pass, &format!("viscous_scale#{k}"), Bind::Sim, Slots::SINGLE);
        self.dispatch(pass, &format!("viscous_cell_fractions#{k}"), Bind::Sim, Slots::CELLS);
        self.dispatch(pass, &format!("viscous_edge_fractions#{k}"), Bind::Sim, Slots::NODES);
        self.dispatch(pass, &format!("viscous_face_fractions#{k}"), Bind::Sim, Slots::FACES);
        self.dispatch(pass, "viscous_diagonal", Bind::Sim, Slots::FACES);
        self.dispatch(pass, "viscous_load", Bind::Sim, Slots::FACES);
        self.dispatch_direct(pass, &format!("viscous_open#{k}"), Bind::Plan, 1);
        self.dispatch(pass, "viscous_residual", Bind::Sim, faces);
        self.viscous_reduce(pass, 3);
        self.viscous_reduce(pass, 2);
        self.dispatch_direct(pass, &format!("viscous_check#{k}"), Bind::Plan, 1);
        self.viscous_reduce(pass, 0);
        self.dispatch(pass, "pcg_start", Bind::Sim, single);
        self.dispatch(pass, "viscous_direction#1", Bind::Sim, faces);

        for _ in 0..self.config.viscosity_iterations {
            self.dispatch(pass, "viscous_apply", Bind::Sim, faces);
            self.viscous_reduce(pass, 1);
            self.dispatch(pass, "pcg_alpha", Bind::Sim, single);
            self.dispatch(pass, "viscous_update", Bind::Sim, faces);
            self.viscous_reduce(pass, 2);
            self.dispatch_direct(pass, &format!("viscous_check#{k}"), Bind::Plan, 1);
            self.viscous_reduce(pass, 0);
            self.dispatch(pass, "pcg_beta", Bind::Sim, single);
            self.dispatch(pass, "viscous_direction#0", Bind::Sim, faces);
        }

        self.dispatch(pass, "viscous_finish", Bind::Sim, Slots::FACES);
        self.encode_extrapolate(pass, self.config.extrapolation_layers);
        self.dispatch(pass, "constrain_solid", Bind::Sim, Slots::FACES);
    }
}
