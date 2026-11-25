use super::*;

impl GpuSolver {
    pub fn dispatch(&self, pass: &mut wgpu::ComputePass<'_>, key: &str, bind: Bind, slot: u32) {
        self.prepare(pass, key, bind);
        pass.dispatch_workgroups_indirect(&self.buffers.indirect, slot as u64 * 12);
    }

    pub fn dispatch_direct(&self, pass: &mut wgpu::ComputePass<'_>, key: &str, bind: Bind, groups: u32) {
        self.prepare(pass, key, bind);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    pub fn encode_frame(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>) {
        self.encode_frame_within(encoder, profiler, self.config.max_substeps);
    }

    pub fn encode_frame_within(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>, budget: u32) {
        self.encode_frame_with(encoder, profiler, budget, None);
    }

    pub(crate) fn encode_frame_with(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>, budget: u32, sdf: Option<&SurfaceSdf>) {
        let late = self.config.late_surface;
        let mut passes = Passes::new(encoder, profiler);
        let pass = passes.section(Section::Plan, 0);
        self.dispatch_direct(pass, "begin_frame", Bind::Plan, 1);

        for substep in 0..budget.clamp(1, self.config.max_substeps) {
            let pass = passes.section(Section::Sort, substep);

            if substep > 0 {
                self.dispatch(pass, "reset_speed", Bind::Sim, Slots::SINGLE);
            }

            if self.is_coupled() {
                self.dispatch(pass, "measure_inflow", Bind::Coupling, Slots::PARTICLES);
                self.dispatch_direct(pass, "plan_substep#grow", Bind::Plan, 1);
            } else {
                self.dispatch(pass, "measure_flow", Bind::Sim, Slots::PARTICLES);
                self.dispatch_direct(pass, "plan_substep", Bind::Plan, 1);
            }

            if late && substep == 0 {
                self.encode_sorted(pass, "sort_copy#surface", true);
                self.encode_surface(&mut passes, sdf);
                self.encode_whitewater_into(&mut passes);
            } else {
                self.encode_sorted(pass, "sort_copy", true);
            }

            self.encode_substep_into(&mut passes, substep);
        }

        if late {
            let pass = passes.current();

            if self.is_coupled() && self.config.has_band() {
                self.dispatch_direct(pass, "adopt_band", Bind::Sim, groups(self.params.dims[3], WG).min(MAX_GROUPS));
            }

            self.encode_columns(pass);
            return;
        }

        let pass = passes.section(Section::Surface, 0);
        self.dispatch_direct(pass, "open_all", Bind::Plan, 1);
        self.encode_sorted(pass, "sort_copy#surface", false);

        if !self.config.has_band() {
            self.dispatch(pass, "liquid_phi", Bind::Sim, Slots::GRID);
        }

        self.encode_surface_field(pass);
        self.encode_columns(pass);
        self.encode_whitewater_into(&mut passes);
    }

    pub fn encode_sort(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.encode_sorted(pass, "sort_copy", false);
    }

    pub(super) fn encode_surface(&self, passes: &mut Passes<'_>, sdf: Option<&SurfaceSdf>) {
        let pass = passes.section(Section::Surface, 0);

        if !self.config.has_band() {
            self.dispatch(pass, "liquid_phi", Bind::Sim, Slots::GRID);
        }

        self.encode_surface_field(pass);

        if let Some(sdf) = sdf {
            sdf.record(pass);
        }
    }

    pub(crate) fn encode_sorted(&self, pass: &mut wgpu::ComputePass<'_>, copy: &str, emit: bool) {
        self.dispatch(pass, "sort_count", Bind::Sim, Slots::PARTICLES);

        if emit {
            self.encode_emission(pass);
        }

        self.dispatch_direct(pass, "sort_scan", Bind::Plan, groups(self.params.dims[3], SCAN_BLOCK));
        self.dispatch_direct(pass, "sort_link", Bind::LevelPlan(0), groups(27 * self.brick_count(), WG).max(64));
        self.dispatch(pass, "scatter_particles", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, "rank_particles", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, copy, Bind::Sim, Slots::PARTICLES);
    }

    pub(crate) fn phi_groups(&self) -> u32 {
        let cells = if self.config.sparse { self.params.pool[0] * BRICK_CELLS } else { self.params.dims[3] };

        groups(cells, 28).min(MAX_GROUPS)
    }

    pub(super) fn p2g_groups(&self) -> u32 {
        let nodes = if self.config.sparse { self.params.pool[0] * BRICK_NODES } else { self.params.node_dims[3] };

        groups(16 * nodes, WG).min(MAX_GROUPS)
    }

    pub fn encode_extrapolate(&self, pass: &mut wgpu::ComputePass<'_>, layers: u32) {
        for _ in 0..layers {
            self.dispatch(pass, "extrapolate_pass", Bind::Sim, Slots::FACES);
            self.dispatch(pass, "copy_scratch", Bind::Sim, Slots::FACES);
        }
    }

    pub(crate) fn extrapolated(&self, pass: &mut wgpu::ComputePass<'_>, layers: u32, last: &str) -> bool {
        if layers == 0 || layers % 2 == 1 {
            self.encode_extrapolate(pass, layers);
            return false;
        }

        for k in 0..layers - 1 {
            self.dispatch(pass, if k % 2 == 0 { "extrapolate_pass" } else { "extrapolate_back" }, Bind::Sim, Slots::FACES);
        }

        self.dispatch(pass, last, Bind::Sim, Slots::FACES);

        true
    }

    pub fn encode_geometry(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.encode_phi(pass);
        self.dispatch(pass, "face_weights", Bind::Sim, Slots::FACES);
        self.encode_boundary(pass);
    }

    pub(super) fn encode_phi(&self, pass: &mut wgpu::ComputePass<'_>) {
        let solid_nodes = if self.config.static_solid { "solid_nodes_static" } else { "solid_nodes" };

        if self.config.has_band() {
            self.dispatch_direct(pass, "band_liquid_phi", Bind::Sim, self.phi_groups());
            self.dispatch(pass, solid_nodes, Bind::Sim, Slots::NODES);
            self.encode_redistance(pass);
            self.dispatch(pass, "band_adopt", Bind::Sim, Slots::CELLS);
        } else {
            self.dispatch(pass, "liquid_phi", Bind::Sim, Slots::GRID);
            self.dispatch(pass, solid_nodes, Bind::Sim, Slots::NODES);
            self.dispatch(pass, "mark_solid_cells", Bind::Sim, Slots::CELLS);
        }

        self.dispatch(pass, "extend_into_solid", Bind::Sim, Slots::CELLS);
        self.dispatch(pass, "extend_from_scratch", Bind::Sim, Slots::CELLS);
        self.dispatch(pass, "extend_into_solid", Bind::Sim, Slots::CELLS);
        self.dispatch(pass, "settle_from_scratch", Bind::Sim, Slots::CELLS);
    }

    pub fn encode_hierarchy(&self, pass: &mut wgpu::ComputePass<'_>) {
        let band = self.config.smooths_boundary();

        for l in 0..self.levels.len() {
            let slot = self.slots.level(l);

            if l == 0 {
                self.dispatch(pass, "level_from_fine", Bind::Level(0), slot);
            } else {
                self.dispatch(pass, "coarsen", Bind::Pair(l - 1), slot);
            }

            self.dispatch(pass, "level_classify", Bind::Level(l), slot);
            self.dispatch(pass, "level_assemble", Bind::Level(l), slot);

            if band {
                self.dispatch(pass, "level_band", Bind::Level(l), slot);
            }
        }

        if self.config.open_floor.is_some() {
            let columns = groups(self.config.dims.x * self.config.dims.z, WG);
            self.dispatch_direct(pass, "floor_surface", Bind::Level(0), columns);
            self.dispatch_direct(pass, "floor_mean", Bind::Level(0), 1);
            self.dispatch_direct(pass, "floor_ghost", Bind::Level(0), columns);
        }

        if self.levels.len() > 1 {
            for l in self.tail..self.levels.len() {
                self.dispatch(pass, "tail_pack", Bind::Level(l), self.slots.level(l));
            }
        }
    }


    pub(super) fn smooth(&self, pass: &mut wgpu::ComputePass<'_>, l: usize, colors: [u32; 2], lines: bool) {
        if lines && l < self.tall_levels() {
            self.relax_lines(pass, l, colors);
        } else {
            self.relax(pass, "mg_smooth", l, colors);
        }
    }

    pub(super) fn relax(&self, pass: &mut wgpu::ComputePass<'_>, kind: &str, l: usize, colors: [u32; 2]) {
        let bind = Bind::Multigrid(l);
        let slot = self.slots.solve_level(l);

        for color in colors {
            self.dispatch(pass, &self.variant(&format!("{kind}#{color}"), l), bind, slot);
        }
    }

    pub(super) fn variant(&self, key: &str, l: usize) -> String {
        if self.has_tall() && l >= self.tall_levels() { format!("{key}~plain") } else { key.to_string() }
    }

    pub(super) fn boundary_sweeps(&self) -> u32 {
        if self.config.smooths_boundary() { self.config.boundary_sweeps } else { 0 }
    }

    pub(super) fn plain(&self, l: usize) -> bool {
        self.config.smoothing > 0 && !(self.has_tall() && l < self.tall_levels())
    }

    pub fn encode_vcycle(&self, pass: &mut wgpu::ComputePass<'_>, lines: bool) {
        let last = self.levels.len() - 1;

        if last == 0 || !self.plain(0) {
            self.dispatch(pass, "mg_zero", Bind::Multigrid(0), self.slots.solve_level(0));
        }

        if last == 0 {
            for _ in 0..self.config.coarse_sweeps {
                self.smooth(pass, last, [0, 1], lines);
            }

            for _ in 0..self.config.coarse_sweeps {
                self.smooth(pass, last, [1, 0], lines);
            }

            return;
        }

        for l in 0..self.tail {
            for s in 0..self.config.smoothing {
                if s == 0 && self.plain(l) {
                    self.dispatch(pass, &self.variant("mg_presmooth", l), Bind::Multigrid(l), self.slots.solve_level(l));
                } else {
                    self.smooth(pass, l, [0, 1], lines);
                }
            }

            for _ in 0..self.boundary_sweeps() {
                self.relax(pass, "mg_band", l, [0, 1]);
            }

            self.dispatch(pass, &self.variant("mg_residual", l), Bind::Multigrid(l), self.slots.solve_level(l));
            self.dispatch(pass, &self.variant("mg_restrict", l + 1), Bind::Pair(l), self.slots.solve_level(l + 1));
            self.gather_tall(pass, l + 1);
        }

        self.dispatch(pass, "mg_tail", Bind::Level(self.tail), self.slots.solve_level(last));

        for l in (0..self.tail).rev() {
            let correct = self.plain(l) && self.boundary_sweeps() == 0;
            let entry = if correct { "mg_correct" } else { "mg_prolong" };
            self.dispatch(pass, &self.variant(entry, l), Bind::Pair(l), self.slots.solve_level(l));

            for _ in 0..self.boundary_sweeps() {
                self.relax(pass, "mg_band", l, [1, 0]);
            }

            for s in 0..self.config.smoothing {
                let black = !(s == 0 && correct);
                let finish = l == 0 && s + 1 == self.config.smoothing && self.fuses_dot(lines);

                if black && !finish {
                    self.smooth(pass, l, [1, 0], lines);
                    continue;
                }

                if black {
                    self.dispatch(pass, &self.variant("mg_smooth#1", l), Bind::Multigrid(l), self.slots.solve_level(l));
                }

                if finish {
                    self.dispatch(pass, "mg_finish", Bind::Multigrid(0), self.slots.solve_grid());
                } else {
                    self.dispatch(pass, &self.variant("mg_smooth#0", l), Bind::Multigrid(l), self.slots.solve_level(l));
                }
            }
        }
    }

    pub(super) fn fuses_dot(&self, lines: bool) -> bool {
        self.config.smoothing > 0 && self.levels.len() > 1 && !(lines && self.has_tall())
    }

    pub(super) fn encode_precondition(&self, pass: &mut wgpu::ComputePass<'_>, lines: bool) {
        if self.blocks() {
            self.encode_blocks(pass, None, false);
            return;
        }

        self.encode_vcycle(pass, lines);

        if !self.fuses_dot(lines) {
            self.dispatch(pass, "pcg_dot", Bind::Level(0), self.slots.solve_grid());
        }
    }

    pub fn blocks(&self) -> bool {
        self.config.block_smoothing && self.levels.len() > 1 && !self.has_tall() && !self.config.smooths_boundary()
    }

    pub(super) fn encode_blocks(&self, pass: &mut wgpu::ComputePass<'_>, update: Option<(bool, bool)>, check: bool) {
        let down = match update {
            Some((swap, coupled)) => format!("mg_block_down#update{}{}", if swap { "#swap" } else { "" }, if coupled { "#coupled" } else { "" }),
            None => "mg_block_down".to_string(),
        };
        let last = self.levels.len() - 1;
        self.dispatch(pass, &down, Bind::Multigrid(0), self.slots.solve_block(0));

        if check {
            self.dispatch_direct(pass, "pcg_check", Bind::LevelPlan(0), 1);
        }

        self.dispatch(pass, "mg_block_complete", Bind::Multigrid(0), self.slots.solve_level(0));

        for l in 1..self.tail {
            self.dispatch(pass, "mg_block_coarse", Bind::Pair(l - 1), self.slots.solve_block(l));
            self.dispatch(pass, "mg_block_complete", Bind::Multigrid(l), self.slots.solve_level(l));
        }

        self.dispatch(pass, "mg_block_restrict", Bind::Pair(self.tail - 1), self.slots.solve_block(self.tail));
        self.dispatch(pass, "mg_tail", Bind::Level(self.tail), self.slots.solve_level(last));

        for l in (1..self.tail).rev() {
            self.dispatch(pass, "mg_block_up", Bind::Pair(l), self.slots.solve_block(l));
        }

        self.dispatch(pass, "mg_block_finish", Bind::Pair(0), self.slots.solve_block(0));
    }

    pub fn encode_solve(&self, pass: &mut wgpu::ComputePass<'_>, coupling: bool, iterations: u32, lines: bool) {
        let cells = self.slots.solve_level(0);
        let grid = self.slots.solve_grid();
        let tiles = self.slots.solve_bodies();
        let coupled = |key: &str| if coupling { format!("{key}#coupled") } else { key.to_string() };

        if coupling {
            self.dispatch(pass, "body_mark", Bind::Level(0), Slots::CELLS);
        }

        // wgpu skips rebinding group 3 when group 1 becomes a layout hole, so plan dispatches here keep group 1 bound
        self.dispatch_direct(pass, &coupled("pcg_open"), Bind::LevelPlan(0), 1);
        self.dispatch(pass, &coupled("pcg_init_residual"), Bind::Level(0), cells);

        if coupling {
            self.dispatch(pass, "body_gather#0", Bind::Level(0), tiles);
            self.dispatch_direct(pass, "pcg_scalar#lead#coupled", Bind::Level(0), 1);
            self.dispatch(pass, "pcg_update#lead#coupled", Bind::Level(0), cells);
        }

        let checked = self.config.tolerance > 0.0;

        if checked {
            self.dispatch_direct(pass, "pcg_check#lead", Bind::LevelPlan(0), 1);
        }

        if iterations > 0 {
            self.encode_precondition(pass, lines);
        }

        for i in 0..iterations {
            let bind = if i % 2 == 0 { Bind::Level(0) } else { Bind::Swapped(0) };
            let last = i + 1 == iterations;

            self.dispatch(pass, if i == 0 { "pcg_apply#lead" } else { "pcg_apply" }, bind, grid);

            if coupling {
                self.dispatch(pass, "body_gather#1", bind, tiles);
            }

            if self.blocks() && !last {
                self.encode_blocks(pass, Some((i % 2 == 1, coupling)), checked);
                continue;
            }

            self.dispatch_direct(pass, &coupled("pcg_scalar"), bind, 1);
            self.dispatch(pass, &coupled("pcg_update"), bind, cells);

            if checked {
                self.dispatch_direct(pass, "pcg_check", Bind::LevelPlan(0), 1);
            }

            if !last {
                self.encode_precondition(pass, lines);
            }
        }

        if !checked {
            self.dispatch_direct(pass, &format!("pcg_check#tally{iterations}"), Bind::LevelPlan(0), 1);
        }
    }

    pub fn encode_substep(&self, encoder: &mut wgpu::CommandEncoder, profiler: Option<&Profiler>, substep: u32) {
        self.encode_substep_into(&mut Passes::new(encoder, profiler), substep);
    }

    pub(super) fn encode_substep_into(&self, passes: &mut Passes<'_>, substep: u32) {
        let has_bodies = self.params.body_count > 0;
        let band = self.config.has_band();
        let tension = self.config.has_tension();
        let shared = self.shares_density_grid();

        if shared {
            let pass = passes.section(Section::Geometry, substep);
            self.encode_geometry(pass);
            // the density solve reuses this hierarchy, valid only while S_GRADIENT / S_SCALE stays dx in pressure_scale
            self.dispatch(pass, "set_pressure_scale", Bind::Sim, Slots::SINGLE);
            self.encode_hierarchy(pass);

            let pass = passes.section(Section::Density, substep);
            self.encode_density_projection(pass);
            self.encode_sort(pass);
        }

        {
            let pass = passes.section(Section::Transfer, substep);

            if band {
                self.encode_band_advection(pass);
            }

            if band {
                self.dispatch_direct(pass, "p2g", Bind::Sim, self.p2g_groups());
                self.dispatch(pass, "redistance_seed", Bind::Sim, Slots::CELLS);
                self.encode_seeded_redistance(pass, "redistance_project#adopt");
                self.dispatch(pass, "extend_from_scratch", Bind::Sim, Slots::CELLS);
                self.dispatch(pass, "extend_into_solid", Bind::Sim, Slots::CELLS);
                self.dispatch(pass, "extend_settle", Bind::Sim, Slots::CELLS);
                self.dispatch(pass, "band_pick", Bind::Sim, Slots::FACES);
            } else {
                self.dispatch_direct(pass, "p2g", Bind::Sim, self.p2g_groups());
            }

            let finish = if band { "faces" } else { "bodies" };
            self.dispatch(pass, "extrapolate_pass", Bind::Sim, Slots::FACES);

            if self.is_coupled() {
                self.dispatch(pass, &format!("extrapolate_inflow#{finish}"), Bind::Coupling, Slots::FACES);
            } else {
                self.dispatch(pass, &format!("extrapolate_back#{finish}"), Bind::Sim, Slots::FACES);
            }
        }

        {
            let pass = passes.section(if shared { Section::Pressure } else { Section::Geometry }, substep);

            if !band && !shared {
                self.encode_geometry(pass);
            }

            if self.config.static_solid && self.config.has_adhesion() {
                self.dispatch(pass, "adhere_static", Bind::Sim, Slots::FACES);
            } else if has_bodies && self.config.has_adhesion() {
                self.dispatch(pass, "adhere", Bind::Sim, Slots::FACES);
            }

            if !shared {
                self.dispatch(pass, if band { "set_pressure_scale_gravity" } else { "set_pressure_scale" }, Bind::Sim, Slots::SINGLE);
                self.encode_hierarchy(pass);
            }

            self.encode_tall(pass);
            let floor = self.config.open_floor.is_some();
            let volume = band && !floor && (!self.is_coupled() || self.tiers_linked());

            if volume {
                self.dispatch(pass, "divergence_volume", Bind::Level(0), Slots::CELLS);

                if self.tiers_linked() {
                    self.dispatch(pass, "band_volume_target", Bind::Coupling, Slots::SINGLE);
                } else {
                    self.dispatch(pass, "band_volume_final", Bind::Sim, Slots::SINGLE);
                }

                self.dispatch(pass, "band_volume_rhs", Bind::Level(0), Slots::CELLS);
            } else {
                self.dispatch(pass, "divergence", Bind::Level(0), Slots::CELLS);
            }

            if floor {
                let columns = groups(self.config.dims.x * self.config.dims.z, WG);
                self.dispatch_direct(pass, "floor_rhs", Bind::Level(0), columns);
            }

            if tension {
                self.encode_redistance(pass);
                self.encode_curvature(pass);
                self.encode_jump(pass);
            }

            self.encode_tall_rhs(pass);
        }

        let pass = if shared { passes.current() } else { passes.section(Section::Pressure, substep) };
        self.encode_solve(pass, has_bodies, self.config.pressure_iterations, self.has_tall());

        let fused = !tension && !self.has_tall();

        if has_bodies {
            self.dispatch(pass, "body_gather#0", Bind::Level(0), Slots::BODIES);
            self.dispatch(pass, "body_impulse", Bind::Level(0), Slots::SINGLE);

            if !fused {
                self.dispatch(pass, "refresh_solid_velocity", Bind::Sim, Slots::FACES);
                self.encode_boundary(pass);
            }
        }

        if fused && has_bodies && self.is_coupled() {
            self.dispatch(pass, &format!("gradient_inflow#{}", u8::from(band)), Bind::LevelCoupling(0), Slots::FACES);
        } else if fused {
            self.dispatch(pass, &format!("gradient_flow#{}{}", u8::from(has_bodies), u8::from(band)), Bind::Level(0), Slots::FACES);
        } else {
            self.encode_tall_expand(pass);
            self.dispatch(pass, "apply_gradient", Bind::Level(0), Slots::FACES);

            if tension {
                self.dispatch(pass, "jump_gradient", Bind::Level(0), Slots::FACES);
            }

            if band {
                self.dispatch(pass, "blend_partial", Bind::Sim, Slots::FACES);
            }
        }

        if self.config.open_floor.is_some() {
            self.dispatch_direct(pass, "floor_flux", Bind::Level(0), 1);
        }

        if !self.extrapolated(pass, self.config.extrapolation_layers, "extrapolate_back#constrain") {
            self.dispatch(pass, "constrain_solid", Bind::Sim, Slots::FACES);
        }

        if self.config.diffuses() {
            let pass = passes.section(Section::Viscosity, substep);
            let kinds: Vec<Diffusion> = [(Diffusion::Capillary, tension), (Diffusion::Viscosity, self.config.has_viscosity())]
                .into_iter()
                .filter(|(_, on)| *on)
                .map(|(kind, _)| kind)
                .collect();
            self.encode_diffusion(pass, &kinds);
        }

        let pass = passes.section(Section::Advect, substep);
        let floor = self.config.open_floor.is_some();

        if floor && !band {
            let spawned = self.config.particles_per_cell * self.config.dims.x * self.config.dims.z;
            self.dispatch_direct(pass, "floor_spawn", Bind::Sim, groups(spawned, WG));
            self.dispatch_direct(pass, "floor_grow", Bind::Plan, 1);
        }

        if band {
            self.dispatch(pass, "g2p#delete", Bind::Sim, Slots::PARTICLES);
            self.encode_reseed(pass);
        } else {
            self.dispatch(pass, "g2p", Bind::Sim, Slots::PARTICLES);
        }

        self.dispatch(pass, "advect#bodies", Bind::Sim, Slots::PARTICLES);

        if floor {
            self.dispatch(pass, "floor_cull", Bind::Sim, Slots::PARTICLES);
        }

        if self.is_coupled() {
            self.dispatch(pass, "keep_inflow", Bind::Coupling, Slots::PARTICLES);
        } else {
            self.dispatch(pass, "keep_inside", Bind::Sim, Slots::PARTICLES);
        }

        self.encode_expel(pass);

        if self.heats() {
            self.dispatch(pass, "heat_particles", Bind::Sim, Slots::PARTICLES);
        }

        if band {
            self.dispatch(pass, "band_blend#store", Bind::Sim, Slots::FACES);
        }

        if self.config.density_projection && !band && !shared {
            let pass = passes.section(Section::Density, substep);
            self.encode_density_projection(pass);
        }
    }

    pub(super) fn shares_density_grid(&self) -> bool {
        let c = &self.config;

        let simple = !c.has_band() && c.open_floor.is_none() && !self.has_tall() && !self.free_bodies && self.coupling.is_none();

        c.shared_density_grid && c.density_projection && simple
    }

    pub fn encode_density_projection(&self, pass: &mut wgpu::ComputePass<'_>) {
        self.dispatch(pass, "save_pressure", Bind::Level(0), Slots::CELLS);

        if !self.shares_density_grid() {
            self.encode_sort(pass);
            self.encode_geometry(pass);

            if self.config.open_floor.is_some() {
                self.dispatch(pass, "floor_close", Bind::Sim, Slots::FACES);
            }

            self.dispatch(pass, "set_density_scale", Bind::Sim, Slots::SINGLE);
            self.encode_hierarchy(pass);
            self.encode_tall_reset(pass);
        }

        self.dispatch(pass, "density_error", Bind::Level(0), Slots::GRID);
        self.encode_solve(pass, false, self.config.density_iterations, false);
        self.dispatch(pass, "zero_velocity", Bind::Sim, Slots::FACES);
        self.dispatch(pass, "apply_gradient", Bind::Level(0), Slots::FACES);

        if !self.extrapolated(pass, 2, "extrapolate_back#zero") {
            self.dispatch(pass, "constrain_zero", Bind::Sim, Slots::FACES);
        }
        self.dispatch(pass, "displace", Bind::Sim, Slots::PARTICLES);
        self.dispatch(pass, "keep_inside", Bind::Sim, Slots::PARTICLES);
        self.encode_expel(pass);
        self.dispatch(pass, "restore_pressure", Bind::Level(0), Slots::CELLS);
    }

    pub(super) fn encode_expel(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.config.static_solid {
            self.dispatch(pass, "expel_static", Bind::Sim, Slots::PARTICLES);
        }
    }
}
