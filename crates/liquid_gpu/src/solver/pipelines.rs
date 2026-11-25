use super::*;

pub const SHADER_SOURCES: [&str; 25] = [
    include_str!("../shaders/common.wgsl"),
    include_str!("../shaders/bindings.wgsl"),
    include_str!("../shaders/level.wgsl"),
    include_str!("../shaders/plan.wgsl"),
    include_str!("../shaders/sort.wgsl"),
    include_str!("../shaders/brick.wgsl"),
    include_str!("../shaders/transfer.wgsl"),
    include_str!("../shaders/levelset.wgsl"),
    include_str!("../shaders/velocity.wgsl"),
    include_str!("../shaders/mg.wgsl"),
    include_str!("../shaders/pcg.wgsl"),
    include_str!("../shaders/rigid.wgsl"),
    include_str!("../shaders/idp.wgsl"),
    include_str!("../shaders/surface.wgsl"),
    include_str!("../shaders/coupling.wgsl"),
    include_str!("../shaders/whitewater.wgsl"),
    include_str!("../shaders/redistance.wgsl"),
    include_str!("../shaders/band.wgsl"),
    include_str!("../shaders/tension.wgsl"),
    include_str!("../shaders/viscosity.wgsl"),
    include_str!("../shaders/adhesion.wgsl"),
    include_str!("../shaders/floor.wgsl"),
    include_str!("../shaders/tall.wgsl"),
    include_str!("../shaders/heat.wgsl"),
    include_str!("../shaders/emitter.wgsl"),
];

pub fn shader_source(subgroups: bool, tail_words: u32) -> String {
    let reduce = if subgroups {
        include_str!("../shaders/reduce_subgroup.wgsl")
    } else {
        include_str!("../shaders/reduce_shared.wgsl")
    };

    [SHADER_SOURCES.join("\n"), reduce.to_string(), format!("const TAIL_WORDS: u32 = {tail_words}u;")].join("\n")
}

pub(super) fn pipeline_layout(device: &wgpu::Device, layouts: &[Option<&wgpu::BindGroupLayout>]) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("liquid pipeline layout"),
        bind_group_layouts: layouts,
        immediate_size: 0,
    })
}

pub(crate) type Spec<'a> = (String, &'static str, Vec<(&'a str, f64)>);

pub(super) const PLAIN: [&str; 7] = ["mg_smooth#", "mg_band#", "mg_residual", "mg_restrict", "mg_prolong", "mg_presmooth", "mg_correct"];

pub(super) fn pipeline_specs<'a>(plan: &[(&'a str, f64)], config: &SolverConfig, words: u32) -> Vec<Spec<'a>> {
    let sim = [
        "scatter_particles",
        "rank_particles",
        "measure_speed",
        "measure_flow",
        "set_pressure_scale",
        "set_density_scale",
        "p2g",
        "g2p",
        "advect",
        "keep_inside",
        "liquid_phi",
        "solid_nodes",
        "mark_solid_cells",
        "extend_into_solid",
        "extend_from_scratch",
        "extend_settle",
        "settle_from_scratch",
        "face_weights",
        "refresh_solid_velocity",
        "extrapolate_pass",
        "extrapolate_back",
        "set_pressure_scale_gravity",
        "copy_scratch",
        "constrain_solid",
        "blend_partial",
        "constrain_zero",
        "zero_velocity",
        "displace",
        "aniso_prepare",
        "surface_splat",
        "surface_field",
        "white_spawn",
        "white_copy",
        "heat_particles",
    ];
    let level = [
        "divergence",
        "divergence_volume",
        "apply_gradient",
        "level_from_fine",
        "body_mark",
        "body_impulse",
        "density_error",
        "save_pressure",
        "restore_pressure",
    ];
    let pair = ["coarsen", "mg_restrict", "mg_prolong", "mg_correct"];
    let solve = ["reset_speed"];
    let solve_level = [
        "level_classify",
        "level_assemble",
        "mg_zero",
        "mg_residual",
        "mg_presmooth",
        "tail_pack",
    ];
    let solve_plan = ["begin_frame", "open_all", "plan_substep"];
    let plan_entries = ["white_close"];

    let mut specs: Vec<Spec<'a>> = Vec::new();

    specs.extend(sim.iter().map(|e| (e.to_string(), *e, Vec::new())));
    specs.extend(level.iter().map(|e| (e.to_string(), *e, Vec::new())));
    specs.extend(pair.iter().map(|e| (e.to_string(), *e, Vec::new())));
    specs.extend(solve.iter().map(|e| (e.to_string(), *e, Vec::new())));
    specs.extend(solve_level.iter().map(|e| (e.to_string(), *e, Vec::new())));
    specs.extend(plan_entries.iter().map(|e| (e.to_string(), *e, plan.to_vec())));
    specs.extend(solve_plan.iter().map(|e| (e.to_string(), *e, plan.to_vec())));
    specs.push(("plan_substep#grow".to_string(), "plan_substep", [plan, &[("PLAN_GROW", 1.0)]].concat()));
    specs.push(("sort_count".to_string(), "sort_count", Vec::new()));
    specs.extend(face_finish_specs());
    specs.push(("advect#bodies".to_string(), "advect", vec![("ADVECT_BODIES", 1.0)]));
    specs.push(("surface_field#white".to_string(), "surface_field", vec![("SURFACE_WHITE", 1.0)]));
    specs.push(("white_spawn#advect".to_string(), "white_spawn", vec![("WHITE_ADVECT", 1.0)]));

    if config.has_band() {
        specs.push(("g2p#delete".to_string(), "g2p", vec![("G2P_DELETE", 1.0)]));
        specs.push(("band_blend#store".to_string(), "band_blend", vec![("BAND_STORE", 1.0)]));
    }
    specs.push(("sort_scan".to_string(), "sort_scan", plan.to_vec()));
    specs.push(("sort_link".to_string(), "sort_link", plan.to_vec()));
    specs.push(("sort_copy".to_string(), "sort_copy", Vec::new()));
    specs.push(("sort_copy#surface".to_string(), "sort_copy", vec![("SORT_PACK", 1.0)]));
    specs.extend(pcg_specs(plan, &block_open(config, words)));
    specs.extend(
        ["couple_boundary", "couple_columns", "measure_inflow", "keep_inflow", "band_volume_target", "couple_seed"]
            .iter()
            .map(|e| (e.to_string(), *e, Vec::new())),
    );

    for color in 0..2u32 {
        specs.push((format!("mg_smooth#{color}"), "mg_smooth", vec![("COLOR", color as f64)]));
    }

    if config.smooths_boundary() {
        let reach = config.boundary_width as f64 - 1.0;
        specs.push(("level_band".to_string(), "level_band", vec![("BOUNDARY_REACH", reach)]));

        for color in 0..2u32 {
            specs.push((format!("mg_band#{color}"), "mg_smooth", vec![("COLOR", color as f64), ("MASK", 4.0)]));
        }
    }

    if config.open_floor.is_some() {
        specs.extend(
            ["floor_surface", "floor_mean", "floor_ghost", "floor_rhs", "floor_flux"]
                .iter()
                .map(|e| (e.to_string(), *e, Vec::new())),
        );
        specs.extend(["floor_spawn", "floor_cull", "floor_close"].iter().map(|e| (e.to_string(), *e, Vec::new())));
        specs.push(("floor_grow".to_string(), "floor_grow", plan.to_vec()));
    }

    specs.extend(tail_specs(config, words));
    specs.extend(block_specs(plan, config));
    specs.extend(feature_specs(plan, config));
    specs.extend(crate::emitter::specs(plan, config));

    if config.has_band() {
        specs.push(("redistance_project#adopt".to_string(), "redistance_project", vec![("PROJECT_ADOPT", 1.0)]));
    }

    if config.tall_cells.is_some() {
        for entry in ["tall_low", "tall_cap", "tall_rows", "tall_lines", "tall_assemble", "tall_rhs", "tall_expand", "tall_reset"] {
            specs.push((entry.to_string(), entry, Vec::new()));
        }

        specs.push(("tall_coarse".to_string(), "tall_coarse", Vec::new()));
        specs.push(("tall_gather".to_string(), "tall_gather", Vec::new()));
        for mut spec in variants("mg_line", "mg_line", "COLOR", &[0, 1]) {
            spec.2.push(("LINE_ROWS", (config.dims.y + 2) as f64));
            specs.push(spec);
        }
    }

    let plain: Vec<Spec<'a>> = if config.tall_cells.is_some() {
        specs
            .iter()
            .filter(|s| PLAIN.iter().any(|p| s.0.starts_with(p)))
            .map(|s| (format!("{}~plain", s.0), s.1, s.2.clone()))
            .collect()
    } else {
        Vec::new()
    };

    for spec in &mut specs {
        if config.sparse {
            spec.2.push(("SPARSE", 1.0));
        }

        if let Some(reach) = config.open_floor {
            spec.2.extend([("OPEN_FLOOR", 1.0), ("FLOOR_REACH", reach as f64)]);
        }

        if let Some(band) = config.tall_cells {
            spec.2.extend([("TALL", 1.0), ("TALL_BAND", band as f64)]);
        }

        spec.2.extend(override_constants(config));
    }

    for mut spec in plain {
        if config.sparse {
            spec.2.push(("SPARSE", 1.0));
        }

        if let Some(reach) = config.open_floor {
            spec.2.extend([("OPEN_FLOOR", 1.0), ("FLOOR_REACH", reach as f64)]);
        }

        spec.2.extend(override_constants(config));
        specs.push(spec);
    }

    specs
}

pub(super) fn override_constants(config: &SolverConfig) -> Vec<(&'static str, f64)> {
    let defaults = SolverConfig::default();
    let mut out = Vec::new();

    if config.particles_per_cell != defaults.particles_per_cell {
        out.push(("PER_CELL", config.particles_per_cell as f64));
    }

    if config.backtrace_order != defaults.backtrace_order {
        out.push(("BACKTRACE_ORDER", config.backtrace_order as f64));
    }

    if config.quantized {
        out.push(("QUANTIZED", 1.0));
    }

    if config.late_surface {
        out.push(("LATE_SURFACE", 1.0));
    }

    if config.cull_crowded {
        out.push(("CULL_CROWDED", 1.0));
    }

    if config.static_solid {
        out.push(("STATIC_SOLID", 1.0));
    }

    out
}

pub(super) fn face_finish_specs<'a>() -> Vec<Spec<'a>> {
    let on = |name: &'a str| (name, 1.0);
    let mut specs: Vec<Spec<'a>> = vec![
        ("extrapolate_back#faces".to_string(), "extrapolate_back", vec![on("EXTRAP_GRAVITY"), on("EXTRAP_FACES")]),
        ("extrapolate_back#bodies".to_string(), "extrapolate_back", vec![on("EXTRAP_GRAVITY"), on("EXTRAP_BODIES")]),
        ("extrapolate_back#constrain".to_string(), "extrapolate_back", vec![("EXTRAP_CONSTRAIN", 1.0)]),
        ("extrapolate_back#zero".to_string(), "extrapolate_back", vec![("EXTRAP_CONSTRAIN", 2.0)]),
        ("extrapolate_inflow#faces".to_string(), "extrapolate_inflow", vec![on("EXTRAP_GRAVITY"), on("EXTRAP_FACES")]),
        ("extrapolate_inflow#bodies".to_string(), "extrapolate_inflow", vec![on("EXTRAP_GRAVITY"), on("EXTRAP_BODIES")]),
    ];

    for blend in 0..2u8 {
        for refresh in 0..2u8 {
            let constants = vec![("GRADIENT_REFRESH", f64::from(refresh)), ("GRADIENT_BLEND", f64::from(blend))];
            specs.push((format!("gradient_flow#{refresh}{blend}"), "gradient_flow", constants));
        }

        specs.push((format!("gradient_inflow#{blend}"), "gradient_inflow", vec![("GRADIENT_BLEND", f64::from(blend))]));
    }

    specs
}

pub(super) fn pcg_specs<'a>(plan: &[(&'a str, f64)], open: &[(&'a str, f64)]) -> Vec<Spec<'a>> {
    let flags = |coupled: bool, lead: bool| vec![("COUPLED", f64::from(u8::from(coupled))), ("LEAD", f64::from(u8::from(lead)))];
    let mut specs: Vec<Spec<'a>> = vec![
        ("pcg_apply#lead".to_string(), "pcg_apply", flags(false, true)),
        ("pcg_apply".to_string(), "pcg_apply", flags(false, false)),
        ("pcg_dot".to_string(), "pcg_dot", Vec::new()),
        ("mg_finish".to_string(), "mg_finish", vec![("COLOR", 0.0)]),
        ("pcg_scalar#lead#coupled".to_string(), "pcg_scalar", flags(true, true)),
        ("pcg_update#lead#coupled".to_string(), "pcg_update", flags(true, true)),
        ("pcg_check#lead".to_string(), "pcg_check", [plan, &flags(false, true)].concat()),
        ("pcg_check".to_string(), "pcg_check", [plan, &flags(false, false)].concat()),
    ];

    for coupled in [false, true] {
        let suffix = if coupled { "#coupled" } else { "" };

        for entry in ["pcg_init_residual", "pcg_scalar", "pcg_update"] {
            specs.push((format!("{entry}{suffix}"), entry, flags(coupled, false)));
        }

        specs.push((format!("pcg_open{suffix}"), "pcg_open", [plan, &flags(coupled, false), open].concat()));
    }

    for vector in 0..2u32 {
        specs.push((format!("body_gather#{vector}"), "body_gather", vec![("VECTOR", vector as f64)]));
    }

    specs
}

pub(super) fn tail_specs<'a>(config: &SolverConfig, words: u32) -> Vec<Spec<'a>> {
    let dims = level_dims(config.dims, config.multigrid_min_dim);
    let tall = if config.tall_cells.is_some() { TALL_LEVELS.min(dims.len() - 1) } else { 0 };
    let tail = tail_start(&dims, tall, words);

    if tail >= dims.len() {
        return Vec::new();
    }

    let band = if config.smooths_boundary() { config.boundary_sweeps } else { 0 };
    let constants = vec![
        ("TAIL_COUNT", (dims.len() - tail) as f64),
        ("TAIL_CELLS", dims[tail..].iter().map(|d| d.element_product()).sum::<u32>() as f64),
        ("SWEEPS", config.coarse_sweeps as f64),
        ("SMOOTHING", config.smoothing as f64),
        ("BAND_SWEEPS", band as f64),
    ];

    vec![("mg_tail".to_string(), "mg_tail", constants)]
}

pub(super) fn block_open(config: &SolverConfig, words: u32) -> Vec<(&'static str, f64)> {
    if !config.block_smoothing {
        return Vec::new();
    }

    let dims = level_dims(config.dims, config.multigrid_min_dim);
    let first = Slots { levels: dims.len() as u32 }.solve_block(0);

    vec![("BLOCK_LEVELS", tail_start(&dims, 0, words) as f64), ("SOLVE_BLOCK_SLOT", first as f64)]
}

pub(super) fn block_specs<'a>(plan: &[(&'a str, f64)], config: &SolverConfig) -> Vec<Spec<'a>> {
    let mut counts = vec![config.pressure_iterations, config.density_iterations];
    counts.sort_unstable();
    counts.dedup();

    let mut specs: Vec<Spec<'a>> = counts
        .into_iter()
        .map(|n| (format!("pcg_check#tally{n}"), "pcg_check", [plan, &[("LEAD", 1.0), ("TALLY", n as f64 + 1.0)]].concat()))
        .collect();

    if !config.block_smoothing {
        return specs;
    }

    let sweeps = ("BLOCK_SWEEPS", config.smoothing as f64);

    specs.extend([
        ("mg_block_down".to_string(), "mg_block_down", vec![sweeps]),
        ("mg_block_coarse".to_string(), "mg_block_coarse", vec![sweeps]),
        ("mg_block_restrict".to_string(), "mg_block_coarse", vec![sweeps, ("RESTRICT_ONLY", 1.0)]),
        ("mg_block_up".to_string(), "mg_block_up", vec![sweeps]),
        ("mg_block_finish".to_string(), "mg_block_up", vec![sweeps, ("FINISH", 1.0)]),
        ("mg_block_complete".to_string(), "mg_block_complete", Vec::new()),
    ]);

    for swap in [false, true] {
        for coupled in [false, true] {
            let key = format!("mg_block_down#update{}{}", if swap { "#swap" } else { "" }, if coupled { "#coupled" } else { "" });
            let bits = [("UPDATE", 1.0), ("SWAP", f64::from(u8::from(swap))), ("COUPLED", f64::from(u8::from(coupled)))];
            specs.push((key, "mg_block_down", [&[sweeps][..], &bits].concat()));
        }
    }

    specs
}

pub(super) fn variants<'a>(name: &str, entry: &'static str, constant: &'a str, values: &[u32]) -> Vec<Spec<'a>> {
    values.iter().map(|v| (format!("{name}#{v}"), entry, vec![(constant, *v as f64)])).collect()
}

pub(super) fn feature_specs<'a>(plan: &[(&'a str, f64)], config: &SolverConfig) -> Vec<Spec<'a>> {
    let mut specs: Vec<Spec<'a>> = Vec::new();
    let mut plain = |names: &[&'static str]| {
        specs.extend(names.iter().map(|e| (e.to_string(), *e, Vec::new())));
    };

    if config.redistances() {
        plain(&["redistance_seed", "redistance_finish", "redistance_project"]);
    }

    if config.has_band() {
        plain(&[
            "band_backward",
            "band_pick",
            "band_volume_partial",
            "band_volume_final",
            "band_delete",
            "band_count",
            "band_scan_sums",
            "band_grow",
            "band_seed",
        ]);
        plain(&["band_forward", "band_liquid_phi", "band_adopt", "band_scan_blocks", "band_spawn", "band_blend"]);
        plain(&["band_volume_rhs"]);

        if config.late_surface {
            plain(&["adopt_band"]);
        }
    }

    if config.has_tension() {
        plain(&["curvature"]);
        plain(&["jump_seed", "jump_rhs", "jump_gradient"]);
    }

    if config.diffuses() {
        plain(&["viscous_residual"]);
    }

    if config.has_adhesion() {
        specs.push(("adhere".to_string(), "adhere", vec![("ADHESION", config.adhesion as f64)]));
    }

    if config.static_solid {
        specs.extend(["solid_nodes_static", "expel_static"].map(|e| (e.to_string(), e, Vec::new())));
    }

    if config.static_solid && config.has_adhesion() {
        specs.push(("adhere_static".to_string(), "adhere_static", vec![("ADHESION", config.adhesion as f64)]));
    }

    if config.redistances() {
        specs.extend(variants("redistance_sweep", "redistance_sweep", "FLIP", &[0, 1]));
    }

    if config.has_tension() {
        specs.extend(variants("blur", "blur", "BLUR", &[0, 1, 2]));
        specs.extend(variants("jump_smooth", "jump_smooth", "SMOOTH", &[0, 1]));
    }

    if config.diffuses() {
        let kinds: Vec<u32> = [(Diffusion::Viscosity, config.has_viscosity()), (Diffusion::Capillary, config.has_tension())]
            .iter()
            .filter(|(_, on)| *on)
            .map(|(kind, _)| *kind as u32)
            .collect();

        for entry in ["viscous_samples", "viscous_edge_fractions", "viscous_couplings", "viscous_finish"] {
            specs.extend(variants(entry, entry, "KIND", &kinds));
        }

        if kinds.len() > 1 {
            specs.push(("viscous_couplings#1#only".to_string(), "viscous_couplings", vec![("KIND", 1.0), ("COUPLINGS_ONLY", 1.0)]));
            specs.push(("viscous_couplings#0#add".to_string(), "viscous_couplings", vec![("KIND", 0.0), ("ADD_COUPLINGS", 1.0)]));
        }

        let blocks = Slots { levels: level_dims(config.dims, config.multigrid_min_dim).len() as u32 }.solve_viscous();

        for kind in kinds {
            let with = |extra: &[(&'a str, f64)]| [plan, &[("KIND", kind as f64)], extra].concat();
            specs.push((format!("viscous_open#{kind}"), "viscous_open", with(&[("VISCOUS_SLOT", blocks as f64)])));
            specs.push((format!("viscous_check#{kind}"), "viscous_check", with(&[])));
            specs.push((format!("viscous_check#lead#{kind}"), "viscous_check", with(&[("LEAD", 1.0)])));
        }

        for (key, flags) in [("#first", &[("FIRST", 1.0)][..]), ("", &[]), ("#swap", &[("SWAP", 1.0)])] {
            specs.push((format!("viscous_apply{key}"), "viscous_apply", flags.to_vec()));
        }

        for (key, flags) in [("#lead", &[("LEAD", 1.0)][..]), ("", &[]), ("#swap", &[("SWAP", 1.0)])] {
            specs.push((format!("viscous_block{key}"), "viscous_block", flags.to_vec()));
        }
    }

    specs
}
