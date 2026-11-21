use liquid_wave::analytic::DamBreak;
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::swe::{Boundaries, Boundary, CouplingRegion, Side, Swe};

fn quantize(v: f64) -> f64 {
    (v * 1048576.0).round() / 1048576.0
}

fn island_bed(x: f64, z: f64) -> f64 {
    let bowl = 0.002 * ((x - 16.0).powi(2) + (z - 12.0).powi(2));
    let ridge = 0.25 * (0.7 * x).sin() * (0.45 * z).cos();
    let island = 1.6 * (-((x - 20.0).powi(2) + (z - 10.0).powi(2)) / 18.0).exp();

    bowl + ridge + island
}

fn lake(quantized: bool) -> Swe {
    let grid = Grid::new(64, 48, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, z| {
        let b = island_bed(x, z);
        if quantized { quantize(b) } else { b }
    });
    let state = State::lake(&grid, &bed, 1.0);

    Swe::new(grid, bed, state)
}

fn max_velocity(swe: &Swe) -> f64 {
    swe.max_speed()
}

#[test]
fn lake_at_rest_over_dry_island_stays_at_rest() {
    for quantized in [true, false] {
        let mut swe = lake(quantized);
        let g = swe.grid;
        let dry = swe.state.h.iter().filter(|&&h| h == 0.0).count();
        let partial = (0..g.cells())
            .filter(|&c| swe.state.h[c] > 0.0 && !swe.bathymetry.cell(c).is_full(swe.state.h[c]))
            .count();

        assert!(dry > 20, "the island must leave dry cells, got {dry}");
        assert!(
            partial > 5,
            "the shoreline must cut through cells, got {partial}"
        );

        let initial = swe.state.clone();

        for _ in 0..1000 {
            swe.step(1.0);
        }

        let speed = max_velocity(&swe);
        let level_drift = initial
            .h
            .iter()
            .zip(&swe.state.h)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);

        println!(
            "lake at rest (quantized bed: {quantized}): max |u| = {speed:.3e}, max |dh| = {level_drift:.3e}, dry cells = {dry}, partially flooded = {partial}"
        );

        assert!(speed <= 1e-12, "max |u| = {speed:e}");
        assert!(level_drift <= 1e-12, "max |dh| = {level_drift:e}");
    }
}

fn dam_break(cells: usize, problem: DamBreak, length: f64, t_end: f64) -> f64 {
    let grid = Grid::new(cells, 2, length / cells as f64, length / cells as f64);
    let bed = Bathymetry::flat(&grid, 0.0);
    let mut state = State::zeros(grid.cells());

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, k);
            state.h[grid.cell(i, k)] = problem.depth(x, 0.0);
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.boundaries = Boundaries::uniform(&grid, Boundary::Wall)
        .with(Side::West, Boundary::Transmissive)
        .with(Side::East, Boundary::Transmissive);

    swe.advance(t_end);

    (0..grid.nx)
        .map(|i| {
            let [x, _] = grid.center(i, 0);
            (swe.state.h[grid.cell(i, 0)] - problem.depth(x, t_end)).abs() * grid.dx
        })
        .sum::<f64>()
        / length
}

fn convergence(name: &str, problem: DamBreak, bound: f64) {
    let resolutions = [100, 200, 400, 800];
    let errors: Vec<f64> = resolutions
        .iter()
        .map(|&n| dam_break(n, problem, 50.0, 2.5))
        .collect();

    for (w, n) in errors.windows(2).zip(resolutions.windows(2)) {
        let order = (w[0] / w[1]).log2();
        println!(
            "{name}: N={} L1={:.4e} -> N={} L1={:.4e}, observed order {order:.3}",
            n[0], w[0], n[1], w[1]
        );
        assert!(w[1] < w[0], "{name}: error must decrease with refinement");
        assert!(order > 0.5, "{name}: observed order {order} too low");
    }

    let finest = *errors.last().unwrap();
    assert!(
        finest < bound,
        "{name}: L1 error {finest:e} above {bound:e}"
    );
}

#[test]
fn ritter_dry_bed_dam_break_converges() {
    let problem = DamBreak {
        gravity: liquid_wave::GRAVITY,
        dam: 25.0,
        upstream: 1.0,
        downstream: 0.0,
    };

    convergence("ritter", problem, 2e-3);
}

#[test]
fn stoker_wet_bed_dam_break_converges() {
    let problem = DamBreak {
        gravity: liquid_wave::GRAVITY,
        dam: 25.0,
        upstream: 1.0,
        downstream: 0.1,
    };

    convergence("stoker", problem, 2e-3);
}

fn sloshing_basin() -> Swe {
    let grid = Grid::new(48, 40, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, island_bed);
    let mut state = State::lake(&grid, &bed, 1.0);

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let c = grid.cell(i, k);
            if i < 12 && state.h[c] > 0.0 {
                state.h[c] += 0.8;
            }
        }
    }

    Swe::new(grid, bed, state)
}

#[test]
fn mass_is_conserved_and_depth_stays_positive() {
    let mut swe = sloshing_basin();
    let initial = swe.state.mass(&swe.grid);
    let mut min_depth: f64 = 0.0;

    for _ in 0..5000 {
        swe.step(0.05);
        min_depth = min_depth.min(swe.state.h.iter().copied().fold(f64::INFINITY, f64::min));
    }

    let drift = (swe.state.mass(&swe.grid) - initial).abs() / initial;

    println!(
        "closed basin, 5000 steps, t = {:.2} s: relative mass drift {drift:.3e}, min h {min_depth:e}",
        swe.time
    );

    assert!(drift < 1e-12, "relative mass drift {drift:e}");
    assert!(min_depth >= 0.0, "negative depth {min_depth:e}");
}

#[test]
fn injected_sources_are_accounted_exactly() {
    let mut swe = sloshing_basin();
    let g = swe.grid;
    let initial = swe.state.mass(&g);
    let mut min_depth: f64 = 0.0;

    for step in 0..3000 {
        for k in 0..g.nz {
            for i in 0..g.nx {
                let c = g.cell(i, k);
                let phase = 0.37 * (i as f64) + 0.53 * (k as f64) + 0.011 * step as f64;
                swe.source.h[c] = 2e-4 * phase.sin();
                swe.source.hu[c] = 1e-4 * phase.cos();
            }
        }

        swe.step(0.05);
        min_depth = min_depth.min(swe.state.h.iter().copied().fold(f64::INFINITY, f64::min));
    }

    let change = swe.state.mass(&g) - initial;
    let mismatch = (change - swe.injected_mass).abs() / initial;

    println!(
        "sources: mass change {change:.6e}, injected {:.6e}, relative mismatch {mismatch:.3e}, min h {min_depth:e}",
        swe.injected_mass
    );

    assert!(mismatch < 1e-12, "relative mismatch {mismatch:e}");
    assert!(min_depth >= 0.0, "negative depth {min_depth:e}");
}

#[test]
fn dirichlet_rest_state_keeps_lake_at_rest() {
    let grid = Grid::new(32, 16, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, _| quantize(0.02 * x));
    let state = State::lake(&grid, &bed, 1.0);
    let mut swe = Swe::new(grid, bed, state);

    swe.boundaries = swe
        .boundaries
        .with(Side::East, Boundary::Dirichlet)
        .with(Side::West, Boundary::Transmissive);

    for k in 0..grid.nz {
        let b = swe.bathymetry.face_x[grid.x_face(grid.nx, k)];
        swe.boundaries.dirichlet[Side::East.index()][k] = [1.0 - b, 0.0, 0.0];
    }

    for _ in 0..500 {
        swe.step(1.0);
    }

    let speed = swe.max_speed();
    println!("Dirichlet lake at rest: max |u| = {speed:.3e}");
    assert!(speed <= 1e-12);
}

#[test]
fn dirichlet_inflow_raises_the_level() {
    let grid = Grid::new(40, 4, 0.5, 0.5);
    let bed = Bathymetry::flat(&grid, 0.0);
    let state = State::lake(&grid, &bed, 1.0);
    let mut swe = Swe::new(grid, bed, state);

    swe.boundaries = swe.boundaries.with(Side::West, Boundary::Dirichlet);
    swe.boundaries.dirichlet[Side::West.index()].fill([1.5, 0.8, 0.0]);

    let before = swe.state.mass(&grid);
    swe.advance(2.0);
    let after = swe.state.mass(&grid);

    println!("Dirichlet inflow: mass {before:.3} -> {after:.3}");
    assert!(after > before + 1.0);
}

#[test]
fn coupling_region_imposes_external_state() {
    let grid = Grid::new(24, 24, 0.5, 0.5);
    let bed = Bathymetry::flat(&grid, 0.0);
    let state = State::lake(&grid, &bed, 1.0);
    let mut swe = Swe::new(grid, bed, state);

    swe.coupling = Some(CouplingRegion {
        min: [8, 8],
        max: [16, 16],
        blend: 1.0,
    });
    swe.coupled.h.fill(1.2);

    swe.step(0.01);

    let inside = swe.state.h[grid.cell(10, 10)];
    let outside = swe.state.h[grid.cell(2, 2)];
    let expected = 64.0 * 0.2 * grid.area();

    println!(
        "coupling: inside {inside}, outside {outside}, coupled mass {:.6}",
        swe.coupled_mass
    );
    assert!((inside - 1.2).abs() < 1e-14);
    assert!((outside - 1.0).abs() < 1e-14);
    assert!((swe.coupled_mass - expected).abs() < 1e-12);
}

#[test]
fn manning_friction_decelerates_uniform_flow_at_the_analytic_rate() {
    let grid = Grid::new(16, 4, 1.0, 1.0);
    let bed = Bathymetry::flat(&grid, 0.0);
    let (depth, speed, manning) = (2.0, 3.0, 0.03);
    let mut state = State::lake(&grid, &bed, depth);
    state.hu.fill(depth * speed);

    let mut swe = Swe::new(grid, bed, state);
    swe.params.manning = manning;
    swe.boundaries = Boundaries::uniform(&grid, Boundary::Wall)
        .with(Side::West, Boundary::Transmissive)
        .with(Side::East, Boundary::Transmissive);

    swe.advance(10.0);

    let rate = liquid_wave::GRAVITY * manning * manning / depth.powf(4.0 / 3.0);
    let expected = speed / (1.0 + rate * speed * swe.time);
    let measured = swe.velocity(grid.cell(8, 2))[0];
    let error = (measured / expected - 1.0).abs();

    println!(
        "Manning friction: u(10 s) = {measured:.6} m/s, analytic {expected:.6} m/s, error {:.3}%",
        100.0 * error
    );
    assert!(error < 1e-3);
}

fn coupled_basin() -> Swe {
    let mut swe = sloshing_basin();
    let g = swe.grid;

    swe.coupling = Some(CouplingRegion {
        min: [20, 14],
        max: [32, 26],
        blend: 0.5,
    });

    for k in 0..g.nz {
        for i in 0..g.nx {
            let c = g.cell(i, k);
            swe.coupled.h[c] = swe.state.h[c] + 0.05 * (0.4 * i as f64).sin();
            swe.coupled.hu[c] = 0.02;
        }
    }

    swe
}

fn inject_outside(swe: &mut Swe, frame: usize) {
    let g = swe.grid;

    for k in 0..g.nz {
        for i in 0..g.nx {
            if swe.coupling.is_some_and(|r| r.contains(i, k)) {
                continue;
            }

            let angle = 0.31 * (i as f64) + 0.47 * (k as f64) + 0.013 * frame as f64;
            swe.source.h[g.cell(i, k)] = 1e-4 * angle.sin();
        }
    }
}

#[test]
fn interface_ledger_is_exact_in_flux_form() {
    let mut swe = coupled_basin();
    let start = swe.ledger();
    let invariant = |l: liquid_wave::swe::Ledger| l.outside + l.interface - l.injected;

    for frame in 0..2000 {
        inject_outside(&mut swe, frame);
        swe.step(0.05);
    }

    let end = swe.ledger();
    let drift = (invariant(end) - invariant(start)).abs() / start.outside;
    let total = (end.mass - start.mass - end.injected - end.coupled).abs() / start.mass;

    println!(
        "coupled basin, t = {:.2} s: interface {:.6e}, injected {:.6e}, coupled {:.6e}, outside drift {drift:.3e}, total drift {total:.3e}",
        swe.time, end.interface, end.injected, end.coupled
    );

    assert!(end.interface.abs() > 1e-3, "the interface must carry mass");
    assert!(drift < 1e-12, "outside + interface - injected drifted by {drift:e}");
    assert!(total < 1e-12, "mass - injected - coupled drifted by {total:e}");
}

#[test]
fn relocation_hands_mass_between_tiers_exactly() {
    let mut swe = coupled_basin();
    let g = swe.grid;

    for _ in 0..300 {
        swe.step(0.05);
    }

    let before = swe.ledger();
    let target = CouplingRegion {
        min: [6, 4],
        max: [18, 16],
        blend: 0.5,
    };

    for k in 0..g.nz {
        for i in 0..g.nx {
            let c = g.cell(i, k);

            if target.contains(i, k) && !swe.coupling.is_some_and(|r| r.contains(i, k)) {
                swe.coupled.h[c] = swe.state.h[c];
            }
        }
    }

    swe.relocate(Some(target));

    let after = swe.ledger();
    let invariant = |l: liquid_wave::swe::Ledger| l.outside + l.interface - l.injected;
    let tiers = (after.conserved() - before.conserved()).abs() / before.mass;
    let flux = (invariant(after) - invariant(before)).abs() / before.mass;

    println!(
        "relocation: tier {:.6} -> {:.6}, outside {:.6} -> {:.6}, interface {:.6e} -> {:.6e}, tier ledger error {tiers:.3e}, flux ledger error {flux:.3e}",
        before.tier, after.tier, before.outside, after.outside, before.interface, after.interface
    );

    assert!(tiers < 1e-14);
    assert!(flux < 1e-14);
}
