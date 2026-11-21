use glam::{DVec3, UVec3};
use liquid_ref::grid::{Grid, sample_cells};
use liquid_ref::levelset::{face_weights, fill_solid_cells, solid_phi_nodes};
use liquid_ref::narrow_band::NarrowBand;
use liquid_ref::particles::{hash_unit, seed_box};
use liquid_ref::pressure::{Level, Multigrid, PressureSystem, dot, pcg};
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{SEED_JITTER, dam_break, moving, tank, with_body};
use liquid_ref::sim::{Params, Simulation};
use liquid_ref::tall::TALL_LEVELS;

fn sloped_problem(band: Option<u32>) -> (Multigrid, Vec<f64>) {
    let dx = 1.0 / 32.0;
    let grid = Grid::new(UVec3::new(24, 40, 20), dx);
    let size = grid.size();
    let block = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.06, 0.05, 0.04) }, DVec3::new(0.5, 0.55, 0.3), Motion::Static, 1000.0);
    let nodes = solid_phi_nodes(&grid, &[block]);
    let mut phi = grid.cells();

    for n in 0..phi.data.len() {
        let x = grid.cell_center(phi.coords(n));
        let eta = 0.8 * size.y + 0.12 * size.y * (x.x / size.x - 0.5) + 0.02 * (9.0 * x.z).sin();
        phi.data[n] = x.y - eta;
    }

    fill_solid_cells(&mut phi, &nodes);

    let weights = face_weights(&grid, &nodes);
    let level = Level::from_fine(&grid, &phi, &weights, 1.0 / (dx * dx));
    let rhs: Vec<f64> = (0..level.len()).map(|n| if level.active[n] { hash_unit(n as u32 ^ 77) - 0.4 } else { 0.0 }).collect();
    let multigrid = Multigrid::new(level, 4).with_tall(&weights, band);

    (multigrid, rhs)
}

fn random_on(level: &Level, salt: u32) -> Vec<f64> {
    (0..level.len()).map(|n| if level.unknown(n) { hash_unit(n as u32 ^ salt) - 0.5 } else { 0.0 }).collect()
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |m, x| m.max(x.abs()))
}

#[test]
fn reduced_operator_is_the_galerkin_product_and_stays_spd() {
    let (multigrid, _) = sloped_problem(Some(3));
    let tall_levels = multigrid.levels.iter().take_while(|l| !l.tall.is_empty()).count();

    assert_eq!(tall_levels, TALL_LEVELS);
    assert!(multigrid.levels[tall_levels..].iter().all(|l| l.tall.is_empty()));

    for (l, level) in multigrid.levels.iter().enumerate().take(tall_levels) {
        let mut regular = level.clone();
        regular.tall.clear();

        let x = random_on(level, 17 + l as u32);
        let y = random_on(level, 9001 + l as u32);
        let mut reduced = vec![0.0; level.len()];
        level.apply(&x, &mut reduced);

        let mut expanded = x.clone();
        level.expand(&mut expanded);
        let mut galerkin = vec![0.0; level.len()];
        regular.apply(&expanded, &mut galerkin);
        level.gather(&mut galerkin, false);

        let gap = reduced.iter().zip(&galerkin).fold(0.0f64, |m, (a, b)| m.max((a - b).abs())) / max_abs(&galerkin);
        let mut ay = vec![0.0; level.len()];
        level.apply(&y, &mut ay);
        let (xay, yax) = (dot(&x, &ay), dot(&y, &reduced));
        let asymmetry = (xay - yax).abs() / xay.abs().max(yax.abs());
        let tops: Vec<u32> = (0..level.columns()).map(|c| level.top(c)).filter(|t| *t > 0).collect();
        let spread = (tops.iter().min().copied().unwrap_or(0), tops.iter().max().copied().unwrap_or(0));
        let active = level.active.iter().filter(|a| **a).count();

        println!(
            "level {l} dims {}: {} of {} columns tall, tops {spread:?}, unknowns {} of {active} active cells; |A_hat x - R^T A R x| / |R^T A R x| {gap:.2e}, symmetry gap {asymmetry:.2e}, x.A_hat x {:.4e}",
            level.dims,
            tops.len(),
            level.columns(),
            level.unknowns(),
            dot(&x, &reduced)
        );

        assert!(!tops.is_empty() && spread.0 < spread.1 && level.unknowns() < active);
        assert!(gap < 1e-12, "level {l}: Galerkin gap {gap}");
        assert!(asymmetry < 1e-12);
        assert!(dot(&x, &reduced) > 0.0);
    }
}

#[test]
fn tall_preconditioner_is_symmetric_and_cuts_unknowns() {
    let (multigrid, _) = sloped_problem(Some(3));
    let fine = &multigrid.levels[0];
    let x = random_on(fine, 17);
    let y = random_on(fine, 9001);
    let a = dot(&multigrid.precondition(&x), &y);
    let b = dot(&x, &multigrid.precondition(&y));
    let relative = (a - b).abs() / a.abs().max(b.abs());

    println!("tall multigrid: <Mx,y> = {a:e}, <x,My> = {b:e}, relative gap {relative:.3e}, <Mx,x> = {:e}", dot(&multigrid.precondition(&x), &x));

    assert!(relative < 1e-12);
    assert!(dot(&multigrid.precondition(&x), &x) > 0.0);
}

fn iterations(band: Option<u32>) -> (usize, usize, f64) {
    let (multigrid, mut rhs) = sloped_problem(band);
    let system = PressureSystem { multigrid: &multigrid, couplings: &[], bodies: &[], coupling_scale: 0.0 };
    let level = system.level();
    let scale = max_abs(&rhs);
    let tolerance = level.gather_rhs(&mut rhs, 1e-8);

    let mut p = vec![0.0; rhs.len()];
    let stats = pcg(&system, &rhs, &mut p, tolerance, 400, true);

    (stats.iterations, level.unknowns(), stats.residual / scale)
}

#[test]
fn tall_mgpcg_converges_like_regular() {
    let (regular, tall) = (iterations(None), iterations(Some(3)));

    println!(
        "mgpcg to 1e-8: regular {} iterations on {} unknowns (residual {:.1e}), tall cells {} iterations on {} unknowns (residual {:.1e})",
        regular.0, regular.1, regular.2, tall.0, tall.1, tall.2
    );

    assert!(tall.2 <= 1e-8);
    assert!(tall.1 * 2 < regular.1);
    assert!(tall.0 <= 2 * regular.0 + 4);
}

fn band_params(tall: Option<u32>) -> Params {
    Params { narrow_band: Some(NarrowBand::default()), tall_cells: tall, ..Params::default() }
}

fn column_surface(sim: &Simulation, i: u32, k: u32) -> f64 {
    let phi = |j: u32| sim.phi.get(i, j, k);

    (1..sim.grid.dims.y)
        .find(|&j| phi(j) >= 0.0)
        .map(|j| (j as f64 - 0.5 + phi(j - 1) / (phi(j - 1) - phi(j))) * sim.grid.dx)
        .unwrap_or(sim.grid.size().y)
}

fn liquid_speed(sim: &Simulation) -> f64 {
    let mut worst = 0.0f64;

    for (axis, field) in sim.velocity.axes.iter().enumerate() {
        for n in 0..field.data.len() {
            let x = sim.grid.face_position(axis, field.coords(n));

            if sample_cells(&sim.phi, &sim.grid, x) < 0.0 {
                worst = worst.max(field.data[n].abs());
            }
        }
    }

    worst
}

struct Rest {
    particle_speed: f64,
    face_speed: f64,
    pressure_error: f64,
    unknowns: usize,
    divergence: f64,
}

fn deep_rest(params: Params) -> Rest {
    let dx = 1.0 / 32.0;
    let depth = 36.0 * dx;
    let mut sim = tank(UVec3::new(16, 40, 16), dx, depth, params);

    sim.advance(0.3);

    let d = sim.grid.dims;
    let mut error = 0.0f64;

    for k in 0..d.z {
        for i in 0..d.x {
            let eta = column_surface(&sim, i, k);

            for j in 0..d.y {
                let y = (j as f64 + 0.5) * dx;
                let n = sim.phi.index(i, j, k);

                if sim.phi.data[n] < 0.0 && y < eta - 2.0 * dx {
                    error = error.max((sim.pressure[n] - 1000.0 * 9.81 * (eta - y)).abs());
                }
            }
        }
    }

    Rest {
        particle_speed: sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length())),
        face_speed: liquid_speed(&sim),
        pressure_error: error / (1000.0 * 9.81 * depth),
        unknowns: sim.report.unknowns,
        divergence: sim.report.max_divergence,
    }
}

#[test]
fn hydrostatic_deep_tank_stays_at_rest_with_tall_cells() {
    let wave_speed = (9.81 * 36.0 / 32.0f64).sqrt();
    let cases = [
        ("narrow band, regular", band_params(None)),
        ("narrow band, tall band 4", band_params(Some(4))),
        ("dense, regular", Params::default()),
        ("dense, tall band 4", Params { tall_cells: Some(4), ..Params::default() }),
    ];
    let runs: Vec<Rest> = std::thread::scope(|s| {
        let handles: Vec<_> = cases.iter().map(|(_, p)| s.spawn(move || deep_rest(*p))).collect();

        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    });

    for ((name, _), r) in cases.iter().zip(&runs) {
        println!(
            "{name}: after 0.3 s max particle speed {:.3e} m/s, max liquid face speed {:.3e} m/s ({:.3e} of sqrt(gh)), max |p - rho g (eta - y)| {:.3e} of rho g h, pressure unknowns {}, max divergence {:.3e}",
            r.particle_speed,
            r.face_speed,
            r.face_speed / wave_speed,
            r.pressure_error,
            r.unknowns,
            r.divergence
        );
    }

    for (r, plain) in [(&runs[1], &runs[0]), (&runs[3], &runs[2])] {
        assert!(r.unknowns * 3 < plain.unknowns);
        assert!(r.face_speed < 0.01 * wave_speed);
        assert!(r.particle_speed < 0.01 * wave_speed);
        assert!(r.pressure_error < 2e-3);
        assert!(r.divergence < 1e-4f64.max(1.1 * plain.divergence));
    }
}

fn zero_crossing_omega(samples: &[(f64, f64)]) -> f64 {
    let crossings: Vec<f64> = samples
        .windows(2)
        .filter(|w| w[0].1.signum() != w[1].1.signum())
        .map(|w| w[0].0 + (w[1].0 - w[0].0) * w[0].1 / (w[0].1 - w[1].1))
        .collect();
    let spans = crossings.len() - 1;

    std::f64::consts::PI * spans as f64 / (crossings[spans] - crossings[0])
}

struct Wave {
    omega: f64,
    airy: f64,
    kh: f64,
    unknowns: usize,
}

fn deep_wave(tall: Option<u32>) -> Wave {
    let cells = 32;
    let depth = 0.1;
    let dx = depth / cells as f64;
    let kh = std::f64::consts::PI;
    let length = (std::f64::consts::PI * cells as f64 / kh).round() as u32;
    let k = std::f64::consts::PI / (length as f64 * dx);
    let airy = (9.81 * k * (k * depth).tanh()).sqrt();
    let speed = 0.05 * depth * airy / (k * depth).sinh();
    let grid = Grid::new(UVec3::new(length, cells + 4, 4), dx);
    let mut water = grid.size();
    water.y = depth;

    let particles = moving(seed_box(&grid, DVec3::ZERO, water, SEED_JITTER), |x| {
        DVec3::new(-speed * (k * x.y).cosh() * (k * x.x).sin(), speed * (k * x.y).sinh() * (k * x.x).cos(), 0.0)
    });
    let mut sim = Simulation::new(grid, band_params(tall), particles, Vec::new());
    let mode = |sim: &Simulation| sim.particles.iter().map(|p| p.v.x * (k * p.x.x).sin()).sum::<f64>();
    let frame = 2.0 * std::f64::consts::PI / airy / 40.0;
    let mut samples = vec![(0.0, mode(&sim))];
    let mut unknowns = 0;

    for _ in 0..128 {
        sim.advance(frame);
        samples.push((sim.time, mode(&sim)));
        unknowns = unknowns.max(sim.report.unknowns);
    }

    Wave { omega: zero_crossing_omega(&samples), airy, kh: k * depth, unknowns }
}

#[test]
fn deep_standing_wave_with_tall_cells_keeps_airy_dispersion() {
    let bands = [None, Some(8), Some(12), Some(16)];
    let runs: Vec<Wave> = std::thread::scope(|s| {
        let handles: Vec<_> = bands.iter().map(|b| s.spawn(move || deep_wave(*b))).collect();

        handles.into_iter().map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))).collect()
    });
    let regular = &runs[0];

    for (band, r) in bands.iter().zip(&runs) {
        println!(
            "kh {:.3}, 32 cells per depth, tall band {band:?}: omega {:.5} rad/s, Airy {:.5} ({:+.2}%), vs regular {:+.3}%, peak pressure unknowns {}",
            r.kh,
            r.omega,
            r.airy,
            100.0 * (r.omega / r.airy - 1.0),
            100.0 * (r.omega / regular.omega - 1.0),
            r.unknowns
        );
    }

    let gaps: Vec<f64> = runs[1..].iter().map(|r| (r.omega / regular.omega - 1.0).abs()).collect();
    let tall = &runs[2];

    assert!(gaps.windows(2).all(|w| w[1] < w[0]), "gap to the regular grid must shrink as the band widens: {gaps:?}");
    assert!(tall.unknowns * 2 < regular.unknowns);
    assert!(gaps[1] < 0.015);
    assert!((tall.omega / tall.airy - 1.0).abs() < 0.4 / 32.0);
}

fn front_slope(params: Params) -> (f64, usize) {
    let a = 0.057;
    let cells = 16;
    let dx = a / cells as f64;
    let mut sim = dam_break(UVec3::new(5 * cells, 3 * cells, 4), dx, DVec3::new(a, 2.0 * a, 0.0), params);
    let t_scale = (a / (2.0 * 9.81)).sqrt();
    let mut samples = Vec::new();
    let mut unknowns = 0;

    for n in 1..=40 {
        sim.advance(0.1 * t_scale);
        let front = sim.particles.iter().filter(|p| p.x.y < 0.25 * a).fold(0.0f64, |m, p| m.max(p.x.x));
        samples.push((0.1 * n as f64, front / a));
        unknowns = unknowns.max(sim.report.unknowns);
    }

    let window: Vec<(f64, f64)> = samples.iter().copied().filter(|(_, z)| (1.5..=3.5).contains(z)).collect();
    let n = window.len() as f64;
    let mt = window.iter().map(|(t, _)| t).sum::<f64>() / n;
    let mz = window.iter().map(|(_, z)| z).sum::<f64>() / n;
    let num: f64 = window.iter().map(|(t, z)| (t - mt) * (z - mz)).sum();
    let den: f64 = window.iter().map(|(t, _)| (t - mt).powi(2)).sum();

    (num / den, unknowns)
}

fn floating_block(params: Params) -> (f64, f64) {
    let body = RigidBody::new(Shape::Cuboid { half: DVec3::new(0.08, 0.05, 0.1) }, DVec3::new(0.25, 0.36, 0.25), Motion::Dynamic, 450.0);
    let mut sim = with_body(tank(UVec3::new(20, 24, 20), 0.025, 0.3, params), body);
    let mut heights = Vec::new();
    let mut fewest = usize::MAX;

    while sim.time < 3.0 {
        let dt = sim.cfl_dt();
        sim.substep(dt);
        fewest = fewest.min(sim.report.unknowns);

        if sim.time > 2.0 {
            heights.push(sim.bodies[0].position.y);
        }
    }

    (heights.iter().sum::<f64>() / heights.len() as f64, fewest as f64)
}

#[test]
fn dam_break_and_floating_block_track_regular_with_tall_cells() {
    let ((tall_front, regular_front), (tall_block, regular_block)) = std::thread::scope(|s| {
        let fronts = [None, Some(4)].map(|t| s.spawn(move || front_slope(band_params(t))));
        let blocks = [None, Some(4)].map(|t| s.spawn(move || floating_block(band_params(t))));
        let [regular_front, tall_front] = fronts.map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)));
        let [regular_block, tall_block] = blocks.map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)));

        ((tall_front, regular_front), (tall_block, regular_block))
    });

    println!(
        "Martin & Moyce front slope: regular {:.3} (peak unknowns {}), tall band 4 {:.3} (peak unknowns {}), gap {:+.2}%",
        regular_front.0,
        regular_front.1,
        tall_front.0,
        tall_front.1,
        100.0 * (tall_front.0 / regular_front.0 - 1.0)
    );
    println!(
        "floating block rest height over 2..3 s: regular {:.4} m (fewest unknowns {}), tall band 4 {:.4} m (fewest unknowns {}), gap {:.3} dx",
        regular_block.0,
        regular_block.1,
        tall_block.0,
        tall_block.1,
        (tall_block.0 - regular_block.0) / 0.025
    );

    assert!(tall_front.1 < regular_front.1);
    assert!((tall_front.0 / regular_front.0 - 1.0).abs() < 0.05);
    assert!(tall_block.1 < regular_block.1);
    assert!((tall_block.0 - regular_block.0).abs() < 0.2 * 0.025);
}

#[test]
fn tall_cells_off_leave_the_solve_untouched() {
    let (multigrid, rhs) = sloped_problem(None);
    let level = &multigrid.levels[0];
    let mut expanded = rhs.clone();

    level.expand(&mut expanded);
    level.gather(&mut expanded, true);

    assert!(level.tall.is_empty());
    assert_eq!(level.unknowns(), level.active.iter().filter(|a| **a).count());
    assert_eq!(expanded, rhs);
}
