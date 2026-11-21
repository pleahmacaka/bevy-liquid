use glam::{DVec3, UVec3};
use liquid_ref::grid::{Grid, MacField};
use liquid_ref::particles::{hash_unit, seed_box};
use liquid_ref::pressure::{Level, Multigrid, PressureSystem, dot, pcg};
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{SEED_JITTER, dam_break, moving, tank, with_body};
use liquid_ref::sim::{Params, Simulation};

fn sphere_problem(n: u32) -> (Grid, Level, Vec<f64>, Vec<f64>) {
    let dx = 1.0 / n as f64;
    let grid = Grid::new(UVec3::splat(n), dx);
    let center = DVec3::splat(0.5);
    let radius = 0.37;

    let mut phi = grid.cells();

    for i in 0..phi.data.len() {
        let x = grid.cell_center(phi.coords(i));
        phi.data[i] = (x - center).length() - radius;
    }

    let mut weights = MacField::new(&grid);
    weights.fill(1.0);

    let level = Level::from_fine(&grid, &phi, &weights, 1.0 / (dx * dx));

    let exact: Vec<f64> = (0..grid.cell_count())
        .map(|i| {
            let x = grid.cell_center(phi.coords(i));
            radius * radius - (x - center).length_squared()
        })
        .collect();

    let rhs: Vec<f64> = level
        .active
        .iter()
        .map(|a| if *a { 6.0 } else { 0.0 })
        .collect();

    (grid, level, exact, rhs)
}

fn solve_error(n: u32) -> f64 {
    let (_, level, exact, rhs) = sphere_problem(n);
    let mg = Multigrid::new(level, 4);
    let system = PressureSystem {
        multigrid: &mg,
        couplings: &[],
        bodies: &[],
        coupling_scale: 0.0,
    };

    let mut p = vec![0.0; rhs.len()];
    let stats = pcg(&system, &rhs, &mut p, 1e-12, 200, true);
    assert!(
        stats.residual <= 1e-12 * 6.0,
        "pcg did not converge: {stats:?}"
    );

    let active = &mg.levels[0].active;

    p.iter()
        .zip(&exact)
        .zip(active)
        .filter(|(_, a)| **a)
        .fold(0.0f64, |m, ((p, e), _)| m.max((p - e).abs()))
}

#[test]
fn ghost_fluid_dirichlet_converges_second_order() {
    let errors: Vec<f64> = [16, 32, 64].iter().map(|&n| solve_error(n)).collect();
    let order_1 = (errors[0] / errors[1]).log2();
    let order_2 = (errors[1] / errors[2]).log2();

    println!("errors {errors:?} orders {order_1:.3} {order_2:.3}");

    assert!(order_2 > 1.7, "observed order {order_2}");
    assert!(errors[2] < 1e-3);
}

#[test]
fn multigrid_preconditioner_is_symmetric() {
    let (_, level, _, _) = sphere_problem(32);
    let mg = Multigrid::new(level, 4);
    let active = mg.levels[0].active.clone();

    let random = |salt: u32| -> Vec<f64> {
        (0..active.len())
            .map(|i| {
                if active[i] {
                    hash_unit(i as u32 ^ salt) - 0.5
                } else {
                    0.0
                }
            })
            .collect()
    };

    let x = random(17);
    let y = random(9001);
    let mx = mg.precondition(&x);
    let my = mg.precondition(&y);

    let a = dot(&mx, &y);
    let b = dot(&x, &my);

    println!("<Mx,y> = {a:e}, <x,My> = {b:e}");

    assert!((a - b).abs() <= 1e-10 * a.abs().max(b.abs()));
    assert!(dot(&mx, &x) > 0.0);
}

#[test]
fn multigrid_cuts_iterations() {
    let (_, level, _, rhs) = sphere_problem(64);
    let mg = Multigrid::new(level, 4);
    let system = PressureSystem {
        multigrid: &mg,
        couplings: &[],
        bodies: &[],
        coupling_scale: 0.0,
    };

    let mut p = vec![0.0; rhs.len()];
    let with_mg = pcg(&system, &rhs, &mut p, 1e-8, 500, true);

    let mut p = vec![0.0; rhs.len()];
    let plain = pcg(&system, &rhs, &mut p, 1e-8, 2000, false);

    println!(
        "mgpcg {} iterations, cg {} iterations",
        with_mg.iterations, plain.iterations
    );

    assert!(with_mg.iterations <= 20);
    assert!(with_mg.iterations * 5 < plain.iterations);
}

fn random_on(active: &[bool], salt: u32) -> Vec<f64> {
    (0..active.len())
        .map(|i| {
            if active[i] {
                hash_unit(i as u32 ^ salt) - 0.5
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn boundary_smoothing_keeps_preconditioner_symmetric() {
    let (_, level, _, _) = sphere_problem(32);
    let mg = Multigrid::new(level, 4).with_boundary_smoothing(2, 2);
    let active = mg.levels[0].active.clone();
    let band = mg.bands[0].iter().filter(|b| **b).count();
    let interior = active.iter().filter(|a| **a).count() - band;

    let x = random_on(&active, 17);
    let y = random_on(&active, 9001);
    let a = dot(&mg.precondition(&x), &y);
    let b = dot(&x, &mg.precondition(&y));
    let relative = (a - b).abs() / a.abs().max(b.abs());

    println!("band cells {band}, interior cells {interior}; <Mx,y> = {a:e}, <x,My> = {b:e}, relative gap {relative:.3e}");

    assert!(band > 0 && interior > 0);
    assert!(relative <= 1e-12);
    assert!(dot(&mg.precondition(&x), &x) > 0.0);
}

fn scene_iterations(scene: impl Fn(Params) -> Simulation, warm: usize, width: u32, sweeps: usize) -> [usize; 2] {
    let mut plain = scene(Params::default());
    let mut banded = scene(Params::default());

    for sim in [&mut plain, &mut banded] {
        for _ in 0..warm {
            let dt = sim.cfl_dt();
            sim.substep(dt);
        }
    }

    banded.params.boundary_width = width;
    banded.params.boundary_sweeps = sweeps;

    let dt = plain.cfl_dt();
    plain.substep(dt);
    banded.substep(dt);

    [plain.report.pressure.iterations, banded.report.pressure.iterations]
}

fn dam_break_scene(params: Params) -> Simulation {
    dam_break(UVec3::new(48, 32, 16), 0.01, DVec3::new(0.16, 0.24, 0.0), params)
}

fn sphere_in_tank(params: Params) -> Simulation {
    let body = RigidBody::new(Shape::Sphere { radius: 0.1 }, DVec3::new(0.25, 0.15, 0.25), Motion::Static, 1000.0);

    with_body(tank(UVec3::new(20, 24, 20), 0.025, 0.3, params), body)
}

#[test]
fn boundary_smoothing_cuts_pcg_iterations() {
    for (name, warm, scene) in [
        ("dam break", 20, dam_break_scene as fn(Params) -> Simulation),
        ("sphere in tank", 5, sphere_in_tank),
    ] {
        let [plain, banded] = scene_iterations(scene, warm, 2, 2);

        println!("{name}: pcg iterations to 1e-8 without boundary smoothing {plain}, with width 2 / 2 sweeps {banded}");

        assert!(banded <= plain, "{name}: {banded} > {plain}");
    }
}

fn column_surface(sim: &Simulation, i: u32, k: u32) -> f64 {
    let phi = |j: u32| sim.phi.get(i, j, k);

    (1..sim.grid.dims.y)
        .find(|&j| phi(j) >= 0.0)
        .map(|j| (j as f64 - 0.5 + phi(j - 1) / (phi(j - 1) - phi(j))) * sim.grid.dx)
        .unwrap_or(sim.grid.size().y)
}

fn mean_surface(sim: &Simulation) -> f64 {
    let d = sim.grid.dims;
    let total: f64 = (0..d.z).flat_map(|k| (0..d.x).map(move |i| (i, k))).map(|(i, k)| column_surface(sim, i, k)).sum();

    total / (d.x * d.z) as f64
}

struct Rest {
    speed: f64,
    local: f64,
    mean: f64,
    drift: f64,
    particles: [usize; 2],
    divergence: f64,
}

fn rest(open_floor: Option<f64>, dx: f64, depth: f64) -> Rest {
    let params = Params { open_floor, ..Params::default() };
    let mut sim = tank(UVec3::new(16, 24, 16), dx, depth, params);
    let count = sim.particles.len();
    let start = mean_surface(&sim);

    sim.advance(0.5);

    let d = sim.grid.dims;
    let surface = mean_surface(&sim);
    let hydrostatic = |eta: f64, y: f64| 1000.0 * 9.81 * (eta - y);
    let (mut local, mut mean) = (0.0f64, 0.0f64);

    for k in 0..d.z {
        for i in 0..d.x {
            let eta = column_surface(&sim, i, k);

            for j in 0..d.y {
                let n = sim.phi.index(i, j, k);
                let y = (j as f64 + 0.5) * dx;

                if sim.phi.data[n] < 0.0 && y < surface - 2.0 * dx {
                    local = local.max((sim.pressure[n] - hydrostatic(eta, y)).abs());
                    mean = mean.max((sim.pressure[n] - hydrostatic(surface, y)).abs());
                }
            }
        }
    }

    Rest {
        speed: sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length())),
        local,
        mean,
        drift: surface - start,
        particles: [count, sim.particles.len()],
        divergence: sim.report.max_divergence,
    }
}

#[test]
fn open_floor_keeps_deep_water_at_rest() {
    let dx: f64 = 1.0 / 32.0;
    let depth = 12.0 * dx;
    let wave_speed = (9.81 * depth).sqrt();
    let scale = 1000.0 * 9.81 * depth;
    let [closed, open] = std::thread::scope(|s| {
        [None, Some(3.0)]
            .map(|floor| s.spawn(move || rest(floor, dx, depth)))
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
    });

    for (name, r) in [("closed floor", &closed), ("open floor", &open)] {
        println!(
            "{name} at rest after 0.5 s: max speed {:.3e} m/s ({:.3e} of sqrt(gh)), max |p - rho g (eta_column - y)| {:.3e} of rho g h, max |p - rho g (eta_mean - y)| {:.3e} of rho g h, surface drift {:+.3e} m, particles {:?}, max divergence {:.3e}",
            r.speed,
            r.speed / wave_speed,
            r.local / scale,
            r.mean / scale,
            r.drift,
            r.particles,
            r.divergence
        );
    }

    assert!(open.speed < 0.01 * wave_speed);
    assert!(open.mean < 2e-3 * scale);
    assert!(open.drift.abs() < 0.1 * dx);
    assert!(open.divergence < 1e-4);
}

struct DeepWave {
    period: f64,
    particles: [usize; 2],
}

fn deep_wave(open_floor: Option<f64>, full_depth: f64) -> DeepWave {
    let dx = 1.0 / 32.0;
    let depth = 10.0 * dx;
    let grid = Grid::new(UVec3::new(32, 16, 4), dx);
    let k = std::f64::consts::PI / grid.size().x;
    let omega = (9.81 * k * (k * full_depth).tanh()).sqrt();
    let amplitude = 0.5 * dx;
    let a = amplitude * 9.81 * k / (omega * (k * full_depth).cosh());
    let mut water = grid.size();
    water.y = depth;

    let particles = moving(seed_box(&grid, DVec3::ZERO, water, SEED_JITTER), |x| {
        let y = x.y - depth + full_depth;

        DVec3::new(-a * (k * y).cosh() * (k * x.x).sin(), a * (k * y).sinh() * (k * x.x).cos(), 0.0)
    });
    let params = Params { open_floor, ..Params::default() };
    let mut sim = Simulation::new(grid, params, particles, Vec::new());
    let start = sim.particles.len();
    let mode = |sim: &Simulation| sim.particles.iter().map(|p| p.v.x * (k * p.x.x).sin()).sum::<f64>();
    let frame = 2.0 * std::f64::consts::PI / omega / 40.0;
    let mut samples = vec![(0.0, mode(&sim))];

    for _ in 0..100 {
        sim.advance(frame);
        samples.push((sim.time, mode(&sim)));
    }

    let crossings: Vec<f64> = samples
        .windows(2)
        .filter(|w| w[0].1.signum() != w[1].1.signum())
        .map(|w| w[0].0 + (w[1].0 - w[0].0) * w[0].1 / (w[0].1 - w[1].1))
        .collect();
    let spans = crossings.len() - 1;

    DeepWave { period: 2.0 * (crossings[spans] - crossings[0]) / spans as f64, particles: [start, sim.particles.len()] }
}

#[test]
fn open_floor_standing_wave_follows_full_depth_dispersion() {
    let full_depth = 1.0;
    let dx = 1.0 / 32.0;
    let depth = 10.0 * dx;
    let k = std::f64::consts::PI;
    let period = |d: f64| 2.0 * std::f64::consts::PI / (9.81 * k * (k * d).tanh()).sqrt();
    let [closed, open, column] = std::thread::scope(|s| {
        [None, Some(3.0), Some(0.0)]
            .map(|floor| s.spawn(move || deep_wave(floor, full_depth)))
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
    });
    let exact = period(full_depth);

    println!(
        "standing wave k = pi/m, domain depth {depth:.4} m of full depth {full_depth} m: analytic full depth {exact:.4} s, analytic closed floor {:.4} s, analytic column-local hydrostatic {:.4} s",
        period(depth),
        2.0 * std::f64::consts::PI / (9.81 * k * (0.5 * k * depth).tanh()).sqrt()
    );
    println!(
        "measured: closed floor {:.4} s ({:+.2}%), open floor {:.4} s ({:+.2}%), column-local open floor {:.4} s ({:+.2}%), open floor particles {:?}",
        closed.period,
        100.0 * (closed.period / exact - 1.0),
        open.period,
        100.0 * (open.period / exact - 1.0),
        column.period,
        100.0 * (column.period / exact - 1.0),
        open.particles
    );

    assert!((open.period - exact).abs() < (closed.period - exact).abs());
    assert!((open.period / exact - 1.0).abs() < 0.06);
}
