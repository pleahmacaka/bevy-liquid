use glam::{DVec3, UVec3};
use liquid_ref::grid::Grid;
use liquid_ref::particles::{Particle, seed_box};
use liquid_ref::scene::SEED_JITTER;
use liquid_ref::sim::{Params, Simulation};

const RADIUS: f64 = 0.01;

const SIGMA: f64 = 0.07;

const MU: f64 = 5.0;

fn droplet(cells_per_radius: u32, stretch: f64) -> Simulation {
    let dx = RADIUS / cells_per_radius as f64;
    let grid = Grid::new(UVec3::splat(2 * cells_per_radius + 8), dx);
    let center = grid.size() * 0.5;
    let axes = DVec3::new(stretch, 1.0 / stretch.sqrt(), 1.0 / stretch.sqrt()) * RADIUS;

    let particles: Vec<Particle> = seed_box(&grid, DVec3::ZERO, grid.size(), SEED_JITTER)
        .into_iter()
        .filter(|p| ((p.x - center) / axes).length_squared() < 1.0)
        .collect();
    let params = Params {
        gravity: DVec3::ZERO,
        surface_tension: SIGMA,
        viscosity: MU,
        ..Params::default()
    };

    Simulation::new(grid, params, particles, Vec::new())
}

fn equivalent_radius(sim: &Simulation) -> f64 {
    let volume = sim.particles.len() as f64 * sim.grid.dx.powi(3) / 8.0;

    (3.0 * volume / (4.0 * std::f64::consts::PI)).cbrt()
}

fn core_pressure(sim: &Simulation, radius: f64) -> f64 {
    let center = sim.grid.size() * 0.5;
    let cells = sim.grid.cells();
    let mut sum = 0.0;
    let mut count = 0.0;

    for n in 0..cells.data.len() {
        if (sim.grid.cell_center(cells.coords(n)) - center).length() < 0.5 * radius {
            sum += sim.pressure[n];
            count += 1.0;
        }
    }

    sum / count
}

fn max_speed(sim: &Simulation) -> f64 {
    sim.particles
        .iter()
        .fold(0.0f64, |m, p| m.max(p.v.length()))
}

fn axis_moment(sim: &Simulation) -> f64 {
    let n = sim.particles.len() as f64;
    let center: DVec3 = sim.particles.iter().map(|p| p.x).sum::<DVec3>() / n;

    sim.particles
        .iter()
        .map(|p| {
            let r = p.x - center;
            r.x * r.x - 0.5 * (r.y * r.y + r.z * r.z)
        })
        .sum::<f64>()
        / n
}

#[test]
fn viscous_static_droplet_holds_laplace_pressure_in_one_solve() {
    let mut sim = droplet(8, 1.0);
    let radius = equivalent_radius(&sim);
    let laplace = 2.0 * SIGMA / radius;
    let mut pressure = 0.0;
    let mut speed = 0.0f64;

    for n in 0..20 {
        let dt = sim.cfl_dt();
        sim.substep(dt);

        assert!(sim.report.viscosity.is_some());
        assert!(sim.report.capillary.is_none());

        if n >= 10 {
            pressure += core_pressure(&sim, radius) / 10.0;
            speed = speed.max(max_speed(&sim));
        }
    }

    let error = (pressure - laplace).abs() / laplace;
    let capillary_number = MU * speed / SIGMA;

    println!(
        "radius 8 cells, mu {MU} Pa s: core pressure {pressure:.3} Pa vs 2 sigma/R {laplace:.3} Pa, error {:.2}%, max spurious speed {speed:.2e} m/s (capillary number {capillary_number:.2e})",
        100.0 * error
    );

    assert!(error < 0.05);
    assert!(speed < 1e-3);
}

struct Relaxation {
    rate: f64,

    exact: f64,

    monotone: bool,

    drift: f64,

    substeps: usize,
}

fn relax(cells_per_radius: u32) -> Relaxation {
    let mut sim = droplet(cells_per_radius, 1.15);
    let exact = 20.0 * SIGMA / (19.0 * MU * equivalent_radius(&sim));
    let v0 = sim.liquid_volume();
    let mut samples = vec![(0.0, axis_moment(&sim))];
    let mut drift = 0.0f64;

    while sim.time < 0.15 / exact {
        let dt = sim.cfl_dt();
        sim.substep(dt);

        samples.push((sim.time, axis_moment(&sim)));
        drift = drift.max((sim.liquid_volume() / v0 - 1.0).abs());
    }

    let start = samples
        .iter()
        .position(|(t, _)| *t > 0.05 / exact)
        .unwrap_or(0);
    let (t0, m0) = samples[start];
    let (t1, m1) = samples[samples.len() - 1];

    Relaxation {
        rate: (m0 / m1).ln() / (t1 - t0),
        exact,
        monotone: samples.windows(2).all(|w| w[1].1 < w[0].1 && w[1].1 > 0.0),
        drift,
        substeps: samples.len() - 1,
    }
}

#[test]
fn stretched_viscous_droplet_relaxes_at_creeping_flow_rate() {
    let runs: Vec<(u32, Relaxation)> = std::thread::scope(|s| {
        let handles: Vec<_> = [6, 8]
            .into_iter()
            .map(|c| s.spawn(move || (c, relax(c))))
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| unreachable!()))
            .collect()
    });
    let errors: Vec<f64> = runs
        .iter()
        .map(|(_, run)| (run.rate - run.exact).abs() / run.exact)
        .collect();

    for ((c, run), e) in runs.iter().zip(&errors) {
        println!(
            "radius {c} cells, mu {MU} Pa s: relaxation rate {:.4} 1/s over {} substeps vs 20 sigma/(19 mu R) {:.4} 1/s, error {:.2}%, moment monotone {}, worst volume drift {:.4}",
            run.rate,
            run.substeps,
            run.exact,
            100.0 * e,
            run.monotone,
            run.drift
        );
    }

    assert!(runs.iter().all(|(_, run)| run.monotone));
    assert!(runs.iter().all(|(_, run)| run.drift < 0.01));
    assert!(errors[1] < errors[0]);
    assert!(errors[1] < 0.2);
}
