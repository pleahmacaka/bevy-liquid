use glam::{DVec3, UVec3};
use liquid_ref::grid::Grid;
use liquid_ref::particles::{Particle, seed_box};
use liquid_ref::scene::SEED_JITTER;
use liquid_ref::sim::{Params, Simulation};

const RADIUS: f64 = 0.01;

const SIGMA: f64 = 0.07;

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

fn laplace_error(cells_per_radius: u32) -> (f64, f64, f64) {
    let mut sim = droplet(cells_per_radius, 1.0);
    let radius = equivalent_radius(&sim);
    let mut pressure = 0.0;
    let mut speed = 0.0f64;

    for n in 0..20 {
        let dt = sim.cfl_dt();
        sim.substep(dt);

        if n >= 10 {
            pressure += core_pressure(&sim, radius) / 10.0;
            speed = sim.particles.iter().fold(speed, |m, p| m.max(p.v.length()));
        }
    }

    let laplace = 2.0 * SIGMA / radius;

    (pressure, laplace, speed)
}

#[test]
fn static_droplet_pressure_matches_laplace() {
    let results: Vec<(u32, f64, f64, f64)> = [4, 8, 16]
        .iter()
        .map(|&c| {
            let (p, laplace, speed) = laplace_error(c);
            (c, p, laplace, speed)
        })
        .collect();
    let errors: Vec<f64> = results
        .iter()
        .map(|(_, p, l, _)| (p - l).abs() / l)
        .collect();

    for ((c, p, l, speed), e) in results.iter().zip(&errors) {
        println!(
            "radius {c} cells: core pressure {p:.3} Pa vs 2 sigma/R {l:.3} Pa, error {:.2}%, max spurious speed {speed:.2e} m/s",
            100.0 * e
        );
    }

    assert!(errors[2] < errors[0]);
    assert!(errors[2] < 0.05);
}

fn rayleigh_period(sim: &Simulation) -> f64 {
    let omega = (8.0 * SIGMA / (1000.0 * equivalent_radius(sim).powi(3))).sqrt();

    2.0 * std::f64::consts::PI / omega
}

fn max_speed(sim: &Simulation) -> f64 {
    sim.particles
        .iter()
        .fold(0.0f64, |m, p| m.max(p.v.length()))
}

struct Stillness {
    startup: f64,

    settled: f64,

    drift: f64,
}

fn still_droplet(cells_per_radius: u32) -> Stillness {
    let mut sim = droplet(cells_per_radius, 1.0);
    let duration = 3.0 * rayleigh_period(&sim);
    let settle = 0.5 * rayleigh_period(&sim);
    let v0 = sim.liquid_volume();
    let mut still = Stillness {
        startup: 0.0,
        settled: 0.0,
        drift: 0.0,
    };

    while sim.time < duration {
        let dt = sim.cfl_dt();
        sim.substep(dt);

        let speed = max_speed(&sim);

        if sim.time < settle {
            still.startup = still.startup.max(speed);
        } else {
            still.settled = still.settled.max(speed);
        }

        still.drift = still.drift.max((sim.liquid_volume() / v0 - 1.0).abs());
    }

    still
}

#[test]
fn static_droplet_stays_still_for_three_periods() {
    let runs: Vec<(u32, Stillness)> = std::thread::scope(|s| {
        let handles: Vec<_> = [12, 16]
            .into_iter()
            .map(|c| s.spawn(move || (c, still_droplet(c))))
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| unreachable!()))
            .collect()
    });

    for (c, run) in &runs {
        println!(
            "radius {c} cells, 3 Rayleigh periods: max speed {:.4} m/s in the first half period, {:.4} m/s afterwards, worst volume drift {:.4}",
            run.startup, run.settled, run.drift
        );
    }

    for (_, run) in &runs {
        assert!(run.settled < 0.006);
        assert!(run.drift < 0.01);
    }
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

fn measured_period(cells_per_radius: u32) -> (f64, f64, usize) {
    let mut sim = droplet(cells_per_radius, 1.15);
    let period = rayleigh_period(&sim);

    let mut crossings = Vec::new();
    let mut previous = (0.0, axis_moment(&sim));

    while sim.time < 3.0 * period {
        let dt = sim.cfl_dt();
        sim.substep(dt);

        let m = axis_moment(&sim);

        if (m < 0.0) != (previous.1 < 0.0) {
            let t = previous.0 + (sim.time - previous.0) * previous.1 / (previous.1 - m);
            crossings.push(t);
        }

        previous = (sim.time, m);
    }

    let spacing = (crossings[crossings.len() - 1] - crossings[0]) / (crossings.len() - 1) as f64;

    (2.0 * spacing, period, crossings.len())
}

#[test]
fn oscillating_droplet_follows_rayleigh_period() {
    let runs: Vec<(u32, (f64, f64, usize))> = std::thread::scope(|s| {
        let handles: Vec<_> = [8, 12]
            .into_iter()
            .map(|c| s.spawn(move || (c, measured_period(c))))
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|_| unreachable!()))
            .collect()
    });

    let errors: Vec<f64> = runs
        .iter()
        .map(|(_, (measured, period, _))| (measured - period).abs() / period)
        .collect();

    for ((c, (measured, period, count)), e) in runs.iter().zip(&errors) {
        println!(
            "radius {c} cells: period {measured:.4} s from {count} zero crossings vs Rayleigh {period:.4} s, error {:.2}%",
            100.0 * e
        );
    }

    assert!(runs.iter().all(|(_, (_, _, count))| *count >= 4));
    assert!(errors[1] < errors[0]);
    assert!(errors[1] < 0.05);
}
