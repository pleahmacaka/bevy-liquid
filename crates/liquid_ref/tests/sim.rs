use glam::{DVec3, UVec3};
use liquid_ref::grid::Grid;
use liquid_ref::narrow_band::NarrowBand;
use liquid_ref::particles::{Particle, seed_box};
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{SEED_JITTER, dam_break, moving, tank, with_body};
use liquid_ref::sim::{Params, Simulation};
use liquid_ref::transfer::Transfer;

fn max_speed(sim: &Simulation) -> f64 {
    sim.particles.iter().fold(0.0, |m, p| m.max(p.v.length()))
}

#[test]
fn hydrostatic_tank_stays_at_rest() {
    let dx = 1.0 / 32.0;
    let depth = 12.0 * dx;
    let mut sim = tank(UVec3::new(16, 24, 16), dx, depth, Params::default());

    let v0 = sim.liquid_volume();
    sim.advance(0.3);

    let mut bottom = 0.0;
    let frames = 10;

    for _ in 0..frames {
        sim.advance(0.02);
        bottom += mean_bottom_pressure(&sim) / frames as f64;
    }

    let v1 = sim.liquid_volume();
    let wave_speed = (9.81 * depth).sqrt();
    let speed = max_speed(&sim);
    let expected = 1000.0 * 9.81 * (depth - 0.5 * dx);
    let drift = (v1 - v0).abs() / v0;

    println!(
        "max speed {speed:.3e} m/s ({:.3e} of sqrt(gh)), bottom p {bottom:.1} vs {expected:.1}, volume drift {drift:.3e}",
        speed / wave_speed
    );

    assert!(speed < 0.01 * wave_speed);
    assert!((bottom - expected).abs() < 0.01 * expected);
    assert!(drift < 0.005);
    assert!(sim.report.max_divergence < 1e-4);
}

fn mean_bottom_pressure(sim: &Simulation) -> f64 {
    let cells = sim.grid.cells();
    let dims = sim.grid.dims;
    let mut sum = 0.0;

    for k in 0..dims.z {
        for i in 0..dims.x {
            sum += sim.pressure[cells.index(i, 0, k)];
        }
    }

    sum / (dims.x * dims.z) as f64
}

fn front_position(sim: &Simulation, a: f64) -> f64 {
    let floor = 0.25 * a;

    sim.particles
        .iter()
        .filter(|p| p.x.y < floor)
        .fold(0.0, |m, p| m.max(p.x.x))
}

#[test]
fn dam_break_front_matches_martin_moyce_slope() {
    let a = 0.057;
    let cells = 16;
    let dx = a / cells as f64;
    let column = DVec3::new(a, 2.0 * a, 0.0);
    let mut sim = dam_break(
        UVec3::new(5 * cells, 3 * cells, 4),
        dx,
        column,
        Params::default(),
    );

    let t_scale = (a / (2.0 * 9.81)).sqrt();
    let mut samples = Vec::new();

    for n in 1..=40 {
        sim.advance(0.1 * t_scale);

        let z = front_position(&sim, a) / a;
        samples.push((0.1 * n as f64, z));
    }

    let window: Vec<(f64, f64)> = samples
        .iter()
        .copied()
        .filter(|(_, z)| (1.5..=3.5).contains(z))
        .collect();
    let n = window.len() as f64;
    let (st, sz) = window
        .iter()
        .fold((0.0, 0.0), |(a, b), (t, z)| (a + t, b + z));
    let (mt, mz) = (st / n, sz / n);
    let num: f64 = window.iter().map(|(t, z)| (t - mt) * (z - mz)).sum();
    let den: f64 = window.iter().map(|(t, _)| (t - mt).powi(2)).sum();
    let slope = num / den;

    println!("samples {samples:?}");
    println!(
        "front slope dZ/dT = {slope:.3} (Martin & Moyce 1952: 1.29), {} points",
        window.len()
    );

    assert!(window.len() >= 5);
    assert!((slope - 1.29).abs() < 0.15 * 1.29);
}

#[test]
fn identical_runs_are_bitwise_identical() {
    let run = || {
        let dx = 0.02;
        let mut sim = dam_break(
            UVec3::new(24, 16, 8),
            dx,
            DVec3::new(0.16, 0.24, 0.0),
            Params::default(),
        );

        for _ in 0..5 {
            let dt = sim.cfl_dt();
            sim.substep(dt);
        }

        sim.particles
    };

    let a = run();
    let b = run();

    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(&b).all(|(p, q)| p == q));
}

#[test]
fn translating_blob_keeps_momentum_in_zero_gravity() {
    let grid = Grid::new(UVec3::splat(24), 1.0 / 24.0);
    let v0 = DVec3::new(0.3, -0.2, 0.1);
    let particles = moving(
        seed_box(&grid, DVec3::splat(0.3), DVec3::splat(0.6), 0.8),
        |_| v0,
    );
    let params = Params {
        gravity: DVec3::ZERO,
        ..Params::default()
    };
    let mut sim = Simulation::new(grid, params, particles, Vec::new());

    let p0 = sim.momentum();
    let e0 = sim.kinetic_energy();
    sim.advance(0.2);

    let p1 = sim.momentum();
    let e1 = sim.kinetic_energy();

    println!("momentum {p0} -> {p1}, energy {e0:.6e} -> {e1:.6e}");

    assert!((p1 - p0).length() < 1e-6 * p0.length());
    assert!(e1 <= e0 * (1.0 + 1e-6));
}

fn taylor_green_energy_ratio(transfer: Transfer, n: u32) -> f64 {
    let grid = Grid::new(UVec3::new(n, n, 2), 1.0 / n as f64);
    let pi = std::f64::consts::PI;
    let field = |x: DVec3| {
        DVec3::new(
            (pi * x.x).sin() * (pi * x.y).cos(),
            -(pi * x.x).cos() * (pi * x.y).sin(),
            0.0,
        )
    };
    let particles = moving(seed_box(&grid, DVec3::ZERO, grid.size(), 0.8), field);
    let params = Params {
        gravity: DVec3::ZERO,
        transfer,
        density_projection: false,
        ..Params::default()
    };
    let mut sim = Simulation::new(grid, params, particles, Vec::new());

    let e0 = sim.kinetic_energy();
    sim.advance(1.0);

    sim.kinetic_energy() / e0
}

#[test]
fn apic_keeps_taylor_green_energy_better_than_pic() {
    let apic_32 = taylor_green_energy_ratio(Transfer::Apic, 32);
    let apic_64 = taylor_green_energy_ratio(Transfer::Apic, 64);
    let pic_32 = taylor_green_energy_ratio(Transfer::Pic, 32);

    println!("energy kept after 1 s: apic 32 {apic_32:.4}, apic 64 {apic_64:.4}, pic 32 {pic_32:.4}");

    assert!(apic_64 <= 1.0 + 1e-6);
    assert!(apic_32 > 0.85);
    assert!(pic_32 < apic_32 - 0.2);
}

fn floating_sphere(density: f64, start_depth: f64) -> Simulation {
    let dx = 0.025;
    let water = 0.3;
    let radius = 0.1;
    let params = Params::default();
    let mut body = RigidBody::new(
        Shape::Sphere { radius },
        DVec3::new(0.25, water - start_depth, 0.25),
        Motion::Dynamic,
        density,
    );
    body.lock_rotation = true;

    with_body(tank(UVec3::new(20, 24, 20), dx, water, params), body)
}

#[test]
fn half_density_sphere_floats_half_submerged() {
    let mut sim = floating_sphere(500.0, 0.0);
    sim.advance(1.5);

    let mut heights = Vec::new();

    for _ in 0..30 {
        sim.advance(1.0 / 30.0);
        heights.push(sim.bodies[0].position.y);
    }

    let mean = heights.iter().sum::<f64>() / heights.len() as f64;
    let surface = surface_height(&sim);
    let offset = (mean - surface) / 0.1;

    println!("sphere center {mean:.4}, surface {surface:.4}, offset/radius {offset:.3}");

    assert!(offset.abs() < 0.15);
}

fn surface_height(sim: &Simulation) -> f64 {
    let grid = sim.grid;
    let column = |i: u32, k: u32| {
        (0..grid.dims.y)
            .rev()
            .find_map(|j| {
                let below = sim.phi.get(i, j, k);
                let above = if j + 1 < grid.dims.y {
                    sim.phi.get(i, j + 1, k)
                } else {
                    grid.dx
                };

                (below < 0.0 && above >= 0.0).then(|| {
                    let t = below / (below - above);
                    (j as f64 + 0.5 + t) * grid.dx
                })
            })
            .unwrap_or(0.0)
    };

    (column(2, 2) + column(17, 17) + column(2, 17) + column(17, 2)) / 4.0
}

fn first_step_acceleration(cells_per_radius: u32) -> f64 {
    let dx = 0.1 / cells_per_radius as f64;
    let n = |m: f64| (m / dx).round() as u32;
    let mut body = RigidBody::new(
        Shape::Sphere { radius: 0.1 },
        DVec3::new(0.3, 0.275, 0.3),
        Motion::Dynamic,
        100.0,
    );
    body.lock_rotation = true;

    let mut sim = with_body(
        tank(UVec3::new(n(0.6), n(0.7), n(0.6)), dx, 0.55, Params::default()),
        body,
    );
    let dt = 0.002;
    sim.substep(dt);

    assert!(sim.bodies[0].velocity.is_finite());

    sim.bodies[0].velocity.y / dt
}

#[test]
fn light_sphere_accelerates_with_added_mass() {
    let theory = 9.81 * (1000.0 - 100.0) / (100.0 + 0.5 * 1000.0);
    let coarse = first_step_acceleration(4);
    let fine = first_step_acceleration(8);

    println!("acceleration r=4 cells {coarse:.2}, r=8 cells {fine:.2}, added-mass theory {theory:.2}, without added mass {:.2}", 9.81 * 9.0);

    assert!((fine - theory).abs() < (coarse - theory).abs());
    assert!((fine - theory).abs() < 0.03 * theory);
}

fn dam_break_volume_ratio(density_projection: bool) -> f64 {
    let params = Params {
        density_projection,
        ..Params::default()
    };
    let mut sim = dam_break(UVec3::new(48, 32, 4), 0.01, DVec3::new(0.16, 0.24, 0.0), params);

    let v0 = sim.liquid_volume();
    sim.advance(1.5);

    sim.liquid_volume() / v0
}

#[test]
fn density_projection_preserves_volume_through_splashing() {
    let with_idp = dam_break_volume_ratio(true);
    let without = dam_break_volume_ratio(false);

    println!("volume after 1.5 s: with IDP {with_idp:.4}, without {without:.4}");

    assert!((with_idp - 1.0).abs() < 0.05);
    assert!((without - 1.0).abs() > 3.0 * (with_idp - 1.0).abs());
}

struct Sloshing {
    kh: f64,

    cells: u32,

    measured: f64,

    exact: f64,
}

impl Sloshing {
    fn error(&self) -> f64 {
        self.measured / self.exact - 1.0
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

fn sloshing(kh: f64, cells: u32) -> Sloshing {
    let depth = 0.1;
    let dx = depth / cells as f64;
    let length = (std::f64::consts::PI * cells as f64 / kh).round() as u32;
    let k = std::f64::consts::PI / (length as f64 * dx);
    let exact = (9.81 * k * (k * depth).tanh()).sqrt();
    let speed = 0.05 * depth * exact / (k * depth).sinh();
    let grid = Grid::new(UVec3::new(length, cells + 4, 4), dx);
    let mut water = grid.size();
    water.y = depth;

    let particles = moving(seed_box(&grid, DVec3::ZERO, water, SEED_JITTER), |x| {
        DVec3::new(
            -speed * (k * x.y).cosh() * (k * x.x).sin(),
            speed * (k * x.y).sinh() * (k * x.x).cos(),
            0.0,
        )
    });
    let params = Params {
        narrow_band: Some(NarrowBand::default()),
        ..Params::default()
    };
    let mut sim = Simulation::new(grid, params, particles, Vec::new());

    let mode = |sim: &Simulation| {
        sim.particles
            .iter()
            .map(|p| p.v.x * (k * p.x.x).sin())
            .sum::<f64>()
    };
    let frame = 2.0 * std::f64::consts::PI / exact / 40.0;
    let mut samples = vec![(0.0, mode(&sim))];

    for _ in 0..128 {
        sim.advance(frame);
        samples.push((sim.time, mode(&sim)));
    }

    Sloshing {
        kh: k * depth,
        cells,
        measured: zero_crossing_omega(&samples),
        exact,
    }
}

#[test]
fn standing_wave_frequency_follows_airy_dispersion() {
    let cases = [
        (0.3, 8),
        (0.3, 16),
        (1.0, 8),
        (1.0, 16),
        (3.0, 8),
        (3.0, 16),
    ];
    let runs: Vec<Sloshing> = std::thread::scope(|s| {
        let handles: Vec<_> = cases
            .map(|(kh, cells)| s.spawn(move || sloshing(kh, cells)))
            .into_iter()
            .collect();

        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    });

    for r in &runs {
        println!(
            "kh {:.4}, {:2} cells per depth: omega {:.5} rad/s, Airy {:.5} rad/s, error {:+.2}%",
            r.kh,
            r.cells,
            r.measured,
            r.exact,
            100.0 * r.error()
        );
    }

    for r in &runs {
        let tolerance = 0.4 / r.cells as f64;

        assert!(
            r.error().abs() < tolerance,
            "kh {:.3} at {} cells per depth: error {:+.2}% exceeds {:.1}%",
            r.kh,
            r.cells,
            100.0 * r.error(),
            100.0 * tolerance
        );
    }
}

fn floor_box() -> RigidBody {
    RigidBody::new(
        Shape::Cuboid { half: DVec3::new(0.0605, 0.0505, 0.0805) },
        DVec3::new(0.703, 0.0505, 0.151),
        Motion::Static,
        1000.0,
    )
}

fn dam_break_with_floor_box() -> Simulation {
    with_body(
        dam_break(UVec3::new(50, 20, 15), 0.02, DVec3::new(0.3, 0.3, 0.0), Params::default()),
        floor_box(),
    )
}

fn worst_clearance(sim: &Simulation) -> f64 {
    sim.particles
        .iter()
        .map(|p| sim.bodies[0].sdf(p.x))
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn particles_trapped_in_a_floor_box_leave_through_free_faces() {
    let mut sim = dam_break_with_floor_box();
    let margin = sim.params.solid_margin_cells * sim.grid.dx;

    for (x, z) in [(0.645, 0.1), (0.655, 0.15), (0.66, 0.2), (0.7, 0.15), (0.76, 0.08)] {
        sim.particles.push(Particle::at_rest(DVec3::new(x, margin, z)));
    }

    let before = worst_clearance(&sim);
    sim.substep(0.004);

    let after = worst_clearance(&sim);
    let size = sim.grid.size();
    let inside = sim
        .particles
        .iter()
        .all(|p| p.x.cmpge(DVec3::splat(margin) * 0.999).all() && p.x.cmple(size - margin * 0.999).all());

    println!("floor box clearance before {before:.4e} m, after one substep {after:.4e} m (margin {margin:.1e} m)");

    assert!(before < 0.0);
    assert!(after >= 0.99 * margin);
    assert!(inside);
}

#[test]
fn dam_break_against_a_floor_box_stays_bounded() {
    let mut sim = dam_break_with_floor_box();
    let margin = sim.params.solid_margin_cells * sim.grid.dx;
    let bound = 4.0 * (2.0 * 9.81 * 0.3f64).sqrt();
    let mut fastest = 0.0f64;
    let mut closest = f64::INFINITY;

    for _ in 0..60 {
        sim.advance(1.0 / 60.0);
        fastest = fastest.max(max_speed(&sim));
        closest = closest.min(worst_clearance(&sim));
    }

    println!("dam break onto a floor box for 1 s: max particle speed {fastest:.3} m/s (bound {bound:.3}), min box clearance {closest:.4e} m (margin {margin:.1e} m)");

    assert!(fastest < bound);
    assert!(closest >= 0.99 * margin);
}

fn floating_offset(density_ratio: f64) -> f64 {
    let (mut lo, mut hi) = (0.0, 2.0);

    for _ in 0..60 {
        let d = 0.5 * (lo + hi);

        if d * d * (3.0 - d) < 4.0 * density_ratio {
            lo = d;
        } else {
            hi = d;
        }
    }

    1.0 - 0.5 * (lo + hi)
}

#[test]
fn tenth_density_sphere_rises_and_floats_stably() {
    let radius = 0.1;
    let mut sim = floating_sphere(100.0, 0.15);
    let mut body_peak = 0.0f64;
    let mut liquid_peak = 0.0f64;
    let mut earlier = Vec::new();
    let mut later = Vec::new();

    while sim.time < 3.0 - 1e-9 {
        sim.advance(1.0 / 30.0);

        let b = &sim.bodies[0];
        assert!(b.position.is_finite() && b.velocity.is_finite(), "diverged at {:.3} s", sim.time);

        body_peak = body_peak.max(b.velocity.length());
        liquid_peak = liquid_peak.max(max_speed(&sim));

        if sim.time > 2.0 {
            later.push(b.position.y);
        } else if sim.time > 1.0 {
            earlier.push(b.position.y);
        }
    }

    let swing = |h: &[f64]| {
        let mean = h.iter().sum::<f64>() / h.len() as f64;
        (mean, h.iter().fold(0.0f64, |m, v| m.max((v - mean).abs())) / radius)
    };
    let (_, earlier_swing) = swing(&earlier);
    let (mean, later_swing) = swing(&later);
    let offset = (mean - surface_height(&sim)) / radius;
    let expected = floating_offset(0.1);

    println!(
        "density ratio 0.1: peak body speed {body_peak:.3} m/s, peak liquid speed {liquid_peak:.3} m/s, centre above surface {offset:.3} r (Archimedes {expected:.3} r), heave swing {earlier_swing:.3} r over 1..2 s, {later_swing:.3} r over 2..3 s"
    );

    assert!(body_peak < 4.0);
    assert!(liquid_peak < 8.0);
    assert!((offset - expected).abs() < 0.15);
    assert!(later_swing < earlier_swing);
}
