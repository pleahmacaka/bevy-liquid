use glam::{DVec3, UVec3};
use liquid_ref::levelset::redistance;
use liquid_ref::narrow_band::NarrowBand;
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::{dam_break, tank, with_body};
use liquid_ref::sim::{Params, Simulation};

fn band_params() -> Params {
    Params {
        narrow_band: Some(NarrowBand::default()),
        ..Params::default()
    }
}

fn front_position(sim: &Simulation, a: f64) -> f64 {
    sim.particles
        .iter()
        .filter(|p| p.x.y < 0.25 * a)
        .fold(0.0, |m, p| m.max(p.x.x))
}

fn fitted_slope(samples: &[(f64, f64)]) -> (f64, usize) {
    let window: Vec<(f64, f64)> = samples
        .iter()
        .copied()
        .filter(|(_, z)| (1.5..=3.5).contains(z))
        .collect();
    let n = window.len() as f64;
    let mt = window.iter().map(|(t, _)| t).sum::<f64>() / n;
    let mz = window.iter().map(|(_, z)| z).sum::<f64>() / n;
    let num: f64 = window.iter().map(|(t, z)| (t - mt) * (z - mz)).sum();
    let den: f64 = window.iter().map(|(t, _)| (t - mt).powi(2)).sum();

    (num / den, window.len())
}

struct FrontRun {
    slope: f64,

    points: usize,

    initial_particles: usize,

    mean_particles: f64,
}

fn martin_moyce(params: Params) -> FrontRun {
    let a = 0.057;
    let cells = 16;
    let dx = a / cells as f64;
    let column = DVec3::new(a, 2.0 * a, 0.0);
    let mut sim = dam_break(UVec3::new(5 * cells, 3 * cells, 4), dx, column, params);

    let initial_particles = sim.particles.len();
    let t_scale = (a / (2.0 * 9.81)).sqrt();
    let mut samples = Vec::new();
    let mut count = 0.0;

    for n in 1..=40 {
        sim.advance(0.1 * t_scale);
        samples.push((0.1 * n as f64, front_position(&sim, a) / a));
        count += sim.particles.len() as f64 / 40.0;
    }

    let (slope, points) = fitted_slope(&samples);

    FrontRun {
        slope,
        points,
        initial_particles,
        mean_particles: count,
    }
}

#[test]
fn narrow_band_dam_break_front_matches_dense() {
    let (band, dense) = std::thread::scope(|s| {
        let dense = s.spawn(|| martin_moyce(Params::default()));
        let band = martin_moyce(band_params());

        (band, dense.join().unwrap_or_else(|_| unreachable!()))
    });

    let ratio = band.mean_particles / dense.mean_particles;

    println!(
        "front slope: narrow band {:.3} ({} points), dense {:.3} ({} points), Martin & Moyce 1.29",
        band.slope, band.points, dense.slope, dense.points
    );
    println!(
        "particles: dense {} constant, narrow band mean {:.0} from {} seeded, ratio {ratio:.3}",
        dense.initial_particles, band.mean_particles, band.initial_particles
    );

    assert!(band.points >= 5);
    assert!((band.slope - dense.slope).abs() < 0.1 * dense.slope);
    assert!((band.slope - 1.29).abs() < 0.15 * 1.29);
    assert!(ratio < 0.6);
}

fn redistanced_volume(sim: &Simulation) -> f64 {
    let distance = redistance(&sim.phi, sim.grid.dx);
    let cells: f64 = distance
        .data
        .iter()
        .map(|phi| (0.5 - phi / sim.grid.dx).clamp(0.0, 1.0))
        .sum();

    cells * sim.grid.dx.powi(3)
}

struct VolumeRun {
    last: f64,

    worst: f64,

    particles: usize,
}

fn volume_run(params: Params) -> VolumeRun {
    let mut sim = dam_break(
        UVec3::new(48, 32, 4),
        0.01,
        DVec3::new(0.16, 0.24, 0.0),
        params,
    );

    let v0 = redistanced_volume(&sim);
    let mut worst = 0.0f64;

    for _ in 0..15 {
        sim.advance(0.1);
        worst = worst.max((redistanced_volume(&sim) / v0 - 1.0).abs());
    }

    VolumeRun {
        last: redistanced_volume(&sim) / v0,
        worst,
        particles: sim.particles.len(),
    }
}

#[test]
fn narrow_band_dam_break_keeps_volume_like_dense() {
    let without_idp = Params {
        density_projection: false,
        ..Params::default()
    };

    let (band, dense, plain) = std::thread::scope(|s| {
        let dense = s.spawn(|| volume_run(Params::default()));
        let plain = s.spawn(|| volume_run(without_idp));
        let band = volume_run(band_params());
        let join = |h: std::thread::ScopedJoinHandle<'_, VolumeRun>| {
            h.join().unwrap_or_else(|_| unreachable!())
        };

        (band, join(dense), join(plain))
    });

    println!(
        "volume after 1.5 s (worst deviation over 0.1 s samples): narrow band {:.4} ({:.4}, {} particles), dense with IDP {:.4} ({:.4}), dense without IDP {:.4} ({:.4}, {} particles)",
        band.last,
        band.worst,
        band.particles,
        dense.last,
        dense.worst,
        plain.last,
        plain.worst,
        plain.particles
    );

    assert!((band.last - 1.0).abs() < 0.03);
    assert!(band.worst < 0.03);
}

#[test]
fn narrow_band_hydrostatic_tank_stays_at_rest() {
    let dx = 1.0 / 32.0;
    let depth = 12.0 * dx;
    let mut sim = tank(UVec3::new(16, 24, 16), dx, depth, band_params());
    let seeded = sim.particles.len();

    let v0 = sim.liquid_volume();
    sim.advance(0.3);

    let mut bottom = 0.0;
    let frames = 10;

    for _ in 0..frames {
        sim.advance(0.02);
        bottom += mean_bottom_pressure(&sim) / frames as f64;
    }

    let wave_speed = (9.81 * depth).sqrt();
    let particle_speed = sim
        .particles
        .iter()
        .fold(0.0f64, |m, p| m.max(p.v.length()));
    let grid_speed = max_liquid_speed(&sim);
    let expected = 1000.0 * 9.81 * (depth - 0.5 * dx);
    let drift = (sim.liquid_volume() - v0).abs() / v0;
    let ratio = sim.particles.len() as f64 / seeded as f64;

    println!(
        "max particle speed {particle_speed:.3e} m/s, max liquid face speed {grid_speed:.3e} m/s ({:.3e} of sqrt(gh)), bottom p {bottom:.1} vs {expected:.1}, volume drift {drift:.3e}, particles {} of {seeded} ({ratio:.3})",
        grid_speed / wave_speed,
        sim.particles.len(),
    );

    assert!(particle_speed < 0.01 * wave_speed);
    assert!(grid_speed < 0.01 * wave_speed);
    assert!((bottom - expected).abs() < 0.01 * expected);
    assert!(drift < 0.005);
    assert!(ratio < 0.35);
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

fn max_liquid_speed(sim: &Simulation) -> f64 {
    let mut worst = 0.0f64;

    for (axis, field) in sim.velocity.axes.iter().enumerate() {
        for n in 0..field.data.len() {
            let f = field.coords(n);
            let x = sim.grid.face_position(axis, f);
            let inside = liquid_ref::grid::sample_cells(&sim.phi, &sim.grid, x) < 0.0;

            if inside {
                worst = worst.max(field.data[n].abs());
            }
        }
    }

    worst
}

struct FloatRun {
    rest: f64,
    peak_speed: f64,
    finite: bool,
}

fn floating_block(params: Params) -> FloatRun {
    let dx = 0.025;
    let body = RigidBody::new(
        Shape::Cuboid {
            half: DVec3::new(0.08, 0.05, 0.1),
        },
        DVec3::new(0.25, 0.36, 0.25),
        Motion::Dynamic,
        450.0,
    );
    let mut sim = with_body(tank(UVec3::new(20, 24, 20), dx, 0.3, params), body);
    let mut peak_speed = 0.0f64;
    let mut heights = Vec::new();

    while sim.time < 3.0 {
        let dt = sim.cfl_dt();
        sim.substep(dt);

        let speed = sim.particles.iter().fold(0.0f64, |m, p| m.max(p.v.length()));
        peak_speed = peak_speed.max(speed);

        if !speed.is_finite() || !sim.bodies[0].position.y.is_finite() {
            return FloatRun {
                rest: f64::NAN,
                peak_speed,
                finite: false,
            };
        }

        if sim.time > 2.0 {
            heights.push(sim.bodies[0].position.y);
        }
    }

    FloatRun {
        rest: heights.iter().sum::<f64>() / heights.len() as f64,
        peak_speed,
        finite: true,
    }
}

#[test]
fn narrow_band_floating_block_settles_like_dense() {
    let (band, dense) = std::thread::scope(|s| {
        let dense = s.spawn(|| floating_block(Params::default()));
        let band = floating_block(band_params());
        (band, dense.join().unwrap_or_else(|_| unreachable!()))
    });

    println!(
        "block rest height over 2..3 s: narrow band {:.4}, dense {:.4}; peak particle speed band {:.3}, dense {:.3} m/s",
        band.rest, dense.rest, band.peak_speed, dense.peak_speed
    );

    assert!(band.finite && dense.finite);
    assert!(band.peak_speed < 2.0 * dense.peak_speed.max(0.5));
    assert!((band.rest - dense.rest).abs() < 0.4 * 0.025);
}
