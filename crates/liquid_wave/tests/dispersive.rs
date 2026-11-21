use std::f64::consts::PI;

use liquid_wave::GRAVITY;
use liquid_wave::airy;
use liquid_wave::dispersive::{Dispersive, DispersiveParams};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::swe::Swe;

const MODE: f64 = 2.0;
const CELLS: usize = 64;

struct Basin {
    grid: Grid,
    k: f64,
    omega: f64,
}

fn basin(kh: f64, depth: f64, amplitude: f64) -> (Basin, Swe) {
    let k = kh / depth;
    let length = MODE * PI / k;
    let dx = length / CELLS as f64;
    let grid = Grid::new(CELLS, 4, dx, dx);
    let bed = Bathymetry::flat(&grid, 0.0);
    let mut state = State::zeros(grid.cells());

    for kk in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, kk);
            state.h[grid.cell(i, kk)] = depth + amplitude * (k * x).cos();
        }
    }

    let omega = airy::omega(k, depth, GRAVITY);

    (Basin { grid, k, omega }, Swe::new(grid, bed, state))
}

fn mode_amplitude(basin: &Basin, h: &[f64], depth: f64) -> f64 {
    let g = &basin.grid;
    let (mut num, mut den) = (0.0, 0.0);

    for i in 0..g.nx {
        let [x, _] = g.center(i, 0);
        let c = (basin.k * x).cos();
        num += (h[g.cell(i, 0)] - depth) * c;
        den += c * c;
    }

    num / den
}

fn measure(basin: &Basin, depth: f64, mut step: impl FnMut(f64) -> (f64, Vec<f64>)) -> f64 {
    let period = 2.0 * PI / basin.omega;
    let dt_max = period / 200.0;
    let mut t = 0.0;
    let mut previous = (0.0, mode_amplitude(basin, &step(0.0).1, depth));
    let mut crossings = Vec::new();

    while t < 6.0 * period {
        let (dt, h) = step(dt_max);
        t += dt;
        let a = mode_amplitude(basin, &h, depth);

        if previous.1.signum() != a.signum() && a != 0.0 {
            crossings.push(previous.0 + (t - previous.0) * previous.1 / (previous.1 - a));
        }

        previous = (t, a);
    }

    let spans = crossings.len() - 1;

    PI * spans as f64 / (crossings[spans] - crossings[0])
}

fn dispersive_omega(kh: f64, depth: f64, params: DispersiveParams) -> (f64, f64) {
    let (b, swe) = basin(kh, depth, 1e-3 * depth);
    let mut model = Dispersive::new(swe, params);

    let measured = measure(&b, depth, |dt_max| {
        if dt_max == 0.0 {
            return (0.0, model.swe.state.h.clone());
        }

        let dt = model.step(dt_max);
        (dt, model.swe.state.h.clone())
    });

    (measured, b.omega)
}

fn swe_omega(kh: f64, depth: f64) -> (f64, f64) {
    let (b, mut swe) = basin(kh, depth, 1e-3 * depth);

    let measured = measure(&b, depth, |dt_max| {
        if dt_max == 0.0 {
            return (0.0, swe.state.h.clone());
        }

        let dt = swe.step(dt_max);
        (dt, swe.state.h.clone())
    });

    (measured, b.omega)
}

#[test]
fn standing_waves_follow_airy_dispersion() {
    for kh in [0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0] {
        let (measured, exact) = dispersive_omega(kh, 4.0, DispersiveParams::default());
        let error = (measured / exact - 1.0).abs();

        println!(
            "dispersive kh={kh:5.2}: omega {measured:.6} vs exact {exact:.6}, error {:.3}%",
            100.0 * error
        );
        assert!(error < 0.01, "kh = {kh}: error {:.3}%", 100.0 * error);
    }
}

#[test]
fn pure_shallow_water_fails_in_deep_water() {
    for kh in [0.1, 1.0, 3.0, 10.0] {
        let (measured, exact) = swe_omega(kh, 4.0);
        let error = measured / exact - 1.0;
        let theory = airy::shallow_omega(kh / 4.0, 4.0, GRAVITY) / exact - 1.0;

        println!(
            "pure SWE   kh={kh:5.2}: omega {measured:.6} vs exact {exact:.6}, error {:+.2}% (linear SWE theory {:+.2}%)",
            100.0 * error,
            100.0 * theory
        );

        if kh >= 3.0 {
            assert!(error > 0.5, "SWE should be far too fast at kh = {kh}");
        }
    }
}

#[test]
fn depth_between_reference_samples() {
    let (measured, exact) = dispersive_omega(1.0, 8.0, DispersiveParams::default());
    let error = (measured / exact - 1.0).abs();

    println!(
        "dispersive at 8 m (between 4 m and 16 m samples), kh=1: error {:.2}%",
        100.0 * error
    );
    assert!(error < 0.12);
}

fn island_basin() -> Swe {
    let grid = Grid::new(64, 32, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, z| {
        0.6 * (-((x - 20.0).powi(2) + (z - 8.0).powi(2)) / 6.0).exp() * 3.0 + 0.002 * x * x
    });
    let mut state = State::lake(&grid, &bed, 1.5);

    for k in 0..grid.nz {
        for i in 0..10 {
            state.h[grid.cell(i, k)] += 1.0;
        }
    }

    Swe::new(grid, bed, state)
}

#[test]
fn dispersive_model_conserves_mass_and_positivity() {
    let mut model = Dispersive::new(island_basin(), DispersiveParams::default());
    let initial = model.swe.state.mass(&model.swe.grid);
    let mut min_depth: f64 = 0.0;

    for _ in 0..1500 {
        model.step(0.05);
        min_depth = min_depth.min(
            model
                .swe
                .state
                .h
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min),
        );
    }

    let drift = (model.swe.state.mass(&model.swe.grid) - initial).abs() / initial;

    println!(
        "dispersive dam break, t = {:.2} s: relative mass drift {drift:.3e}, min h {min_depth:e}",
        model.swe.time
    );
    assert!(drift < 1e-12);
    assert!(min_depth >= 0.0);
}

#[test]
fn dispersive_lake_at_rest() {
    let grid = Grid::new(64, 32, 0.5, 0.5);
    let bed = Bathymetry::from_fn(&grid, |x, z| {
        1.8 * (-((x - 16.0).powi(2) + (z - 8.0).powi(2)) / 10.0).exp()
    });
    let state = State::lake(&grid, &bed, 1.0);
    let mut model = Dispersive::new(Swe::new(grid, bed, state), DispersiveParams::default());

    for _ in 0..1000 {
        model.step(1.0);
    }

    let speed = model.swe.max_speed();
    println!("dispersive lake at rest over a dry island, 1000 steps: max |u| = {speed:.3e}");
    assert!(speed < 1e-10);
}

#[test]
fn coupling_boundary_velocity_follows_the_airy_profile() {
    let depth = 4.0;
    let kh = 2.0;
    let amplitude = 1e-3 * depth;
    let (basin, mut swe) = basin(kh, depth, amplitude);
    let g = basin.grid;
    let speed = amplitude * basin.omega;

    for kk in 0..g.nz {
        for i in 0..g.nx {
            let [x, _] = g.center(i, kk);
            swe.state.hu[g.cell(i, kk)] = speed / basin.k * (basin.k * x).cos();
        }
    }

    let model = Dispersive::new(swe, DispersiveParams::default());
    let parts = model.decompose(&model.swe.state);
    let modes = model.surface_modes(&parts.surface);
    let boundary = 8;
    let [x, _] = g.center(boundary, 2);
    let c = g.cell(boundary, 2);
    let column = model.swe.state.h[c];

    println!(
        "local wavenumber {:.5} (exact {:.5}), column depth {column:.5}",
        modes[c].k, basin.k
    );

    assert!((modes[c].k / basin.k - 1.0).abs() < 1e-3);

    for fraction in [0.0, 0.25, 0.5, 0.75, 0.999] {
        let z = -fraction * depth;
        let got = model.column_velocity(&modes, boundary, 2, column + z);
        let u = speed * airy::horizontal_profile(basin.k, z, depth) * (basin.k * x).cos();
        let w = speed * airy::vertical_profile(basin.k, z, depth) * (basin.k * x).sin();
        let (eu, ew) = ((got[0] - u).abs() / speed, (got[1] - w).abs() / speed);

        println!(
            "z = {z:6.3}: u {:.5e} vs Airy {u:.5e} (error {:.2}%), w {:.5e} vs Airy {w:.5e} (error {:.2}%)",
            got[0],
            100.0 * eu,
            got[1],
            100.0 * ew
        );

        assert!(eu < 0.02 && ew < 0.02, "z = {z}: u error {eu:e}, w error {ew:e}");
    }
}

#[test]
fn dispersive_interface_ledger_is_exact() {
    let mut swe = island_basin();
    let g = swe.grid;

    swe.coupling = Some(liquid_wave::swe::CouplingRegion {
        min: [36, 10],
        max: [52, 26],
        blend: 0.5,
    });
    swe.coupled = swe.state.clone();

    let mut model = Dispersive::new(swe, DispersiveParams::default());
    let invariant = |l: liquid_wave::swe::Ledger| l.outside + l.interface - l.injected;
    let start = model.swe.ledger();

    for _ in 0..600 {
        model.step(0.05);
    }

    let end = model.swe.ledger();
    let drift = (invariant(end) - invariant(start)).abs() / start.outside;

    println!(
        "dispersive coupled basin on a {}x{} grid, t = {:.2} s: interface {:.6e}, outside drift {drift:.3e}",
        g.nx, g.nz, model.swe.time, end.interface
    );

    assert!(end.interface.abs() > 1e-3);
    assert!(drift < 1e-12, "outside + interface drifted by {drift:e}");
}
