use liquid_wave::airy::{AmbientField, WaveComponent};
use liquid_wave::dispersive::{Dispersive, DispersiveParams};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::sponge::{AmbientFrame, Sponge};
use liquid_wave::swe::{Boundaries, Boundary, Side, Swe};
use liquid_wave::{GRAVITY, airy};

const DEPTH: f64 = 1.0;
const LENGTH: f64 = 120.0;
const SPONGE: f64 = 16.0;

fn channel(packet: impl Fn(f64) -> (f64, f64), absorbing: bool) -> Swe {
    let grid = Grid::new(480, 2, LENGTH / 480.0, LENGTH / 480.0);
    let bed = Bathymetry::flat(&grid, 0.0);
    let rest = State::lake(&grid, &bed, DEPTH);
    let mut state = rest.clone();

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, k);
            let (eta, q) = packet(x);
            let c = grid.cell(i, k);
            state.h[c] += eta;
            state.hu[c] = q;
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.boundaries =
        Boundaries::uniform(&grid, Boundary::Wall).with(Side::West, Boundary::Transmissive);

    if absorbing {
        swe.sponge = Some(Sponge::new(&grid, [0.0, SPONGE, 0.0, 0.0], 4.0, 2.0, rest));
    }

    swe
}

fn interior_energy(swe: &Swe) -> f64 {
    let g = &swe.grid;

    (0..g.nx)
        .filter(|&i| g.center(i, 0)[0] < LENGTH - SPONGE)
        .map(|i| {
            let c = g.cell(i, 0);
            let eta = swe.state.h[c] - DEPTH;
            let q = swe.state.hu[c];
            0.5 * (GRAVITY * eta * eta + q * q / DEPTH) * g.dx
        })
        .sum()
}

fn reflection(
    mut step: impl FnMut() -> f64,
    energy: impl Fn() -> f64,
    t_incident: f64,
    t_reflected: f64,
) -> f64 {
    let mut t = 0.0;

    while t < t_incident {
        t += step();
    }

    let incident = energy();

    while t < t_reflected {
        t += step();
    }

    (energy() / incident).sqrt()
}

fn swe_reflection(absorbing: bool) -> f64 {
    let c = (GRAVITY * DEPTH).sqrt();
    let pulse = |x: f64| {
        let eta = 0.01 * (-((x - 40.0) / 3.0).powi(2)).exp();
        (eta, c * eta)
    };

    let mut swe = channel(pulse, absorbing);
    let travel = (LENGTH - SPONGE - 40.0) / c;
    let cell = std::cell::RefCell::new(&mut swe);

    reflection(
        || cell.borrow_mut().step(0.05),
        || interior_energy(&cell.borrow()),
        0.6 * travel,
        travel + 2.0 * SPONGE / c + 45.0 / c,
    )
}

#[test]
fn sponge_absorbs_shallow_water_pulse() {
    let r = swe_reflection(true);
    let wall = swe_reflection(false);

    println!(
        "sponge, SWE pulse into a walled end: amplitude reflection {:.3}% (no sponge: {:.1}%)",
        100.0 * r,
        100.0 * wall
    );
    assert!(r < 0.05, "reflection {r}");
    assert!(wall > 0.5, "control run must reflect, got {wall}");
}

fn dispersive_reflection(absorbing: bool) -> (f64, f64, f64) {
    let wavelength = 6.0;
    let k = 2.0 * std::f64::consts::PI / wavelength;
    let omega = airy::omega(k, DEPTH, GRAVITY);
    let group = airy::group_derivative(k, DEPTH, GRAVITY);
    let packet = |x: f64| {
        let envelope = 0.01 * (-((x - 40.0) / 8.0).powi(2)).exp();
        let eta = envelope * (k * x).cos();
        (eta, omega / k * eta)
    };

    let swe = channel(packet, absorbing);
    let mut model = Dispersive::new(swe, DispersiveParams::default());
    let travel = (LENGTH - SPONGE - 40.0) / group;
    let cell = std::cell::RefCell::new(&mut model);

    let r = reflection(
        || cell.borrow_mut().step(0.05),
        || interior_energy(&cell.borrow().swe),
        0.5 * travel,
        travel + 2.0 * SPONGE / group + 50.0 / group,
    );

    (r, k * DEPTH, group)
}

#[test]
fn sponge_absorbs_dispersive_wave_packet() {
    let (r, kh, group) = dispersive_reflection(true);
    let (wall, _, _) = dispersive_reflection(false);

    println!(
        "sponge, dispersive packet (kh = {kh:.2}, group speed {group:.2} m/s) into a walled end: amplitude reflection {:.3}% (no sponge: {:.1}%)",
        100.0 * r,
        100.0 * wall
    );
    assert!(r < 0.05, "reflection {r}");
    assert!(wall > 0.5, "control run must reflect, got {wall}");
}

fn ambient_channel(pulse: bool) -> (Dispersive, AmbientFrame, AmbientField) {
    let grid = Grid::new(480, 2, LENGTH / 480.0, LENGTH / 480.0);
    let bed = Bathymetry::flat(&grid, 0.0);
    let k = 2.0 * std::f64::consts::PI / 8.0;
    let field = AmbientField::new(
        vec![WaveComponent {
            k: [k, 0.0],
            amplitude: 0.004,
            phase: 0.0,
            omega: airy::omega(k, DEPTH, GRAVITY),
        }],
        DEPTH,
    );
    let frame = AmbientFrame {
        origin: [0.0, 0.0],
        level: DEPTH,
    };
    let mut state = frame.state(&grid, &bed, &field, 0.0);

    if pulse {
        for (c, h) in state.h.iter_mut().enumerate() {
            let [x, _] = grid.center(c % grid.nx, c / grid.nx);
            *h += 0.01 * (-((x - 60.0) / 3.0).powi(2)).exp();
        }
    }

    let mut swe = Swe::new(grid, bed, state.clone());
    swe.boundaries = Boundaries::uniform(&grid, Boundary::Wall);
    let mut sponge = Sponge::new(&grid, [SPONGE, SPONGE, 0.0, 0.0], 4.0, 2.0, state);
    sponge.ambient = Some((frame, field.clone()));
    swe.sponge = Some(sponge);

    (
        Dispersive::new(swe, DispersiveParams::default()),
        frame,
        field,
    )
}

fn interior_cells(model: &Dispersive) -> Vec<usize> {
    let g = &model.swe.grid;

    (0..g.nx)
        .filter(|&i| {
            let x = g.center(i, 0)[0];
            x > SPONGE && x < LENGTH - SPONGE
        })
        .map(|i| g.cell(i, 0))
        .collect()
}

fn energy(eta: f64, q: f64) -> f64 {
    0.5 * (GRAVITY * eta * eta + q * q / DEPTH)
}

#[test]
fn sponge_absorbs_only_the_perturbation() {
    let (mut calm, frame, field) = ambient_channel(false);
    let (mut disturbed, _, _) = ambient_channel(true);
    let cells = interior_cells(&calm);

    let difference = |a: &Dispersive, b: &Dispersive| -> f64 {
        cells
            .iter()
            .map(|&c| {
                energy(
                    a.swe.state.h[c] - b.swe.state.h[c],
                    a.swe.state.hu[c] - b.swe.state.hu[c],
                )
            })
            .sum()
    };

    let initial = difference(&disturbed, &calm);

    while calm.swe.time < 60.0 {
        let a = calm.step(0.015);
        let b = disturbed.step(0.015);
        assert_eq!(a, b, "both runs must stay on the same time grid");
    }

    let residual = (difference(&disturbed, &calm) / initial).sqrt();

    let swe = &calm.swe;
    let (mut drift, mut reference) = (0.0, 0.0);

    for &c in &cells {
        let [h, hu, _] = frame.cell_state(&swe.grid, &swe.bathymetry, &field, c, swe.time);
        drift += energy(swe.state.h[c] - h, swe.state.hu[c] - hu);
        reference += energy(h - DEPTH, hu);
    }

    let ambient_error = (drift / reference).sqrt();

    println!(
        "ambient + perturbation: perturbation amplitude left after 60 s {:.3}%, simulated ambient wave vs analytic {:.2}%",
        100.0 * residual,
        100.0 * ambient_error
    );
    assert!(residual < 0.05, "perturbation not absorbed: {residual}");
    assert!(
        ambient_error < 0.1,
        "ambient wave lost in the sponge: {ambient_error}"
    );
}
