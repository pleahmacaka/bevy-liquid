use glam::{DVec3, UVec3};
use liquid_ref::gas::{Emitter, GasDomain, GasParams, Region};
use liquid_ref::grid::Grid;
use liquid_ref::rigid::Shape;

const DX: f64 = 1.0 / 16.0;

const FRAME: f64 = 1.0 / 60.0;

const VAPOR: f64 = 373.15;

fn domain(dims: UVec3, params: GasParams) -> GasDomain {
    GasDomain::new(Grid::new(dims, DX), DVec3::ZERO, params)
}

fn sphere(center: DVec3, radius: f64) -> Region {
    Region {
        shape: Shape::Sphere { radius },
        center,
    }
}

fn plume(dims: UVec3, params: GasParams) -> GasDomain {
    let mut gas = domain(dims, params);
    let base = DVec3::new(0.5 * dims.x as f64 * DX, 0.25, 0.5 * dims.z as f64 * DX);

    gas.emitters.push(Emitter {
        region: sphere(base, 0.15),
        density_rate: 4.0,
        heat_rate: 400.0,
    });

    gas
}

fn run(gas: &mut GasDomain, frames: usize) -> usize {
    (0..frames).map(|_| gas.advance(FRAME)).sum()
}

fn hot_blob(gas: &mut GasDomain, center: DVec3, radius: f64, heat: f64) {
    for n in 0..gas.grid.cell_count() {
        if (gas.cell_position(n) - center).length() <= radius {
            gas.density.data[n] = 1.0;
            gas.temperature.data[n] += heat;
        }
    }
}

fn centroid(gas: &GasDomain) -> DVec3 {
    let total: f64 = gas.density.data.iter().sum();

    (0..gas.grid.cell_count())
        .map(|n| gas.cell_position(n) * gas.density.data[n])
        .sum::<DVec3>()
        / total
}

fn lateral_spread(gas: &GasDomain) -> f64 {
    let c = centroid(gas);
    let total: f64 = gas.density.data.iter().sum();
    let moment: f64 = (0..gas.grid.cell_count())
        .map(|n| {
            let d = gas.cell_position(n) - c;

            gas.density.data[n] * (d.x * d.x + d.z * d.z)
        })
        .sum();

    (moment / total).sqrt()
}

fn enstrophy(gas: &GasDomain) -> f64 {
    gas.vorticity()
        .iter()
        .map(|w| w.length_squared())
        .sum::<f64>()
        * DX.powi(3)
}

fn wall_speed(gas: &GasDomain, axis: usize, high: bool) -> f64 {
    let faces = &gas.velocity.axes[axis];
    let layer = if high { gas.grid.dims[axis] } else { 0 };

    (0..faces.data.len())
        .filter(|&n| faces.coords(n)[axis] == layer)
        .fold(0.0, |m, n| m.max(faces.data[n].abs()))
}

fn top_outflow(gas: &GasDomain) -> f64 {
    let d = gas.grid.dims;
    let faces = &gas.velocity.axes[1];
    let mut flux = 0.0;

    for k in 0..d.z {
        for i in 0..d.x {
            flux += faces.get(i, d.y, k).max(0.0) * gas.density.get(i, d.y - 1, k);
        }
    }

    flux * DX * DX
}

fn bits(gas: &GasDomain) -> Vec<u64> {
    gas.velocity
        .axes
        .iter()
        .chain([&gas.density, &gas.temperature])
        .flat_map(|f| f.data.iter().map(|v| v.to_bits()))
        .chain([gas.time.to_bits()])
        .collect()
}

#[test]
fn ambient_gas_stays_at_rest() {
    let mut gas = domain(UVec3::new(16, 24, 16), GasParams::default());
    let ambient = gas.params.ambient_temperature;
    let substeps = run(&mut gas, 120);

    println!("substeps {substeps}, max |u| {:e}", gas.velocity.max_abs());

    assert_eq!(substeps, 120);
    assert_eq!(gas.velocity.max_abs(), 0.0);
    assert!(gas.density.data.iter().all(|r| *r == 0.0));
    assert!(gas.temperature.data.iter().all(|t| *t == ambient));
}

#[test]
fn hot_blob_rises_and_spreads() {
    let mut gas = domain(UVec3::new(24, 40, 24), GasParams::default());
    hot_blob(&mut gas, DVec3::new(0.75, 0.5, 0.75), 0.25, 100.0);

    let mut heights = vec![centroid(&gas).y];
    let mut spreads = vec![lateral_spread(&gas)];

    for _ in 0..40 {
        gas.advance(FRAME);
        heights.push(centroid(&gas).y);
        spreads.push(lateral_spread(&gas));
    }

    let rise = heights[40] - heights[0];
    let growth = spreads[40] / spreads[0];

    println!(
        "centroid y {:.5} -> {:.5} -> {:.5}, rise {rise:.4} m, lateral spread {:.5} -> {:.5} (x{growth:.4})",
        heights[0], heights[1], heights[40], spreads[0], spreads[40]
    );

    assert!((heights[1] - heights[0]).abs() < 1e-12);
    assert!(heights[1..].windows(2).all(|w| w[1] > w[0]));
    assert!(rise > 0.2);
    assert!(growth > 1.1);
}

#[test]
fn projection_leaves_divergence_below_tolerance() {
    let mut gas = plume(UVec3::new(24, 40, 24), GasParams::default());
    gas.obstacles.push(Region {
        shape: Shape::Cuboid {
            half: DVec3::new(0.2, 0.05, 0.2),
        },
        center: DVec3::new(0.9, 1.0, 0.75),
    });

    run(&mut gas, 40);

    let solid = gas.solid();
    let worst = gas
        .divergence()
        .data
        .iter()
        .zip(&solid)
        .filter(|(_, s)| !**s)
        .fold(0.0f64, |m, (d, _)| m.max(d.abs()));
    let bound = gas.params.tolerance * gas.report.input_divergence;

    println!(
        "max |div| {worst:e} vs tolerance bound {bound:e}, divergence before projection {:e}, {} pcg iterations",
        gas.report.input_divergence, gas.report.pressure.iterations
    );

    assert!(solid.iter().any(|s| *s));
    assert!(gas.report.input_divergence > 1.0);
    assert!(worst <= bound);
}

#[test]
fn still_gas_cools_and_fades_exponentially() {
    let params = GasParams {
        density_buoyancy: 0.0,
        thermal_buoyancy: 0.0,
        ..GasParams::default()
    };
    let ambient = params.ambient_temperature;
    let mut gas = domain(UVec3::new(12, 12, 12), params);

    gas.density.fill(2.0);
    gas.temperature.fill(ambient + 50.0);

    let substeps = run(&mut gas, 90);
    let rho = 2.0 * (-params.density_decay * gas.time).exp();
    let excess = 50.0 * (-params.cooling * gas.time).exp();
    let rho_error = gas
        .density
        .data
        .iter()
        .fold(0.0f64, |m, r| m.max((r - rho).abs() / rho));
    let t_error = gas
        .temperature
        .data
        .iter()
        .fold(0.0f64, |m, t| m.max((t - ambient - excess).abs() / excess));

    println!(
        "{substeps} substeps to t {:.4}: density {rho:.12} rel err {rho_error:e}, excess T {excess:.12} rel err {t_error:e}",
        gas.time
    );

    assert_eq!(gas.velocity.max_abs(), 0.0);
    assert!(rho_error < 1e-12);
    assert!(t_error < 1e-12);
}

#[test]
fn confinement_raises_enstrophy() {
    let measure = |epsilon: f64| {
        let params = GasParams {
            vorticity_confinement: epsilon,
            ..GasParams::default()
        };
        let mut gas = plume(UVec3::new(24, 40, 24), params);

        for _ in 0..90 {
            gas.substep(1.0 / 120.0);
        }

        enstrophy(&gas)
    };

    let off = measure(0.0);
    let on = measure(GasParams::default().vorticity_confinement);

    println!("enstrophy off {off:.5}, on {on:.5}, ratio {:.4}", on / off);

    assert!(on > 1.03 * off);
}

#[test]
fn closed_walls_block_normal_flow() {
    for top_open in [true, false] {
        let mut params = GasParams::default();
        params.open_high[1] = top_open;

        let mut gas = plume(UVec3::new(16, 24, 16), params);
        run(&mut gas, 60);

        let top = wall_speed(&gas, 1, true);

        println!(
            "top open {top_open}: max |u_n| on the top {top:e}, pcg {:?}",
            gas.report.pressure
        );

        for axis in 0..3 {
            for high in [false, true] {
                if axis == 1 && high {
                    continue;
                }

                assert_eq!(wall_speed(&gas, axis, high), 0.0);
            }
        }

        assert_eq!(top > 0.0, top_open);
        assert!(gas.report.pressure.residual <= params.tolerance * gas.report.input_divergence);
    }
}

#[test]
fn open_top_lets_smoke_leave() {
    let finish = |top_open: bool| {
        let mut params = GasParams {
            density_decay: 0.0,
            ..GasParams::default()
        };
        params.open_high[1] = top_open;

        let mut gas = plume(UVec3::new(16, 20, 16), params);
        run(&mut gas, 180);

        (gas.mass(), top_outflow(&gas))
    };

    let (open_mass, open_flux) = finish(true);
    let (closed_mass, closed_flux) = finish(false);

    println!(
        "after 3 s: open top mass {open_mass:.4} outflow {open_flux:.4e} kg/s, closed top mass {closed_mass:.4} outflow {closed_flux:e}"
    );

    assert!(open_flux > 0.0);
    assert_eq!(closed_flux, 0.0);
    assert!(open_mass < 0.5 * closed_mass);
}

#[test]
fn inject_adds_exact_mass_inside_and_drops_outside() {
    let mut gas = domain(UVec3::new(8, 8, 8), GasParams::default());
    gas.obstacles.push(sphere(DVec3::splat(0.375), 0.08));

    let m = 0.0123;
    let inside = [
        DVec3::new(0.21, 0.33, 0.27),
        DVec3::new(0.01, 0.49, 0.001),
        DVec3::ZERO,
        DVec3::splat(0.5),
        DVec3::new(0.3, 0.375, 0.375),
    ];

    for p in inside {
        let before = gas.mass();
        let accepted = gas.inject(p, m, VAPOR, DVec3::Y);
        let error = (gas.mass() - before - m).abs() / m;

        println!("inside {p}: rel mass error {error:e}");

        assert!(accepted);
        assert!(error < 1e-14);
    }

    let outside = [
        DVec3::new(-0.01, 0.2, 0.2),
        DVec3::new(0.2, 0.51, 0.2),
        DVec3::new(0.2, 0.2, 1.0),
        DVec3::splat(0.375),
    ];

    for p in outside {
        let before = bits(&gas);

        assert!(!gas.inject(p, m, VAPOR, DVec3::Y));
        assert!(bits(&gas) == before);
    }
}

#[test]
fn identical_runs_are_bit_identical() {
    let make = || {
        let mut gas = plume(UVec3::new(16, 24, 16), GasParams::default());
        gas.obstacles.push(sphere(DVec3::new(0.6, 0.8, 0.5), 0.12));
        run(&mut gas, 40);
        gas.inject(DVec3::new(0.3, 0.4, 0.5), 0.01, VAPOR, DVec3::Y);
        run(&mut gas, 20);

        gas
    };

    let a = make();
    let b = make();

    println!("{} values compared at t {}", bits(&a).len(), a.time);

    assert!(bits(&a) == bits(&b));
}
