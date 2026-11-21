use liquid_wave::GRAVITY;
use liquid_wave::airy::{self, AmbientField, WaveComponent};
use liquid_wave::ocean::{Ocean, OceanParams};

#[test]
fn dispersion_has_deep_and_shallow_limits() {
    let k = 0.7;

    let deep = airy::omega(k, 1000.0, GRAVITY);
    assert!((deep - (GRAVITY * k).sqrt()).abs() < 1e-12);

    let shallow = airy::omega(k, 1e-3, GRAVITY);
    let swe = airy::shallow_omega(k, 1e-3, GRAVITY);
    assert!((shallow / swe - 1.0).abs() < 1e-6);

    let dk = 1e-6;
    let numeric =
        (airy::omega(k + dk, 3.0, GRAVITY) - airy::omega(k - dk, 3.0, GRAVITY)) / (2.0 * dk);
    assert!((numeric - airy::group_derivative(k, 3.0, GRAVITY)).abs() < 1e-8);
}

#[test]
fn ambient_field_reproduces_fft_ocean() {
    let ocean = Ocean::new(OceanParams {
        resolution: 32,
        patch_sizes: vec![400.0, 60.0],
        ..OceanParams::default()
    });
    let t = 4.2;

    for (index, cascade) in ocean.cascades.iter().enumerate() {
        let eta = ocean.fields(index, t).eta;
        let field = AmbientField::new(cascade.components().collect(), ocean.params.sea.depth);
        let rms = (eta.iter().map(|v| v * v).sum::<f64>() / eta.len() as f64).sqrt();
        let spacing = cascade.patch / cascade.n as f64;
        let mut worst: f64 = 0.0;

        for (i, j) in [(0, 0), (5, 9), (31, 17), (12, 30)] {
            let direct = field.elevation(i as f64 * spacing, j as f64 * spacing, t);
            worst = worst.max((direct - eta[j * cascade.n + i]).abs());
        }

        println!(
            "cascade L={:6.1}: analytic components vs FFT elevation, max deviation {worst:.3e} (rms {rms:.3e})",
            cascade.patch
        );
        assert!(worst < 1e-10 * rms);
    }

    let ambient = ocean.ambient();
    assert_eq!(
        ambient.components.len(),
        ocean
            .cascades
            .iter()
            .map(|c| c.components().count())
            .sum::<usize>()
    );
}

fn single_with(depth: f64, amplitude: f64) -> AmbientField {
    let k = [0.3, 0.4];
    let magnitude = 0.5;

    AmbientField::new(
        vec![WaveComponent {
            k,
            amplitude,
            phase: 0.3,
            omega: airy::omega(magnitude, depth, GRAVITY),
        }],
        depth,
    )
}

fn single(depth: f64) -> AmbientField {
    single_with(depth, 0.05)
}

#[test]
fn velocity_is_divergence_free_and_satisfies_surface_kinematics() {
    let (x, z, t) = (1.3, -2.1, 0.7);
    let e = 1e-3;
    let linear = single_with(6.0, 1e-4);
    let k = linear.components[0].wavenumber();

    for y in [-0.5, -2.0, -5.0] {
        let d = |dx: f64, dy: f64, dz: f64, axis: usize| {
            linear.sample(x + dx, y + dy, z + dz, t).velocity[axis]
        };
        let divergence = (d(e, 0.0, 0.0, 0) - d(-e, 0.0, 0.0, 0)) / (2.0 * e)
            + (d(0.0, e, 0.0, 1) - d(0.0, -e, 0.0, 1)) / (2.0 * e)
            + (d(0.0, 0.0, e, 2) - d(0.0, 0.0, -e, 2)) / (2.0 * e);

        let u = linear.sample(x, y, z, t).velocity;
        let scale = k * (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        assert!(
            divergence.abs() < 1e-3 * scale,
            "divergence {divergence} vs {scale} at y = {y}"
        );
    }

    let field = single(6.0);
    let e = 1e-4;
    let surface = field.elevation(x, z, t);
    let deta_dt = (field.elevation(x, z, t + e) - field.elevation(x, z, t - e)) / (2.0 * e);
    let w = field.sample(x, surface, z, t).velocity[1];

    println!("surface kinematics: d(eta)/dt = {deta_dt:.6e}, w(eta) = {w:.6e}");
    assert!((deta_dt - w).abs() < 1e-6);
}

#[test]
fn deep_water_profile_decays_exponentially() {
    let field = single(500.0);
    let c = field.components[0];
    let k = c.wavenumber();
    let (x, z, t) = (0.0, 0.0, 0.0);
    let theta = c.angle(x, z, t);
    let eta = field.elevation(x, z, t);

    for y in [-1.0, -3.0, -6.0] {
        let stretched = (y - eta) * field.depth / (field.depth + eta);
        let expected = c.amplitude * c.omega * (k * stretched).exp() * theta.cos();
        let horizontal = field.sample(x, y, z, t).velocity;
        let along = (horizontal[0] * c.k[0] + horizontal[2] * c.k[1]) / k;

        assert!(
            (along - expected).abs() < 1e-9,
            "y = {y}: {along} vs {expected}"
        );
    }
}

#[test]
fn depth_integrated_flux_matches_profile() {
    let depth = 4.0;
    let field = single(depth);
    let (x, z, t) = (0.4, 0.9, 0.2);
    let eta = field.elevation(x, z, t);
    let n = 4000;
    let dy = (depth + eta) / n as f64;

    let integral = (0..n).fold([0.0; 2], |acc, i| {
        let y = -depth + (i as f64 + 0.5) * dy;
        let u = field.sample(x, y, z, t).velocity;
        [acc[0] + u[0] * dy, acc[1] + u[2] * dy]
    });

    let flux = field.flux(x, z, t);
    let stretch = (depth + eta) / depth;

    for axis in 0..2 {
        assert!((integral[axis] - stretch * flux[axis]).abs() < 1e-6 * flux[axis].abs().max(1e-6));
    }
}
