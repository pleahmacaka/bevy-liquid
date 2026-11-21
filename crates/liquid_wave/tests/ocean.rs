use std::f64::consts::PI;

use liquid_wave::fft::{Complex64, Fft2};
use liquid_wave::ocean::{Foam, FoamParams, Ocean, OceanParams, cascade_bands};

fn params(seed: u64) -> OceanParams {
    OceanParams {
        resolution: 128,
        seed,
        ..OceanParams::default()
    }
}

fn spatial_variance(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;

    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

#[test]
fn bands_are_contiguous_and_disjoint() {
    let p = params(1);
    let bands = cascade_bands(&p.patch_sizes, p.resolution);

    for pair in bands.windows(2) {
        assert_eq!(pair[0][1], pair[1][0], "bands must touch without overlap");
    }

    for (band, &patch) in bands.iter().zip(&p.patch_sizes) {
        assert!(band[0] < band[1]);
        assert!(band[1] <= PI * p.resolution as f64 / patch + 1e-12);
    }

    println!("cascade bands (rad/m): {bands:?}");
}

#[test]
fn discrete_spectrum_matches_continuous_integral() {
    let ocean = Ocean::new(params(1));
    let sea = ocean.params.sea;
    let mut discrete_total = 0.0;
    let mut integral_total = 0.0;

    for cascade in &ocean.cascades {
        let dk = 2.0 * PI / cascade.patch;
        let mut discrete = 0.0;

        for b in 0..cascade.n {
            for a in 0..cascade.n {
                let k = cascade.wavevector(a, b);
                let m = k[0].hypot(k[1]);
                let edge = a == cascade.n / 2 || b == cascade.n / 2;

                if m > 0.0 && m >= cascade.band[0] && m < cascade.band[1] && !edge {
                    discrete += sea.wavevector_density(k[0], k[1]) * dk * dk;
                }
            }
        }

        let integral = sea.band_variance(cascade.band[0].max(dk), cascade.band[1], 400);
        discrete_total += discrete;
        integral_total += integral;

        println!(
            "cascade L={:7.1} m band [{:.4}, {:.4}]: grid sum {discrete:.5e} m^2, integral {integral:.5e} m^2",
            cascade.patch, cascade.band[0], cascade.band[1]
        );
    }

    let error = (discrete_total / integral_total - 1.0).abs();
    println!(
        "total grid sum {discrete_total:.5e}, integral {integral_total:.5e}, error {:.2}%",
        100.0 * error
    );
    assert!(error < 0.03);
}

#[test]
fn elevation_variance_matches_spectrum_integral() {
    let seeds = 24;
    let mut realized = 0.0;

    for seed in 0..seeds {
        let ocean = Ocean::new(params(seed));

        for (c, _) in ocean.cascades.iter().enumerate() {
            realized += spatial_variance(&ocean.fields(c, 3.7).eta) / seeds as f64;
        }
    }

    let ocean = Ocean::new(params(0));
    let integral: f64 = ocean
        .cascades
        .iter()
        .map(|c| c.band_variance_integral(&ocean.params.sea))
        .sum();

    let error = (realized / integral - 1.0).abs();
    println!(
        "elevation variance over {seeds} seeds: {realized:.5e} m^2 vs spectrum integral {integral:.5e} m^2 (Hs = {:.3} m), error {:.2}%",
        4.0 * integral.sqrt(),
        100.0 * error
    );
    assert!(error < 0.05);
}

#[test]
fn spectrum_is_hermitian_and_output_is_real() {
    let ocean = Ocean::new(params(7));
    let fft = Fft2::new(128, 128);

    for cascade in &ocean.cascades {
        let mut data = cascade.spectrum(12.3);
        fft.inverse_unnormalized(&mut data);

        let re = data.iter().map(|v| v.re.abs()).fold(0.0, f64::max);
        let im = data.iter().map(|v| v.im.abs()).fold(0.0, f64::max);

        println!(
            "cascade L={:7.1}: max |Re| {re:.4e}, max |Im| {im:.3e}",
            cascade.patch
        );
        assert!(im <= 1e-12 * re);
    }
}

#[test]
fn single_mode_matches_gerstner_kinematics() {
    let mut ocean = Ocean::new(OceanParams {
        resolution: 32,
        choppiness: 0.8,
        ..OceanParams::default()
    });
    let cascade = &mut ocean.cascades[0];
    cascade
        .h0
        .iter_mut()
        .for_each(|h| *h = Complex64::default());

    let (a, b) = (3, 2);
    let amplitude = 0.7;
    let phase = 0.4;
    cascade.h0[b * cascade.n + a] = Complex64::from_polar(0.5 * amplitude, phase);

    let k = cascade.wavevector(a, b);
    let omega = cascade.omega[b * cascade.n + a];
    let magnitude = k[0].hypot(k[1]);
    let t = 1.9;
    let fields = ocean.fields(0, t);
    let cascade = &ocean.cascades[0];
    let lambda = ocean.params.choppiness;
    let mut worst: f64 = 0.0;

    for j in 0..cascade.n {
        for i in 0..cascade.n {
            let x = i as f64 * cascade.patch / cascade.n as f64;
            let z = j as f64 * cascade.patch / cascade.n as f64;
            let theta = k[0] * x + k[1] * z - omega * t + phase;
            let c = j * cascade.n + i;

            let eta = amplitude * theta.cos();
            let dx = -lambda * amplitude * k[0] / magnitude * theta.sin();
            let jacobian = 1.0 - lambda * amplitude * magnitude * theta.cos();

            worst = worst
                .max((fields.eta[c] - eta).abs())
                .max((fields.dx[c] - dx).abs())
                .max((fields.jacobian[c] - jacobian).abs());
        }
    }

    println!("single mode: max deviation from analytic eta / Dx / J = {worst:.3e}");
    assert!(worst < 1e-12);
}

#[test]
fn jacobian_averages_to_one() {
    let ocean = Ocean::new(params(3));

    for (c, cascade) in ocean.cascades.iter().enumerate() {
        let fields = ocean.fields(c, 5.0);
        let mean = fields.jacobian.iter().sum::<f64>() / fields.jacobian.len() as f64;
        let min = fields
            .jacobian
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);

        println!(
            "cascade L={:7.1}: mean J = {mean:.15}, min J = {min:.3}",
            cascade.patch
        );
        assert!((mean - 1.0).abs() < 1e-12);
    }
}

#[test]
fn foam_grows_under_folding_and_decays_elsewhere() {
    let params = FoamParams {
        threshold: 0.6,
        gain: 4.0,
        decay: 2.0,
    };
    let jacobian = [0.2, 1.0];
    let mut foam = Foam::new(2);

    for _ in 0..10 {
        foam.update(&jacobian, 0.05, &params);
    }

    assert!(foam.value[0] > 0.5);
    assert_eq!(foam.value[1], 0.0);

    let start = foam.value[0];
    foam.update(&[1.0, 1.0], 1.0, &params);
    let expected = start * (-0.5f64).exp();

    println!(
        "foam after folding {start:.4}, after 1 s calm {:.4} (expected {expected:.4})",
        foam.value[0]
    );
    assert!((foam.value[0] - expected).abs() < 1e-12);
}
