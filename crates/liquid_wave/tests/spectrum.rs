use std::f64::consts::PI;

use liquid_wave::spectrum::{SeaState, Spreading, ln_gamma};

fn spreading_integral(sea: &SeaState, omega: f64) -> f64 {
    let n = 20_000;
    let dtheta = 2.0 * PI / n as f64;

    (0..n)
        .map(|i| sea.direction(omega, -PI + (i as f64 + 0.5) * dtheta) * dtheta)
        .sum()
}

#[test]
fn directional_spreading_is_normalized() {
    for spreading in [Spreading::Hasselmann, Spreading::DonelanBanner] {
        for swell in [0.0, 0.6] {
            let sea = SeaState {
                spreading,
                swell,
                ..SeaState::default()
            };
            let wp = sea.peak_omega();

            for ratio in [0.6, 0.9, 1.0, 1.3, 2.0, 4.0] {
                let integral = spreading_integral(&sea, ratio * wp);
                assert!(
                    (integral - 1.0).abs() < 1e-6,
                    "{spreading:?} swell {swell} ratio {ratio}: {integral}"
                );
            }
        }
    }
}

#[test]
fn swell_narrows_the_spread() {
    let wind = SeaState::default();
    let swell = SeaState { swell: 1.0, ..wind };
    let wp = wind.peak_omega();

    assert!(swell.direction(wp, 0.0) > wind.direction(wp, 0.0));
    assert!(swell.spreading_exponent(wp) > wind.spreading_exponent(wp) + 10.0);
}

#[test]
fn jonswap_peaks_at_peak_frequency() {
    let sea = SeaState::default();
    let wp = sea.peak_omega();
    let samples: Vec<f64> = (1..4000).map(|i| i as f64 * 0.001).collect();
    let best = samples
        .iter()
        .copied()
        .max_by(|a, b| sea.jonswap(*a).total_cmp(&sea.jonswap(*b)))
        .unwrap();

    println!("JONSWAP peak at {best:.3} rad/s, predicted {wp:.3} rad/s");
    assert!((best - wp).abs() < 0.002);
}

#[test]
fn tma_reduces_to_jonswap_in_deep_water() {
    let sea = SeaState {
        depth: 50.0,
        ..SeaState::default()
    };
    let deep = 2.0 * (sea.gravity / sea.depth).sqrt();

    for omega in [deep * 1.01, deep * 2.0, deep * 5.0] {
        assert_eq!(sea.tma(omega), sea.jonswap(omega));
    }

    let shallow = 0.3 * (sea.gravity / sea.depth).sqrt();
    let wh = shallow * (sea.depth / sea.gravity).sqrt();
    assert!((sea.depth_attenuation(shallow) - 0.5 * wh * wh).abs() < 1e-15);
}

#[test]
fn gamma_function_matches_known_values() {
    assert!((ln_gamma(5.0) - 24f64.ln()).abs() < 1e-12);
    assert!((ln_gamma(0.5) - PI.sqrt().ln()).abs() < 1e-12);
    assert!((ln_gamma(10.5) - 1_133_278.388_948_785_5f64.ln()).abs() < 1e-10);
}
