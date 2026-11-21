use liquid_wave::GRAVITY;
use liquid_wave::analytic::DamBreak;

#[test]
fn stoker_middle_state_satisfies_rankine_hugoniot() {
    let problem = DamBreak {
        gravity: GRAVITY,
        dam: 0.0,
        upstream: 1.0,
        downstream: 0.1,
    };
    let m = problem.stoker_middle();
    let (h1, g) = (problem.downstream, GRAVITY);

    let mass = m.shock_speed * (m.depth - h1) - m.depth * m.velocity;
    let momentum = m.shock_speed * m.depth * m.velocity
        - (m.depth * m.velocity * m.velocity + 0.5 * g * (m.depth * m.depth - h1 * h1));
    let riemann = m.velocity + 2.0 * (g * m.depth).sqrt() - 2.0 * (g * problem.upstream).sqrt();

    println!(
        "Stoker middle state: h = {:.6}, u = {:.6}, shock speed = {:.6}",
        m.depth, m.velocity, m.shock_speed
    );
    assert!(mass.abs() < 1e-12 && momentum.abs() < 1e-12 && riemann.abs() < 1e-12);
}

#[test]
fn ritter_front_moves_at_twice_the_wave_speed() {
    let problem = DamBreak {
        gravity: GRAVITY,
        dam: 0.0,
        upstream: 2.0,
        downstream: 0.0,
    };
    let c0 = (GRAVITY * 2.0).sqrt();

    assert!(problem.depth(2.0 * c0 * 1.0 - 1e-9, 1.0) < 1e-12);
    assert_eq!(problem.depth(2.0 * c0 + 1e-6, 1.0), 0.0);
    assert!((problem.depth(-c0, 1.0) - 2.0).abs() < 1e-12);
    assert!((problem.depth(0.0, 1.0) - 4.0 / 9.0 * 2.0).abs() < 1e-12);
}
