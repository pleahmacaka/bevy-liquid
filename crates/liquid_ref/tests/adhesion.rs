use glam::{DVec3, UVec3};
use liquid_ref::adhesion::{RANGE_CELLS, profile};
use liquid_ref::grid::Grid;
use liquid_ref::particles::seed_box;
use liquid_ref::rigid::{Motion, RigidBody, Shape};
use liquid_ref::scene::SEED_JITTER;
use liquid_ref::sim::{Params, Simulation};

const DX: f64 = 0.025;

const CEILING: f64 = 0.3;

const FILM: f64 = 0.075;

#[test]
fn profile_pulls_with_unit_area_inside_the_range() {
    let samples = 100_000;
    let q = |i: usize| (i as f64 + 0.5) / samples as f64;
    let area = (0..samples).map(|i| profile(q(i))).sum::<f64>() / samples as f64;
    let peak = (0..samples).map(q).max_by(|a, b| profile(*a).total_cmp(&profile(*b))).unwrap_or(0.0);

    println!("profile area {area:.6}, peak at q = {peak:.3}");

    assert!((area - 1.0).abs() < 1e-4);
    assert!((0..samples).all(|i| profile(q(i)) >= 0.0));
    assert!([-0.5, 0.0, 1.0, 1.5].iter().all(|q| profile(*q) == 0.0));
    assert!((peak - 0.54).abs() < 0.01);
}

fn ceiling(adhesion: f64) -> Simulation {
    let grid = Grid::new(UVec3::new(16, 16, 8), DX);
    let size = grid.size();
    let slab = RigidBody::new(
        Shape::Cuboid { half: DVec3::new(size.x, 0.05, size.z) },
        DVec3::new(0.5 * size.x, CEILING + 0.05, 0.5 * size.z),
        Motion::Static,
        1000.0,
    );
    let lo = DVec3::new(0.1, CEILING - FILM, 0.0);
    let hi = DVec3::new(size.x - 0.1, CEILING, size.z);
    let particles = seed_box(&grid, lo, hi, SEED_JITTER);
    let params = Params { adhesion, ..Params::default() };

    Simulation::new(grid, params, particles, vec![slab])
}

fn attached(sim: &Simulation) -> f64 {
    let reach = CEILING - FILM - RANGE_CELLS * DX;
    let near = sim.particles.iter().filter(|p| p.x.y > reach).count();

    near as f64 / sim.particles.len() as f64
}

#[test]
fn adhesion_holds_a_film_under_a_ceiling() {
    let mut free = ceiling(0.0);
    let mut sticky = ceiling(1.0);

    free.advance(1.0);
    sticky.advance(1.0);

    println!(
        "film still within reach of the ceiling after 1 s: {:.3} without adhesion, {:.3} with 1 J/kg",
        attached(&free),
        attached(&sticky)
    );

    assert!(attached(&sticky) > 0.85);
    assert!(attached(&free) < 0.1);
}
