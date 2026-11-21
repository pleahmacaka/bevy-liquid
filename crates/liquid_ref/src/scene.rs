use glam::{DVec3, UVec3};

use crate::grid::Grid;
use crate::particles::{Particle, seed_box};
use crate::rigid::RigidBody;
use crate::sim::{Params, Simulation};

pub const SEED_JITTER: f64 = 0.8;

pub fn tank(dims: UVec3, dx: f64, water_depth: f64, params: Params) -> Simulation {
    let grid = Grid::new(dims, dx);
    let mut hi = grid.size();
    hi.y = water_depth;

    let particles = seed_box(&grid, DVec3::ZERO, hi, SEED_JITTER);

    Simulation::new(grid, params, particles, Vec::new())
}

pub fn dam_break(dims: UVec3, dx: f64, column: DVec3, params: Params) -> Simulation {
    let grid = Grid::new(dims, dx);
    let mut hi = column;
    hi.z = grid.size().z;

    let particles = seed_box(&grid, DVec3::ZERO, hi, SEED_JITTER);

    Simulation::new(grid, params, particles, Vec::new())
}

pub fn with_body(mut sim: Simulation, body: RigidBody) -> Simulation {
    sim.bodies.push(body);
    sim.particles.retain(|p| body.sdf(p.x) > 0.0);
    sim.refresh_phi();

    sim
}

pub fn moving(particles: Vec<Particle>, velocity: impl Fn(DVec3) -> DVec3) -> Vec<Particle> {
    particles
        .into_iter()
        .map(|p| Particle {
            v: velocity(p.x),
            ..p
        })
        .collect()
}
