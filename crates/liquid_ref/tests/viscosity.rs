use glam::{DVec3, UVec3};
use liquid_ref::grid::{Grid, MacField};
use liquid_ref::levelset::{face_weights, solid_phi_nodes};
use liquid_ref::scene::dam_break;
use liquid_ref::sim::{Params, Simulation};
use liquid_ref::viscosity::{Fractions, ViscositySolve};

const LAMB_ROOT: f64 = 5.763_459_196_894_55;

const RADIUS: f64 = 0.1;

const NU: f64 = 0.01;

struct Ball {
    grid: Grid,

    fractions: Fractions,

    weights: MacField,

    center: DVec3,
}

impl Ball {
    fn new(cells_per_radius: u32) -> Self {
        let n = 2 * cells_per_radius + 6;
        let grid = Grid::new(UVec3::splat(n), RADIUS / cells_per_radius as f64);
        let center = grid.size() * 0.5;

        let mut phi = grid.cells();

        for i in 0..phi.data.len() {
            phi.data[i] = (grid.cell_center(phi.coords(i)) - center).length() - RADIUS;
        }

        let weights = face_weights(&grid, &solid_phi_nodes(&grid, &[]));
        let fractions = Fractions::new(&grid, &phi);

        Self {
            grid,
            fractions,
            weights,
            center,
        }
    }

    fn field(&self, f: impl Fn(DVec3) -> DVec3) -> MacField {
        let mut u = MacField::new(&self.grid);

        for (axis, field) in u.axes.iter_mut().enumerate() {
            for n in 0..field.data.len() {
                if self.fractions.faces.axes[axis].data[n] > 0.0 {
                    let x = self.grid.face_position(axis, field.coords(n));
                    field.data[n] = f(x - self.center)[axis];
                }
            }
        }

        u
    }

    fn inner(&self, a: &MacField, b: &MacField) -> f64 {
        (0..3)
            .map(|axis| {
                let w = &self.fractions.faces.axes[axis].data;
                let (a, b) = (&a.axes[axis].data, &b.axes[axis].data);

                (0..w.len()).map(|n| w[n] * a[n] * b[n]).sum::<f64>()
            })
            .sum()
    }

    fn angular_momentum(&self, u: &MacField) -> f64 {
        self.inner(u, &self.field(|r| DVec3::new(-r.y, r.x, 0.0)))
    }

    fn diffuse(&self, u: &mut MacField, dt: f64) {
        let stats = ViscositySolve {
            grid: &self.grid,
            fractions: &self.fractions,
            weights: &self.weights,
            kinematic_viscosity: NU,
            dt,
            tolerance: 1e-10,
            max_iterations: 2000,
        }
        .apply(u);

        assert!(
            stats.iterations < 2000,
            "viscosity solve did not converge: {stats:?}"
        );
    }
}

fn spherical_j1(x: f64) -> f64 {
    if x < 1e-6 {
        x / 3.0
    } else {
        x.sin() / (x * x) - x.cos() / x
    }
}

fn toroidal_decay_rate(cells_per_radius: u32) -> f64 {
    let ball = Ball::new(cells_per_radius);
    let k = LAMB_ROOT / RADIUS;
    let lambda = NU * k * k;
    let dt = 0.1 / lambda;
    let solves = 8;

    let mode = ball.field(|r| {
        let s = r.length();
        let swirl = DVec3::new(-r.y, r.x, 0.0);

        if s > 0.0 {
            swirl * spherical_j1(k * s) / s
        } else {
            DVec3::ZERO
        }
    });
    let mut u = mode.clone();

    for _ in 0..solves {
        ball.diffuse(&mut u, dt);
    }

    let amplitude = ball.inner(&u, &mode) / ball.inner(&mode, &mode);

    ((1.0 / amplitude).powf(1.0 / solves as f64) - 1.0) / dt
}

#[test]
fn free_surface_toroidal_mode_decays_at_lamb_rate() {
    let k = LAMB_ROOT / RADIUS;
    let exact = NU * k * k;
    let rates: Vec<(u32, f64)> = [5, 10, 20]
        .iter()
        .map(|&c| (c, toroidal_decay_rate(c)))
        .collect();
    let errors: Vec<f64> = rates
        .iter()
        .map(|(_, r)| (r - exact).abs() / exact)
        .collect();

    for ((c, r), e) in rates.iter().zip(&errors) {
        println!(
            "radius {c} cells: decay rate {r:.4} 1/s vs Lamb {exact:.4} 1/s, error {:.2}%",
            100.0 * e
        );
    }

    assert!(errors[1] < errors[0] && errors[2] < errors[1]);
    assert!(errors[2] < 0.05);
}

#[test]
fn rigid_rotation_survives_strong_viscosity() {
    let ball = Ball::new(10);
    let dt = 50.0 * ball.grid.dx * ball.grid.dx / NU;
    let mut u = ball.field(|r| DVec3::new(-r.y, r.x, 0.0));

    let l0 = ball.angular_momentum(&u);

    for _ in 0..5 {
        ball.diffuse(&mut u, dt);
    }

    let ratio = ball.angular_momentum(&u) / l0;

    println!("angular momentum kept after 5 solves at dt = 50 dx^2/nu: {ratio:.9}");

    assert!((ratio - 1.0).abs() < 1e-6);
}

fn front_position(sim: &Simulation, height: f64) -> f64 {
    sim.particles
        .iter()
        .filter(|p| p.x.y < 0.25 * height)
        .fold(0.0, |m, p| m.max(p.x.x))
}

struct ColumnRun {
    front: f64,

    max_speed: f64,

    min_dt: f64,

    max_iterations: usize,

    finite: bool,
}

fn collapse_column(viscosity: f64) -> ColumnRun {
    let dx = 0.005;
    let column = DVec3::new(0.06, 0.1, 0.0);
    let params = Params {
        viscosity,
        ..Params::default()
    };
    let mut sim = dam_break(UVec3::new(40, 32, 4), dx, column, params);

    let mut run = ColumnRun {
        front: 0.0,
        max_speed: 0.0,
        min_dt: f64::INFINITY,
        max_iterations: 0,
        finite: true,
    };

    while sim.time < 0.4 {
        let dt = sim.cfl_dt();
        sim.substep(dt);

        run.min_dt = run.min_dt.min(dt);
        run.max_iterations = run
            .max_iterations
            .max(sim.report.viscosity.map_or(0, |s| s.iterations));
        run.finite &= sim
            .particles
            .iter()
            .all(|p| p.x.is_finite() && p.v.is_finite());
        run.max_speed = sim
            .particles
            .iter()
            .fold(run.max_speed, |m, p| m.max(p.v.length()));
    }

    run.front = front_position(&sim, column.y);

    run
}

#[test]
fn stiff_viscosity_is_stable_far_above_explicit_limit() {
    let mu = 10.0;
    let nu = mu / 1000.0;
    let dx: f64 = 0.005;
    let explicit_limit = dx * dx / (6.0 * nu);

    let viscous = collapse_column(mu);
    let inviscid = collapse_column(0.0);
    let free_fall = (2.0 * 9.81 * 0.1f64).sqrt();

    println!(
        "mu {mu} Pa s: min dt {:.2e} s = {:.1}x explicit limit {explicit_limit:.2e} s, max cg iterations {}, max speed {:.3} m/s (free fall {free_fall:.3}), front {:.4} m vs inviscid {:.4} m from 0.06 m",
        viscous.min_dt,
        viscous.min_dt / explicit_limit,
        viscous.max_iterations,
        viscous.max_speed,
        viscous.front,
        inviscid.front,
    );

    assert!(viscous.finite);
    assert!(viscous.min_dt > 10.0 * explicit_limit);
    assert!(viscous.max_iterations < 200);
    assert!(viscous.max_speed < free_fall);
    assert!(viscous.front - 0.06 < 0.5 * (inviscid.front - 0.06));
}
