use glam::{DMat3, DVec3, UVec3};
use liquid_ref::grid::{Grid, MacField};
use liquid_ref::particles::{CellIndex, Particle, seed_box};
use liquid_ref::transfer::{Transfer, g2p, p2g};

fn affine(x: DVec3) -> DVec3 {
    let b = DMat3::from_cols(
        DVec3::new(0.3, -1.1, 0.4),
        DVec3::new(0.7, 0.2, -0.5),
        DVec3::new(-0.6, 0.9, 0.1),
    );

    DVec3::new(0.5, -0.25, 1.5) + b * x
}

fn gradient_row(axis: usize) -> DVec3 {
    let h = 1e-3;

    DVec3::new(
        (affine(DVec3::X * h)[axis] - affine(-DVec3::X * h)[axis]) / (2.0 * h),
        (affine(DVec3::Y * h)[axis] - affine(-DVec3::Y * h)[axis]) / (2.0 * h),
        (affine(DVec3::Z * h)[axis] - affine(-DVec3::Z * h)[axis]) / (2.0 * h),
    )
}

#[test]
fn g2p_recovers_affine_field_exactly() {
    let grid = Grid::new(UVec3::splat(12), 0.1);
    let mut velocity = MacField::new(&grid);

    for (axis, field) in velocity.axes.iter_mut().enumerate() {

        for n in 0..field.data.len() {
            let x = grid.face_position(axis, field.coords(n));
            field.data[n] = affine(x)[axis];
        }
    }

    let mut particles = seed_box(&grid, DVec3::splat(0.3), DVec3::splat(0.9), 0.8);
    g2p(&grid, &velocity, &mut particles, Transfer::Apic);

    for p in &particles {
        assert!((p.v - affine(p.x)).length() < 1e-12);

        for (axis, c) in p.c.iter().enumerate() {
            assert!((*c - gradient_row(axis)).length() < 1e-7);
        }
    }
}

#[test]
fn p2g_reproduces_affine_field_exactly() {
    let grid = Grid::new(UVec3::splat(12), 0.1);
    let particles: Vec<Particle> = seed_box(&grid, DVec3::ZERO, grid.size(), 0.8)
        .into_iter()
        .map(|p| Particle {
            x: p.x,
            v: affine(p.x),
            c: [gradient_row(0), gradient_row(1), gradient_row(2)],
        })
        .collect();

    let index = CellIndex::build(&grid, &particles);
    let out = p2g(&grid, &particles, &index);

    let mut worst = 0.0f64;

    for (axis, field) in out.velocity.axes.iter().enumerate() {

        for n in 0..field.data.len() {
            let f = field.coords(n);

            if f.min_element() < 2 || f.max_element() > 9 {
                continue;
            }

            let x = grid.face_position(axis, f);
            worst = worst.max((field.data[n] - affine(x)[axis]).abs());
        }
    }

    assert!(worst < 1e-7, "max error {worst}");
}

#[test]
fn p2g_conserves_mass_and_momentum() {
    let grid = Grid::new(UVec3::splat(10), 0.1);
    let particles: Vec<Particle> = seed_box(&grid, DVec3::splat(0.25), DVec3::splat(0.75), 0.8)
        .into_iter()
        .map(|p| Particle {
            x: p.x,
            v: DVec3::new(p.x.y, -p.x.z, p.x.x),
            c: [DVec3::ZERO; 3],
        })
        .collect();

    let index = CellIndex::build(&grid, &particles);
    let out = p2g(&grid, &particles, &index);

    for axis in 0..3 {
        let mass: f64 = out.mass.axes[axis].data.iter().sum();
        let momentum: f64 = out.mass.axes[axis]
            .data
            .iter()
            .zip(&out.velocity.axes[axis].data)
            .map(|(m, v)| m * v)
            .sum();
        let expected: f64 = particles.iter().map(|p| p.v[axis]).sum();

        assert!((mass - particles.len() as f64).abs() < 1e-9);
        assert!((momentum - expected).abs() < 1e-9);
    }
}
