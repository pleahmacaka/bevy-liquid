use glam::{DVec3, UVec3};
use liquid_ref::grid::Grid;
use liquid_ref::levelset::{face_weights, redistance, solid_phi_nodes};
use liquid_ref::rigid::{Motion, RigidBody, Shape};

#[test]
fn redistance_recovers_sphere_distance_from_distorted_field() {
    let n = 32;
    let grid = Grid::new(UVec3::splat(n), 1.0 / n as f64);
    let center = DVec3::splat(0.5);
    let radius = 0.3;

    let mut phi = grid.cells();

    for i in 0..phi.data.len() {
        let r = (grid.cell_center(phi.coords(i)) - center).length();
        phi.data[i] = (r * r - radius * radius) * 7.0;
    }

    let distance = redistance(&phi, grid.dx);
    let band = 2.5 * grid.dx;
    let worst = distance
        .data
        .iter()
        .enumerate()
        .fold(0.0f64, |worst, (i, d)| {
            let exact = (grid.cell_center(phi.coords(i)) - center).length() - radius;

            if exact.abs() < band {
                worst.max((d - exact).abs())
            } else {
                worst
            }
        });

    println!(
        "max redistance error within 2.5 cells: {:.3} dx",
        worst / grid.dx
    );

    assert!(worst < 0.05 * grid.dx);
}

#[test]
fn grid_aligned_box_closes_covered_faces_and_opens_touching_ones() {
    let grid = Grid::new(UVec3::splat(16), 1.0 / 32.0);
    let body = RigidBody::new(
        Shape::Cuboid {
            half: DVec3::new(0.125, 0.0625, 0.125),
        },
        DVec3::new(0.25, 0.125, 0.25),
        Motion::Static,
        1000.0,
    );
    let nodes = solid_phi_nodes(&grid, &[body]);
    let weights = face_weights(&grid, &nodes);
    let mut covered = 0;
    let mut touching = 0;

    for axis in 0..3 {
        let field = &weights.axes[axis];

        for n in 0..field.data.len() {
            let f = field.coords(n);

            if f[axis] == 0 || f[axis] == grid.dims[axis] {
                continue;
            }

            let corners: Vec<f64> = (0..4u32)
                .map(|k| {
                    let mut p = f;
                    p[(axis + 1) % 3] += k & 1;
                    p[(axis + 2) % 3] += k >> 1;
                    nodes.get(p.x, p.y, p.z)
                })
                .collect();
            let lo = corners.iter().copied().fold(f64::MAX, f64::min);
            let hi = corners.iter().copied().fold(f64::MIN, f64::max);

            if hi <= 0.0 {
                covered += 1;
                assert_eq!(field.data[n], 0.0, "covered face {axis} {f} open");
            } else if lo >= 0.0 && corners.contains(&0.0) {
                touching += 1;
                assert_eq!(field.data[n], 1.0, "touching face {axis} {f} blocked");
            }
        }
    }

    println!("covered faces {covered}, touching faces {touching}");

    assert!(covered > 0 && touching > 0);
}
