use glam::{DVec3, IVec3, UVec3};

use crate::grid::{Field3, Grid, MacField, sample_grid_space};
use crate::particles::{CellIndex, Particle};
use crate::rigid::RigidBody;

pub const KERNEL_RADIUS_CELLS: f64 = 1.0;

pub const PARTICLE_RADIUS_CELLS: f64 = 0.4 * KERNEL_RADIUS_CELLS;

pub const EMPTY_PHI_CELLS: f64 = 3.0;

pub fn liquid_phi(grid: &Grid, particles: &[Particle], index: &CellIndex) -> Field3 {
    let mut phi = grid.cells();
    let radius = KERNEL_RADIUS_CELLS * grid.dx;
    let r_particle = PARTICLE_RADIUS_CELLS * grid.dx;

    phi.data = crate::exec::map(grid.cell_count(), |n| {
        let c = phi.coords(n);
        let x = grid.cell_center(c);
        zhu_bridson(grid, particles, index, c, x, radius, r_particle)
    });

    phi
}

fn zhu_bridson(
    grid: &Grid,
    particles: &[Particle],
    index: &CellIndex,
    c: UVec3,
    x: DVec3,
    radius: f64,
    r_particle: f64,
) -> f64 {
    let lo = (c.as_ivec3() - 1).max(IVec3::ZERO);
    let hi = (c.as_ivec3() + 1).min(grid.dims.as_ivec3() - 1);

    let mut wsum = 0.0;
    let mut centroid = DVec3::ZERO;

    for cz in lo.z..=hi.z {
        for cy in lo.y..=hi.y {
            for cx in lo.x..=hi.x {
                let cell =
                    (cx as u32 + grid.dims.x * (cy as u32 + grid.dims.y * cz as u32)) as usize;

                for &pi in index.cell(cell) {
                    let p = particles[pi as usize].x;
                    let s2 = (x - p).length_squared() / (radius * radius);

                    if s2 < 1.0 {
                        let k = (1.0 - s2).powi(3);
                        wsum += k;
                        centroid += k * p;
                    }
                }
            }
        }
    }

    if wsum <= 0.0 {
        return EMPTY_PHI_CELLS * grid.dx;
    }

    (x - centroid / wsum).length() - r_particle
}

pub fn solid_phi_nodes(grid: &Grid, bodies: &[RigidBody]) -> Field3 {
    let mut phi = Field3::new(grid.node_dims(), f64::MAX);

    phi.data = crate::exec::map(phi.data.len(), |n| {
        let x = phi.coords(n).as_dvec3() * grid.dx;
        bodies.iter().map(|b| b.sdf(x)).fold(f64::MAX, f64::min)
    });

    phi
}

pub fn solid_phi_at_cell(nodes: &Field3, c: UVec3) -> f64 {
    let mut sum = 0.0;

    for corner in 0..8u32 {
        let o = UVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let p = c + o;
        sum += nodes.get(p.x, p.y, p.z);
    }

    sum / 8.0
}

fn covered(v: f64) -> bool {
    v <= 0.0
}

pub fn fraction_inside_edge(a: f64, b: f64) -> f64 {
    match (covered(a), covered(b)) {
        (true, true) => 1.0,
        (true, false) => a / (a - b),
        (false, true) => b / (b - a),
        (false, false) => 0.0,
    }
}

pub fn fraction_inside_square(bl: f64, br: f64, tl: f64, tr: f64) -> f64 {
    let inside = [bl, tl, br, tr].iter().filter(|v| covered(**v)).count();
    let mut list = [bl, br, tr, tl];

    match inside {
        4 => 1.0,
        3 => {
            while covered(list[0]) {
                list.rotate_left(1);
            }

            let side0 = 1.0 - fraction_inside_edge(list[0], list[3]);
            let side1 = 1.0 - fraction_inside_edge(list[0], list[1]);

            1.0 - 0.5 * side0 * side1
        }
        2 => two_inside(list),
        1 => {
            while !covered(list[0]) {
                list.rotate_left(1);
            }

            let side0 = fraction_inside_edge(list[0], list[3]);
            let side1 = fraction_inside_edge(list[0], list[1]);

            0.5 * side0 * side1
        }
        _ => 0.0,
    }
}

fn two_inside(mut list: [f64; 4]) -> f64 {
    while !covered(list[0]) || !(covered(list[1]) || covered(list[2])) {
        list.rotate_left(1);
    }

    if covered(list[1]) {
        let left = fraction_inside_edge(list[0], list[3]);
        let right = fraction_inside_edge(list[1], list[2]);

        return 0.5 * (left + right);
    }

    let middle = 0.25 * list.iter().sum::<f64>();

    if middle < 0.0 {
        let side1 = 1.0 - fraction_inside_edge(list[0], list[3]);
        let side3 = 1.0 - fraction_inside_edge(list[2], list[3]);
        let side2 = 1.0 - fraction_inside_edge(list[2], list[1]);
        let side0 = 1.0 - fraction_inside_edge(list[0], list[1]);

        1.0 - 0.5 * side1 * side3 - 0.5 * side0 * side2
    } else {
        let side0 = fraction_inside_edge(list[0], list[1]);
        let side1 = fraction_inside_edge(list[0], list[3]);
        let side2 = fraction_inside_edge(list[2], list[1]);
        let side3 = fraction_inside_edge(list[2], list[3]);

        0.5 * side0 * side1 + 0.5 * side2 * side3
    }
}

pub fn face_weights(grid: &Grid, nodes: &Field3) -> MacField {
    let mut weights = MacField::new(grid);

    for axis in 0..3 {
        let field = &mut weights.axes[axis];
        let (t1, t2) = ((axis + 1) % 3, (axis + 2) % 3);

        field.data = crate::exec::map(field.data.len(), |n| {
            let f = field.coords(n);

            if f[axis] == 0 {
                return 0.0;
            }

            if f[axis] == grid.dims[axis] {
                return 0.0;
            }

            let corner = |a: u32, b: u32| {
                let mut p = f;
                p[t1] += a;
                p[t2] += b;
                nodes.get(p.x, p.y, p.z)
            };

            1.0 - fraction_inside_square(corner(0, 0), corner(1, 0), corner(0, 1), corner(1, 1))
        });
    }

    weights
}

pub fn face_owners(grid: &Grid, bodies: &[RigidBody], weights: &MacField) -> [Vec<i32>; 3] {
    std::array::from_fn(|axis| {
        let w = &weights.axes[axis];

        crate::exec::map(w.data.len(), |n| {
            if w.data[n] >= 1.0 {
                return -1;
            }

            let x = grid.face_position(axis, w.coords(n));

            let reach = std::f64::consts::FRAC_1_SQRT_2 * grid.dx;

            bodies
                .iter()
                .enumerate()
                .map(|(i, b)| (i, b.sdf(x)))
                .filter(|(_, s)| *s < reach)
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .map_or(-1, |(i, _)| i as i32)
        })
    })
}

pub const SOLID_EXTRAPOLATION_PASSES: usize = 3;

pub fn fill_solid_cells(phi: &mut Field3, nodes: &Field3) {
    let mut known: Vec<bool> = (0..phi.data.len())
        .map(|n| solid_phi_at_cell(nodes, phi.coords(n)) >= 0.0)
        .collect();
    let dims = phi.dims.as_ivec3();

    for _ in 0..SOLID_EXTRAPOLATION_PASSES {
        let mut updates = Vec::new();

        for n in 0..phi.data.len() {
            if known[n] {
                continue;
            }

            let c = phi.coords(n).as_ivec3();
            let mut sum = 0.0;
            let mut count = 0.0;

            for axis in 0..3 {
                for step in [-1, 1] {
                    let mut p = c;
                    p[axis] += step;

                    if p.cmplt(IVec3::ZERO).any() || p.cmpge(dims).any() {
                        continue;
                    }

                    let m = phi.index(p.x as u32, p.y as u32, p.z as u32);

                    if known[m] {
                        sum += phi.data[m];
                        count += 1.0;
                    }
                }
            }

            if count > 0.0 {
                updates.push((n, sum / count));
            }
        }

        for (n, v) in updates {
            phi.data[n] = v;
            known[n] = true;
        }
    }
}

pub fn redistance(phi: &Field3, dx: f64) -> Field3 {
    let mut dist: Vec<f64> = (0..phi.data.len())
        .map(|n| interface_distance(phi, n, dx).unwrap_or(f64::INFINITY))
        .collect();
    let fixed: Vec<bool> = dist.iter().map(|d| d.is_finite()).collect();
    let dims = phi.dims.as_ivec3();

    for sweep in 0..8 {
        let flip = IVec3::new(sweep & 1, (sweep >> 1) & 1, (sweep >> 2) & 1);

        for k in 0..dims.z {
            for j in 0..dims.y {
                for i in 0..dims.x {
                    let c = IVec3::new(i, j, k);
                    let c = IVec3::select(flip.cmpeq(IVec3::ONE), dims - 1 - c, c);
                    let n = phi.index(c.x as u32, c.y as u32, c.z as u32);

                    if !fixed[n] {
                        dist[n] = dist[n].min(eikonal_update(phi, &dist, c, dx));
                    }
                }
            }
        }
    }

    let far = dims.as_dvec3().length() * dx;
    let mut out = phi.clone();

    for (o, d) in out.data.iter_mut().zip(dist) {
        let d = d.min(far);
        *o = if *o < 0.0 { -d } else { d };
    }

    let swept = out.clone();

    out.data = crate::exec::map(swept.data.len(), |n| {
        let d = swept.data[n];

        if d.abs() > PROJECTION_BAND_CELLS * dx {
            return d;
        }

        surface_distance(phi, &swept, n, dx).map_or(d, |s| s.copysign(d))
    });

    out
}

pub const PROJECTION_BAND_CELLS: f64 = 3.0;

fn surface_distance(input: &Field3, swept: &Field3, n: usize, dx: f64) -> Option<f64> {
    let start = swept.coords(n).as_dvec3();
    let normal = DVec3::new(
        sample_grid_space(swept, start + DVec3::X * 0.5)
            - sample_grid_space(swept, start - DVec3::X * 0.5),
        sample_grid_space(swept, start + DVec3::Y * 0.5)
            - sample_grid_space(swept, start - DVec3::Y * 0.5),
        sample_grid_space(swept, start + DVec3::Z * 0.5)
            - sample_grid_space(swept, start - DVec3::Z * 0.5),
    )
    .try_normalize()?;

    let inside = |t: f64| sample_grid_space(input, start + normal * t) < 0.0;
    let origin = inside(0.0);
    let reach = if origin { 2.0 } else { -2.0 } * (swept.data[n] / dx).abs().max(0.5);

    if inside(reach) == origin {
        return None;
    }

    let (mut near, mut far) = (0.0, reach);

    for _ in 0..40 {
        let mid = 0.5 * (near + far);

        if inside(mid) == origin {
            near = mid;
        } else {
            far = mid;
        }
    }

    Some((0.5 * (near + far)).abs() * dx)
}

fn interface_distance(phi: &Field3, n: usize, dx: f64) -> Option<f64> {
    let c = phi.coords(n).as_ivec3();
    let v = phi.data[n];
    let mut inverse_square = 0.0;
    let mut crossed = false;

    for axis in 0..3 {
        let mut nearest = f64::INFINITY;

        for side in [-1, 1] {
            let mut p = c;
            p[axis] += side;

            let Some(w) = phi.at(p) else {
                continue;
            };

            if (w < 0.0) != (v < 0.0) {
                crossed = true;
                nearest = nearest.min(v / (v - w) * dx);
            } else if w.abs() < v.abs() {
                nearest = nearest.min(v.abs() / (v.abs() - w.abs()) * dx);
            }
        }

        if nearest.is_finite() {
            inverse_square += 1.0 / (nearest * nearest);
        }
    }

    crossed.then(|| 1.0 / inverse_square.sqrt())
}

fn eikonal_update(phi: &Field3, dist: &[f64], c: IVec3, dx: f64) -> f64 {
    let mut a = [f64::INFINITY; 3];

    for (axis, slot) in a.iter_mut().enumerate() {
        for side in [-1, 1] {
            let mut p = c;
            p[axis] += side;

            if phi.contains(p) {
                *slot = slot.min(dist[phi.index(p.x as u32, p.y as u32, p.z as u32)]);
            }
        }
    }

    a.sort_by(f64::total_cmp);

    let d = a[0] + dx;

    if d <= a[1] {
        return d;
    }

    let d = 0.5 * (a[0] + a[1] + (2.0 * dx * dx - (a[0] - a[1]).powi(2)).sqrt());

    if d <= a[2] {
        return d;
    }

    let s = a[0] + a[1] + a[2];
    let q = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];

    (s + (s * s - 3.0 * (q - dx * dx)).sqrt()) / 3.0
}
