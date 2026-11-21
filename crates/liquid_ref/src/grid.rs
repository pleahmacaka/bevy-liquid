use glam::{DVec3, IVec3, UVec3};

#[derive(Clone, Debug, PartialEq)]
pub struct Field3 {
    pub dims: UVec3,

    pub data: Vec<f64>,
}

impl Field3 {
    pub fn new(dims: UVec3, value: f64) -> Self {
        let len = (dims.x * dims.y * dims.z) as usize;

        Self {
            dims,
            data: vec![value; len],
        }
    }

    pub fn index(&self, i: u32, j: u32, k: u32) -> usize {
        (i + self.dims.x * (j + self.dims.y * k)) as usize
    }

    pub fn contains(&self, p: IVec3) -> bool {
        p.x >= 0
            && p.y >= 0
            && p.z >= 0
            && (p.x as u32) < self.dims.x
            && (p.y as u32) < self.dims.y
            && (p.z as u32) < self.dims.z
    }

    pub fn get(&self, i: u32, j: u32, k: u32) -> f64 {
        self.data[self.index(i, j, k)]
    }

    pub fn set(&mut self, i: u32, j: u32, k: u32, value: f64) {
        let n = self.index(i, j, k);
        self.data[n] = value;
    }

    pub fn at(&self, p: IVec3) -> Option<f64> {
        self.contains(p)
            .then(|| self.get(p.x as u32, p.y as u32, p.z as u32))
    }

    pub fn coords(&self, n: usize) -> UVec3 {
        let n = n as u32;
        let i = n % self.dims.x;
        let j = (n / self.dims.x) % self.dims.y;
        let k = n / (self.dims.x * self.dims.y);

        UVec3::new(i, j, k)
    }

    pub fn fill(&mut self, value: f64) {
        self.data.fill(value);
    }

    pub fn max_abs(&self) -> f64 {
        self.data.iter().fold(0.0, |m, v| m.max(v.abs()))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub dims: UVec3,

    pub dx: f64,
}

impl Grid {
    pub fn new(dims: UVec3, dx: f64) -> Self {
        Self { dims, dx }
    }

    pub fn size(&self) -> DVec3 {
        self.dims.as_dvec3() * self.dx
    }

    pub fn cell_count(&self) -> usize {
        (self.dims.x * self.dims.y * self.dims.z) as usize
    }

    pub fn cell_center(&self, c: UVec3) -> DVec3 {
        (c.as_dvec3() + 0.5) * self.dx
    }

    pub fn face_dims(&self, axis: usize) -> UVec3 {
        let mut d = self.dims;
        d[axis] += 1;

        d
    }

    pub fn face_offset(axis: usize) -> DVec3 {
        let mut o = DVec3::splat(0.5);
        o[axis] = 0.0;

        o
    }

    pub fn face_position(&self, axis: usize, f: UVec3) -> DVec3 {
        (f.as_dvec3() + Self::face_offset(axis)) * self.dx
    }

    pub fn node_dims(&self) -> UVec3 {
        self.dims + 1
    }

    pub fn cell_of(&self, x: DVec3) -> IVec3 {
        (x / self.dx).floor().as_ivec3()
    }

    pub fn cells(&self) -> Field3 {
        Field3::new(self.dims, 0.0)
    }

    pub fn faces(&self, axis: usize) -> Field3 {
        Field3::new(self.face_dims(axis), 0.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MacField {
    pub axes: [Field3; 3],
}

impl MacField {
    pub fn new(grid: &Grid) -> Self {
        Self {
            axes: [grid.faces(0), grid.faces(1), grid.faces(2)],
        }
    }

    pub fn fill(&mut self, value: f64) {
        for a in &mut self.axes {
            a.fill(value);
        }
    }

    pub fn sample(&self, grid: &Grid, x: DVec3) -> DVec3 {
        DVec3::new(
            sample_trilinear(&self.axes[0], grid, x, 0),
            sample_trilinear(&self.axes[1], grid, x, 1),
            sample_trilinear(&self.axes[2], grid, x, 2),
        )
    }

    pub fn max_abs(&self) -> f64 {
        self.axes.iter().fold(0.0, |m, a| m.max(a.max_abs()))
    }
}

pub fn sample_trilinear(field: &Field3, grid: &Grid, x: DVec3, axis: usize) -> f64 {
    let g = x / grid.dx - Grid::face_offset(axis);
    sample_grid_space(field, g)
}

pub fn sample_cells(field: &Field3, grid: &Grid, x: DVec3) -> f64 {
    sample_grid_space(field, x / grid.dx - 0.5)
}

pub fn sample_grid_space(field: &Field3, g: DVec3) -> f64 {
    trilinear_corners(field, g)
        .iter()
        .fold(0.0, |acc, (w, v)| acc + w * v)
}

pub fn trilinear_corners(field: &Field3, g: DVec3) -> [(f64, f64); 8] {
    let max = (field.dims.as_dvec3() - 1.0).max(DVec3::ZERO);
    let g = g.clamp(DVec3::ZERO, max);
    let base = g.floor().min((max - 1.0).max(DVec3::ZERO));
    let t = g - base;
    let b = base.as_uvec3();
    let hi = (b + 1).min(field.dims - 1);

    std::array::from_fn(|corner| {
        let corner = corner as u32;
        let pick = UVec3::new(corner & 1, (corner >> 1) & 1, (corner >> 2) & 1);
        let p = UVec3::select(pick.cmpeq(UVec3::ONE), hi, b);
        let w = DVec3::select(pick.cmpeq(UVec3::ONE), t, 1.0 - t);

        (w.x * w.y * w.z, field.get(p.x, p.y, p.z))
    })
}
