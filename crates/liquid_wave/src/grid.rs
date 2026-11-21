#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub nx: usize,
    pub nz: usize,
    pub dx: f64,
    pub dz: f64,
}

impl Grid {
    pub fn new(nx: usize, nz: usize, dx: f64, dz: f64) -> Self {
        assert!(nx > 1 && nz > 0, "grid needs at least 2x1 cells");
        assert!(dx > 0.0 && dz > 0.0, "cell size must be positive");

        Self { nx, nz, dx, dz }
    }

    pub fn cells(&self) -> usize {
        self.nx * self.nz
    }

    pub fn cell(&self, i: usize, k: usize) -> usize {
        k * self.nx + i
    }

    pub fn x_faces(&self) -> usize {
        (self.nx + 1) * self.nz
    }

    pub fn x_face(&self, i: usize, k: usize) -> usize {
        k * (self.nx + 1) + i
    }

    pub fn z_faces(&self) -> usize {
        self.nx * (self.nz + 1)
    }

    pub fn z_face(&self, i: usize, k: usize) -> usize {
        k * self.nx + i
    }

    pub fn corners(&self) -> usize {
        (self.nx + 1) * (self.nz + 1)
    }

    pub fn corner(&self, i: usize, k: usize) -> usize {
        k * (self.nx + 1) + i
    }

    pub fn center(&self, i: usize, k: usize) -> [f64; 2] {
        [(i as f64 + 0.5) * self.dx, (k as f64 + 0.5) * self.dz]
    }

    pub fn area(&self) -> f64 {
        self.dx * self.dz
    }

    pub fn length(&self) -> [f64; 2] {
        [self.nx as f64 * self.dx, self.nz as f64 * self.dz]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CellBed {
    pub center: f64,
    pub min: f64,
    pub max: f64,
}

impl CellBed {
    fn shape(&self) -> (f64, f64) {
        let range = self.max - self.min;
        let exponent = (self.center - self.min) / (self.max - self.center) + 1.0;

        (range, exponent)
    }

    pub fn full_depth(&self) -> f64 {
        self.max - self.center
    }

    pub fn is_full(&self, h: f64) -> bool {
        h >= self.full_depth()
    }

    pub fn level(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return self.min;
        }

        if self.is_full(h) {
            return h + self.center;
        }

        let (range, exponent) = self.shape();

        self.min + range * (h * exponent / range).powf(1.0 / exponent)
    }

    pub fn depth(&self, level: f64) -> f64 {
        if level <= self.min {
            return 0.0;
        }

        if level >= self.max {
            return level - self.center;
        }

        let (range, exponent) = self.shape();

        range * ((level - self.min) / range).powf(exponent) / exponent
    }
}

#[derive(Clone, Debug)]
pub struct Bathymetry {
    pub corners: Vec<f64>,
    pub center: Vec<f64>,
    pub cell_min: Vec<f64>,
    pub cell_max: Vec<f64>,
    pub face_x: Vec<f64>,
    pub face_z: Vec<f64>,
}

impl Bathymetry {
    pub fn from_corners(grid: &Grid, corners: Vec<f64>) -> Self {
        assert_eq!(
            corners.len(),
            grid.corners(),
            "corner count must be (nx+1)*(nz+1)"
        );

        let at = |i: usize, k: usize| corners[grid.corner(i, k)];

        let mut center = vec![0.0; grid.cells()];
        let mut cell_min = vec![0.0; grid.cells()];
        let mut cell_max = vec![0.0; grid.cells()];
        let mut face_x = vec![0.0; grid.x_faces()];
        let mut face_z = vec![0.0; grid.z_faces()];

        for k in 0..grid.nz {
            for i in 0..grid.nx {
                let c = grid.cell(i, k);
                let quad = [at(i, k), at(i + 1, k), at(i, k + 1), at(i + 1, k + 1)];

                center[c] = 0.25 * (quad[0] + quad[1] + quad[2] + quad[3]);
                cell_min[c] = quad.iter().copied().fold(f64::INFINITY, f64::min);
                cell_max[c] = quad.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            }

            for i in 0..=grid.nx {
                face_x[grid.x_face(i, k)] = 0.5 * (at(i, k) + at(i, k + 1));
            }
        }

        for k in 0..=grid.nz {
            for i in 0..grid.nx {
                face_z[grid.z_face(i, k)] = 0.5 * (at(i, k) + at(i + 1, k));
            }
        }

        Self {
            corners,
            center,
            cell_min,
            cell_max,
            face_x,
            face_z,
        }
    }

    pub fn cell(&self, c: usize) -> CellBed {
        CellBed {
            center: self.center[c],
            min: self.cell_min[c],
            max: self.cell_max[c],
        }
    }

    pub fn from_fn(grid: &Grid, height: impl Fn(f64, f64) -> f64) -> Self {
        let mut corners = vec![0.0; grid.corners()];

        for k in 0..=grid.nz {
            for i in 0..=grid.nx {
                corners[grid.corner(i, k)] = height(i as f64 * grid.dx, k as f64 * grid.dz);
            }
        }

        Self::from_corners(grid, corners)
    }

    pub fn flat(grid: &Grid, height: f64) -> Self {
        Self::from_fn(grid, |_, _| height)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct State {
    pub h: Vec<f64>,
    pub hu: Vec<f64>,
    pub hv: Vec<f64>,
}

impl State {
    pub fn zeros(cells: usize) -> Self {
        Self {
            h: vec![0.0; cells],
            hu: vec![0.0; cells],
            hv: vec![0.0; cells],
        }
    }

    pub fn lake(grid: &Grid, bathymetry: &Bathymetry, level: f64) -> Self {
        let mut state = Self::zeros(grid.cells());

        for (c, h) in state.h.iter_mut().enumerate() {
            *h = bathymetry.cell(c).depth(level);
        }

        state
    }

    pub fn mass(&self, grid: &Grid) -> f64 {
        self.h.iter().sum::<f64>() * grid.area()
    }

    pub fn clear(&mut self) {
        self.h.fill(0.0);
        self.hu.fill(0.0);
        self.hv.fill(0.0);
    }
}
