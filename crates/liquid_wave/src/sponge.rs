use crate::airy::AmbientField;
use crate::grid::{Bathymetry, Grid, State};
use crate::swe::Side;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmbientFrame {
    pub origin: [f64; 2],
    pub level: f64,
}

impl AmbientFrame {
    pub fn cell_state(
        &self,
        grid: &Grid,
        bathymetry: &Bathymetry,
        field: &AmbientField,
        c: usize,
        t: f64,
    ) -> [f64; 3] {
        let [x, z] = grid.center(c % grid.nx, c / grid.nx);
        let (x, z) = (x + self.origin[0], z + self.origin[1]);
        let eta = field.elevation(x, z, t);
        let q = field.flux(x, z, t);

        [bathymetry.cell(c).depth(self.level + eta), q[0], q[1]]
    }

    pub fn state(
        &self,
        grid: &Grid,
        bathymetry: &Bathymetry,
        field: &AmbientField,
        t: f64,
    ) -> State {
        let mut state = State::zeros(grid.cells());

        for c in 0..grid.cells() {
            let [h, hu, hv] = self.cell_state(grid, bathymetry, field, c, t);
            state.h[c] = h;
            state.hu[c] = hu;
            state.hv[c] = hv;
        }

        state
    }
}

#[derive(Clone, Debug)]
pub struct Sponge {
    pub widths: [f64; 4],
    pub strength: f64,
    pub power: f64,
    pub sigma: Vec<f64>,
    pub target: State,
    pub ambient: Option<(AmbientFrame, AmbientField)>,
}

impl Sponge {
    pub fn new(grid: &Grid, widths: [f64; 4], strength: f64, power: f64, target: State) -> Self {
        assert_eq!(
            target.h.len(),
            grid.cells(),
            "sponge target must cover the grid"
        );

        let mut sigma = vec![0.0; grid.cells()];
        let [lx, lz] = grid.length();

        for k in 0..grid.nz {
            for i in 0..grid.nx {
                let [x, z] = grid.center(i, k);
                let distances = [x, lx - x, z, lz - z];

                sigma[grid.cell(i, k)] = Side::ALL
                    .iter()
                    .map(|s| profile(distances[s.index()], widths[s.index()], strength, power))
                    .fold(0.0, f64::max);
            }
        }

        Self {
            widths,
            strength,
            power,
            sigma,
            target,
            ambient: None,
        }
    }

    pub fn update_target(&mut self, grid: &Grid, bathymetry: &Bathymetry, t: f64) {
        let Some((frame, field)) = &self.ambient else {
            return;
        };

        for c in 0..grid.cells() {
            if self.sigma[c] > 0.0 {
                let [h, hu, hv] = frame.cell_state(grid, bathymetry, field, c, t);
                self.target.h[c] = h;
                self.target.hu[c] = hu;
                self.target.hv[c] = hv;
            }
        }
    }

    pub fn apply(&self, state: &mut State, dt: f64) {
        for (c, &s) in self.sigma.iter().enumerate() {
            if s <= 0.0 {
                continue;
            }

            let keep = 1.0 / (1.0 + s * dt);

            state.h[c] = self.target.h[c] + (state.h[c] - self.target.h[c]) * keep;
            state.hu[c] = self.target.hu[c] + (state.hu[c] - self.target.hu[c]) * keep;
            state.hv[c] = self.target.hv[c] + (state.hv[c] - self.target.hv[c]) * keep;
        }
    }
}

pub fn profile(distance: f64, width: f64, strength: f64, power: f64) -> f64 {
    if width <= 0.0 || distance >= width {
        return 0.0;
    }

    strength * ((width - distance) / width).powf(power)
}
