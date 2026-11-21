use crate::grid::{Bathymetry, Grid, State};
use crate::kp07::{self, Cell, Kp07Params, Point, Reconstruction};
use crate::sponge::Sponge;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Boundary {
    Wall,
    Transmissive,
    Dirichlet,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    West,
    East,
    South,
    North,
}

impl Side {
    pub const ALL: [Side; 4] = [Side::West, Side::East, Side::South, Side::North];

    pub fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Debug)]
pub struct Boundaries {
    pub kinds: [Boundary; 4],
    pub dirichlet: [Vec<[f64; 3]>; 4],
}

impl Boundaries {
    pub fn uniform(grid: &Grid, kind: Boundary) -> Self {
        Self {
            kinds: [kind; 4],
            dirichlet: [
                vec![[0.0; 3]; grid.nz],
                vec![[0.0; 3]; grid.nz],
                vec![[0.0; 3]; grid.nx],
                vec![[0.0; 3]; grid.nx],
            ],
        }
    }

    pub fn with(mut self, side: Side, kind: Boundary) -> Self {
        self.kinds[side.index()] = kind;
        self
    }

    pub fn kind(&self, side: Side) -> Boundary {
        self.kinds[side.index()]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CouplingRegion {
    pub min: [usize; 2],
    pub max: [usize; 2],
    pub blend: f64,
}

impl CouplingRegion {
    pub fn contains(&self, i: usize, k: usize) -> bool {
        i >= self.min[0] && i < self.max[0] && k >= self.min[1] && k < self.max[1]
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Ledger {
    pub interface: f64,
    pub injected: f64,
    pub coupled: f64,
    pub mass: f64,
    pub outside: f64,
    pub tier: f64,
}

impl Ledger {
    pub fn conserved(&self) -> f64 {
        self.tier + self.outside - self.injected
    }
}

fn inside(region: Option<CouplingRegion>, i: usize, k: usize) -> bool {
    region.is_some_and(|r| r.contains(i, k))
}

#[derive(Clone, Debug)]
pub struct Fluxes {
    pub x: Vec<[f64; 3]>,
    pub z: Vec<[f64; 3]>,
    pub rate: f64,
}

#[derive(Clone, Debug)]
pub struct FaceField {
    pub x: Vec<f64>,
    pub z: Vec<f64>,
}

impl FaceField {
    pub fn zeros(grid: &Grid) -> Self {
        Self {
            x: vec![0.0; grid.x_faces()],
            z: vec![0.0; grid.z_faces()],
        }
    }
}

#[derive(Clone, Copy)]
enum Axis {
    X,
    Z,
}

#[derive(Clone, Debug)]
pub struct Swe {
    pub grid: Grid,
    pub bathymetry: Bathymetry,
    pub params: Kp07Params,
    pub boundaries: Boundaries,
    pub state: State,
    pub source: State,
    pub coupling: Option<CouplingRegion>,
    pub coupled: State,
    pub sponge: Option<Sponge>,
    pub time: f64,
    pub injected_mass: f64,
    pub coupled_mass: f64,
    pub interface_mass: f64,
}

impl Swe {
    pub fn new(grid: Grid, bathymetry: Bathymetry, state: State) -> Self {
        assert_eq!(
            state.h.len(),
            grid.cells(),
            "state size must match the grid"
        );

        Self {
            boundaries: Boundaries::uniform(&grid, Boundary::Wall),
            source: State::zeros(grid.cells()),
            coupled: State::zeros(grid.cells()),
            params: Kp07Params::default(),
            coupling: None,
            sponge: None,
            time: 0.0,
            injected_mass: 0.0,
            coupled_mass: 0.0,
            interface_mass: 0.0,
            grid,
            bathymetry,
            state,
        }
    }

    pub fn step(&mut self, dt_max: f64) -> f64 {
        let mut state = std::mem::take(&mut self.state);
        let mut flux = self.coupling.map(|_| FaceField::zeros(&self.grid));
        let dt = self.heun(&mut state, dt_max, flux.as_mut());
        self.state = state;

        if let Some(flux) = &flux {
            self.interface_mass += self.interface_flux(flux);
        }

        self.finish_step(dt);

        dt
    }

    pub fn interface_flux(&self, flux: &FaceField) -> f64 {
        let Some(region) = self.coupling else {
            return 0.0;
        };

        let g = &self.grid;
        let (i0, i1) = (region.min[0], region.max[0].min(g.nx));
        let (k0, k1) = (region.min[1], region.max[1].min(g.nz));
        let mut total = 0.0;

        for k in k0..k1 {
            if i0 > 0 && i0 < i1 {
                total += flux.x[g.x_face(i0, k)] * g.dz;
            }

            if i1 < g.nx && i0 < i1 {
                total -= flux.x[g.x_face(i1, k)] * g.dz;
            }
        }

        for i in i0..i1 {
            if k0 > 0 && k0 < k1 {
                total += flux.z[g.z_face(i, k0)] * g.dx;
            }

            if k1 < g.nz && k0 < k1 {
                total -= flux.z[g.z_face(i, k1)] * g.dx;
            }
        }

        total
    }

    pub fn ledger(&self) -> Ledger {
        let g = &self.grid;
        let area = g.area();
        let mut ledger = Ledger {
            interface: self.interface_mass,
            injected: self.injected_mass,
            coupled: self.coupled_mass,
            ..Ledger::default()
        };

        for k in 0..g.nz {
            for i in 0..g.nx {
                let c = g.cell(i, k);
                let mass = self.state.h[c] * area;

                ledger.mass += mass;

                if inside(self.coupling, i, k) {
                    ledger.tier += self.coupled.h[c] * area;
                } else {
                    ledger.outside += mass;
                }
            }
        }

        ledger
    }

    pub fn relocate(&mut self, region: Option<CouplingRegion>) {
        let g = self.grid;
        let area = g.area();
        let old = self.coupling;

        for k in 0..g.nz {
            for i in 0..g.nx {
                let c = g.cell(i, k);

                match (inside(old, i, k), inside(region, i, k)) {
                    (true, false) => {
                        self.state.h[c] = self.coupled.h[c];
                        self.state.hu[c] = self.coupled.hu[c];
                        self.state.hv[c] = self.coupled.hv[c];
                        self.interface_mass -= self.coupled.h[c] * area;
                    }
                    (false, true) => self.interface_mass += self.state.h[c] * area,
                    _ => {}
                }
            }
        }

        self.coupling = region;
    }

    pub fn advance(&mut self, duration: f64) -> usize {
        let end = self.time + duration;
        let mut steps = 0;

        while self.time < end - 1e-12 * duration.abs().max(1.0) {
            self.step(end - self.time);
            steps += 1;
        }

        steps
    }

    pub fn finish_step(&mut self, dt: f64) {
        self.apply_source();
        self.apply_coupling();

        if let Some(sponge) = &mut self.sponge {
            sponge.update_target(&self.grid, &self.bathymetry, self.time + dt);
            sponge.apply(&mut self.state, dt);
        }

        self.time += dt;
    }

    pub fn velocity(&self, cell: usize) -> [f64; 2] {
        let epsilon = self.params.epsilon();
        let h = self.state.h[cell];

        [
            kp07::desingularize(h, self.state.hu[cell], epsilon),
            kp07::desingularize(h, self.state.hv[cell], epsilon),
        ]
    }

    pub fn max_speed(&self) -> f64 {
        (0..self.grid.cells())
            .map(|c| {
                let [u, v] = self.velocity(c);
                u.hypot(v)
            })
            .fold(0.0, f64::max)
    }

    pub fn heun(
        &self,
        state: &mut State,
        dt_max: f64,
        mut accumulated: Option<&mut FaceField>,
    ) -> f64 {
        let first = self.fluxes(state);
        let dt = self.stable_dt(&first, dt_max);

        let mut stage = State::zeros(self.grid.cells());
        let factors = self.drain_factors(state, &first, dt);
        self.update(state, None, &first, &factors, dt, &mut stage);

        if let Some(acc) = accumulated.as_deref_mut() {
            self.accumulate(&first, &factors, 0.5 * dt, acc);
        }

        let second = self.fluxes(&stage);
        let factors = self.drain_factors(&stage, &second, dt);
        let mut next = State::zeros(self.grid.cells());
        self.update(&stage, Some(state), &second, &factors, dt, &mut next);

        if let Some(acc) = accumulated {
            self.accumulate(&second, &factors, 0.5 * dt, acc);
        }

        *state = next;

        dt
    }

    pub fn stable_dt(&self, fluxes: &Fluxes, dt_max: f64) -> f64 {
        if fluxes.rate > 0.0 {
            (self.params.cfl / fluxes.rate).min(dt_max)
        } else {
            dt_max
        }
    }

    pub fn fluxes(&self, state: &State) -> Fluxes {
        let g = &self.grid;
        let mut x = vec![[0.0; 3]; g.x_faces()];
        let mut z = vec![[0.0; 3]; g.z_faces()];
        let mut rate: f64 = 0.0;

        for k in 0..g.nz {
            for i in 0..=g.nx {
                let (flux, speed) = self.face_flux(state, Axis::X, i, k);
                x[g.x_face(i, k)] = flux;
                rate = rate.max(speed / g.dx);
            }
        }

        for k in 0..=g.nz {
            for i in 0..g.nx {
                let (flux, speed) = self.face_flux(state, Axis::Z, i, k);
                z[g.z_face(i, k)] = flux;
                rate = rate.max(speed / g.dz);
            }
        }

        Fluxes { x, z, rate }
    }

    pub fn drain_factors(&self, state: &State, fluxes: &Fluxes, dt: f64) -> Vec<f64> {
        let g = &self.grid;
        let mut factors = vec![1.0; g.cells()];

        for k in 0..g.nz {
            for i in 0..g.nx {
                let c = g.cell(i, k);
                let outflow = dt * cell_outflow(g, &fluxes.x, &fluxes.z, i, k, |f| f[0]);
                let available = state.h[c] * g.area();

                if outflow > available {
                    factors[c] = available / outflow;
                }
            }
        }

        factors
    }

    pub fn x_face_factor(&self, fluxes: &Fluxes, factors: &[f64], i: usize, k: usize) -> f64 {
        donor_factor(
            &self.grid,
            factors,
            Axis::X,
            i,
            k,
            fluxes.x[self.grid.x_face(i, k)][0],
        )
    }

    pub fn z_face_factor(&self, fluxes: &Fluxes, factors: &[f64], i: usize, k: usize) -> f64 {
        donor_factor(
            &self.grid,
            factors,
            Axis::Z,
            i,
            k,
            fluxes.z[self.grid.z_face(i, k)][0],
        )
    }

    pub fn update(
        &self,
        stage: &State,
        base: Option<&State>,
        fluxes: &Fluxes,
        factors: &[f64],
        dt: f64,
        out: &mut State,
    ) {
        let g = &self.grid;
        let gravity = self.params.gravity;

        for k in 0..g.nz {
            for i in 0..g.nx {
                let c = g.cell(i, k);
                let west = scale(
                    fluxes.x[g.x_face(i, k)],
                    self.x_face_factor(fluxes, factors, i, k),
                );
                let east = scale(
                    fluxes.x[g.x_face(i + 1, k)],
                    self.x_face_factor(fluxes, factors, i + 1, k),
                );
                let south = scale(
                    fluxes.z[g.z_face(i, k)],
                    self.z_face_factor(fluxes, factors, i, k),
                );
                let north = scale(
                    fluxes.z[g.z_face(i, k + 1)],
                    self.z_face_factor(fluxes, factors, i, k + 1),
                );

                let rx = self.reconstruct(stage, Axis::X, i as isize, k as isize);
                let rz = self.reconstruct(stage, Axis::Z, i as isize, k as isize);

                let dh = -(east[0] - west[0]) / g.dx - (north[0] - south[0]) / g.dz;
                let dhu = -(east[1] - west[1]) / g.dx - (north[1] - south[1]) / g.dz
                    + kp07::bed_source(&rx, gravity, g.dx);
                let dhv = -(east[2] - west[2]) / g.dx - (north[2] - south[2]) / g.dz
                    + kp07::bed_source(&rz, gravity, g.dz);

                let mut h = stage.h[c] + dt * dh;
                let friction = kp07::friction_factor(
                    h,
                    stage.hu[c] + dt * dhu,
                    stage.hv[c] + dt * dhv,
                    dt,
                    &self.params,
                );
                let mut hu = (stage.hu[c] + dt * dhu) * friction;
                let mut hv = (stage.hv[c] + dt * dhv) * friction;

                if let Some(base) = base {
                    h = 0.5 * (base.h[c] + h);
                    hu = 0.5 * (base.hu[c] + hu);
                    hv = 0.5 * (base.hv[c] + hv);
                }

                if h <= 0.0 {
                    h = 0.0;
                    hu = 0.0;
                    hv = 0.0;
                }

                out.h[c] = h;
                out.hu[c] = hu;
                out.hv[c] = hv;
            }
        }
    }

    fn accumulate(&self, fluxes: &Fluxes, factors: &[f64], weight: f64, acc: &mut FaceField) {
        let g = &self.grid;

        for k in 0..g.nz {
            for i in 0..=g.nx {
                let f = g.x_face(i, k);
                acc.x[f] += weight * fluxes.x[f][0] * self.x_face_factor(fluxes, factors, i, k);
            }
        }

        for k in 0..=g.nz {
            for i in 0..g.nx {
                let f = g.z_face(i, k);
                acc.z[f] += weight * fluxes.z[f][0] * self.z_face_factor(fluxes, factors, i, k);
            }
        }
    }

    fn apply_source(&mut self) {
        let area = self.grid.area();

        for c in 0..self.grid.cells() {
            let h = self.state.h[c];
            let dh = self.source.h[c].max(-h);

            self.state.h[c] = h + dh;
            self.state.hu[c] += self.source.hu[c];
            self.state.hv[c] += self.source.hv[c];
            self.injected_mass += dh * area;

            if self.state.h[c] <= 0.0 {
                self.state.hu[c] = 0.0;
                self.state.hv[c] = 0.0;
            }
        }

        self.source.clear();
    }

    fn apply_coupling(&mut self) {
        let Some(region) = self.coupling else {
            return;
        };

        let g = self.grid;

        for k in region.min[1]..region.max[1].min(g.nz) {
            for i in region.min[0]..region.max[0].min(g.nx) {
                let c = g.cell(i, k);
                let before = self.state.h[c];

                self.state.h[c] = (before + region.blend * (self.coupled.h[c] - before)).max(0.0);
                self.state.hu[c] += region.blend * (self.coupled.hu[c] - self.state.hu[c]);
                self.state.hv[c] += region.blend * (self.coupled.hv[c] - self.state.hv[c]);
                self.coupled_mass += (self.state.h[c] - before) * g.area();
            }
        }
    }

    fn cell_along(&self, state: &State, axis: Axis, i: isize, k: isize) -> Cell {
        let g = &self.grid;
        let (along, limit) = match axis {
            Axis::X => (i, g.nx as isize),
            Axis::Z => (k, g.nz as isize),
        };

        if along < 0 || along >= limit {
            return self.ghost_cell(state, axis, i, k);
        }

        let c = g.cell(i as usize, k as usize);
        let bed = self.bathymetry.cell(c);
        let h = state.h[c];
        let (level, full) = (bed.level(h), bed.is_full(h));

        match axis {
            Axis::X => Cell {
                h,
                qn: state.hu[c],
                qt: state.hv[c],
                level,
                full,
            },
            Axis::Z => Cell {
                h,
                qn: state.hv[c],
                qt: state.hu[c],
                level,
                full,
            },
        }
    }

    fn ghost_cell(&self, state: &State, axis: Axis, i: isize, k: isize) -> Cell {
        let g = &self.grid;
        let (side, inner_i, inner_k) = match axis {
            Axis::X if i < 0 => (Side::West, 0, k),
            Axis::X => (Side::East, g.nx as isize - 1, k),
            Axis::Z if k < 0 => (Side::South, i, 0),
            Axis::Z => (Side::North, i, g.nz as isize - 1),
        };

        let inner = self.cell_along(state, axis, inner_i, inner_k);

        match self.boundaries.kind(side) {
            Boundary::Wall => Cell {
                qn: -inner.qn,
                ..inner
            },
            Boundary::Transmissive => inner,
            Boundary::Dirichlet => {
                let along = match axis {
                    Axis::X => inner_k as usize,
                    Axis::Z => inner_i as usize,
                };
                let d = self.boundaries.dirichlet[side.index()][along];
                let (qn, qt) = match axis {
                    Axis::X => (d[1], d[2]),
                    Axis::Z => (d[2], d[1]),
                };

                Cell {
                    h: d[0],
                    qn,
                    qt,
                    level: d[0] + self.boundary_bottom(side, along),
                    full: true,
                }
            }
        }
    }

    fn boundary_bottom(&self, side: Side, along: usize) -> f64 {
        let g = &self.grid;

        match side {
            Side::West => self.bathymetry.face_x[g.x_face(0, along)],
            Side::East => self.bathymetry.face_x[g.x_face(g.nx, along)],
            Side::South => self.bathymetry.face_z[g.z_face(along, 0)],
            Side::North => self.bathymetry.face_z[g.z_face(along, g.nz)],
        }
    }

    fn reconstruct(&self, state: &State, axis: Axis, i: isize, k: isize) -> Reconstruction {
        let g = &self.grid;
        let (m, p) = match axis {
            Axis::X => ((i - 1, k), (i + 1, k)),
            Axis::Z => ((i, k - 1), (i, k + 1)),
        };

        let (iu, ku) = (i as usize, k as usize);
        let (b_minus, b_plus) = match axis {
            Axis::X => (
                self.bathymetry.face_x[g.x_face(iu, ku)],
                self.bathymetry.face_x[g.x_face(iu + 1, ku)],
            ),
            Axis::Z => (
                self.bathymetry.face_z[g.z_face(iu, ku)],
                self.bathymetry.face_z[g.z_face(iu, ku + 1)],
            ),
        };

        kp07::reconstruct(
            self.cell_along(state, axis, m.0, m.1),
            self.cell_along(state, axis, i, k),
            self.cell_along(state, axis, p.0, p.1),
            b_minus,
            b_plus,
            &self.params,
        )
    }

    fn ghost_point(&self, side: Side, along: usize, inner: Point) -> Point {
        match self.boundaries.kind(side) {
            Boundary::Wall => inner.mirrored(),
            Boundary::Transmissive => inner,
            Boundary::Dirichlet => {
                let d = self.boundaries.dirichlet[side.index()][along];
                let (qn, qt) = match side {
                    Side::West | Side::East => (d[1], d[2]),
                    Side::South | Side::North => (d[2], d[1]),
                };

                Point::new(d[0], qn, qt, self.params.epsilon())
            }
        }
    }

    fn face_flux(&self, state: &State, axis: Axis, i: usize, k: usize) -> ([f64; 3], f64) {
        let (along, limit, low_side, high_side) = match axis {
            Axis::X => (i, self.grid.nx, Side::West, Side::East),
            Axis::Z => (k, self.grid.nz, Side::South, Side::North),
        };
        let row = match axis {
            Axis::X => k,
            Axis::Z => i,
        };

        let (ii, kk) = (i as isize, k as isize);
        let (left_i, left_k) = match axis {
            Axis::X => (ii - 1, kk),
            Axis::Z => (ii, kk - 1),
        };

        let right = (along < limit).then(|| self.reconstruct(state, axis, ii, kk).minus);
        let left = (along > 0).then(|| self.reconstruct(state, axis, left_i, left_k).plus);

        let (left, right) = match (left, right) {
            (Some(l), Some(r)) => (l, r),
            (None, Some(r)) => (self.ghost_point(low_side, row, r), r),
            (Some(l), None) => (l, self.ghost_point(high_side, row, l)),
            (None, None) => unreachable!("a face always has an interior cell"),
        };

        let (f, speed) = kp07::central_upwind(&left, &right, self.params.gravity);

        let flux = match axis {
            Axis::X => f,
            Axis::Z => [f[0], f[2], f[1]],
        };

        (flux, speed)
    }
}

fn scale(f: [f64; 3], s: f64) -> [f64; 3] {
    [f[0] * s, f[1] * s, f[2] * s]
}

fn donor_factor(grid: &Grid, factors: &[f64], axis: Axis, i: usize, k: usize, mass: f64) -> f64 {
    let (along, limit) = match axis {
        Axis::X => (i, grid.nx),
        Axis::Z => (k, grid.nz),
    };

    let donor_is_low = mass > 0.0;

    if donor_is_low && along > 0 {
        match axis {
            Axis::X => factors[grid.cell(i - 1, k)],
            Axis::Z => factors[grid.cell(i, k - 1)],
        }
    } else if !donor_is_low && along < limit {
        factors[grid.cell(i, k)]
    } else {
        1.0
    }
}

pub fn limit_outflow(grid: &Grid, h: &[f64], flux: &mut FaceField) {
    let mut factors = vec![1.0; grid.cells()];

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let c = grid.cell(i, k);
            let outflow = cell_outflow(grid, &flux.x, &flux.z, i, k, |f| *f);
            let available = h[c] * grid.area();

            if outflow > available {
                factors[c] = available / outflow;
            }
        }
    }

    for k in 0..grid.nz {
        for i in 0..=grid.nx {
            let f = grid.x_face(i, k);
            flux.x[f] *= donor_factor(grid, &factors, Axis::X, i, k, flux.x[f]);
        }
    }

    for k in 0..=grid.nz {
        for i in 0..grid.nx {
            let f = grid.z_face(i, k);
            flux.z[f] *= donor_factor(grid, &factors, Axis::Z, i, k, flux.z[f]);
        }
    }
}

pub fn cell_outflow<T>(
    grid: &Grid,
    x: &[T],
    z: &[T],
    i: usize,
    k: usize,
    mass: impl Fn(&T) -> f64,
) -> f64 {
    let west = mass(&x[grid.x_face(i, k)]);
    let east = mass(&x[grid.x_face(i + 1, k)]);
    let south = mass(&z[grid.z_face(i, k)]);
    let north = mass(&z[grid.z_face(i, k + 1)]);

    (east.max(0.0) - west.min(0.0)) * grid.dz + (north.max(0.0) - south.min(0.0)) * grid.dx
}
