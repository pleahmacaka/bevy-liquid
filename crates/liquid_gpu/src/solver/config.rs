use super::*;

#[derive(Clone, Debug, PartialEq)]
pub struct SolverConfig {
    pub dims: UVec3,
    pub dx: f32,
    pub max_particles: u32,
    pub density: f32,
    pub gravity: Vec3,
    pub cfl: f32,
    pub max_dt: f32,
    pub max_substeps: u32,
    pub pressure_iterations: u32,
    pub density_iterations: u32,
    pub tolerance: f32,
    pub extrapolation_layers: u32,
    pub multigrid_min_dim: u32,
    pub smoothing: u32,
    pub coarse_sweeps: u32,
    pub block_smoothing: bool,
    pub density_projection: bool,
    pub kernel_radius: f32,
    pub particle_radius: f32,
    pub margin: f32,
    pub whitewater_capacity: u32,
    pub sparse: bool,
    pub brick_capacity: Option<u32>,
    pub narrow_band: Option<f32>,
    pub viscosity: f32,
    pub surface_tension: f32,
    pub adhesion: f32,
    pub temperature: f32,
    pub boiling_point: f32,
    pub latent_heat: f32,
    pub expansion: f32,
    pub cooling: f32,
    pub ambient_temperature: f32,
    pub viscosity_iterations: u32,
    pub redistance_iterations: u32,
    pub subgroups: bool,
    pub boundary_width: u32,
    pub boundary_sweeps: u32,
    pub open_floor: Option<f32>,
    pub tall_cells: Option<u32>,
    pub particles_per_cell: u32,
    pub backtrace_order: u32,
    pub quantized: bool,
    pub late_surface: bool,
    pub shared_density_grid: bool,
    pub cull_crowded: bool,
    pub static_solid: bool,
    pub quality: Quality,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quality {
    #[default]
    High,
    Game,
}

impl Quality {
    pub fn name(self) -> &'static str {
        match self {
            Quality::High => "high",
            Quality::Game => "game",
        }
    }
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            dims: UVec3::new(64, 48, 64),
            dx: 1.0 / 32.0,
            max_particles: 1 << 20,
            density: 1000.0,
            gravity: Vec3::new(0.0, -9.81, 0.0),
            cfl: 1.0,
            max_dt: 1.0 / 60.0,
            max_substeps: 4,
            pressure_iterations: 24,
            density_iterations: 8,
            tolerance: 1e-5,
            extrapolation_layers: 4,
            multigrid_min_dim: 4,
            smoothing: 2,
            coarse_sweeps: 8,
            block_smoothing: false,
            density_projection: true,
            kernel_radius: 1.0,
            particle_radius: 0.4,
            margin: 0.1,
            whitewater_capacity: 0,
            sparse: true,
            brick_capacity: None,
            narrow_band: None,
            viscosity: 0.0,
            surface_tension: 0.0,
            adhesion: 0.0,
            temperature: 293.15,
            boiling_point: 373.15,
            latent_heat: 540.0,
            expansion: 1600.0,
            cooling: 0.02,
            ambient_temperature: 293.15,
            viscosity_iterations: 50,
            redistance_iterations: 16,
            subgroups: true,
            boundary_width: 0,
            boundary_sweeps: 0,
            open_floor: None,
            tall_cells: None,
            particles_per_cell: 8,
            backtrace_order: 4,
            quantized: false,
            late_surface: false,
            shared_density_grid: false,
            cull_crowded: false,
            static_solid: false,
            quality: Quality::High,
        }
    }
}

pub(super) const GAME_COARSENING: f32 = 0.75;

pub(super) const GAME_VISCOSITY_ITERATIONS: u32 = 12;
pub(super) const GAME_DENSITY_ITERATIONS: u32 = 1;

impl SolverConfig {
    pub fn with_quality(self, quality: Quality) -> Self {
        if quality == Quality::High || self.quality == Quality::Game {
            return Self { quality, ..self };
        }

        let dims = (self.dims.as_vec3() * GAME_COARSENING).ceil().as_uvec3();
        let scale = (0..3).map(|k| self.dims[k] as f32 / dims[k] as f32).fold(f32::INFINITY, f32::min);

        Self {
            quality,
            dims,
            dx: self.dx * scale,
            pressure_iterations: self.pressure_iterations.min(2),
            density_iterations: self.density_iterations.min(GAME_DENSITY_ITERATIONS),
            viscosity_iterations: self.viscosity_iterations.min(GAME_VISCOSITY_ITERATIONS),
            redistance_iterations: self.redistance_iterations.min(8),
            whitewater_capacity: self.whitewater_capacity.min(1 << 15),
            particles_per_cell: 4,
            backtrace_order: 2,
            late_surface: true,
            shared_density_grid: true,
            cull_crowded: true,
            ..self
        }
    }

    pub fn has_band(&self) -> bool {
        self.narrow_band.is_some()
    }

    pub fn has_tension(&self) -> bool {
        self.surface_tension > 0.0
    }

    pub fn has_viscosity(&self) -> bool {
        self.viscosity > 0.0
    }

    pub fn has_adhesion(&self) -> bool {
        self.adhesion > 0.0
    }

    pub fn redistances(&self) -> bool {
        self.has_band() || self.has_tension()
    }

    pub fn diffuses(&self) -> bool {
        self.has_viscosity() || self.has_tension()
    }

    pub fn smooths_boundary(&self) -> bool {
        self.boundary_width > 0 && self.boundary_sweeps > 0
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slots {
    pub levels: u32,
}

impl Slots {
    pub const PARTICLES: u32 = 0;
    pub const CELLS: u32 = 1;
    pub const FACES: u32 = 2;
    pub const NODES: u32 = 3;
    pub const SINGLE: u32 = 4;
    pub const SCAN: u32 = 5;
    pub const REDUCE: u32 = 6;
    pub const GRID: u32 = 7;
    pub const BRICKS: u32 = 8;
    pub const CLEAR: u32 = 9;
    pub const BODIES: u32 = 10;
    const FIRST_LEVEL: u32 = 11;

    pub fn level(&self, l: usize) -> u32 {
        Self::FIRST_LEVEL + l as u32
    }

    pub fn solve_first(&self) -> u32 {
        Self::FIRST_LEVEL + self.levels
    }

    pub fn solve_single(&self) -> u32 {
        self.solve_first()
    }

    pub fn solve_reduce(&self) -> u32 {
        self.solve_first() + 1
    }

    pub fn solve_level(&self, l: usize) -> u32 {
        self.solve_first() + 2 + l as u32
    }

    pub fn solve_faces(&self) -> u32 {
        self.solve_first() + 2 + self.levels
    }

    pub fn solve_bodies(&self) -> u32 {
        self.solve_first() + 3 + self.levels
    }

    pub fn solve_grid(&self) -> u32 {
        self.solve_first() + 4 + self.levels
    }

    pub fn solve_block(&self, l: usize) -> u32 {
        self.solve_first() + 5 + self.levels + l as u32
    }

    pub fn solve_viscous(&self) -> u32 {
        self.solve_first() + 5 + 2 * self.levels
    }

    pub fn count(&self) -> u32 {
        self.solve_viscous() + 1
    }
}
