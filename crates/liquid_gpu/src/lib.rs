pub mod caustics;
pub mod coupling;
pub mod diagnostics;
pub mod emitter;
pub mod feedback;
pub mod gas;
pub mod gas_render;
pub mod heat;
pub mod layout;
pub mod narrow_band;
pub mod plugin;
pub mod redistance;
pub mod render;
pub mod solver;
pub mod surface_tension;
pub mod tall;
pub mod viscosity;
pub mod whitewater;

pub use liquid_ocean::WaterLook;

pub mod prelude {
    pub use crate::WaterLook;
    pub use crate::emitter::LiquidEmitter;
    pub use crate::gas::{GasConfig, GasEmitter, GasParams, GasRegion, GasShape};
    pub use crate::gas_render::{GasDomain, GasLook, GasPlugin, GasSampling};
    pub use crate::heat::HeatSource;
    pub use crate::plugin::{
        HeightfieldCoupling, LiquidBody, LiquidDomain, LiquidImpulse, LiquidPlugin, LiquidProperties, LiquidShape,
        LiquidSolid, seed_box, seed_where, wgpu_settings,
    };
    pub use crate::solver::{Quality, SolverConfig};
}
