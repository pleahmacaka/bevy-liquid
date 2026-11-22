pub const FFT: &str = concat!(
    include_str!("../shaders/fft.wgsl"),
    include_str!("../shaders/complex.wgsl")
);

pub const SWE: &str = concat!(
    include_str!("../shaders/swe.wgsl"),
    include_str!("../shaders/bed.wgsl")
);

pub const OCEAN: &str = concat!(
    include_str!("../shaders/ocean.wgsl"),
    include_str!("../shaders/complex.wgsl")
);

pub const DISPERSIVE: &str = concat!(
    include_str!("../shaders/dispersive.wgsl"),
    include_str!("../shaders/bed.wgsl"),
    include_str!("../shaders/complex.wgsl"),
);

pub const PROMOTION: &str = concat!(
    include_str!("../shaders/promotion.wgsl"),
    include_str!("../shaders/bed.wgsl"),
    include_str!("../shaders/complex.wgsl")
);

pub const BED: &str = include_str!("../shaders/bed.wgsl");

pub const SPONGE: &str = include_str!("../shaders/sponge.wgsl");

pub const AMBIENT: &str = include_str!("../shaders/ambient.wgsl");
