use liquid_wave::GRAVITY;
use liquid_wave::dispersive::{Dispersive, DispersiveParams};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::promotion::{self, Body, Criteria, Detector, Indicators, Promotion, Surface};
use liquid_wave::swe::{Boundaries, Boundary, Side, Swe};

fn detect(detector: &mut Detector, swe: &Swe, dt: f64, bodies: &[Body]) -> Vec<Indicators> {
    let surface = Surface {
        grid: &swe.grid,
        bathymetry: &swe.bathymetry,
        state: &swe.state,
        modes: None,
        gravity: swe.params.gravity,
    };

    detector.evaluate(&surface, dt, bodies)
}

fn run_frames(swe: &mut Swe, frames: usize, frame: f64, window: [usize; 2]) -> (Option<Promotion>, f64) {
    let mut detector = Detector::new(Criteria::default());
    let mut result = None;
    let mut peak: f64 = 0.0;

    for _ in 0..frames {
        swe.advance(frame);
        let indicators = detect(&mut detector, swe, frame, &[]);
        peak = indicators.iter().map(|i| i.score()).fold(peak, f64::max);
        result = promotion::promote(&swe.grid, &indicators, window);
    }

    (result, peak)
}

fn island_lake() -> Swe {
    let grid = Grid::new(64, 48, 0.25, 0.25);
    let bed = Bathymetry::from_fn(&grid, |x, z| {
        1.4 * (-((x - 8.0).powi(2) + (z - 6.0).powi(2)) / 4.0).exp()
    });
    let state = State::lake(&grid, &bed, 1.0);

    Swe::new(grid, bed, state)
}

#[test]
fn calm_lake_is_not_promoted() {
    let mut swe = island_lake();
    let (result, peak) = run_frames(&mut swe, 20, 1.0 / 60.0, [16, 16]);

    println!("calm lake: peak score {peak:.3e}, promotion {result:?}");

    assert!(result.is_none());
    assert!(peak < 1e-6);
}

pub fn airy_basin(kh: f64, depth: f64, steepness: f64) -> Dispersive {
    let k = kh / depth;
    let cells = 128;
    let dx = 4.0 * std::f64::consts::PI / k / cells as f64;
    let grid = Grid::new(cells, 8, dx, dx);
    let bed = Bathymetry::flat(&grid, 0.0);
    let mut state = State::zeros(grid.cells());
    let amplitude = steepness / k;
    for kk in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, kk);
            let c = grid.cell(i, kk);
            state.h[c] = depth + amplitude * (k * x).cos();
        }
    }

    Dispersive::new(Swe::new(grid, bed, state), DispersiveParams::default())
}

#[test]
fn small_airy_wave_is_not_promoted() {
    let mut model = airy_basin(1.5, 2.0, 0.05);
    let mut detector = Detector::new(Criteria::default());
    let frame = 1.0 / 60.0;
    let mut peak = Indicators::default();
    let mut result = None;

    for _ in 0..60 {
        let end = model.swe.time + frame;

        while model.swe.time < end - 1e-9 {
            model.step(end - model.swe.time);
        }

        let parts = model.decompose(&model.swe.state);
        let modes = model.surface_modes(&parts.surface);
        let surface = Surface {
            grid: &model.swe.grid,
            bathymetry: &model.swe.bathymetry,
            state: &model.swe.state,
            modes: Some(&modes),
            gravity: GRAVITY,
        };
        let indicators = detector.evaluate(&surface, frame, &[]);

        for i in &indicators {
            peak.breaking = peak.breaking.max(i.breaking);
            peak.acceleration = peak.acceleration.max(i.acceleration);
            peak.falling = peak.falling.max(i.falling);
        }

        result = promotion::promote(&model.swe.grid, &indicators, [32, 8]).or(result);
    }

    println!("Airy wave ka = 0.05, kh = 1.5: peak indicators {peak:?}, promotion {result:?}");

    assert!(result.is_none());
    assert!(peak.score() < 0.5);
}

pub fn bore(upstream: f64, downstream: f64) -> Swe {
    let grid = Grid::new(160, 8, 0.1, 0.1);
    let bed = Bathymetry::flat(&grid, 0.0);
    let mut state = State::zeros(grid.cells());

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, k);
            state.h[grid.cell(i, k)] = if x < 4.0 { upstream } else { downstream };
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.boundaries = Boundaries::uniform(&grid, Boundary::Wall).with(Side::East, Boundary::Transmissive);

    swe
}

fn front(swe: &Swe, level: f64) -> usize {
    (0..swe.grid.nx)
        .rev()
        .find(|&i| swe.state.h[swe.grid.cell(i, 4)] > level)
        .unwrap_or(0)
}

#[test]
fn breaking_bore_is_promoted() {
    let mut swe = bore(1.0, 0.1);
    let (result, peak) = run_frames(&mut swe, 40, 1.0 / 60.0, [40, 8]);
    let result = result.expect("the bore must be flagged");
    let at = front(&swe, 0.2) as u32;

    println!(
        "bore 1.0 / 0.1 m after {:.2} s: front at cell {at}, peak score {peak:.2}, promotion {result:?}",
        swe.time
    );

    assert!(result.window[0] <= at && at < result.window[2]);
    assert!(result.bounds[0] <= at && at < result.bounds[2]);
}

#[test]
fn piercing_body_is_promoted() {
    let mut swe = island_lake();
    let mut detector = Detector::new(Criteria::default());
    let body = Body {
        center: [11.0, 1.1, 8.0],
        radius: 0.4,
    };

    swe.advance(0.1);
    let indicators = detect(&mut detector, &swe, 0.1, &[body]);
    let result = promotion::promote(&swe.grid, &indicators, [16, 16]).expect("the body must be flagged");
    let centre = [(11.0 / 0.25) as u32, (8.0 / 0.25) as u32];

    println!("piercing body: {result:?}");

    assert!(result.flagged >= 4);
    assert!(result.window[0] < centre[0] && centre[0] < result.window[2]);
    assert!(result.window[1] < centre[1] && centre[1] < result.window[3]);
}

#[test]
fn water_falling_off_a_ledge_is_promoted() {
    let grid = Grid::new(96, 8, 0.1, 0.1);
    let bed = Bathymetry::from_fn(&grid, |x, _| if x < 5.0 { 1.0 } else { 0.0 });
    let mut state = State::lake(&grid, &bed, 0.2);

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let c = grid.cell(i, k);
            if grid.center(i, k)[0] < 5.0 {
                state.h[c] = 0.2;
                state.hu[c] = 0.2;
            }
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    let (result, peak) = run_frames(&mut swe, 20, 1.0 / 60.0, [24, 8]);
    let result = result.expect("the ledge must be flagged");

    println!("ledge: peak score {peak:.2}, promotion {result:?}");

    assert!(result.window[0] <= 50 && 50 < result.window[2]);
}

