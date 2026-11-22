mod common;

use liquid_ocean::dispersive::GpuDispersive;
use liquid_ocean::promotion::{GpuPromotion, PromotionInputs};
use liquid_ocean::swe::{self, GpuSwe};
use liquid_wave::GRAVITY;
use liquid_wave::dispersive::{Dispersive, DispersiveParams};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::promotion::{self, Criteria, Detector, Indicators, Promotion, Surface};
use liquid_wave::swe::{Boundaries, Boundary, Side, Swe};

const FRAME: f64 = 1.0 / 60.0;
const SLICES: usize = 4;

fn bore() -> Swe {
    let grid = Grid::new(160, 8, 0.1, 0.1);
    let bed = Bathymetry::flat(&grid, 0.0);
    let mut state = State::zeros(grid.cells());

    for k in 0..grid.nz {
        for i in 0..grid.nx {
            let [x, _] = grid.center(i, k);
            state.h[grid.cell(i, k)] = if x < 4.0 { 1.0 } else { 0.1 };
        }
    }

    let mut swe = Swe::new(grid, bed, state);
    swe.boundaries = Boundaries::uniform(&grid, Boundary::Wall).with(Side::East, Boundary::Transmissive);

    swe
}

fn mask_agreement(gpu: &common::Gpu, detector: &GpuPromotion, cpu: &[Indicators]) -> (usize, usize, f64) {
    let got: Vec<[f32; 4]> = gpu.read(&detector.indicators);
    let breaking: Vec<f64> = got.iter().map(|g| g[0] as f64).collect();
    let reference: Vec<f64> = cpu.iter().map(|c| c.breaking).collect();
    let mut disagree = 0;
    let mut flagged = 0;

    for (g, c) in got.iter().zip(cpu) {
        let gpu_flag = g.iter().copied().fold(0.0f32, f32::max) >= 1.0;
        let cpu_flag = c.score() >= 1.0;
        flagged += cpu_flag as usize;
        disagree += (gpu_flag != cpu_flag) as usize;
    }

    (flagged, disagree, common::relative_error(&breaking, &reference))
}

#[cfg(windows)]
#[test]
fn detector_compiles_under_fxc() {
    common::assert_compiles_under_fxc(|gpu| {
        let cpu = bore();
        let solver = GpuSwe::new(&gpu.device, &cpu, 0.01, swe::shared_buffers(&gpu.device, &cpu));
        let inputs = PromotionInputs {
            state: &solver.buffers.state,
            bed: &solver.buffers.bed,
            profile: &solver.buffers.profile,
        };
        GpuPromotion::new(&gpu.device, &cpu.grid, GRAVITY, &Criteria::default(), inputs);
    });
}

#[test]
fn gpu_detector_matches_cpu_on_a_breaking_bore() {
    let gpu = common::gpu();
    let mut cpu = bore();
    let dt = FRAME / SLICES as f64;
    let mut solver = GpuSwe::new(&gpu.device, &cpu, dt as f32, swe::shared_buffers(&gpu.device, &cpu));
    let criteria = Criteria::default();
    let mut detector = GpuPromotion::new(
        &gpu.device,
        &cpu.grid,
        GRAVITY,
        &criteria,
        PromotionInputs {
            state: &solver.buffers.state,
            bed: &solver.buffers.bed,
            profile: &solver.buffers.profile,
        },
    );
    let mut reference = Detector::new(criteria);
    let window = [40usize, 8];
    let mut last: (Option<Promotion>, Option<Promotion>) = (None, None);
    let mut worst: f64 = 0.0;

    solver.write_params(&gpu.queue, FRAME as f32);

    for frame in 0..30 {
        detector.write(&gpu.queue, FRAME as f32, [window[0] as u32, window[1] as u32], &[]);
        gpu.submit(|e| {
            solver.encode_frame(e, SLICES as u32);
            detector.encode(e);
        });

        for _ in 0..SLICES {
            assert_eq!(cpu.step(dt), dt);
        }

        let surface = Surface {
            grid: &cpu.grid,
            bathymetry: &cpu.bathymetry,
            state: &cpu.state,
            modes: None,
            gravity: GRAVITY,
        };
        let indicators = reference.evaluate(&surface, FRAME, &[]);
        let (flagged, disagree, breaking) = mask_agreement(&gpu, &detector, &indicators);
        let want = promotion::promote(&cpu.grid, &indicators, window);
        let got = detector.read(&gpu.device, &gpu.queue).promotion();

        if flagged > 0 {
            worst = worst.max(disagree as f64 / flagged as f64);
        }

        if frame % 10 == 9 {
            println!("frame {frame}: CPU flagged {flagged} cells, mask disagreement {disagree}, breaking indicator error {breaking:.2e}");
            println!("    CPU {want:?}");
            println!("    GPU {got:?}");
        }

        last = (want, got);
    }

    let want = last.0.expect("CPU must flag the bore");
    let got = last.1.expect("GPU must flag the bore");
    let shift = (want.window[0] as i64 - got.window[0] as i64).abs();

    assert!(worst < 0.05, "mask disagreement {worst}");
    assert!(shift <= 2, "windows differ by {shift} cells");
    assert!((want.flagged as f64 / got.flagged as f64 - 1.0).abs() < 0.05);
}

#[test]
fn gpu_detector_ignores_calm_water_and_a_small_airy_wave() {
    let gpu = common::gpu();
    let depth = 2.0;
    let k = 1.5 / depth;
    let grid = Grid::new(128, 8, 4.0 * std::f64::consts::PI / k / 128.0, 0.2);
    let bed = Bathymetry::flat(&grid, 0.0);

    for steepness in [0.0, 0.05] {
        let mut state = State::zeros(grid.cells());

        for kk in 0..grid.nz {
            for i in 0..grid.nx {
                let [x, _] = grid.center(i, kk);
                state.h[grid.cell(i, kk)] = depth + steepness / k * (k * x).cos();
            }
        }

        let model = Dispersive::new(Swe::new(grid, bed.clone(), state), DispersiveParams::default());
        let mut solver = GpuDispersive::new(
            &gpu.device,
            &model,
            (FRAME / SLICES as f64) as f32,
            swe::shared_buffers(&gpu.device, &model.swe),
        );
        let mut detector = GpuPromotion::new(
            &gpu.device,
            &grid,
            GRAVITY,
            &Criteria::default(),
            PromotionInputs {
                state: &solver.total,
                bed: &solver.bulk.buffers.bed,
                profile: &solver.bulk.buffers.profile,
            },
        );
        let mut peak: f32 = 0.0;
        let mut promoted = None;

        solver.write_params(&gpu.queue, FRAME as f32);

        for _ in 0..60 {
            detector.write(&gpu.queue, FRAME as f32, [32, 8], &[]);
            gpu.submit(|e| {
                solver.encode_frame(e, SLICES as u32);
                detector.encode(e);
            });

            let scores: Vec<[f32; 4]> = gpu.read(&detector.indicators);
            peak = scores.iter().flatten().copied().fold(peak, f32::max);
            promoted = detector.read(&gpu.device, &gpu.queue).promotion().or(promoted);
        }

        println!("GPU dispersive, ka = {steepness}: peak indicator {peak:.3}, promotion {promoted:?}");

        assert!(promoted.is_none());
        assert!(peak < 0.5);
    }
}
