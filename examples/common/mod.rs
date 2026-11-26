use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use bevy::diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::diagnostic::RenderDiagnosticsPlugin;
use bevy::time::TimeUpdateStrategy;
use bevy::window::PresentMode;
use liquid_gpu::prelude::*;

const BENCH_WARMUP: usize = 120;
const BENCH_WINDOW: usize = 120;
const RENDER_PATHS: [&str; 2] = ["render/gas/elapsed_gpu", "render/main_transparent_pass_3d/elapsed_gpu"];

#[derive(Resource, Clone, Copy)]
pub struct Preset(pub Quality);

#[derive(Resource)]
struct Bench {
    frames: usize,
    seen: usize,
    last: Option<Instant>,
    wall: Vec<f64>,
    samples: BTreeMap<String, Vec<f64>>,
}

pub fn flag(name: &str) -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let at = args.iter().position(|a| a == name)?;

    Some(args.get(at + 1).cloned().unwrap_or_default())
}

pub fn choice<T: Copy>(name: &str, options: &[(&str, T)]) -> T {
    let Some(value) = flag(name) else {
        return options[0].1;
    };

    options.iter().find(|(key, _)| *key == value).map(|(_, v)| *v).unwrap_or_else(|| {
        let keys: Vec<&str> = options.iter().map(|(key, _)| *key).collect();
        eprintln!("unknown {name} {value}, expected {}", keys.join(" or "));
        std::process::exit(2);
    })
}

pub fn app(window: Window) -> App {
    let quality = choice("--quality", &[("game", Quality::Game), ("high", Quality::High)]);
    let fixed = ["LIQUID_CAPTURE", "LIQUID_RECORD"].iter().any(|k| std::env::var_os(k).is_some());
    let bench = std::env::var("LIQUID_BENCH").ok().and_then(|v| v.parse::<usize>().ok());
    let diagnostics = bench.is_some() || std::env::var_os("LIQUID_DIAGNOSTICS").is_some();
    let mut app = App::new();

    app.insert_resource(Preset(quality)).add_plugins(
        DefaultPlugins
            .set(WindowPlugin {
                primary_window: Some(Window {
                    present_mode: if diagnostics { PresentMode::AutoNoVsync } else { PresentMode::AutoVsync },
                    ..window
                }),
                ..default()
            })
            .set(RenderPlugin {
                render_creation: wgpu_settings().into(),
                ..default()
            }),
    );

    if fixed || bench.is_some() {
        app.insert_resource(TimeUpdateStrategy::ManualDuration(Duration::from_secs_f64(1.0 / 60.0)));
    }

    if let Some(frames) = bench {
        app.add_plugins(RenderDiagnosticsPlugin)
            .insert_resource(Bench { frames, seen: 0, last: None, wall: Vec::new(), samples: BTreeMap::new() })
            .add_systems(Last, record_bench);
    } else if diagnostics {
        app.add_plugins((FrameTimeDiagnosticsPlugin::default(), LogDiagnosticsPlugin::default(), RenderDiagnosticsPlugin));
    }

    app
}

fn record_bench(preset: Res<Preset>, store: Res<DiagnosticsStore>, mut bench: ResMut<Bench>, mut exit: MessageWriter<AppExit>) {
    let bench = &mut *bench;
    bench.seen += 1;
    let now = Instant::now();
    let previous = bench.last.replace(now);

    if bench.seen <= BENCH_WARMUP {
        return;
    }

    if let Some(previous) = previous {
        bench.wall.push((now - previous).as_secs_f64() * 1000.0);
    }

    for diagnostic in store.iter() {
        let path = diagnostic.path().as_str();
        let wanted = path.starts_with("liquid/") || RENDER_PATHS.contains(&path);

        if let Some(value) = diagnostic.value().filter(|_| wanted) {
            bench.samples.entry(path.to_string()).or_default().push(value);
        }
    }

    if (bench.seen - BENCH_WARMUP).is_multiple_of(BENCH_WINDOW) {
        report_window(bench);
    }

    if bench.seen < BENCH_WARMUP + bench.frames {
        return;
    }

    let (median, p95, ..) = stats(&bench.wall);
    info!(
        "bench frames {} preset {}: wall median {median:.2} ms p95 {p95:.2} ms ({:.1} fps)",
        bench.wall.len(),
        preset.0.name(),
        1000.0 / median
    );

    for (path, samples) in &bench.samples {
        let (median, p95, mean, max) = stats(samples);

        if median > 0.0 {
            info!("bench {path}: median {median:.2} p95 {p95:.2} mean {mean:.2} max {max:.2}");
        }
    }

    exit.write(AppExit::Success);
}

fn report_window(bench: &Bench) {
    let recent = |values: &[f64]| stats(&values[values.len().saturating_sub(BENCH_WINDOW)..]).0;
    let sections: Vec<String> = bench
        .samples
        .iter()
        .map(|(path, samples)| (path.replace("liquid/", ""), recent(samples)))
        .filter(|(_, value)| *value > 0.0)
        .map(|(name, value)| format!("{name} {value:.2}"))
        .collect();

    info!("window wall {:.2} ms | {}", recent(&bench.wall), sections.join(", "));
}

fn stats(values: &[f64]) -> (f64, f64, f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let at = |q: f64| sorted.get(((sorted.len() as f64 - 1.0) * q) as usize).copied().unwrap_or(0.0);

    (at(0.5), at(0.95), sorted.iter().sum::<f64>() / sorted.len().max(1) as f64, at(1.0))
}
