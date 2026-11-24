use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use bevy::diagnostic::{Diagnostic, DiagnosticPath, Diagnostics, RegisterDiagnostic};
use bevy::prelude::*;

use crate::layout::GpuState;
use crate::solver::{GpuSolver, Profiler, Section};

const RING: usize = 4;
const PROFILE_PERIOD: u32 = 4;
const FREE: u8 = 0;
const ENCODED: u8 = 1;
const MAPPING: u8 = 2;
const MAPPED: u8 = 3;
const FAILED: u8 = 4;

pub const GPU_TIME: DiagnosticPath = DiagnosticPath::const_new("liquid/gpu");
pub const PARTICLES: DiagnosticPath = DiagnosticPath::const_new("liquid/particles");
pub const SUBSTEPS: DiagnosticPath = DiagnosticPath::const_new("liquid/substeps");
pub const ENCODE_TIME: DiagnosticPath = DiagnosticPath::const_new("liquid/encode");
pub const MAX_SPEED: DiagnosticPath = DiagnosticPath::const_new("liquid/max_speed");
pub const SECTIONS: [DiagnosticPath; 10] = [
    DiagnosticPath::const_new("liquid/plan"),
    DiagnosticPath::const_new("liquid/sort"),
    DiagnosticPath::const_new("liquid/transfer"),
    DiagnosticPath::const_new("liquid/geometry"),
    DiagnosticPath::const_new("liquid/pressure"),
    DiagnosticPath::const_new("liquid/advect"),
    DiagnosticPath::const_new("liquid/density"),
    DiagnosticPath::const_new("liquid/surface"),
    DiagnosticPath::const_new("liquid/whitewater"),
    DiagnosticPath::const_new("liquid/viscosity"),
];

pub struct Sample {
    pub sections: Vec<(Section, f32)>,
    pub particles: u32,
    pub substeps: u32,
    pub max_speed: f32,
    pub wanted: u32,
    pub encode_ms: f32,
}

struct Slot {
    profiler: Option<Profiler>,
    profiled: bool,
    state: wgpu::Buffer,
    status: Arc<AtomicU8>,
    encoded: u32,
    encode_ms: f32,
}

impl Slot {
    fn buffers(&self) -> impl Iterator<Item = &wgpu::Buffer> {
        let profiler = self.profiler.as_ref().filter(|_| self.profiled);

        std::iter::once(&self.state).chain(profiler.map(Profiler::readback))
    }
}

#[derive(Resource)]
pub(crate) struct DiagnosticsRing {
    slots: Mutex<(u32, Vec<Slot>)>,
    frames: AtomicU32,
    budget: AtomicU32,
    sender: Mutex<mpsc::Sender<Sample>>,
}

#[derive(Resource)]
struct DiagnosticsInbox(Mutex<mpsc::Receiver<Sample>>);

pub(crate) fn build(app: &mut App) -> DiagnosticsRing {
    let (sender, receiver) = mpsc::channel();

    for path in [GPU_TIME, ENCODE_TIME].into_iter().chain(SECTIONS) {
        app.register_diagnostic(Diagnostic::new(path).with_suffix(" ms"));
    }

    app.register_diagnostic(Diagnostic::new(PARTICLES))
        .register_diagnostic(Diagnostic::new(SUBSTEPS))
        .register_diagnostic(Diagnostic::new(MAX_SPEED).with_suffix(" m/s"))
        .insert_resource(DiagnosticsInbox(Mutex::new(receiver)))
        .add_systems(PreUpdate, record);

    DiagnosticsRing {
        slots: Mutex::new((0, Vec::new())),
        frames: AtomicU32::new(0),
        budget: AtomicU32::new(u32::MAX),
        sender: Mutex::new(sender),
    }
}

fn record(inbox: Res<DiagnosticsInbox>, mut diagnostics: Diagnostics) {
    let Ok(receiver) = inbox.0.lock() else {
        return;
    };

    for sample in receiver.try_iter() {
        if !sample.sections.is_empty() {
            let total: f32 = sample.sections.iter().map(|s| s.1).sum();
            diagnostics.add_measurement(&GPU_TIME, || total as f64);
        }

        for (path, (_, ms)) in SECTIONS.iter().zip(&sample.sections) {
            diagnostics.add_measurement(path, || *ms as f64);
        }

        diagnostics.add_measurement(&PARTICLES, || sample.particles as f64);
        diagnostics.add_measurement(&SUBSTEPS, || sample.substeps as f64);
        diagnostics.add_measurement(&MAX_SPEED, || sample.max_speed as f64);
        diagnostics.add_measurement(&ENCODE_TIME, || sample.encode_ms as f64);
    }
}

impl DiagnosticsRing {
    pub(crate) fn encode_frame(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        solver: &GpuSolver,
    ) {
        let Ok(mut guard) = self.slots.lock() else {
            solver.encode_frame_within(encoder, None, self.budget.load(Ordering::Acquire));
            return;
        };
        let (substeps, slots) = &mut *guard;

        if *substeps != solver.config.max_substeps || slots.is_empty() {
            *substeps = solver.config.max_substeps;
            *slots = (0..RING).map(|_| new_slot(device, queue, *substeps)).collect();
        }

        for slot in slots.iter_mut() {
            if slot.status.load(Ordering::Acquire) == FAILED {
                *slot = new_slot(device, queue, *substeps);
            } else {
                self.advance(slot);
            }
        }

        let budget = self.budget.load(Ordering::Acquire);
        let Some(slot) = slots.iter_mut().find(|s| s.status.load(Ordering::Acquire) == FREE) else {
            solver.encode_frame_within(encoder, None, budget);
            return;
        };

        let profiler = slot.profiler.as_ref().filter(|_| self.frames.fetch_add(1, Ordering::AcqRel).is_multiple_of(PROFILE_PERIOD));
        let start = std::time::Instant::now();
        solver.encode_frame_within(encoder, profiler, budget);
        slot.encode_ms = start.elapsed().as_secs_f32() * 1000.0;
        slot.encoded = budget;
        slot.profiled = profiler.is_some();

        if let Some(profiler) = profiler {
            profiler.resolve(encoder);
        }

        encoder.copy_buffer_to_buffer(&solver.buffers.state, 0, &slot.state, 0, state_bytes());
        slot.status.store(ENCODED, Ordering::Release);
    }

    fn advance(&self, slot: &Slot) {
        match slot.status.load(Ordering::Acquire) {
            ENCODED => {
                slot.status.store(MAPPING, Ordering::Release);
                let remaining = Arc::new(AtomicU8::new(slot.buffers().count() as u8));

                for buffer in slot.buffers() {
                    let status = slot.status.clone();
                    let remaining = remaining.clone();

                    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                        if result.is_err() {
                            status.store(FAILED, Ordering::Release);
                        } else if remaining.fetch_sub(1, Ordering::AcqRel) == 1 {
                            let _ = status.compare_exchange(MAPPING, MAPPED, Ordering::AcqRel, Ordering::Acquire);
                        }
                    });
                }
            }
            MAPPED => {
                if let Some(sample) = read(slot) {
                    self.budget.store(sample.wanted, Ordering::Release);

                    if let Ok(sender) = self.sender.lock() {
                        let _ = sender.send(sample);
                    }
                }

                for buffer in slot.buffers() {
                    buffer.unmap();
                }

                slot.status.store(FREE, Ordering::Release);
            }
            _ => {}
        }
    }
}

fn state_bytes() -> u64 {
    std::mem::size_of::<GpuState>() as u64
}

fn new_slot(device: &wgpu::Device, queue: &wgpu::Queue, substeps: u32) -> Slot {
    let profiler = device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY)
        .then(|| Profiler::new(device, queue, substeps));

    Slot {
        profiler,
        profiled: false,
        state: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid diagnostics state"),
            size: state_bytes(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        status: Arc::new(AtomicU8::new(FREE)),
        encoded: substeps,
        encode_ms: 0.0,
    }
}

fn read(slot: &Slot) -> Option<Sample> {
    let state: GpuState = {
        let view = slot.state.get_mapped_range(..).ok()?;
        *bytemuck::from_bytes(&view[..state_bytes() as usize])
    };
    let sections = match slot.profiler.as_ref().filter(|_| slot.profiled) {
        Some(profiler) => {
            let view = profiler.readback().get_mapped_range(..).ok()?;
            profiler.milliseconds(bytemuck::cast_slice(&view), slot.encoded)
        }
        None => Vec::new(),
    };

    Some(Sample {
        sections,
        particles: state.particle_count,
        substeps: state.substeps,
        max_speed: f32::from_bits(state.vmax_bits),
        wanted: budget_for(&state),
        encode_ms: slot.encode_ms,
    })
}

fn budget_for(state: &GpuState) -> u32 {
    if state.remaining <= 0.0 {
        return state.substeps;
    }

    if state.dt <= 0.0 {
        return u32::MAX;
    }

    state.substeps + (state.remaining / state.dt).ceil() as u32
}
