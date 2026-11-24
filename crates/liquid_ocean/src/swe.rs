use bytemuck::{Pod, Zeroable};
use liquid_wave::grid::{Bathymetry, Grid, State};
use liquid_wave::swe::{Boundary, CouplingRegion, Ledger, Side, Swe};

use crate::kernel::{self, Binding, Kernel};
use crate::shaders;
use crate::sponge::{SpongeBuffers, SpongeTracker};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SweParams {
    pub nx: u32,
    pub nz: u32,
    pub dx: f32,
    pub dz: f32,
    pub gravity: f32,
    pub theta: f32,
    pub cfl: f32,
    pub epsilon: f32,
    pub dt_max: f32,
    pub frame_time: f32,
    pub sponge: u32,
    pub coupling_blend: f32,
    pub kinds: [u32; 4],
    pub coupling_min: [u32; 2],
    pub coupling_max: [u32; 2],
    pub manning: f32,
    pub pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Clock {
    pub dt: f32,
    pub remaining: f32,
    pub time: f32,
    pub elapsed: f32,
    pub rate: u32,
    pub steps: u32,
    pub pad: [u32; 2],
}

#[derive(Clone)]
pub struct SharedBuffers {
    pub state: wgpu::Buffer,
    pub bed: wgpu::Buffer,
}

#[derive(Clone)]
pub struct SweBuffers {
    pub params: wgpu::Buffer,
    pub state: wgpu::Buffer,
    pub stage: wgpu::Buffer,
    pub bed: wgpu::Buffer,
    pub face_bed: wgpu::Buffer,
    pub flux: wgpu::Buffer,
    pub factors: wgpu::Buffer,
    pub clock: wgpu::Buffer,
    pub boundary: wgpu::Buffer,
    pub source: wgpu::Buffer,
    pub coupled: wgpu::Buffer,
    pub sponge: wgpu::Buffer,
    pub accumulated: wgpu::Buffer,
    pub ledger: wgpu::Buffer,
    pub profile: wgpu::Buffer,
    pub steps: wgpu::Buffer,
}

impl SweBuffers {
    fn ordered_with_state<'a>(&'a self, state: &'a wgpu::Buffer) -> [&'a wgpu::Buffer; 14] {
        [
            &self.params,
            state,
            &self.stage,
            &self.bed,
            &self.face_bed,
            &self.flux,
            &self.factors,
            &self.clock,
            &self.boundary,
            &self.source,
            &self.coupled,
            &self.sponge,
            &self.accumulated,
            &self.ledger,
        ]
    }
}

const CELL_ARGS: u64 = 0;
const FACE_ARGS: u64 = 3;
const SINGLE_ARGS: u64 = 6;
const NEXT_ARGS: u64 = 9;

pub const BINDINGS: [Binding; 14] = [
    Binding::Uniform,
    Binding::Storage,
    Binding::Storage,
    Binding::ReadOnly,
    Binding::ReadOnly,
    Binding::Storage,
    Binding::Storage,
    Binding::Storage,
    Binding::ReadOnly,
    Binding::Storage,
    Binding::ReadOnly,
    Binding::ReadOnly,
    Binding::Storage,
    Binding::Storage,
];

const ENTRIES: [&str; 15] = [
    "flux_state",
    "flux_stage",
    "begin_frame",
    "drain_state",
    "drain_stage",
    "update_first",
    "update_second",
    "accumulate_flux",
    "clear_accumulated",
    "finish",
    "ledger_switch_cells",
    "ledger_switch",
    "ledger_faces_flux",
    "ledger_faces_accumulated",
    "ledger_finish",
];

pub const LEDGER_SLOTS: usize = 32;
pub const L_REGION: usize = 10;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuLedger {
    pub slots: [f32; LEDGER_SLOTS],
}

impl GpuLedger {
    pub fn new(coupling: Option<CouplingRegion>) -> Self {
        let (min, max, _) = coupling_fields(coupling);
        let mut slots = [0.0; LEDGER_SLOTS];

        for (slot, value) in slots[L_REGION..L_REGION + 4].iter_mut().zip([min[0], min[1], max[0], max[1]]) {
            *slot = value as f32;
        }

        Self { slots }
    }

    pub fn ledger(&self) -> Ledger {
        let s = &self.slots;
        let compensated = |slot: usize| (s[slot] - s[slot + 1]) as f64;

        Ledger {
            interface: compensated(0),
            injected: compensated(2),
            coupled: compensated(4),
            mass: s[6] as f64,
            outside: s[7] as f64,
            tier: s[8] as f64,
        }
    }

    pub fn frames(&self) -> u32 {
        self.slots[9] as u32
    }
}

pub fn boundary_kind(kind: Boundary) -> u32 {
    match kind {
        Boundary::Wall => 0,
        Boundary::Transmissive => 1,
        Boundary::Dirichlet => 2,
    }
}

pub fn state_data(state: &State) -> Vec<[f32; 4]> {
    (0..state.h.len())
        .map(|c| {
            [
                state.h[c] as f32,
                state.hu[c] as f32,
                state.hv[c] as f32,
                0.0,
            ]
        })
        .collect()
}

pub fn bed_data(bathymetry: &Bathymetry) -> Vec<[f32; 4]> {
    (0..bathymetry.center.len())
        .map(|c| {
            [
                bathymetry.center[c] as f32,
                bathymetry.cell_min[c] as f32,
                bathymetry.cell_max[c] as f32,
                0.0,
            ]
        })
        .collect()
}

pub fn face_bed_data(bathymetry: &Bathymetry) -> Vec<f32> {
    bathymetry
        .face_x
        .iter()
        .chain(&bathymetry.face_z)
        .map(|&b| b as f32)
        .collect()
}

pub fn boundary_data(swe: &Swe) -> Vec<[f32; 4]> {
    Side::ALL
        .iter()
        .flat_map(|side| swe.boundaries.dirichlet[side.index()].iter())
        .map(|d| [d[0] as f32, d[1] as f32, d[2] as f32, 0.0])
        .collect()
}

pub fn sponge_data(swe: &Swe) -> Vec<[f32; 4]> {
    let cells = swe.grid.cells();

    match &swe.sponge {
        Some(s) => (0..cells)
            .map(|c| {
                [
                    s.target.h[c] as f32,
                    s.target.hu[c] as f32,
                    s.target.hv[c] as f32,
                    s.sigma[c] as f32,
                ]
            })
            .collect(),
        None => vec![[0.0; 4]; cells],
    }
}

pub fn shared_buffers(device: &wgpu::Device, swe: &Swe) -> SharedBuffers {
    SharedBuffers {
        state: kernel::storage(
            device,
            "swe state",
            kernel::vec4_bytes(&state_data(&swe.state)),
        ),
        bed: kernel::storage(
            device,
            "swe bed",
            kernel::vec4_bytes(&bed_data(&swe.bathymetry)),
        ),
    }
}

pub fn params(swe: &Swe, dt_max: f32, frame_time: f32) -> SweParams {
    let g = &swe.grid;
    let p = &swe.params;
    let (coupling_min, coupling_max, blend) = coupling_fields(swe.coupling);

    SweParams {
        nx: g.nx as u32,
        nz: g.nz as u32,
        dx: g.dx as f32,
        dz: g.dz as f32,
        gravity: p.gravity as f32,
        theta: p.theta as f32,
        cfl: p.cfl as f32,
        epsilon: p.epsilon() as f32,
        dt_max,
        frame_time,
        sponge: swe.sponge.is_some() as u32,
        coupling_blend: blend,
        kinds: Side::ALL.map(|s| boundary_kind(swe.boundaries.kind(s))),
        coupling_min,
        coupling_max,
        manning: p.manning as f32,
        pad: [0.0; 3],
    }
}

fn coupling_fields(coupling: Option<CouplingRegion>) -> ([u32; 2], [u32; 2], f32) {
    match coupling {
        Some(c) => (
            [c.min[0] as u32, c.min[1] as u32],
            [c.max[0] as u32, c.max[1] as u32],
            c.blend as f32,
        ),
        None => ([0, 0], [0, 0], 0.0),
    }
}

pub struct GpuSwe {
    kernel: Kernel,
    pub buffers: SweBuffers,
    group: wgpu::BindGroup,
    steps: wgpu::BindGroup,
    pub tracker: Option<SpongeTracker>,
    pub params: SweParams,
    pub cells: u32,
    pub faces: u32,
}

impl GpuSwe {
    pub fn new(device: &wgpu::Device, swe: &Swe, dt_max: f32, shared: SharedBuffers) -> Self {
        let kernel = Kernel::with_group(device, "swe", shaders::SWE, &BINDINGS, &ENTRIES, &[Binding::Storage], &["choose_dt"]);
        let grid: Grid = swe.grid;
        let cells = grid.cells() as u32;
        let faces = (grid.x_faces() + grid.z_faces()) as u32;
        let params = params(swe, dt_max, 0.0);

        let buffers = SweBuffers {
            params: kernel::uniform(device, "swe params", bytemuck::bytes_of(&params)),
            state: shared.state,
            stage: kernel::zeroed_storage(device, "swe stage", 16 * cells as u64),
            bed: shared.bed,
            face_bed: kernel::storage(
                device,
                "swe face bed",
                bytemuck::cast_slice(&face_bed_data(&swe.bathymetry)),
            ),
            flux: kernel::zeroed_storage(device, "swe flux", 16 * faces as u64),
            factors: kernel::zeroed_storage(device, "swe drain", 4 * cells as u64),
            clock: kernel::storage(
                device,
                "swe clock",
                bytemuck::bytes_of(&Clock {
                    time: swe.time as f32,
                    ..Clock::default()
                }),
            ),
            boundary: kernel::storage(
                device,
                "swe boundary",
                kernel::vec4_bytes(&boundary_data(swe)),
            ),
            source: kernel::zeroed_storage(device, "swe source", 16 * cells as u64),
            coupled: kernel::storage(
                device,
                "swe coupled",
                kernel::vec4_bytes(&state_data(&swe.coupled)),
            ),
            sponge: kernel::storage(device, "swe sponge", kernel::vec4_bytes(&sponge_data(swe))),
            accumulated: kernel::zeroed_storage(device, "swe accumulated", 4 * faces as u64),
            ledger: kernel::storage(
                device,
                "swe ledger",
                bytemuck::bytes_of(&GpuLedger::new(swe.coupling)),
            ),
            profile: kernel::zeroed_storage(device, "swe profile", 32 * cells as u64),
            steps: kernel::indirect_storage(device, "swe steps", 72),
        };

        let group = kernel.bind(device, "swe", &buffers.ordered_with_state(&buffers.state));
        let steps = kernel.bind_extra(device, "swe steps", &[&buffers.steps]).expect("swe steps layout");
        let tracker = swe
            .sponge
            .as_ref()
            .and_then(|s| s.ambient.as_ref())
            .map(|(frame, field)| {
                let target = SpongeBuffers {
                    sponge: &buffers.sponge,
                    bed: &buffers.bed,
                    clock: &buffers.clock,
                };
                SpongeTracker::new(device, &grid, frame, field, target)
            });

        Self {
            kernel,
            buffers,
            group,
            steps,
            tracker,
            params,
            cells,
            faces,
        }
    }

    pub fn bind_state(&self, device: &wgpu::Device, state: &wgpu::Buffer) -> wgpu::BindGroup {
        self.kernel.bind(
            device,
            "swe alternate state",
            &self.buffers.ordered_with_state(state),
        )
    }

    pub fn set_coupling(&mut self, coupling: Option<CouplingRegion>) {
        let (min, max, blend) = coupling_fields(coupling);
        self.params.coupling_min = min;
        self.params.coupling_max = max;
        self.params.coupling_blend = blend;
    }

    pub fn write_params(&mut self, queue: &wgpu::Queue, frame_time: f32) {
        self.params.frame_time = frame_time;
        queue.write_buffer(&self.buffers.params, 0, bytemuck::bytes_of(&self.params));
    }

    pub fn encode_frame(&self, encoder: &mut wgpu::CommandEncoder, substeps: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("swe frame"),
            timestamp_writes: None,
        });

        self.encode_begin(&mut pass, &self.group);

        for substep in 0..substeps {
            self.encode_steps(&mut pass, &self.group, false, substep > 0);
        }

        self.encode_finish(&mut pass, &self.group);
    }

    pub fn encode_begin(&self, pass: &mut wgpu::ComputePass, group: &wgpu::BindGroup) {
        let k = &self.kernel;

        k.dispatch(pass, "begin_frame", group, 1);
        k.dispatch(pass, "ledger_switch_cells", group, self.cells);
        k.dispatch_grid(pass, "ledger_switch", group, [1, 1]);
    }

    pub fn encode_substep(
        &self,
        pass: &mut wgpu::ComputePass,
        group: &wgpu::BindGroup,
        accumulate: bool,
    ) {
        self.encode_steps(pass, group, accumulate, false);
    }

    fn encode_steps(&self, pass: &mut wgpu::ComputePass, group: &wgpu::BindGroup, accumulate: bool, gated: bool) {
        let k = &self.kernel;
        let args = &self.buffers.steps;
        let run = |pass: &mut wgpu::ComputePass, entry: &str, threads: u32, slot: u64| {
            if gated {
                k.dispatch_indirect(pass, entry, group, args, slot);
            } else {
                k.dispatch(pass, entry, group, threads);
            }
        };
        let interface = |pass: &mut wgpu::ComputePass| {
            if accumulate {
                run(pass, "accumulate_flux", self.faces, FACE_ARGS);
            } else if gated {
                k.dispatch_indirect(pass, "ledger_faces_flux", group, args, SINGLE_ARGS);
            } else {
                k.dispatch_grid(pass, "ledger_faces_flux", group, [1, 1]);
            }
        };

        run(pass, "flux_state", self.faces, NEXT_ARGS + FACE_ARGS);
        pass.set_bind_group(1, &self.steps, &[]);
        k.dispatch(pass, "choose_dt", group, 1);
        run(pass, "drain_state", self.cells, CELL_ARGS);
        interface(pass);
        run(pass, "update_first", self.cells, CELL_ARGS);
        run(pass, "flux_stage", self.faces, FACE_ARGS);
        run(pass, "drain_stage", self.cells, CELL_ARGS);
        interface(pass);
        run(pass, "update_second", self.cells, CELL_ARGS);
    }

    pub fn encode_clear_accumulated(&self, pass: &mut wgpu::ComputePass, group: &wgpu::BindGroup) {
        self.kernel
            .dispatch(pass, "clear_accumulated", group, self.faces);
    }

    pub fn encode_finish(&self, pass: &mut wgpu::ComputePass, group: &wgpu::BindGroup) {
        if let Some(tracker) = &self.tracker {
            tracker.encode(pass);
        }

        self.kernel.dispatch(pass, "finish", group, self.cells);
        self.kernel.dispatch_grid(pass, "ledger_finish", group, [1, 1]);
    }

    pub fn encode_interface_accumulated(&self, pass: &mut wgpu::ComputePass) {
        self.kernel
            .dispatch_grid(pass, "ledger_faces_accumulated", &self.group, [1, 1]);
    }

    pub fn reset_ledger(&self, queue: &wgpu::Queue, coupling: Option<CouplingRegion>) {
        queue.write_buffer(
            &self.buffers.ledger,
            0,
            bytemuck::bytes_of(&GpuLedger::new(coupling)),
        );
    }

    pub fn read_ledger(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> GpuLedger {
        kernel::read_pod(device, queue, &self.buffers.ledger)
    }

    pub fn group(&self) -> &wgpu::BindGroup {
        &self.group
    }
}
