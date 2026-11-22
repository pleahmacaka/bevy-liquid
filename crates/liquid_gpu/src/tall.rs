use crate::layout::{WG, groups};
use crate::solver::{Bind, GpuSolver};

pub const TALL_STRIDE: u32 = 16;

pub const TALL_LEVELS: usize = 3;

const LINE_SPAN: u32 = 32768;

const LINE_GROUPS: u32 = 8;

impl GpuSolver {
    pub fn has_tall(&self) -> bool {
        self.tall_levels() > 0
    }

    pub fn tall_levels(&self) -> usize {
        if self.config.tall_cells.is_some() { TALL_LEVELS.min(self.levels.len() - 1) } else { 0 }
    }

    fn column_groups(&self, l: usize) -> u32 {
        let d = self.levels[l].dims;

        groups(d.x * d.z, WG)
    }

    pub(crate) fn encode_tall(&self, pass: &mut wgpu::ComputePass<'_>) {
        if !self.has_tall() {
            return;
        }

        let d = self.levels[0].dims;
        let columns = self.column_groups(0);

        self.dispatch_direct(pass, "tall_low", Bind::Level(0), columns);
        self.dispatch_direct(pass, "tall_cap", Bind::Level(0), columns);
        self.dispatch_direct(pass, "tall_rows", Bind::Level(0), groups(d.z, WG));
        self.dispatch_direct(pass, "tall_lines", Bind::Level(0), groups(d.x, WG));

        for l in 1..self.tall_levels() {
            self.dispatch_direct(pass, "tall_coarse", Bind::Pair(l - 1), self.column_groups(l));
        }

        for l in 0..self.tall_levels() {
            self.dispatch_direct(pass, "tall_assemble", Bind::Level(l), self.column_groups(l));
        }
    }

    pub(crate) fn encode_tall_rhs(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.has_tall() {
            self.dispatch_direct(pass, "tall_rhs", Bind::Level(0), self.column_groups(0));
        }
    }

    pub(crate) fn encode_tall_expand(&self, pass: &mut wgpu::ComputePass<'_>) {
        if self.has_tall() {
            self.dispatch_direct(pass, "tall_expand", Bind::Level(0), self.column_groups(0));
        }
    }

    pub(crate) fn encode_tall_reset(&self, pass: &mut wgpu::ComputePass<'_>) {
        for l in 0..self.tall_levels() {
            self.dispatch_direct(pass, "tall_reset", Bind::Level(l), self.column_groups(l));
        }
    }

    pub(crate) fn relax_lines(&self, pass: &mut wgpu::ComputePass<'_>, l: usize, colors: [u32; 2]) {
        let d = self.levels[l].dims;
        let lines = (d.x.div_ceil(2) * d.z).div_ceil(LINE_GROUPS);

        for color in colors {
            self.prepare(pass, &format!("mg_line#{color}"), Bind::Multigrid(l));
            pass.dispatch_workgroups(lines.min(LINE_SPAN), lines.div_ceil(LINE_SPAN), 1);
        }
    }

    pub(crate) fn gather_tall(&self, pass: &mut wgpu::ComputePass<'_>, l: usize) {
        if l < self.tall_levels() {
            self.dispatch_direct(pass, "tall_gather", Bind::Multigrid(l), self.column_groups(l));
        }
    }
}
