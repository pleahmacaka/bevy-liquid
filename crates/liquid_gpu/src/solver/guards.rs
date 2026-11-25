use super::*;

impl GpuSolver {
    pub(super) fn guarded(&self) -> Vec<&wgpu::Buffer> {
        let b = &self.buffers;
        let mut out = vec![
            &b.particles,
            &b.particles_out,
            &b.cell_counter,
            &b.cell_start,
            &b.sorted,
            &b.velocity,
            &b.valid,
            &b.scratch,
            &b.weights,
            &b.owners,
            &b.solid_velocity,
            &b.phi,
            &b.nodes,
            &b.bodies,
            &b.state,
            &b.scalars,
            &b.partials,
            &b.scratch_valid,
            &b.pressure_keep,
            &b.ellipsoids,
            &b.packed,
            &b.brick_flags,
            &b.brick_list,
        ];

        out.extend(b.extra().map(|(_, buffer)| buffer));
        out.extend(&b.static_solid);

        out.push(&self.pack);

        for l in &self.levels {
            out.extend([&l.phi, &l.coef, &l.active, &l.diag, &l.off, &l.x, &l.b, &l.r, &l.z, &l.d, &l.q, &l.t, &l.tall, &l.e]);
        }

        out
    }

    pub fn allocated_bytes(&self) -> u64 {
        let b = &self.buffers;
        let rest = [&b.params, &b.surface, &b.indirect, &b.white, &b.white_out, &b.white_state];

        self.guarded().into_iter().chain(rest).map(|buffer| buffer.size()).sum()
    }

    pub fn grid_bytes(&self) -> u64 {
        let b = &self.buffers;
        let fine = [
            &b.cell_counter,
            &b.cell_start,
            &b.velocity,
            &b.valid,
            &b.scratch,
            &b.weights,
            &b.owners,
            &b.solid_velocity,
            &b.phi,
            &b.nodes,
            &b.scratch_valid,
            &b.pressure_keep,
            &b.surface,
            &b.brick_flags,
            &b.brick_list,
        ];
        let levels = self.levels.iter().flat_map(|l| [&l.phi, &l.coef, &l.active, &l.diag, &l.off, &l.x, &l.b, &l.r, &l.z, &l.d, &l.q, &l.t, &l.tall, &l.e]);
        let extra = b.extra().map(|(_, buffer)| buffer);

        fine.into_iter().chain(extra).chain(levels).map(|buffer| buffer.size()).sum()
    }

    pub fn guard_violations(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> Vec<String> {
        let buffers = self.guarded();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("liquid guard readback"),
            size: buffers.len() as u64 * GUARD_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&Default::default());

        for (i, buffer) in buffers.iter().enumerate() {
            let guard = buffer.size() - GUARD_BYTES;
            encoder.copy_buffer_to_buffer(buffer, guard, &staging, i as u64 * GUARD_BYTES, GUARD_BYTES);
        }

        queue.submit([encoder.finish()]);
        staging.map_async(wgpu::MapMode::Read, .., |r| r.expect("guard map failed"));
        let _ = device.poll(wgpu::PollType::wait_indefinitely());

        let words: Vec<u32> = {
            let view = staging.get_mapped_range(..).expect("guard range");
            bytemuck::cast_slice(&view).to_vec()
        };

        let per = (GUARD_BYTES / 4) as usize;

        buffers
            .iter()
            .enumerate()
            .filter(|(i, _)| words[i * per..(i + 1) * per].iter().any(|w| *w != GUARD_WORD))
            .map(|(i, b)| format!("buffer #{i} ({} bytes) guard overwritten", b.size()))
            .collect()
    }
}
