use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

use bevy::prelude::*;

pub const FEEDBACK_DEPTH: u64 = 2;

const SLOTS: u64 = FEEDBACK_DEPTH + 2;
const EMPTY: u8 = 0;
const ENCODED: u8 = 1;
const MAPPING: u8 = 2;
const MAPPED: u8 = 3;
const FAILED: u8 = 4;

struct Slot {
    buffer: wgpu::Buffer,
    frame: u64,
    status: Arc<AtomicU8>,
}

#[derive(Resource, Clone, Default)]
pub(crate) struct Feedback {
    slots: Arc<Mutex<Vec<Slot>>>,
    bytes: u64,
}

impl Feedback {
    pub(crate) fn new(bytes: u64) -> Self {
        Self { slots: Arc::default(), bytes }
    }

    pub(crate) fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        frame: u64,
        copy: impl FnOnce(&mut wgpu::CommandEncoder, &wgpu::Buffer),
    ) {
        let Ok(mut slots) = self.slots.lock() else {
            return;
        };

        if slots.is_empty() {
            *slots = (0..SLOTS)
                .map(|_| Slot {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("liquid body feedback"),
                        size: self.bytes,
                        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    frame: u64::MAX,
                    status: Arc::new(AtomicU8::new(EMPTY)),
                })
                .collect();
        }

        let slot = &mut slots[(frame % SLOTS) as usize];

        if matches!(slot.status.load(Ordering::Acquire), MAPPING | MAPPED) {
            slot.buffer.unmap();
        }

        copy(encoder, &slot.buffer);
        slot.frame = frame;
        slot.status.store(ENCODED, Ordering::Release);
    }

    pub(crate) fn take(&self, device: &wgpu::Device, frame: u64) -> Option<Vec<f32>> {
        let wanted = frame.checked_sub(FEEDBACK_DEPTH)?;
        let (buffer, status) = {
            let slots = self.slots.lock().ok()?;
            let slot = slots.iter().find(|s| s.frame == wanted)?;

            if slot.status.load(Ordering::Acquire) == ENCODED {
                slot.status.store(MAPPING, Ordering::Release);
                let status = slot.status.clone();

                slot.buffer.map_async(wgpu::MapMode::Read, .., move |result| {
                    let _ = status.compare_exchange(
                        MAPPING,
                        if result.is_ok() { MAPPED } else { FAILED },
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    );
                });
            }

            (slot.buffer.clone(), slot.status.clone())
        };

        while status.load(Ordering::Acquire) == MAPPING {
            if device.poll(wgpu::PollType::Poll).is_err() {
                return None;
            }

            std::thread::yield_now();
        }

        if status.compare_exchange(MAPPED, EMPTY, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return None;
        }

        let values = buffer.get_mapped_range(..).ok().map(|view| bytemuck::cast_slice(&view).to_vec());
        buffer.unmap();

        values
    }
}
