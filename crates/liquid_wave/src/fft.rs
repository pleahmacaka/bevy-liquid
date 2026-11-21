use std::f64::consts::TAU;
use std::sync::Arc;

pub use rustfft::num_complex::Complex64;
use rustfft::{Fft, FftPlanner};

pub struct Fft2 {
    pub nx: usize,
    pub nz: usize,
    forward_x: Arc<dyn Fft<f64>>,
    inverse_x: Arc<dyn Fft<f64>>,
    forward_z: Arc<dyn Fft<f64>>,
    inverse_z: Arc<dyn Fft<f64>>,
}

impl Fft2 {
    pub fn new(nx: usize, nz: usize) -> Self {
        let mut planner = FftPlanner::new();

        Self {
            nx,
            nz,
            forward_x: planner.plan_fft_forward(nx),
            inverse_x: planner.plan_fft_inverse(nx),
            forward_z: planner.plan_fft_forward(nz),
            inverse_z: planner.plan_fft_inverse(nz),
        }
    }

    pub fn len(&self) -> usize {
        self.nx * self.nz
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn forward(&self, data: &mut [Complex64]) {
        self.transform(data, self.forward_x.as_ref(), self.forward_z.as_ref());
    }

    pub fn inverse_unnormalized(&self, data: &mut [Complex64]) {
        self.transform(data, self.inverse_x.as_ref(), self.inverse_z.as_ref());
    }

    pub fn inverse(&self, data: &mut [Complex64]) {
        self.inverse_unnormalized(data);

        let scale = 1.0 / self.len() as f64;
        data.iter_mut().for_each(|v| *v *= scale);
    }

    fn transform(&self, data: &mut [Complex64], along_x: &dyn Fft<f64>, along_z: &dyn Fft<f64>) {
        assert_eq!(data.len(), self.len(), "buffer must match the FFT size");

        along_x.process(data);

        let mut columns = vec![Complex64::default(); self.len()];

        for k in 0..self.nz {
            for i in 0..self.nx {
                columns[i * self.nz + k] = data[k * self.nx + i];
            }
        }

        along_z.process(&mut columns);

        for k in 0..self.nz {
            for i in 0..self.nx {
                data[k * self.nx + i] = columns[i * self.nz + k];
            }
        }
    }
}

pub fn signed_index(index: usize, n: usize) -> i64 {
    if index < n / 2 {
        index as i64
    } else {
        index as i64 - n as i64
    }
}

pub fn wavenumber(index: usize, n: usize, length: f64) -> f64 {
    TAU * signed_index(index, n) as f64 / length
}

pub fn is_nyquist(index: usize, n: usize) -> bool {
    n.is_multiple_of(2) && index == n / 2
}
