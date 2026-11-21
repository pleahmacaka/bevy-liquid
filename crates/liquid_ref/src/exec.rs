use rayon::prelude::*;

pub const PARALLEL_MIN: usize = 1 << 15;

pub fn map<T: Send>(len: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if len < PARALLEL_MIN {
        (0..len).map(f).collect()
    } else {
        (0..len).into_par_iter().map(f).collect()
    }
}

pub fn each_mut<T: Send>(items: &mut [T], f: impl Fn(usize, &mut T) + Sync + Send) {
    if items.len() < PARALLEL_MIN {
        items.iter_mut().enumerate().for_each(|(n, t)| f(n, t));
    } else {
        items.par_iter_mut().enumerate().for_each(|(n, t)| f(n, t));
    }
}
