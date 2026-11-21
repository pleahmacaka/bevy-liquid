use std::f64::consts::TAU;

pub fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);

    x ^ (x >> 31)
}

pub fn hash(keys: &[u64]) -> u64 {
    keys.iter()
        .fold(0x2545_f491_4f6c_dd1d, |acc, &k| splitmix64(acc ^ k))
}

pub fn unit_open(bits: u64) -> f64 {
    ((bits >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

pub fn gaussian_pair(keys: &[u64]) -> [f64; 2] {
    let a = hash(keys);
    let b = splitmix64(a);
    let radius = (-2.0 * unit_open(a).ln()).sqrt();
    let angle = TAU * unit_open(b);

    [radius * angle.cos(), radius * angle.sin()]
}
