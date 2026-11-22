fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn times_i(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.y, a.x);
}

// WARP's DXIL tanh exceeds 1 above x = 2.4 and FXC's exp ratio turns NaN for large x, so build it from exp(-2x)
fn tanh_positive(x: f32) -> f32 {
    if x < 0.125 {
        let x2 = x * x;
        return x * (1.0 - x2 * (1.0 / 3.0 - x2 * (2.0 / 15.0)));
    }

    let e = exp(-2.0 * x);
    return (1.0 - e) / (1.0 + e);
}

fn signed_index(index: u32, n: u32) -> f32 {
    if index < n / 2u {
        return f32(index);
    }
    return f32(index) - f32(n);
}
