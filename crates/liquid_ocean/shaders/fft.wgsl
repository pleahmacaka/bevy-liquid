struct FftParams {
    n: u32,
    log2n: u32,
    stride: u32,
    line_stride: u32,
    field_stride: u32,
    inverse: u32,
    twiddle_stride: u32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> fft: FftParams;
@group(0) @binding(1) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> twiddles: array<vec2<f32>>;

@compute @workgroup_size(256)
fn fft_lines(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let base = wid.y * fft.field_stride + wid.x * fft.line_stride;
    let shift = 32u - fft.log2n;

    for (var j = lid.x; j < fft.n; j = j + THREADS) {
        line[reverseBits(j) >> shift] = data[base + j * fft.stride];
    }

    workgroupBarrier();
    butterflies(lid.x, fft.n, fft.log2n, fft.inverse != 0u, fft.twiddle_stride);

    for (var j = lid.x; j < fft.n; j = j + THREADS) {
        data[base + j * fft.stride] = line[j];
    }
}
