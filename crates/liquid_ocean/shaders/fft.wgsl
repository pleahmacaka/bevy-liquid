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

const THREADS: u32 = 256u;

var<workgroup> line: array<vec2<f32>, 1024>;

@compute @workgroup_size(256)
fn fft_lines(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let base = wid.y * fft.field_stride + wid.x * fft.line_stride;
    let shift = 32u - fft.log2n;

    for (var j = lid.x; j < fft.n; j = j + THREADS) {
        line[reverseBits(j) >> shift] = data[base + j * fft.stride];
    }

    workgroupBarrier();

    for (var s = 1u; s <= fft.log2n; s = s + 1u) {
        let half = 1u << (s - 1u);
        let step = fft.n / (2u * half);

        for (var t = lid.x; t < fft.n / 2u; t = t + THREADS) {
            let pos = t % half;
            let a = (t / half) * 2u * half + pos;
            let b = a + half;
            var w = twiddles[pos * step * fft.twiddle_stride];
            if fft.inverse != 0u {
                w.y = -w.y;
            }
            let x = line[a];
            let y = complex_mul(line[b], w);
            line[a] = x + y;
            line[b] = x - y;
        }

        workgroupBarrier();
    }

    for (var j = lid.x; j < fft.n; j = j + THREADS) {
        data[base + j * fft.stride] = line[j];
    }
}
