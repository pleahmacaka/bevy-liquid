const THREADS: u32 = 256u;

var<workgroup> line: array<vec2<f32>, 1024>;

fn butterflies(lane: u32, n: u32, log2n: u32, inverse: bool, twiddle_stride: u32) {
    for (var s = 1u; s <= log2n; s = s + 1u) {
        let half = 1u << (s - 1u);
        let spacing = n / (2u * half);

        for (var t = lane; t < n / 2u; t = t + THREADS) {
            let pos = t % half;
            let a = (t / half) * 2u * half + pos;
            let b = a + half;
            var w = twiddles[pos * spacing * twiddle_stride];
            if inverse {
                w.y = -w.y;
            }
            let x = line[a];
            let y = complex_mul(line[b], w);
            line[a] = x + y;
            line[b] = x - y;
        }

        workgroupBarrier();
    }
}
