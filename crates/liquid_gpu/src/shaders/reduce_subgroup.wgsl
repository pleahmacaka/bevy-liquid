struct Lane {
    @builtin(local_invocation_index) index: u32,
}

struct Wave {
    sub: u32,
    size: u32,
    group: u32,
    groups: u32,
}

var<workgroup> reduce_shared: array<f32, 64>;
var<workgroup> scan_shared: array<u32, 64>;

// naga's HLSL output gives subgroup builtins inside an entry-point struct no semantic, so DXC rejects them; derive them here instead
fn wave(lane: Lane) -> Wave {
    let size = subgroupAdd(1u);
    let base = subgroupMin(lane.index);

    return Wave(subgroupExclusiveAdd(1u), size, base / size, WG / size);
}

var<workgroup> body_shared: array<vec4<f32>, 128>;

fn lanes(value: f32, largest: bool) -> f32 {
    if largest {
        return subgroupMax(value);
    }

    return subgroupAdd(value);
}

fn group_reduce(lane: Lane, value: f32, largest: bool) -> f32 {
    let w = wave(lane);
    let part = lanes(value, largest);

    if w.sub == 0u {
        reduce_shared[w.group] = part;
    }

    workgroupBarrier();

    var acc = 0.0;

    for (var i = w.sub; i < w.groups; i = i + w.size) {
        acc = select(acc + reduce_shared[i], max(acc, reduce_shared[i]), largest);
    }

    let total = lanes(acc, largest);
    workgroupBarrier();

    return total;
}

fn store_body(lane: Lane, lin: vec3<f32>, ang: vec3<f32>, base: u32) {
    let w = wave(lane);
    let a = subgroupAdd(lin);
    let b = subgroupAdd(ang);

    if w.sub == 0u {
        body_shared[2u * w.group] = vec4<f32>(a, 0.0);
        body_shared[2u * w.group + 1u] = vec4<f32>(b, 0.0);
    }

    workgroupBarrier();

    if lane.index < 6u {
        var acc = 0.0;

        for (var g = 0u; g < w.groups; g = g + 1u) {
            acc = acc + body_shared[2u * g + lane.index / 3u][lane.index % 3u];
        }

        partials[base + lane.index] = acc;
    }

    workgroupBarrier();
}

var<workgroup> coarse_fit: u32;

fn tail_coarse(t: Tail, lane: Lane) {
    let w = wave(lane);

    if lane.index == 0u {
        coarse_fit = select(0u, 1u, w.size >= t.len);
    }

    if workgroupUniformLoad(&coarse_fit) == 0u {
        tail_sweeps(t, lane.index);
        return;
    }

    let i = min(w.sub, t.len - 1u);
    let c = tail_coords(t, i);
    let stride = vec3<u32>(1u, u32(t.dims.x), u32(t.dims.x * t.dims.y));
    let hi = min(vec3<u32>(i) + stride, vec3<u32>(t.len - 1u));
    let lo = select(vec3<u32>(i), vec3<u32>(i) - stride, c > vec3<i32>(0));
    let k_hi = vec3<f32>(tail_coef(t, i, 1u), tail_coef(t, i, 2u), tail_coef(t, i, 3u));
    let k_lo = select(vec3<f32>(0.0), vec3<f32>(tail_coef(t, lo.x, 1u), tail_coef(t, lo.y, 2u), tail_coef(t, lo.z, 3u)), c > vec3<i32>(0));
    let b = tail_lds[tail_b(t, i)];
    let diag = abs(tail_coef(t, i, 0u));
    let live = w.sub < t.len && tail_live(t, i, 1u);
    let parity = u32(c.x + c.y + c.z) % 2u;
    var x = tail_lds[tail_x(t, i)];

    for (var sweep = 0u; sweep < 2u * SWEEPS; sweep = sweep + 1u) {
        for (var k = 0u; k < 2u; k = k + 1u) {
            let x_hi = vec3<f32>(subgroupShuffle(x, hi.x), subgroupShuffle(x, hi.y), subgroupShuffle(x, hi.z));
            let x_lo = vec3<f32>(subgroupShuffle(x, lo.x), subgroupShuffle(x, lo.y), subgroupShuffle(x, lo.z));

            if live && parity == (k + sweep / SWEEPS) % 2u {
                x = (b + dot(k_hi, x_hi) + dot(k_lo, x_lo)) / diag;
            }
        }
    }

    if w.group == 0u && w.sub < t.len {
        tail_lds[tail_x(t, w.sub)] = x;
    }

    workgroupBarrier();
}

fn lane_rank(lane: Lane) -> u32 {
    let w = wave(lane);

    return w.group * w.size + w.sub;
}

fn block_scan(lane: Lane, value: u32) -> Scan {
    let w = wave(lane);
    let exclusive = subgroupExclusiveAdd(value);

    if w.sub + 1u == w.size {
        scan_shared[w.group] = exclusive + value;
    }

    workgroupBarrier();

    var before = 0u;
    var total = 0u;

    for (var g = 0u; g < w.groups; g = g + 1u) {
        let sum = scan_shared[g];
        before = before + select(0u, sum, g < w.group);
        total = total + sum;
    }

    workgroupBarrier();

    return Scan(before + exclusive, total);
}
