struct OceanParams {
    n: u32,
    cascades: u32,
    time: f32,
    dt: f32,
    choppiness: f32,
    foam_threshold: f32,
    foam_gain: f32,
    foam_decay: f32,
    patch_size: vec4<f32>,
}

@group(0) @binding(0) var<uniform> ocean: OceanParams;
@group(0) @binding(1) var<storage, read> h0: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read> omega: array<f32>;
@group(0) @binding(3) var<storage, read_write> spectrum: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> surface: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> twiddles: array<vec2<f32>>;

const TAU: f32 = 6.28318530718;

fn mirror(index: u32, n: u32) -> u32 {
    let a = index % n;
    let b = index / n;
    return ((n - b) % n) * n + (n - a) % n;
}

fn rotation(angle: f32) -> vec2<f32> {
    return vec2<f32>(cos(angle), sin(angle));
}

const COLUMN_ITEMS: u32 = 4u;

fn evolved(cascade: u32, idx: u32, field: u32) -> vec2<f32> {
    let n = ocean.n;
    let cells = n * n;
    let base = cascade * cells;
    let w = omega[base + idx];
    let phase = rotation(-w * ocean.time);
    let a = h0[base + idx];
    let b = h0[base + mirror(idx, n)];
    let mirrored = complex_mul(b, phase);
    let h = complex_mul(a, phase) + vec2<f32>(mirrored.x, -mirrored.y);

    let dk = TAU / ocean.patch_size[cascade];
    let kx = signed_index(idx % n, n) * dk;
    let kz = signed_index(idx / n, n) * dk;
    let k = sqrt(kx * kx + kz * kz);
    var ux = 0.0;
    var uz = 0.0;
    if k > 0.0 {
        ux = kx / k;
        uz = kz / k;
    }

    let lambda = ocean.choppiness;
    let eta = h;
    let dx = times_i(h) * (lambda * ux);
    let dz = times_i(h) * (lambda * uz);
    let slope_x = times_i(h) * kx;
    let slope_z = times_i(h) * kz;
    let dxx = h * (-lambda * kx * ux);
    let dzz = h * (-lambda * kz * uz);
    let dxz = h * (-lambda * kx * uz);

    switch field {
        case 0u: {
            return eta + times_i(dx);
        }
        case 1u: {
            return dz + times_i(slope_x);
        }
        case 2u: {
            return slope_z + times_i(dxx);
        }
        default: {
            return dzz + times_i(dxz);
        }
    }
}

@compute @workgroup_size(256)
fn evolve_rows(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = ocean.n;
    let log2n = countTrailingZeros(n);
    let shift = 32u - log2n;
    let row = wid.x;
    let cascade = wid.y / 4u;
    let field = wid.y % 4u;

    for (var j = lid.x; j < n; j = j + THREADS) {
        line[reverseBits(j) >> shift] = evolved(cascade, row * n + j, field);
    }

    workgroupBarrier();
    butterflies(lid.x, n, log2n, true, 1u);

    let base = wid.y * n * n + row * n;

    for (var j = lid.x; j < n; j = j + THREADS) {
        spectrum[base + j] = line[j];
    }
}

@compute @workgroup_size(256)
fn columns_assemble(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let n = ocean.n;
    let cells = n * n;
    let log2n = countTrailingZeros(n);
    let shift = 32u - log2n;
    let column = wid.x;
    let cascade = wid.y;
    var kept: array<vec2<f32>, 16>;

    for (var field = 0u; field < 4u; field = field + 1u) {
        let base = (cascade * 4u + field) * cells + column;

        for (var j = lid.x; j < n; j = j + THREADS) {
            line[reverseBits(j) >> shift] = spectrum[base + j * n];
        }

        workgroupBarrier();
        butterflies(lid.x, n, log2n, true, 1u);

        for (var t = 0u; t < COLUMN_ITEMS; t = t + 1u) {
            let j = lid.x + t * THREADS;

            if j < n {
                kept[field * COLUMN_ITEMS + t] = line[j];
            }
        }

        workgroupBarrier();
    }

    for (var t = 0u; t < COLUMN_ITEMS; t = t + 1u) {
        let j = lid.x + t * THREADS;

        if j < n {
            assemble(cascade, j * n + column, kept[t], kept[COLUMN_ITEMS + t], kept[2u * COLUMN_ITEMS + t], kept[3u * COLUMN_ITEMS + t]);
        }
    }
}

fn assemble(cascade: u32, idx: u32, p0: vec2<f32>, p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) {
    let cells = ocean.n * ocean.n;
    let eta = p0.x;
    let dx = p0.y;
    let dz = p1.x;
    let slope_x = p1.y;
    let slope_z = p2.x;
    let dxx = p2.y;
    let dzz = p3.x;
    let dxz = p3.y;
    let jacobian = (1.0 + dxx) * (1.0 + dzz) - dxz * dxz;

    let out = 2u * (cascade * cells + idx);
    let previous = surface[out + 1u].z;
    let keep = exp(-ocean.dt / ocean.foam_decay);
    let produced = ocean.foam_gain * ocean.dt * max(ocean.foam_threshold - jacobian, 0.0);
    let foam = clamp(previous * keep + produced, 0.0, 1.0);

    surface[out] = vec4<f32>(dx, eta, dz, jacobian);
    surface[out + 1u] = vec4<f32>(slope_x, slope_z, foam, 0.0);
}
