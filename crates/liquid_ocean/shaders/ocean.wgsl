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

const TAU: f32 = 6.28318530718;

fn mirror(index: u32, n: u32) -> u32 {
    let a = index % n;
    let b = index / n;
    return ((n - b) % n) * n + (n - a) % n;
}

fn rotation(angle: f32) -> vec2<f32> {
    return vec2<f32>(cos(angle), sin(angle));
}

@compute @workgroup_size(64)
fn evolve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = ocean.n;
    let cells = n * n;
    let idx = gid.x;
    let cascade = gid.y;
    if idx >= cells || cascade >= ocean.cascades {
        return;
    }

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

    let out = base * 4u + idx;
    spectrum[out] = eta + times_i(dx);
    spectrum[out + cells] = dz + times_i(slope_x);
    spectrum[out + 2u * cells] = slope_z + times_i(dxx);
    spectrum[out + 3u * cells] = dzz + times_i(dxz);
}

@compute @workgroup_size(64)
fn assemble(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = ocean.n;
    let cells = n * n;
    let idx = gid.x;
    let cascade = gid.y;
    if idx >= cells || cascade >= ocean.cascades {
        return;
    }

    let packed = cascade * 4u * cells + idx;
    let p0 = spectrum[packed];
    let p1 = spectrum[packed + cells];
    let p2 = spectrum[packed + 2u * cells];
    let p3 = spectrum[packed + 3u * cells];

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
