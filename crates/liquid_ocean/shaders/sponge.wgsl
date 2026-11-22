struct TrackParams {
    nx: u32,
    nz: u32,
    dx: f32,
    dz: f32,
    origin: vec2<f32>,
    level: f32,
    pad: f32,
}

struct Clock {
    dt: f32,
    remaining: f32,
    time: f32,
    elapsed: f32,
    rate: u32,
    steps: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> track: TrackParams;
@group(0) @binding(3) var<storage, read_write> sponge: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> bed: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> clock: Clock;

@compute @workgroup_size(64)
fn track_ambient(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let c = gid.x + gid.y * groups.x * 64u;
    if c >= track.nx * track.nz {
        return;
    }

    let layer = sponge[c];
    if layer.w <= 0.0 {
        return;
    }

    let i = c % track.nx;
    let k = c / track.nx;
    let x = track.origin.x + (f32(i) + 0.5) * track.dx;
    let z = track.origin.y + (f32(k) + 0.5) * track.dz;
    let eta = ambient_elevation(x, z, clock.time);
    let q = ambient_flux(x, z, clock.time);

    sponge[c] = vec4<f32>(bed_depth(bed[c], track.level + eta), q, layer.w);
}
