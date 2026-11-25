struct HeatSource {
    center: vec4<f32>,
    extent: vec4<f32>,
    heat: vec4<f32>,
}

struct Heating {
    shift: vec4<f32>,
    cell: vec4<f32>,
    air: vec4<f32>,
    gas: vec4<u32>,
    count: vec4<u32>,
    sources: array<HeatSource, 8>,
}

struct GasInjection {
    position_mass: vec4<f32>,
    velocity_temperature: vec4<f32>,
    volume: f32,
}

struct GasInjections {
    count: atomic<u32>,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    items: array<GasInjection>,
}

@group(3) @binding(7) var<uniform> heating: Heating;
@group(3) @binding(8) var<storage, read_write> injections: GasInjections;
@group(3) @binding(9) var<storage, read> gas_flags: array<u32>;

fn in_source(s: HeatSource, x: vec3<f32>) -> bool {
    let p = x - s.center.xyz;

    if s.center.w < 0.5 {
        return length(p) <= s.extent.x;
    }

    return all(abs(p) <= s.extent.xyz);
}

fn relaxed(h: f32, goal: f32, rate: f32, dt: f32) -> f32 {
    return h + (1.0 - exp(-rate * dt)) * (goal - min(h, params.thermal.y));
}

fn gas_open(g: vec3<f32>) -> bool {
    let dims = heating.gas.xyz;
    let c = min(vec3<u32>(g * heating.cell.y), dims - 1u);

    return (gas_flags[c.x + dims.x * (c.y + dims.y * c.z)] & 1u) == 0u;
}

fn surface_above(x: vec3<f32>) -> vec3<f32> {
    var c = cell_of(x);
    let top = i32(params.dims.y) - 1;

    while c.y < top && phi_cell(c) < 0.0 {
        c.y = c.y + 1;
    }

    return vec3<f32>(x.x, max(x.y, (f32(c.y) + 0.5) * params.dx), x.z);
}

fn boil(x: vec3<f32>, v: vec3<f32>) -> bool {
    let g = surface_above(x) + heating.shift.xyz;
    let size = vec3<f32>(heating.gas.xyz) * heating.cell.x;
    let capacity = heating.gas.w;

    if any(g < vec3<f32>(0.0)) || any(g > size) || !gas_open(g) || atomicLoad(&injections.count) >= capacity {
        return false;
    }

    let slot = atomicAdd(&injections.count, 1u);

    if slot >= capacity {
        return false;
    }

    injections.items[slot] = GasInjection(vec4<f32>(g, heating.shift.w), vec4<f32>(v, params.thermal.y), heating.cell.z);

    return true;
}

@compute @workgroup_size(256)
fn heat_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= state.particle_count || particles[i].v.w > 0.5 {
        return;
    }

    let x = particles[i].x.xyz;
    let dt = state.dt;
    var h = particles[i].c0.w;

    for (var k = 0u; k < heating.count.x; k = k + 1u) {
        let s = heating.sources[k];

        if in_source(s, x) {
            h = relaxed(h, s.heat.x, s.heat.y, dt);
        }
    }

    h = min(relaxed(h, heating.air.x, heating.air.y, dt), heating.air.z);
    particles[i].c0.w = h;

    if h > params.thermal.y + params.thermal.z && boil(x, particles[i].v.xyz) {
        particles[i].v.w = 1.0;
    }
}
