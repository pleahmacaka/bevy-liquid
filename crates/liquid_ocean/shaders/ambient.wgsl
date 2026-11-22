struct AmbientParams {
    count: u32,
    depth: f32,
    time: f32,
    pad: f32,
}

struct AmbientComponent {
    k: vec2<f32>,
    amplitude: f32,
    phase: f32,
    omega: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
}

struct AmbientSample {
    elevation: f32,
    velocity: vec3<f32>,
}

fn ambient_profiles(k: f32, z: f32, depth: f32) -> vec2<f32> {
    let a = exp(k * z);
    let b = exp(-k * (z + 2.0 * depth));
    let d = 1.0 - exp(-2.0 * k * depth);
    return vec2<f32>((a + b) / d, (a - b) / d);
}

fn ambient_angle(c: AmbientComponent, x: f32, z: f32, t: f32) -> f32 {
    return c.k.x * x + c.k.y * z - c.omega * t + c.phase;
}

fn ambient_elevation(x: f32, z: f32, t: f32) -> f32 {
    var eta = 0.0;
    for (var i = 0u; i < ambient.count; i = i + 1u) {
        let c = ambient_components[i];
        eta = eta + c.amplitude * cos(ambient_angle(c, x, z, t));
    }
    return eta;
}

fn ambient_flux(x: f32, z: f32, t: f32) -> vec2<f32> {
    var q = vec2<f32>(0.0);
    for (var i = 0u; i < ambient.count; i = i + 1u) {
        let c = ambient_components[i];
        let k = length(c.k);
        q = q + c.k / k * (c.amplitude * c.omega / k * cos(ambient_angle(c, x, z, t)));
    }
    return q;
}

fn ambient_sample(position: vec3<f32>, t: f32) -> AmbientSample {
    let eta = ambient_elevation(position.x, position.z, t);
    let depth = ambient.depth;
    let stretched = (position.y - eta) * depth / (depth + eta);
    var velocity = vec3<f32>(0.0);

    for (var i = 0u; i < ambient.count; i = i + 1u) {
        let c = ambient_components[i];
        let k = length(c.k);
        let angle = ambient_angle(c, position.x, position.z, t);
        let profile = ambient_profiles(k, stretched, depth);
        let horizontal = c.amplitude * c.omega * profile.x * cos(angle);
        let vertical = c.amplitude * c.omega * profile.y * sin(angle);
        velocity = velocity + vec3<f32>(horizontal * c.k.x / k, vertical, horizontal * c.k.y / k);
    }

    return AmbientSample(eta, velocity);
}
