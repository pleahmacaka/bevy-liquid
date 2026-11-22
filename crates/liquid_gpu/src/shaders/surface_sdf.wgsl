struct Params {
    dims: vec4<u32>,
    bricks: vec4<u32>,
    dx: f32,
    iso: f32,
    band: f32,
    frame_dt: f32,
}

struct Closest {
    point: vec4<f32>,
    normal: vec4<f32>,
}

struct Probe {
    value: f32,
    gradient: vec3<f32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> density: array<f32>;
@group(0) @binding(2) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cell_start: array<u32>;
@group(0) @binding(4) var<storage, read> closest_in: array<Closest>;
@group(0) @binding(5) var<storage, read_write> closest_out: array<Closest>;
@group(0) @binding(6) var<storage, read_write> distances: array<f32>;
@group(0) @binding(7) var<storage, read_write> bricks: array<f32>;
@group(0) @binding(8) var<storage, read_write> motion: array<vec4<f32>>;

const BRICK: i32 = 8;
const MOTION_BAND: f32 = 2.0;
const NEWTON: i32 = 4;
const NEWTON_TOLERANCE: f32 = 1.0e-3;
const NEWTON_REACH: f32 = 2.0;

fn dims() -> vec3<i32> {
    return vec3<i32>(params.dims.xyz);
}

fn cell_index(c: vec3<i32>) -> u32 {
    let d = dims();
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn cell_coords(i: u32) -> vec3<i32> {
    let d = params.dims.xyz;
    return vec3<i32>(vec3<u32>(i % d.x, (i / d.x) % d.y, i / (d.x * d.y)));
}

fn in_grid(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < dims());
}

fn density_at(c: vec3<i32>) -> f32 {
    return density[cell_index(clamp(c, vec3<i32>(0), dims() - 1))];
}

fn probe(p: vec3<f32>) -> Probe {
    let b = floor(p);
    let t = p - b;
    let c = vec3<i32>(b);
    let v000 = density_at(c);
    let v100 = density_at(c + vec3<i32>(1, 0, 0));
    let v010 = density_at(c + vec3<i32>(0, 1, 0));
    let v110 = density_at(c + vec3<i32>(1, 1, 0));
    let v001 = density_at(c + vec3<i32>(0, 0, 1));
    let v101 = density_at(c + vec3<i32>(1, 0, 1));
    let v011 = density_at(c + vec3<i32>(0, 1, 1));
    let v111 = density_at(c + vec3<i32>(1, 1, 1));

    let x00 = mix(v000, v100, t.x);
    let x10 = mix(v010, v110, t.x);
    let x01 = mix(v001, v101, t.x);
    let x11 = mix(v011, v111, t.x);
    let y0 = mix(x00, x10, t.y);
    let y1 = mix(x01, x11, t.y);

    let gx = mix(mix(v100 - v000, v110 - v010, t.y), mix(v101 - v001, v111 - v011, t.y), t.z);
    let gy = mix(x10 - x00, x11 - x01, t.z);
    return Probe(mix(y0, y1, t.z), vec3<f32>(gx, gy, y1 - y0));
}

fn intercepts(c: vec3<i32>) -> vec3<f32> {
    let here = density[cell_index(c)];
    let inside = here >= params.iso;
    var q = vec3<f32>(0.0);

    for (var axis = 0; axis < 3; axis = axis + 1) {
        let along = vec3<i32>(axis) == vec3<i32>(0, 1, 2);
        var nearest = 2.0;
        var side = 0.0;

        for (var s = -1; s <= 1; s = s + 2) {
            let n = c + s * vec3<i32>(along);

            if !in_grid(n) {
                continue;
            }

            let there = density[cell_index(n)];

            if (there >= params.iso) != inside {
                let h = (params.iso - here) / (there - here);

                if h < nearest {
                    nearest = h;
                    side = f32(s);
                }
            }
        }

        if side != 0.0 {
            q = select(q, vec3<f32>(side / max(nearest, 1.0e-4)), along);
        }
    }

    return q;
}

@compute @workgroup_size(256)
fn seed_closest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= params.dims.w {
        return;
    }

    let c = cell_coords(i);
    let q = intercepts(c);
    var out = Closest(vec4<f32>(0.0), vec4<f32>(0.0));

    if any(q != vec3<f32>(0.0)) {
        let start = vec3<f32>(c);
        var p = start;
        var at = probe(p);

        for (var k = 0; k < NEWTON; k = k + 1) {
            let g2 = dot(at.gradient, at.gradient);

            if g2 < 1.0e-12 {
                break;
            }

            p = p - (at.value - params.iso) * at.gradient / g2;
            at = probe(p);
        }

        let converged = abs(at.value - params.iso) < NEWTON_TOLERANCE
            && length(p - start) < NEWTON_REACH
            && dot(at.gradient, at.gradient) > 1.0e-12;

        if converged {
            out = Closest(vec4<f32>(p, 1.0), vec4<f32>(normalize(at.gradient), 0.0));
        } else {
            out = Closest(vec4<f32>(start + q / dot(q, q), 1.0), vec4<f32>(normalize(q), 0.0));
        }
    }

    closest_out[i] = out;
}

@compute @workgroup_size(256)
fn spread_closest(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= params.dims.w {
        return;
    }

    let c = cell_coords(i);
    let here = vec3<f32>(c);
    var best = i;
    var best_d2 = 3.0e38;
    let own = closest_in[i].point;

    if own.w > 0.5 {
        let e = own.xyz - here;
        best_d2 = dot(e, e);
    }

    for (var z = -1; z <= 1; z = z + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            for (var x = -1; x <= 1; x = x + 1) {
                let n = c + vec3<i32>(x, y, z);

                if !in_grid(n) {
                    continue;
                }

                let k = cell_index(n);
                let candidate = closest_in[k].point;

                if candidate.w < 0.5 {
                    continue;
                }

                let e = candidate.xyz - here;
                let d2 = dot(e, e);

                if d2 < best_d2 {
                    best = k;
                    best_d2 = d2;
                }
            }
        }
    }

    closest_out[i] = closest_in[best];
}

fn displacement(cell: u32) -> vec4<f32> {
    var sum = vec4<f32>(0.0);

    let end = cell_start[cell + 1u];
    var j = cell_start[cell];

    while j + 4u <= end {
        let a = velocities[j].xyz;
        let b = velocities[j + 1u].xyz;
        let c = velocities[j + 2u].xyz;
        let d = velocities[j + 3u].xyz;
        sum = sum + vec4<f32>(a * params.frame_dt, 1.0);
        sum = sum + vec4<f32>(b * params.frame_dt, 1.0);
        sum = sum + vec4<f32>(c * params.frame_dt, 1.0);
        sum = sum + vec4<f32>(d * params.frame_dt, 1.0);
        j = j + 4u;
    }

    while j < end {
        sum = sum + vec4<f32>(velocities[j].xyz * params.frame_dt, 1.0);
        j = j + 1u;
    }

    return sum;
}

@compute @workgroup_size(256)
fn write_distance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;

    if i >= params.dims.w {
        return;
    }

    let c = cell_coords(i);
    let nearest = closest_in[i];
    var d = params.band;

    if nearest.point.w > 0.5 {
        let plane = abs(dot(nearest.normal.xyz, vec3<f32>(c) - nearest.point.xyz));
        d = min(plane * params.dx, params.band);
    }

    distances[i] = select(d, -d, density[i] >= params.iso);

    var m = vec4<f32>(0.0);

    if d < MOTION_BAND * params.dx {
        m = displacement(i);
    }

    motion[i] = m;
}

var<workgroup> lowest_rows: array<f32, 100>;

@compute @workgroup_size(10, 10, 1)
fn reduce_bricks(@builtin(workgroup_id) brick: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = vec2<i32>(brick.xy * u32(BRICK) + local.xy) - 1;
    let lane = local.x + 10u * local.y;
    var lowest = 3.0e38;

    if all(row >= vec2<i32>(0)) && all(row < dims().xy) {
        let z0 = max(i32(brick.z) * BRICK - 1, 0);
        let z1 = min(i32(brick.z) * BRICK + BRICK, dims().z - 1);

        for (var z = z0; z <= z1; z = z + 1) {
            lowest = min(lowest, distances[cell_index(vec3<i32>(row, z))]);
        }
    }

    lowest_rows[lane] = lowest;
    workgroupBarrier();

    if lane == 0u {
        for (var k = 1u; k < 100u; k = k + 1u) {
            lowest = min(lowest, lowest_rows[k]);
        }

        let n = params.bricks.xyz;
        bricks[brick.x + n.x * (brick.y + n.y * brick.z)] = lowest;
    }
}
