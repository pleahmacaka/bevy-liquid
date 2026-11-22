struct LevelParams {
    dims: vec4<u32>,
    regularization: f32,
    groups: u32,
    body_groups: u32,
    sparse: u32,
    bricks: u32,
    tall: u32,
    pack: u32,
    body_base: u32,
}

@group(1) @binding(0) var<uniform> lv: LevelParams;
@group(1) @binding(1) var<storage, read_write> l_phi: array<f32>;
@group(1) @binding(2) var<storage, read_write> l_coef: array<f32>;
@group(1) @binding(3) var<storage, read_write> l_active: array<u32>;
@group(1) @binding(4) var<storage, read_write> l_diag: array<f32>;
@group(1) @binding(5) var<storage, read_write> l_off: array<f32>;
@group(1) @binding(6) var<storage, read_write> l_x: array<f32>;
@group(1) @binding(7) var<storage, read_write> l_b: array<f32>;
@group(1) @binding(8) var<storage, read_write> l_r: array<f32>;
@group(1) @binding(9) var<storage, read_write> l_z: array<f32>;
@group(1) @binding(10) var<storage, read_write> l_d: array<f32>;
@group(1) @binding(11) var<storage, read_write> l_q: array<f32>;
@group(1) @binding(12) var<storage, read_write> l_pack: array<f32>;
@group(1) @binding(13) var<storage, read_write> l_e: array<f32>;

fn l_len() -> u32 {
    return lv.dims.w;
}

fn l_coords(n: u32) -> vec3<i32> {
    if level_bricked() {
        return cell_coords(n);
    }

    let d = lv.dims.xyz;
    return vec3<i32>(vec3<u32>(n % d.x, (n / d.x) % d.y, n / (d.x * d.y)));
}

fn l_index(c: vec3<i32>) -> u32 {
    if level_bricked() {
        return cell_index(c);
    }

    let d = vec3<i32>(lv.dims.xyz);
    return u32(c.x + d.x * (c.y + d.y * c.z));
}

fn l_inside(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(lv.dims.xyz));
}

fn l_columns() -> u32 {
    return lv.dims.x * lv.dims.z;
}

fn l_column(c: vec3<i32>) -> u32 {
    return u32(c.x) + lv.dims.x * u32(c.z);
}

fn floor_slot(col: u32) -> u32 {
    return 3u * l_len() + col;
}

fn ghost_slot(col: u32) -> u32 {
    return 3u * l_len() + l_columns() + col;
}

fn surface_slot(col: u32) -> u32 {
    return 3u * l_len() + 2u * l_columns() + col;
}

fn l_floor(c: vec3<i32>) -> f32 {
    if !OPEN_FLOOR || c.y != 0 {
        return 0.0;
    }

    return l_coef[floor_slot(l_column(c))];
}

fn level_bricked() -> bool {
    return SPARSE && lv.sparse != 0u;
}

fn level_items() -> u32 {
    if level_bricked() {
        return grid_cells();
    }

    return l_len();
}

fn level_item(m: u32) -> u32 {
    if level_bricked() {
        return masked_cell(m);
    }

    return m;
}

fn level_cell(t: u32) -> u32 {
    if level_bricked() {
        return brick_cell(t);
    }

    return t;
}

fn set_cell(t: u32, dims: vec3<u32>, members: u32) -> u32 {
    let count = brick_list[members];

    if count == NONE {
        return select(NONE, t, t < dims.x * dims.y * dims.z);
    }

    let s = t / BRICK_CELLS;

    if s >= count {
        return NONE;
    }

    let r = t % BRICK_CELLS;
    let c = vec3<u32>(unpack_origin(brick_list[members + 1u + s])) + vec3<u32>(r % BRICK, (r / BRICK) % BRICK, r / (BRICK * BRICK));

    if any(c >= dims) {
        return NONE;
    }

    return c.x + dims.x * (c.y + dims.y * c.z);
}

fn listed_cell(t: u32) -> u32 {
    if SPARSE && lv.bricks != 0u {
        return set_cell(t, lv.dims.xyz, lv.bricks);
    }

    return level_cell(t);
}

fn l_step(n: u32, c: vec3<i32>, a: u32, step: i32) -> u32 {
    let p = bump(c, a, step);

    if level_bricked() {
        let local = (c[a] & 7) + step;

        if local >= 0 && local < 8 {
            return u32(i32(n) + step * (1 << (3u * a)));
        }
    }

    return l_index(p);
}

fn l_neighbor_sum(n: u32, which: u32) -> f32 {
    let c = l_coords(n);
    let len = l_len();
    var s = 0.0;

    for (var a = 0u; a < 3u; a = a + 1u) {
        let k_hi = l_off[a * len + n];

        if k_hi != 0.0 {
            s = s + k_hi * l_value(which, l_step(n, c, a, 1), bump(c, a, 1));
        }

        if c[a] > 0 {
            let mi = l_step(n, c, a, -1);
            let k_lo = l_off[a * len + mi];

            if k_lo != 0.0 {
                s = s + k_lo * l_value(which, mi, bump(c, a, -1));
            }
        }
    }

    return s;
}

fn l_neighbor_sum_x(n: u32) -> f32 {
    return l_neighbor_sum(n, 0u);
}
