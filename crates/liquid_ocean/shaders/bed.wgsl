fn bed_shape(b: vec4<f32>) -> vec2<f32> {
    return vec2<f32>(b.z - b.y, (b.x - b.y) / (b.z - b.x) + 1.0);
}

fn bed_full(b: vec4<f32>, h: f32) -> bool {
    return h >= b.z - b.x;
}

fn bed_level(b: vec4<f32>, h: f32) -> f32 {
    if h <= 0.0 {
        return b.y;
    }
    if bed_full(b, h) {
        return h + b.x;
    }
    let shape = bed_shape(b);
    return b.y + shape.x * pow(h * shape.y / shape.x, 1.0 / shape.y);
}

fn bed_depth(b: vec4<f32>, level: f32) -> f32 {
    if level <= b.y {
        return 0.0;
    }
    if level >= b.z {
        return level - b.x;
    }
    let shape = bed_shape(b);
    return shape.x * pow((level - b.y) / shape.x, shape.y) / shape.y;
}
