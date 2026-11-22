struct Lane {
    @builtin(local_invocation_index) index: u32,
}

var<workgroup> reduce_shared: array<f32, 256>;
var<workgroup> scan_shared: array<u32, 256>;

fn group_reduce(lane: Lane, value: f32, largest: bool) -> f32 {
    let lid = lane.index;
    reduce_shared[lid] = value;
    workgroupBarrier();

    var stride = WG / 2u;

    loop {
        if stride == 0u {
            break;
        }

        if lid < stride {
            let a = reduce_shared[lid];
            let b = reduce_shared[lid + stride];
            reduce_shared[lid] = select(a + b, max(a, b), largest);
        }

        workgroupBarrier();
        stride = stride / 2u;
    }

    let total = reduce_shared[0];
    workgroupBarrier();

    return total;
}

fn store_body(lane: Lane, lin: vec3<f32>, ang: vec3<f32>, base: u32) {
    let terms = array<f32, 6>(lin.x, lin.y, lin.z, ang.x, ang.y, ang.z);

    for (var e = 0u; e < 6u; e = e + 1u) {
        let total = group_reduce(lane, terms[e], false);

        if lane.index == 0u {
            partials[base + e] = total;
        }
    }
}

fn tail_coarse(t: Tail, lane: Lane) {
    tail_sweeps(t, lane.index);
}

fn lane_rank(lane: Lane) -> u32 {
    return lane.index;
}

fn block_scan(lane: Lane, value: u32) -> Scan {
    let lid = lane.index;
    scan_shared[lid] = value;
    workgroupBarrier();

    for (var offset = 1u; offset < WG; offset = offset * 2u) {
        var add = 0u;

        if lid >= offset {
            add = scan_shared[lid - offset];
        }

        workgroupBarrier();
        scan_shared[lid] = scan_shared[lid] + add;
        workgroupBarrier();
    }

    let inclusive = scan_shared[lid];
    let total = scan_shared[WG - 1u];
    workgroupBarrier();

    return Scan(inclusive - value, total);
}
