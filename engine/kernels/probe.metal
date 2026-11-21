#include <metal_stdlib>
using namespace metal;

constant uint PATTERN = 2654435761u;

kernel void fill_pattern(device uint4* data [[buffer(0)]],
                         constant uint& count4 [[buffer(1)]],
                         uint gid [[thread_position_in_grid]],
                         uint grid [[threads_per_grid]]) {
    for (uint i = gid; i < count4; i += grid) {
        uint base = i * 4u;
        data[i] = uint4(base, base + 1u, base + 2u, base + 3u) * PATTERN;
    }
}

kernel void sum_stream(device const uint4* data [[buffer(0)]],
                       constant uint& count4 [[buffer(1)]],
                       device atomic_uint* total [[buffer(2)]],
                       uint gid [[thread_position_in_grid]],
                       uint grid [[threads_per_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    uint4 acc = 0;
    for (uint i = gid; i < count4; i += grid) {
        acc += data[i];
    }
    uint lane_total = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) {
        atomic_fetch_add_explicit(total, lane_total, memory_order_relaxed);
    }
}