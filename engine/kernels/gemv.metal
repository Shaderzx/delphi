#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant uint GOLDEN = 0x9e3779b9u;
constant uint INPUT_SALT = 0xa5a5a5a5u;
constant uint VALUES_EXACT = 0u;
constant uint MAX_ROWS = 32u;
constant uint ROWS [[function_constant(0)]];

struct FillShape {
    uint seed;
    uint values;
    uint base;
    uint count;
};

struct GemvShape {
    uint rows;
    uint n;
    uint k;
    float scale;
};

uint lowbias32(uint a) {
    a ^= a >> 16;
    a *= 0x7feb352du;
    a ^= a >> 15;
    a *= 0x846ca68bu;
    a ^= a >> 16;
    return a;
}

float sample(uint seed, uint values, uint index) {
    uint h = lowbias32(seed * GOLDEN + index);
    if (values == VALUES_EXACT) {
        return float(int(h >> 24) - 128) / 128.0f;
    }
    return as_type<float>((h >> 9) | 0x3f800000u) - 1.5f;
}

kernel void fill_weights(device ushort* w [[buffer(0)]],
                         constant FillShape& f [[buffer(1)]],
                         uint gid [[thread_position_in_grid]],
                         uint grid [[threads_per_grid]]) {
    for (uint i = gid; i < f.count; i += grid) {
        w[i] = ushort(as_type<uint>(sample(f.seed, f.values, f.base + i)) >> 16);
    }
}

kernel void fill_inputs(device float* x [[buffer(0)]],
                        constant FillShape& f [[buffer(1)]],
                        uint gid [[thread_position_in_grid]],
                        uint grid [[threads_per_grid]]) {
    for (uint i = gid; i < f.count; i += grid) {
        x[i] = sample(f.seed ^ INPUT_SALT, f.values, f.base + i);
    }
}

float dot8(uint4 packed, device const float4* x) {
    float4 lo = float4(as_type<float>(packed.x << 16), as_type<float>(packed.x & 0xffff0000u),
                       as_type<float>(packed.y << 16), as_type<float>(packed.y & 0xffff0000u));
    float4 hi = float4(as_type<float>(packed.z << 16), as_type<float>(packed.z & 0xffff0000u),
                       as_type<float>(packed.w << 16), as_type<float>(packed.w & 0xffff0000u));
    float4 a = lo * x[0];
    float4 b = hi * x[1];
    return ((a.x + a.y) + (a.z + a.w)) + ((b.x + b.y) + (b.z + b.w));
}

kernel void gemv_rows(device const uint4* w [[buffer(0)]],
                      device const float4* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant GemvShape& s [[buffer(3)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint simds [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= s.n) {
        return;
    }
    uint chunks = s.k / 8u;
    device const uint4* wr = w + row * chunks;
    float acc[MAX_ROWS];
    for (uint m = 0; m < ROWS; ++m) {
        acc[m] = 0.0f;
    }
    for (uint c = lane; c < chunks; c += 32u) {
        uint4 packed = wr[c];
        for (uint m = 0; m < ROWS; ++m) {
            acc[m] += dot8(packed, x + (m * chunks + c) * 2u);
        }
    }
    for (uint m = 0; m < ROWS; ++m) {
        float total = simd_sum(acc[m]);
        if (lane == 0) {
            y[m * s.n + row] = total * s.scale;
        }
    }
}