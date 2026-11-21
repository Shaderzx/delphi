#include <metal_stdlib>
using namespace metal;

constant float FP4_VALUES[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};
constant float FP4_BOUNDS[7] = {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5};

inline float rbf(float x) {
    uint u = as_type<uint>(x);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return as_type<float>(u);
}

inline float bfv(ushort w) { return as_type<float>(uint(w) << 16); }

inline float sigm(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

inline float e4m3_rne(float x) {
    if (!(x > 0.0f)) return 0.0f;
    if (x >= 448.0f) return 448.0f;
    float step = x < 0.015625f ? 0.001953125f : exp2(floor(log2(x)) - 3.0f);
    return min(rint(x / step) * step, 448.0f);
}

struct Gemv { uint rows; uint cols; uint round_out; float scale; };

kernel void gemv_bf16(device const uint4* w [[buffer(0)]],
                      device const float4* x [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant Gemv& g [[buffer(3)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint simds [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= g.rows) return;
    uint chunks = g.cols / 8u;
    device const uint4* wr = w + ulong(row) * chunks;
    float acc = 0.0f;
    for (uint c = lane; c < chunks; c += 32u) {
        uint4 p = wr[c];
        float4 a = x[c * 2u];
        float4 b = x[c * 2u + 1u];
        acc += as_type<float>(p.x << 16) * a.x + as_type<float>(p.x & 0xffff0000u) * a.y
             + as_type<float>(p.y << 16) * a.z + as_type<float>(p.y & 0xffff0000u) * a.w
             + as_type<float>(p.z << 16) * b.x + as_type<float>(p.z & 0xffff0000u) * b.y
             + as_type<float>(p.w << 16) * b.z + as_type<float>(p.w & 0xffff0000u) * b.w;
    }
    float total = simd_sum(acc) * g.scale;
    if (lane == 0) y[row] = g.round_out ? rbf(total) : total;
}

struct Norm { uint n; uint has_weight; float eps; };

kernel void rms_norm(device const float* x [[buffer(0)]],
                     device const ushort* w [[buffer(1)]],
                     device float* y [[buffer(2)]],
                     constant Norm& p [[buffer(3)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32];
    float s = 0.0f;
    for (uint i = tid; i < p.n; i += threads) s += x[i] * x[i];
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) total += part[i];
    float r = 1.0f / precise::sqrt(total / float(p.n) + p.eps);
    for (uint i = tid; i < p.n; i += threads) {
        float v = x[i] * r;
        y[i] = rbf(p.has_weight ? v * bfv(w[i]) : v);
    }
}

struct Res { uint n; uint sources; int addend; uint queries; uint normalize; uint offs[8]; uint outs[3]; };

kernel void attn_res(device float* arena [[buffer(0)]],
                     device const float* q [[buffer(1)]],
                     constant Res& p [[buffer(2)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32][8][4];
    threadgroup float stats[8][4];
    uint ns = p.sources;
    float ss[8], dot[8][3];
    for (uint s = 0; s < ns; ++s) { ss[s] = 0; for (uint k = 0; k < 3; ++k) dot[s][k] = 0; }
    for (uint i = tid; i < p.n; i += threads) {
        for (uint s = 0; s < ns; ++s) {
            float v = arena[p.offs[s] + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v += arena[uint(p.addend) + i];
            ss[s] += v * v;
            for (uint k = 0; k < p.queries; ++k) dot[s][k] += v * q[k * p.n + i];
        }
    }
    for (uint s = 0; s < ns; ++s) {
        float a = simd_sum(ss[s]);
        float b0 = simd_sum(dot[s][0]);
        float b1 = simd_sum(dot[s][1]);
        float b2 = simd_sum(dot[s][2]);
        if (lane == 0) { part[simd][s][0] = a; part[simd][s][1] = b0; part[simd][s][2] = b1; part[simd][s][3] = b2; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < ns * 4u) {
        uint s = tid / 4u, k = tid % 4u;
        float t = 0;
        for (uint g = 0; g < (threads + 31u) / 32u; ++g) t += part[g][s][k];
        stats[s][k] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rs[8], wt[3][8], inv[3];
    for (uint s = 0; s < ns; ++s) rs[s] = 1.0f / precise::sqrt(stats[s][0] / float(p.n) + 1e-6f);
    for (uint k = 0; k < p.queries; ++k) {
        float m = -INFINITY;
        for (uint s = 0; s < ns; ++s) m = max(m, stats[s][k + 1] * rs[s]);
        float tot = 0;
        for (uint s = 0; s < ns; ++s) { wt[k][s] = precise::exp(stats[s][k + 1] * rs[s] - m); tot += wt[k][s]; }
        inv[k] = 1.0f / tot;
        if (p.normalize) for (uint s = 0; s < ns; ++s) wt[k][s] *= rs[s];
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < p.n; i += threads) {
        float v[8];
        for (uint s = 0; s < ns; ++s) {
            v[s] = arena[p.offs[s] + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v[s] += arena[uint(p.addend) + i];
        }
        for (uint k = 0; k < p.queries; ++k) {
            float acc = 0;
            for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], v[s], acc);
            arena[p.outs[k] + i] = acc * inv[k];
        }
    }
}

struct Conv { uint channels; uint dynamic; uint head_dim; float scale; };

kernel void sconv(device const float* x [[buffer(0)]],
                  device float* state [[buffer(1)]],
                  device const ushort* w [[buffer(2)]],
                  device const float* dyn [[buffer(3)]],
                  device float* y [[buffer(4)]],
                  constant Conv& p [[buffer(5)]],
                  uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) return;
    float h[4] = {state[c], state[p.channels + c], state[2 * p.channels + c], x[c]};
    float acc = 0;
    if (p.dynamic) {
        uint head = c / p.head_dim;
        for (uint lag = 0; lag < 4; ++lag) acc += h[3 - lag] * (bfv(w[lag * p.channels + c]) + dyn[head * 4 + lag]);
    } else {
        for (uint tap = 0; tap < 4; ++tap) acc = fma(h[tap], bfv(w[c * 4 + tap]), acc);
    }
    y[c] = rbf((acc + h[3]) * p.scale);
    state[c] = h[1];
    state[p.channels + c] = h[2];
    state[2 * p.channels + c] = h[3];
}

struct Clip { uint width; uint has_scale; float threshold; float tau; uint mode; };

kernel void clip_rows(device const float* x [[buffer(0)]],
                      device const ushort* scale [[buffer(1)]],
                      device float* y [[buffer(2)]],
                      constant Clip& p [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    float v[4];
    float s = 0;
    uint n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) {
        float a = x[row * p.width + i];
        if (p.has_scale) a *= bfv(scale[row * p.width + i]);
        v[n] = a;
        s += a * a;
    }
    s = simd_sum(s);
    float coeff;
    if (p.mode == 1u) coeff = 1.0f / precise::sqrt(s / float(p.width) + 1e-6f);
    else coeff = min(p.threshold / (1e-6f + precise::sqrt(s)), 1.0f);
    n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) {
        float out = v[n] * coeff;
        if (p.tau != 1.0f) out = rbf(out) * p.tau;
        y[row * p.width + i] = rbf(out);
    }
}

kernel void rel_bias(device const float* r [[buffer(0)]],
                     device const ushort* proj [[buffer(1)]],
                     device float* rel [[buffer(2)]],
                     constant uint2& dims [[buffer(3)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint d = gid.x, h = gid.y;
    if (d >= dims.y) return;
    float acc = 0;
    for (uint i = 0; i < dims.x; ++i) acc += r[h * dims.x + i] * bfv(proj[i * dims.y + d]);
    rel[h * dims.y + d] = rbf(acc);
}

struct Attn { uint pos; uint start; uint kv_heads; uint group; uint head_dim; uint extent; float scale; };

kernel void attend(device const float* q [[buffer(0)]],
                   device const float* kc [[buffer(1)]],
                   device const float* vc [[buffer(2)]],
                   device const float* rel [[buffer(3)]],
                   device float* out [[buffer(4)]],
                   device float* lse [[buffer(5)]],
                   constant Attn& p [[buffer(6)]],
                   uint h [[threadgroup_position_in_grid]],
                   uint tid [[thread_position_in_threadgroup]],
                   uint threads [[threads_per_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[4096];
    threadgroup float part[32];
    threadgroup float qs[128];
    uint kvh = h / p.group;
    uint stride = p.kv_heads * p.head_dim;
    for (uint d = tid; d < p.head_dim; d += threads) qs[d] = q[h * p.head_dim + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint count = p.pos - p.start + 1u;
    float m = -INFINITY;
    for (uint t = tid; t < count; t += threads) {
        uint kp = p.start + t;
        device const float* k = kc + ulong(kp) * stride + kvh * p.head_dim;
        float s = 0;
        for (uint d = 0; d < p.head_dim; ++d) s += k[d] * qs[d];
        uint dist = p.pos - kp;
        s = s * p.scale + (dist < p.extent ? rel[h * p.extent + dist] : 0.0f);
        scores[t] = s;
        m = max(m, s);
    }
    m = simd_max(m);
    if (lane == 0) part[simd] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = -INFINITY;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) mx = max(mx, part[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint t = tid; t < count; t += threads) { float e = precise::exp(scores[t] - mx); scores[t] = e; sum += e; }
    sum = simd_sum(sum);
    if (lane == 0) part[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) tot += part[i];
    for (uint d = tid; d < p.head_dim; d += threads) {
        float acc = 0;
        for (uint t = 0; t < count; ++t) acc += scores[t] * vc[ulong(p.start + t) * stride + kvh * p.head_dim + d];
        out[h * p.head_dim + d] = acc / tot;
    }
    if (tid == 0) lse[h] = mx + precise::log(tot);
}

kernel void gate_out(device const float* o [[buffer(0)]],
                     device const float* gate [[buffer(1)]],
                     device const float* lse [[buffer(2)]],
                     device const float* sinks [[buffer(3)]],
                     device float* y [[buffer(4)]],
                     constant uint2& dims [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= dims.x * dims.y) return;
    uint h = i / dims.y;
    y[i] = rbf(o[i] * sigm(gate[i]) * sigm(lse[h] - sinks[h]));
}

kernel void kv_store(device const float* k [[buffer(0)]],
                     device const float* v [[buffer(1)]],
                     device float* kc [[buffer(2)]],
                     device float* vc [[buffer(3)]],
                     constant uint2& p [[buffer(4)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= p.y) return;
    kc[ulong(p.x) * p.y + i] = k[i];
    vc[ulong(p.x) * p.y + i] = v[i];
}

struct Route { uint experts; uint top; float scale; };

kernel void route(device const float* logits [[buffer(0)]],
                  device const float* bias [[buffer(1)]],
                  device uint* ids [[buffer(2)]],
                  device float* gammas [[buffer(3)]],
                  constant Route& p [[buffer(4)]],
                  uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float sel[256];
    threadgroup uint chosen[16];
    if (tid < p.experts) sel[tid] = sigm(logits[tid]) + bias[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint k = 0; k < p.top; ++k) {
            uint best = 0;
            float bv = -INFINITY;
            for (uint e = 0; e < p.experts; ++e) if (sel[e] > bv) { bv = sel[e]; best = e; }
            chosen[k] = best;
            sel[best] = -INFINITY;
        }
        float lp[16], m = -INFINITY;
        for (uint k = 0; k < p.top; ++k) {
            float l = logits[chosen[k]];
            lp[k] = min(l, 0.0f) - precise::log(1.0f + precise::exp(-fabs(l)));
            m = max(m, lp[k]);
        }
        float s[16];
        for (uint k = 0; k < p.top; ++k) s[k] = precise::exp(lp[k] - m);
        float tot = ((s[0] + s[2]) + (s[1] + s[3])) + ((s[4] + s[6]) + (s[5] + s[7]));
        for (uint k = 0; k < p.top; ++k) {
            ids[k] = chosen[k];
            gammas[k] = precise::exp(lp[k] - (m + precise::log(tot))) * p.scale;
        }
    }
}

kernel void quant_nvfp4(device const float* x [[buffer(0)]],
                        device float* operand [[buffer(1)]],
                        device float* outer [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32];
    device const float* xr = x + row * n;
    device float* opr = operand + row * n;
    float a = 0;
    for (uint i = tid; i < n; i += threads) a = max(a, fabs(xr[i]));
    a = simd_max(a);
    if (lane == 0) part[simd] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) amax = max(amax, part[i]);
    if (amax == 0.0f) {
        for (uint i = tid; i < n; i += threads) opr[i] = 0;
        if (tid == 0) outer[row] = 0;
        return;
    }
    float enc = min(1536.0f / amax, FLT_MAX);
    float dec = 1.0f / enc;
    float out_scale = amax * (1.0f / 1536.0f);
    if (tid == 0) outer[row] = out_scale;
    for (uint b = tid; b < n / 16u; b += threads) {
        float blk[16];
        float bmax = 0;
        for (uint j = 0; j < 16; ++j) { blk[j] = xr[b * 16 + j]; bmax = max(bmax, fabs(blk[j])); }
        float best_err[2];
        float vals[2][16];
        float scs[2];
        float ims[2] = {1.0f / 6.0f, 0.25f};
        for (uint c = 0; c < 2; ++c) {
            float sc = e4m3_rne(min(bmax * (enc * ims[c]), 448.0f));
            float inv = min(1.0f / (sc * dec), FLT_MAX);
            float err[16];
            for (uint j = 0; j < 16; ++j) {
                float v = blk[j] * inv;
                float av = fabs(v);
                uint code = 7;
                for (uint k = 0; k < 7; ++k) if (av <= FP4_BOUNDS[k]) { code = k; break; }
                if (code < 7 && av == FP4_BOUNDS[code] && (code & 1u)) code += 1;
                float r = FP4_VALUES[code] * (signbit(v) ? -1.0f : 1.0f);
                vals[c][j] = r * sc;
                err[j] = fabs((r * sc) * out_scale - blk[j]);
            }
            for (uint w = 16; w > 1; w /= 2) for (uint j = 0; j < w / 2; ++j) err[j] = err[2 * j] + err[2 * j + 1];
            best_err[c] = err[0];
            scs[c] = sc;
        }
        uint pick = best_err[1] < best_err[0] ? 1u : 0u;
        for (uint j = 0; j < 16; ++j) opr[b * 16 + j] = vals[pick][j];
    }
}

kernel void quant_mxfp8(device const float* x [[buffer(0)]],
                        device float* operand [[buffer(1)]],
                        constant uint& n [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint b = gid.x, row = gid.y;
    if (b >= n / 32u) return;
    device const float* xr = x + row * n + b * 32u;
    float bmax = 0;
    for (uint j = 0; j < 32; ++j) bmax = max(bmax, fabs(xr[j]));
    float scale = bmax > 0 ? exp2(clamp(floor(log2(bmax)) - 8.0f, -127.0f, 127.0f)) : exp2(-127.0f);
    for (uint j = 0; j < 32; ++j) {
        float v = xr[j] / scale;
        operand[row * n + b * 32u + j] = (v < 0 ? -1.0f : 1.0f) * e4m3_rne(fabs(v)) * scale;
    }
}

inline float fp4(uint code) {
    uint mag = code & 7u;
    float v = mag < 2u ? float(mag) * 0.5f : float(as_type<half>(ushort((mag << 9) + 0x3800u)));
    return (code & 8u) ? -v : v;
}

inline float4 fp4x4(uint bits) {
    return float4(fp4(bits & 15u), fp4((bits >> 4) & 15u), fp4((bits >> 8) & 15u), fp4((bits >> 12) & 15u));
}

struct Expert { uint rows; uint cols; uint in_stride; uint out_stride; uint per_row_input; ulong expert_bytes; ulong scale_bytes; };

kernel void expert_nvfp4(device const uchar* w [[buffer(0)]],
                         device const uchar* sc [[buffer(1)]],
                         device const float* amax [[buffer(2)]],
                         device const float* denom [[buffer(3)]],
                         device const float* x [[buffer(4)]],
                         device const float* outer [[buffer(5)]],
                         device const uint* ids [[buffer(6)]],
                         device float* y [[buffer(7)]],
                         constant Expert& p [[buffer(8)]],
                         constant float* e4m3 [[buffer(9)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint simds [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * simds + simd;
    uint k = group.y;
    if (row >= p.rows) return;
    uint e = ids[k];
    device const uchar* wr = w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const float* xr = x + (p.per_row_input ? k * p.in_stride : 0u);
    float acc = 0;
    device const float4* x4 = (device const float4*)xr;
    for (uint b = lane; b < p.cols / 16u; b += 32u) {
        uint2 packed = *(device const uint2*)(wr + b * 8u);
        float s = e4m3[sr[b]];
        float4 part = fp4x4(packed.x) * x4[b * 4u] + fp4x4(packed.x >> 16) * x4[b * 4u + 1u]
                    + fp4x4(packed.y) * x4[b * 4u + 2u] + fp4x4(packed.y >> 16) * x4[b * 4u + 3u];
        acc += ((part.x + part.y) + (part.z + part.w)) * s;
    }
    float total = simd_sum(acc);
    if (lane == 0) {
        float o = outer[p.per_row_input ? k : 0u];
        float r = total * o * (amax[e] / denom[0]);
        y[k * p.out_stride + row] = p.per_row_input ? rbf(r) : r;
    }
}

kernel void expert_mxfp8(device const uchar* w [[buffer(0)]],
                         device const uchar* sc [[buffer(1)]],
                         device const float* x [[buffer(4)]],
                         device const uint* ids [[buffer(6)]],
                         device float* y [[buffer(7)]],
                         constant Expert& p [[buffer(8)]],
                         constant float* e4m3 [[buffer(9)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint simds [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * simds + simd;
    uint k = group.y;
    if (row >= p.rows) return;
    uint e = ids[k];
    device const uchar* wr = w + ulong(e) * p.expert_bytes + ulong(row) * p.cols;
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 32u);
    device const float* xr = x + (p.per_row_input ? k * p.in_stride : 0u);
    float acc = 0;
    for (uint b = lane; b < p.cols / 32u; b += 32u) {
        float part = 0;
        for (uint j = 0; j < 32; ++j) part += e4m3[wr[b * 32u + j]] * xr[b * 32u + j];
        acc += part * exp2(float(sr[b]) - 127.0f);
    }
    float total = simd_sum(acc);
    if (lane == 0) y[k * p.out_stride + row] = rbf(total);
}

struct Swiglu { uint inter; uint top; float gate_cap; float up_cap; uint round_out; };

kernel void swiglu(device const float* gu [[buffer(0)]],
                   device const float* gammas [[buffer(1)]],
                   device float* h [[buffer(2)]],
                   constant Swiglu& p [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x, k = gid.y;
    if (j >= p.inter) return;
    float g = gu[k * 2u * p.inter + 2u * j];
    float u = gu[k * 2u * p.inter + 2u * j + 1u];
    float sg = sigm(g);
    float gc = p.gate_cap * precise::tanh(g / p.gate_cap);
    float uc = p.up_cap * precise::tanh(u / p.up_cap);
    float v = gc * sg * uc * gammas[k];
    h[k * p.inter + j] = p.round_out ? rbf(v) : v;
}

kernel void round_rows(device float* x [[buffer(0)]], constant uint& n [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = rbf(x[i]);
}

struct Combine { uint n; uint top; uint boundary; };

kernel void combine(device const float* rows [[buffer(0)]],
                    device const uint* ids [[buffer(1)]],
                    device const float* attention [[buffer(2)]],
                    device const float* partial [[buffer(3)]],
                    device float* conv_state [[buffer(4)]],
                    device const ushort* w [[buffer(5)]],
                    device float* out [[buffer(6)]],
                    constant Combine& p [[buffer(7)]],
                    uint c [[thread_position_in_grid]]) {
    if (c >= p.n) return;
    uint order[16];
    for (uint k = 0; k < p.top; ++k) order[k] = k;
    for (uint a = 1; a < p.top; ++a) for (uint b = a; b > 0 && ids[order[b - 1]] > ids[order[b]]; --b) { uint t = order[b]; order[b] = order[b - 1]; order[b - 1] = t; }
    float g = 0;
    for (uint k = 0; k < p.top; ++k) g += rows[order[k] * p.n + c];
    float cur = rbf(g);
    float h[4] = {conv_state[c], conv_state[p.n + c], conv_state[2 * p.n + c], cur};
    float acc = 0;
    for (uint tap = 0; tap < 4; ++tap) acc = fma(h[tap], bfv(w[c * 4 + tap]), acc);
    float yv = rbf((acc + cur) * 0.4472135955f);
    conv_state[c] = h[1];
    conv_state[p.n + c] = h[2];
    conv_state[2 * p.n + c] = cur;
    out[c] = p.boundary ? rbf(attention[c] + yv) : rbf((partial[c] + attention[c]) + yv);
}

struct Ctl { uint n; uint rank; };

kernel void ctl_normed(device const float* d [[buffer(0)]],
                       device const float* x [[buffer(1)]],
                       device float* normed [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2][32];
    float a = 0, b = 0;
    for (uint i = tid; i < n; i += threads) { a += d[i] * d[i]; b += x[i] * x[i]; }
    a = simd_sum(a); b = simd_sum(b);
    if (lane == 0) { part[0][simd] = a; part[1][simd] = b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ta = 0, tb = 0;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) { ta += part[0][i]; tb += part[1][i]; }
    float rd = 1.0f / precise::sqrt(ta / float(n) + 1e-6f);
    float rx = 1.0f / precise::sqrt(tb / float(n) + 1e-6f);
    for (uint i = tid; i < n; i += threads) { normed[i] = rbf(d[i] * rd); normed[n + i] = rbf(x[i] * rx); }
}

kernel void ctl_mix(device const float* d [[buffer(0)]],
                    device const float* x [[buffer(1)]],
                    device const float* gain [[buffer(2)]],
                    device const float* forget [[buffer(3)]],
                    device float* out [[buffer(4)]],
                    constant uint& n [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = (1.0f - precise::tanh(forget[i])) * d[i] + precise::tanh(gain[i]) * x[i];
}

kernel void embed_row(device const ushort* table [[buffer(0)]],
                      device const uint* token [[buffer(1)]],
                      device float* x [[buffer(2)]],
                      constant uint& n [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = bfv(table[ulong(token[0]) * n + i]);
}

struct Logits { uint n; uint valid; float cap; };

kernel void softcap_logits(device float* l [[buffer(0)]], constant Logits& p [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    l[i] = i < p.valid ? rbf(p.cap * precise::tanh(l[i] / p.cap)) : -INFINITY;
}

kernel void argmax_logits(device const float* l [[buffer(0)]],
                          device uint* token [[buffer(1)]],
                          constant uint& n [[buffer(2)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float bv[32];
    threadgroup uint bi[32];
    float best = -INFINITY;
    uint idx = 0;
    for (uint i = tid; i < n; i += threads) if (l[i] > best) { best = l[i]; idx = i; }
    for (uint off = 16; off > 0; off /= 2) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(idx, off);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (lane == 0) { bv[simd] = best; bi[simd] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint i = 1; i < (threads + 31u) / 32u; ++i) if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
        best = bv[0]; idx = bi[0];
        for (uint i = 1; i < (threads + 31u) / 32u; ++i) if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
        token[0] = idx;
    }
}