#include <metal_stdlib>
using namespace metal;

constant float FP4_VALUES[16] = {0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6};

inline float rbf(float x) {
    uint u = as_type<uint>(x);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) & 0xffff0000u;
    return as_type<float>(u);
}

inline ushort f2bf(float x) {
    uint u = as_type<uint>(x);
    u = (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
    return ushort(u);
}

inline float bfv(ushort w) { return as_type<float>(uint(w) << 16); }

inline float sigm(float x) { return 1.0f / (1.0f + precise::exp(-x)); }

inline float e4m3_rne(float x) {
    if (!(x > 0.0f)) return 0.0f;
    if (x >= 448.0f) return 448.0f;
    if (x < 0.015625f) return rint(x * 512.0f) * 0.001953125f;
    uint u = as_type<uint>(x);
    u = (u + 0x7ffffu + ((u >> 20) & 1u)) & 0xfff00000u;
    return min(as_type<float>(u), 448.0f);
}

inline float e4m3_decode(uchar b) {
    uint e = (b >> 3) & 15u;
    uint m = b & 7u;
    float v = select(as_type<float>(((e + 120u) << 23) | (m << 20)), float(m) * 0.001953125f, e == 0u);
    return select(v, -v, (b & 0x80u) != 0u);
}

inline float fp4_rne_val(float v) {
    float av = fabs(v);
    uint code = uint(av > 0.25f) + uint(av >= 0.75f) + uint(av > 1.25f) + uint(av >= 1.75f)
              + uint(av > 2.5f) + uint(av >= 3.5f) + uint(av > 5.0f);
    return FP4_VALUES[code] * (signbit(v) ? -1.0f : 1.0f);
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

kernel void rms_norm(device const float4* x [[buffer(0)]],
                     device const ushort* w [[buffer(1)]],
                     device float4* y [[buffer(2)]],
                     constant Norm& p [[buffer(3)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32];
    uint n4 = p.n / 4u;
    float4 v = tid < n4 ? x[tid] : float4(0.0f);
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 t = x[i];
        s += dot(t, t);
    }
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) total += part[i];
    float r = 1.0f / precise::sqrt(total / float(p.n) + p.eps);
    if (tid < n4) {
        float4 o = v * r;
        if (p.has_weight) {
            uint b = tid * 4u;
            o *= float4(bfv(w[b]), bfv(w[b + 1u]), bfv(w[b + 2u]), bfv(w[b + 3u]));
        }
        y[tid] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 o = x[i] * r;
        if (p.has_weight) {
            uint b = i * 4u;
            o *= float4(bfv(w[b]), bfv(w[b + 1u]), bfv(w[b + 2u]), bfv(w[b + 3u]));
        }
        y[i] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
}

struct Res { uint n; uint sources; int addend; uint queries; uint normalize; uint has_norm; uint offs[8]; uint outs[3]; uint pad; ulong norm_offs[3]; };

kernel void attn_res(device float* arena [[buffer(0)]],
                     device const float* q [[buffer(1)]],
                     device const ushort* weights [[buffer(2)]],
                     constant Res& p [[buffer(3)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20][8][4];
    threadgroup float stats[8][4];
    uint ns = p.sources;
    uint n4 = p.n / 4u;
    device float4* arena4 = (device float4*)arena;
    device const float4* q4 = (device const float4*)q;
    float ss[8], dotv[8][3];
    float4 cached[8];
    for (uint s = 0; s < ns; ++s) { ss[s] = 0; for (uint k = 0; k < 3; ++k) dotv[s][k] = 0; }
    if (tid < n4) {
        for (uint s = 0; s < ns; ++s) {
            float4 v = arena4[p.offs[s] / 4u + tid];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v += arena4[uint(p.addend) / 4u + tid];
            cached[s] = v;
            ss[s] = dot(v, v);
            for (uint k = 0; k < p.queries; ++k) dotv[s][k] = dot(v, q4[k * n4 + tid]);
        }
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        for (uint s = 0; s < ns; ++s) {
            float4 v = arena4[p.offs[s] / 4u + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v += arena4[uint(p.addend) / 4u + i];
            ss[s] += dot(v, v);
            for (uint k = 0; k < p.queries; ++k) dotv[s][k] += dot(v, q4[k * n4 + i]);
        }
    }
    for (uint s = 0; s < ns; ++s) {
        float a = simd_sum(ss[s]);
        float b0 = simd_sum(dotv[s][0]);
        float b1 = simd_sum(dotv[s][1]);
        float b2 = simd_sum(dotv[s][2]);
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
    if (!p.has_norm) {
        if (tid < n4) {
            for (uint k = 0; k < p.queries; ++k) {
                float4 acc = 0;
                for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], cached[s], acc);
                arena4[p.outs[k] / 4u + tid] = acc * inv[k];
            }
        }
        for (uint i = tid + threads; i < n4; i += threads) {
            float4 v[8];
            for (uint s = 0; s < ns; ++s) {
                v[s] = arena4[p.offs[s] / 4u + i];
                if (int(s) == int(ns) - 1 && p.addend >= 0) v[s] += arena4[uint(p.addend) / 4u + i];
            }
            for (uint k = 0; k < p.queries; ++k) {
                float4 acc = 0;
                for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], v[s], acc);
                arena4[p.outs[k] / 4u + i] = acc * inv[k];
            }
        }
        return;
    }
    float4 outv[3];
    float nss[3] = {0.0f, 0.0f, 0.0f};
    if (tid < n4) {
        for (uint k = 0; k < p.queries; ++k) {
            float4 acc = 0;
            for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], cached[s], acc);
            outv[k] = acc * inv[k];
            nss[k] = dot(outv[k], outv[k]);
        }
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 v[8];
        for (uint s = 0; s < ns; ++s) {
            v[s] = arena4[p.offs[s] / 4u + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v[s] += arena4[uint(p.addend) / 4u + i];
        }
        for (uint k = 0; k < p.queries; ++k) {
            float4 acc = 0;
            for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], v[s], acc);
            float4 ov = acc * inv[k];
            nss[k] += dot(ov, ov);
        }
    }
    for (uint k = 0; k < p.queries; ++k) {
        float a = simd_sum(nss[k]);
        if (lane == 0) part[simd][0][k] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float nr[3];
    for (uint k = 0; k < p.queries; ++k) {
        float t = 0.0f;
        for (uint g = 0; g < (threads + 31u) / 32u; ++g) t += part[g][0][k];
        nr[k] = 1.0f / precise::sqrt(t / float(p.n) + 1e-6f);
    }
    if (tid < n4) {
        uint b = tid * 4u;
        for (uint k = 0; k < p.queries; ++k) {
            device const ushort* nw = weights + p.norm_offs[k] / 2u;
            float4 o = outv[k] * nr[k] * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
            arena4[p.outs[k] / 4u + tid] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
        }
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 v[8];
        for (uint s = 0; s < ns; ++s) {
            v[s] = arena4[p.offs[s] / 4u + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v[s] += arena4[uint(p.addend) / 4u + i];
        }
        uint b = i * 4u;
        for (uint k = 0; k < p.queries; ++k) {
            float4 acc = 0;
            for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], v[s], acc);
            device const ushort* nw = weights + p.norm_offs[k] / 2u;
            float4 o = (acc * inv[k]) * nr[k] * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
            arena4[p.outs[k] / 4u + i] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
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

struct ClipKv { uint width; uint pos; uint stride; uint mode; uint kv_cap; float threshold; };

kernel void clip_kv_store(device const float* x [[buffer(0)]],
                          device ushort* cache [[buffer(1)]],
                          constant ClipKv& p [[buffer(2)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    float v[4];
    float s = 0;
    uint n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) {
        float a = x[row * p.width + i];
        v[n] = a;
        s += a * a;
    }
    s = simd_sum(s);
    float coeff = p.mode == 1u ? (1.0f / precise::sqrt(s / float(p.width) + 1e-6f)) : min(p.threshold / (1e-6f + precise::sqrt(s)), 1.0f);
    device ushort* dst = cache + ulong(p.pos % p.kv_cap) * p.stride + row * p.width;
    n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) {
        dst[i] = f2bf(v[n] * coeff);
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

struct Attn { uint pos; uint start; uint kv_heads; uint group; uint head_dim; uint extent; uint kv_cap; float scale; };

kernel void attend_gated(device const float4* q [[buffer(0)]],
                         device const ushort4* kc [[buffer(1)]],
                         device const ushort4* vc [[buffer(2)]],
                         device const float* rel [[buffer(3)]],
                         device const float4* gate [[buffer(4)]],
                         device const float* sinks [[buffer(5)]],
                         device float4* gated [[buffer(6)]],
                         constant Attn& p [[buffer(7)]],
                         uint h [[threadgroup_position_in_grid]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[4];
    threadgroup float sl[4];
    threadgroup float4 so[4][32];
    uint kvh = h / p.group;
    float4 q4 = q[h * 32u + lane];
    uint count = p.pos - p.start + 1u;
    float m = -INFINITY;
    float l = 0.0f;
    float4 o4 = 0.0f;
    for (uint t = simd; t < count; t += 4u) {
        uint kp = p.start + t;
        ulong base = (ulong(kp % p.kv_cap) * p.kv_heads + kvh) * 32u + lane;
        ushort4 ku = kc[base];
        float4 k4 = float4(bfv(ku.x), bfv(ku.y), bfv(ku.z), bfv(ku.w));
        float s = simd_sum(dot(q4, k4)) * p.scale;
        uint dist = p.pos - kp;
        s += dist < p.extent ? rel[h * p.extent + dist] : 0.0f;
        float mn = max(m, s);
        float a = m == -INFINITY ? 0.0f : precise::exp(m - mn);
        float b = precise::exp(s - mn);
        ushort4 vu = vc[base];
        float4 v4 = float4(bfv(vu.x), bfv(vu.y), bfv(vu.z), bfv(vu.w));
        o4 = o4 * a + v4 * b;
        l = l * a + b;
        m = mn;
    }
    if (lane == 0) { sm[simd] = m; sl[simd] = l; }
    so[simd][lane] = o4;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float mx = max(max(sm[0], sm[1]), max(sm[2], sm[3]));
        float tot = 0.0f;
        float4 acc = 0.0f;
        for (uint s = 0; s < 4u; ++s) {
            if (sm[s] != -INFINITY) {
                float w = precise::exp(sm[s] - mx);
                tot += sl[s] * w;
                acc += so[s][lane] * w;
            }
        }
        float lse = mx + precise::log(tot);
        float ssink = sigm(lse - sinks[h]);
        float4 g4 = gate[h * 32u + lane];
        float4 sg = float4(sigm(g4.x), sigm(g4.y), sigm(g4.z), sigm(g4.w));
        float4 y4 = (acc / tot) * sg * ssink;
        gated[h * 32u + lane] = float4(rbf(y4.x), rbf(y4.y), rbf(y4.z), rbf(y4.w));
    }
}

struct Route { uint experts; uint top; float scale; };

kernel void route(device const float* logits [[buffer(0)]],
                  device const float* bias [[buffer(1)]],
                  device uint* ids [[buffer(2)]],
                  device float* gammas [[buffer(3)]],
                  constant Route& p [[buffer(4)]],
                  uint lane [[thread_index_in_simdgroup]]) {
    float val[8];
    uint idx[8];
    uint count = (p.experts + 31u) / 32u;
    for (uint j = 0; j < count; ++j) {
        uint e = lane + j * 32u;
        idx[j] = e;
        val[j] = e < p.experts ? (sigm(logits[e]) + bias[e]) : -INFINITY;
    }
    uint chosen[16];
    for (uint k = 0; k < p.top; ++k) {
        float bv = -INFINITY;
        uint bi = 0xffffffffu;
        uint bj = 0;
        for (uint j = 0; j < count; ++j) {
            if (val[j] > bv || (val[j] == bv && idx[j] < bi)) { bv = val[j]; bi = idx[j]; bj = j; }
        }
        float gv = bv;
        uint gi = bi;
        for (uint off = 16; off > 0; off /= 2) {
            float ov = simd_shuffle_xor(gv, off);
            uint oi = simd_shuffle_xor(gi, off);
            if (ov > gv || (ov == gv && oi < gi)) { gv = ov; gi = oi; }
        }
        if (bi == gi) val[bj] = -INFINITY;
        chosen[k] = gi;
    }
    if (lane == 0) {
        float lp[16], m = -INFINITY;
        for (uint k = 0; k < p.top; ++k) {
            float l = logits[chosen[k]];
            lp[k] = min(l, 0.0f) - precise::log(1.0f + precise::exp(-fabs(l)));
            m = max(m, lp[k]);
        }
        float s[16];
        for (uint k = 0; k < p.top; ++k) s[k] = precise::exp(lp[k] - m);
        float tot = 0.0f;
        if (p.top == 8u) tot = ((s[0] + s[2]) + (s[1] + s[3])) + ((s[4] + s[6]) + (s[5] + s[7]));
        else for (uint k = 0; k < p.top; ++k) tot += s[k];
        for (uint k = 0; k < p.top; ++k) {
            ids[k] = chosen[k];
            gammas[k] = precise::exp(lp[k] - (m + precise::log(tot))) * p.scale;
        }
    }
}

inline float4 fp4_rne_val4(float4 v) {
    float4 av = fabs(v);
    uint4 code = uint4(av > 0.25f) + uint4(av >= 0.75f) + uint4(av > 1.25f) + uint4(av >= 1.75f)
               + uint4(av > 2.5f) + uint4(av >= 3.5f) + uint4(av > 5.0f);
    float4 m = float4(FP4_VALUES[code.x], FP4_VALUES[code.y], FP4_VALUES[code.z], FP4_VALUES[code.w]);
    return copysign(m, v);
}

inline void quant_block16_v4(float4 b0, float4 b1, float4 b2, float4 b3, float enc, float dec, float out_scale, device half4* dst) {
    float4 am = max(max(fabs(b0), fabs(b1)), max(fabs(b2), fabs(b3)));
    float bmax = max(max(am.x, am.y), max(am.z, am.w));
    float sc0 = e4m3_rne(min(bmax * (enc * (1.0f / 6.0f)), 448.0f));
    float inv0 = min(1.0f / (sc0 * dec), FLT_MAX);
    float4 q00 = fp4_rne_val4(b0 * inv0) * sc0;
    float4 q01 = fp4_rne_val4(b1 * inv0) * sc0;
    float4 q02 = fp4_rne_val4(b2 * inv0) * sc0;
    float4 q03 = fp4_rne_val4(b3 * inv0) * sc0;
    float4 e00 = fabs(q00 * out_scale - b0), e01 = fabs(q01 * out_scale - b1);
    float4 e02 = fabs(q02 * out_scale - b2), e03 = fabs(q03 * out_scale - b3);
    float err0 = (((e00.x + e00.y) + (e00.z + e00.w)) + ((e01.x + e01.y) + (e01.z + e01.w)))
               + (((e02.x + e02.y) + (e02.z + e02.w)) + ((e03.x + e03.y) + (e03.z + e03.w)));
    float sc1 = e4m3_rne(min(bmax * (enc * 0.25f), 448.0f));
    float inv1 = min(1.0f / (sc1 * dec), FLT_MAX);
    float4 q10 = fp4_rne_val4(b0 * inv1) * sc1;
    float4 q11 = fp4_rne_val4(b1 * inv1) * sc1;
    float4 q12 = fp4_rne_val4(b2 * inv1) * sc1;
    float4 q13 = fp4_rne_val4(b3 * inv1) * sc1;
    float4 e10 = fabs(q10 * out_scale - b0), e11 = fabs(q11 * out_scale - b1);
    float4 e12 = fabs(q12 * out_scale - b2), e13 = fabs(q13 * out_scale - b3);
    float err1 = (((e10.x + e10.y) + (e10.z + e10.w)) + ((e11.x + e11.y) + (e11.z + e11.w)))
               + (((e12.x + e12.y) + (e12.z + e12.w)) + ((e13.x + e13.y) + (e13.z + e13.w)));
    if (err1 < err0) {
        dst[0] = half4(q10); dst[1] = half4(q11); dst[2] = half4(q12); dst[3] = half4(q13);
    } else {
        dst[0] = half4(q00); dst[1] = half4(q01); dst[2] = half4(q02); dst[3] = half4(q03);
    }
}

kernel void quant_nvfp4(device const float* x [[buffer(0)]],
                        device half* operand [[buffer(1)]],
                        device float* outer [[buffer(2)]],
                        constant uint& n [[buffer(3)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    device const float4* xr4 = (device const float4*)(x + row * n);
    device half4* opr4 = (device half4*)(operand + row * n);
    uint n4 = n / 4u;
    float a = 0.0f;
    for (uint i = tid; i < n4; i += threads) {
        float4 v = fabs(xr4[i]);
        a = max(a, max(max(v.x, v.y), max(v.z, v.w)));
    }
    a = simd_max(a);
    if (lane == 0) part[simd] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) amax = max(amax, part[i]);
    if (amax == 0.0f) {
        for (uint i = tid; i < n4; i += threads) opr4[i] = half4(0);
        if (tid == 0) outer[row] = 0.0f;
        return;
    }
    float enc = min(1536.0f / amax, FLT_MAX);
    float dec = 1.0f / enc;
    float out_scale = amax * (1.0f / 1536.0f);
    if (tid == 0) outer[row] = out_scale;
    for (uint b = tid; b < n / 16u; b += threads) {
        uint b4 = b * 4u;
        quant_block16_v4(xr4[b4], xr4[b4 + 1u], xr4[b4 + 2u], xr4[b4 + 3u], enc, dec, out_scale, opr4 + b4);
    }
}

struct Swiglu { uint inter; uint top; float gate_cap; float up_cap; uint round_out; };

kernel void swiglu_quant_nvfp4(device const float* gu [[buffer(0)]],
                               device const float* gammas [[buffer(1)]],
                               device half* hq [[buffer(2)]],
                               device float* houter [[buffer(3)]],
                               constant Swiglu& p [[buffer(4)]],
                               uint k [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint threads [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float hs[2048];
    threadgroup float part[8];
    float gamma = gammas[k];
    float inv_gc = 1.0f / p.gate_cap;
    float inv_uc = 1.0f / p.up_cap;
    float a = 0.0f;
    device const float2* row_gu = (device const float2*)(gu + k * 2u * p.inter);
    for (uint j = tid; j < p.inter; j += threads) {
        float2 guv = row_gu[j];
        float g = guv.x;
        float u = guv.y;
        float sg = sigm(g);
        float gc = p.gate_cap * precise::tanh(g * inv_gc);
        float uc = p.up_cap * precise::tanh(u * inv_uc);
        float v = rbf(gc * sg * uc * gamma);
        hs[j] = v;
        a = max(a, fabs(v));
    }
    a = simd_max(a);
    if (lane == 0) part[simd] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) amax = max(amax, part[i]);
    device half4* opr4 = (device half4*)(hq + k * p.inter);
    if (amax == 0.0f) {
        for (uint i = tid; i < p.inter / 4u; i += threads) opr4[i] = half4(0);
        if (tid == 0) houter[k] = 0.0f;
        return;
    }
    float enc = min(1536.0f / amax, FLT_MAX);
    float dec = 1.0f / enc;
    float out_scale = amax * (1.0f / 1536.0f);
    if (tid == 0) houter[k] = out_scale;
    threadgroup const float4* hs4 = (threadgroup const float4*)hs;
    for (uint b = tid; b < p.inter / 16u; b += threads) {
        uint b4 = b * 4u;
        quant_block16_v4(hs4[b4], hs4[b4 + 1u], hs4[b4 + 2u], hs4[b4 + 3u], enc, dec, out_scale, opr4 + b4);
    }
}

kernel void swiglu_quant_mxfp8(device const float* gu [[buffer(0)]],
                               device const float* gammas [[buffer(1)]],
                               device float* operand [[buffer(2)]],
                               constant Swiglu& p [[buffer(3)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint b = gid.x, k = gid.y;
    if (b >= p.inter / 32u) return;
    float gamma = gammas[k];
    float inv_gc = 1.0f / p.gate_cap;
    float inv_uc = 1.0f / p.up_cap;
    device const float2* row_gu = (device const float2*)(gu + k * 2u * p.inter) + b * 32u;
    float vals[32];
    float bmax = 0.0f;
    for (uint j = 0; j < 32u; ++j) {
        float2 guv = row_gu[j];
        float g = rbf(guv.x);
        float u = rbf(guv.y);
        float sg = sigm(g);
        float gc = p.gate_cap * precise::tanh(g * inv_gc);
        float uc = p.up_cap * precise::tanh(u * inv_uc);
        float v = gc * sg * uc * gamma;
        vals[j] = v;
        bmax = max(bmax, fabs(v));
    }
    float scale = bmax > 0.0f ? exp2(clamp(floor(log2(bmax)) - 8.0f, -127.0f, 127.0f)) : exp2(-127.0f);
    float inv_s = 1.0f / scale;
    device float* dst = operand + k * p.inter + b * 32u;
    for (uint j = 0; j < 32u; ++j) {
        float v = vals[j] * inv_s;
        dst[j] = (v < 0.0f ? -1.0f : 1.0f) * e4m3_rne(fabs(v)) * scale;
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

inline half fp4_h(uint code) {
    uint mag = code & 7u;
    half v = mag < 2u ? half(mag) * half(0.5h) : as_type<half>(ushort((mag << 9) + 0x3800u));
    return (code & 8u) ? -v : v;
}

inline half4 fp4x4_h(uint bits) {
    return half4(fp4_h(bits & 15u), fp4_h((bits >> 4) & 15u), fp4_h((bits >> 8) & 15u), fp4_h((bits >> 12) & 15u));
}

inline float simd16_sum(float v) {
    v += simd_shuffle_xor(v, 8u);
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

inline float simd8_sum(float v) {
    v += simd_shuffle_xor(v, 4u);
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
}

inline float simd4_sum(float v) {
    v += simd_shuffle_xor(v, 2u);
    v += simd_shuffle_xor(v, 1u);
    return v;
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

kernel void expert_nvfp4_16(device const uchar* w [[buffer(0)]],
                            device const uchar* sc [[buffer(1)]],
                            device const float* amax [[buffer(2)]],
                            device const float* denom [[buffer(3)]],
                            device const half* x [[buffer(4)]],
                            device const float* outer [[buffer(5)]],
                            device const uint* ids [[buffer(6)]],
                            device float* y [[buffer(7)]],
                            constant Expert& p [[buffer(8)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint simds [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane >> 4;
    uint sl = lane & 15u;
    uint row = (group.x * simds + simd) * 2u + sub;
    uint k = group.y;
    if (row >= p.rows) return;
    uint e = ids[k];
    device const uint2* wr = (device const uint2*)(w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u));
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const half4* x4 = (device const half4*)(x + (p.per_row_input ? k * p.in_stride : 0u));
    float acc = 0.0f;
    uint blocks = p.cols / 16u;
    for (uint b = sl; b < blocks; b += 16u) {
        uint2 pk = wr[b];
        float s = e4m3_decode(sr[b]);
        uint b4 = b * 4u;
        float d = 0.0f;
        { float4 q = float4(fp4x4_h(pk.x) * x4[b4]);            d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.x >> 16) * x4[b4 + 1u]); d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.y) * x4[b4 + 2u]);       d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.y >> 16) * x4[b4 + 3u]); d += (q.x + q.y) + (q.z + q.w); }
        acc += d * s;
    }
    float total = simd16_sum(acc);
    if (sl == 0u) {
        float o = outer[p.per_row_input ? k : 0u];
        float r = total * o * (amax[e] / denom[0]);
        y[k * p.out_stride + row] = p.per_row_input ? rbf(r) : r;
    }
}

kernel void expert_nvfp4_8(device const uchar* w [[buffer(0)]],
                           device const uchar* sc [[buffer(1)]],
                           device const float* amax [[buffer(2)]],
                           device const float* denom [[buffer(3)]],
                           device const half* x [[buffer(4)]],
                           device const float* outer [[buffer(5)]],
                           device const uint* ids [[buffer(6)]],
                           device float* y [[buffer(7)]],
                           constant Expert& p [[buffer(8)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane >> 3;
    uint sl = lane & 7u;
    uint row = (group.x * simds + simd) * 4u + sub;
    uint k = group.y;
    if (row >= p.rows) return;
    uint e = ids[k];
    device const uint2* wr = (device const uint2*)(w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u));
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const half4* x4 = (device const half4*)(x + (p.per_row_input ? k * p.in_stride : 0u));
    float acc = 0.0f;
    uint blocks = p.cols / 16u;
    for (uint b = sl; b < blocks; b += 8u) {
        uint2 pk = wr[b];
        float s = e4m3_decode(sr[b]);
        uint b4 = b * 4u;
        float d = 0.0f;
        { float4 q = float4(fp4x4_h(pk.x) * x4[b4]);            d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.x >> 16) * x4[b4 + 1u]); d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.y) * x4[b4 + 2u]);       d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.y >> 16) * x4[b4 + 3u]); d += (q.x + q.y) + (q.z + q.w); }
        acc += d * s;
    }
    float total = simd8_sum(acc);
    if (sl == 0u) {
        float o = outer[p.per_row_input ? k : 0u];
        float r = total * o * (amax[e] / denom[0]);
        y[k * p.out_stride + row] = p.per_row_input ? rbf(r) : r;
    }
}

kernel void expert_nvfp4(device const uchar* w [[buffer(0)]],
                         device const uchar* sc [[buffer(1)]],
                         device const float* amax [[buffer(2)]],
                         device const float* denom [[buffer(3)]],
                         device const half* x [[buffer(4)]],
                         device const float* outer [[buffer(5)]],
                         device const uint* ids [[buffer(6)]],
                         device float* y [[buffer(7)]],
                         constant Expert& p [[buffer(8)]],
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
    device const half4* x4 = (device const half4*)(x + (p.per_row_input ? k * p.in_stride : 0u));
    float acc = 0.0f;
    uint blocks = p.cols / 16u;
    for (uint b = lane; b < blocks; b += 32u) {
        uint2 packed = *(device const uint2*)(wr + b * 8u);
        float s = e4m3_decode(sr[b]);
        uint b4 = b * 4u;
        float4 part = fp4x4(packed.x) * float4(x4[b4]) + fp4x4(packed.x >> 16) * float4(x4[b4 + 1u])
                    + fp4x4(packed.y) * float4(x4[b4 + 2u]) + fp4x4(packed.y >> 16) * float4(x4[b4 + 3u]);
        acc += ((part.x + part.y) + (part.z + part.w)) * s;
    }
    float total = simd_sum(acc);
    if (lane == 0) {
        float o = outer[p.per_row_input ? k : 0u];
        float r = total * o * (amax[e] / denom[0]);
        y[k * p.out_stride + row] = p.per_row_input ? rbf(r) : r;
    }
}

inline float4 e4m3x4(uint word) {
    return float4(e4m3_decode(uchar(word)), e4m3_decode(uchar(word >> 8)), e4m3_decode(uchar(word >> 16)), e4m3_decode(uchar(word >> 24)));
}

kernel void expert_mxfp8_8(device const uchar* w [[buffer(0)]],
                           device const uchar* sc [[buffer(1)]],
                           device const float* x [[buffer(4)]],
                           device const uint* ids [[buffer(6)]],
                           device float* y [[buffer(7)]],
                           constant Expert& p [[buffer(8)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane >> 3;
    uint sl = lane & 7u;
    uint row = (group.x * simds + simd) * 4u + sub;
    uint k = group.y;
    if (row >= p.rows) return;
    uint e = ids[k];
    device const uint4* wr4 = (device const uint4*)(w + ulong(e) * p.expert_bytes + ulong(row) * p.cols);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 32u);
    device const float4* x4 = (device const float4*)(x + (p.per_row_input ? k * p.in_stride : 0u));
    float acc = 0.0f;
    uint blocks = p.cols / 32u;
    for (uint b = sl; b < blocks; b += 8u) {
        uint4 w0 = wr4[b * 2u];
        uint4 w1 = wr4[b * 2u + 1u];
        uint b8 = b * 8u;
        float d = dot(e4m3x4(w0.x), x4[b8]) + dot(e4m3x4(w0.y), x4[b8 + 1u])
                + dot(e4m3x4(w0.z), x4[b8 + 2u]) + dot(e4m3x4(w0.w), x4[b8 + 3u])
                + dot(e4m3x4(w1.x), x4[b8 + 4u]) + dot(e4m3x4(w1.y), x4[b8 + 5u])
                + dot(e4m3x4(w1.z), x4[b8 + 6u]) + dot(e4m3x4(w1.w), x4[b8 + 7u]);
        acc += d * as_type<float>(uint(sr[b]) << 23);
    }
    float total = simd8_sum(acc);
    if (sl == 0u) y[k * p.out_stride + row] = rbf(total);
}

kernel void expert_mxfp8_4(device const uchar* w [[buffer(0)]],
                           device const uchar* sc [[buffer(1)]],
                           device const float* x [[buffer(4)]],
                           device const uint* ids [[buffer(6)]],
                           device float* y [[buffer(7)]],
                           constant Expert& p [[buffer(8)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane >> 2;
    uint sl = lane & 3u;
    uint row = (group.x * simds + simd) * 8u + sub;
    uint k = group.y;
    if (row >= p.rows) return;
    uint e = ids[k];
    device const uint4* wr4 = (device const uint4*)(w + ulong(e) * p.expert_bytes + ulong(row) * p.cols);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 32u);
    device const float4* x4 = (device const float4*)(x + (p.per_row_input ? k * p.in_stride : 0u));
    float acc = 0.0f;
    uint blocks = p.cols / 32u;
    for (uint b = sl; b < blocks; b += 4u) {
        uint4 w0 = wr4[b * 2u];
        uint4 w1 = wr4[b * 2u + 1u];
        uint b8 = b * 8u;
        float d = dot(e4m3x4(w0.x), x4[b8]) + dot(e4m3x4(w0.y), x4[b8 + 1u])
                + dot(e4m3x4(w0.z), x4[b8 + 2u]) + dot(e4m3x4(w0.w), x4[b8 + 3u])
                + dot(e4m3x4(w1.x), x4[b8 + 4u]) + dot(e4m3x4(w1.y), x4[b8 + 5u])
                + dot(e4m3x4(w1.z), x4[b8 + 6u]) + dot(e4m3x4(w1.w), x4[b8 + 7u]);
        acc += d * as_type<float>(uint(sr[b]) << 23);
    }
    float total = simd4_sum(acc);
    if (sl == 0u) y[k * p.out_stride + row] = rbf(total);
}

kernel void expert_mxfp8(device const uchar* w [[buffer(0)]],
                         device const uchar* sc [[buffer(1)]],
                         device const float* x [[buffer(4)]],
                         device const uint* ids [[buffer(6)]],
                         device float* y [[buffer(7)]],
                         constant Expert& p [[buffer(8)]],
                         uint2 group [[threadgroup_position_in_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint simds [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * simds + simd;
    uint k = group.y;
    if (row >= p.rows) return;
    uint e = ids[k];
    device const uint4* wr4 = (device const uint4*)(w + ulong(e) * p.expert_bytes + ulong(row) * p.cols);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 32u);
    device const float4* x4 = (device const float4*)(x + (p.per_row_input ? k * p.in_stride : 0u));
    float acc = 0.0f;
    uint blocks = p.cols / 32u;
    for (uint b = lane; b < blocks; b += 32u) {
        uint4 w0 = wr4[b * 2u];
        uint4 w1 = wr4[b * 2u + 1u];
        float4 p0 = e4m3x4(w0.x) * x4[b * 8u] + e4m3x4(w0.y) * x4[b * 8u + 1u]
                  + e4m3x4(w0.z) * x4[b * 8u + 2u] + e4m3x4(w0.w) * x4[b * 8u + 3u];
        float4 p1 = e4m3x4(w1.x) * x4[b * 8u + 4u] + e4m3x4(w1.y) * x4[b * 8u + 5u]
                  + e4m3x4(w1.z) * x4[b * 8u + 6u] + e4m3x4(w1.w) * x4[b * 8u + 7u];
        float4 pt = p0 + p1;
        acc += ((pt.x + pt.y) + (pt.z + pt.w)) * as_type<float>(uint(sr[b]) << 23);
    }
    float total = simd_sum(acc);
    if (lane == 0) y[k * p.out_stride + row] = rbf(total);
}

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
                    uint c [[thread_position_in_grid]],
                    uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint order[16];
    if (tid == 0) {
        for (uint k = 0; k < p.top; ++k) order[k] = k;
        for (uint a = 1; a < p.top; ++a) for (uint b = a; b > 0 && ids[order[b - 1]] > ids[order[b]]; --b) { uint t = order[b]; order[b] = order[b - 1]; order[b - 1] = t; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (c >= p.n) return;
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

kernel void ctl_normed(device const float4* d [[buffer(0)]],
                       device const float4* x [[buffer(1)]],
                       device float4* normed [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint threads [[threads_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2][20];
    uint n4 = n / 4u;
    float4 dv = tid < n4 ? d[tid] : float4(0.0f);
    float4 xv = tid < n4 ? x[tid] : float4(0.0f);
    float a = dot(dv, dv), b = dot(xv, xv);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 dt = d[i], xt = x[i];
        a += dot(dt, dt); b += dot(xt, xt);
    }
    a = simd_sum(a); b = simd_sum(b);
    if (lane == 0) { part[0][simd] = a; part[1][simd] = b; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ta = 0, tb = 0;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) { ta += part[0][i]; tb += part[1][i]; }
    float rd = 1.0f / precise::sqrt(ta / float(n) + 1e-6f);
    float rx = 1.0f / precise::sqrt(tb / float(n) + 1e-6f);
    if (tid < n4) {
        float4 od = dv * rd, ox = xv * rx;
        normed[tid] = float4(rbf(od.x), rbf(od.y), rbf(od.z), rbf(od.w));
        normed[n4 + tid] = float4(rbf(ox.x), rbf(ox.y), rbf(ox.z), rbf(ox.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 od = d[i] * rd, ox = x[i] * rx;
        normed[i] = float4(rbf(od.x), rbf(od.y), rbf(od.z), rbf(od.w));
        normed[n4 + i] = float4(rbf(ox.x), rbf(ox.y), rbf(ox.z), rbf(ox.w));
    }
}

kernel void ctl_mix_norm(device const float4* d [[buffer(0)]],
                         device const float4* x [[buffer(1)]],
                         device const float4* gain [[buffer(2)]],
                         device const float4* forget [[buffer(3)]],
                         device const ushort* nw [[buffer(4)]],
                         device float4* out [[buffer(5)]],
                         constant uint& n [[buffer(6)]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = n / 4u;
    float4 v = 0.0f;
    if (tid < n4) {
        float4 fg = forget[tid], gn = gain[tid];
        float4 tf = float4(precise::tanh(fg.x), precise::tanh(fg.y), precise::tanh(fg.z), precise::tanh(fg.w));
        float4 tg = float4(precise::tanh(gn.x), precise::tanh(gn.y), precise::tanh(gn.z), precise::tanh(gn.w));
        v = (1.0f - tf) * d[tid] + tg * x[tid];
    }
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 fg = forget[i], gn = gain[i];
        float4 tf = float4(precise::tanh(fg.x), precise::tanh(fg.y), precise::tanh(fg.z), precise::tanh(fg.w));
        float4 tg = float4(precise::tanh(gn.x), precise::tanh(gn.y), precise::tanh(gn.z), precise::tanh(gn.w));
        float4 vt = (1.0f - tf) * d[i] + tg * x[i];
        s += dot(vt, vt);
    }
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) total += part[i];
    float r = 1.0f / precise::sqrt(total / float(n) + 1e-6f);
    if (tid < n4) {
        uint b = tid * 4u;
        float4 o = v * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        out[tid] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 fg = forget[i], gn = gain[i];
        float4 tf = float4(precise::tanh(fg.x), precise::tanh(fg.y), precise::tanh(fg.z), precise::tanh(fg.w));
        float4 tg = float4(precise::tanh(gn.x), precise::tanh(gn.y), precise::tanh(gn.z), precise::tanh(gn.w));
        float4 vt = (1.0f - tf) * d[i] + tg * x[i];
        uint b = i * 4u;
        float4 o = vt * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        out[i] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
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
        best = bv[0]; idx = bi[0];
        for (uint i = 1; i < (threads + 31u) / 32u; ++i) if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
        token[0] = idx;
    }
}

struct CertHead { uint vocab; uint cols; uint ng; float cap; };

inline float4 u4_lo(uint w) {
    return float4(float(w & 15u), float((w >> 4) & 15u), float((w >> 8) & 15u), float((w >> 12) & 15u)) - 8.0f;
}

inline float4 u4_hi(uint w) {
    return float4(float((w >> 16) & 15u), float((w >> 20) & 15u), float((w >> 24) & 15u), float((w >> 28) & 15u)) - 8.0f;
}

inline half4 u4_lo_h_fast(uint w) {
    ushort4 u = ushort4(w & 15u, (w >> 4) & 15u, (w >> 8) & 15u, (w >> 12) & 15u) | ushort4(0x6400u);
    return as_type<half4>(u) - half4(1032.0h);
}

inline half4 u4_hi_h_fast(uint w) {
    ushort4 u = ushort4((w >> 16) & 15u, (w >> 20) & 15u, (w >> 24) & 15u, (w >> 28) & 15u) | ushort4(0x6400u);
    return as_type<half4>(u) - half4(1032.0h);
}

kernel void build_cert_head(device const uint4* w [[buffer(0)]],
                            device uint4* q4 [[buffer(1)]],
                            device half* sc [[buffer(2)]],
                            device float* err [[buffer(3)]],
                            constant CertHead& p [[buffer(4)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint simds [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= p.vocab) return;
    uint chunks = p.cols / 8u;
    device const uint4* wr = w + ulong(row) * chunks;
    float err2 = 0.0f;
    float w2 = 0.0f;
    for (uint g = lane; g < p.ng; g += 32u) {
        float vals[32];
        float amax = 0.0f;
        for (uint u = 0; u < 4u; ++u) {
            uint4 pk = wr[g * 4u + u];
            uint b = u * 8u;
            vals[b + 0u] = as_type<float>(pk.x << 16);
            vals[b + 1u] = as_type<float>(pk.x & 0xffff0000u);
            vals[b + 2u] = as_type<float>(pk.y << 16);
            vals[b + 3u] = as_type<float>(pk.y & 0xffff0000u);
            vals[b + 4u] = as_type<float>(pk.z << 16);
            vals[b + 5u] = as_type<float>(pk.z & 0xffff0000u);
            vals[b + 6u] = as_type<float>(pk.w << 16);
            vals[b + 7u] = as_type<float>(pk.w & 0xffff0000u);
            for (uint j = 0; j < 8u; ++j) amax = max(amax, fabs(vals[b + j]));
        }
        half sh = half(amax / 7.0f);
        float s = float(sh);
        float inv = s > 0.0f ? (1.0f / s) : 0.0f;
        uint packed[4] = {0u, 0u, 0u, 0u};
        for (uint u = 0; u < 4u; ++u) {
            uint word = 0u;
            for (uint j = 0; j < 8u; ++j) {
                float v = vals[u * 8u + j];
                int q = s > 0.0f ? clamp(int(rint(v * inv)), -7, 7) : 0;
                float qv = float(q) * s;
                float diff = v - qv;
                err2 += diff * diff;
                w2 += qv * qv;
                word |= uint(q + 8) << (j * 4u);
            }
            packed[u] = word;
        }
        q4[ulong(row) * p.ng + g] = uint4(packed[0], packed[1], packed[2], packed[3]);
        sc[ulong(row) * p.ng + g] = sh;
    }
    float total = simd_sum(err2);
    float total_w2 = simd_sum(w2);
    if (lane == 0) err[row] = precise::sqrt(total) * 1.002f + precise::sqrt(total_w2) * 0.0025f + 1e-4f;
}

kernel void cert_head_xnorm(device const float4* x [[buffer(0)]],
                            device float* state [[buffer(1)]],
                            device uint* count [[buffer(2)]],
                            constant uint& n [[buffer(3)]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = n / 4u;
    float s = tid < n4 ? dot(x[tid], x[tid]) : 0.0f;
    for (uint i = tid + threads; i < n4; i += threads) s += dot(x[i], x[i]);
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (uint i = 0; i < (threads + 31u) / 32u; ++i) tot += part[i];
        state[2] = precise::sqrt(tot);
        count[0] = 0u;
    }
}

kernel void cert_head_scan(device const uint4* q4 [[buffer(0)]],
                           device const half* sc [[buffer(1)]],
                           device const float* err [[buffer(2)]],
                           device const float4* x4 [[buffer(3)]],
                           device const float* state [[buffer(4)]],
                           device float* ub_out [[buffer(5)]],
                           device float* tg_val [[buffer(6)]],
                           device uint* tg_row [[buffer(7)]],
                           constant CertHead& p [[buffer(8)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float svals[8];
    threadgroup uint srows[8];
    uint row = group * simds + simd;
    if (row < p.vocab) {
        device const uint4* qr = q4 + ulong(row) * p.ng;
        device const half* sr = sc + ulong(row) * p.ng;
        float acc = 0.0f;
        for (uint g = lane; g < p.ng; g += 32u) {
            uint4 pk = qr[g];
            float s = float(sr[g]);
            device const float4* xg = x4 + g * 8u;
            float4 d = u4_lo(pk.x) * xg[0] + u4_hi(pk.x) * xg[1]
                     + u4_lo(pk.y) * xg[2] + u4_hi(pk.y) * xg[3]
                     + u4_lo(pk.z) * xg[4] + u4_hi(pk.z) * xg[5]
                     + u4_lo(pk.w) * xg[6] + u4_hi(pk.w) * xg[7];
            acc += ((d.x + d.y) + (d.z + d.w)) * s;
        }
        float shat = simd_sum(acc);
        if (lane == 0) {
            ub_out[row] = shat + state[2] * err[row];
            svals[simd] = shat;
            srows[simd] = row;
        }
    } else if (lane == 0) {
        svals[simd] = -INFINITY;
        srows[simd] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
        float bv = svals[0];
        uint br = srows[0];
        for (uint i = 1; i < simds; ++i) {
            if (svals[i] > bv || (svals[i] == bv && srows[i] < br)) { bv = svals[i]; br = srows[i]; }
        }
        tg_val[group] = bv;
        tg_row[group] = br;
    }
}

struct CertTau { uint ntg; uint cols; float cap; };

kernel void cert_head_tau(device const float* tg_val [[buffer(0)]],
                          device const uint* tg_row [[buffer(1)]],
                          device const uint4* w [[buffer(2)]],
                          device const float4* x4 [[buffer(3)]],
                          device float* state [[buffer(4)]],
                          constant CertTau& p [[buffer(5)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float exact_s[16];
    float bv = -INFINITY;
    uint br = 0xffffffffu;
    for (uint i = tid; i < p.ntg; i += 512u) {
        float v = tg_val[i];
        uint r = tg_row[i];
        if (v > bv || (v == bv && r < br)) { bv = v; br = r; }
    }
    for (uint off = 16; off > 0; off /= 2) {
        float ov = simd_shuffle_xor(bv, off);
        uint oi = simd_shuffle_xor(br, off);
        if (ov > bv || (ov == bv && oi < br)) { bv = ov; br = oi; }
    }
    float sv = -INFINITY;
    if (br != 0xffffffffu) {
        uint chunks = p.cols / 8u;
        device const uint4* wr = w + ulong(br) * chunks;
        float acc = 0.0f;
        for (uint c = lane; c < chunks; c += 32u) {
            uint4 pk = wr[c];
            float4 a = x4[c * 2u];
            float4 b = x4[c * 2u + 1u];
            acc += as_type<float>(pk.x << 16) * a.x + as_type<float>(pk.x & 0xffff0000u) * a.y
                 + as_type<float>(pk.y << 16) * a.z + as_type<float>(pk.y & 0xffff0000u) * a.w
                 + as_type<float>(pk.z << 16) * b.x + as_type<float>(pk.z & 0xffff0000u) * b.y
                 + as_type<float>(pk.w << 16) * b.z + as_type<float>(pk.w & 0xffff0000u) * b.w;
        }
        float z = rbf(simd_sum(acc));
        sv = rbf(p.cap * precise::tanh(z / p.cap));
    }
    if (lane == 0) exact_s[simd] = sv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tau_s = exact_s[0];
        for (uint i = 1; i < 16u; ++i) tau_s = max(tau_s, exact_s[i]);
        uint bits = as_type<uint>(tau_s);
        uint exp = (bits >> 23) & 0xffu;
        float ulp = exp > 7u ? as_type<float>((exp - 7u) << 23) : 1e-6f;
        float s_lo = tau_s - ulp;
        if (s_lo <= -p.cap) {
            state[0] = -INFINITY;
        } else {
            float u = clamp(s_lo / p.cap, -0.9999f, 0.9999f);
            float z_lo = p.cap * 0.5f * precise::log((1.0f + u) / (1.0f - u));
            state[0] = z_lo - fabs(z_lo) * 0.005f - 0.01f;
        }
        state[1] = tau_s;
    }
}

kernel void cert_head_filter(device const float* ub [[buffer(0)]],
                             device const float* state [[buffer(1)]],
                             device atomic_uint* count [[buffer(2)]],
                             device uint* cands [[buffer(3)]],
                             constant uint& vocab [[buffer(4)]],
                             uint gid [[thread_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    bool keep = gid < vocab && ub[gid] >= state[0];
    uint rank = simd_prefix_exclusive_sum(uint(keep));
    uint tot = simd_sum(uint(keep));
    uint base = 0u;
    if (lane == 0 && tot > 0u) base = atomic_fetch_add_explicit(count, tot, memory_order_relaxed);
    base = simd_broadcast_first(base);
    if (keep) cands[base + rank] = gid;
}

kernel void cert_head_setup(device const uint* count [[buffer(0)]],
                            device uint* indirect [[buffer(1)]]) {
    uint c = count[0];
    indirect[0] = max((c + 7u) / 8u, 1u);
    indirect[1] = 1u;
    indirect[2] = 1u;
}

kernel void cert_head_rescore(device const uint4* w [[buffer(0)]],
                              device const float4* x4 [[buffer(1)]],
                              device const uint* cands [[buffer(2)]],
                              device const uint* count [[buffer(3)]],
                              device float* out_val [[buffer(4)]],
                              device uint* out_row [[buffer(5)]],
                              constant CertTau& p [[buffer(6)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint simds [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float svals[8];
    threadgroup uint srows[8];
    uint idx = group * simds + simd;
    uint cnt = count[0];
    if (idx < cnt) {
        uint row = cands[idx];
        uint chunks = p.cols / 8u;
        device const uint4* wr = w + ulong(row) * chunks;
        float acc = 0.0f;
        for (uint c = lane; c < chunks; c += 32u) {
            uint4 pk = wr[c];
            float4 a = x4[c * 2u];
            float4 b = x4[c * 2u + 1u];
            acc += as_type<float>(pk.x << 16) * a.x + as_type<float>(pk.x & 0xffff0000u) * a.y
                 + as_type<float>(pk.y << 16) * a.z + as_type<float>(pk.y & 0xffff0000u) * a.w
                 + as_type<float>(pk.z << 16) * b.x + as_type<float>(pk.z & 0xffff0000u) * b.y
                 + as_type<float>(pk.w << 16) * b.z + as_type<float>(pk.w & 0xffff0000u) * b.w;
        }
        float z = rbf(simd_sum(acc));
        float sv = rbf(p.cap * precise::tanh(z / p.cap));
        if (lane == 0) { svals[simd] = sv; srows[simd] = row; }
    } else if (lane == 0) {
        svals[simd] = -INFINITY;
        srows[simd] = 0xffffffffu;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0 && lane == 0) {
        float bv = svals[0];
        uint br = srows[0];
        for (uint i = 1; i < simds; ++i) {
            if (svals[i] > bv || (svals[i] == bv && srows[i] < br)) { bv = svals[i]; br = srows[i]; }
        }
        out_val[group] = bv;
        out_row[group] = br;
    }
}

kernel void cert_head_argmax(device const float* in_val [[buffer(0)]],
                             device const uint* in_row [[buffer(1)]],
                             device const uint* indirect [[buffer(2)]],
                             device uint* token [[buffer(3)]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float bv[8];
    threadgroup uint bi[8];
    uint ngroups = indirect[0];
    float best = -INFINITY;
    uint idx = 0xffffffffu;
    for (uint i = tid; i < ngroups; i += threads) {
        float v = in_val[i];
        uint r = in_row[i];
        if (v > best || (v == best && r < idx)) { best = v; idx = r; }
    }
    for (uint off = 16; off > 0; off /= 2) {
        float ov = simd_shuffle_down(best, off);
        uint oi = simd_shuffle_down(idx, off);
        if (ov > best || (ov == best && oi < idx)) { best = ov; idx = oi; }
    }
    if (lane == 0) { bv[simd] = best; bi[simd] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        best = bv[0]; idx = bi[0];
        for (uint i = 1; i < (threads + 31u) / 32u; ++i) {
            if (bv[i] > best || (bv[i] == best && bi[i] < idx)) { best = bv[i]; idx = bi[i]; }
        }
        token[0] = idx;
    }
}

kernel void bf12_hist(device const uint4* w [[buffer(0)]],
                      device atomic_uint* hist [[buffer(1)]],
                      constant uint& count8 [[buffer(2)]],
                      uint gid [[thread_position_in_grid]],
                      uint gsize [[threads_per_grid]],
                      uint tid [[thread_position_in_threadgroup]]) {
    threadgroup atomic_uint local[256];
    if (tid < 256u) atomic_store_explicit(&local[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = gid; i < count8; i += gsize) {
        uint4 pk = w[i];
        uint ws[4] = {pk.x, pk.y, pk.z, pk.w};
        for (uint j = 0; j < 4u; ++j) {
            uint lo = ws[j] & 0xffffu;
            uint hi = ws[j] >> 16;
            if ((lo & 0x7fffu) != 0u) atomic_fetch_add_explicit(&local[(lo >> 7) & 0xffu], 1u, memory_order_relaxed);
            if ((hi & 0x7fffu) != 0u) atomic_fetch_add_explicit(&local[(hi >> 7) & 0xffu], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 256u) {
        uint v = atomic_load_explicit(&local[tid], memory_order_relaxed);
        if (v) atomic_fetch_add_explicit(&hist[tid], v, memory_order_relaxed);
    }
}

struct Bf12Pack { uint src_rows; uint cols; uint row_base; uint out_cap; };

kernel void bf12_pack(device const uint4* w [[buffer(0)]],
                      device const uchar* exp_map [[buffer(1)]],
                      device uint4* sm_out [[buffer(2)]],
                      device uint2* ex_out [[buffer(3)]],
                      device atomic_uint* out_cnt [[buffer(4)]],
                      device uint2* out_raw [[buffer(5)]],
                      constant Bf12Pack& p [[buffer(6)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint simds [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint r = group * simds + simd;
    if (r >= p.src_rows) return;
    uint chunks16 = p.cols / 16u;
    device const uint4* wr = w + ulong(r) * (chunks16 * 2u);
    uint dst_row = p.row_base + r;
    for (uint c = lane; c < chunks16; c += 32u) {
        uint4 a = wr[c * 2u];
        uint4 b = wr[c * 2u + 1u];
        uint words[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        uint smw[4] = {0u, 0u, 0u, 0u};
        uint exw[2] = {0u, 0u};
        for (uint j = 0; j < 16u; ++j) {
            uint raw = (j & 1u) ? (words[j >> 1] >> 16) : (words[j >> 1] & 0xffffu);
            uint smb = 0u;
            uint idx = 15u;
            if ((raw & 0x7fffu) != 0u) {
                idx = uint(exp_map[(raw >> 7) & 0xffu]);
                if (idx == 15u) {
                    uint pos = atomic_fetch_add_explicit(out_cnt, 1u, memory_order_relaxed);
                    if (pos < p.out_cap) out_raw[pos] = uint2(dst_row, ((c * 16u + j) << 16) | raw);
                } else {
                    smb = ((raw >> 8) & 0x80u) | (raw & 0x7fu);
                }
            }
            smw[j >> 2] |= smb << ((j & 3u) * 8u);
            exw[j >> 3] |= idx << ((j & 7u) * 4u);
        }
        sm_out[ulong(dst_row) * chunks16 + c] = uint4(smw[0], smw[1], smw[2], smw[3]);
        ex_out[ulong(dst_row) * chunks16 + c] = uint2(exw[0], exw[1]);
    }
}

struct Gemv12 { uint rows; uint cols; uint round_out; uint has_outliers; uint elut[16]; };

inline float bf12_dot4(uint smw, uint ex16, float4 xv, constant uint* elut) {
    float w0 = as_type<float>(((smw << 24) & 0x80000000u) | ((smw << 16) & 0x007f0000u) | elut[ex16 & 15u]);
    float w1 = as_type<float>(((smw << 16) & 0x80000000u) | ((smw << 8) & 0x007f0000u) | elut[(ex16 >> 4) & 15u]);
    float w2 = as_type<float>(((smw << 8) & 0x80000000u) | (smw & 0x007f0000u) | elut[(ex16 >> 8) & 15u]);
    float w3 = as_type<float>((smw & 0x80000000u) | ((smw >> 8) & 0x007f0000u) | elut[(ex16 >> 12) & 15u]);
    return w0 * xv.x + w1 * xv.y + w2 * xv.z + w3 * xv.w;
}

kernel void gemv_bf12(device const uint4* sm_buf [[buffer(0)]],
                      device const uint2* ex_buf [[buffer(1)]],
                      device const uint* ro_buf [[buffer(2)]],
                      device const uint* op_buf [[buffer(3)]],
                      device const float4* x4 [[buffer(4)]],
                      device float* y [[buffer(5)]],
                      constant Gemv12& g [[buffer(6)]],
                      uint group [[threadgroup_position_in_grid]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint simds [[simdgroups_per_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= g.rows) return;
    uint chunks16 = g.cols / 16u;
    device const uint4* smr = sm_buf + ulong(row) * chunks16;
    device const uint2* exr = ex_buf + ulong(row) * chunks16;
    float acc = 0.0f;
    for (uint c = lane; c < chunks16; c += 32u) {
        uint4 sm = smr[c];
        uint2 ex = exr[c];
        acc += bf12_dot4(sm.x, ex.x & 0xffffu, x4[c * 4u], g.elut)
             + bf12_dot4(sm.y, ex.x >> 16, x4[c * 4u + 1u], g.elut)
             + bf12_dot4(sm.z, ex.y & 0xffffu, x4[c * 4u + 2u], g.elut)
             + bf12_dot4(sm.w, ex.y >> 16, x4[c * 4u + 3u], g.elut);
    }
    if (g.has_outliers) {
        uint s = ro_buf[row];
        uint e = ro_buf[row + 1u];
        device const float* xf = (device const float*)x4;
        for (uint j = s + lane; j < e; j += 32u) {
            uint pk = op_buf[j];
            acc += bfv(ushort(pk & 0xffffu)) * xf[pk >> 16];
        }
    }
    float total = simd_sum(acc);
    if (lane == 0) y[row] = g.round_out ? rbf(total) : total;
}
