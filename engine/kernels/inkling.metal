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
    float v = e == 0u ? float(m) * 0.001953125f : as_type<float>(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
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

struct ClipKv { uint width; uint pos; uint stride; uint mode; float threshold; };

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
    device ushort* dst = cache + ulong(p.pos) * p.stride + row * p.width;
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

struct Attn { uint pos; uint start; uint kv_heads; uint group; uint head_dim; uint extent; float scale; };

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
        ulong base = (ulong(kp) * p.kv_heads + kvh) * 32u + lane;
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
        float tot = ((s[0] + s[2]) + (s[1] + s[3])) + ((s[4] + s[6]) + (s[5] + s[7]));
        for (uint k = 0; k < p.top; ++k) {
            ids[k] = chosen[k];
            gammas[k] = precise::exp(lp[k] - (m + precise::log(tot))) * p.scale;
        }
    }
}

inline void quant_block16(const thread float* blk, float enc, float dec, float out_scale, device float* dst) {
    float bmax = 0;
    for (uint j = 0; j < 16; ++j) bmax = max(bmax, fabs(blk[j]));
    float best_err[2];
    float vals[2][16];
    float ims[2] = {1.0f / 6.0f, 0.25f};
    for (uint c = 0; c < 2; ++c) {
        float sc = e4m3_rne(min(bmax * (enc * ims[c]), 448.0f));
        float inv = min(1.0f / (sc * dec), FLT_MAX);
        float err[16];
        for (uint j = 0; j < 16; ++j) {
            float r = fp4_rne_val(blk[j] * inv);
            float qv = r * sc;
            vals[c][j] = qv;
            err[j] = fabs(qv * out_scale - blk[j]);
        }
        for (uint w = 16; w > 1; w /= 2) for (uint j = 0; j < w / 2; ++j) err[j] = err[2 * j] + err[2 * j + 1];
        best_err[c] = err[0];
    }
    uint pick = best_err[1] < best_err[0] ? 1u : 0u;
    for (uint j = 0; j < 16; ++j) dst[j] = vals[pick][j];
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
    threadgroup float part[8];
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
        for (uint j = 0; j < 16; ++j) blk[j] = xr[b * 16u + j];
        quant_block16(blk, enc, dec, out_scale, opr + b * 16u);
    }
}

struct Swiglu { uint inter; uint top; float gate_cap; float up_cap; uint round_out; };

kernel void swiglu_quant_nvfp4(device const float* gu [[buffer(0)]],
                               device const float* gammas [[buffer(1)]],
                               device float* hq [[buffer(2)]],
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
    float a = 0.0f;
    device const float* row_gu = gu + k * 2u * p.inter;
    for (uint j = tid; j < p.inter; j += threads) {
        float g = row_gu[2u * j];
        float u = row_gu[2u * j + 1u];
        float sg = sigm(g);
        float gc = p.gate_cap * precise::tanh(g / p.gate_cap);
        float uc = p.up_cap * precise::tanh(u / p.up_cap);
        float v = rbf(gc * sg * uc * gamma);
        hs[j] = v;
        a = max(a, fabs(v));
    }
    a = simd_max(a);
    if (lane == 0) part[simd] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float amax = 0;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) amax = max(amax, part[i]);
    device float* opr = hq + k * p.inter;
    if (amax == 0.0f) {
        for (uint i = tid; i < p.inter; i += threads) opr[i] = 0;
        if (tid == 0) houter[k] = 0;
        return;
    }
    float enc = min(1536.0f / amax, FLT_MAX);
    float dec = 1.0f / enc;
    float out_scale = amax * (1.0f / 1536.0f);
    if (tid == 0) houter[k] = out_scale;
    for (uint b = tid; b < p.inter / 16u; b += threads) {
        float blk[16];
        for (uint j = 0; j < 16; ++j) blk[j] = hs[b * 16u + j];
        quant_block16(blk, enc, dec, out_scale, opr + b * 16u);
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
    device const float4* x4 = (device const float4*)(x + (p.per_row_input ? k * p.in_stride : 0u));
    float acc = 0.0f;
    uint blocks = p.cols / 16u;
    for (uint b = lane; b < blocks; b += 32u) {
        uint2 packed = *(device const uint2*)(wr + b * 8u);
        float s = e4m3_decode(sr[b]);
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

inline float4 e4m3x4(uint word) {
    return float4(e4m3_decode(uchar(word)), e4m3_decode(uchar(word >> 8)), e4m3_decode(uchar(word >> 16)), e4m3_decode(uchar(word >> 24)));
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
