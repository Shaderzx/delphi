struct EmbedB { uint n; uint rows; };

kernel void embed_rows(device const ushort* table [[buffer(0)]],
                       device const uint* tokens [[buffer(1)]],
                       device float* x [[buffer(2)]],
                       constant EmbedB& p [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, m = gid.y;
    if (i < p.n && m < p.rows) x[m * p.n + i] = bfv(table[ulong(tokens[m]) * p.n + i]);
}

kernel void rms_norm_batch(device const float4* x [[buffer(0)]],
                           device const ushort* w [[buffer(1)]],
                           device float4* y [[buffer(2)]],
                           constant Norm& p [[buffer(3)]],
                           uint m [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[32];
    uint n4 = p.n / 4u;
    device const float4* xm = x + ulong(m) * n4;
    device float4* ym = y + ulong(m) * n4;
    float4 v = tid < n4 ? xm[tid] : float4(0.0f);
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 t = xm[i];
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
        ym[tid] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 o = xm[i] * r;
        if (p.has_weight) {
            uint b = i * 4u;
            o *= float4(bfv(w[b]), bfv(w[b + 1u]), bfv(w[b + 2u]), bfv(w[b + 3u]));
        }
        ym[i] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
}

kernel void attn_res_batch(device float* arena [[buffer(0)]],
                           device const float* q [[buffer(1)]],
                           device const ushort* weights [[buffer(2)]],
                           constant Res& p [[buffer(3)]],
                           uint m [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20][8][4];
    threadgroup float stats[8][4];
    uint ns = p.sources;
    uint n4 = p.n / 4u;
    uint m4 = m * n4;
    device float4* arena4 = (device float4*)arena;
    device const float4* q4 = (device const float4*)q;
    float ss[8], dotv[8][3];
    float4 cached[8];
    for (uint s = 0; s < ns; ++s) { ss[s] = 0; for (uint k = 0; k < 3; ++k) dotv[s][k] = 0; }
    if (tid < n4) {
        for (uint s = 0; s < ns; ++s) {
            float4 v = arena4[p.offs[s] / 4u + m4 + tid];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v += arena4[uint(p.addend) / 4u + m4 + tid];
            cached[s] = v;
            ss[s] = dot(v, v);
            for (uint k = 0; k < p.queries; ++k) dotv[s][k] = dot(v, q4[k * n4 + tid]);
        }
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        for (uint s = 0; s < ns; ++s) {
            float4 v = arena4[p.offs[s] / 4u + m4 + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v += arena4[uint(p.addend) / 4u + m4 + i];
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
        float mx = -INFINITY;
        for (uint s = 0; s < ns; ++s) mx = max(mx, stats[s][k + 1] * rs[s]);
        float tot = 0;
        for (uint s = 0; s < ns; ++s) { wt[k][s] = precise::exp(stats[s][k + 1] * rs[s] - mx); tot += wt[k][s]; }
        inv[k] = 1.0f / tot;
        if (p.normalize) for (uint s = 0; s < ns; ++s) wt[k][s] *= rs[s];
    }
    if (!p.has_norm) {
        if (tid < n4) {
            for (uint k = 0; k < p.queries; ++k) {
                float4 acc = 0;
                for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], cached[s], acc);
                arena4[p.outs[k] / 4u + m4 + tid] = acc * inv[k];
            }
        }
        for (uint i = tid + threads; i < n4; i += threads) {
            float4 v[8];
            for (uint s = 0; s < ns; ++s) {
                v[s] = arena4[p.offs[s] / 4u + m4 + i];
                if (int(s) == int(ns) - 1 && p.addend >= 0) v[s] += arena4[uint(p.addend) / 4u + m4 + i];
            }
            for (uint k = 0; k < p.queries; ++k) {
                float4 acc = 0;
                for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], v[s], acc);
                arena4[p.outs[k] / 4u + m4 + i] = acc * inv[k];
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
            v[s] = arena4[p.offs[s] / 4u + m4 + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v[s] += arena4[uint(p.addend) / 4u + m4 + i];
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
            arena4[p.outs[k] / 4u + m4 + tid] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
        }
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 v[8];
        for (uint s = 0; s < ns; ++s) {
            v[s] = arena4[p.offs[s] / 4u + m4 + i];
            if (int(s) == int(ns) - 1 && p.addend >= 0) v[s] += arena4[uint(p.addend) / 4u + m4 + i];
        }
        uint b = i * 4u;
        for (uint k = 0; k < p.queries; ++k) {
            float4 acc = 0;
            for (uint s = 0; s < ns; ++s) acc = fma(wt[k][s], v[s], acc);
            device const ushort* nw = weights + p.norm_offs[k] / 2u;
            float4 o = (acc * inv[k]) * nr[k] * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
            arena4[p.outs[k] / 4u + m4 + i] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
        }
    }
}

kernel void ctl_normed_batch(device const float4* d [[buffer(0)]],
                             device const float4* x [[buffer(1)]],
                             device float4* normed [[buffer(2)]],
                             constant uint& n [[buffer(3)]],
                             uint m [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[2][20];
    uint n4 = n / 4u;
    device const float4* dm = d + ulong(m) * n4;
    device const float4* xm = x + ulong(m) * n4;
    device float4* nm = normed + ulong(m) * (2u * n4);
    float4 dv = tid < n4 ? dm[tid] : float4(0.0f);
    float4 xv = tid < n4 ? xm[tid] : float4(0.0f);
    float a = dot(dv, dv), b = dot(xv, xv);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 dt = dm[i], xt = xm[i];
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
        nm[tid] = float4(rbf(od.x), rbf(od.y), rbf(od.z), rbf(od.w));
        nm[n4 + tid] = float4(rbf(ox.x), rbf(ox.y), rbf(ox.z), rbf(ox.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 od = dm[i] * rd, ox = xm[i] * rx;
        nm[i] = float4(rbf(od.x), rbf(od.y), rbf(od.z), rbf(od.w));
        nm[n4 + i] = float4(rbf(ox.x), rbf(ox.y), rbf(ox.z), rbf(ox.w));
    }
}

struct CtlMixB { uint n; uint gf_stride; };

kernel void ctl_mix_norm_batch(device const float4* d [[buffer(0)]],
                               device const float4* x [[buffer(1)]],
                               device const float4* gain [[buffer(2)]],
                               device const float4* forget [[buffer(3)]],
                               device const ushort* nw [[buffer(4)]],
                               device float4* out [[buffer(5)]],
                               constant CtlMixB& p [[buffer(6)]],
                               uint m [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint threads [[threads_per_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = p.n / 4u;
    uint gfs4 = p.gf_stride / 4u;
    device const float4* dm = d + ulong(m) * n4;
    device const float4* xm = x + ulong(m) * n4;
    device const float4* gm = gain + ulong(m) * gfs4;
    device const float4* fm = forget + ulong(m) * gfs4;
    device float4* om = out + ulong(m) * n4;
    float4 v = 0.0f;
    if (tid < n4) {
        float4 fg = fm[tid], gn = gm[tid];
        float4 tf = float4(precise::tanh(fg.x), precise::tanh(fg.y), precise::tanh(fg.z), precise::tanh(fg.w));
        float4 tg = float4(precise::tanh(gn.x), precise::tanh(gn.y), precise::tanh(gn.z), precise::tanh(gn.w));
        v = (1.0f - tf) * dm[tid] + tg * xm[tid];
    }
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 fg = fm[i], gn = gm[i];
        float4 tf = float4(precise::tanh(fg.x), precise::tanh(fg.y), precise::tanh(fg.z), precise::tanh(fg.w));
        float4 tg = float4(precise::tanh(gn.x), precise::tanh(gn.y), precise::tanh(gn.z), precise::tanh(gn.w));
        float4 vt = (1.0f - tf) * dm[i] + tg * xm[i];
        s += dot(vt, vt);
    }
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) total += part[i];
    float r = 1.0f / precise::sqrt(total / float(p.n) + 1e-6f);
    if (tid < n4) {
        uint b = tid * 4u;
        float4 o = v * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        om[tid] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 fg = fm[i], gn = gm[i];
        float4 tf = float4(precise::tanh(fg.x), precise::tanh(fg.y), precise::tanh(fg.z), precise::tanh(fg.w));
        float4 tg = float4(precise::tanh(gn.x), precise::tanh(gn.y), precise::tanh(gn.z), precise::tanh(gn.w));
        float4 vt = (1.0f - tf) * dm[i] + tg * xm[i];
        uint b = i * 4u;
        float4 o = vt * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        om[i] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
}

inline float4 bf12_dec4(uint smw, uint ex16, constant uint* elut) {
    return float4(
        as_type<float>(((smw << 24) & 0x80000000u) | ((smw << 16) & 0x007f0000u) | elut[ex16 & 15u]),
        as_type<float>(((smw << 16) & 0x80000000u) | ((smw << 8) & 0x007f0000u) | elut[(ex16 >> 4) & 15u]),
        as_type<float>(((smw << 8) & 0x80000000u) | (smw & 0x007f0000u) | elut[(ex16 >> 8) & 15u]),
        as_type<float>((smw & 0x80000000u) | ((smw >> 8) & 0x007f0000u) | elut[(ex16 >> 12) & 15u])
    );
}

struct Gemv12B { uint rows; uint cols; uint round_out; uint has_outliers; uint x_stride; uint y_stride; uint elut[16]; };

kernel void gemv_bf12_rows(device const uint4* sm_buf [[buffer(0)]],
                           device const uint2* ex_buf [[buffer(1)]],
                           device const uint* ro_buf [[buffer(2)]],
                           device const uint* op_buf [[buffer(3)]],
                           device const float4* x4 [[buffer(4)]],
                           device float* y [[buffer(5)]],
                           constant Gemv12B& g [[buffer(6)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= g.rows) return;
    uint chunks16 = g.cols / 16u;
    uint xs4 = g.x_stride / 4u;
    device const uint4* smr = sm_buf + ulong(row) * chunks16;
    device const uint2* exr = ex_buf + ulong(row) * chunks16;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint c = lane; c < chunks16; c += 32u) {
        uint4 sm = smr[c];
        uint2 ex = exr[c];
        float4 w0 = bf12_dec4(sm.x, ex.x & 0xffffu, g.elut);
        float4 w1 = bf12_dec4(sm.y, ex.x >> 16, g.elut);
        float4 w2 = bf12_dec4(sm.z, ex.y & 0xffffu, g.elut);
        float4 w3 = bf12_dec4(sm.w, ex.y >> 16, g.elut);
        uint c4 = c * 4u;
        if (ROWS > 0u) { device const float4* xm = x4 + 0ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a0 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
        if (ROWS > 1u) { device const float4* xm = x4 + 1ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a1 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
        if (ROWS > 2u) { device const float4* xm = x4 + 2ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a2 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
        if (ROWS > 3u) { device const float4* xm = x4 + 3ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a3 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
        if (ROWS > 4u) { device const float4* xm = x4 + 4ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a4 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
        if (ROWS > 5u) { device const float4* xm = x4 + 5ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a5 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
        if (ROWS > 6u) { device const float4* xm = x4 + 6ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a6 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
        if (ROWS > 7u) { device const float4* xm = x4 + 7ul * xs4 + c4; float4 a = xm[0], b = xm[1], cc = xm[2], d = xm[3]; a7 += (w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w) + (w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w) + (w2.x*cc.x + w2.y*cc.y + w2.z*cc.z + w2.w*cc.w) + (w3.x*d.x + w3.y*d.y + w3.z*d.z + w3.w*d.w); }
    }
    if (g.has_outliers) {
        uint s = ro_buf[row];
        uint e = ro_buf[row + 1u];
        device const float* xf = (device const float*)x4;
        for (uint j = s + lane; j < e; j += 32u) {
            uint pk = op_buf[j];
            float wv = bfv(ushort(pk & 0xffffu));
            uint col = pk >> 16;
            if (ROWS > 0u) a0 += wv * xf[0ul * g.x_stride + col];
            if (ROWS > 1u) a1 += wv * xf[1ul * g.x_stride + col];
            if (ROWS > 2u) a2 += wv * xf[2ul * g.x_stride + col];
            if (ROWS > 3u) a3 += wv * xf[3ul * g.x_stride + col];
            if (ROWS > 4u) a4 += wv * xf[4ul * g.x_stride + col];
            if (ROWS > 5u) a5 += wv * xf[5ul * g.x_stride + col];
            if (ROWS > 6u) a6 += wv * xf[6ul * g.x_stride + col];
            if (ROWS > 7u) a7 += wv * xf[7ul * g.x_stride + col];
        }
    }
    if (ROWS > 0u) { float t = simd_sum(a0); if (lane == 0) y[0ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd_sum(a1); if (lane == 0) y[1ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd_sum(a2); if (lane == 0) y[2ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd_sum(a3); if (lane == 0) y[3ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd_sum(a4); if (lane == 0) y[4ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd_sum(a5); if (lane == 0) y[5ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd_sum(a6); if (lane == 0) y[6ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd_sum(a7); if (lane == 0) y[7ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
}

struct GemvB { uint rows; uint cols; uint round_out; uint x_stride; uint y_stride; float scale; };

kernel void gemv_bf16_rows(device const uint4* w [[buffer(0)]],
                           device const float4* x4 [[buffer(1)]],
                           device float* y [[buffer(2)]],
                           constant GemvB& g [[buffer(3)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint simds [[simdgroups_per_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= g.rows) return;
    uint chunks = g.cols / 8u;
    uint xs4 = g.x_stride / 4u;
    device const uint4* wr = w + ulong(row) * chunks;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint c = lane; c < chunks; c += 32u) {
        uint4 p = wr[c];
        float4 w0 = float4(as_type<float>(p.x << 16), as_type<float>(p.x & 0xffff0000u),
                           as_type<float>(p.y << 16), as_type<float>(p.y & 0xffff0000u));
        float4 w1 = float4(as_type<float>(p.z << 16), as_type<float>(p.z & 0xffff0000u),
                           as_type<float>(p.w << 16), as_type<float>(p.w & 0xffff0000u));
        uint c2 = c * 2u;
        if (ROWS > 0u) { device const float4* xm = x4 + 0ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a0 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
        if (ROWS > 1u) { device const float4* xm = x4 + 1ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a1 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
        if (ROWS > 2u) { device const float4* xm = x4 + 2ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a2 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
        if (ROWS > 3u) { device const float4* xm = x4 + 3ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a3 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
        if (ROWS > 4u) { device const float4* xm = x4 + 4ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a4 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
        if (ROWS > 5u) { device const float4* xm = x4 + 5ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a5 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
        if (ROWS > 6u) { device const float4* xm = x4 + 6ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a6 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
        if (ROWS > 7u) { device const float4* xm = x4 + 7ul * xs4 + c2; float4 a = xm[0], b = xm[1]; a7 += w0.x*a.x + w0.y*a.y + w0.z*a.z + w0.w*a.w + w1.x*b.x + w1.y*b.y + w1.z*b.z + w1.w*b.w; }
    }
    if (ROWS > 0u) { float t = simd_sum(a0) * g.scale; if (lane == 0) y[0ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd_sum(a1) * g.scale; if (lane == 0) y[1ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd_sum(a2) * g.scale; if (lane == 0) y[2ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd_sum(a3) * g.scale; if (lane == 0) y[3ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd_sum(a4) * g.scale; if (lane == 0) y[4ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd_sum(a5) * g.scale; if (lane == 0) y[5ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd_sum(a6) * g.scale; if (lane == 0) y[6ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd_sum(a7) * g.scale; if (lane == 0) y[7ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
}

struct ConvB { uint channels; uint dynamic; uint head_dim; uint rows; uint x_stride; uint dyn_stride; uint y_stride; float scale; };

kernel void sconv_batch(device const float* x [[buffer(0)]],
                        device float* state [[buffer(1)]],
                        device const ushort* w [[buffer(2)]],
                        device const float* dyn [[buffer(3)]],
                        device float* y [[buffer(4)]],
                        constant ConvB& p [[buffer(5)]],
                        uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) return;
    float h0 = state[c];
    float h1 = state[p.channels + c];
    float h2 = state[2u * p.channels + c];
    uint head = p.dynamic ? (c / p.head_dim) : 0u;
    float w0 = p.dynamic ? bfv(w[0u * p.channels + c]) : bfv(w[c * 4u + 0u]);
    float w1 = p.dynamic ? bfv(w[1u * p.channels + c]) : bfv(w[c * 4u + 1u]);
    float w2 = p.dynamic ? bfv(w[2u * p.channels + c]) : bfv(w[c * 4u + 2u]);
    float w3 = p.dynamic ? bfv(w[3u * p.channels + c]) : bfv(w[c * 4u + 3u]);
    for (uint m = 0; m < p.rows; ++m) {
        float xm = x[ulong(m) * p.x_stride + c];
        state[(3u + m) * p.channels + c] = xm;
        float acc = 0.0f;
        if (p.dynamic) {
            device const float* dm = dyn + ulong(m) * p.dyn_stride + head * 4u;
            acc += xm * (w0 + dm[0]);
            acc += h2 * (w1 + dm[1]);
            acc += h1 * (w2 + dm[2]);
            acc += h0 * (w3 + dm[3]);
        } else {
            acc = fma(h0, w0, acc);
            acc = fma(h1, w1, acc);
            acc = fma(h2, w2, acc);
            acc = fma(xm, w3, acc);
        }
        y[ulong(m) * p.y_stride + c] = rbf((acc + xm) * p.scale);
        h0 = h1;
        h1 = h2;
        h2 = xm;
    }
}

struct ConvMeta { uint offset_f; uint width; };

kernel void conv_commit_all(device float* conv_buf [[buffer(0)]],
                            device const ConvMeta* meta [[buffer(1)]],
                            constant uint& commit [[buffer(2)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]]) {
    ConvMeta cm = meta[group];
    device float* st = conv_buf + cm.offset_f;
    for (uint c = tid; c < cm.width; c += threads) {
        float s0 = st[(commit + 0u) * cm.width + c];
        float s1 = st[(commit + 1u) * cm.width + c];
        float s2 = st[(commit + 2u) * cm.width + c];
        st[0u * cm.width + c] = s0;
        st[1u * cm.width + c] = s1;
        st[2u * cm.width + c] = s2;
    }
}

struct ClipB { uint width; uint has_scale; uint pos0; uint is_local; uint n_floor; uint in_stride; uint out_stride; uint mode; float threshold; float alpha; };

kernel void clip_rows_batch(device const float* x [[buffer(0)]],
                            device const ushort* scale [[buffer(1)]],
                            device float* y [[buffer(2)]],
                            constant ClipB& p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x, m = group.y;
    device const float* xm = x + ulong(m) * p.in_stride + row * p.width;
    device float* ym = y + ulong(m) * p.out_stride + row * p.width;
    float v[4];
    float s = 0;
    uint n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) {
        float a = xm[i];
        if (p.has_scale) a *= bfv(scale[row * p.width + i]);
        v[n] = a;
        s += a * a;
    }
    s = simd_sum(s);
    float coeff;
    if (p.mode == 1u) coeff = 1.0f / precise::sqrt(s / float(p.width) + 1e-6f);
    else coeff = min(p.threshold / (1e-6f + precise::sqrt(s)), 1.0f);
    float tau = p.is_local ? 1.0f : (1.0f + p.alpha * precise::log(max((float(p.pos0 + m) + 1.0f) / float(p.n_floor), 1.0f)));
    n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) {
        float out = v[n] * coeff;
        if (tau != 1.0f) out = rbf(out) * tau;
        ym[i] = rbf(out);
    }
}

struct ClipKvB { uint width; uint pos0; uint kv_stride; uint in_stride; uint mode; uint kv_cap; float threshold; };

kernel void clip_kv_store_batch(device const float* x [[buffer(0)]],
                                device ushort* cache [[buffer(1)]],
                                constant ClipKvB& p [[buffer(2)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x, m = group.y;
    device const float* xm = x + ulong(m) * p.in_stride + row * p.width;
    float v[4];
    float s = 0;
    uint n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) {
        float a = xm[i];
        v[n] = a;
        s += a * a;
    }
    s = simd_sum(s);
    float coeff = p.mode == 1u ? (1.0f / precise::sqrt(s / float(p.width) + 1e-6f)) : min(p.threshold / (1e-6f + precise::sqrt(s)), 1.0f);
    device ushort* dst = cache + ulong((p.pos0 + m) % p.kv_cap) * p.kv_stride + row * p.width;
    n = 0;
    for (uint i = lane; i < p.width; i += 32u, ++n) dst[i] = f2bf(v[n] * coeff);
}

struct RelB { uint d_rel; uint extent; uint heads; };

kernel void rel_bias_batch(device const float* r [[buffer(0)]],
                           device const ushort* proj [[buffer(1)]],
                           device float* rel [[buffer(2)]],
                           constant RelB& p [[buffer(3)]],
                           uint3 gid [[thread_position_in_grid]]) {
    uint d = gid.x, h = gid.y, m = gid.z;
    if (d >= p.extent) return;
    device const float* rm = r + ulong(m) * p.heads * p.d_rel + h * p.d_rel;
    float acc = 0;
    for (uint i = 0; i < p.d_rel; ++i) acc += rm[i] * bfv(proj[i * p.extent + d]);
    rel[ulong(m) * p.heads * p.extent + h * p.extent + d] = rbf(acc);
}

struct AttnB { uint pos0; uint rows; uint window; uint is_local; uint kv_heads; uint group; uint head_dim; uint extent; uint heads; uint q_stride; uint gate_stride; uint gate_off; uint sink_off; uint out_stride; uint kv_cap; float scale; };

kernel void attend_gated_batch(device const float4* q [[buffer(0)]],
                               device const ushort4* kc [[buffer(1)]],
                               device const ushort4* vc [[buffer(2)]],
                               device const float* rel [[buffer(3)]],
                               device const float* qbundle [[buffer(4)]],
                               device float4* gated [[buffer(5)]],
                               constant AttnB& p [[buffer(6)]],
                               uint2 group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[4];
    threadgroup float sl[4];
    threadgroup float4 so[4][32];
    uint h = group.x;
    uint m0 = group.y * 4u;
    uint mb = min(4u, p.rows - m0);
    uint kvh = h / p.group;
    uint qs4 = p.q_stride / 4u;
    float4 q0 = q[ulong(m0) * qs4 + h * 32u + lane];
    float4 q1 = mb > 1u ? q[ulong(m0 + 1u) * qs4 + h * 32u + lane] : float4(0.0f);
    float4 q2 = mb > 2u ? q[ulong(m0 + 2u) * qs4 + h * 32u + lane] : float4(0.0f);
    float4 q3 = mb > 3u ? q[ulong(m0 + 3u) * qs4 + h * 32u + lane] : float4(0.0f);
    float mv0 = -INFINITY, mv1 = -INFINITY, mv2 = -INFINITY, mv3 = -INFINITY;
    float lv0 = 0.0f, lv1 = 0.0f, lv2 = 0.0f, lv3 = 0.0f;
    float4 o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    uint pos_first = p.pos0 + m0;
    uint pos_last = pos_first + mb - 1u;
    uint kv_start = (p.is_local && pos_first + 1u > p.window) ? (pos_first + 1u - p.window) : 0u;
    uint count = pos_last - kv_start + 1u;
    for (uint t = simd; t < count; t += 4u) {
        uint kp = kv_start + t;
        ulong base = (ulong(kp % p.kv_cap) * p.kv_heads + kvh) * 32u + lane;
        ushort4 ku = kc[base];
        float4 k4 = float4(bfv(ku.x), bfv(ku.y), bfv(ku.z), bfv(ku.w));
        ushort4 vu = vc[base];
        float4 v4 = float4(bfv(vu.x), bfv(vu.y), bfv(vu.z), bfv(vu.w));
        {
            uint qpos = pos_first;
            uint qstart = (p.is_local && qpos + 1u > p.window) ? (qpos + 1u - p.window) : 0u;
            if (kp >= qstart && kp <= qpos) {
                float s = simd_sum(dot(q0, k4)) * p.scale;
                uint dist = qpos - kp;
                s += dist < p.extent ? rel[(ulong(m0) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv0, s);
                float a = mv0 == -INFINITY ? 0.0f : precise::exp(mv0 - mn);
                float b = precise::exp(s - mn);
                o0 = o0 * a + v4 * b; lv0 = lv0 * a + b; mv0 = mn;
            }
        }
        if (mb > 1u) {
            uint qpos = pos_first + 1u;
            uint qstart = (p.is_local && qpos + 1u > p.window) ? (qpos + 1u - p.window) : 0u;
            if (kp >= qstart && kp <= qpos) {
                float s = simd_sum(dot(q1, k4)) * p.scale;
                uint dist = qpos - kp;
                s += dist < p.extent ? rel[(ulong(m0 + 1u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv1, s);
                float a = mv1 == -INFINITY ? 0.0f : precise::exp(mv1 - mn);
                float b = precise::exp(s - mn);
                o1 = o1 * a + v4 * b; lv1 = lv1 * a + b; mv1 = mn;
            }
        }
        if (mb > 2u) {
            uint qpos = pos_first + 2u;
            uint qstart = (p.is_local && qpos + 1u > p.window) ? (qpos + 1u - p.window) : 0u;
            if (kp >= qstart && kp <= qpos) {
                float s = simd_sum(dot(q2, k4)) * p.scale;
                uint dist = qpos - kp;
                s += dist < p.extent ? rel[(ulong(m0 + 2u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv2, s);
                float a = mv2 == -INFINITY ? 0.0f : precise::exp(mv2 - mn);
                float b = precise::exp(s - mn);
                o2 = o2 * a + v4 * b; lv2 = lv2 * a + b; mv2 = mn;
            }
        }
        if (mb > 3u) {
            uint qpos = pos_first + 3u;
            uint qstart = (p.is_local && qpos + 1u > p.window) ? (qpos + 1u - p.window) : 0u;
            if (kp >= qstart && kp <= qpos) {
                float s = simd_sum(dot(q3, k4)) * p.scale;
                uint dist = qpos - kp;
                s += dist < p.extent ? rel[(ulong(m0 + 3u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv3, s);
                float a = mv3 == -INFINITY ? 0.0f : precise::exp(mv3 - mn);
                float b = precise::exp(s - mn);
                o3 = o3 * a + v4 * b; lv3 = lv3 * a + b; mv3 = mn;
            }
        }
    }
    uint os4 = p.out_stride / 4u;
    for (uint j = 0; j < mb; ++j) {
        float mj = j == 0u ? mv0 : (j == 1u ? mv1 : (j == 2u ? mv2 : mv3));
        float lj = j == 0u ? lv0 : (j == 1u ? lv1 : (j == 2u ? lv2 : lv3));
        float4 oj = j == 0u ? o0 : (j == 1u ? o1 : (j == 2u ? o2 : o3));
        if (lane == 0) { sm[simd] = mj; sl[simd] = lj; }
        so[simd][lane] = oj;
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
            uint gm = m0 + j;
            float sink = qbundle[ulong(gm) * p.gate_stride + p.sink_off + h];
            float ssink = sigm(lse - sink);
            device const float4* gptr = (device const float4*)(qbundle + ulong(gm) * p.gate_stride + p.gate_off);
            float4 g4 = gptr[h * 32u + lane];
            float4 sg = float4(sigm(g4.x), sigm(g4.y), sigm(g4.z), sigm(g4.w));
            float4 y4 = (acc / tot) * sg * ssink;
            gated[ulong(gm) * os4 + h * 32u + lane] = float4(rbf(y4.x), rbf(y4.y), rbf(y4.z), rbf(y4.w));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void route_batch(device const float* logits [[buffer(0)]],
                        device const float* bias [[buffer(1)]],
                        device uint* ids [[buffer(2)]],
                        device float* gammas [[buffer(3)]],
                        constant Route& p [[buffer(4)]],
                        uint m [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    device const float* lm = logits + ulong(m) * p.experts;
    device uint* im = ids + ulong(m) * p.top;
    device float* gm = gammas + ulong(m) * p.top;
    float val[8];
    uint idx[8];
    uint count = (p.experts + 31u) / 32u;
    for (uint j = 0; j < count; ++j) {
        uint e = lane + j * 32u;
        idx[j] = e;
        val[j] = e < p.experts ? (sigm(lm[e]) + bias[e]) : -INFINITY;
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
        float lp[16], mx = -INFINITY;
        for (uint k = 0; k < p.top; ++k) {
            float l = lm[chosen[k]];
            lp[k] = min(l, 0.0f) - precise::log(1.0f + precise::exp(-fabs(l)));
            mx = max(mx, lp[k]);
        }
        float s[16];
        for (uint k = 0; k < p.top; ++k) s[k] = precise::exp(lp[k] - mx);
        float tot = 0.0f;
        if (p.top == 8u) tot = ((s[0] + s[2]) + (s[1] + s[3])) + ((s[4] + s[6]) + (s[5] + s[7]));
        else for (uint k = 0; k < p.top; ++k) tot += s[k];
        for (uint k = 0; k < p.top; ++k) {
            im[k] = chosen[k];
            gm[k] = precise::exp(lp[k] - (mx + precise::log(tot))) * p.scale;
        }
    }
}

struct UnionP { uint rows; uint top; uint experts; uint w13_groups; uint w2_groups; };

kernel void build_expert_union(device const uint* ids [[buffer(0)]],
                               device uint* u_exp [[buffer(1)]],
                               device uint* u_cnt [[buffer(2)]],
                               device uint* u_pairs [[buffer(3)]],
                               device uint* ind_w13 [[buffer(4)]],
                               device uint* ind_w2 [[buffer(5)]],
                               constant UnionP& p [[buffer(6)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint all_ids[128];
    threadgroup uint stot[4];
    uint total_mk = p.rows * p.top;
    if (tid < total_mk) all_ids[tid] = ids[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cnt = 0u;
    uint pairs[16];
    if (tid < p.experts) {
        for (uint mk = 0; mk < total_mk; ++mk) {
            if (all_ids[mk] == tid) pairs[cnt++] = mk;
        }
    }
    uint active = cnt > 0u ? 1u : 0u;
    uint lrank = simd_prefix_exclusive_sum(active);
    uint ssum = simd_sum(active);
    if (lane == 0) stot[simd] = ssum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint base = 0u;
    for (uint s = 0; s < simd; ++s) base += stot[s];
    uint u = base + lrank;
    if (active) {
        u_exp[u] = tid;
        u_cnt[u] = cnt;
        for (uint j = 0; j < cnt; ++j) u_pairs[u * 16u + j] = pairs[j];
    }
    if (tid == 0) {
        uint total_u = stot[0] + stot[1] + stot[2] + stot[3];
        ind_w13[0] = p.w13_groups;
        ind_w13[1] = max(total_u, 1u);
        ind_w13[2] = 1u;
        ind_w2[0] = p.w2_groups;
        ind_w2[1] = max(total_u, 1u);
        ind_w2[2] = 1u;
    }
}

struct ExpertG { uint rows; uint cols; uint in_stride; uint out_stride; uint per_row_input; uint top; ulong expert_bytes; ulong scale_bytes; };

kernel void expert_nvfp4_grouped(device const uchar* w [[buffer(0)]],
                                 device const uchar* sc [[buffer(1)]],
                                 device const float* amax [[buffer(2)]],
                                 device const float* denom [[buffer(3)]],
                                 device const half* x [[buffer(4)]],
                                 device const float* outer [[buffer(5)]],
                                 device const uint* u_exp [[buffer(6)]],
                                 device const uint* u_cnt [[buffer(7)]],
                                 device const uint* u_pairs [[buffer(8)]],
                                 device float* y [[buffer(9)]],
                                 constant ExpertG& p [[buffer(10)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint simds [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * simds + simd;
    uint u = group.y;
    if (row >= p.rows) return;
    uint e = u_exp[u];
    uint cnt = u_cnt[u];
    device const uchar* wr = w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const half4* x4 = (device const half4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 16u;
    float alpha = amax[e] / denom[0];
    if (cnt == 1u) {
        uint mk0 = u_pairs[u * 16u];
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        device const half4* x0 = x4 + ulong(i0) * is4;
        float a0 = 0.0f;
        for (uint b = lane; b < blocks; b += 32u) {
            uint2 packed = *(device const uint2*)(wr + b * 8u);
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float4 part = fp4x4(packed.x) * float4(x0[b4]) + fp4x4(packed.x >> 16) * float4(x0[b4 + 1u])
                        + fp4x4(packed.y) * float4(x0[b4 + 2u]) + fp4x4(packed.y >> 16) * float4(x0[b4 + 3u]);
            a0 += ((part.x + part.y) + (part.z + part.w)) * s;
        }
        float t0 = simd_sum(a0);
        if (lane == 0) { float r = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; }
        return;
    }
    if (cnt == 2u) {
        uint mk0 = u_pairs[u * 16u], mk1 = u_pairs[u * 16u + 1u];
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        device const half4* x0 = x4 + ulong(i0) * is4;
        device const half4* x1 = x4 + ulong(i1) * is4;
        float a0 = 0.0f, a1 = 0.0f;
        for (uint b = lane; b < blocks; b += 32u) {
            uint2 packed = *(device const uint2*)(wr + b * 8u);
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float4 p0 = 0.0f, p1 = 0.0f;
            { float4 f = fp4x4(packed.x);       p0 += f * float4(x0[b4]);      p1 += f * float4(x1[b4]); }
            { float4 f = fp4x4(packed.x >> 16); p0 += f * float4(x0[b4 + 1u]); p1 += f * float4(x1[b4 + 1u]); }
            { float4 f = fp4x4(packed.y);       p0 += f * float4(x0[b4 + 2u]); p1 += f * float4(x1[b4 + 2u]); }
            { float4 f = fp4x4(packed.y >> 16); p0 += f * float4(x0[b4 + 3u]); p1 += f * float4(x1[b4 + 3u]); }
            a0 += ((p0.x + p0.y) + (p0.z + p0.w)) * s;
            a1 += ((p1.x + p1.y) + (p1.z + p1.w)) * s;
        }
        float t0 = simd_sum(a0), t1 = simd_sum(a1);
        if (lane == 0) {
            float r0 = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r0) : r0;
            float r1 = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r1) : r1;
        }
        return;
    }
    if (cnt == 3u) {
        uint mk0 = u_pairs[u * 16u], mk1 = u_pairs[u * 16u + 1u], mk2 = u_pairs[u * 16u + 2u];
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
        device const half4* x0 = x4 + ulong(i0) * is4;
        device const half4* x1 = x4 + ulong(i1) * is4;
        device const half4* x2 = x4 + ulong(i2) * is4;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (uint b = lane; b < blocks; b += 32u) {
            uint2 packed = *(device const uint2*)(wr + b * 8u);
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float4 p0 = 0.0f, p1 = 0.0f, p2 = 0.0f;
            { float4 f = fp4x4(packed.x);       p0 += f * float4(x0[b4]);      p1 += f * float4(x1[b4]);      p2 += f * float4(x2[b4]); }
            { float4 f = fp4x4(packed.x >> 16); p0 += f * float4(x0[b4 + 1u]); p1 += f * float4(x1[b4 + 1u]); p2 += f * float4(x2[b4 + 1u]); }
            { float4 f = fp4x4(packed.y);       p0 += f * float4(x0[b4 + 2u]); p1 += f * float4(x1[b4 + 2u]); p2 += f * float4(x2[b4 + 2u]); }
            { float4 f = fp4x4(packed.y >> 16); p0 += f * float4(x0[b4 + 3u]); p1 += f * float4(x1[b4 + 3u]); p2 += f * float4(x2[b4 + 3u]); }
            a0 += ((p0.x + p0.y) + (p0.z + p0.w)) * s;
            a1 += ((p1.x + p1.y) + (p1.z + p1.w)) * s;
            a2 += ((p2.x + p2.y) + (p2.z + p2.w)) * s;
        }
        float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2);
        if (lane == 0) {
            float r0 = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r0) : r0;
            float r1 = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r1) : r1;
            float r2 = t2 * outer[i2] * alpha; y[ulong(mk2) * p.out_stride + row] = p.per_row_input ? rbf(r2) : r2;
        }
        return;
    }
    if (cnt == 4u) {
        uint mk0 = u_pairs[u * 16u], mk1 = u_pairs[u * 16u + 1u], mk2 = u_pairs[u * 16u + 2u], mk3 = u_pairs[u * 16u + 3u];
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
        uint i3 = p.per_row_input ? mk3 : (mk3 / p.top);
        device const half4* x0 = x4 + ulong(i0) * is4;
        device const half4* x1 = x4 + ulong(i1) * is4;
        device const half4* x2 = x4 + ulong(i2) * is4;
        device const half4* x3 = x4 + ulong(i3) * is4;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = lane; b < blocks; b += 32u) {
            uint2 packed = *(device const uint2*)(wr + b * 8u);
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float4 p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
            { float4 f = fp4x4(packed.x);       p0 += f * float4(x0[b4]);      p1 += f * float4(x1[b4]);      p2 += f * float4(x2[b4]);      p3 += f * float4(x3[b4]); }
            { float4 f = fp4x4(packed.x >> 16); p0 += f * float4(x0[b4 + 1u]); p1 += f * float4(x1[b4 + 1u]); p2 += f * float4(x2[b4 + 1u]); p3 += f * float4(x3[b4 + 1u]); }
            { float4 f = fp4x4(packed.y);       p0 += f * float4(x0[b4 + 2u]); p1 += f * float4(x1[b4 + 2u]); p2 += f * float4(x2[b4 + 2u]); p3 += f * float4(x3[b4 + 2u]); }
            { float4 f = fp4x4(packed.y >> 16); p0 += f * float4(x0[b4 + 3u]); p1 += f * float4(x1[b4 + 3u]); p2 += f * float4(x2[b4 + 3u]); p3 += f * float4(x3[b4 + 3u]); }
            a0 += ((p0.x + p0.y) + (p0.z + p0.w)) * s;
            a1 += ((p1.x + p1.y) + (p1.z + p1.w)) * s;
            a2 += ((p2.x + p2.y) + (p2.z + p2.w)) * s;
            a3 += ((p3.x + p3.y) + (p3.z + p3.w)) * s;
        }
        float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3);
        if (lane == 0) {
            float r0 = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r0) : r0;
            float r1 = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r1) : r1;
            float r2 = t2 * outer[i2] * alpha; y[ulong(mk2) * p.out_stride + row] = p.per_row_input ? rbf(r2) : r2;
            float r3 = t3 * outer[i3] * alpha; y[ulong(mk3) * p.out_stride + row] = p.per_row_input ? rbf(r3) : r3;
        }
        return;
    }
    for (uint j0 = 0; j0 < cnt; j0 += 8u) {
        uint rem = min(8u, cnt - j0);
        uint mk0 = u_pairs[u * 16u + j0];
        uint mk1 = rem > 1u ? u_pairs[u * 16u + j0 + 1u] : mk0;
        uint mk2 = rem > 2u ? u_pairs[u * 16u + j0 + 2u] : mk0;
        uint mk3 = rem > 3u ? u_pairs[u * 16u + j0 + 3u] : mk0;
        uint mk4 = rem > 4u ? u_pairs[u * 16u + j0 + 4u] : mk0;
        uint mk5 = rem > 5u ? u_pairs[u * 16u + j0 + 5u] : mk0;
        uint mk6 = rem > 6u ? u_pairs[u * 16u + j0 + 6u] : mk0;
        uint mk7 = rem > 7u ? u_pairs[u * 16u + j0 + 7u] : mk0;
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
        uint i3 = p.per_row_input ? mk3 : (mk3 / p.top);
        uint i4 = p.per_row_input ? mk4 : (mk4 / p.top);
        uint i5 = p.per_row_input ? mk5 : (mk5 / p.top);
        uint i6 = p.per_row_input ? mk6 : (mk6 / p.top);
        uint i7 = p.per_row_input ? mk7 : (mk7 / p.top);
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (uint b = lane; b < blocks; b += 32u) {
            uint2 packed = *(device const uint2*)(wr + b * 8u);
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float4 p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0, p6 = 0, p7 = 0;
            { float4 f = fp4x4(packed.x);       p0 += f * float4(x4[ulong(i0)*is4+b4]);      if (rem > 1u) p1 += f * float4(x4[ulong(i1)*is4+b4]);      if (rem > 2u) p2 += f * float4(x4[ulong(i2)*is4+b4]);      if (rem > 3u) p3 += f * float4(x4[ulong(i3)*is4+b4]);      if (rem > 4u) p4 += f * float4(x4[ulong(i4)*is4+b4]);      if (rem > 5u) p5 += f * float4(x4[ulong(i5)*is4+b4]);      if (rem > 6u) p6 += f * float4(x4[ulong(i6)*is4+b4]);      if (rem > 7u) p7 += f * float4(x4[ulong(i7)*is4+b4]); }
            { float4 f = fp4x4(packed.x >> 16); uint k = b4 + 1u; p0 += f * float4(x4[ulong(i0)*is4+k]); if (rem > 1u) p1 += f * float4(x4[ulong(i1)*is4+k]); if (rem > 2u) p2 += f * float4(x4[ulong(i2)*is4+k]); if (rem > 3u) p3 += f * float4(x4[ulong(i3)*is4+k]); if (rem > 4u) p4 += f * float4(x4[ulong(i4)*is4+k]); if (rem > 5u) p5 += f * float4(x4[ulong(i5)*is4+k]); if (rem > 6u) p6 += f * float4(x4[ulong(i6)*is4+k]); if (rem > 7u) p7 += f * float4(x4[ulong(i7)*is4+k]); }
            { float4 f = fp4x4(packed.y);       uint k = b4 + 2u; p0 += f * float4(x4[ulong(i0)*is4+k]); if (rem > 1u) p1 += f * float4(x4[ulong(i1)*is4+k]); if (rem > 2u) p2 += f * float4(x4[ulong(i2)*is4+k]); if (rem > 3u) p3 += f * float4(x4[ulong(i3)*is4+k]); if (rem > 4u) p4 += f * float4(x4[ulong(i4)*is4+k]); if (rem > 5u) p5 += f * float4(x4[ulong(i5)*is4+k]); if (rem > 6u) p6 += f * float4(x4[ulong(i6)*is4+k]); if (rem > 7u) p7 += f * float4(x4[ulong(i7)*is4+k]); }
            { float4 f = fp4x4(packed.y >> 16); uint k = b4 + 3u; p0 += f * float4(x4[ulong(i0)*is4+k]); if (rem > 1u) p1 += f * float4(x4[ulong(i1)*is4+k]); if (rem > 2u) p2 += f * float4(x4[ulong(i2)*is4+k]); if (rem > 3u) p3 += f * float4(x4[ulong(i3)*is4+k]); if (rem > 4u) p4 += f * float4(x4[ulong(i4)*is4+k]); if (rem > 5u) p5 += f * float4(x4[ulong(i5)*is4+k]); if (rem > 6u) p6 += f * float4(x4[ulong(i6)*is4+k]); if (rem > 7u) p7 += f * float4(x4[ulong(i7)*is4+k]); }
            a0 += ((p0.x + p0.y) + (p0.z + p0.w)) * s;
            if (rem > 1u) a1 += ((p1.x + p1.y) + (p1.z + p1.w)) * s;
            if (rem > 2u) a2 += ((p2.x + p2.y) + (p2.z + p2.w)) * s;
            if (rem > 3u) a3 += ((p3.x + p3.y) + (p3.z + p3.w)) * s;
            if (rem > 4u) a4 += ((p4.x + p4.y) + (p4.z + p4.w)) * s;
            if (rem > 5u) a5 += ((p5.x + p5.y) + (p5.z + p5.w)) * s;
            if (rem > 6u) a6 += ((p6.x + p6.y) + (p6.z + p6.w)) * s;
            if (rem > 7u) a7 += ((p7.x + p7.y) + (p7.z + p7.w)) * s;
        }
        float t0 = simd_sum(a0); if (lane == 0) { float r = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; }
        if (rem > 1u) { float t1 = simd_sum(a1); if (lane == 0) { float r = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 2u) { float t2 = simd_sum(a2); if (lane == 0) { float r = t2 * outer[i2] * alpha; y[ulong(mk2) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 3u) { float t3 = simd_sum(a3); if (lane == 0) { float r = t3 * outer[i3] * alpha; y[ulong(mk3) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 4u) { float t4 = simd_sum(a4); if (lane == 0) { float r = t4 * outer[i4] * alpha; y[ulong(mk4) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 5u) { float t5 = simd_sum(a5); if (lane == 0) { float r = t5 * outer[i5] * alpha; y[ulong(mk5) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 6u) { float t6 = simd_sum(a6); if (lane == 0) { float r = t6 * outer[i6] * alpha; y[ulong(mk6) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 7u) { float t7 = simd_sum(a7); if (lane == 0) { float r = t7 * outer[i7] * alpha; y[ulong(mk7) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
    }
}

kernel void expert_mxfp8_grouped(device const uchar* w [[buffer(0)]],
                                 device const uchar* sc [[buffer(1)]],
                                 device const float* x [[buffer(4)]],
                                 device const uint* u_exp [[buffer(6)]],
                                 device const uint* u_cnt [[buffer(7)]],
                                 device const uint* u_pairs [[buffer(8)]],
                                 device float* y [[buffer(9)]],
                                 constant ExpertG& p [[buffer(10)]],
                                 uint2 group [[threadgroup_position_in_grid]],
                                 uint simd [[simdgroup_index_in_threadgroup]],
                                 uint simds [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    uint row = group.x * simds + simd;
    uint u = group.y;
    if (row >= p.rows) return;
    uint e = u_exp[u];
    uint cnt = u_cnt[u];
    device const uint4* wr4 = (device const uint4*)(w + ulong(e) * p.expert_bytes + ulong(row) * p.cols);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 32u);
    device const float4* x4 = (device const float4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 32u;
    for (uint j0 = 0; j0 < cnt; j0 += 4u) {
        uint rem = min(4u, cnt - j0);
        uint mk0 = u_pairs[u * 16u + j0];
        uint mk1 = rem > 1u ? u_pairs[u * 16u + j0 + 1u] : mk0;
        uint mk2 = rem > 2u ? u_pairs[u * 16u + j0 + 2u] : mk0;
        uint mk3 = rem > 3u ? u_pairs[u * 16u + j0 + 3u] : mk0;
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
        uint i3 = p.per_row_input ? mk3 : (mk3 / p.top);
        device const float4* x0 = x4 + ulong(i0) * is4;
        device const float4* x1 = x4 + ulong(i1) * is4;
        device const float4* x2 = x4 + ulong(i2) * is4;
        device const float4* x3 = x4 + ulong(i3) * is4;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = lane; b < blocks; b += 32u) {
            uint4 w0 = wr4[b * 2u];
            uint4 w1 = wr4[b * 2u + 1u];
            float4 f0 = e4m3x4(w0.x), f1 = e4m3x4(w0.y), f2 = e4m3x4(w0.z), f3 = e4m3x4(w0.w);
            float4 f4 = e4m3x4(w1.x), f5 = e4m3x4(w1.y), f6 = e4m3x4(w1.z), f7 = e4m3x4(w1.w);
            float s = as_type<float>(uint(sr[b]) << 23);
            uint b8 = b * 8u;
            { float4 pt = (f0*x0[b8] + f1*x0[b8+1u] + f2*x0[b8+2u] + f3*x0[b8+3u]) + (f4*x0[b8+4u] + f5*x0[b8+5u] + f6*x0[b8+6u] + f7*x0[b8+7u]); a0 += ((pt.x + pt.y) + (pt.z + pt.w)) * s; }
            if (rem > 1u) { float4 pt = (f0*x1[b8] + f1*x1[b8+1u] + f2*x1[b8+2u] + f3*x1[b8+3u]) + (f4*x1[b8+4u] + f5*x1[b8+5u] + f6*x1[b8+6u] + f7*x1[b8+7u]); a1 += ((pt.x + pt.y) + (pt.z + pt.w)) * s; }
            if (rem > 2u) { float4 pt = (f0*x2[b8] + f1*x2[b8+1u] + f2*x2[b8+2u] + f3*x2[b8+3u]) + (f4*x2[b8+4u] + f5*x2[b8+5u] + f6*x2[b8+6u] + f7*x2[b8+7u]); a2 += ((pt.x + pt.y) + (pt.z + pt.w)) * s; }
            if (rem > 3u) { float4 pt = (f0*x3[b8] + f1*x3[b8+1u] + f2*x3[b8+2u] + f3*x3[b8+3u]) + (f4*x3[b8+4u] + f5*x3[b8+5u] + f6*x3[b8+6u] + f7*x3[b8+7u]); a3 += ((pt.x + pt.y) + (pt.z + pt.w)) * s; }
        }
        float t0 = simd_sum(a0); if (lane == 0) y[ulong(mk0) * p.out_stride + row] = rbf(t0);
        if (rem > 1u) { float t1 = simd_sum(a1); if (lane == 0) y[ulong(mk1) * p.out_stride + row] = rbf(t1); }
        if (rem > 2u) { float t2 = simd_sum(a2); if (lane == 0) y[ulong(mk2) * p.out_stride + row] = rbf(t2); }
        if (rem > 3u) { float t3 = simd_sum(a3); if (lane == 0) y[ulong(mk3) * p.out_stride + row] = rbf(t3); }
    }
}

struct CombineB { uint n; uint top; uint boundary; uint rows; };

kernel void combine_batch(device const float* rows_buf [[buffer(0)]],
                          device const uint* ids [[buffer(1)]],
                          device const float* attention [[buffer(2)]],
                          device const float* partial [[buffer(3)]],
                          device float* conv_state [[buffer(4)]],
                          device const ushort* w [[buffer(5)]],
                          device float* out [[buffer(6)]],
                          constant CombineB& p [[buffer(7)]],
                          uint c [[thread_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint order[16];
    float h0 = c < p.n ? conv_state[c] : 0.0f;
    float h1 = c < p.n ? conv_state[p.n + c] : 0.0f;
    float h2 = c < p.n ? conv_state[2u * p.n + c] : 0.0f;
    float w0 = c < p.n ? bfv(w[c * 4u + 0u]) : 0.0f;
    float w1 = c < p.n ? bfv(w[c * 4u + 1u]) : 0.0f;
    float w2 = c < p.n ? bfv(w[c * 4u + 2u]) : 0.0f;
    float w3 = c < p.n ? bfv(w[c * 4u + 3u]) : 0.0f;
    for (uint m = 0; m < p.rows; ++m) {
        if (tid == 0) {
            device const uint* im = ids + ulong(m) * p.top;
            for (uint k = 0; k < p.top; ++k) order[k] = k;
            for (uint a = 1; a < p.top; ++a) {
                for (uint b = a; b > 0 && im[order[b - 1]] > im[order[b]]; --b) {
                    uint t = order[b]; order[b] = order[b - 1]; order[b - 1] = t;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (c < p.n) {
            device const float* rm = rows_buf + ulong(m) * p.top * p.n;
            float g = 0.0f;
            for (uint k = 0; k < p.top; ++k) g += rm[order[k] * p.n + c];
            float cur = rbf(g);
            conv_state[(3u + m) * p.n + c] = cur;
            float acc = 0.0f;
            acc = fma(h0, w0, acc);
            acc = fma(h1, w1, acc);
            acc = fma(h2, w2, acc);
            acc = fma(cur, w3, acc);
            float yv = rbf((acc + cur) * 0.4472135955f);
            float attv = attention[ulong(m) * p.n + c];
            float partv = partial[ulong(m) * p.n + c];
            out[ulong(m) * p.n + c] = p.boundary ? rbf(attv + yv) : rbf((partv + attv) + yv);
            h0 = h1; h1 = h2; h2 = cur;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void cert_head_xnorm_batch(device const float4* x [[buffer(0)]],
                                  device float* state [[buffer(1)]],
                                  device uint* count [[buffer(2)]],
                                  constant uint& n [[buffer(3)]],
                                  uint m [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = n / 4u;
    device const float4* xm = x + ulong(m) * n4;
    float s = tid < n4 ? dot(xm[tid], xm[tid]) : 0.0f;
    for (uint i = tid + threads; i < n4; i += threads) s += dot(xm[i], xm[i]);
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        for (uint i = 0; i < (threads + 31u) / 32u; ++i) tot += part[i];
        state[m * 4u + 2u] = precise::sqrt(tot);
        count[m] = 0u;
    }
}

struct CertHeadB { uint vocab; uint cols; uint ng; uint ntg; uint m_base; float cap; };

kernel void cert_head_scan_rows(device const uint4* q4 [[buffer(0)]],
                                device const half* sc [[buffer(1)]],
                                device const float* err [[buffer(2)]],
                                device const float4* x4 [[buffer(3)]],
                                device const float* state [[buffer(4)]],
                                device float* ub_out [[buffer(5)]],
                                device float* tg_val [[buffer(6)]],
                                device uint* tg_row [[buffer(7)]],
                                constant CertHeadB& p [[buffer(8)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float svals[8];
    threadgroup uint srows[8];
    uint row = group * simds + simd;
    uint xs4 = p.cols / 4u;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    if (row < p.vocab) {
        device const uint4* qr = q4 + ulong(row) * p.ng;
        device const half* sr = sc + ulong(row) * p.ng;
        for (uint g = lane; g < p.ng; g += 32u) {
            uint4 pk = qr[g];
            float s = float(sr[g]);
            float4 w0 = u4_lo(pk.x), w1 = u4_hi(pk.x), w2 = u4_lo(pk.y), w3 = u4_hi(pk.y);
            float4 w4 = u4_lo(pk.z), w5 = u4_hi(pk.z), w6 = u4_lo(pk.w), w7 = u4_hi(pk.w);
            uint g8 = g * 8u;
            if (ROWS > 0u) { device const float4* xg = x4 + ulong(p.m_base + 0u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a0 += ((d.x+d.y)+(d.z+d.w))*s; }
            if (ROWS > 1u) { device const float4* xg = x4 + ulong(p.m_base + 1u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a1 += ((d.x+d.y)+(d.z+d.w))*s; }
            if (ROWS > 2u) { device const float4* xg = x4 + ulong(p.m_base + 2u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a2 += ((d.x+d.y)+(d.z+d.w))*s; }
            if (ROWS > 3u) { device const float4* xg = x4 + ulong(p.m_base + 3u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a3 += ((d.x+d.y)+(d.z+d.w))*s; }
            if (ROWS > 4u) { device const float4* xg = x4 + ulong(p.m_base + 4u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a4 += ((d.x+d.y)+(d.z+d.w))*s; }
            if (ROWS > 5u) { device const float4* xg = x4 + ulong(p.m_base + 5u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a5 += ((d.x+d.y)+(d.z+d.w))*s; }
            if (ROWS > 6u) { device const float4* xg = x4 + ulong(p.m_base + 6u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a6 += ((d.x+d.y)+(d.z+d.w))*s; }
            if (ROWS > 7u) { device const float4* xg = x4 + ulong(p.m_base + 7u) * xs4 + g8; float4 d = w0*xg[0]+w1*xg[1]+w2*xg[2]+w3*xg[3]+w4*xg[4]+w5*xg[5]+w6*xg[6]+w7*xg[7]; a7 += ((d.x+d.y)+(d.z+d.w))*s; }
        }
    }
    float er = row < p.vocab ? err[row] : 0.0f;
    for (uint m = 0; m < ROWS; ++m) {
        uint gm = p.m_base + m;
        float av = m == 0u ? a0 : (m == 1u ? a1 : (m == 2u ? a2 : (m == 3u ? a3 : (m == 4u ? a4 : (m == 5u ? a5 : (m == 6u ? a6 : a7))))));
        if (row < p.vocab) {
            float shat = simd_sum(av);
            if (lane == 0) {
                ub_out[ulong(gm) * p.vocab + row] = shat + state[gm * 4u + 2u] * er;
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
            tg_val[ulong(gm) * p.ntg + group] = bv;
            tg_row[ulong(gm) * p.ntg + group] = br;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void cert_head_tau_batch(device const float* tg_val [[buffer(0)]],
                                device const uint* tg_row [[buffer(1)]],
                                device const uint4* w [[buffer(2)]],
                                device const float4* x4 [[buffer(3)]],
                                device float* state [[buffer(4)]],
                                constant CertTau& p [[buffer(5)]],
                                uint m [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float exact_s[16];
    device const float* tv = tg_val + ulong(m) * p.ntg;
    device const uint* tr = tg_row + ulong(m) * p.ntg;
    device const float4* xm = x4 + ulong(m) * (p.cols / 4u);
    float bv = -INFINITY;
    uint br = 0xffffffffu;
    for (uint i = tid; i < p.ntg; i += 512u) {
        float v = tv[i];
        uint r = tr[i];
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
            float4 a = xm[c * 2u];
            float4 b = xm[c * 2u + 1u];
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
        float u = clamp(s_lo / p.cap, -0.9999f, 0.9999f);
        float z_lo = p.cap * 0.5f * precise::log((1.0f + u) / (1.0f - u));
        state[m * 4u + 0u] = z_lo - fabs(z_lo) * 0.005f - 0.01f;
        state[m * 4u + 1u] = tau_s;
    }
}

kernel void cert_head_filter_batch(device const float* ub [[buffer(0)]],
                                   device const float* state [[buffer(1)]],
                                   device atomic_uint* count [[buffer(2)]],
                                   device uint* cands [[buffer(3)]],
                                   constant uint& vocab [[buffer(4)]],
                                   uint2 gid [[thread_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    uint i = gid.x, m = gid.y;
    bool keep = i < vocab && ub[ulong(m) * vocab + i] >= state[m * 4u + 0u];
    uint rank = simd_prefix_exclusive_sum(uint(keep));
    uint tot = simd_sum(uint(keep));
    uint base = 0u;
    if (lane == 0 && tot > 0u) base = atomic_fetch_add_explicit(count + m, tot, memory_order_relaxed);
    base = simd_broadcast_first(base);
    if (keep) cands[ulong(m) * vocab + base + rank] = i;
}

kernel void cert_head_setup_batch(device const uint* count [[buffer(0)]],
                                  device uint* indirect [[buffer(1)]],
                                  uint m [[threadgroup_position_in_grid]]) {
    uint c = count[m];
    indirect[m * 4u + 0u] = max((c + 7u) / 8u, 1u);
    indirect[m * 4u + 1u] = 1u;
    indirect[m * 4u + 2u] = 1u;
    indirect[m * 4u + 3u] = 0u;
}

struct TapCopy { uint n4; uint tap_idx; };

kernel void copy_tap_batch(device const float4* src [[buffer(0)]],
                           device float4* dst [[buffer(1)]],
                           constant TapCopy& p [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, m = gid.y;
    if (i < p.n4) dst[(ulong(m) * 4u + p.tap_idx) * p.n4 + i] = src[ulong(m) * p.n4 + i];
}

struct GemvQ4B { uint rows; uint cols; uint ng; uint x_stride; uint y_stride; uint round_out; };

kernel void gemv_q4_f32_rows(device const uint4* q4 [[buffer(0)]],
                             device const half* sc [[buffer(1)]],
                             device const float4* x4 [[buffer(2)]],
                             device float* y [[buffer(3)]],
                             constant GemvQ4B& p [[buffer(4)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint simds [[simdgroups_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= p.rows) return;
    uint xs4 = p.x_stride / 4u;
    device const uint4* qr = q4 + ulong(row) * p.ng;
    device const half* sr = sc + ulong(row) * p.ng;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint g = lane; g < p.ng; g += 32u) {
        uint4 pk = qr[g];
        float s = float(sr[g]);
        uint g8 = g * 8u;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0;
        { float4 w = u4_lo(pk.x); if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + g8]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + g8]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + g8]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + g8]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + g8]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + g8]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + g8]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + g8]); }
        { float4 w = u4_hi(pk.x); uint k = g8 + 1u; if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k]); }
        { float4 w = u4_lo(pk.y); uint k = g8 + 2u; if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k]); }
        { float4 w = u4_hi(pk.y); uint k = g8 + 3u; if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k]); }
        { float4 w = u4_lo(pk.z); uint k = g8 + 4u; if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k]); }
        { float4 w = u4_hi(pk.z); uint k = g8 + 5u; if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k]); }
        { float4 w = u4_lo(pk.w); uint k = g8 + 6u; if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k]); }
        { float4 w = u4_hi(pk.w); uint k = g8 + 7u; if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k]); if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k]); if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k]); if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k]); if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k]); if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k]); if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k]); if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k]); }
        if (ROWS > 0u) a0 += d0 * s;
        if (ROWS > 1u) a1 += d1 * s;
        if (ROWS > 2u) a2 += d2 * s;
        if (ROWS > 3u) a3 += d3 * s;
        if (ROWS > 4u) a4 += d4 * s;
        if (ROWS > 5u) a5 += d5 * s;
        if (ROWS > 6u) a6 += d6 * s;
        if (ROWS > 7u) a7 += d7 * s;
    }
    if (ROWS > 0u) { float t = simd_sum(a0); if (lane == 0) y[0ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd_sum(a1); if (lane == 0) y[1ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd_sum(a2); if (lane == 0) y[2ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd_sum(a3); if (lane == 0) y[3ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd_sum(a4); if (lane == 0) y[4ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd_sum(a5); if (lane == 0) y[5ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd_sum(a6); if (lane == 0) y[6ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd_sum(a7); if (lane == 0) y[7ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
}
