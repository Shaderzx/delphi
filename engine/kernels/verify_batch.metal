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

kernel void gemv_bf12_16_rows(device const uint4* sm_buf [[buffer(0)]],
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
    uint sub = lane >> 4;
    uint sl = lane & 15u;
    uint row = (group * simds + simd) * 2u + sub;
    if (row >= g.rows) return;
    uint chunks16 = g.cols / 16u;
    uint xs4 = g.x_stride / 4u;
    device const uint4* smr = sm_buf + ulong(row) * chunks16;
    device const uint2* exr = ex_buf + ulong(row) * chunks16;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint c = sl; c < chunks16; c += 16u) {
        uint4 sm = smr[c];
        uint2 ex = exr[c];
        uint c4 = c * 4u;
        {
            float4 w = bf12_dec4(sm.x, ex.x & 0xffffu, g.elut);
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + c4]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + c4]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + c4]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + c4]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + c4]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + c4]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + c4]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + c4]);
        }
        {
            float4 w = bf12_dec4(sm.y, ex.x >> 16, g.elut);
            uint k = c4 + 1u;
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + k]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + k]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + k]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + k]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + k]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + k]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + k]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + k]);
        }
        {
            float4 w = bf12_dec4(sm.z, ex.y & 0xffffu, g.elut);
            uint k = c4 + 2u;
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + k]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + k]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + k]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + k]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + k]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + k]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + k]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + k]);
        }
        {
            float4 w = bf12_dec4(sm.w, ex.y >> 16, g.elut);
            uint k = c4 + 3u;
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + k]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + k]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + k]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + k]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + k]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + k]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + k]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + k]);
        }
    }
    if (g.has_outliers) {
        uint s = ro_buf[row];
        uint e = ro_buf[row + 1u];
        device const float* xf = (device const float*)x4;
        for (uint j = s + sl; j < e; j += 16u) {
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
    if (ROWS > 0u) { float t = simd16_sum(a0); if (sl == 0u) y[0ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd16_sum(a1); if (sl == 0u) y[1ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd16_sum(a2); if (sl == 0u) y[2ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd16_sum(a3); if (sl == 0u) y[3ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd16_sum(a4); if (sl == 0u) y[4ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd16_sum(a5); if (sl == 0u) y[5ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd16_sum(a6); if (sl == 0u) y[6ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd16_sum(a7); if (sl == 0u) y[7ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
}

kernel void gemv_bf12_4_rows(device const uint4* sm_buf [[buffer(0)]],
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
    uint sub = lane >> 2;
    uint sl = lane & 3u;
    uint row = (group * simds + simd) * 8u + sub;
    if (row >= g.rows) return;
    uint chunks16 = g.cols / 16u;
    uint xs4 = g.x_stride / 4u;
    device const uint4* smr = sm_buf + ulong(row) * chunks16;
    device const uint2* exr = ex_buf + ulong(row) * chunks16;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint c = sl; c < chunks16; c += 4u) {
        uint4 sm = smr[c];
        uint2 ex = exr[c];
        uint c4 = c * 4u;
        {
            float4 w = bf12_dec4(sm.x, ex.x & 0xffffu, g.elut);
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + c4]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + c4]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + c4]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + c4]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + c4]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + c4]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + c4]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + c4]);
        }
        {
            float4 w = bf12_dec4(sm.y, ex.x >> 16, g.elut);
            uint k = c4 + 1u;
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + k]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + k]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + k]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + k]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + k]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + k]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + k]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + k]);
        }
        {
            float4 w = bf12_dec4(sm.z, ex.y & 0xffffu, g.elut);
            uint k = c4 + 2u;
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + k]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + k]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + k]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + k]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + k]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + k]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + k]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + k]);
        }
        {
            float4 w = bf12_dec4(sm.w, ex.y >> 16, g.elut);
            uint k = c4 + 3u;
            if (ROWS > 0u) a0 += dot(w, x4[0ul * xs4 + k]);
            if (ROWS > 1u) a1 += dot(w, x4[1ul * xs4 + k]);
            if (ROWS > 2u) a2 += dot(w, x4[2ul * xs4 + k]);
            if (ROWS > 3u) a3 += dot(w, x4[3ul * xs4 + k]);
            if (ROWS > 4u) a4 += dot(w, x4[4ul * xs4 + k]);
            if (ROWS > 5u) a5 += dot(w, x4[5ul * xs4 + k]);
            if (ROWS > 6u) a6 += dot(w, x4[6ul * xs4 + k]);
            if (ROWS > 7u) a7 += dot(w, x4[7ul * xs4 + k]);
        }
    }
    if (g.has_outliers) {
        uint s = ro_buf[row];
        uint e = ro_buf[row + 1u];
        device const float* xf = (device const float*)x4;
        for (uint j = s + sl; j < e; j += 4u) {
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
    if (ROWS > 0u) { float t = simd4_sum(a0); if (sl == 0u) y[0ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd4_sum(a1); if (sl == 0u) y[1ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd4_sum(a2); if (sl == 0u) y[2ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd4_sum(a3); if (sl == 0u) y[3ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd4_sum(a4); if (sl == 0u) y[4ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd4_sum(a5); if (sl == 0u) y[5ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd4_sum(a6); if (sl == 0u) y[6ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd4_sum(a7); if (sl == 0u) y[7ul * g.y_stride + row] = g.round_out ? rbf(t) : t; }
}

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

struct ConvB { uint channels; uint dynamic; uint head_dim; uint rows; uint x_stride; uint dyn_stride; uint y_stride; uint is_tree; float scale; uint p1[64]; uint p2[64]; uint p3[64]; };

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
        float th0 = p.is_tree ? state[p.p3[m] * p.channels + c] : h0;
        float th1 = p.is_tree ? state[p.p2[m] * p.channels + c] : h1;
        float th2 = p.is_tree ? state[p.p1[m] * p.channels + c] : h2;
        float acc = 0.0f;
        if (p.dynamic) {
            device const float* dm = dyn + ulong(m) * p.dyn_stride + head * 4u;
            acc += xm * (w0 + dm[0]);
            acc += th2 * (w1 + dm[1]);
            acc += th1 * (w2 + dm[2]);
            acc += th0 * (w3 + dm[3]);
        } else {
            acc = fma(th0, w0, acc);
            acc = fma(th1, w1, acc);
            acc = fma(th2, w2, acc);
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
                            constant uint4& csrc [[buffer(2)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]]) {
    ConvMeta cm = meta[group];
    device float* st = conv_buf + cm.offset_f;
    for (uint c = tid; c < cm.width; c += threads) {
        float s0 = st[csrc.x * cm.width + c];
        float s1 = st[csrc.y * cm.width + c];
        float s2 = st[csrc.z * cm.width + c];
        st[0u * cm.width + c] = s0;
        st[1u * cm.width + c] = s1;
        st[2u * cm.width + c] = s2;
    }
}

struct KvCompactP { uint pos0; uint kv_cap; uint kv_stride; uint count; uint src[16]; };

kernel void kv_compact_tree(device ushort* kc [[buffer(0)]],
                            device ushort* vc [[buffer(1)]],
                            constant KvCompactP& p [[buffer(2)]],
                            uint c [[thread_position_in_grid]]) {
    if (c >= p.kv_stride) return;
    for (uint j = 1u; j < p.count; ++j) {
        uint s = p.src[j];
        if (s != j) {
            ulong sidx = ulong((p.pos0 + s) % p.kv_cap) * p.kv_stride + c;
            ulong didx = ulong((p.pos0 + j) % p.kv_cap) * p.kv_stride + c;
            kc[didx] = kc[sidx];
            vc[didx] = vc[sidx];
        }
    }
}

struct ClipB { uint width; uint has_scale; uint pos0; uint is_local; uint n_floor; uint in_stride; uint out_stride; uint mode; float threshold; float alpha; uint is_tree; uint dep[64]; };

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
    uint qpos = p.pos0 + (p.is_tree ? p.dep[m] : m);
    float tau = p.is_local ? 1.0f : (1.0f + p.alpha * precise::log(max((float(qpos) + 1.0f) / float(p.n_floor), 1.0f)));
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

struct AttnB { uint pos0; uint rows; uint window; uint is_local; uint kv_heads; uint group; uint head_dim; uint extent; uint heads; uint q_stride; uint gate_stride; uint gate_off; uint sink_off; uint out_stride; uint kv_cap; float scale; uint is_tree; uint dep[64]; uint mask[64]; };

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
    uint qp0 = p.pos0 + (p.is_tree ? p.dep[m0] : m0);
    uint qp1 = mb > 1u ? (p.pos0 + (p.is_tree ? p.dep[m0 + 1u] : (m0 + 1u))) : qp0;
    uint qp2 = mb > 2u ? (p.pos0 + (p.is_tree ? p.dep[m0 + 2u] : (m0 + 2u))) : qp0;
    uint qp3 = mb > 3u ? (p.pos0 + (p.is_tree ? p.dep[m0 + 3u] : (m0 + 3u))) : qp0;
    uint min_qp = min(min(qp0, qp1), min(qp2, qp3));
    uint kv_start = (p.is_local && min_qp + 1u > p.window) ? (min_qp + 1u - p.window) : 0u;
    if (!p.is_tree) {
        uint pos_first = p.pos0 + m0;
        uint pos_last = pos_first + mb - 1u;
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
    } else {
        uint qs0 = (p.is_local && qp0 + 1u > p.window) ? (qp0 + 1u - p.window) : 0u;
        uint qs1 = (p.is_local && qp1 + 1u > p.window) ? (qp1 + 1u - p.window) : 0u;
        uint qs2 = (p.is_local && qp2 + 1u > p.window) ? (qp2 + 1u - p.window) : 0u;
        uint qs3 = (p.is_local && qp3 + 1u > p.window) ? (qp3 + 1u - p.window) : 0u;
        uint pcnt = p.pos0 > kv_start ? (p.pos0 - kv_start) : 0u;
        for (uint t = simd; t < pcnt; t += 4u) {
            uint kp = kv_start + t;
            ulong base = (ulong(kp % p.kv_cap) * p.kv_heads + kvh) * 32u + lane;
            ushort4 ku = kc[base];
            float4 k4 = float4(bfv(ku.x), bfv(ku.y), bfv(ku.z), bfv(ku.w));
            ushort4 vu = vc[base];
            float4 v4 = float4(bfv(vu.x), bfv(vu.y), bfv(vu.z), bfv(vu.w));
            if (kp >= qs0) {
                float s = simd_sum(dot(q0, k4)) * p.scale;
                uint dist = qp0 - kp;
                s += dist < p.extent ? rel[(ulong(m0) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv0, s);
                float a = mv0 == -INFINITY ? 0.0f : precise::exp(mv0 - mn);
                float b = precise::exp(s - mn);
                o0 = o0 * a + v4 * b; lv0 = lv0 * a + b; mv0 = mn;
            }
            if (mb > 1u && kp >= qs1) {
                float s = simd_sum(dot(q1, k4)) * p.scale;
                uint dist = qp1 - kp;
                s += dist < p.extent ? rel[(ulong(m0 + 1u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv1, s);
                float a = mv1 == -INFINITY ? 0.0f : precise::exp(mv1 - mn);
                float b = precise::exp(s - mn);
                o1 = o1 * a + v4 * b; lv1 = lv1 * a + b; mv1 = mn;
            }
            if (mb > 2u && kp >= qs2) {
                float s = simd_sum(dot(q2, k4)) * p.scale;
                uint dist = qp2 - kp;
                s += dist < p.extent ? rel[(ulong(m0 + 2u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv2, s);
                float a = mv2 == -INFINITY ? 0.0f : precise::exp(mv2 - mn);
                float b = precise::exp(s - mn);
                o2 = o2 * a + v4 * b; lv2 = lv2 * a + b; mv2 = mn;
            }
            if (mb > 3u && kp >= qs3) {
                float s = simd_sum(dot(q3, k4)) * p.scale;
                uint dist = qp3 - kp;
                s += dist < p.extent ? rel[(ulong(m0 + 3u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv3, s);
                float a = mv3 == -INFINITY ? 0.0f : precise::exp(mv3 - mn);
                float b = precise::exp(s - mn);
                o3 = o3 * a + v4 * b; lv3 = lv3 * a + b; mv3 = mn;
            }
        }
        uint mk0 = p.mask[m0];
        uint mk1 = mb > 1u ? p.mask[m0 + 1u] : 0u;
        uint mk2 = mb > 2u ? p.mask[m0 + 2u] : 0u;
        uint mk3 = mb > 3u ? p.mask[m0 + 3u] : 0u;
        uint d0 = p.dep[m0];
        uint d1 = mb > 1u ? p.dep[m0 + 1u] : 0u;
        uint d2 = mb > 2u ? p.dep[m0 + 2u] : 0u;
        uint d3 = mb > 3u ? p.dep[m0 + 3u] : 0u;
        for (uint k = 0; k < p.rows; ++k) {
            uint true_kp = p.pos0 + p.dep[k];
            if (((true_kp - kv_start) & 3u) != simd) continue;
            uint bit = 1u << k;
            if (((mk0 | mk1 | mk2 | mk3) & bit) == 0u) continue;
            ulong base = (ulong((p.pos0 + k) % p.kv_cap) * p.kv_heads + kvh) * 32u + lane;
            ushort4 ku = kc[base];
            float4 k4 = float4(bfv(ku.x), bfv(ku.y), bfv(ku.z), bfv(ku.w));
            ushort4 vu = vc[base];
            float4 v4 = float4(bfv(vu.x), bfv(vu.y), bfv(vu.z), bfv(vu.w));
            uint dk = p.dep[k];
            if ((mk0 & bit) != 0u && (!p.is_local || true_kp >= qs0)) {
                float s = simd_sum(dot(q0, k4)) * p.scale;
                uint dist = d0 - dk;
                s += dist < p.extent ? rel[(ulong(m0) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv0, s);
                float a = mv0 == -INFINITY ? 0.0f : precise::exp(mv0 - mn);
                float b = precise::exp(s - mn);
                o0 = o0 * a + v4 * b; lv0 = lv0 * a + b; mv0 = mn;
            }
            if ((mk1 & bit) != 0u && (!p.is_local || true_kp >= qs1)) {
                float s = simd_sum(dot(q1, k4)) * p.scale;
                uint dist = d1 - dk;
                s += dist < p.extent ? rel[(ulong(m0 + 1u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv1, s);
                float a = mv1 == -INFINITY ? 0.0f : precise::exp(mv1 - mn);
                float b = precise::exp(s - mn);
                o1 = o1 * a + v4 * b; lv1 = lv1 * a + b; mv1 = mn;
            }
            if ((mk2 & bit) != 0u && (!p.is_local || true_kp >= qs2)) {
                float s = simd_sum(dot(q2, k4)) * p.scale;
                uint dist = d2 - dk;
                s += dist < p.extent ? rel[(ulong(m0 + 2u) * p.heads + h) * p.extent + dist] : 0.0f;
                float mn = max(mv2, s);
                float a = mv2 == -INFINITY ? 0.0f : precise::exp(mv2 - mn);
                float b = precise::exp(s - mn);
                o2 = o2 * a + v4 * b; lv2 = lv2 * a + b; mv2 = mn;
            }
            if ((mk3 & bit) != 0u && (!p.is_local || true_kp >= qs3)) {
                float s = simd_sum(dot(q3, k4)) * p.scale;
                uint dist = d3 - dk;
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
            uint qpos = j == 0u ? qp0 : (j == 1u ? qp1 : (j == 2u ? qp2 : qp3));
            uint qstart = (p.is_local && qpos + 1u > p.window) ? (qpos + 1u - p.window) : 0u;
            uint shift = (qstart - kv_start) & 3u;
            float mx = max(max(sm[shift], sm[(shift + 1u) & 3u]), max(sm[(shift + 2u) & 3u], sm[(shift + 3u) & 3u]));
            float tot = 0.0f;
            float4 acc = 0.0f;
            for (uint s = 0; s < 4u; ++s) {
                uint bs = (s + shift) & 3u;
                if (sm[bs] != -INFINITY) {
                    float w = precise::exp(sm[bs] - mx);
                    tot += sl[bs] * w;
                    acc += so[bs][lane] * w;
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

struct UnionP { uint rows; uint top; uint experts; uint w13_groups; uint w2_groups; uint w13_g5p; uint w2_g5p; };

kernel void build_expert_union(device const uint* ids [[buffer(0)]],
                               device uint* u_exp [[buffer(1)]],
                               device uint* u_cnt [[buffer(2)]],
                               device uint* u_off [[buffer(3)]],
                               device uint* u_pairs [[buffer(4)]],
                               device uint* u_bidx [[buffer(5)]],
                               device uint* ind_w13 [[buffer(6)]],
                               device uint* ind_w2 [[buffer(7)]],
                               device uint* ind_b [[buffer(8)]],
                               constant UnionP& p [[buffer(9)]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint all_ids[1024];
    threadgroup uint stot_u[4];
    threadgroup uint stot_o[4];
    threadgroup uint stot_b1[4];
    threadgroup uint stot_b24[4];
    threadgroup uint stot_b5p[4];
    uint total_mk = p.rows * p.top;
    for (uint i = tid; i < total_mk; i += 128u) all_ids[i] = ids[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint cnt = 0u;
    if (tid < p.experts) {
        for (uint mk = 0; mk < total_mk; ++mk) {
            if (all_ids[mk] == tid) cnt++;
        }
    }
    uint active = cnt > 0u ? 1u : 0u;
    uint is_b1 = cnt == 1u ? 1u : 0u;
    uint is_b24 = (cnt >= 2u && cnt <= 4u) ? 1u : 0u;
    uint is_b5p = cnt >= 5u ? 1u : 0u;
    uint lr_u = simd_prefix_exclusive_sum(active);
    uint lr_o = simd_prefix_exclusive_sum(cnt);
    uint lr_b1 = simd_prefix_exclusive_sum(is_b1);
    uint lr_b24 = simd_prefix_exclusive_sum(is_b24);
    uint lr_b5p = simd_prefix_exclusive_sum(is_b5p);
    uint su = simd_sum(active);
    uint so = simd_sum(cnt);
    uint sb1 = simd_sum(is_b1);
    uint sb24 = simd_sum(is_b24);
    uint sb5p = simd_sum(is_b5p);
    if (lane == 0u) {
        stot_u[simd] = su;
        stot_o[simd] = so;
        stot_b1[simd] = sb1;
        stot_b24[simd] = sb24;
        stot_b5p[simd] = sb5p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint bu = 0u, bo = 0u, bb1 = 0u, bb24 = 0u, bb5p = 0u;
    for (uint s = 0; s < simd; ++s) {
        bu += stot_u[s];
        bo += stot_o[s];
        bb1 += stot_b1[s];
        bb24 += stot_b24[s];
        bb5p += stot_b5p[s];
    }
    uint u = bu + lr_u;
    uint off = bo + lr_o;
    if (active) {
        u_exp[u] = tid;
        u_cnt[u] = cnt;
        u_off[u] = off;
        uint w = off;
        for (uint mk = 0; mk < total_mk; ++mk) {
            if (all_ids[mk] == tid) u_pairs[w++] = mk;
        }
        if (is_b1) u_bidx[bb1 + lr_b1] = u;
        else if (is_b24) u_bidx[128u + bb24 + lr_b24] = u;
        else u_bidx[256u + bb5p + lr_b5p] = u;
    }
    if (tid == 0u) {
        uint total_u = stot_u[0] + stot_u[1] + stot_u[2] + stot_u[3];
        uint n1 = stot_b1[0] + stot_b1[1] + stot_b1[2] + stot_b1[3];
        uint n24 = stot_b24[0] + stot_b24[1] + stot_b24[2] + stot_b24[3];
        uint n5p = stot_b5p[0] + stot_b5p[1] + stot_b5p[2] + stot_b5p[3];
        ind_w13[0] = p.w13_groups; ind_w13[1] = max(total_u, 1u); ind_w13[2] = 1u; ind_w13[3] = total_u;
        ind_w2[0] = p.w2_groups;   ind_w2[1] = max(total_u, 1u);  ind_w2[2] = 1u;  ind_w2[3] = total_u;
        ind_b[0]  = p.w13_groups; ind_b[1]  = max(n1, 1u);  ind_b[2]  = 1u; ind_b[3]  = n1;
        ind_b[4]  = p.w13_groups; ind_b[5]  = max(n24, 1u); ind_b[6]  = 1u; ind_b[7]  = n24;
        ind_b[8]  = p.w13_g5p;    ind_b[9]  = max(n5p, 1u); ind_b[10] = 1u; ind_b[11] = n5p;
        ind_b[12] = p.w2_groups;  ind_b[13] = max(n1, 1u);  ind_b[14] = 1u; ind_b[15] = n1;
        ind_b[16] = p.w2_groups;  ind_b[17] = max(n24, 1u); ind_b[18] = 1u; ind_b[19] = n24;
        ind_b[20] = p.w2_g5p;     ind_b[21] = max(n5p, 1u); ind_b[22] = 1u; ind_b[23] = n5p;
    }
}

struct ExpertG { uint rows; uint cols; uint in_stride; uint out_stride; uint per_row_input; uint top; uint skip_5p; uint pad; ulong expert_bytes; ulong scale_bytes; };

inline void nvfp4_g1_body(device const uchar* w, device const uchar* sc, device const float* amax,
                          device const float* denom, device const half* x, device const float* outer,
                          device const uint* u_exp, device const uint* u_off, device const uint* u_pairs,
                          device const uint* u_bidx, device const uint* ind_slot,
                          device float* y, constant ExpertG& p, uint row, uint bi, uint sl, uint stride_lanes) {
    if (row >= p.rows || bi >= ind_slot[3]) return;
    uint u = u_bidx[bi];
    uint e = u_exp[u];
    uint mk0 = u_pairs[u_off[u]];
    uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
    device const uint2* wr = (device const uint2*)(w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u));
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const half4* x0 = (device const half4*)x + ulong(i0) * (p.in_stride / 4u);
    uint blocks = p.cols / 16u;
    float a0 = 0.0f;
    for (uint b = sl; b < blocks; b += stride_lanes) {
        uint2 pk = wr[b];
        float s = e4m3_decode(sr[b]);
        uint b4 = b * 4u;
        float d = 0.0f;
        { float4 q = float4(fp4x4_h(pk.x) * x0[b4]);            d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.x >> 16) * x0[b4 + 1u]); d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.y) * x0[b4 + 2u]);       d += (q.x + q.y) + (q.z + q.w); }
        { float4 q = float4(fp4x4_h(pk.y >> 16) * x0[b4 + 3u]); d += (q.x + q.y) + (q.z + q.w); }
        a0 += d * s;
    }
    float t0 = stride_lanes == 16u ? simd16_sum(a0) : simd8_sum(a0);
    if (sl == 0u) {
        float r = t0 * outer[i0] * (amax[e] / denom[0]);
        y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r) : r;
    }
}

kernel void expert_nvfp4_g1_16(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                               device const float* amax [[buffer(2)]], device const float* denom [[buffer(3)]],
                               device const half* x [[buffer(4)]], device const float* outer [[buffer(5)]],
                               device const uint* u_exp [[buffer(6)]], device const uint* u_off [[buffer(7)]],
                               device const uint* u_pairs [[buffer(8)]], device const uint* u_bidx [[buffer(9)]],
                               device const uint* ind_slot [[buffer(10)]], device float* y [[buffer(11)]],
                               constant ExpertG& p [[buffer(12)]],
                               uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 2u + (lane >> 4);
    nvfp4_g1_body(w, sc, amax, denom, x, outer, u_exp, u_off, u_pairs, u_bidx, ind_slot, y, p, row, group.y, lane & 15u, 16u);
}

kernel void expert_nvfp4_g1_8(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                              device const float* amax [[buffer(2)]], device const float* denom [[buffer(3)]],
                              device const half* x [[buffer(4)]], device const float* outer [[buffer(5)]],
                              device const uint* u_exp [[buffer(6)]], device const uint* u_off [[buffer(7)]],
                              device const uint* u_pairs [[buffer(8)]], device const uint* u_bidx [[buffer(9)]],
                              device const uint* ind_slot [[buffer(10)]], device float* y [[buffer(11)]],
                              constant ExpertG& p [[buffer(12)]],
                              uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                              uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 4u + (lane >> 3);
    nvfp4_g1_body(w, sc, amax, denom, x, outer, u_exp, u_off, u_pairs, u_bidx, ind_slot, y, p, row, group.y, lane & 7u, 8u);
}

inline void nvfp4_g24_body(device const uchar* w, device const uchar* sc, device const float* amax,
                           device const float* denom, device const half* x, device const float* outer,
                           device const uint* u_exp, device const uint* u_cnt, device const uint* u_off,
                           device const uint* u_pairs, device const uint* u_bidx, device const uint* ind_slot,
                           device float* y, constant ExpertG& p, uint row, uint bi, uint sl, uint stride_lanes) {
    if (row >= p.rows || bi >= ind_slot[3]) return;
    uint u = u_bidx[bi];
    uint e = u_exp[u];
    uint cnt = u_cnt[u];
    uint off = u_off[u];
    device const uint2* wr = (device const uint2*)(w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u));
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const half4* x4 = (device const half4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 16u;
    float alpha = amax[e] / denom[0];
    uint mk0 = u_pairs[off], mk1 = u_pairs[off + 1u];
    uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
    uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
    device const half4* x0 = x4 + ulong(i0) * is4;
    device const half4* x1 = x4 + ulong(i1) * is4;
    if (cnt == 2u) {
        float a0 = 0.0f, a1 = 0.0f;
        for (uint b = sl; b < blocks; b += stride_lanes) {
            uint2 pk = wr[b];
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float d0 = 0.0f, d1 = 0.0f;
            { half4 f = fp4x4_h(pk.x);       float4 q; q = float4(f * x0[b4]);      d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[b4]);      d1 += (q.x + q.y) + (q.z + q.w); }
            { half4 f = fp4x4_h(pk.x >> 16); uint k = b4 + 1u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); }
            { half4 f = fp4x4_h(pk.y);       uint k = b4 + 2u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); }
            { half4 f = fp4x4_h(pk.y >> 16); uint k = b4 + 3u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); }
            a0 += d0 * s;
            a1 += d1 * s;
        }
        float t0 = stride_lanes == 16u ? simd16_sum(a0) : simd8_sum(a0);
        float t1 = stride_lanes == 16u ? simd16_sum(a1) : simd8_sum(a1);
        if (sl == 0u) {
            float r0 = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r0) : r0;
            float r1 = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r1) : r1;
        }
        return;
    }
    uint mk2 = u_pairs[off + 2u];
    uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
    device const half4* x2 = x4 + ulong(i2) * is4;
    if (cnt == 3u) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
        for (uint b = sl; b < blocks; b += stride_lanes) {
            uint2 pk = wr[b];
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
            { half4 f = fp4x4_h(pk.x);       float4 q; q = float4(f * x0[b4]);      d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[b4]);      d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[b4]);      d2 += (q.x + q.y) + (q.z + q.w); }
            { half4 f = fp4x4_h(pk.x >> 16); uint k = b4 + 1u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[k]); d2 += (q.x + q.y) + (q.z + q.w); }
            { half4 f = fp4x4_h(pk.y);       uint k = b4 + 2u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[k]); d2 += (q.x + q.y) + (q.z + q.w); }
            { half4 f = fp4x4_h(pk.y >> 16); uint k = b4 + 3u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[k]); d2 += (q.x + q.y) + (q.z + q.w); }
            a0 += d0 * s;
            a1 += d1 * s;
            a2 += d2 * s;
        }
        float t0 = stride_lanes == 16u ? simd16_sum(a0) : simd8_sum(a0);
        float t1 = stride_lanes == 16u ? simd16_sum(a1) : simd8_sum(a1);
        float t2 = stride_lanes == 16u ? simd16_sum(a2) : simd8_sum(a2);
        if (sl == 0u) {
            float r0 = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r0) : r0;
            float r1 = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r1) : r1;
            float r2 = t2 * outer[i2] * alpha; y[ulong(mk2) * p.out_stride + row] = p.per_row_input ? rbf(r2) : r2;
        }
        return;
    }
    uint mk3 = u_pairs[off + 3u];
    uint i3 = p.per_row_input ? mk3 : (mk3 / p.top);
    device const half4* x3 = x4 + ulong(i3) * is4;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (uint b = sl; b < blocks; b += stride_lanes) {
        uint2 pk = wr[b];
        float s = e4m3_decode(sr[b]);
        uint b4 = b * 4u;
        float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
        { half4 f = fp4x4_h(pk.x);       float4 q; q = float4(f * x0[b4]);      d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[b4]);      d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[b4]);      d2 += (q.x + q.y) + (q.z + q.w); q = float4(f * x3[b4]);      d3 += (q.x + q.y) + (q.z + q.w); }
        { half4 f = fp4x4_h(pk.x >> 16); uint k = b4 + 1u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[k]); d2 += (q.x + q.y) + (q.z + q.w); q = float4(f * x3[k]); d3 += (q.x + q.y) + (q.z + q.w); }
        { half4 f = fp4x4_h(pk.y);       uint k = b4 + 2u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[k]); d2 += (q.x + q.y) + (q.z + q.w); q = float4(f * x3[k]); d3 += (q.x + q.y) + (q.z + q.w); }
        { half4 f = fp4x4_h(pk.y >> 16); uint k = b4 + 3u; float4 q; q = float4(f * x0[k]); d0 += (q.x + q.y) + (q.z + q.w); q = float4(f * x1[k]); d1 += (q.x + q.y) + (q.z + q.w); q = float4(f * x2[k]); d2 += (q.x + q.y) + (q.z + q.w); q = float4(f * x3[k]); d3 += (q.x + q.y) + (q.z + q.w); }
        a0 += d0 * s;
        a1 += d1 * s;
        a2 += d2 * s;
        a3 += d3 * s;
    }
    float t0 = stride_lanes == 16u ? simd16_sum(a0) : simd8_sum(a0);
    float t1 = stride_lanes == 16u ? simd16_sum(a1) : simd8_sum(a1);
    float t2 = stride_lanes == 16u ? simd16_sum(a2) : simd8_sum(a2);
    float t3 = stride_lanes == 16u ? simd16_sum(a3) : simd8_sum(a3);
    if (sl == 0u) {
        float r0 = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r0) : r0;
        float r1 = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r1) : r1;
        float r2 = t2 * outer[i2] * alpha; y[ulong(mk2) * p.out_stride + row] = p.per_row_input ? rbf(r2) : r2;
        float r3 = t3 * outer[i3] * alpha; y[ulong(mk3) * p.out_stride + row] = p.per_row_input ? rbf(r3) : r3;
    }
}

kernel void expert_nvfp4_g24_16(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                                device const float* amax [[buffer(2)]], device const float* denom [[buffer(3)]],
                                device const half* x [[buffer(4)]], device const float* outer [[buffer(5)]],
                                device const uint* u_exp [[buffer(6)]], device const uint* u_cnt [[buffer(7)]],
                                device const uint* u_off [[buffer(8)]], device const uint* u_pairs [[buffer(9)]],
                                device const uint* u_bidx [[buffer(10)]], device const uint* ind_slot [[buffer(11)]],
                                device float* y [[buffer(12)]], constant ExpertG& p [[buffer(13)]],
                                uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 2u + (lane >> 4);
    nvfp4_g24_body(w, sc, amax, denom, x, outer, u_exp, u_cnt, u_off, u_pairs, u_bidx, ind_slot, y, p, row, group.y, lane & 15u, 16u);
}

kernel void expert_nvfp4_g24_8(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                               device const float* amax [[buffer(2)]], device const float* denom [[buffer(3)]],
                               device const half* x [[buffer(4)]], device const float* outer [[buffer(5)]],
                               device const uint* u_exp [[buffer(6)]], device const uint* u_cnt [[buffer(7)]],
                               device const uint* u_off [[buffer(8)]], device const uint* u_pairs [[buffer(9)]],
                               device const uint* u_bidx [[buffer(10)]], device const uint* ind_slot [[buffer(11)]],
                               device float* y [[buffer(12)]], constant ExpertG& p [[buffer(13)]],
                               uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 4u + (lane >> 3);
    nvfp4_g24_body(w, sc, amax, denom, x, outer, u_exp, u_cnt, u_off, u_pairs, u_bidx, ind_slot, y, p, row, group.y, lane & 7u, 8u);
}

inline void nvfp4_g5p_body(device const uchar* w, device const uchar* sc, device const float* amax,
                           device const float* denom, device const half* x, device const float* outer,
                           device const uint* u_exp, device const uint* u_cnt, device const uint* u_off,
                           device const uint* u_pairs, device const uint* u_bidx, device const uint* ind_slot,
                           device float* y, constant ExpertG& p, uint row, uint bi, uint sl, uint stride_lanes) {
    if (row >= p.rows || bi >= ind_slot[3]) return;
    uint u = u_bidx[bi];
    uint e = u_exp[u];
    uint cnt = u_cnt[u];
    uint off = u_off[u];
    device const uint2* wr = (device const uint2*)(w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u));
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const half4* x4 = (device const half4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 16u;
    float alpha = amax[e] / denom[0];
    auto red = [&](float v) { return stride_lanes == 16u ? simd16_sum(v) : simd8_sum(v); };
    for (uint j0 = 0; j0 < cnt; j0 += 8u) {
        uint rem = min(8u, cnt - j0);
        uint mk0 = u_pairs[off + j0];
        uint mk1 = rem > 1u ? u_pairs[off + j0 + 1u] : mk0;
        uint mk2 = rem > 2u ? u_pairs[off + j0 + 2u] : mk0;
        uint mk3 = rem > 3u ? u_pairs[off + j0 + 3u] : mk0;
        uint mk4 = rem > 4u ? u_pairs[off + j0 + 4u] : mk0;
        uint mk5 = rem > 5u ? u_pairs[off + j0 + 5u] : mk0;
        uint mk6 = rem > 6u ? u_pairs[off + j0 + 6u] : mk0;
        uint mk7 = rem > 7u ? u_pairs[off + j0 + 7u] : mk0;
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
        uint i3 = p.per_row_input ? mk3 : (mk3 / p.top);
        uint i4 = p.per_row_input ? mk4 : (mk4 / p.top);
        uint i5 = p.per_row_input ? mk5 : (mk5 / p.top);
        uint i6 = p.per_row_input ? mk6 : (mk6 / p.top);
        uint i7 = p.per_row_input ? mk7 : (mk7 / p.top);
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
        for (uint b = sl; b < blocks; b += stride_lanes) {
            uint2 pk = wr[b];
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0;
            half4 fs[4] = {fp4x4_h(pk.x), fp4x4_h(pk.x >> 16), fp4x4_h(pk.y), fp4x4_h(pk.y >> 16)};
            for (uint t = 0; t < 4u; ++t) {
                half4 f = fs[t];
                uint k = b4 + t;
                float4 q;
                q = float4(f * x4[ulong(i0) * is4 + k]); d0 += (q.x + q.y) + (q.z + q.w);
                if (rem > 1u) { q = float4(f * x4[ulong(i1) * is4 + k]); d1 += (q.x + q.y) + (q.z + q.w); }
                if (rem > 2u) { q = float4(f * x4[ulong(i2) * is4 + k]); d2 += (q.x + q.y) + (q.z + q.w); }
                if (rem > 3u) { q = float4(f * x4[ulong(i3) * is4 + k]); d3 += (q.x + q.y) + (q.z + q.w); }
                if (rem > 4u) { q = float4(f * x4[ulong(i4) * is4 + k]); d4 += (q.x + q.y) + (q.z + q.w); }
                if (rem > 5u) { q = float4(f * x4[ulong(i5) * is4 + k]); d5 += (q.x + q.y) + (q.z + q.w); }
                if (rem > 6u) { q = float4(f * x4[ulong(i6) * is4 + k]); d6 += (q.x + q.y) + (q.z + q.w); }
                if (rem > 7u) { q = float4(f * x4[ulong(i7) * is4 + k]); d7 += (q.x + q.y) + (q.z + q.w); }
            }
            a0 += d0 * s;
            if (rem > 1u) a1 += d1 * s;
            if (rem > 2u) a2 += d2 * s;
            if (rem > 3u) a3 += d3 * s;
            if (rem > 4u) a4 += d4 * s;
            if (rem > 5u) a5 += d5 * s;
            if (rem > 6u) a6 += d6 * s;
            if (rem > 7u) a7 += d7 * s;
        }
        float t0 = red(a0); if (sl == 0u) { float r = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; }
        if (rem > 1u) { float t1 = red(a1); if (sl == 0u) { float r = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 2u) { float t2 = red(a2); if (sl == 0u) { float r = t2 * outer[i2] * alpha; y[ulong(mk2) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 3u) { float t3 = red(a3); if (sl == 0u) { float r = t3 * outer[i3] * alpha; y[ulong(mk3) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 4u) { float t4 = red(a4); if (sl == 0u) { float r = t4 * outer[i4] * alpha; y[ulong(mk4) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 5u) { float t5 = red(a5); if (sl == 0u) { float r = t5 * outer[i5] * alpha; y[ulong(mk5) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 6u) { float t6 = red(a6); if (sl == 0u) { float r = t6 * outer[i6] * alpha; y[ulong(mk6) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 7u) { float t7 = red(a7); if (sl == 0u) { float r = t7 * outer[i7] * alpha; y[ulong(mk7) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
    }
}

kernel void expert_nvfp4_g5p_16(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                                device const float* amax [[buffer(2)]], device const float* denom [[buffer(3)]],
                                device const half* x [[buffer(4)]], device const float* outer [[buffer(5)]],
                                device const uint* u_exp [[buffer(6)]], device const uint* u_cnt [[buffer(7)]],
                                device const uint* u_off [[buffer(8)]], device const uint* u_pairs [[buffer(9)]],
                                device const uint* u_bidx [[buffer(10)]], device const uint* ind_slot [[buffer(11)]],
                                device float* y [[buffer(12)]], constant ExpertG& p [[buffer(13)]],
                                uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 2u + (lane >> 4);
    nvfp4_g5p_body(w, sc, amax, denom, x, outer, u_exp, u_cnt, u_off, u_pairs, u_bidx, ind_slot, y, p, row, group.y, lane & 15u, 16u);
}

kernel void expert_nvfp4_g5p_8(device const uchar* w [[buffer(0)]], device const uchar* sc [[buffer(1)]],
                               device const float* amax [[buffer(2)]], device const float* denom [[buffer(3)]],
                               device const half* x [[buffer(4)]], device const float* outer [[buffer(5)]],
                               device const uint* u_exp [[buffer(6)]], device const uint* u_cnt [[buffer(7)]],
                               device const uint* u_off [[buffer(8)]], device const uint* u_pairs [[buffer(9)]],
                               device const uint* u_bidx [[buffer(10)]], device const uint* ind_slot [[buffer(11)]],
                               device float* y [[buffer(12)]], constant ExpertG& p [[buffer(13)]],
                               uint2 group [[threadgroup_position_in_grid]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint simds [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 4u + (lane >> 3);
    nvfp4_g5p_body(w, sc, amax, denom, x, outer, u_exp, u_cnt, u_off, u_pairs, u_bidx, ind_slot, y, p, row, group.y, lane & 7u, 8u);
}

inline void mxfp8_grouped_body(device const uchar* w, device const uchar* sc, device const float* x,
                               device const uint* u_exp, device const uint* u_cnt, device const uint* u_off, device const uint* u_pairs,
                               device float* y, constant ExpertG& p, uint row, uint u, uint sl, uint stride_lanes) {
    if (row >= p.rows) return;
    uint e = u_exp[u];
    uint cnt = u_cnt[u];
    if (p.skip_5p && cnt >= 5u) return;
    uint off = u_off[u];
    device const uint4* wr4 = (device const uint4*)(w + ulong(e) * p.expert_bytes + ulong(row) * p.cols);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 32u);
    device const float4* x4 = (device const float4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 32u;
    auto red = [&](float v) {
        if (stride_lanes == 8u) return simd8_sum(v);
        return simd4_sum(v);
    };
    for (uint j0 = 0; j0 < cnt; j0 += 4u) {
        uint rem = min(4u, cnt - j0);
        uint mk0 = u_pairs[off + j0];
        uint mk1 = rem > 1u ? u_pairs[off + j0 + 1u] : mk0;
        uint mk2 = rem > 2u ? u_pairs[off + j0 + 2u] : mk0;
        uint mk3 = rem > 3u ? u_pairs[off + j0 + 3u] : mk0;
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
        uint i3 = p.per_row_input ? mk3 : (mk3 / p.top);
        device const float4* x0 = x4 + ulong(i0) * is4;
        device const float4* x1 = x4 + ulong(i1) * is4;
        device const float4* x2 = x4 + ulong(i2) * is4;
        device const float4* x3 = x4 + ulong(i3) * is4;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (uint b = sl; b < blocks; b += stride_lanes) {
            uint4 w0 = wr4[b * 2u];
            uint4 w1 = wr4[b * 2u + 1u];
            float s = as_type<float>(uint(sr[b]) << 23);
            uint b8 = b * 8u;
            uint ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (uint t = 0; t < 8u; ++t) {
                float4 f = e4m3x4(ws[t]);
                uint k = b8 + t;
                d0 += dot(f, x0[k]);
                if (rem > 1u) d1 += dot(f, x1[k]);
                if (rem > 2u) d2 += dot(f, x2[k]);
                if (rem > 3u) d3 += dot(f, x3[k]);
            }
            a0 += d0 * s;
            if (rem > 1u) a1 += d1 * s;
            if (rem > 2u) a2 += d2 * s;
            if (rem > 3u) a3 += d3 * s;
        }
        float t0 = red(a0); if (sl == 0u) y[ulong(mk0) * p.out_stride + row] = rbf(t0);
        if (rem > 1u) { float t1 = red(a1); if (sl == 0u) y[ulong(mk1) * p.out_stride + row] = rbf(t1); }
        if (rem > 2u) { float t2 = red(a2); if (sl == 0u) y[ulong(mk2) * p.out_stride + row] = rbf(t2); }
        if (rem > 3u) { float t3 = red(a3); if (sl == 0u) y[ulong(mk3) * p.out_stride + row] = rbf(t3); }
    }
}

kernel void expert_mxfp8_grouped_8(device const uchar* w [[buffer(0)]],
                                   device const uchar* sc [[buffer(1)]],
                                   device const float* x [[buffer(4)]],
                                   device const uint* u_exp [[buffer(6)]],
                                   device const uint* u_cnt [[buffer(7)]],
                                   device const uint* u_off [[buffer(11)]],
                                   device const uint* u_pairs [[buffer(8)]],
                                   device float* y [[buffer(9)]],
                                   constant ExpertG& p [[buffer(10)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint simds [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 4u + (lane >> 3);
    mxfp8_grouped_body(w, sc, x, u_exp, u_cnt, u_off, u_pairs, y, p, row, group.y, lane & 7u, 8u);
}

kernel void expert_mxfp8_grouped_4(device const uchar* w [[buffer(0)]],
                                   device const uchar* sc [[buffer(1)]],
                                   device const float* x [[buffer(4)]],
                                   device const uint* u_exp [[buffer(6)]],
                                   device const uint* u_cnt [[buffer(7)]],
                                   device const uint* u_off [[buffer(11)]],
                                   device const uint* u_pairs [[buffer(8)]],
                                   device float* y [[buffer(9)]],
                                   constant ExpertG& p [[buffer(10)]],
                                   uint2 group [[threadgroup_position_in_grid]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint simds [[simdgroups_per_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    uint row = (group.x * simds + simd) * 8u + (lane >> 2);
    mxfp8_grouped_body(w, sc, x, u_exp, u_cnt, u_off, u_pairs, y, p, row, group.y, lane & 3u, 4u);
}

kernel void expert_nvfp4_grouped(device const uchar* w [[buffer(0)]],
                                 device const uchar* sc [[buffer(1)]],
                                 device const float* amax [[buffer(2)]],
                                 device const float* denom [[buffer(3)]],
                                 device const half* x [[buffer(4)]],
                                 device const float* outer [[buffer(5)]],
                                 device const uint* u_exp [[buffer(6)]],
                                 device const uint* u_cnt [[buffer(7)]],
                                 device const uint* u_off [[buffer(11)]],
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
    uint off = u_off[u];
    device const uchar* wr = w + ulong(e) * p.expert_bytes + ulong(row) * (p.cols / 2u);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 16u);
    device const half4* x4 = (device const half4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 16u;
    float alpha = amax[e] / denom[0];
    for (uint j0 = 0; j0 < cnt; j0 += 4u) {
        uint rem = min(4u, cnt - j0);
        uint mk0 = u_pairs[off + j0];
        uint mk1 = rem > 1u ? u_pairs[off + j0 + 1u] : mk0;
        uint mk2 = rem > 2u ? u_pairs[off + j0 + 2u] : mk0;
        uint mk3 = rem > 3u ? u_pairs[off + j0 + 3u] : mk0;
        uint i0 = p.per_row_input ? mk0 : (mk0 / p.top);
        uint i1 = p.per_row_input ? mk1 : (mk1 / p.top);
        uint i2 = p.per_row_input ? mk2 : (mk2 / p.top);
        uint i3 = p.per_row_input ? mk3 : (mk3 / p.top);
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (uint b = lane; b < blocks; b += 32u) {
            uint2 packed = *(device const uint2*)(wr + b * 8u);
            float s = e4m3_decode(sr[b]);
            uint b4 = b * 4u;
            float4 p0 = 0, p1 = 0, p2 = 0, p3 = 0;
            half4 fs[4] = {fp4x4_h(packed.x), fp4x4_h(packed.x >> 16), fp4x4_h(packed.y), fp4x4_h(packed.y >> 16)};
            for (uint t = 0; t < 4u; ++t) {
                half4 f = fs[t];
                uint k = b4 + t;
                p0 += float4(f * x4[ulong(i0) * is4 + k]);
                if (rem > 1u) p1 += float4(f * x4[ulong(i1) * is4 + k]);
                if (rem > 2u) p2 += float4(f * x4[ulong(i2) * is4 + k]);
                if (rem > 3u) p3 += float4(f * x4[ulong(i3) * is4 + k]);
            }
            a0 += ((p0.x + p0.y) + (p0.z + p0.w)) * s;
            if (rem > 1u) a1 += ((p1.x + p1.y) + (p1.z + p1.w)) * s;
            if (rem > 2u) a2 += ((p2.x + p2.y) + (p2.z + p2.w)) * s;
            if (rem > 3u) a3 += ((p3.x + p3.y) + (p3.z + p3.w)) * s;
        }
        float t0 = simd_sum(a0); if (lane == 0) { float r = t0 * outer[i0] * alpha; y[ulong(mk0) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; }
        if (rem > 1u) { float t1 = simd_sum(a1); if (lane == 0) { float r = t1 * outer[i1] * alpha; y[ulong(mk1) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 2u) { float t2 = simd_sum(a2); if (lane == 0) { float r = t2 * outer[i2] * alpha; y[ulong(mk2) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
        if (rem > 3u) { float t3 = simd_sum(a3); if (lane == 0) { float r = t3 * outer[i3] * alpha; y[ulong(mk3) * p.out_stride + row] = p.per_row_input ? rbf(r) : r; } }
    }
}

kernel void expert_mxfp8_grouped(device const uchar* w [[buffer(0)]],
                                 device const uchar* sc [[buffer(1)]],
                                 device const float* x [[buffer(4)]],
                                 device const uint* u_exp [[buffer(6)]],
                                 device const uint* u_cnt [[buffer(7)]],
                                 device const uint* u_off [[buffer(11)]],
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
    uint off = u_off[u];
    device const uint4* wr4 = (device const uint4*)(w + ulong(e) * p.expert_bytes + ulong(row) * p.cols);
    device const uchar* sr = sc + ulong(e) * p.scale_bytes + ulong(row) * (p.cols / 32u);
    device const float4* x4 = (device const float4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 32u;
    for (uint j0 = 0; j0 < cnt; j0 += 4u) {
        uint rem = min(4u, cnt - j0);
        uint mk0 = u_pairs[off + j0];
        uint mk1 = rem > 1u ? u_pairs[off + j0 + 1u] : mk0;
        uint mk2 = rem > 2u ? u_pairs[off + j0 + 2u] : mk0;
        uint mk3 = rem > 3u ? u_pairs[off + j0 + 3u] : mk0;
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
            float s = as_type<float>(uint(sr[b]) << 23);
            uint b8 = b * 8u;
            uint ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            for (uint t = 0; t < 8u; ++t) {
                float4 f = e4m3x4(ws[t]);
                uint k = b8 + t;
                d0 += dot(f, x0[k]);
                if (rem > 1u) d1 += dot(f, x1[k]);
                if (rem > 2u) d2 += dot(f, x2[k]);
                if (rem > 3u) d3 += dot(f, x3[k]);
            }
            a0 += d0 * s;
            if (rem > 1u) a1 += d1 * s;
            if (rem > 2u) a2 += d2 * s;
            if (rem > 3u) a3 += d3 * s;
        }
        float t0 = simd_sum(a0); if (lane == 0) y[ulong(mk0) * p.out_stride + row] = rbf(t0);
        if (rem > 1u) { float t1 = simd_sum(a1); if (lane == 0) y[ulong(mk1) * p.out_stride + row] = rbf(t1); }
        if (rem > 2u) { float t2 = simd_sum(a2); if (lane == 0) y[ulong(mk2) * p.out_stride + row] = rbf(t2); }
        if (rem > 3u) { float t3 = simd_sum(a3); if (lane == 0) y[ulong(mk3) * p.out_stride + row] = rbf(t3); }
    }
}

struct CombineB { uint n; uint top; uint boundary; uint rows; uint is_tree; uint p1[64]; uint p2[64]; uint p3[64]; };

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
            float th0 = p.is_tree ? conv_state[p.p3[m] * p.n + c] : h0;
            float th1 = p.is_tree ? conv_state[p.p2[m] * p.n + c] : h1;
            float th2 = p.is_tree ? conv_state[p.p1[m] * p.n + c] : h2;
            float acc = 0.0f;
            acc = fma(th0, w0, acc);
            acc = fma(th1, w1, acc);
            acc = fma(th2, w2, acc);
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
                                  device half4* xh [[buffer(3)]],
                                  constant uint& n [[buffer(4)]],
                                  uint m [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    threadgroup float part_dx[20];
    uint n4 = n / 4u;
    device const float4* xm = x + ulong(m) * n4;
    device half4* xhm = xh + ulong(m) * n4;
    float s = 0.0f;
    float sd = 0.0f;
    for (uint i = tid; i < n4; i += threads) {
        float4 v = xm[i];
        half4 vh = select(half4(v), half4(0.0h), fabs(v) > 65000.0f);
        xhm[i] = vh;
        float4 dv = v - float4(vh);
        s += dot(v, v);
        sd += dot(dv, dv);
    }
    s = simd_sum(s);
    sd = simd_sum(sd);
    if (lane == 0) { part[simd] = s; part_dx[simd] = sd; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float tot = 0.0f;
        float tot_dx = 0.0f;
        for (uint i = 0; i < (threads + 31u) / 32u; ++i) { tot += part[i]; tot_dx += part_dx[i]; }
        state[m * 4u + 2u] = precise::sqrt(tot);
        state[m * 4u + 3u] = precise::sqrt(tot_dx);
        count[m] = 0u;
    }
}

struct CertHeadB { uint vocab; uint cols; uint ng; uint ntg; uint m_base; float cap; };

kernel void cert_head_scan_rows(device const uint4* q4 [[buffer(0)]],
                                device const half* sc [[buffer(1)]],
                                device const float2* err [[buffer(2)]],
                                device const half4* x4 [[buffer(3)]],
                                device const float* state [[buffer(4)]],
                                device float* ub_out [[buffer(5)]],
                                device float* tg_val [[buffer(6)]],
                                device uint* tg_row [[buffer(7)]],
                                constant CertHeadB& p [[buffer(8)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float svals[32];
    threadgroup uint srows[32];
    uint sub = lane >> 3;
    uint sl = lane & 7u;
    uint slot = simd * 4u + sub;
    uint row = (group * simds + simd) * 4u + sub;
    uint xs4 = p.cols / 4u;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    if (row < p.vocab) {
        device const uint4* qr = q4 + ulong(row) * p.ng;
        device const half* sr = sc + ulong(row) * p.ng;
        for (uint g = sl; g < p.ng; g += 8u) {
            uint4 pk = qr[g];
            float s = float(sr[g]);
            uint g8 = g * 8u;
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0;
            uint ws[4] = {pk.x, pk.y, pk.z, pk.w};
            for (uint t = 0; t < 4u; ++t) {
                uint ww = ws[t];
                uint k0 = g8 + t * 2u;
                {
                    float4 w = float4(u4_lo_h_fast(ww));
                    if (ROWS > 0u) d0 += dot(w, float4(x4[ulong(p.m_base + 0u) * xs4 + k0]));
                    if (ROWS > 1u) d1 += dot(w, float4(x4[ulong(p.m_base + 1u) * xs4 + k0]));
                    if (ROWS > 2u) d2 += dot(w, float4(x4[ulong(p.m_base + 2u) * xs4 + k0]));
                    if (ROWS > 3u) d3 += dot(w, float4(x4[ulong(p.m_base + 3u) * xs4 + k0]));
                    if (ROWS > 4u) d4 += dot(w, float4(x4[ulong(p.m_base + 4u) * xs4 + k0]));
                    if (ROWS > 5u) d5 += dot(w, float4(x4[ulong(p.m_base + 5u) * xs4 + k0]));
                    if (ROWS > 6u) d6 += dot(w, float4(x4[ulong(p.m_base + 6u) * xs4 + k0]));
                    if (ROWS > 7u) d7 += dot(w, float4(x4[ulong(p.m_base + 7u) * xs4 + k0]));
                }
                {
                    float4 w = float4(u4_hi_h_fast(ww));
                    uint k1 = k0 + 1u;
                    if (ROWS > 0u) d0 += dot(w, float4(x4[ulong(p.m_base + 0u) * xs4 + k1]));
                    if (ROWS > 1u) d1 += dot(w, float4(x4[ulong(p.m_base + 1u) * xs4 + k1]));
                    if (ROWS > 2u) d2 += dot(w, float4(x4[ulong(p.m_base + 2u) * xs4 + k1]));
                    if (ROWS > 3u) d3 += dot(w, float4(x4[ulong(p.m_base + 3u) * xs4 + k1]));
                    if (ROWS > 4u) d4 += dot(w, float4(x4[ulong(p.m_base + 4u) * xs4 + k1]));
                    if (ROWS > 5u) d5 += dot(w, float4(x4[ulong(p.m_base + 5u) * xs4 + k1]));
                    if (ROWS > 6u) d6 += dot(w, float4(x4[ulong(p.m_base + 6u) * xs4 + k1]));
                    if (ROWS > 7u) d7 += dot(w, float4(x4[ulong(p.m_base + 7u) * xs4 + k1]));
                }
            }
            if (ROWS > 0u) a0 += d0 * s;
            if (ROWS > 1u) a1 += d1 * s;
            if (ROWS > 2u) a2 += d2 * s;
            if (ROWS > 3u) a3 += d3 * s;
            if (ROWS > 4u) a4 += d4 * s;
            if (ROWS > 5u) a5 += d5 * s;
            if (ROWS > 6u) a6 += d6 * s;
            if (ROWS > 7u) a7 += d7 * s;
        }
    }
    float2 er = row < p.vocab ? err[row] : float2(0.0f);
    for (uint m = 0; m < ROWS; ++m) {
        uint gm = p.m_base + m;
        float av = m == 0u ? a0 : (m == 1u ? a1 : (m == 2u ? a2 : (m == 3u ? a3 : (m == 4u ? a4 : (m == 5u ? a5 : (m == 6u ? a6 : a7))))));
        float shat = simd8_sum(av);
        if (sl == 0u) {
            if (row < p.vocab) {
                ub_out[ulong(gm) * p.vocab + row] = shat + state[gm * 4u + 2u] * er.x + state[gm * 4u + 3u] * er.y + 1e-4f;
                svals[slot] = shat;
                srows[slot] = row;
            } else {
                svals[slot] = -INFINITY;
                srows[slot] = 0xffffffffu;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            float bv = lane < (simds * 4u) ? svals[lane] : -INFINITY;
            uint br = lane < (simds * 4u) ? srows[lane] : 0xffffffffu;
            for (uint off = 16u; off > 0u; off >>= 1) {
                float ov = simd_shuffle_xor(bv, off);
                uint oi = simd_shuffle_xor(br, off);
                if (ov > bv || (ov == bv && oi < br)) { bv = ov; br = oi; }
            }
            if (lane == 0u) {
                tg_val[ulong(gm) * p.ntg + group] = bv;
                tg_row[ulong(gm) * p.ntg + group] = br;
            }
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
        if (s_lo <= -p.cap) {
            state[m * 4u + 0u] = -INFINITY;
        } else {
            float u = clamp(s_lo / p.cap, -0.9999f, 0.9999f);
            float z_lo = p.cap * 0.5f * precise::log((1.0f + u) / (1.0f - u));
            state[m * 4u + 0u] = z_lo - fabs(z_lo) * 0.005f - 0.01f;
        }
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

struct CastF16P { uint M; uint M_pad; uint K; uint x_stride; };

kernel void f32_to_f16_rows(device const float* x [[buffer(0)]],
                            device half* y [[buffer(1)]],
                            constant CastF16P& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x, m = gid.y;
    if (k >= p.K || m >= p.M_pad) return;
    y[ulong(m) * p.K + k] = m < p.M ? half(x[ulong(m) * p.x_stride + k]) : half(0);
}

kernel void gemv_q4_h8_rows(device const uint4* q4 [[buffer(0)]],
                            device const half* sc [[buffer(1)]],
                            device const half4* x4 [[buffer(2)]],
                            device float* y [[buffer(3)]],
                            constant GemvQ4B& p [[buffer(4)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint simds [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane >> 3;
    uint sl = lane & 7u;
    uint row = (group * simds + simd) * 4u + sub;
    if (row >= p.rows) return;
    uint xs4 = p.cols / 4u;
    device const uint4* qr = q4 + ulong(row) * p.ng;
    device const half* sr = sc + ulong(row) * p.ng;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint g = sl; g < p.ng; g += 8u) {
        uint4 pk = qr[g];
        float s = float(sr[g]);
        uint g8 = g * 8u;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0;
        uint ws[4] = {pk.x, pk.y, pk.z, pk.w};
        for (uint t = 0; t < 4u; ++t) {
            uint ww = ws[t];
            uint k0 = g8 + t * 2u;
            {
                half4 w = u4_lo_h_fast(ww);
                if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k0]));
                if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k0]));
                if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k0]));
                if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k0]));
                if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k0]));
                if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k0]));
                if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k0]));
                if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k0]));
            }
            {
                half4 w = u4_hi_h_fast(ww);
                uint k1 = k0 + 1u;
                if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k1]));
                if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k1]));
                if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k1]));
                if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k1]));
                if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k1]));
                if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k1]));
                if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k1]));
                if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k1]));
            }
        }
        if (ROWS > 0u) a0 += d0 * s;
        if (ROWS > 1u) a1 += d1 * s;
        if (ROWS > 2u) a2 += d2 * s;
        if (ROWS > 3u) a3 += d3 * s;
        if (ROWS > 4u) a4 += d4 * s;
        if (ROWS > 5u) a5 += d5 * s;
        if (ROWS > 6u) a6 += d6 * s;
        if (ROWS > 7u) a7 += d7 * s;
    }
    if (ROWS > 0u) { float t = simd8_sum(a0); if (sl == 0u) y[0ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd8_sum(a1); if (sl == 0u) y[1ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd8_sum(a2); if (sl == 0u) y[2ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd8_sum(a3); if (sl == 0u) y[3ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd8_sum(a4); if (sl == 0u) y[4ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd8_sum(a5); if (sl == 0u) y[5ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd8_sum(a6); if (sl == 0u) y[6ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd8_sum(a7); if (sl == 0u) y[7ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
}

kernel void gemv_q4_h2_rows(device const uint4* q4 [[buffer(0)]],
                            device const half* sc [[buffer(1)]],
                            device const half4* x4 [[buffer(2)]],
                            device float* y [[buffer(3)]],
                            constant GemvQ4B& p [[buffer(4)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint simds [[simdgroups_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint sub = lane >> 1;
    uint sl = lane & 1u;
    uint row = (group * simds + simd) * 16u + sub;
    if (row >= p.rows) return;
    uint xs4 = p.cols / 4u;
    device const uint4* qr = q4 + ulong(row) * p.ng;
    device const half* sr = sc + ulong(row) * p.ng;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint g = sl; g < p.ng; g += 2u) {
        uint4 pk = qr[g];
        float s = float(sr[g]);
        uint g8 = g * 8u;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0;
        uint ws[4] = {pk.x, pk.y, pk.z, pk.w};
        for (uint t = 0; t < 4u; ++t) {
            uint ww = ws[t];
            uint k0 = g8 + t * 2u;
            {
                half4 w = u4_lo_h_fast(ww);
                if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k0]));
                if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k0]));
                if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k0]));
                if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k0]));
                if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k0]));
                if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k0]));
                if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k0]));
                if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k0]));
            }
            {
                half4 w = u4_hi_h_fast(ww);
                uint k1 = k0 + 1u;
                if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k1]));
                if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k1]));
                if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k1]));
                if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k1]));
                if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k1]));
                if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k1]));
                if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k1]));
                if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k1]));
            }
        }
        if (ROWS > 0u) a0 += d0 * s;
        if (ROWS > 1u) a1 += d1 * s;
        if (ROWS > 2u) a2 += d2 * s;
        if (ROWS > 3u) a3 += d3 * s;
        if (ROWS > 4u) a4 += d4 * s;
        if (ROWS > 5u) a5 += d5 * s;
        if (ROWS > 6u) a6 += d6 * s;
        if (ROWS > 7u) a7 += d7 * s;
    }
    if (ROWS > 0u) { float t = a0 + simd_shuffle_xor(a0, 1u); if (sl == 0u) y[0ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = a1 + simd_shuffle_xor(a1, 1u); if (sl == 0u) y[1ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = a2 + simd_shuffle_xor(a2, 1u); if (sl == 0u) y[2ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = a3 + simd_shuffle_xor(a3, 1u); if (sl == 0u) y[3ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = a4 + simd_shuffle_xor(a4, 1u); if (sl == 0u) y[4ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = a5 + simd_shuffle_xor(a5, 1u); if (sl == 0u) y[5ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = a6 + simd_shuffle_xor(a6, 1u); if (sl == 0u) y[6ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = a7 + simd_shuffle_xor(a7, 1u); if (sl == 0u) y[7ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
}

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 400
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

struct CastBf16P { uint M; uint M_pad; uint K; uint x_stride; };

kernel void f32_to_bf16_rows(device const float* x [[buffer(0)]],
                             device bfloat* y [[buffer(1)]],
                             constant CastBf16P& p [[buffer(2)]],
                             uint2 gid [[thread_position_in_grid]]) {
    uint k = gid.x, m = gid.y;
    if (k >= p.K || m >= p.M_pad) return;
    y[ulong(m) * p.K + k] = m < p.M ? bfloat(x[ulong(m) * p.x_stride + k]) : bfloat(0.0f);
}

struct M4Bf12P { uint M; uint M_pad; uint N; uint K; uint round_out; uint has_outliers; uint elut[16]; };

inline ushort4 bf12_u16_4(uint smw, uint ex16, constant uint* elut) {
    return ushort4(
        ushort(((smw << 8) & 0x8000u) | (smw & 0x7fu) | (elut[ex16 & 15u] >> 16)),
        ushort((smw & 0x8000u) | ((smw >> 8) & 0x7fu) | (elut[(ex16 >> 4) & 15u] >> 16)),
        ushort(((smw >> 8) & 0x8000u) | ((smw >> 16) & 0x7fu) | (elut[(ex16 >> 8) & 15u] >> 16)),
        ushort(((smw >> 16) & 0x8000u) | ((smw >> 24) & 0x7fu) | (elut[(ex16 >> 12) & 15u] >> 16))
    );
}

kernel void m4_bf12_mm32(device const uint4* SM [[buffer(0)]],
                         device const uint2* EX [[buffer(1)]],
                         device bfloat* X [[buffer(2)]],
                         device float* Y [[buffer(3)]],
                         constant M4Bf12P& p [[buffer(4)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat Wtg[64 * 32];
    int n0 = int(tgid.x) * 64;
    int m0 = int(tgid.y) * 32;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tX(X, dextents<int32_t, 2>(int(p.K), int(p.M_pad)));
    tensor<threadgroup bfloat, extents<int32_t, 32, 64>, tensor_inline> tW(Wtg, extents<int32_t, 32, 64>());
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(int(p.N), int(p.M)));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32, 64, 32, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto cT = op.get_destination_cooperative_tensor<decltype(tX.slice<32, 32>(0, 0)), decltype(tW), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    uint chunks16 = p.K / 16u;
    uint r = tid >> 1;
    uint sub16 = tid & 1u;
    uint row = uint(n0) + r;
    for (uint c0 = 0; c0 < chunks16; c0 += 2u) {
        if (row < p.N) {
            uint c = c0 + sub16;
            uint4 sm = SM[ulong(row) * chunks16 + c];
            uint2 ex = EX[ulong(row) * chunks16 + c];
            threadgroup ushort4* dst = (threadgroup ushort4*)(Wtg + r * 32u + sub16 * 16u);
            dst[0] = bf12_u16_4(sm.x, ex.x & 0xffffu, p.elut);
            dst[1] = bf12_u16_4(sm.y, ex.x >> 16, p.elut);
            dst[2] = bf12_u16_4(sm.z, ex.y & 0xffffu, p.elut);
            dst[3] = bf12_u16_4(sm.w, ex.y >> 16, p.elut);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.slice<32, 32>(int(c0 * 16u), m0);
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (p.round_out && !p.has_outliers) {
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = rbf(cT[i]);
    }
    auto sY = tY.slice(n0, m0);
    cT.store(sY);
}

struct OutlierPatchP { uint M; uint N; uint x_stride; uint y_stride; uint round_out; };

kernel void bf12_outlier_patch_rows(device const uint* ro_buf [[buffer(0)]],
                                    device const uint* op_buf [[buffer(1)]],
                                    device const float* xf [[buffer(2)]],
                                    device float* y [[buffer(3)]],
                                    constant OutlierPatchP& p [[buffer(4)]],
                                    uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.x, m = gid.y;
    if (row >= p.N || m >= p.M) return;
    uint s = ro_buf[row];
    uint e = ro_buf[row + 1u];
    float v = y[ulong(m) * p.y_stride + row];
    if (s < e) {
        device const float* xm = xf + ulong(m) * p.x_stride;
        for (uint j = s; j < e; ++j) {
            uint pk = op_buf[j];
            v += bfv(ushort(pk & 0xffffu)) * xm[pk >> 16];
        }
    }
    y[ulong(m) * p.y_stride + row] = p.round_out ? rbf(v) : v;
}

kernel void m4_nvfp4_g5p(device const uchar* w [[buffer(0)]],
                         device const uchar* sc [[buffer(1)]],
                         device const float* amax [[buffer(2)]],
                         device const float* denom [[buffer(3)]],
                         device const half* x [[buffer(4)]],
                         device const float* outer [[buffer(5)]],
                         device const uint* u_exp [[buffer(6)]],
                         device const uint* u_cnt [[buffer(7)]],
                         device const uint* u_off [[buffer(8)]],
                         device const uint* u_pairs [[buffer(9)]],
                         device const uint* u_bidx [[buffer(10)]],
                         device const uint* ind_slot [[buffer(11)]],
                         device float* y [[buffer(12)]],
                         constant ExpertG& p [[buffer(13)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    uint bi = tgid.y;
    if (bi >= ind_slot[3]) return;
    threadgroup half Wtg[64 * 32];
    threadgroup half Xtg[16 * 32];
    threadgroup float Ytg[16 * 64];
    threadgroup uint mks[16];
    threadgroup uint ins[16];
    uint u = u_bidx[bi];
    uint e = u_exp[u];
    uint cnt = u_cnt[u];
    uint off = u_off[u];
    uint n0 = tgid.x * 64u;
    device const uchar* we = w + ulong(e) * p.expert_bytes;
    device const uchar* se = sc + ulong(e) * p.scale_bytes;
    device const half4* x4 = (device const half4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 16u;
    float alpha = amax[e] / denom[0];
    tensor<threadgroup half, extents<int32_t, 32, 16>, tensor_inline> tX(Xtg, extents<int32_t, 32, 16>());
    tensor<threadgroup half, extents<int32_t, 32, 64>, tensor_inline> tW(Wtg, extents<int32_t, 32, 64>());
    tensor<threadgroup float, extents<int32_t, 64, 16>, tensor_inline> tY(Ytg, extents<int32_t, 64, 16>());
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 64, 32, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    uint wr = tid >> 1;
    uint sub16 = tid & 1u;
    uint row = n0 + wr;
    uint xr = tid >> 3;
    uint xc4 = tid & 7u;
    for (uint j0 = 0; j0 < cnt; j0 += 16u) {
        uint rem = min(16u, cnt - j0);
        if (tid < rem) {
            uint mk = u_pairs[off + j0 + tid];
            mks[tid] = mk;
            ins[tid] = p.per_row_input ? mk : (mk / p.top);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto cT = op.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        for (uint b0 = 0; b0 < blocks; b0 += 2u) {
            if (row < p.rows) {
                uint b = b0 + sub16;
                uint2 pk = *(device const uint2*)(we + ulong(row) * (p.cols / 2u) + b * 8u);
                half s = half(e4m3_decode(se[ulong(row) * blocks + b]));
                threadgroup half4* dst = (threadgroup half4*)(Wtg + wr * 32u + sub16 * 16u);
                dst[0] = fp4x4_h(pk.x) * s;
                dst[1] = fp4x4_h(pk.x >> 16) * s;
                dst[2] = fp4x4_h(pk.y) * s;
                dst[3] = fp4x4_h(pk.y >> 16) * s;
            }
            threadgroup half4* xdst = (threadgroup half4*)(Xtg + xr * 32u);
            xdst[xc4] = xr < rem ? x4[ulong(ins[xr]) * is4 + b0 * 4u + xc4] : half4(0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(tX, tW, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        cT.store(tY);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rem * 64u; idx += 128u) {
            uint j = idx >> 6;
            uint c = idx & 63u;
            uint out_r = n0 + c;
            if (out_r < p.rows) {
                float r = Ytg[idx] * outer[ins[j]] * alpha;
                y[ulong(mks[j]) * p.out_stride + out_r] = p.per_row_input ? rbf(r) : r;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void m4_mxfp8_g5p(device const uchar* w [[buffer(0)]],
                         device const uchar* sc [[buffer(1)]],
                         device const float* x [[buffer(4)]],
                         device const uint* u_exp [[buffer(6)]],
                         device const uint* u_cnt [[buffer(7)]],
                         device const uint* u_off [[buffer(8)]],
                         device const uint* u_pairs [[buffer(9)]],
                         device const uint* u_bidx [[buffer(10)]],
                         device const uint* ind_slot [[buffer(11)]],
                         device float* y [[buffer(12)]],
                         constant ExpertG& p [[buffer(13)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    uint bi = tgid.y;
    if (bi >= ind_slot[3]) return;
    threadgroup bfloat Wtg[64 * 32];
    threadgroup bfloat Xtg[16 * 32];
    threadgroup float Ytg[16 * 64];
    threadgroup uint mks[16];
    threadgroup uint ins[16];
    uint u = u_bidx[bi];
    uint e = u_exp[u];
    uint cnt = u_cnt[u];
    uint off = u_off[u];
    uint n0 = tgid.x * 64u;
    device const uchar* we = w + ulong(e) * p.expert_bytes;
    device const uchar* se = sc + ulong(e) * p.scale_bytes;
    device const float4* x4 = (device const float4*)x;
    uint is4 = p.in_stride / 4u;
    uint blocks = p.cols / 32u;
    tensor<threadgroup bfloat, extents<int32_t, 32, 16>, tensor_inline> tX(Xtg, extents<int32_t, 32, 16>());
    tensor<threadgroup bfloat, extents<int32_t, 32, 64>, tensor_inline> tW(Wtg, extents<int32_t, 32, 64>());
    tensor<threadgroup float, extents<int32_t, 64, 16>, tensor_inline> tY(Ytg, extents<int32_t, 64, 16>());
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 64, 32, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    uint wr = tid >> 1;
    uint sub16 = tid & 1u;
    uint row = n0 + wr;
    uint xr = tid >> 3;
    uint xc4 = tid & 7u;
    for (uint j0 = 0; j0 < cnt; j0 += 16u) {
        uint rem = min(16u, cnt - j0);
        if (tid < rem) {
            uint mk = u_pairs[off + j0 + tid];
            mks[tid] = mk;
            ins[tid] = p.per_row_input ? mk : (mk / p.top);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto cT = op.get_destination_cooperative_tensor<decltype(tX), decltype(tW), float>();
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
        for (uint b = 0; b < blocks; ++b) {
            if (row < p.rows) {
                uint4 pk = *(device const uint4*)(we + ulong(row) * p.cols + b * 32u + sub16 * 16u);
                float s = as_type<float>(uint(se[ulong(row) * blocks + b]) << 23);
                threadgroup bfloat4* dst = (threadgroup bfloat4*)(Wtg + wr * 32u + sub16 * 16u);
                dst[0] = bfloat4(e4m3x4(pk.x) * s);
                dst[1] = bfloat4(e4m3x4(pk.y) * s);
                dst[2] = bfloat4(e4m3x4(pk.z) * s);
                dst[3] = bfloat4(e4m3x4(pk.w) * s);
            }
            threadgroup bfloat4* xdst = (threadgroup bfloat4*)(Xtg + xr * 32u);
            xdst[xc4] = xr < rem ? bfloat4(x4[ulong(ins[xr]) * is4 + b * 8u + xc4]) : bfloat4(bfloat(0.0f));
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(tX, tW, cT);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        cT.store(tY);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = tid; idx < rem * 64u; idx += 128u) {
            uint j = idx >> 6;
            uint c = idx & 63u;
            uint out_r = n0 + c;
            if (out_r < p.rows) y[ulong(mks[j]) * p.out_stride + out_r] = rbf(Ytg[idx]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

struct M4Q4P { uint M; uint M_pad; uint N; uint K; uint ng; uint round_out; };

kernel void m4_q4_mm32(device const uint4* q4 [[buffer(0)]],
                       device const half* sc [[buffer(1)]],
                       device bfloat* X [[buffer(2)]],
                       device float* Y [[buffer(3)]],
                       constant M4Q4P& p [[buffer(4)]],
                       uint2 tgid [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bfloat Wtg[64 * 32];
    int n0 = int(tgid.x) * 64;
    int m0 = int(tgid.y) * 32;
    tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tX(X, dextents<int32_t, 2>(int(p.K), int(p.M_pad)));
    tensor<threadgroup bfloat, extents<int32_t, 32, 64>, tensor_inline> tW(Wtg, extents<int32_t, 32, 64>());
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(Y, dextents<int32_t, 2>(int(p.N), int(p.M)));
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(32, 64, 32, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto cT = op.get_destination_cooperative_tensor<decltype(tX.slice<32, 32>(0, 0)), decltype(tW), float>();
    #pragma clang loop unroll(full)
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = 0.0f;
    uint r = tid >> 1;
    uint sub16 = tid & 1u;
    uint row = uint(n0) + r;
    for (uint g = 0; g < p.ng; ++g) {
        if (row < p.N) {
            uint4 pk = q4[ulong(row) * p.ng + g];
            float s = float(sc[ulong(row) * p.ng + g]);
            uint w0 = sub16 ? pk.z : pk.x;
            uint w1 = sub16 ? pk.w : pk.y;
            threadgroup bfloat4* dst = (threadgroup bfloat4*)(Wtg + r * 32u + sub16 * 16u);
            dst[0] = bfloat4(float4(u4_lo_h_fast(w0)) * s);
            dst[1] = bfloat4(float4(u4_hi_h_fast(w0)) * s);
            dst[2] = bfloat4(float4(u4_lo_h_fast(w1)) * s);
            dst[3] = bfloat4(float4(u4_hi_h_fast(w1)) * s);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto sX = tX.slice<32, 32>(int(g * 32u), m0);
        op.run(sX, tW, cT);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (p.round_out) {
        #pragma clang loop unroll(full)
        for (uint16_t i = 0; i < cT.get_capacity(); ++i) if (cT.is_valid_element(i)) cT[i] = rbf(cT[i]);
    }
    auto sY = tY.slice(n0, m0);
    cT.store(sY);
}
#endif

struct FanConvMeta { uint state_off; uint hist_off; uint width; uint pad; };
struct FanConvP { uint branches; uint hist_stride; uint rows; uint pad; };

kernel void fan_conv_init(device float* conv_buf [[buffer(0)]],
                          device float* hist [[buffer(1)]],
                          device const FanConvMeta* meta [[buffer(2)]],
                          constant FanConvP& p [[buffer(3)]],
                          uint2 group [[threadgroup_position_in_grid]],
                          uint2 tid2 [[thread_position_in_threadgroup]],
                          uint2 threads2 [[threads_per_threadgroup]]) {
    uint tid = tid2.x;
    uint threads = threads2.x;
    FanConvMeta cm = meta[group.x];
    device const float* st = conv_buf + cm.state_off;
    device float* h = hist + ulong(group.y) * p.hist_stride + cm.hist_off;
    for (uint i = tid; i < 3u * cm.width; i += threads) h[i] = st[i];
}

kernel void fan_conv_stage(device float* conv_buf [[buffer(0)]],
                           device float* hist [[buffer(1)]],
                           device const FanConvMeta* meta [[buffer(2)]],
                           constant FanConvP& p [[buffer(3)]],
                           uint2 group [[threadgroup_position_in_grid]],
                           uint2 tid2 [[thread_position_in_threadgroup]],
                           uint2 threads2 [[threads_per_threadgroup]]) {
    uint tid = tid2.x;
    uint threads = threads2.x;
    FanConvMeta cm = meta[group.x];
    uint b = group.y;
    device float* st = conv_buf + cm.state_off + ulong(3u + p.rows + 3u * b) * cm.width;
    device const float* h = hist + ulong(b) * p.hist_stride + cm.hist_off;
    for (uint i = tid; i < 3u * cm.width; i += threads) st[i] = h[i];
}

kernel void fan_conv_commit(device float* conv_buf [[buffer(0)]],
                            device float* hist [[buffer(1)]],
                            device const FanConvMeta* meta [[buffer(2)]],
                            constant FanConvP& p [[buffer(3)]],
                            uint2 group [[threadgroup_position_in_grid]],
                            uint2 tid2 [[thread_position_in_threadgroup]],
                            uint2 threads2 [[threads_per_threadgroup]]) {
    uint tid = tid2.x;
    uint threads = threads2.x;
    FanConvMeta cm = meta[group.x];
    uint b = group.y;
    device const float* cur = conv_buf + cm.state_off + ulong(3u + b) * cm.width;
    device float* h = hist + ulong(b) * p.hist_stride + cm.hist_off;
    for (uint c = tid; c < cm.width; c += threads) {
        float older = h[cm.width + c];
        float newer = h[2u * cm.width + c];
        h[c] = older;
        h[cm.width + c] = newer;
        h[2u * cm.width + c] = cur[c];
    }
}

struct AttnFan { uint prefix; uint stride; uint qpos; uint rows; uint window; uint is_local; uint kv_heads; uint group; uint extent; uint heads; uint q_stride; uint gate_stride; uint gate_off; uint sink_off; uint out_stride; uint kv_cap; float scale; };

kernel void attend_gated_fan(device const float4* q [[buffer(0)]],
                             device const ushort4* kc [[buffer(1)]],
                             device const ushort4* vc [[buffer(2)]],
                             device const float* rel [[buffer(3)]],
                             device const float* qbundle [[buffer(4)]],
                             device float4* gated [[buffer(5)]],
                             constant AttnFan& p [[buffer(6)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[4][4];
    threadgroup float sl[4][4];
    threadgroup float4 so[4][4][32];
    uint kvh = group.x;
    uint m = group.y;
    uint heads_here = min(p.group, 4u);
    float4 qv[4];
    float mv[4];
    float lv[4];
    float4 o[4];
    for (uint g = 0; g < heads_here; ++g) {
        uint h = kvh * p.group + g;
        qv[g] = q[ulong(m) * (p.q_stride / 4u) + h * 32u + lane];
        mv[g] = -INFINITY;
        lv[g] = 0.0f;
        o[g] = 0.0f;
    }
    uint qp = p.qpos;
    uint qstart = (p.is_local && qp + 1u > p.window) ? (qp + 1u - p.window) : 0u;
    uint count = qp - qstart + 1u;
    for (uint t = simd; t < count; t += 4u) {
        uint kp = qstart + t;
        uint phys = kp < p.prefix ? kp : (p.prefix + (kp - p.prefix) * p.stride + m);
        ulong base = (ulong(phys % p.kv_cap) * p.kv_heads + kvh) * 32u + lane;
        ushort4 ku = kc[base];
        float4 k4 = float4(bfv(ku.x), bfv(ku.y), bfv(ku.z), bfv(ku.w));
        ushort4 vu = vc[base];
        float4 v4 = float4(bfv(vu.x), bfv(vu.y), bfv(vu.z), bfv(vu.w));
        uint dist = qp - kp;
        for (uint g = 0; g < heads_here; ++g) {
            uint h = kvh * p.group + g;
            float s = simd_sum(dot(qv[g], k4)) * p.scale;
            s += dist < p.extent ? rel[(ulong(m) * p.heads + h) * p.extent + dist] : 0.0f;
            float mn = max(mv[g], s);
            float a = mv[g] == -INFINITY ? 0.0f : precise::exp(mv[g] - mn);
            float b = precise::exp(s - mn);
            o[g] = o[g] * a + v4 * b; lv[g] = lv[g] * a + b; mv[g] = mn;
        }
    }
    for (uint g = 0; g < heads_here; ++g) {
        if (lane == 0) { sm[g][simd] = mv[g]; sl[g][simd] = lv[g]; }
        so[g][simd][lane] = o[g];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd < heads_here) {
        uint g = simd;
        uint h = kvh * p.group + g;
        float mx = max(max(sm[g][0], sm[g][1]), max(sm[g][2], sm[g][3]));
        float tot = 0.0f;
        float4 acc = 0.0f;
        for (uint s = 0; s < 4u; ++s) {
            if (sm[g][s] != -INFINITY) {
                float w = precise::exp(sm[g][s] - mx);
                tot += sl[g][s] * w;
                acc += so[g][s][lane] * w;
            }
        }
        float lse = mx + precise::log(tot);
        float sink = qbundle[ulong(m) * p.gate_stride + p.sink_off + h];
        float ssink = sigm(lse - sink);
        device const float4* gptr = (device const float4*)(qbundle + ulong(m) * p.gate_stride + p.gate_off);
        float4 g4 = gptr[h * 32u + lane];
        float4 sg = float4(sigm(g4.x), sigm(g4.y), sigm(g4.z), sigm(g4.w));
        float4 y4 = (acc / tot) * sg * ssink;
        gated[ulong(m) * (p.out_stride / 4u) + h * 32u + lane] = float4(rbf(y4.x), rbf(y4.y), rbf(y4.z), rbf(y4.w));
    }
}
