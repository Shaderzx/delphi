struct MtpNormP { uint vocab; uint rank; };

kernel void mtp_w1_norms(device const ushort* w1 [[buffer(0)]],
                         device float* norms [[buffer(1)]],
                         constant MtpNormP& p [[buffer(2)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint simds [[simdgroups_per_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= p.vocab) return;
    device const ushort* wr = w1 + ulong(row) * p.rank;
    float s = 0.0f;
    for (uint i = lane; i < p.rank; i += 32u) {
        float v = bfv(wr[i]);
        s += v * v;
    }
    float t = simd_sum(s);
    if (lane == 0u) norms[row] = t;
}

struct MtpPackHeadP { uint dvocab; uint ng; };

kernel void mtp_pack_head_sub(device const uint4* src_q4 [[buffer(0)]],
                              device const half* src_sc [[buffer(1)]],
                              device const uint* vocab_ids [[buffer(2)]],
                              device uint4* dst_q4 [[buffer(3)]],
                              device half* dst_sc [[buffer(4)]],
                              constant MtpPackHeadP& p [[buffer(5)]],
                              uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint threads [[threads_per_threadgroup]]) {
    if (row >= p.dvocab) return;
    uint orig = vocab_ids[row];
    for (uint g = tid; g < p.ng; g += threads) {
        dst_q4[ulong(row) * p.ng + g] = src_q4[ulong(orig) * p.ng + g];
        dst_sc[ulong(row) * p.ng + g] = src_sc[ulong(orig) * p.ng + g];
    }
}

struct MtpQ4Build { uint rows; uint cols; uint ng; uint row_off; };

kernel void mtp_quant_q4(device const ushort* src [[buffer(0)]],
                         device uint4* q4 [[buffer(1)]],
                         device half* sc [[buffer(2)]],
                         constant MtpQ4Build& p [[buffer(3)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]]) {
    if (row >= p.rows) return;
    ulong dst_row = ulong(p.row_off + row) * p.ng;
    device const ushort* wr = src + ulong(row) * p.cols;
    for (uint g = tid; g < p.ng; g += threads) {
        float v[32];
        float amax = 0.0f;
        uint base = g * 32u;
        for (uint j = 0; j < 32u; ++j) {
            v[j] = bfv(wr[base + j]);
            amax = max(amax, fabs(v[j]));
        }
        float s = amax > 0.0f ? amax / 7.0f : 1.0f;
        half sh = half(s);
        float sf = float(sh);
        float inv = sf > 0.0f ? 1.0f / sf : 0.0f;
        uint words[4] = {0u, 0u, 0u, 0u};
        for (uint j = 0; j < 32u; ++j) {
            int q = int(clamp(rint(v[j] * inv), -7.0f, 7.0f));
            words[j >> 3] |= uint(q + 8) << ((j & 7u) * 4u);
        }
        q4[dst_row + g] = uint4(words[0], words[1], words[2], words[3]);
        sc[dst_row + g] = sh;
    }
}

struct MtpFoldP { uint dvocab; uint cols; uint rank; };

kernel void mtp_build_fold(device const ushort* unembed [[buffer(0)]],
                           device const ushort* proj [[buffer(1)]],
                           device const uint* vocab_ids [[buffer(2)]],
                           device uint4* fold_q4 [[buffer(3)]],
                           device half* fold_sc [[buffer(4)]],
                           constant MtpFoldP& p [[buffer(5)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half Us[8][24];
    threadgroup half Ps[16][256];
    threadgroup float Out[8][256];
    uint v0 = group * 8u;
    uint tid = simd * 32u + lane;
    simdgroup_float8x8 a0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 a1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 a2 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_float8x8 a3 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    uint ur = tid >> 4;
    uint uc = tid & 15u;
    uint orig_v = (v0 + ur < p.dvocab) ? vocab_ids[v0 + ur] : 0u;
    for (uint k0 = 0; k0 < p.cols; k0 += 16u) {
        if (tid < 128u) {
            Us[ur][uc] = (v0 + ur < p.dvocab) ? half(bfv(unembed[ulong(orig_v) * p.cols + k0 + uc])) : half(0);
        }
        for (uint row = 0; row < 16u; ++row) {
            Ps[row][tid] = half(bfv(proj[ulong(k0 + row) * p.rank + tid]));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint rbase = simd * 32u;
        for (uint kk = 0; kk < 16u; kk += 8u) {
            simdgroup_half8x8 mu, mp0, mp1, mp2, mp3;
            simdgroup_load(mu, &Us[0][kk], 24);
            simdgroup_load(mp0, &Ps[kk][rbase + 0u], 256);
            simdgroup_load(mp1, &Ps[kk][rbase + 8u], 256);
            simdgroup_load(mp2, &Ps[kk][rbase + 16u], 256);
            simdgroup_load(mp3, &Ps[kk][rbase + 24u], 256);
            simdgroup_multiply_accumulate(a0, mu, mp0, a0);
            simdgroup_multiply_accumulate(a1, mu, mp1, a1);
            simdgroup_multiply_accumulate(a2, mu, mp2, a2);
            simdgroup_multiply_accumulate(a3, mu, mp3, a3);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint rbase = simd * 32u;
    simdgroup_store(a0, &Out[0][rbase + 0u], 256);
    simdgroup_store(a1, &Out[0][rbase + 8u], 256);
    simdgroup_store(a2, &Out[0][rbase + 16u], 256);
    simdgroup_store(a3, &Out[0][rbase + 24u], 256);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64u) {
        uint lr = tid >> 3;
        uint g = tid & 7u;
        uint vr = v0 + lr;
        if (vr < p.dvocab) {
            uint rb = g * 32u;
            float amax = 0.0f;
            for (uint j = 0; j < 32u; ++j) amax = max(amax, fabs(Out[lr][rb + j]));
            float s = amax > 0.0f ? amax / 7.0f : 1.0f;
            half sh = half(s);
            float sf = float(sh);
            float inv = sf > 0.0f ? 1.0f / sf : 0.0f;
            uint words[4] = {0u, 0u, 0u, 0u};
            for (uint j = 0; j < 32u; ++j) {
                int q = int(clamp(rint(Out[lr][rb + j] * inv), -7.0f, 7.0f));
                words[j >> 3] |= uint(q + 8) << ((j & 7u) * 4u);
            }
            fold_q4[ulong(vr) * 8u + g] = uint4(words[0], words[1], words[2], words[3]);
            fold_sc[ulong(vr) * 8u + g] = sh;
        }
    }
}

struct MtpBiasP { uint dvocab; uint cols; };

kernel void mtp_build_fold_bias(device const uint4* unembed [[buffer(0)]],
                                device const ushort* bias [[buffer(1)]],
                                device const uint* vocab_ids [[buffer(2)]],
                                device float* fold_b [[buffer(3)]],
                                constant MtpBiasP& p [[buffer(4)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint simds [[simdgroups_per_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= p.dvocab) return;
    uint orig = vocab_ids[row];
    uint chunks = p.cols / 8u;
    device const uint4* wr = unembed + ulong(orig) * chunks;
    float acc = 0.0f;
    for (uint c = lane; c < chunks; c += 32u) {
        uint4 pk = wr[c];
        uint base = c * 8u;
        acc += as_type<float>(pk.x << 16) * bfv(bias[base + 0u])
             + as_type<float>(pk.x & 0xffff0000u) * bfv(bias[base + 1u])
             + as_type<float>(pk.y << 16) * bfv(bias[base + 2u])
             + as_type<float>(pk.y & 0xffff0000u) * bfv(bias[base + 3u])
             + as_type<float>(pk.z << 16) * bfv(bias[base + 4u])
             + as_type<float>(pk.z & 0xffff0000u) * bfv(bias[base + 5u])
             + as_type<float>(pk.w << 16) * bfv(bias[base + 6u])
             + as_type<float>(pk.w & 0xffff0000u) * bfv(bias[base + 7u]);
    }
    float t = simd_sum(acc);
    if (lane == 0u) fold_b[row] = t;
}

struct MtpMixP { uint hidden; uint alpha_row; uint row_start; float eps; };

kernel void mtp_mix_rmsnorm_h(device const float4* taps [[buffer(0)]],
                              device const float* alpha [[buffer(1)]],
                              device const ushort* nw [[buffer(2)]],
                              device half4* out [[buffer(3)]],
                              constant MtpMixP& p [[buffer(4)]],
                              uint m [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint threads [[threads_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = p.hidden / 4u;
    uint sm = p.row_start + m;
    float a0 = alpha[p.alpha_row * 4u + 0u];
    float a1 = alpha[p.alpha_row * 4u + 1u];
    float a2 = alpha[p.alpha_row * 4u + 2u];
    float a3 = alpha[p.alpha_row * 4u + 3u];
    device const float4* t0 = taps + (ulong(sm) * 4u + 0u) * n4;
    device const float4* t1 = taps + (ulong(sm) * 4u + 1u) * n4;
    device const float4* t2 = taps + (ulong(sm) * 4u + 2u) * n4;
    device const float4* t3 = taps + (ulong(sm) * 4u + 3u) * n4;
    device half4* om = out + ulong(m) * n4;
    float4 v = tid < n4 ? (t0[tid] * a0 + t1[tid] * a1 + t2[tid] * a2 + t3[tid] * a3) : float4(0.0f);
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 vt = t0[i] * a0 + t1[i] * a1 + t2[i] * a2 + t3[i] * a3;
        s += dot(vt, vt);
    }
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) tot += part[i];
    float r = 1.0f / precise::sqrt(tot / float(p.hidden) + p.eps);
    if (tid < n4) {
        uint b = tid * 4u;
        float4 o = v * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        om[tid] = half4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 vt = t0[i] * a0 + t1[i] * a1 + t2[i] * a2 + t3[i] * a3;
        uint b = i * 4u;
        float4 o = vt * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        om[i] = half4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
}

kernel void mtp_rmsnorm_h(device const float4* x [[buffer(0)]],
                          device const ushort* nw [[buffer(1)]],
                          device half4* y [[buffer(2)]],
                          constant Norm& p [[buffer(3)]],
                          uint m [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = p.n / 4u;
    device const float4* xm = x + ulong(m) * n4;
    device half4* ym = y + ulong(m) * n4;
    float4 v = tid < n4 ? xm[tid] : float4(0.0f);
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 vt = xm[i];
        s += dot(vt, vt);
    }
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) tot += part[i];
    float r = 1.0f / precise::sqrt(tot / float(p.n) + p.eps);
    if (tid < n4) {
        uint b = tid * 4u;
        float4 o = v * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        ym[tid] = half4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        uint b = i * 4u;
        float4 o = xm[i] * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        ym[i] = half4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
}

inline half4 u4_lo_h(uint word) {
    return half4(half(int(word & 15u) - 8), half(int((word >> 4) & 15u) - 8),
                 half(int((word >> 8) & 15u) - 8), half(int((word >> 12) & 15u) - 8));
}

inline half4 u4_hi_h(uint word) {
    return half4(half(int((word >> 16) & 15u) - 8), half(int((word >> 20) & 15u) - 8),
                 half(int((word >> 24) & 15u) - 8), half(int((word >> 28) & 15u) - 8));
}

struct MtpGemvQ4P { uint rows; uint cols; uint ng; uint x_stride; uint y_stride; uint round_out; uint has_bias; };

kernel void mtp_gemv_q4_rows(device const uint4* q4 [[buffer(0)]],
                             device const half* sc [[buffer(1)]],
                             device const half4* x4 [[buffer(2)]],
                             device const ushort* bias [[buffer(3)]],
                             device float* y [[buffer(4)]],
                             constant MtpGemvQ4P& p [[buffer(5)]],
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
        { half4 w = u4_lo_h(pk.x); if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + g8])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + g8])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + g8])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + g8])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + g8])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + g8])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + g8])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + g8])); }
        { half4 w = u4_hi_h(pk.x); uint k = g8 + 1u; if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k])); }
        { half4 w = u4_lo_h(pk.y); uint k = g8 + 2u; if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k])); }
        { half4 w = u4_hi_h(pk.y); uint k = g8 + 3u; if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k])); }
        { half4 w = u4_lo_h(pk.z); uint k = g8 + 4u; if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k])); }
        { half4 w = u4_hi_h(pk.z); uint k = g8 + 5u; if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k])); }
        { half4 w = u4_lo_h(pk.w); uint k = g8 + 6u; if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k])); }
        { half4 w = u4_hi_h(pk.w); uint k = g8 + 7u; if (ROWS > 0u) d0 += float(dot(w, x4[0ul*xs4 + k])); if (ROWS > 1u) d1 += float(dot(w, x4[1ul*xs4 + k])); if (ROWS > 2u) d2 += float(dot(w, x4[2ul*xs4 + k])); if (ROWS > 3u) d3 += float(dot(w, x4[3ul*xs4 + k])); if (ROWS > 4u) d4 += float(dot(w, x4[4ul*xs4 + k])); if (ROWS > 5u) d5 += float(dot(w, x4[5ul*xs4 + k])); if (ROWS > 6u) d6 += float(dot(w, x4[6ul*xs4 + k])); if (ROWS > 7u) d7 += float(dot(w, x4[7ul*xs4 + k])); }
        if (ROWS > 0u) a0 += d0 * s;
        if (ROWS > 1u) a1 += d1 * s;
        if (ROWS > 2u) a2 += d2 * s;
        if (ROWS > 3u) a3 += d3 * s;
        if (ROWS > 4u) a4 += d4 * s;
        if (ROWS > 5u) a5 += d5 * s;
        if (ROWS > 6u) a6 += d6 * s;
        if (ROWS > 7u) a7 += d7 * s;
    }
    float bv = p.has_bias ? bfv(bias[row]) : 0.0f;
    if (ROWS > 0u) { float t = simd_sum(a0) + bv; if (lane == 0) y[0ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd_sum(a1) + bv; if (lane == 0) y[1ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd_sum(a2) + bv; if (lane == 0) y[2ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd_sum(a3) + bv; if (lane == 0) y[3ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd_sum(a4) + bv; if (lane == 0) y[4ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd_sum(a5) + bv; if (lane == 0) y[5ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd_sum(a6) + bv; if (lane == 0) y[6ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd_sum(a7) + bv; if (lane == 0) y[7ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
}

struct MtpCtxKvP { uint kv_heads; uint head_dim; uint pos0; float eps; float log_theta; };

kernel void mtp_ctx_kv_store(device const float* ctx_kv [[buffer(0)]],
                             device const ushort* k_norm [[buffer(1)]],
                             device ushort* kcache [[buffer(2)]],
                             device ushort* vcache [[buffer(3)]],
                             constant MtpCtxKvP& p [[buffer(4)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint h = group.x, m = group.y;
    uint kvw = p.kv_heads * p.head_dim;
    device const float* ks = ctx_kv + ulong(m) * (2u * kvw) + h * p.head_dim;
    device const float* vs = ctx_kv + ulong(m) * (2u * kvw) + kvw + h * p.head_dim;
    float a0 = ks[lane], a1 = ks[lane + 32u], a2 = ks[lane + 64u], a3 = ks[lane + 96u];
    float ss = simd_sum(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3);
    float r = 1.0f / precise::sqrt(ss / float(p.head_dim) + p.eps);
    a0 = rbf(a0 * r * bfv(k_norm[lane]));
    a1 = rbf(a1 * r * bfv(k_norm[lane + 32u]));
    a2 = rbf(a2 * r * bfv(k_norm[lane + 64u]));
    a3 = rbf(a3 * r * bfv(k_norm[lane + 96u]));
    float pos = float(p.pos0 + m);
    float ang0 = pos * precise::exp(-float(2u * lane) / float(p.head_dim) * p.log_theta);
    float ang1 = pos * precise::exp(-float(2u * (lane + 32u)) / float(p.head_dim) * p.log_theta);
    float c0 = precise::cos(ang0), s0 = precise::sin(ang0);
    float c1 = precise::cos(ang1), s1 = precise::sin(ang1);
    float r0 = a0 * c0 - a2 * s0;
    float r2 = a2 * c0 + a0 * s0;
    float r1 = a1 * c1 - a3 * s1;
    float r3 = a3 * c1 + a1 * s1;
    ulong dst = ulong(p.pos0 + m) * kvw + h * p.head_dim;
    kcache[dst + lane] = f2bf(r0);
    kcache[dst + lane + 32u] = f2bf(r1);
    kcache[dst + lane + 64u] = f2bf(r2);
    kcache[dst + lane + 96u] = f2bf(r3);
    vcache[dst + lane] = f2bf(vs[lane]);
    vcache[dst + lane + 32u] = f2bf(vs[lane + 32u]);
    vcache[dst + lane + 64u] = f2bf(vs[lane + 64u]);
    vcache[dst + lane + 96u] = f2bf(vs[lane + 96u]);
}

struct MtpEmbedP { uint hidden; float eps; };

kernel void mtp_embed_block(device const ushort* embed [[buffer(0)]],
                            device const ushort* embed_norm [[buffer(1)]],
                            device const ushort* mask_embed [[buffer(2)]],
                            device const uint* anchor [[buffer(3)]],
                            device float4* out [[buffer(4)]],
                            constant MtpEmbedP& p [[buffer(5)]],
                            uint m [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = p.hidden / 4u;
    device float4* om = out + ulong(m) * n4;
    if (m > 0u) {
        for (uint i = tid; i < n4; i += threads) {
            uint b = i * 4u;
            om[i] = float4(bfv(mask_embed[b]), bfv(mask_embed[b + 1u]), bfv(mask_embed[b + 2u]), bfv(mask_embed[b + 3u]));
        }
        return;
    }
    uint tok = anchor[0];
    device const ushort* er = embed + ulong(tok) * p.hidden;
    float4 v = 0.0f;
    if (tid < n4) {
        uint b = tid * 4u;
        v = float4(bfv(er[b]), bfv(er[b + 1u]), bfv(er[b + 2u]), bfv(er[b + 3u]));
    }
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        uint b = i * 4u;
        float4 vt = float4(bfv(er[b]), bfv(er[b + 1u]), bfv(er[b + 2u]), bfv(er[b + 3u]));
        s += dot(vt, vt);
    }
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) tot += part[i];
    float r = 1.0f / precise::sqrt(tot / float(p.hidden) + p.eps);
    if (tid < n4) {
        uint b = tid * 4u;
        float4 o = v * r * float4(bfv(embed_norm[b]), bfv(embed_norm[b + 1u]), bfv(embed_norm[b + 2u]), bfv(embed_norm[b + 3u]));
        om[tid] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        uint b = i * 4u;
        float4 vt = float4(bfv(er[b]), bfv(er[b + 1u]), bfv(er[b + 2u]), bfv(er[b + 3u]));
        float4 o = vt * r * float4(bfv(embed_norm[b]), bfv(embed_norm[b + 1u]), bfv(embed_norm[b + 2u]), bfv(embed_norm[b + 3u]));
        om[i] = float4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
}

struct MtpQkRopeP { uint q_heads; uint kv_heads; uint head_dim; uint qkv_stride; uint pos0; float eps; float log_theta; };

kernel void mtp_qk_norm_rope(device float* qkv [[buffer(0)]],
                             device const ushort* q_norm [[buffer(1)]],
                             device const ushort* k_norm [[buffer(2)]],
                             constant MtpQkRopeP& p [[buffer(3)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    uint idx = group.x, m = group.y;
    bool is_k = idx >= p.q_heads;
    device const ushort* nw = is_k ? k_norm : q_norm;
    device float* ptr = qkv + ulong(m) * p.qkv_stride + idx * p.head_dim;
    float a0 = ptr[lane], a1 = ptr[lane + 32u], a2 = ptr[lane + 64u], a3 = ptr[lane + 96u];
    float ss = simd_sum(a0 * a0 + a1 * a1 + a2 * a2 + a3 * a3);
    float r = 1.0f / precise::sqrt(ss / float(p.head_dim) + p.eps);
    a0 = rbf(a0 * r * bfv(nw[lane]));
    a1 = rbf(a1 * r * bfv(nw[lane + 32u]));
    a2 = rbf(a2 * r * bfv(nw[lane + 64u]));
    a3 = rbf(a3 * r * bfv(nw[lane + 96u]));
    float pos = float(p.pos0 + m);
    float ang0 = pos * precise::exp(-float(2u * lane) / float(p.head_dim) * p.log_theta);
    float ang1 = pos * precise::exp(-float(2u * (lane + 32u)) / float(p.head_dim) * p.log_theta);
    float c0 = precise::cos(ang0), s0 = precise::sin(ang0);
    float c1 = precise::cos(ang1), s1 = precise::sin(ang1);
    ptr[lane] = rbf(a0 * c0 - a2 * s0);
    ptr[lane + 32u] = rbf(a1 * c1 - a3 * s1);
    ptr[lane + 64u] = rbf(a2 * c0 + a0 * s0);
    ptr[lane + 96u] = rbf(a3 * c1 + a1 * s1);
}

struct MtpAttnP { uint prefix_len; uint ctx_start; uint q_heads; uint kv_heads; uint group; uint qkv_stride; uint q_dim; uint kv_dim; float scale; };

kernel void mtp_attend_block(device const float* qkv [[buffer(0)]],
                             device const ushort4* kc [[buffer(1)]],
                             device const ushort4* vc [[buffer(2)]],
                             device half4* out [[buffer(3)]],
                             constant MtpAttnP& p [[buffer(4)]],
                             uint2 group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float sm[4];
    threadgroup float sl[4];
    threadgroup float4 so[4][32];
    uint h = group.x, m = group.y;
    uint kvh = h / p.group;
    device const float4* q4p = (device const float4*)(qkv + ulong(m) * p.qkv_stride + h * 128u);
    float4 q4 = q4p[lane];
    uint ctx_cnt = p.prefix_len > p.ctx_start ? (p.prefix_len - p.ctx_start) : 0u;
    uint tot_cnt = ctx_cnt + 8u;
    float mv = -INFINITY, lv = 0.0f;
    float4 o4 = 0.0f;
    for (uint t = simd; t < tot_cnt; t += 4u) {
        float4 k4, v4;
        if (t < ctx_cnt) {
            uint kp = p.ctx_start + t;
            ulong base = (ulong(kp) * p.kv_heads + kvh) * 32u + lane;
            ushort4 ku = kc[base];
            ushort4 vu = vc[base];
            k4 = float4(bfv(ku.x), bfv(ku.y), bfv(ku.z), bfv(ku.w));
            v4 = float4(bfv(vu.x), bfv(vu.y), bfv(vu.z), bfv(vu.w));
        } else {
            uint bm = t - ctx_cnt;
            device const float4* kp4 = (device const float4*)(qkv + ulong(bm) * p.qkv_stride + p.q_dim + kvh * 128u);
            device const float4* vp4 = (device const float4*)(qkv + ulong(bm) * p.qkv_stride + p.q_dim + p.kv_dim + kvh * 128u);
            k4 = kp4[lane];
            v4 = vp4[lane];
        }
        float s = simd_sum(dot(q4, k4)) * p.scale;
        float mn = max(mv, s);
        float a = mv == -INFINITY ? 0.0f : precise::exp(mv - mn);
        float b = precise::exp(s - mn);
        o4 = o4 * a + v4 * b;
        lv = lv * a + b;
        mv = mn;
    }
    if (lane == 0) { sm[simd] = mv; sl[simd] = lv; }
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
        float4 y4 = acc / tot;
        out[(ulong(m) * p.q_heads + h) * 32u + lane] = half4(rbf(y4.x), rbf(y4.y), rbf(y4.z), rbf(y4.w));
    }
}

kernel void mtp_add_rmsnorm(device float4* res [[buffer(0)]],
                            device const float4* delta [[buffer(1)]],
                            device const ushort* nw [[buffer(2)]],
                            device half4* normed [[buffer(3)]],
                            constant Norm& p [[buffer(4)]],
                            uint m [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint threads [[threads_per_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[20];
    uint n4 = p.n / 4u;
    device float4* rm = res + ulong(m) * n4;
    device const float4* dm = delta + ulong(m) * n4;
    device half4* ym = normed + ulong(m) * n4;
    float4 v = 0.0f;
    if (tid < n4) {
        float4 t = rm[tid] + dm[tid];
        v = float4(rbf(t.x), rbf(t.y), rbf(t.z), rbf(t.w));
        rm[tid] = v;
    }
    float s = dot(v, v);
    for (uint i = tid + threads; i < n4; i += threads) {
        float4 t = rm[i] + dm[i];
        float4 vr = float4(rbf(t.x), rbf(t.y), rbf(t.z), rbf(t.w));
        rm[i] = vr;
        s += dot(vr, vr);
    }
    s = simd_sum(s);
    if (lane == 0) part[simd] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint i = 0; i < (threads + 31u) / 32u; ++i) tot += part[i];
    float r = 1.0f / precise::sqrt(tot / float(p.n) + p.eps);
    if (tid < n4) {
        uint b = tid * 4u;
        float4 o = v * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        ym[tid] = half4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
    for (uint i = tid + threads; i < n4; i += threads) {
        uint b = i * 4u;
        float4 o = rm[i] * r * float4(bfv(nw[b]), bfv(nw[b + 1u]), bfv(nw[b + 2u]), bfv(nw[b + 3u]));
        ym[i] = half4(rbf(o.x), rbf(o.y), rbf(o.z), rbf(o.w));
    }
}

kernel void mtp_add_inplace(device float4* a [[buffer(0)]],
                            device const float4* b [[buffer(1)]],
                            constant uint& n4 [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]]) {
    uint i = gid.x, m = gid.y;
    if (i < n4) {
        ulong idx = ulong(m) * n4 + i;
        float4 t = a[idx] + b[idx];
        a[idx] = float4(rbf(t.x), rbf(t.y), rbf(t.z), rbf(t.w));
    }
}

kernel void mtp_silu_mul(device const float* gu [[buffer(0)]],
                         device half* out [[buffer(1)]],
                         constant uint& inter [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint j = gid.x, m = gid.y;
    if (j >= inter) return;
    float g = gu[ulong(m) * (2u * inter) + j];
    float u = gu[ulong(m) * (2u * inter) + inter + j];
    out[ulong(m) * inter + j] = half(rbf((g * sigm(g)) * u));
}

struct MtpBaseP { uint dvocab; uint ng; uint xs4; };

kernel void mtp_base_logits_7(device const uint4* q4 [[buffer(0)]],
                              device const half* sc [[buffer(1)]],
                              device const half4* x4 [[buffer(2)]],
                              device half* base_out [[buffer(3)]],
                              constant MtpBaseP& p [[buffer(4)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint simds [[simdgroups_per_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * simds + simd;
    if (row >= p.dvocab) return;
    device const uint4* qr = q4 + ulong(row) * p.ng;
    device const half* sr = sc + ulong(row) * p.ng;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0;
    for (uint g = lane; g < p.ng; g += 32u) {
        uint4 pk = qr[g];
        float s = float(sr[g]);
        uint g8 = g * 8u;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0;
        { half4 w = u4_lo_h(pk.x); d0 += float(dot(w, x4[1ul*p.xs4 + g8])); d1 += float(dot(w, x4[2ul*p.xs4 + g8])); d2 += float(dot(w, x4[3ul*p.xs4 + g8])); d3 += float(dot(w, x4[4ul*p.xs4 + g8])); d4 += float(dot(w, x4[5ul*p.xs4 + g8])); d5 += float(dot(w, x4[6ul*p.xs4 + g8])); d6 += float(dot(w, x4[7ul*p.xs4 + g8])); }
        { half4 w = u4_hi_h(pk.x); uint k = g8 + 1u; d0 += float(dot(w, x4[1ul*p.xs4 + k])); d1 += float(dot(w, x4[2ul*p.xs4 + k])); d2 += float(dot(w, x4[3ul*p.xs4 + k])); d3 += float(dot(w, x4[4ul*p.xs4 + k])); d4 += float(dot(w, x4[5ul*p.xs4 + k])); d5 += float(dot(w, x4[6ul*p.xs4 + k])); d6 += float(dot(w, x4[7ul*p.xs4 + k])); }
        { half4 w = u4_lo_h(pk.y); uint k = g8 + 2u; d0 += float(dot(w, x4[1ul*p.xs4 + k])); d1 += float(dot(w, x4[2ul*p.xs4 + k])); d2 += float(dot(w, x4[3ul*p.xs4 + k])); d3 += float(dot(w, x4[4ul*p.xs4 + k])); d4 += float(dot(w, x4[5ul*p.xs4 + k])); d5 += float(dot(w, x4[6ul*p.xs4 + k])); d6 += float(dot(w, x4[7ul*p.xs4 + k])); }
        { half4 w = u4_hi_h(pk.y); uint k = g8 + 3u; d0 += float(dot(w, x4[1ul*p.xs4 + k])); d1 += float(dot(w, x4[2ul*p.xs4 + k])); d2 += float(dot(w, x4[3ul*p.xs4 + k])); d3 += float(dot(w, x4[4ul*p.xs4 + k])); d4 += float(dot(w, x4[5ul*p.xs4 + k])); d5 += float(dot(w, x4[6ul*p.xs4 + k])); d6 += float(dot(w, x4[7ul*p.xs4 + k])); }
        { half4 w = u4_lo_h(pk.z); uint k = g8 + 4u; d0 += float(dot(w, x4[1ul*p.xs4 + k])); d1 += float(dot(w, x4[2ul*p.xs4 + k])); d2 += float(dot(w, x4[3ul*p.xs4 + k])); d3 += float(dot(w, x4[4ul*p.xs4 + k])); d4 += float(dot(w, x4[5ul*p.xs4 + k])); d5 += float(dot(w, x4[6ul*p.xs4 + k])); d6 += float(dot(w, x4[7ul*p.xs4 + k])); }
        { half4 w = u4_hi_h(pk.z); uint k = g8 + 5u; d0 += float(dot(w, x4[1ul*p.xs4 + k])); d1 += float(dot(w, x4[2ul*p.xs4 + k])); d2 += float(dot(w, x4[3ul*p.xs4 + k])); d3 += float(dot(w, x4[4ul*p.xs4 + k])); d4 += float(dot(w, x4[5ul*p.xs4 + k])); d5 += float(dot(w, x4[6ul*p.xs4 + k])); d6 += float(dot(w, x4[7ul*p.xs4 + k])); }
        { half4 w = u4_lo_h(pk.w); uint k = g8 + 6u; d0 += float(dot(w, x4[1ul*p.xs4 + k])); d1 += float(dot(w, x4[2ul*p.xs4 + k])); d2 += float(dot(w, x4[3ul*p.xs4 + k])); d3 += float(dot(w, x4[4ul*p.xs4 + k])); d4 += float(dot(w, x4[5ul*p.xs4 + k])); d5 += float(dot(w, x4[6ul*p.xs4 + k])); d6 += float(dot(w, x4[7ul*p.xs4 + k])); }
        { half4 w = u4_hi_h(pk.w); uint k = g8 + 7u; d0 += float(dot(w, x4[1ul*p.xs4 + k])); d1 += float(dot(w, x4[2ul*p.xs4 + k])); d2 += float(dot(w, x4[3ul*p.xs4 + k])); d3 += float(dot(w, x4[4ul*p.xs4 + k])); d4 += float(dot(w, x4[5ul*p.xs4 + k])); d5 += float(dot(w, x4[6ul*p.xs4 + k])); d6 += float(dot(w, x4[7ul*p.xs4 + k])); }
        a0 += d0 * s; a1 += d1 * s; a2 += d2 * s; a3 += d3 * s; a4 += d4 * s; a5 += d5 * s; a6 += d6 * s;
    }
    float t0 = simd_sum(a0), t1 = simd_sum(a1), t2 = simd_sum(a2), t3 = simd_sum(a3), t4 = simd_sum(a4), t5 = simd_sum(a5), t6 = simd_sum(a6);
    if (lane == 0) {
        base_out[0ul * p.dvocab + row] = half(t0);
        base_out[1ul * p.dvocab + row] = half(t1);
        base_out[2ul * p.dvocab + row] = half(t2);
        base_out[3ul * p.dvocab + row] = half(t3);
        base_out[4ul * p.dvocab + row] = half(t4);
        base_out[5ul * p.dvocab + row] = half(t5);
        base_out[6ul * p.dvocab + row] = half(t6);
    }
}

struct MtpRnnInP { uint rank; uint hidden; uint step; };

kernel void mtp_rnn_input(device const float* state [[buffer(0)]],
                          device const ushort* markov_w1 [[buffer(1)]],
                          device const uint* out_ids [[buffer(2)]],
                          device const half* hn [[buffer(3)]],
                          device const half* f_rnn [[buffer(4)]],
                          device half* rnn_in [[buffer(5)]],
                          constant MtpRnnInP& p [[buffer(6)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    uint prev_tok = out_ids[p.step];
    for (uint i = tid; i < p.rank; i += threads) {
        rnn_in[i] = p.step == 0u ? half(0) : half(state[i]);
        rnn_in[p.rank + i] = half(bfv(markov_w1[ulong(prev_tok) * p.rank + i]));
    }
    device const half* hk = hn + ulong(p.step + 1u) * p.hidden;
    uint off0 = 2u * p.rank;
    uint off1 = off0 + p.hidden;
    for (uint i = tid; i < p.hidden; i += threads) {
        rnn_in[off0 + i] = hk[i];
        rnn_in[off1 + i] = f_rnn[i];
    }
}

kernel void mtp_rnn_gates(device const float* proj [[buffer(0)]],
                          device float* state [[buffer(1)]],
                          device float* latent [[buffer(2)]],
                          constant uint2& rs [[buffer(3)]],
                          uint tid [[thread_position_in_threadgroup]]) {
    uint rank = rs.x, step = rs.y;
    if (tid >= rank) return;
    float st = step == 0u ? 0.0f : state[tid];
    float gate_raw = proj[tid];
    float cand_raw = proj[rank + tid];
    float out_raw = proj[2u * rank + tid];
    float gate = rbf(1.0f / (1.0f + precise::exp(-gate_raw)));
    float kept = rbf(gate * st);
    float blend = rbf(rbf(1.0f - gate) * rbf(precise::tanh(cand_raw)));
    state[tid] = rbf(kept + blend);
    latent[tid] = rbf(precise::tanh(out_raw));
}

struct MtpMarkovP { uint dvocab; uint step; float cap; float inv_temp; };

kernel void mtp_markov_scan(device const uint4* fold_q4 [[buffer(0)]],
                            device const half* fold_sc [[buffer(1)]],
                            device const float* fold_b [[buffer(2)]],
                            device const half* base_logits [[buffer(3)]],
                            device const float4* latent4 [[buffer(4)]],
                            device float* tg_val [[buffer(5)]],
                            device uint* tg_row [[buffer(6)]],
                            device float* tg_sum [[buffer(7)]],
                            constant MtpMarkovP& p [[buffer(8)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[32];
    threadgroup uint sr[32];
    uint sub = lane >> 3;
    uint g = lane & 7u;
    uint slot = simd * 4u + sub;
    uint row = group * 32u + slot;
    float dot_g = 0.0f;
    if (row < p.dvocab) {
        uint4 pk = fold_q4[ulong(row) * 8u + g];
        float s = float(fold_sc[ulong(row) * 8u + g]);
        uint g8 = g * 8u;
        float d = dot(u4_lo(pk.x), latent4[g8 + 0u]) + dot(u4_hi(pk.x), latent4[g8 + 1u])
                + dot(u4_lo(pk.y), latent4[g8 + 2u]) + dot(u4_hi(pk.y), latent4[g8 + 3u])
                + dot(u4_lo(pk.z), latent4[g8 + 4u]) + dot(u4_hi(pk.z), latent4[g8 + 5u])
                + dot(u4_lo(pk.w), latent4[g8 + 6u]) + dot(u4_hi(pk.w), latent4[g8 + 7u]);
        dot_g = d * s;
    }
    dot_g += simd_shuffle_xor(dot_g, 4u);
    dot_g += simd_shuffle_xor(dot_g, 2u);
    dot_g += simd_shuffle_xor(dot_g, 1u);
    if (g == 0u) {
        float val = -INFINITY;
        if (row < p.dvocab) {
            float raw = float(base_logits[ulong(p.step) * p.dvocab + row]) + fold_b[row] + dot_g;
            val = p.cap > 0.0f ? (p.cap * precise::tanh(raw / p.cap)) : raw;
        }
        sv[slot] = val;
        sr[slot] = row;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        float v = sv[lane];
        uint r = sr[lane];
        float bv = v;
        uint br = r;
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_xor(bv, off);
            uint oi = simd_shuffle_xor(br, off);
            if (ov > bv || (ov == bv && oi < br)) { bv = ov; br = oi; }
        }
        float e = v == -INFINITY ? 0.0f : precise::exp((v - bv) * p.inv_temp);
        float esum = simd_sum(e);
        if (lane == 0u) {
            tg_val[group] = bv;
            tg_row[group] = br;
            tg_sum[group] = esum;
        }
    }
}

struct MtpReduceP { uint ntg; uint step; float inv_temp; };

kernel void mtp_markov_reduce(device const float* tg_val [[buffer(0)]],
                              device const uint* tg_row [[buffer(1)]],
                              device const float* tg_sum [[buffer(2)]],
                              device const uint* vocab_ids [[buffer(3)]],
                              device uint* out_ids [[buffer(4)]],
                              device float* out_probs [[buffer(5)]],
                              constant MtpReduceP& p [[buffer(6)]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float mv[16];
    threadgroup uint mr[16];
    threadgroup float ms[16];
    float bv = -INFINITY;
    uint br = 0xffffffffu;
    for (uint i = tid; i < p.ntg; i += 512u) {
        float v = tg_val[i];
        uint r = tg_row[i];
        if (v > bv || (v == bv && r < br)) { bv = v; br = r; }
    }
    for (uint off = 16u; off > 0u; off >>= 1) {
        float ov = simd_shuffle_xor(bv, off);
        uint oi = simd_shuffle_xor(br, off);
        if (ov > bv || (ov == bv && oi < br)) { bv = ov; br = oi; }
    }
    if (lane == 0u) { mv[simd] = bv; mr[simd] = br; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = mv[0];
    uint gr = mr[0];
    for (uint i = 1u; i < 16u; ++i) {
        if (mv[i] > gm || (mv[i] == gm && mr[i] < gr)) { gm = mv[i]; gr = mr[i]; }
    }
    float ls = 0.0f;
    for (uint i = tid; i < p.ntg; i += 512u) {
        float v = tg_val[i];
        if (v != -INFINITY) ls += tg_sum[i] * precise::exp((v - gm) * p.inv_temp);
    }
    ls = simd_sum(ls);
    if (lane == 0u) ms[simd] = ls;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint i = 0u; i < 16u; ++i) tot += ms[i];
        out_ids[p.step + 1u] = vocab_ids[gr];
        out_probs[p.step] = tot > 0.0f ? (1.0f / tot) : 0.0f;
    }
}
