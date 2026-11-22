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

struct MtpMixP { uint hidden; uint alpha_row; uint row_start; float eps; uint use_map; uint map[16]; };

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
    uint sm = p.use_map ? p.map[p.row_start + m] : (p.row_start + m);
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
    uint sub = lane >> 3;
    uint sl = lane & 7u;
    uint row = (group * simds + simd) * 4u + sub;
    if (row >= p.rows) return;
    uint xs4 = p.x_stride / 4u;
    device const uint4* qr = q4 + ulong(row) * p.ng;
    device const half* sr = sc + ulong(row) * p.ng;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    for (uint g = sl; g < p.ng; g += 8u) {
        uint4 pk = qr[g];
        float s = float(sr[g]);
        uint g8 = g * 8u;
        half d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0, d7 = 0;
        uint ws[4] = {pk.x, pk.y, pk.z, pk.w};
        for (uint t = 0; t < 4u; ++t) {
            uint ww = ws[t];
            uint k0 = g8 + t * 2u;
            {
                half4 w = u4_lo_h_fast(ww);
                if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k0]);
                if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k0]);
                if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k0]);
                if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k0]);
                if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k0]);
                if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k0]);
                if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k0]);
                if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k0]);
            }
            {
                half4 w = u4_hi_h_fast(ww);
                uint k1 = k0 + 1u;
                if (ROWS > 0u) d0 += dot(w, x4[0ul*xs4 + k1]);
                if (ROWS > 1u) d1 += dot(w, x4[1ul*xs4 + k1]);
                if (ROWS > 2u) d2 += dot(w, x4[2ul*xs4 + k1]);
                if (ROWS > 3u) d3 += dot(w, x4[3ul*xs4 + k1]);
                if (ROWS > 4u) d4 += dot(w, x4[4ul*xs4 + k1]);
                if (ROWS > 5u) d5 += dot(w, x4[5ul*xs4 + k1]);
                if (ROWS > 6u) d6 += dot(w, x4[6ul*xs4 + k1]);
                if (ROWS > 7u) d7 += dot(w, x4[7ul*xs4 + k1]);
            }
        }
        if (ROWS > 0u) a0 += float(d0) * s;
        if (ROWS > 1u) a1 += float(d1) * s;
        if (ROWS > 2u) a2 += float(d2) * s;
        if (ROWS > 3u) a3 += float(d3) * s;
        if (ROWS > 4u) a4 += float(d4) * s;
        if (ROWS > 5u) a5 += float(d5) * s;
        if (ROWS > 6u) a6 += float(d6) * s;
        if (ROWS > 7u) a7 += float(d7) * s;
    }
    float bv = p.has_bias ? bfv(bias[row]) : 0.0f;
    if (ROWS > 0u) { float t = simd8_sum(a0) + bv; if (sl == 0u) y[0ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 1u) { float t = simd8_sum(a1) + bv; if (sl == 0u) y[1ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 2u) { float t = simd8_sum(a2) + bv; if (sl == 0u) y[2ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 3u) { float t = simd8_sum(a3) + bv; if (sl == 0u) y[3ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 4u) { float t = simd8_sum(a4) + bv; if (sl == 0u) y[4ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 5u) { float t = simd8_sum(a5) + bv; if (sl == 0u) y[5ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 6u) { float t = simd8_sum(a6) + bv; if (sl == 0u) y[6ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
    if (ROWS > 7u) { float t = simd8_sum(a7) + bv; if (sl == 0u) y[7ul * p.y_stride + row] = p.round_out ? rbf(t) : t; }
}

struct MtpCtxKvP { uint kv_heads; uint head_dim; uint pos0; uint kv_cap; float eps; float log_theta; };

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
    ulong dst = ulong((p.pos0 + m) % p.kv_cap) * kvw + h * p.head_dim;
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

struct MtpAttnP { uint prefix_len; uint ctx_start; uint q_heads; uint kv_heads; uint group; uint qkv_stride; uint q_dim; uint kv_dim; uint kv_cap; float scale; };

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
            ulong base = (ulong(kp % p.kv_cap) * p.kv_heads + kvh) * 32u + lane;
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
    uint sub = lane >> 3;
    uint sl = lane & 7u;
    uint row = (group * simds + simd) * 4u + sub;
    if (row >= p.dvocab) return;
    device const uint4* qr = q4 + ulong(row) * p.ng;
    device const half* sr = sc + ulong(row) * p.ng;
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0;
    for (uint g = sl; g < p.ng; g += 8u) {
        uint4 pk = qr[g];
        float s = float(sr[g]);
        uint g8 = g * 8u;
        half d0 = 0, d1 = 0, d2 = 0, d3 = 0, d4 = 0, d5 = 0, d6 = 0;
        uint ws[4] = {pk.x, pk.y, pk.z, pk.w};
        for (uint t = 0; t < 4u; ++t) {
            uint ww = ws[t];
            uint k0 = g8 + t * 2u;
            {
                half4 w = u4_lo_h_fast(ww);
                d0 += dot(w, x4[1ul*p.xs4 + k0]); d1 += dot(w, x4[2ul*p.xs4 + k0]); d2 += dot(w, x4[3ul*p.xs4 + k0]); d3 += dot(w, x4[4ul*p.xs4 + k0]);
                d4 += dot(w, x4[5ul*p.xs4 + k0]); d5 += dot(w, x4[6ul*p.xs4 + k0]); d6 += dot(w, x4[7ul*p.xs4 + k0]);
            }
            {
                half4 w = u4_hi_h_fast(ww);
                uint k1 = k0 + 1u;
                d0 += dot(w, x4[1ul*p.xs4 + k1]); d1 += dot(w, x4[2ul*p.xs4 + k1]); d2 += dot(w, x4[3ul*p.xs4 + k1]); d3 += dot(w, x4[4ul*p.xs4 + k1]);
                d4 += dot(w, x4[5ul*p.xs4 + k1]); d5 += dot(w, x4[6ul*p.xs4 + k1]); d6 += dot(w, x4[7ul*p.xs4 + k1]);
            }
        }
        a0 += float(d0) * s; a1 += float(d1) * s; a2 += float(d2) * s; a3 += float(d3) * s; a4 += float(d4) * s; a5 += float(d5) * s; a6 += float(d6) * s;
    }
    float t0 = simd8_sum(a0), t1 = simd8_sum(a1), t2 = simd8_sum(a2), t3 = simd8_sum(a3), t4 = simd8_sum(a4), t5 = simd8_sum(a5), t6 = simd8_sum(a6);
    if (sl == 0u) {
        base_out[0ul * p.dvocab + row] = half(t0);
        base_out[1ul * p.dvocab + row] = half(t1);
        base_out[2ul * p.dvocab + row] = half(t2);
        base_out[3ul * p.dvocab + row] = half(t3);
        base_out[4ul * p.dvocab + row] = half(t4);
        base_out[5ul * p.dvocab + row] = half(t5);
        base_out[6ul * p.dvocab + row] = half(t6);
    }
}

struct MtpRnnInP { uint rank; uint hidden; uint step; uint beams; uint parent_b[4]; };

kernel void mtp_rnn_input(device const float* state [[buffer(0)]],
                          device const ushort* markov_w1 [[buffer(1)]],
                          device const uint* beam_toks [[buffer(2)]],
                          device const half* hn [[buffer(3)]],
                          device const half* f_rnn [[buffer(4)]],
                          device half* rnn_in [[buffer(5)]],
                          constant MtpRnnInP& p [[buffer(6)]],
                          uint b [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
    uint prev_tok = beam_toks[b];
    uint pb = p.parent_b[b];
    uint dim = 2u * p.rank + 2u * p.hidden;
    device half* dst = rnn_in + ulong(b) * dim;
    device const float* st = state + ulong(pb) * p.rank;
    for (uint i = tid; i < p.rank; i += threads) {
        dst[i] = p.step == 0u ? half(0) : half(st[i]);
        dst[p.rank + i] = half(bfv(markov_w1[ulong(prev_tok) * p.rank + i]));
    }
    device const half* hk = hn + ulong(p.step + 1u) * p.hidden;
    uint off0 = 2u * p.rank;
    uint off1 = off0 + p.hidden;
    for (uint i = tid; i < p.hidden; i += threads) {
        dst[off0 + i] = hk[i];
        dst[off1 + i] = f_rnn[i];
    }
}

struct MtpRnnGateP { uint rank; uint step; uint parent_b[4]; };

kernel void mtp_rnn_gates(device const float* proj [[buffer(0)]],
                          device const float* state_in [[buffer(1)]],
                          device float* state_out [[buffer(2)]],
                          device half* latent [[buffer(3)]],
                          constant MtpRnnGateP& p [[buffer(4)]],
                          uint b [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]]) {
    if (tid >= p.rank) return;
    uint pb = p.parent_b[b];
    float st = p.step == 0u ? 0.0f : state_in[ulong(pb) * p.rank + tid];
    device const float* pr = proj + ulong(b) * (3u * p.rank);
    float gate_raw = pr[tid];
    float cand_raw = pr[p.rank + tid];
    float out_raw = pr[2u * p.rank + tid];
    float gate = rbf(1.0f / (1.0f + precise::exp(-gate_raw)));
    float kept = rbf(gate * st);
    float blend = rbf(rbf(1.0f - gate) * rbf(precise::tanh(cand_raw)));
    state_out[ulong(b) * p.rank + tid] = rbf(kept + blend);
    latent[ulong(b) * p.rank + tid] = half(rbf(precise::tanh(out_raw)));
}

struct MtpMarkovP { uint dvocab; uint step; uint beams; uint ntg; float cap; float inv_temp; };

kernel void mtp_markov_scan(device const uint4* fold_q4 [[buffer(0)]],
                            device const half* fold_sc [[buffer(1)]],
                            device const float* fold_b [[buffer(2)]],
                            device const half* base_logits [[buffer(3)]],
                            device const half4* latent4 [[buffer(4)]],
                            device float* tg_val [[buffer(5)]],
                            device uint* tg_row [[buffer(6)]],
                            device float* tg_sum [[buffer(7)]],
                            constant MtpMarkovP& p [[buffer(8)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[3][32];
    threadgroup uint sr[32];
    uint sub = lane >> 3;
    uint g = lane & 7u;
    uint slot = simd * 4u + sub;
    uint row = group * 32u + slot;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    if (row < p.dvocab) {
        uint4 pk = fold_q4[ulong(row) * 8u + g];
        float s = float(fold_sc[ulong(row) * 8u + g]);
        uint g8 = g * 8u;
        device const half4* l0 = latent4 + g8;
        device const half4* l1 = latent4 + 64u + g8;
        device const half4* l2 = latent4 + 128u + g8;
        half4 w0 = u4_lo_h_fast(pk.x), w1 = u4_hi_h_fast(pk.x);
        half4 w2 = u4_lo_h_fast(pk.y), w3 = u4_hi_h_fast(pk.y);
        half4 w4 = u4_lo_h_fast(pk.z), w5 = u4_hi_h_fast(pk.z);
        half4 w6 = u4_lo_h_fast(pk.w), w7 = u4_hi_h_fast(pk.w);
        half h0 = dot(w0, l0[0]) + dot(w1, l0[1]) + dot(w2, l0[2]) + dot(w3, l0[3])
                + dot(w4, l0[4]) + dot(w5, l0[5]) + dot(w6, l0[6]) + dot(w7, l0[7]);
        d0 = float(h0) * s;
        if (p.beams > 1u) {
            half h1 = dot(w0, l1[0]) + dot(w1, l1[1]) + dot(w2, l1[2]) + dot(w3, l1[3])
                    + dot(w4, l1[4]) + dot(w5, l1[5]) + dot(w6, l1[6]) + dot(w7, l1[7]);
            d1 = float(h1) * s;
        }
        if (p.beams > 2u) {
            half h2 = dot(w0, l2[0]) + dot(w1, l2[1]) + dot(w2, l2[2]) + dot(w3, l2[3])
                    + dot(w4, l2[4]) + dot(w5, l2[5]) + dot(w6, l2[6]) + dot(w7, l2[7]);
            d2 = float(h2) * s;
        }
    }
    d0 = simd8_sum(d0);
    if (p.beams > 1u) d1 = simd8_sum(d1);
    if (p.beams > 2u) d2 = simd8_sum(d2);
    if (g == 0u) {
        sr[slot] = row;
        if (row < p.dvocab) {
            float base = float(base_logits[ulong(p.step) * p.dvocab + row]) + fold_b[row];
            float r0 = base + d0;
            sv[0][slot] = p.cap > 0.0f ? (p.cap * precise::tanh(r0 / p.cap)) : r0;
            if (p.beams > 1u) {
                float r1 = base + d1;
                sv[1][slot] = p.cap > 0.0f ? (p.cap * precise::tanh(r1 / p.cap)) : r1;
            }
            if (p.beams > 2u) {
                float r2 = base + d2;
                sv[2][slot] = p.cap > 0.0f ? (p.cap * precise::tanh(r2 / p.cap)) : r2;
            }
        } else {
            sv[0][slot] = -INFINITY;
            if (p.beams > 1u) sv[1][slot] = -INFINITY;
            if (p.beams > 2u) sv[2][slot] = -INFINITY;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        uint r = sr[lane];
        for (uint b = 0; b < p.beams; ++b) {
            float v = sv[b][lane];
            float v1 = v; uint r1 = r;
            for (uint off = 16u; off > 0u; off >>= 1) {
                float ov = simd_shuffle_xor(v1, off); uint oi = simd_shuffle_xor(r1, off);
                if (ov > v1 || (ov == v1 && oi < r1)) { v1 = ov; r1 = oi; }
            }
            float e = v == -INFINITY ? 0.0f : precise::exp((v - v1) * p.inv_temp);
            float esum = simd_sum(e);
            float v2 = (r == r1) ? -INFINITY : v; uint r2 = (r == r1) ? 0xffffffffu : r;
            for (uint off = 16u; off > 0u; off >>= 1) {
                float ov = simd_shuffle_xor(v2, off); uint oi = simd_shuffle_xor(r2, off);
                if (ov > v2 || (ov == v2 && oi < r2)) { v2 = ov; r2 = oi; }
            }
            float v3 = (r == r1 || r == r2) ? -INFINITY : v; uint r3 = (r == r1 || r == r2) ? 0xffffffffu : r;
            for (uint off = 16u; off > 0u; off >>= 1) {
                float ov = simd_shuffle_xor(v3, off); uint oi = simd_shuffle_xor(r3, off);
                if (ov > v3 || (ov == v3 && oi < r3)) { v3 = ov; r3 = oi; }
            }
            if (lane == 0u) {
                ulong dst = (ulong(b) * p.ntg + group) * 3u;
                tg_val[dst + 0u] = v1; tg_row[dst + 0u] = r1;
                tg_val[dst + 1u] = v2; tg_row[dst + 1u] = r2;
                tg_val[dst + 2u] = v3; tg_row[dst + 2u] = r3;
                tg_sum[ulong(b) * p.ntg + group] = esum;
            }
        }
    }
}

struct MtpReduceP { uint ntg; uint step; uint beams; float inv_temp; };

kernel void mtp_markov_reduce(device const float* tg_val [[buffer(0)]],
                              device const uint* tg_row [[buffer(1)]],
                              device const float* tg_sum [[buffer(2)]],
                              device const uint* vocab_ids [[buffer(3)]],
                              device uint* tree_ids [[buffer(4)]],
                              device float* tree_probs [[buffer(5)]],
                              device uint* next_beam_tok [[buffer(6)]],
                              constant MtpReduceP& p [[buffer(7)]],
                              uint b [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float mv[16][3];
    threadgroup uint mr[16][3];
    threadgroup float ms[16];
    device const float* tv = tg_val + ulong(b) * p.ntg * 3u;
    device const uint* tr = tg_row + ulong(b) * p.ntg * 3u;
    device const float* ts = tg_sum + ulong(b) * p.ntg;
    float lv1 = -INFINITY, lv2 = -INFINITY, lv3 = -INFINITY;
    uint lr1 = 0xffffffffu, lr2 = 0xffffffffu, lr3 = 0xffffffffu;
    for (uint i = tid; i < p.ntg; i += 512u) {
        for (uint j = 0; j < 3u; ++j) {
            float v = tv[i * 3u + j];
            uint r = tr[i * 3u + j];
            if (v > lv1 || (v == lv1 && r < lr1)) {
                lv3 = lv2; lr3 = lr2; lv2 = lv1; lr2 = lr1; lv1 = v; lr1 = r;
            } else if (v > lv2 || (v == lv2 && r < lr2)) {
                lv3 = lv2; lr3 = lr2; lv2 = v; lr2 = r;
            } else if (v > lv3 || (v == lv3 && r < lr3)) {
                lv3 = v; lr3 = r;
            }
        }
    }
    for (uint k = 0; k < 3u; ++k) {
        float bv = lv1;
        uint br = lr1;
        for (uint off = 16u; off > 0u; off >>= 1) {
            float ov = simd_shuffle_xor(bv, off);
            uint oi = simd_shuffle_xor(br, off);
            if (ov > bv || (ov == bv && oi < br)) { bv = ov; br = oi; }
        }
        if (lane == 0u) { mv[simd][k] = bv; mr[simd][k] = br; }
        if (lr1 == br && lv1 == bv) { lv1 = lv2; lr1 = lr2; lv2 = lv3; lr2 = lr3; lv3 = -INFINITY; lr3 = 0xffffffffu; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float g1 = -INFINITY, g2 = -INFINITY, g3 = -INFINITY;
    uint r1 = 0xffffffffu, r2 = 0xffffffffu, r3 = 0xffffffffu;
    for (uint s = 0; s < 16u; ++s) {
        for (uint k = 0; k < 3u; ++k) {
            float v = mv[s][k];
            uint r = mr[s][k];
            if (v > g1 || (v == g1 && r < r1)) {
                g3 = g2; r3 = r2; g2 = g1; r2 = r1; g1 = v; r1 = r;
            } else if (v > g2 || (v == g2 && r < r2)) {
                g3 = g2; r3 = r2; g2 = v; r2 = r;
            } else if (v > g3 || (v == g3 && r < r3)) {
                g3 = v; r3 = r;
            }
        }
    }
    float ls = 0.0f;
    for (uint i = tid; i < p.ntg; i += 512u) {
        float v = tv[i * 3u];
        if (v != -INFINITY) ls += ts[i] * precise::exp((v - g1) * p.inv_temp);
    }
    ls = simd_sum(ls);
    if (lane == 0u) ms[simd] = ls;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint i = 0u; i < 16u; ++i) tot += ms[i];
        float inv_tot = tot > 0.0f ? (1.0f / tot) : 0.0f;
        uint id1 = r1 != 0xffffffffu ? vocab_ids[r1] : 0u;
        uint id2 = r2 != 0xffffffffu ? vocab_ids[r2] : 0u;
        uint id3 = r3 != 0xffffffffu ? vocab_ids[r3] : 0u;
        float p1 = r1 != 0xffffffffu ? inv_tot : 0.0f;
        float p2 = r2 != 0xffffffffu ? (precise::exp((g2 - g1) * p.inv_temp) * inv_tot) : 0.0f;
        float p3 = r3 != 0xffffffffu ? (precise::exp((g3 - g1) * p.inv_temp) * inv_tot) : 0.0f;
        uint base = p.step == 0u ? 0u : (3u + (p.step - 1u) * 9u + b * 3u);
        tree_ids[base + 0u] = id1; tree_probs[base + 0u] = p1;
        tree_ids[base + 1u] = id2; tree_probs[base + 1u] = p2;
        tree_ids[base + 2u] = id3; tree_probs[base + 2u] = p3;
        if (p.step == 0u) {
            next_beam_tok[0] = id1;
            next_beam_tok[1] = id2;
            next_beam_tok[2] = id3;
        } else if (b == 0u) {
            next_beam_tok[0] = id1;
            next_beam_tok[1] = id2;
        } else if (b == 1u) {
            next_beam_tok[2] = id1;
        }
    }
}
