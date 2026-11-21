#include "probes.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#import <Foundation/Foundation.h>

#include "flags.h"
#include "gpu.h"

namespace {

struct Tensor {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    std::string dtype;
    std::vector<uint64_t> shape;
};

struct Mat {
    uint64_t offset = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct Mat12 {
    uint64_t sm_off = 0;
    uint64_t ex_off = 0;
    uint64_t ro_off = 0;
    uint64_t op_off = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t has_outliers = 0;
    uint32_t elut[16] = {};
};

struct MatQ4 {
    uint64_t q4_off = 0;
    uint64_t sc_off = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
};

struct CtlWeights {
    Mat down, gain, forget;
    Mat12 down12, gf12;
    MatQ4 down4, gf4;
};

struct LayerWeights {
    bool local = false;
    bool nvfp4 = false;
    uint64_t attn_norm = 0, q_norm = 0, k_norm = 0, v_norm = 0, mlp_norm = 0;
    Mat wq, wr, gate_attn, sink, wk, wv, wo, dynk, dynv, router;
    Mat12 q12, k12, v12, wo12, router12;
    MatQ4 q4, k4, v4, wo4;
    uint64_t k_sconv = 0, v_sconv = 0, attn_sconv = 0, mlp_sconv = 0;
    uint64_t q_clip = 0, r_clip = 0, rel_proj = 0, router_bias = 0;
    uint64_t w13_w = 0, w13_s = 0, w13_amax = 0, w13_denom = 0, w13_eb = 0, w13_sb = 0;
    uint64_t w2_w = 0, w2_s = 0, w2_amax = 0, w2_denom = 0, w2_eb = 0, w2_sb = 0;
    bool has_ctl = false;
    std::array<CtlWeights, 4> ctl;
};

struct Config {
    uint32_t hidden, layers, heads, kv_heads, head_dim, d_rel, rel_extent, window, block, experts, top, inter, vocab, padded_vocab;
    uint32_t core_start, core_end, passes, n_floor;
    float clip_qk, clip_rel, alpha, gate_cap, up_cap, route_scale, softcap;
    std::vector<bool> local;
};

NSDictionary* json_object(NSData* data) {
    if (!data) return nil;
    id obj = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    return [obj isKindOfClass:[NSDictionary class]] ? obj : nil;
}

uint32_t u(NSDictionary* d, NSString* k) { return [d[k] unsignedIntValue]; }
float f(NSDictionary* d, NSString* k) { return [d[k] floatValue]; }

class Engine {
public:
    static constexpr uint32_t MAX_BATCH = 16;
    enum LogitsMode { LOGITS_NONE = 0, LOGITS_LAST = 1, LOGITS_ALL = 2 };

    Gpu gpu;
    Config c{};
    std::map<std::string, Tensor> tensors;
    std::vector<LayerWeights> lw;
    uint64_t embed_off = 0, embed_norm_off = 0, final_norm_off = 0;
    Mat unembed_mat{};
    id<MTLBuffer> file = nil;
    id<MTLComputePipelineState> p_gemv, p_norm, p_res, p_sconv, p_clip, p_clip_kv, p_rel, p_attend, p_route;
    id<MTLComputePipelineState> p_qnv4, p_qmxf8, p_env4, p_emxf8, p_swiglu, p_swiglu_qnv4, p_combine;
    id<MTLComputePipelineState> p_ctl_norm, p_ctl_mix_norm, p_embed, p_softcap, p_argmax, p_round;
    id<MTLComputePipelineState> p_ch_build, p_ch_xnorm, p_ch_scan, p_ch_tau, p_ch_filter, p_ch_setup, p_ch_rescore, p_ch_argmax;
    id<MTLComputePipelineState> p_bf12_hist, p_bf12_pack, p_gemv12;
    id<MTLComputePipelineState> p_embed_b, p_norm_b, p_res_b, p_ctl_norm_b, p_ctl_mix_b, p_sconv_b, p_conv_commit;
    id<MTLComputePipelineState> p_clip_b, p_clip_kv_b, p_rel_b, p_attend_b, p_route_b, p_union, p_env4_g, p_emxf8_g, p_combine_b;
    id<MTLComputePipelineState> p_ch_xnorm_b, p_ch_tau_b, p_ch_filter_b, p_ch_setup_b, p_copy_tap;
    std::array<id<MTLComputePipelineState>, 9> p_gemv12_rows, p_gemv16_rows, p_ch_scan_rows, p_gemv_q4_rows;
    id<MTLComputePipelineState> p_mtp_w1_norms, p_mtp_pack_head, p_mtp_q4, p_mtp_fold, p_mtp_fold_b, p_mtp_mix, p_mtp_norm_h, p_mtp_ctx_kv, p_mtp_embed;
    id<MTLComputePipelineState> p_mtp_qk_rope, p_mtp_attn, p_mtp_add_norm, p_mtp_add, p_mtp_silu;
    id<MTLComputePipelineState> p_mtp_base7, p_mtp_rnn_in, p_mtp_rnn_gates, p_mtp_mscan, p_mtp_mred;
    std::array<id<MTLComputePipelineState>, 9> p_mtp_q4_rows;
    id<MTLBuffer> arena, queries, logits, token, scratch;
    id<MTLBuffer> head_q4 = nil, head_sc = nil, head_err = nil;
    id<MTLBuffer> ch_ub = nil, ch_tg_val = nil, ch_tg_row = nil, ch_state = nil, ch_count = nil, ch_cands = nil, ch_indirect = nil;
    id<MTLBuffer> bf12_sm = nil, bf12_ex = nil, bf12_ro = nil, bf12_op = nil;
    id<MTLBuffer> lean_q4 = nil, lean_sc = nil;
    id<MTLBuffer> conv_buf = nil, conv_meta = nil;
    id<MTLBuffer> u_exp = nil, u_cnt = nil, u_pairs = nil, u_ind_w13 = nil, u_ind_w2 = nil;
    id<MTLBuffer> tap_buf = nil;
    bool cert_head_ready = false;
    bool cert_head_active = false;
    bool bf12_ready = false;
    bool lean_q4_ready = false;
    bool mtp_ready = false;
    uint32_t eff_top = 8;
    std::array<double, 9> cost_ms = {0.0, 21.6, 30.1, 35.8, 40.3, 47.9, 57.6, 63.4, 71.8};
    struct MtpQ4Mat { uint64_t q4_off = 0, sc_off = 0; uint32_t rows = 0, cols = 0; };
    struct MtpLayer {
        bool local = false;
        uint64_t dflare_norm = 0, in_norm = 0, post_norm = 0, q_norm = 0, k_norm = 0, k_norm_ctx = 0;
        MtpQ4Mat kv_ctx, qkv, wo, gu, down;
    };
    struct MtpCfg {
        uint32_t hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, inter = 0, block_size = 0, rank = 0, window = 0, dvocab = 0;
        float eps = 1e-6f, log_theta = 0.0f;
    } mc{};
    id<MTLBuffer> mtp_file = nil;
    std::map<std::string, Tensor> mtp_tensors;
    std::vector<MtpLayer> mlw;
    MtpQ4Mat m_rnn_proj{};
    uint64_t m_mask_embed = 0, m_final_norm = 0, m_markov_dflare = 0, m_markov_w1 = 0, m_rnn_bias = 0, m_hidden_out_w = 0, m_hidden_out_b = 0;
    id<MTLBuffer> m_alpha = nil, m_q4 = nil, m_sc = nil, m_vocab_ids = nil, m_head_q4 = nil, m_head_sc = nil;
    id<MTLBuffer> m_fold_q4 = nil, m_fold_sc = nil, m_fold_b = nil;
    std::vector<id<MTLBuffer>> m_kcache, m_vcache;
    id<MTLBuffer> m_f_rnn = nil, m_ctx_x = nil, m_ctx_kv = nil, m_h = nil, m_hn = nil, m_h1 = nil;
    id<MTLBuffer> m_qkv = nil, m_attn_out = nil, m_gu = nil, m_mlp_h = nil, m_base_logits = nil;
    id<MTLBuffer> m_rnn_state = nil, m_rnn_in = nil, m_rnn_proj_out = nil, m_latent = nil;
    id<MTLBuffer> m_tg_val = nil, m_tg_row = nil, m_tg_sum = nil, m_out = nil;
    uint32_t q_stride = 0, k_stride_local = 0, k_stride_global = 0;
    uint64_t s_qbundle, s_kbundle, s_vbundle;
    uint64_t s_q, s_qc, s_r, s_rc, s_gate, s_sinks, s_kraw, s_vraw, s_k, s_v, s_dynk, s_dynv, s_rel, s_gated;
    uint64_t s_router, s_ids, s_gammas, s_xq, s_outer, s_gu, s_h, s_hq, s_houter, s_rows;
    uint64_t s_normed[3], s_ctlh[3], s_gf[3], s_gain[3], s_forget[3];
    std::vector<id<MTLBuffer>> kcache, vcache;
    std::vector<uint32_t> kv_cap, m_kv_cap;
    std::vector<std::array<uint64_t, 4>> conv_off;
    uint32_t conv_count = 0;
    uint32_t pending_conv_commit = 0;
    uint32_t max_context = 32768;
    uint32_t next_slot = 0;
    uint32_t slots = 1024;
    std::vector<uint64_t> q_attn_off, q_mlp_off, q_loop_attn_off, q_loop_mlp_off;
    uint64_t q_output_off = 0;
    id<MTLComputeCommandEncoder> enc = nil;
    bool together = false;

    const Tensor& T(const std::string& name) const { return tensors.at("model.llm." + name); }

    Mat make_mat(const std::string& name) const {
        const Tensor& t = T(name);
        return {t.offset, uint32_t(t.shape[0]), uint32_t(t.bytes / 2 / t.shape[0])};
    }

    bool load(const std::string& dir, bool use_cert_head, bool use_bf12, bool want_mtp, const std::string& mode, std::string& error) {
        auto opened = open_gpu(error);
        if (!opened) return false;
        gpu = *opened;
        NSString* base = [NSString stringWithUTF8String:dir.c_str()];
        NSDictionary* cfg = json_object([NSData dataWithContentsOfFile:[base stringByAppendingPathComponent:@"config.json"]]);
        NSDictionary* t = cfg[@"text_config"];
        if (!t) { error = "config.json has no text_config"; return false; }
        NSDictionary* loop = t[@"loop_config"];
        c = {u(t, @"hidden_size"), u(t, @"num_hidden_layers"), u(t, @"num_attention_heads"), u(t, @"num_key_value_heads"), u(t, @"head_dim"),
             u(t, @"d_rel"), u(t, @"rel_extent"), u(t, @"sliding_window_size"), u(t, @"n_layers_per_attn_res_block"), u(t, @"n_routed_experts"),
             u(t, @"num_experts_per_tok"), u(t, @"intermediate_size"), u(t, @"unpadded_vocab_size"), u(t, @"vocab_size"),
             u(loop, @"core_start"), u(loop, @"core_end"), u(loop, @"num_passes"), u(t, @"log_scaling_n_floor"),
             f(t, @"qk_l2_clip_threshold"), f(t, @"rel_l2_clip_threshold"), f(t, @"log_scaling_alpha"), f(t, @"mlp_gate_softcap"),
             f(t, @"mlp_up_softcap"), f(t, @"route_scale"), f(t, @"final_logit_softcapping"), {}};
        c.local.assign(c.layers, false);
        if (c.top != 8) { error = "only num_experts_per_tok == 8 is supported"; return false; }
        if (c.inter > 2048) { error = "intermediate_size > 2048 is not supported"; return false; }
        for (NSNumber* n in t[@"local_layer_ids"]) {
            long long lid = n.longLongValue;
            if (lid < 0 || static_cast<uint64_t>(lid) >= c.layers) { error = "local_layer_ids contains out-of-range layer index"; return false; }
            c.local[static_cast<size_t>(lid)] = true;
        }
        NSArray* files = [[[NSFileManager defaultManager] contentsOfDirectoryAtPath:base error:nil] filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"SELF BEGINSWITH 'model' AND SELF ENDSWITH '.safetensors'"]];
        if (files.count != 1) { error = "expected one model*.safetensors file"; return false; }
        std::string path = std::string(base.UTF8String) + "/" + [files[0] UTF8String];
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { error = "failed to open checkpoint: " + path; return false; }
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 8) { ::close(fd); error = "failed to stat checkpoint or file too small: " + path; return false; }
        uint64_t file_size = static_cast<uint64_t>(st.st_size);
        size_t page = getpagesize();
        size_t mapped = (file_size + page - 1) / page * page;
        void* data = mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (data == MAP_FAILED) { error = "mmap failed"; return false; }
        if (mapped > gpu.device.maxBufferLength) { munmap(data, mapped); error = "model file larger than maxBufferLength"; return false; }
        uint64_t header = *static_cast<uint64_t*>(data);
        if (header > file_size - 8) { munmap(data, mapped); error = "checkpoint header exceeds file size"; return false; }
        NSDictionary* h = json_object([NSData dataWithBytesNoCopy:static_cast<char*>(data) + 8 length:header freeWhenDone:NO]);
        if (![h isKindOfClass:[NSDictionary class]]) { munmap(data, mapped); error = "invalid safetensors header json"; return false; }
        uint64_t data_base = 8 + header;
        for (NSString* name in h) {
            if ([name isEqualToString:@"__metadata__"]) continue;
            NSDictionary* info = h[name];
            NSArray* offs = info[@"data_offsets"];
            if (![offs isKindOfClass:[NSArray class]] || offs.count != 2) { munmap(data, mapped); error = "invalid tensor offsets"; return false; }
            uint64_t start = [offs[0] unsignedLongLongValue];
            uint64_t end = [offs[1] unsignedLongLongValue];
            if (end < start || end > file_size - data_base) { munmap(data, mapped); error = "checkpoint tensor span exceeds file size"; return false; }
            Tensor tensor;
            tensor.offset = data_base + start;
            tensor.bytes = end - start;
            tensor.dtype = [info[@"dtype"] UTF8String];
            for (NSNumber* n in info[@"shape"]) tensor.shape.push_back(n.unsignedLongLongValue);
            if (tensor.dtype == "BF16" && tensor.shape.size() == 2 && (tensor.offset % 16) != 0) {
                munmap(data, mapped);
                error = std::string("BF16 matrix not 16-byte aligned: ") + name.UTF8String;
                return false;
            }
            tensors[name.UTF8String] = tensor;
        }
        file = [gpu.device newBufferWithBytesNoCopy:data length:mapped options:MTLResourceStorageModeShared deallocator:nil];
        if (!file) { munmap(data, mapped); error = "newBufferWithBytesNoCopy failed"; return false; }
        p_gemv = pipeline(gpu, @"gemv_bf16");
        p_norm = pipeline(gpu, @"rms_norm");
        p_res = pipeline(gpu, @"attn_res");
        p_sconv = pipeline(gpu, @"sconv");
        p_clip = pipeline(gpu, @"clip_rows");
        p_clip_kv = pipeline(gpu, @"clip_kv_store");
        p_rel = pipeline(gpu, @"rel_bias");
        p_attend = pipeline(gpu, @"attend_gated");
        p_route = pipeline(gpu, @"route");
        p_qnv4 = pipeline(gpu, @"quant_nvfp4");
        p_qmxf8 = pipeline(gpu, @"quant_mxfp8");
        p_env4 = pipeline(gpu, @"expert_nvfp4");
        p_emxf8 = pipeline(gpu, @"expert_mxfp8");
        p_swiglu = pipeline(gpu, @"swiglu");
        p_swiglu_qnv4 = pipeline(gpu, @"swiglu_quant_nvfp4");
        p_combine = pipeline(gpu, @"combine");
        p_ctl_norm = pipeline(gpu, @"ctl_normed");
        p_ctl_mix_norm = pipeline(gpu, @"ctl_mix_norm");
        p_embed = pipeline(gpu, @"embed_row");
        p_softcap = pipeline(gpu, @"softcap_logits");
        p_argmax = pipeline(gpu, @"argmax_logits");
        p_round = pipeline(gpu, @"round_rows");
        p_ch_build = pipeline(gpu, @"build_cert_head");
        p_ch_xnorm = pipeline(gpu, @"cert_head_xnorm");
        p_ch_scan = pipeline(gpu, @"cert_head_scan");
        p_ch_tau = pipeline(gpu, @"cert_head_tau");
        p_ch_filter = pipeline(gpu, @"cert_head_filter");
        p_ch_setup = pipeline(gpu, @"cert_head_setup");
        p_ch_rescore = pipeline(gpu, @"cert_head_rescore");
        p_ch_argmax = pipeline(gpu, @"cert_head_argmax");
        p_bf12_hist = pipeline(gpu, @"bf12_hist");
        p_bf12_pack = pipeline(gpu, @"bf12_pack");
        p_gemv12 = pipeline(gpu, @"gemv_bf12");
        p_embed_b = pipeline(gpu, @"embed_rows");
        p_norm_b = pipeline(gpu, @"rms_norm_batch");
        p_res_b = pipeline(gpu, @"attn_res_batch");
        p_ctl_norm_b = pipeline(gpu, @"ctl_normed_batch");
        p_ctl_mix_b = pipeline(gpu, @"ctl_mix_norm_batch");
        p_sconv_b = pipeline(gpu, @"sconv_batch");
        p_conv_commit = pipeline(gpu, @"conv_commit_all");
        p_clip_b = pipeline(gpu, @"clip_rows_batch");
        p_clip_kv_b = pipeline(gpu, @"clip_kv_store_batch");
        p_rel_b = pipeline(gpu, @"rel_bias_batch");
        p_attend_b = pipeline(gpu, @"attend_gated_batch");
        p_route_b = pipeline(gpu, @"route_batch");
        p_union = pipeline(gpu, @"build_expert_union");
        p_env4_g = pipeline(gpu, @"expert_nvfp4_grouped");
        p_emxf8_g = pipeline(gpu, @"expert_mxfp8_grouped");
        p_combine_b = pipeline(gpu, @"combine_batch");
        p_ch_xnorm_b = pipeline(gpu, @"cert_head_xnorm_batch");
        p_ch_tau_b = pipeline(gpu, @"cert_head_tau_batch");
        p_ch_filter_b = pipeline(gpu, @"cert_head_filter_batch");
        p_ch_setup_b = pipeline(gpu, @"cert_head_setup_batch");
        p_copy_tap = pipeline(gpu, @"copy_tap_batch");
        p_mtp_w1_norms = pipeline(gpu, @"mtp_w1_norms");
        p_mtp_pack_head = pipeline(gpu, @"mtp_pack_head_sub");
        p_mtp_q4 = pipeline(gpu, @"mtp_quant_q4");
        p_mtp_fold = pipeline(gpu, @"mtp_build_fold");
        p_mtp_fold_b = pipeline(gpu, @"mtp_build_fold_bias");
        p_mtp_mix = pipeline(gpu, @"mtp_mix_rmsnorm_h");
        p_mtp_norm_h = pipeline(gpu, @"mtp_rmsnorm_h");
        p_mtp_ctx_kv = pipeline(gpu, @"mtp_ctx_kv_store");
        p_mtp_embed = pipeline(gpu, @"mtp_embed_block");
        p_mtp_qk_rope = pipeline(gpu, @"mtp_qk_norm_rope");
        p_mtp_attn = pipeline(gpu, @"mtp_attend_block");
        p_mtp_add_norm = pipeline(gpu, @"mtp_add_rmsnorm");
        p_mtp_add = pipeline(gpu, @"mtp_add_inplace");
        p_mtp_silu = pipeline(gpu, @"mtp_silu_mul");
        p_mtp_base7 = pipeline(gpu, @"mtp_base_logits_7");
        p_mtp_rnn_in = pipeline(gpu, @"mtp_rnn_input");
        p_mtp_rnn_gates = pipeline(gpu, @"mtp_rnn_gates");
        p_mtp_mscan = pipeline(gpu, @"mtp_markov_scan");
        p_mtp_mred = pipeline(gpu, @"mtp_markov_reduce");
        for (uint32_t r = 1; r <= 8; ++r) {
            p_gemv12_rows[r] = pipeline_for_rows(gpu, @"gemv_bf12_rows", r);
            p_gemv16_rows[r] = pipeline_for_rows(gpu, @"gemv_bf16_rows", r);
            p_ch_scan_rows[r] = pipeline_for_rows(gpu, @"cert_head_scan_rows", r);
            p_gemv_q4_rows[r] = pipeline_for_rows(gpu, @"gemv_q4_f32_rows", r);
            p_mtp_q4_rows[r] = pipeline_for_rows(gpu, @"mtp_gemv_q4_rows", r);
        }
        arena = [gpu.device newBufferWithLength:uint64_t(slots) * MAX_BATCH * c.hidden * 4 options:MTLResourceStorageModePrivate];
        if (!arena) { error = "arena allocation failed"; return false; }
        build_queries();
        if (!queries) { error = "queries allocation failed"; return false; }
        resolve_weights();
        logits = [gpu.device newBufferWithLength:uint64_t(c.padded_vocab) * 4 options:MTLResourceStorageModeShared];
        token = [gpu.device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        u_exp = [gpu.device newBufferWithLength:128 * 4 options:MTLResourceStorageModePrivate];
        u_cnt = [gpu.device newBufferWithLength:128 * 4 options:MTLResourceStorageModePrivate];
        u_pairs = [gpu.device newBufferWithLength:128 * MAX_BATCH * 4 options:MTLResourceStorageModePrivate];
        u_ind_w13 = [gpu.device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
        u_ind_w2 = [gpu.device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
        tap_buf = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * 4 * c.hidden * 4 options:MTLResourceStorageModePrivate];
        if (!logits || !token || !u_exp || !u_cnt || !u_pairs || !u_ind_w13 || !u_ind_w2 || !tap_buf) { error = "workspace allocation failed"; return false; }
        uint32_t H = c.hidden, I = c.inter, K = c.top;
        uint64_t at = 0;
        auto slot_alloc = [&](uint64_t count) {
            uint64_t off = at;
            at += (count * 4 + 255) / 256 * 256;
            return off;
        };
        q_stride = c.heads * c.head_dim * 2 + c.heads * c.d_rel + c.heads;
        k_stride_local = c.kv_heads * c.head_dim;
        k_stride_global = k_stride_local + c.kv_heads * 4;
        s_qbundle = slot_alloc(uint64_t(MAX_BATCH) * q_stride);
        s_q = s_qbundle;
        s_r = s_q + uint64_t(c.heads * c.head_dim) * 4;
        s_gate = s_r + uint64_t(c.heads * c.d_rel) * 4;
        s_sinks = s_gate + uint64_t(c.heads * c.head_dim) * 4;
        s_qc = slot_alloc(uint64_t(MAX_BATCH) * c.heads * c.head_dim);
        s_rc = slot_alloc(uint64_t(MAX_BATCH) * c.heads * c.d_rel);
        s_kbundle = slot_alloc(uint64_t(MAX_BATCH) * k_stride_global);
        s_kraw = s_kbundle;
        s_dynk = s_kraw + uint64_t(k_stride_local) * 4;
        s_vbundle = slot_alloc(uint64_t(MAX_BATCH) * k_stride_global);
        s_vraw = s_vbundle;
        s_dynv = s_vraw + uint64_t(k_stride_local) * 4;
        s_k = slot_alloc(uint64_t(MAX_BATCH) * k_stride_local);
        s_v = slot_alloc(uint64_t(MAX_BATCH) * k_stride_local);
        s_rel = slot_alloc(uint64_t(MAX_BATCH) * c.heads * c.rel_extent);
        s_gated = slot_alloc(uint64_t(MAX_BATCH) * c.heads * c.head_dim);
        s_router = slot_alloc(uint64_t(MAX_BATCH) * c.experts);
        s_ids = slot_alloc(uint64_t(MAX_BATCH) * 16);
        s_gammas = slot_alloc(uint64_t(MAX_BATCH) * 16);
        s_xq = slot_alloc(uint64_t(MAX_BATCH) * H);
        s_outer = slot_alloc(uint64_t(MAX_BATCH) * 4);
        s_gu = slot_alloc(uint64_t(MAX_BATCH) * K * 2 * I);
        s_h = slot_alloc(uint64_t(MAX_BATCH) * K * I);
        s_hq = slot_alloc(uint64_t(MAX_BATCH) * K * I);
        s_houter = slot_alloc(uint64_t(MAX_BATCH) * K);
        s_rows = slot_alloc(uint64_t(MAX_BATCH) * K * H);
        for (int k = 0; k < 3; ++k) {
            s_normed[k] = slot_alloc(uint64_t(MAX_BATCH) * 2 * H);
            s_ctlh[k] = slot_alloc(uint64_t(MAX_BATCH) * 64);
            s_gf[k] = slot_alloc(uint64_t(MAX_BATCH) * 2 * H);
            s_gain[k] = s_gf[k];
            s_forget[k] = s_gf[k] + uint64_t(H) * 4;
        }
        scratch = [gpu.device newBufferWithLength:at options:MTLResourceStorageModePrivate];
        if (!scratch) { error = "scratch allocation failed"; return false; }
        uint32_t visits = c.layers + (c.passes - 1) * (c.core_end - c.core_start);
        conv_off.resize(visits);
        struct ConvMetaHost { uint32_t offset_f, width; };
        std::vector<ConvMetaHost> cmeta;
        uint64_t conv_floats = 0;
        uint32_t core_len = c.core_end > c.core_start ? (c.core_end - c.core_start) : 1;
        for (uint32_t v = 0; v < visits; ++v) {
            uint32_t lid = v < c.layers ? v : (c.core_start + (v - c.layers) % core_len);
            uint32_t cap = c.local[lid] ? std::min<uint32_t>(max_context, c.window + MAX_BATCH * 2) : max_context;
            kv_cap.push_back(cap);
            id<MTLBuffer> kb = [gpu.device newBufferWithLength:uint64_t(cap) * c.kv_heads * c.head_dim * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> vb = [gpu.device newBufferWithLength:uint64_t(cap) * c.kv_heads * c.head_dim * 2 options:MTLResourceStorageModePrivate];
            if (!kb || !vb) { error = "kv cache allocation failed"; return false; }
            kcache.push_back(kb);
            vcache.push_back(vb);
            for (int s = 0; s < 4; ++s) {
                uint32_t width = s < 2 ? c.kv_heads * c.head_dim : c.hidden;
                conv_off[v][s] = conv_floats * 4;
                cmeta.push_back({uint32_t(conv_floats), width});
                conv_floats += ((3u + MAX_BATCH) * width + 63u) / 64u * 64u;
            }
        }
        conv_count = uint32_t(cmeta.size());
        conv_buf = [gpu.device newBufferWithLength:conv_floats * 4 options:MTLResourceStorageModePrivate];
        conv_meta = [gpu.device newBufferWithBytes:cmeta.data() length:cmeta.size() * sizeof(ConvMetaHost) options:MTLResourceStorageModeShared];
        if (!conv_buf || !conv_meta) { error = "conv state allocation failed"; return false; }
        reset_state();
        eff_top = (mode == "turbo") ? std::min<uint32_t>(4u, c.top) : c.top;
        if (mode == "lean") cost_ms = {0.0, 17.1, 25.5, 31.2, 35.7, 43.3, 53.0, 58.8, 67.2};
        else if (mode == "turbo") cost_ms = {0.0, 13.2, 18.8, 23.0, 26.8, 32.0, 38.2, 43.0, 48.5};
        if ((use_cert_head || want_mtp) && (c.hidden % 32 == 0) && c.vocab >= 64) {
            if (!build_cert_head(error)) return false;
            cert_head_active = use_cert_head;
        }
        if (mode == "lean" || mode == "turbo") {
            if ((c.hidden % 32 == 0) && ((c.heads * c.head_dim) % 32 == 0)) {
                if (!build_lean_q4(error)) return false;
            }
        } else if (use_bf12 && (c.hidden % 16 == 0) && ((c.heads * c.head_dim) % 16 == 0) && ((c.heads * c.d_rel) % 4 == 0)) {
            if (!build_bf12(error)) return false;
        }
        if (want_mtp && cert_head_ready) {
            if (!load_mtp(dir, cfg, error)) return false;
        }
        return true;
    }

    bool build_lean_q4(std::string& error) {
        struct Q4Job { MatQ4* dst; std::vector<Mat> srcs; uint32_t rows = 0, cols = 0; };
        std::vector<Q4Job> jobs;
        auto add_job = [&](MatQ4* dst, std::initializer_list<Mat> mats) {
            Q4Job j;
            j.dst = dst;
            for (const Mat& m : mats) {
                j.srcs.push_back(m);
                j.rows += m.rows;
                j.cols = m.cols;
            }
            jobs.push_back(std::move(j));
        };
        for (uint32_t i = 0; i < c.layers; ++i) {
            LayerWeights& w = lw[i];
            add_job(&w.q4, {w.wq, w.wr, w.gate_attn, w.sink});
            if (w.local) {
                add_job(&w.k4, {w.wk});
                add_job(&w.v4, {w.wv});
            } else {
                add_job(&w.k4, {w.wk, w.dynk});
                add_job(&w.v4, {w.wv, w.dynv});
            }
            add_job(&w.wo4, {w.wo});
            if (w.has_ctl) {
                for (int k = 0; k < 4; ++k) {
                    add_job(&w.ctl[k].down4, {w.ctl[k].down});
                    add_job(&w.ctl[k].gf4, {w.ctl[k].gain, w.ctl[k].forget});
                }
            }
        }
        auto align256 = [](uint64_t v) { return (v + 255) / 256 * 256; };
        uint64_t q4_tot = 0, sc_tot = 0;
        for (Q4Job& j : jobs) {
            uint32_t ng = j.cols / 32;
            j.dst->q4_off = q4_tot;
            j.dst->sc_off = sc_tot;
            j.dst->rows = j.rows;
            j.dst->cols = j.cols;
            q4_tot += align256(uint64_t(j.rows) * ng * 16);
            sc_tot += align256(uint64_t(j.rows) * ng * 2);
        }
        lean_q4 = [gpu.device newBufferWithLength:q4_tot options:MTLResourceStorageModePrivate];
        lean_sc = [gpu.device newBufferWithLength:sc_tot options:MTLResourceStorageModePrivate];
        if (!lean_q4 || !lean_sc) { error = "lean q4 buffer allocation failed"; return false; }
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        struct MtpQ4BuildP { uint32_t rows, cols, ng, row_off; };
        [e setComputePipelineState:p_mtp_q4];
        for (const Q4Job& j : jobs) {
            uint32_t roff = 0;
            uint32_t ng = j.cols / 32;
            for (const Mat& m : j.srcs) {
                MtpQ4BuildP bp{m.rows, j.cols, ng, roff};
                [e setBuffer:file offset:m.offset atIndex:0];
                [e setBuffer:lean_q4 offset:j.dst->q4_off atIndex:1];
                [e setBuffer:lean_sc offset:j.dst->sc_off atIndex:2];
                [e setBytes:&bp length:sizeof(bp) atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(m.rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                roff += m.rows;
            }
        }
        [e endEncoding];
        gpu_seconds(cb);
        lean_q4_ready = true;
        return true;
    }

    bool load_mtp(const std::string& dir, NSDictionary* cfg, std::string& error) {
        NSDictionary* m = cfg[@"mtp_config"];
        if (![m isKindOfClass:[NSDictionary class]]) return true;
        std::string path = dir + "/mtp.safetensors";
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return true;
        struct stat st;
        if (fstat(fd, &st) != 0 || st.st_size < 8) { ::close(fd); return true; }
        uint64_t file_size = static_cast<uint64_t>(st.st_size);
        size_t page = getpagesize();
        size_t mapped = (file_size + page - 1) / page * page;
        void* data = mmap(nullptr, mapped, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (data == MAP_FAILED) { error = "mtp mmap failed"; return false; }
        uint64_t header = *static_cast<uint64_t*>(data);
        if (header > file_size - 8) { munmap(data, mapped); error = "mtp header exceeds file size"; return false; }
        NSDictionary* h = json_object([NSData dataWithBytesNoCopy:static_cast<char*>(data) + 8 length:header freeWhenDone:NO]);
        if (![h isKindOfClass:[NSDictionary class]]) { munmap(data, mapped); error = "invalid mtp safetensors header"; return false; }
        uint64_t data_base = 8 + header;
        for (NSString* name in h) {
            if ([name isEqualToString:@"__metadata__"]) continue;
            NSDictionary* info = h[name];
            NSArray* offs = info[@"data_offsets"];
            if (![offs isKindOfClass:[NSArray class]] || offs.count != 2) { munmap(data, mapped); error = "invalid mtp tensor offsets"; return false; }
            uint64_t start = [offs[0] unsignedLongLongValue];
            uint64_t end = [offs[1] unsignedLongLongValue];
            if (end < start || end > file_size - data_base) { munmap(data, mapped); error = "mtp tensor span exceeds file size"; return false; }
            Tensor tensor;
            tensor.offset = data_base + start;
            tensor.bytes = end - start;
            tensor.dtype = [info[@"dtype"] UTF8String];
            for (NSNumber* n in info[@"shape"]) tensor.shape.push_back(n.unsignedLongLongValue);
            mtp_tensors[name.UTF8String] = tensor;
        }
        mtp_file = [gpu.device newBufferWithBytesNoCopy:data length:mapped options:MTLResourceStorageModeShared deallocator:nil];
        if (!mtp_file) { munmap(data, mapped); error = "mtp metal buffer failed"; return false; }
        mc.hidden = u(m, @"hidden_size");
        mc.layers = u(m, @"num_hidden_layers");
        mc.heads = u(m, @"num_attention_heads");
        mc.kv_heads = u(m, @"num_key_value_heads");
        mc.head_dim = u(m, @"head_dim");
        mc.inter = u(m, @"intermediate_size");
        mc.block_size = u(m, @"block_size");
        mc.rank = u(m, @"markov_rank");
        mc.window = u(m, @"sliding_window_size");
        mc.eps = f(m, @"rms_norm_eps");
        mc.log_theta = std::log(f(m, @"rope_theta"));
        if (mc.hidden != c.hidden || mc.block_size != 8 || mc.rank != 256 || mc.head_dim != 128 || mc.layers == 0) return true;
        auto require_mt = [&](const std::string& n) -> bool {
            if (mtp_tensors.find(n) != mtp_tensors.end()) return true;
            error = "missing mtp tensor: " + n;
            return false;
        };
        for (const std::string& n : std::vector<std::string>{
                 "fuse_weight", "mask_embed", "norm.weight", "markov_head.dflare_norm.weight",
                 "markov_head.markov_w1.weight", "markov_head.rnn_proj.bias", "markov_head.rnn_proj.weight",
                 "markov_head.hidden_out.weight", "markov_head.hidden_out.bias"}) {
            if (!require_mt(n)) return false;
        }
        for (uint32_t i = 0; i < mc.layers; ++i) {
            std::string L = "layers." + std::to_string(i) + ".";
            for (const std::string& s : std::vector<std::string>{
                     "dflare_norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
                     "self_attn.q_norm.weight", "self_attn.k_norm.weight", "self_attn.k_norm_ctx.weight",
                     "self_attn.k_proj_ctx.weight", "self_attn.v_proj_ctx.weight", "self_attn.q_proj.weight",
                     "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                     "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"}) {
                if (!require_mt(L + s)) return false;
            }
        }
        auto MT = [&](const std::string& n) -> const Tensor& { return mtp_tensors.at(n); };
        auto bf16_to_f32 = [](uint16_t b) {
            uint32_t bits = uint32_t(b) << 16;
            float v;
            std::memcpy(&v, &bits, 4);
            return v;
        };
        const Tensor& fw = MT("fuse_weight");
        const uint16_t* fwp = reinterpret_cast<const uint16_t*>(static_cast<const char*>(mtp_file.contents) + fw.offset);
        std::vector<float> alpha((mc.layers + 1) * 4, 0.0f);
        for (uint32_t r = 0; r < mc.layers + 1; ++r) {
            float mx = -INFINITY;
            float vals[4];
            for (uint32_t j = 0; j < 4; ++j) { vals[j] = bf16_to_f32(fwp[r * 4 + j]); mx = std::max(mx, vals[j]); }
            float s = 0.0f;
            for (uint32_t j = 0; j < 4; ++j) { vals[j] = std::exp(vals[j] - mx); s += vals[j]; }
            for (uint32_t j = 0; j < 4; ++j) alpha[r * 4 + j] = vals[j] / s;
        }
        m_alpha = [gpu.device newBufferWithBytes:alpha.data() length:alpha.size() * 4 options:MTLResourceStorageModeShared];
        m_mask_embed = MT("mask_embed").offset;
        m_final_norm = MT("norm.weight").offset;
        m_markov_dflare = MT("markov_head.dflare_norm.weight").offset;
        m_markov_w1 = MT("markov_head.markov_w1.weight").offset;
        m_rnn_bias = MT("markov_head.rnn_proj.bias").offset;
        m_hidden_out_w = MT("markov_head.hidden_out.weight").offset;
        m_hidden_out_b = MT("markov_head.hidden_out.bias").offset;
        mlw.assign(mc.layers, {});
        for (NSNumber* n in m[@"local_layer_ids"]) {
            uint32_t lid = n.unsignedIntValue;
            if (lid < mc.layers) mlw[lid].local = true;
        }
        struct Q4Job { MtpQ4Mat* dst; std::vector<std::string> names; uint32_t rows = 0, cols = 0; };
        std::vector<Q4Job> jobs;
        auto add_q4 = [&](MtpQ4Mat* dst, std::initializer_list<std::string> names) {
            Q4Job j;
            j.dst = dst;
            for (const std::string& n : names) {
                const Tensor& t = MT(n);
                j.names.push_back(n);
                j.rows += uint32_t(t.shape[0]);
                j.cols = uint32_t(t.shape[1]);
            }
            jobs.push_back(std::move(j));
        };
        for (uint32_t i = 0; i < mc.layers; ++i) {
            std::string L = "layers." + std::to_string(i) + ".";
            MtpLayer& ml = mlw[i];
            ml.dflare_norm = MT(L + "dflare_norm.weight").offset;
            ml.in_norm = MT(L + "input_layernorm.weight").offset;
            ml.post_norm = MT(L + "post_attention_layernorm.weight").offset;
            ml.q_norm = MT(L + "self_attn.q_norm.weight").offset;
            ml.k_norm = MT(L + "self_attn.k_norm.weight").offset;
            ml.k_norm_ctx = MT(L + "self_attn.k_norm_ctx.weight").offset;
            add_q4(&ml.kv_ctx, {L + "self_attn.k_proj_ctx.weight", L + "self_attn.v_proj_ctx.weight"});
            add_q4(&ml.qkv, {L + "self_attn.q_proj.weight", L + "self_attn.k_proj.weight", L + "self_attn.v_proj.weight"});
            add_q4(&ml.wo, {L + "self_attn.o_proj.weight"});
            add_q4(&ml.gu, {L + "mlp.gate_proj.weight", L + "mlp.up_proj.weight"});
            add_q4(&ml.down, {L + "mlp.down_proj.weight"});
        }
        add_q4(&m_rnn_proj, {"markov_head.rnn_proj.weight"});
        auto align256 = [](uint64_t v) { return (v + 255) / 256 * 256; };
        uint64_t q4_tot = 0, sc_tot = 0;
        for (Q4Job& j : jobs) {
            uint32_t ng = j.cols / 32;
            j.dst->q4_off = q4_tot;
            j.dst->sc_off = sc_tot;
            j.dst->rows = j.rows;
            j.dst->cols = j.cols;
            q4_tot += align256(uint64_t(j.rows) * ng * 16);
            sc_tot += align256(uint64_t(j.rows) * ng * 2);
        }
        m_q4 = [gpu.device newBufferWithLength:q4_tot options:MTLResourceStorageModePrivate];
        m_sc = [gpu.device newBufferWithLength:sc_tot options:MTLResourceStorageModePrivate];
        uint32_t V = c.vocab;
        uint32_t DV = std::min<uint32_t>(V, 49152u);
        mc.dvocab = DV;
        id<MTLBuffer> norms_buf = [gpu.device newBufferWithLength:uint64_t(V) * 4 options:MTLResourceStorageModeShared];
        if (!norms_buf) { error = "mtp norms buffer allocation failed"; return false; }
        {
            struct MtpNormPP { uint32_t vocab, rank; };
            MtpNormPP np{V, mc.rank};
            id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:p_mtp_w1_norms];
            [e setBuffer:mtp_file offset:m_markov_w1 atIndex:0];
            [e setBuffer:norms_buf offset:0 atIndex:1];
            [e setBytes:&np length:sizeof(np) atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake((V + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];
            gpu_seconds(cb);
        }
        const float* nptr = static_cast<const float*>(norms_buf.contents);
        std::vector<uint32_t> idx(V);
        for (uint32_t i = 0; i < V; ++i) idx[i] = i;
        if (DV < V) {
            std::nth_element(idx.begin(), idx.begin() + DV, idx.end(), [&](uint32_t a, uint32_t b) { return nptr[a] > nptr[b]; });
            idx.resize(DV);
            std::sort(idx.begin(), idx.end());
        }
        m_vocab_ids = [gpu.device newBufferWithBytes:idx.data() length:uint64_t(DV) * 4 options:MTLResourceStorageModeShared];
        uint32_t ng_h = c.hidden / 32;
        m_head_q4 = [gpu.device newBufferWithLength:uint64_t(DV) * ng_h * 16 options:MTLResourceStorageModePrivate];
        m_head_sc = [gpu.device newBufferWithLength:uint64_t(DV) * ng_h * 2 options:MTLResourceStorageModePrivate];
        m_fold_q4 = [gpu.device newBufferWithLength:uint64_t(DV) * 8 * 16 options:MTLResourceStorageModePrivate];
        m_fold_sc = [gpu.device newBufferWithLength:uint64_t(DV) * 8 * 2 options:MTLResourceStorageModePrivate];
        m_fold_b = [gpu.device newBufferWithLength:uint64_t(DV) * 4 options:MTLResourceStorageModePrivate];
        uint32_t kvw = mc.kv_heads * mc.head_dim;
        uint32_t qdim = mc.heads * mc.head_dim;
        for (uint32_t i = 0; i < mc.layers; ++i) {
            uint32_t mcap = mlw[i].local ? std::min<uint32_t>(max_context, mc.window + MAX_BATCH * 2) : max_context;
            m_kv_cap.push_back(mcap);
            id<MTLBuffer> kb = [gpu.device newBufferWithLength:uint64_t(mcap) * kvw * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> vb = [gpu.device newBufferWithLength:uint64_t(mcap) * kvw * 2 options:MTLResourceStorageModePrivate];
            if (!kb || !vb) { error = "mtp kv cache allocation failed"; return false; }
            m_kcache.push_back(kb);
            m_vcache.push_back(vb);
        }
        m_f_rnn = [gpu.device newBufferWithLength:uint64_t(mc.hidden) * 2 options:MTLResourceStorageModePrivate];
        m_ctx_x = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * mc.hidden * 2 options:MTLResourceStorageModePrivate];
        m_ctx_kv = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * 2 * kvw * 4 options:MTLResourceStorageModePrivate];
        m_h = [gpu.device newBufferWithLength:uint64_t(8) * mc.hidden * 4 options:MTLResourceStorageModePrivate];
        m_hn = [gpu.device newBufferWithLength:uint64_t(8) * mc.hidden * 2 options:MTLResourceStorageModePrivate];
        m_h1 = [gpu.device newBufferWithLength:uint64_t(8) * mc.hidden * 4 options:MTLResourceStorageModePrivate];
        m_qkv = [gpu.device newBufferWithLength:uint64_t(8) * (qdim + 2 * kvw) * 4 options:MTLResourceStorageModePrivate];
        m_attn_out = [gpu.device newBufferWithLength:uint64_t(8) * qdim * 2 options:MTLResourceStorageModePrivate];
        m_gu = [gpu.device newBufferWithLength:uint64_t(8) * 2 * mc.inter * 4 options:MTLResourceStorageModePrivate];
        m_mlp_h = [gpu.device newBufferWithLength:uint64_t(8) * mc.inter * 2 options:MTLResourceStorageModePrivate];
        m_base_logits = [gpu.device newBufferWithLength:uint64_t(7) * DV * 2 options:MTLResourceStorageModePrivate];
        m_rnn_state = [gpu.device newBufferWithLength:uint64_t(mc.rank) * 4 options:MTLResourceStorageModePrivate];
        m_rnn_in = [gpu.device newBufferWithLength:uint64_t(2 * mc.rank + 2 * mc.hidden) * 2 options:MTLResourceStorageModePrivate];
        m_rnn_proj_out = [gpu.device newBufferWithLength:uint64_t(3 * mc.rank) * 4 options:MTLResourceStorageModePrivate];
        m_latent = [gpu.device newBufferWithLength:uint64_t(mc.rank) * 4 options:MTLResourceStorageModePrivate];
        uint32_t mntg = (DV + 31) / 32;
        m_tg_val = [gpu.device newBufferWithLength:uint64_t(mntg) * 4 options:MTLResourceStorageModePrivate];
        m_tg_row = [gpu.device newBufferWithLength:uint64_t(mntg) * 4 options:MTLResourceStorageModePrivate];
        m_tg_sum = [gpu.device newBufferWithLength:uint64_t(mntg) * 4 options:MTLResourceStorageModePrivate];
        m_out = [gpu.device newBufferWithLength:128 options:MTLResourceStorageModeShared];
        if (!m_alpha || !m_q4 || !m_sc || !m_vocab_ids || !m_head_q4 || !m_head_sc || !m_fold_q4 || !m_fold_sc || !m_fold_b
            || !m_f_rnn || !m_ctx_x || !m_ctx_kv || !m_h || !m_hn || !m_h1 || !m_qkv || !m_attn_out || !m_gu || !m_mlp_h
            || !m_base_logits || !m_rnn_state || !m_rnn_in || !m_rnn_proj_out || !m_latent || !m_tg_val || !m_tg_row || !m_tg_sum || !m_out) {
            error = "mtp buffer allocation failed";
            return false;
        }
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        struct MtpQ4BuildP { uint32_t rows, cols, ng, row_off; };
        [e setComputePipelineState:p_mtp_q4];
        for (const Q4Job& j : jobs) {
            uint32_t roff = 0;
            uint32_t ng = j.cols / 32;
            for (const std::string& n : j.names) {
                const Tensor& t = MT(n);
                uint32_t r = uint32_t(t.shape[0]);
                MtpQ4BuildP bp{r, j.cols, ng, roff};
                [e setBuffer:mtp_file offset:t.offset atIndex:0];
                [e setBuffer:m_q4 offset:j.dst->q4_off atIndex:1];
                [e setBuffer:m_sc offset:j.dst->sc_off atIndex:2];
                [e setBytes:&bp length:sizeof(bp) atIndex:3];
                [e dispatchThreadgroups:MTLSizeMake(r, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                roff += r;
            }
        }
        struct MtpPackHeadPP { uint32_t dvocab, ng; };
        MtpPackHeadPP php{DV, ng_h};
        [e setComputePipelineState:p_mtp_pack_head];
        [e setBuffer:head_q4 offset:0 atIndex:0];
        [e setBuffer:head_sc offset:0 atIndex:1];
        [e setBuffer:m_vocab_ids offset:0 atIndex:2];
        [e setBuffer:m_head_q4 offset:0 atIndex:3];
        [e setBuffer:m_head_sc offset:0 atIndex:4];
        [e setBytes:&php length:sizeof(php) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake(DV, 1, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        struct MtpFoldPP { uint32_t dvocab, cols, rank; };
        MtpFoldPP fp{DV, c.hidden, mc.rank};
        [e setComputePipelineState:p_mtp_fold];
        [e setBuffer:file offset:unembed_mat.offset atIndex:0];
        [e setBuffer:mtp_file offset:m_hidden_out_w atIndex:1];
        [e setBuffer:m_vocab_ids offset:0 atIndex:2];
        [e setBuffer:m_fold_q4 offset:0 atIndex:3];
        [e setBuffer:m_fold_sc offset:0 atIndex:4];
        [e setBytes:&fp length:sizeof(fp) atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((DV + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        struct MtpBiasPP { uint32_t dvocab, cols; };
        MtpBiasPP fbp{DV, c.hidden};
        [e setComputePipelineState:p_mtp_fold_b];
        [e setBuffer:file offset:unembed_mat.offset atIndex:0];
        [e setBuffer:mtp_file offset:m_hidden_out_b atIndex:1];
        [e setBuffer:m_vocab_ids offset:0 atIndex:2];
        [e setBuffer:m_fold_b offset:0 atIndex:3];
        [e setBytes:&fbp length:sizeof(fbp) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake((DV + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        gpu_seconds(cb);
        mtp_ready = true;
        return true;
    }

    struct Bf12PackP { uint32_t src_rows, cols, row_base, out_cap; };
    struct Gemv12P { uint32_t rows, cols, round_out, has_outliers; uint32_t elut[16]; };

    bool build_bf12(std::string& error) {
        struct Job {
            Mat12* dst;
            std::vector<Mat> srcs;
            uint32_t total_rows = 0;
            uint32_t cols = 0;
            uint64_t out_base = 0;
            uint32_t out_cap = 0;
        };
        std::vector<Job> jobs;
        auto add_job = [&](Mat12* dst, std::initializer_list<Mat> mats) {
            Job j;
            j.dst = dst;
            for (const Mat& m : mats) {
                j.srcs.push_back(m);
                j.total_rows += m.rows;
                j.cols = m.cols;
            }
            jobs.push_back(std::move(j));
        };
        for (uint32_t i = 0; i < c.layers; ++i) {
            LayerWeights& w = lw[i];
            add_job(&w.q12, {w.wq, w.wr, w.gate_attn, w.sink});
            if (w.local) {
                add_job(&w.k12, {w.wk});
                add_job(&w.v12, {w.wv});
            } else {
                add_job(&w.k12, {w.wk, w.dynk});
                add_job(&w.v12, {w.wv, w.dynv});
            }
            add_job(&w.wo12, {w.wo});
            add_job(&w.router12, {w.router});
            if (w.has_ctl) {
                for (int k = 0; k < 4; ++k) {
                    add_job(&w.ctl[k].down12, {w.ctl[k].down});
                    add_job(&w.ctl[k].gf12, {w.ctl[k].gain, w.ctl[k].forget});
                }
            }
        }
        size_t nj = jobs.size();
        id<MTLBuffer> hbuf = [gpu.device newBufferWithLength:nj * 256 * 4 options:MTLResourceStorageModeShared];
        if (!hbuf) { error = "bf12 hist buffer allocation failed"; return false; }
        std::memset(hbuf.contents, 0, hbuf.length);
        {
            id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [e setComputePipelineState:p_bf12_hist];
            for (size_t ji = 0; ji < nj; ++ji) {
                for (const Mat& m : jobs[ji].srcs) {
                    uint32_t count8 = uint32_t((uint64_t(m.rows) * m.cols) / 8);
                    uint32_t groups = std::min<uint32_t>((count8 + 255) / 256, 1024u);
                    [e setBuffer:file offset:m.offset atIndex:0];
                    [e setBuffer:hbuf offset:ji * 256 * 4 atIndex:1];
                    [e setBytes:&count8 length:4 atIndex:2];
                    [e dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                }
            }
            [e endEncoding];
            gpu_seconds(cb);
        }
        id<MTLBuffer> mbuf = [gpu.device newBufferWithLength:nj * 256 options:MTLResourceStorageModeShared];
        if (!mbuf) { error = "bf12 map buffer allocation failed"; return false; }
        const uint32_t* hptr = static_cast<const uint32_t*>(hbuf.contents);
        uint8_t* mptr = static_cast<uint8_t*>(mbuf.contents);
        auto align256 = [](uint64_t v) { return (v + 255) / 256 * 256; };
        uint64_t sm_total = 0, ex_total = 0, ro_total = 0, out_total = 0;
        for (size_t ji = 0; ji < nj; ++ji) {
            Job& j = jobs[ji];
            const uint32_t* hc = hptr + ji * 256;
            std::array<uint32_t, 256> ord;
            for (uint32_t e = 0; e < 256; ++e) ord[e] = e;
            std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return hc[a] > hc[b]; });
            uint8_t* emap = mptr + ji * 256;
            std::memset(emap, 15, 256);
            j.dst->elut[15] = 0;
            uint64_t top_sum = 0, all_sum = 0;
            for (uint32_t e = 0; e < 256; ++e) all_sum += hc[e];
            for (uint32_t k = 0; k < 15; ++k) {
                uint32_t e = ord[k];
                if (hc[e] > 0) {
                    emap[e] = uint8_t(k);
                    j.dst->elut[k] = e << 23;
                    top_sum += hc[e];
                } else {
                    j.dst->elut[k] = 0;
                }
            }
            j.out_cap = uint32_t(all_sum - top_sum);
            j.out_base = out_total;
            out_total += j.out_cap;
            j.dst->rows = j.total_rows;
            j.dst->cols = j.cols;
            j.dst->sm_off = sm_total;
            j.dst->ex_off = ex_total;
            j.dst->ro_off = ro_total;
            j.dst->has_outliers = j.out_cap > 0 ? 1u : 0u;
            sm_total += align256(uint64_t(j.total_rows) * j.cols);
            ex_total += align256(uint64_t(j.total_rows) * (j.cols / 2));
            ro_total += align256(uint64_t(j.total_rows + 1) * 4);
        }
        bf12_sm = [gpu.device newBufferWithLength:sm_total options:MTLResourceStorageModePrivate];
        bf12_ex = [gpu.device newBufferWithLength:ex_total options:MTLResourceStorageModePrivate];
        bf12_ro = [gpu.device newBufferWithLength:ro_total options:MTLResourceStorageModeShared];
        id<MTLBuffer> cbuf = [gpu.device newBufferWithLength:nj * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> rbuf = [gpu.device newBufferWithLength:std::max<uint64_t>(out_total, 1) * 8 options:MTLResourceStorageModeShared];
        if (!bf12_sm || !bf12_ex || !bf12_ro || !cbuf || !rbuf) {
            error = "bf12 packed buffer allocation failed";
            return false;
        }
        std::memset(cbuf.contents, 0, cbuf.length);
        {
            id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
            [e setComputePipelineState:p_bf12_pack];
            for (size_t ji = 0; ji < nj; ++ji) {
                Job& j = jobs[ji];
                uint32_t rbase = 0;
                for (const Mat& m : j.srcs) {
                    Bf12PackP pp{m.rows, m.cols, rbase, j.out_cap};
                    [e setBuffer:file offset:m.offset atIndex:0];
                    [e setBuffer:mbuf offset:ji * 256 atIndex:1];
                    [e setBuffer:bf12_sm offset:j.dst->sm_off atIndex:2];
                    [e setBuffer:bf12_ex offset:j.dst->ex_off atIndex:3];
                    [e setBuffer:cbuf offset:ji * 4 atIndex:4];
                    [e setBuffer:rbuf offset:j.out_base * 8 atIndex:5];
                    [e setBytes:&pp length:sizeof(pp) atIndex:6];
                    [e dispatchThreadgroups:MTLSizeMake((m.rows + 7) / 8, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                    rbase += m.rows;
                }
            }
            [e endEncoding];
            gpu_seconds(cb);
        }
        uint64_t op_total = 0;
        for (size_t ji = 0; ji < nj; ++ji) {
            jobs[ji].dst->op_off = op_total;
            op_total += align256(uint64_t(std::max<uint32_t>(jobs[ji].out_cap, 1)) * 4);
        }
        bf12_op = [gpu.device newBufferWithLength:std::max<uint64_t>(op_total, 256) options:MTLResourceStorageModeShared];
        if (!bf12_op) { error = "bf12 outlier buffer allocation failed"; return false; }
        const uint32_t* cnts = static_cast<const uint32_t*>(cbuf.contents);
        const uint32_t* raws = static_cast<const uint32_t*>(rbuf.contents);
        for (size_t ji = 0; ji < nj; ++ji) {
            Job& j = jobs[ji];
            uint32_t* ro = reinterpret_cast<uint32_t*>(static_cast<char*>(bf12_ro.contents) + j.dst->ro_off);
            uint32_t* op = reinterpret_cast<uint32_t*>(static_cast<char*>(bf12_op.contents) + j.dst->op_off);
            uint32_t n = cnts[ji];
            std::memset(ro, 0, uint64_t(j.total_rows + 1) * 4);
            if (n == 0) continue;
            const uint32_t* src = raws + j.out_base * 2;
            for (uint32_t k = 0; k < n; ++k) ro[src[k * 2] + 1]++;
            for (uint32_t r = 0; r < j.total_rows; ++r) ro[r + 1] += ro[r];
            std::vector<uint32_t> cur(ro, ro + j.total_rows);
            for (uint32_t k = 0; k < n; ++k) {
                uint32_t row = src[k * 2];
                op[cur[row]++] = src[k * 2 + 1];
            }
            for (uint32_t r = 0; r < j.total_rows; ++r) {
                if (ro[r + 1] - ro[r] > 1) std::sort(op + ro[r], op + ro[r + 1]);
            }
        }
        bf12_ready = true;
        return true;
    }

    struct CertHeadP { uint32_t vocab, cols, ng; float cap; };
    struct CertTauP { uint32_t ntg, cols; float cap; };

    bool build_cert_head(std::string& error) {
        uint32_t V = c.vocab;
        uint32_t ng = c.hidden / 32;
        uint32_t ntg = (V + 7) / 8;
        head_q4 = [gpu.device newBufferWithLength:uint64_t(V) * ng * 16 options:MTLResourceStorageModePrivate];
        head_sc = [gpu.device newBufferWithLength:uint64_t(V) * ng * 2 options:MTLResourceStorageModePrivate];
        head_err = [gpu.device newBufferWithLength:uint64_t(V) * 4 options:MTLResourceStorageModePrivate];
        ch_ub = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * V * 4 options:MTLResourceStorageModePrivate];
        ch_tg_val = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * ntg * 4 options:MTLResourceStorageModePrivate];
        ch_tg_row = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * ntg * 4 options:MTLResourceStorageModePrivate];
        ch_state = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * 16 options:MTLResourceStorageModePrivate];
        ch_count = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * 4 options:MTLResourceStorageModeShared];
        ch_cands = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * V * 4 options:MTLResourceStorageModePrivate];
        ch_indirect = [gpu.device newBufferWithLength:uint64_t(MAX_BATCH) * 16 options:MTLResourceStorageModePrivate];
        if (!head_q4 || !head_sc || !head_err || !ch_ub || !ch_tg_val || !ch_tg_row || !ch_state || !ch_count || !ch_cands || !ch_indirect) {
            error = "certified head allocation failed";
            return false;
        }
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        CertHeadP p{V, c.hidden, ng, c.softcap};
        [e setComputePipelineState:p_ch_build];
        [e setBuffer:file offset:unembed_mat.offset atIndex:0];
        [e setBuffer:head_q4 offset:0 atIndex:1];
        [e setBuffer:head_sc offset:0 atIndex:2];
        [e setBuffer:head_err offset:0 atIndex:3];
        [e setBytes:&p length:sizeof(p) atIndex:4];
        [e dispatchThreadgroups:MTLSizeMake(ntg, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [e endEncoding];
        gpu_seconds(cb);
        cert_head_ready = true;
        return true;
    }

    void resolve_weights() {
        embed_off = T("embed.weight").offset;
        embed_norm_off = T("embed_norm.weight").offset;
        final_norm_off = T("norm.weight").offset;
        unembed_mat = make_mat("unembed.weight");
        lw.resize(c.layers);
        const char* cnames[4] = {"attn", "k", "v", "mlp"};
        for (uint32_t i = 0; i < c.layers; ++i) {
            LayerWeights& w = lw[i];
            std::string L = "layers." + std::to_string(i) + ".";
            std::string A = L + "attn.";
            std::string M = L + "mlp.";
            w.local = c.local[i];
            if (w.local) {
                w.attn_norm = T(L + "attn_norm.weight").offset;
            } else {
                w.q_norm = T(L + "q_input_norm.weight").offset;
                w.k_norm = T(L + "k_input_norm.weight").offset;
                w.v_norm = T(L + "v_input_norm.weight").offset;
            }
            w.mlp_norm = T(L + "mlp_norm.weight").offset;
            w.wq = make_mat(A + "wq_du.weight");
            w.wr = make_mat(A + "wr_du.weight");
            w.gate_attn = make_mat(A + "gate_attn.weight");
            w.sink = make_mat(A + "attn_sink_proj.weight");
            w.wk = make_mat(A + "wk_dv.weight");
            w.wv = make_mat(A + "wv_dv.weight");
            w.wo = make_mat(A + "wo_ud.weight");
            if (w.local) {
                w.k_sconv = T(A + "k_sconv.weight").offset;
                w.v_sconv = T(A + "v_sconv.weight").offset;
            } else {
                w.dynk = make_mat(A + "k_sconv.weight_proj");
                w.dynv = make_mat(A + "v_sconv.weight_proj");
                w.k_sconv = T(A + "k_sconv.static_w").offset;
                w.v_sconv = T(A + "v_sconv.static_w").offset;
            }
            w.q_clip = T(A + "q_l2_clip.scale").offset;
            w.r_clip = T(A + "rel_l2_clip.scale").offset;
            w.rel_proj = T(A + "rel_logits_proj.proj").offset;
            w.attn_sconv = T(L + "attn_sconv.weight").offset;
            w.mlp_sconv = T(L + "mlp_sconv.weight").offset;
            w.router = make_mat(M + "gate.weight");
            w.router_bias = T(M + "gate.bias").offset;
            const Tensor& w13 = T(M + "experts.w13_weight");
            const Tensor& s13 = T(M + "experts.w13_weight_scale");
            const Tensor& w2 = T(M + "experts.w2_weight");
            const Tensor& s2 = T(M + "experts.w2_weight_scale");
            w.nvfp4 = w13.dtype == "U8";
            w.w13_w = w13.offset; w.w13_s = s13.offset; w.w13_eb = w13.bytes / c.experts; w.w13_sb = s13.bytes / c.experts;
            w.w2_w = w2.offset; w.w2_s = s2.offset; w.w2_eb = w2.bytes / c.experts; w.w2_sb = s2.bytes / c.experts;
            if (w.nvfp4) {
                w.w13_amax = T(M + "experts.w13_weight_amax").offset;
                w.w13_denom = T(M + "experts.w13_weight_outer_scale_denominator").offset;
                w.w2_amax = T(M + "experts.w2_weight_amax").offset;
                w.w2_denom = T(M + "experts.w2_weight_outer_scale_denominator").offset;
            }
            if (i >= c.core_start && i < c.core_end) {
                w.has_ctl = true;
                for (int k = 0; k < 4; ++k) {
                    std::string cb = L + "controllers." + cnames[k] + ".";
                    w.ctl[k] = {make_mat(cb + "down_weight"), make_mat(cb + "gain_weight"), make_mat(cb + "forget_weight"), {}, {}, {}, {}};
                }
            }
        }
    }

    void build_queries() {
        std::vector<float> all;
        float scale = 1.0f / std::sqrt(float(c.hidden));
        auto append_rows = [&](const std::string& name, uint64_t row_offset, uint64_t rows) {
            const Tensor& t = tensors.at(name);
            const float* src = reinterpret_cast<const float*>(static_cast<const char*>(file.contents) + t.offset) + row_offset * c.hidden;
            for (uint64_t i = 0; i < rows * c.hidden; ++i) all.push_back(src[i] * scale);
        };
        auto align64 = [&]() { while (all.size() % 64) all.push_back(0); };
        std::string p = "model.llm.attn_res.";
        q_attn_off.assign(c.layers, 0);
        for (uint32_t i = 1; i < c.layers; ++i) {
            q_attn_off[i] = all.size() * 4;
            append_rows(p + "attn_queries." + std::to_string(i - 1), 0, c.local[i] ? 1 : 3);
            align64();
        }
        q_mlp_off.assign(c.layers, 0);
        for (uint32_t i = 0; i < c.layers; ++i) {
            q_mlp_off[i] = all.size() * 4;
            append_rows(p + "mlp_queries." + std::to_string(i), 0, 1);
            align64();
        }
        q_output_off = all.size() * 4;
        append_rows(p + "output_query", 0, 1);
        align64();
        uint32_t core = c.core_end - c.core_start;
        q_loop_attn_off.assign(core, 0);
        q_loop_mlp_off.assign(core, 0);
        for (uint32_t l = 0; l < core; ++l) {
            q_loop_attn_off[l] = all.size() * 4;
            append_rows("model.llm.middle_cycle_cross_pass_queries", 2 * l, 1);
            append_rows("model.llm.middle_cycle_cross_pass_kv_queries", 2 * l, 1);
            append_rows("model.llm.middle_cycle_cross_pass_kv_queries", 2 * l + 1, 1);
            align64();
            q_loop_mlp_off[l] = all.size() * 4;
            append_rows("model.llm.middle_cycle_cross_pass_queries", 2 * l + 1, 1);
            align64();
        }
        queries = [gpu.device newBufferWithBytes:all.data() length:all.size() * 4 options:MTLResourceStorageModeShared];
    }

    uint64_t so(uint32_t slot) const { return uint64_t(slot) * MAX_BATCH * c.hidden * 4; }
    uint32_t alloc() { return next_slot++; }

    void barrier() { if (!together) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
    void grid(NSUInteger groups, NSUInteger threads) { barrier(); [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }
    void grid2(NSUInteger gx, NSUInteger gy, NSUInteger threads) { barrier(); [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }
    void grid3(NSUInteger gx, NSUInteger gy, NSUInteger gz, NSUInteger threads) { barrier(); [enc dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }

    struct GemvP { uint32_t rows, cols, round_out; float scale; };
    struct GemvBP { uint32_t rows, cols, round_out, x_stride, y_stride; float scale; };
    struct Gemv12BP { uint32_t rows, cols, round_out, has_outliers, x_stride, y_stride; uint32_t elut[16]; };

    void gemv_ab(const Mat& m, id<MTLBuffer> in_buf, uint64_t in_off, id<MTLBuffer> out_buf, uint64_t out_off, bool round_out = true) {
        GemvP g{m.rows, m.cols, round_out ? 1u : 0u, 1.0f};
        [enc setComputePipelineState:p_gemv];
        [enc setBuffer:file offset:m.offset atIndex:0];
        [enc setBuffer:in_buf offset:in_off atIndex:1];
        [enc setBuffer:out_buf offset:out_off atIndex:2];
        [enc setBytes:&g length:sizeof(g) atIndex:3];
        grid((g.rows + 7) / 8, 256);
    }

    void gemv16_m(const Mat& m, id<MTLBuffer> in_buf, uint64_t in_off, uint32_t x_stride, id<MTLBuffer> out_buf, uint64_t out_off, uint32_t y_stride, uint32_t M, bool round_out = true) {
        if (M == 1) {
            gemv_ab(m, in_buf, in_off, out_buf, out_off, round_out);
            return;
        }
        for (uint32_t m0 = 0; m0 < M; m0 += 8) {
            uint32_t mb = std::min<uint32_t>(8, M - m0);
            GemvBP g{m.rows, m.cols, round_out ? 1u : 0u, x_stride, y_stride, 1.0f};
            [enc setComputePipelineState:p_gemv16_rows[mb]];
            [enc setBuffer:file offset:m.offset atIndex:0];
            [enc setBuffer:in_buf offset:in_off + uint64_t(m0) * x_stride * 4 atIndex:1];
            [enc setBuffer:out_buf offset:out_off + uint64_t(m0) * y_stride * 4 atIndex:2];
            [enc setBytes:&g length:sizeof(g) atIndex:3];
            grid((g.rows + 7) / 8, 256);
        }
    }

    void gemv12_ab(const Mat12& m, id<MTLBuffer> in_buf, uint64_t in_off, id<MTLBuffer> out_buf, uint64_t out_off, bool round_out = true) {
        Gemv12P g{};
        g.rows = m.rows;
        g.cols = m.cols;
        g.round_out = round_out ? 1u : 0u;
        g.has_outliers = m.has_outliers;
        std::memcpy(g.elut, m.elut, sizeof(g.elut));
        [enc setComputePipelineState:p_gemv12];
        [enc setBuffer:bf12_sm offset:m.sm_off atIndex:0];
        [enc setBuffer:bf12_ex offset:m.ex_off atIndex:1];
        [enc setBuffer:bf12_ro offset:m.ro_off atIndex:2];
        [enc setBuffer:bf12_op offset:m.op_off atIndex:3];
        [enc setBuffer:in_buf offset:in_off atIndex:4];
        [enc setBuffer:out_buf offset:out_off atIndex:5];
        [enc setBytes:&g length:sizeof(g) atIndex:6];
        grid((g.rows + 7) / 8, 256);
    }

    void gemv12_m(const Mat12& m, id<MTLBuffer> in_buf, uint64_t in_off, uint32_t x_stride, id<MTLBuffer> out_buf, uint64_t out_off, uint32_t y_stride, uint32_t M, bool round_out = true) {
        if (M == 1) {
            gemv12_ab(m, in_buf, in_off, out_buf, out_off, round_out);
            return;
        }
        for (uint32_t m0 = 0; m0 < M; m0 += 8) {
            uint32_t mb = std::min<uint32_t>(8, M - m0);
            Gemv12BP g{};
            g.rows = m.rows;
            g.cols = m.cols;
            g.round_out = round_out ? 1u : 0u;
            g.has_outliers = m.has_outliers;
            g.x_stride = x_stride;
            g.y_stride = y_stride;
            std::memcpy(g.elut, m.elut, sizeof(g.elut));
            [enc setComputePipelineState:p_gemv12_rows[mb]];
            [enc setBuffer:bf12_sm offset:m.sm_off atIndex:0];
            [enc setBuffer:bf12_ex offset:m.ex_off atIndex:1];
            [enc setBuffer:bf12_ro offset:m.ro_off atIndex:2];
            [enc setBuffer:bf12_op offset:m.op_off atIndex:3];
            [enc setBuffer:in_buf offset:in_off + uint64_t(m0) * x_stride * 4 atIndex:4];
            [enc setBuffer:out_buf offset:out_off + uint64_t(m0) * y_stride * 4 atIndex:5];
            [enc setBytes:&g length:sizeof(g) atIndex:6];
            grid((g.rows + 7) / 8, 256);
        }
    }

    struct GemvQ4BP { uint32_t rows, cols, ng, x_stride, y_stride, round_out; };

    void gemv_q4_m(const MatQ4& m, id<MTLBuffer> in_buf, uint64_t in_off, uint32_t x_stride, id<MTLBuffer> out_buf, uint64_t out_off, uint32_t y_stride, uint32_t M, bool round_out = true) {
        for (uint32_t m0 = 0; m0 < M; m0 += 8) {
            uint32_t mb = std::min<uint32_t>(8, M - m0);
            GemvQ4BP g{m.rows, m.cols, m.cols / 32, x_stride, y_stride, round_out ? 1u : 0u};
            [enc setComputePipelineState:p_gemv_q4_rows[mb]];
            [enc setBuffer:lean_q4 offset:m.q4_off atIndex:0];
            [enc setBuffer:lean_sc offset:m.sc_off atIndex:1];
            [enc setBuffer:in_buf offset:in_off + uint64_t(m0) * x_stride * 4 atIndex:2];
            [enc setBuffer:out_buf offset:out_off + uint64_t(m0) * y_stride * 4 atIndex:3];
            [enc setBytes:&g length:sizeof(g) atIndex:4];
            grid((g.rows + 7) / 8, 256);
        }
    }

    struct NormP { uint32_t n, has_weight; float eps; };
    uint32_t norm(uint32_t in, uint64_t weight_off, uint32_t M) {
        uint32_t out = alloc();
        NormP p{c.hidden, 1, 1e-6f};
        [enc setComputePipelineState:M == 1 ? p_norm : p_norm_b];
        [enc setBuffer:arena offset:so(in) atIndex:0];
        [enc setBuffer:file offset:weight_off atIndex:1];
        [enc setBuffer:arena offset:so(out) atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid(M, 576);
        return out;
    }

    struct ResP { uint32_t n, sources; int32_t addend; uint32_t queries, normalize, has_norm, offs[8], outs[3]; uint32_t pad; uint64_t norm_offs[3]; };
    std::vector<uint32_t> res(const std::vector<uint32_t>& sources, int addend, uint64_t query_off, uint32_t nq, uint32_t M, const uint64_t* norm_offs = nullptr) {
        ResP p{};
        p.n = c.hidden;
        p.sources = uint32_t(sources.size());
        p.addend = addend >= 0 ? int32_t(addend * MAX_BATCH * c.hidden) : -1;
        p.queries = nq;
        p.normalize = 1;
        p.has_norm = norm_offs ? 1u : 0u;
        for (size_t i = 0; i < sources.size(); ++i) p.offs[i] = sources[i] * MAX_BATCH * c.hidden;
        std::vector<uint32_t> outs;
        for (uint32_t k = 0; k < nq; ++k) {
            outs.push_back(alloc());
            p.outs[k] = outs.back() * MAX_BATCH * c.hidden;
            if (norm_offs) p.norm_offs[k] = norm_offs[k];
        }
        [enc setComputePipelineState:M == 1 ? p_res : p_res_b];
        [enc setBuffer:arena offset:0 atIndex:0];
        [enc setBuffer:queries offset:query_off atIndex:1];
        [enc setBuffer:file offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid(M, 576);
        return outs;
    }

    struct CtlMixBP { uint32_t n, gf_stride; };

    std::array<uint32_t, 3> controllers_attn_norm(const LayerWeights& w, const uint32_t dests[3], const uint32_t msgs[3], const uint64_t norm_offs[3], uint32_t M) {
        uint32_t n = c.hidden;
        uint32_t gfs = 2 * n;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        for (int k = 0; k < 3; ++k) {
            [enc setComputePipelineState:M == 1 ? p_ctl_norm : p_ctl_norm_b];
            [enc setBuffer:arena offset:so(dests[k]) atIndex:0];
            [enc setBuffer:arena offset:so(msgs[k]) atIndex:1];
            [enc setBuffer:scratch offset:s_normed[k] atIndex:2];
            [enc setBytes:&n length:4 atIndex:3];
            grid(M, 576);
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        for (int k = 0; k < 3; ++k) {
            if (lean_q4_ready) gemv_q4_m(w.ctl[k].down4, scratch, s_normed[k], 2 * n, scratch, s_ctlh[k], 64, M);
            else if (bf12_ready) gemv12_m(w.ctl[k].down12, scratch, s_normed[k], 2 * n, scratch, s_ctlh[k], 64, M);
            else gemv16_m(w.ctl[k].down, scratch, s_normed[k], 2 * n, scratch, s_ctlh[k], 64, M);
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        for (int k = 0; k < 3; ++k) {
            if (lean_q4_ready) {
                gemv_q4_m(w.ctl[k].gf4, scratch, s_ctlh[k], 64, scratch, s_gf[k], gfs, M, false);
            } else if (bf12_ready) {
                gemv12_m(w.ctl[k].gf12, scratch, s_ctlh[k], 64, scratch, s_gf[k], gfs, M, false);
            } else {
                gemv16_m(w.ctl[k].gain, scratch, s_ctlh[k], 64, scratch, s_gain[k], gfs, M, false);
                gemv16_m(w.ctl[k].forget, scratch, s_ctlh[k], 64, scratch, s_forget[k], gfs, M, false);
            }
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        std::array<uint32_t, 3> outs;
        CtlMixBP cmp{n, gfs};
        for (int k = 0; k < 3; ++k) {
            outs[k] = alloc();
            [enc setComputePipelineState:M == 1 ? p_ctl_mix_norm : p_ctl_mix_b];
            [enc setBuffer:arena offset:so(dests[k]) atIndex:0];
            [enc setBuffer:arena offset:so(msgs[k]) atIndex:1];
            [enc setBuffer:scratch offset:s_gain[k] atIndex:2];
            [enc setBuffer:scratch offset:s_forget[k] atIndex:3];
            [enc setBuffer:file offset:norm_offs[k] atIndex:4];
            [enc setBuffer:arena offset:so(outs[k]) atIndex:5];
            if (M == 1) [enc setBytes:&n length:4 atIndex:6];
            else [enc setBytes:&cmp length:sizeof(cmp) atIndex:6];
            grid(M, 576);
        }
        together = false;
        return outs;
    }

    uint32_t controller_mlp_norm(const LayerWeights& w, uint32_t dest, uint32_t msg, uint32_t M) {
        uint32_t n = c.hidden;
        uint32_t gfs = 2 * n;
        [enc setComputePipelineState:M == 1 ? p_ctl_norm : p_ctl_norm_b];
        [enc setBuffer:arena offset:so(dest) atIndex:0];
        [enc setBuffer:arena offset:so(msg) atIndex:1];
        [enc setBuffer:scratch offset:s_normed[0] atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        grid(M, 576);
        if (lean_q4_ready) {
            gemv_q4_m(w.ctl[3].down4, scratch, s_normed[0], 2 * n, scratch, s_ctlh[0], 64, M);
            gemv_q4_m(w.ctl[3].gf4, scratch, s_ctlh[0], 64, scratch, s_gf[0], gfs, M, false);
        } else if (bf12_ready) {
            gemv12_m(w.ctl[3].down12, scratch, s_normed[0], 2 * n, scratch, s_ctlh[0], 64, M);
            gemv12_m(w.ctl[3].gf12, scratch, s_ctlh[0], 64, scratch, s_gf[0], gfs, M, false);
        } else {
            gemv16_m(w.ctl[3].down, scratch, s_normed[0], 2 * n, scratch, s_ctlh[0], 64, M);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            together = true;
            gemv16_m(w.ctl[3].gain, scratch, s_ctlh[0], 64, scratch, s_gain[0], gfs, M, false);
            gemv16_m(w.ctl[3].forget, scratch, s_ctlh[0], 64, scratch, s_forget[0], gfs, M, false);
            together = false;
        }
        uint32_t out = alloc();
        CtlMixBP cmp{n, gfs};
        [enc setComputePipelineState:M == 1 ? p_ctl_mix_norm : p_ctl_mix_b];
        [enc setBuffer:arena offset:so(dest) atIndex:0];
        [enc setBuffer:arena offset:so(msg) atIndex:1];
        [enc setBuffer:scratch offset:s_gain[0] atIndex:2];
        [enc setBuffer:scratch offset:s_forget[0] atIndex:3];
        [enc setBuffer:file offset:w.mlp_norm atIndex:4];
        [enc setBuffer:arena offset:so(out) atIndex:5];
        if (M == 1) [enc setBytes:&n length:4 atIndex:6];
        else [enc setBytes:&cmp length:sizeof(cmp) atIndex:6];
        grid(M, 576);
        return out;
    }

    struct ConvBP { uint32_t channels, dynamic, head_dim, rows, x_stride, dyn_stride, y_stride; float scale; };
    void sconv_run(id<MTLBuffer> xb, uint64_t xoff, uint64_t state_off, uint64_t woff, uint64_t dynoff, id<MTLBuffer> yb, uint64_t yoff, uint32_t channels, bool dynamic, uint32_t M, uint32_t x_stride, uint32_t dyn_stride, uint32_t y_stride) {
        ConvBP p{channels, dynamic ? 1u : 0u, c.head_dim, M, x_stride, dyn_stride, y_stride, dynamic ? 1.0f / 3.0f : 0.4472135955f};
        [enc setComputePipelineState:p_sconv_b];
        [enc setBuffer:xb offset:xoff atIndex:0];
        [enc setBuffer:conv_buf offset:state_off atIndex:1];
        [enc setBuffer:file offset:woff atIndex:2];
        [enc setBuffer:scratch offset:dynoff atIndex:3];
        [enc setBuffer:yb offset:yoff atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        grid((channels + 255) / 256, 256);
    }

    void conv_commit_encode(uint32_t commit) {
        if (commit == 0) return;
        [enc setComputePipelineState:p_conv_commit];
        [enc setBuffer:conv_buf offset:0 atIndex:0];
        [enc setBuffer:conv_meta offset:0 atIndex:1];
        [enc setBytes:&commit length:4 atIndex:2];
        grid(conv_count, 256);
    }

    struct ClipBP { uint32_t width, has_scale, pos0, is_local, n_floor, in_stride, out_stride, mode; float threshold, alpha; };
    void clip_run(uint64_t in_off, uint64_t scale_off, uint64_t out_off, uint32_t rows, uint32_t width, float thr, uint32_t pos0, bool is_local, uint32_t in_stride, uint32_t out_stride, uint32_t mode, uint32_t M) {
        ClipBP p{width, scale_off ? 1u : 0u, pos0, is_local ? 1u : 0u, c.n_floor, in_stride, out_stride, mode, thr, c.alpha};
        [enc setComputePipelineState:p_clip_b];
        [enc setBuffer:scratch offset:in_off atIndex:0];
        if (scale_off) [enc setBuffer:file offset:scale_off atIndex:1]; else [enc setBuffer:scratch offset:in_off atIndex:1];
        [enc setBuffer:scratch offset:out_off atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid2(rows, M, 32);
    }

    struct ClipKvBP { uint32_t width, pos0, kv_stride, in_stride, mode, kv_cap; float threshold; };
    void clip_kv_run(uint64_t in_off, id<MTLBuffer> cache, uint32_t rows, uint32_t width, uint32_t pos0, uint32_t kv_stride, uint32_t in_stride, uint32_t mode, uint32_t cap, float thr, uint32_t M) {
        ClipKvBP p{width, pos0, kv_stride, in_stride, mode, cap, thr};
        [enc setComputePipelineState:p_clip_kv_b];
        [enc setBuffer:scratch offset:in_off atIndex:0];
        [enc setBuffer:cache offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        grid2(rows, M, 32);
    }

    struct AttnP { uint32_t pos, start, kv_heads, group, head_dim, extent, kv_cap; float scale; };
    struct AttnBP { uint32_t pos0, rows, window, is_local, kv_heads, group, head_dim, extent, heads, q_stride, gate_stride, gate_off, sink_off, out_stride, kv_cap; float scale; };
    struct RelBP { uint32_t d_rel, extent, heads; };

    uint32_t attention(const LayerWeights& w, uint32_t visit, uint32_t sq, uint32_t sk, uint32_t sv, uint32_t pos0, uint32_t M) {
        uint32_t H = c.hidden;
        uint32_t kvw = c.kv_heads * c.head_dim;
        uint32_t kstr = w.local ? k_stride_local : k_stride_global;
        uint32_t qdim = c.heads * c.head_dim;
        uint32_t rdim = c.heads * c.d_rel;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        if (lean_q4_ready) {
            gemv_q4_m(w.q4, arena, so(sq), H, scratch, s_qbundle, q_stride, M);
            gemv_q4_m(w.k4, arena, so(sk), H, scratch, s_kbundle, kstr, M);
            gemv_q4_m(w.v4, arena, so(sv), H, scratch, s_vbundle, kstr, M);
        } else if (bf12_ready) {
            gemv12_m(w.q12, arena, so(sq), H, scratch, s_qbundle, q_stride, M);
            gemv12_m(w.k12, arena, so(sk), H, scratch, s_kbundle, kstr, M);
            gemv12_m(w.v12, arena, so(sv), H, scratch, s_vbundle, kstr, M);
        } else {
            gemv16_m(w.wq, arena, so(sq), H, scratch, s_q, q_stride, M);
            gemv16_m(w.wr, arena, so(sq), H, scratch, s_r, q_stride, M);
            gemv16_m(w.gate_attn, arena, so(sq), H, scratch, s_gate, q_stride, M);
            gemv16_m(w.sink, arena, so(sq), H, scratch, s_sinks, q_stride, M);
            gemv16_m(w.wk, arena, so(sk), H, scratch, s_kraw, kstr, M);
            gemv16_m(w.wv, arena, so(sv), H, scratch, s_vraw, kstr, M);
            if (!w.local) {
                gemv16_m(w.dynk, arena, so(sk), H, scratch, s_dynk, kstr, M);
                gemv16_m(w.dynv, arena, so(sv), H, scratch, s_dynv, kstr, M);
            }
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        sconv_run(scratch, s_kraw, conv_off[visit][0], w.k_sconv, s_dynk, scratch, s_k, kvw, !w.local, M, kstr, kstr, kvw);
        sconv_run(scratch, s_vraw, conv_off[visit][1], w.v_sconv, s_dynv, scratch, s_v, kvw, !w.local, M, kstr, kstr, kvw);
        clip_run(s_q, w.q_clip, s_qc, c.heads, c.head_dim, c.clip_qk, pos0, w.local, q_stride, qdim, 0, M);
        clip_run(s_r, w.r_clip, s_rc, c.heads, c.d_rel, c.clip_rel, pos0, w.local, q_stride, rdim, 0, M);
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        clip_kv_run(s_k, kcache[visit], c.kv_heads, c.head_dim, pos0, kvw, kvw, 0, kv_cap[visit], c.clip_qk, M);
        clip_kv_run(s_v, vcache[visit], c.kv_heads, c.head_dim, pos0, kvw, kvw, 1, kv_cap[visit], 0.0f, M);
        RelBP rbp{c.d_rel, c.rel_extent, c.heads};
        [enc setComputePipelineState:p_rel_b];
        [enc setBuffer:scratch offset:s_rc atIndex:0];
        [enc setBuffer:file offset:w.rel_proj atIndex:1];
        [enc setBuffer:scratch offset:s_rel atIndex:2];
        [enc setBytes:&rbp length:sizeof(rbp) atIndex:3];
        grid3((c.rel_extent + 255) / 256, c.heads, M, 256);
        together = false;
        if (M == 1) {
            uint32_t start = w.local && pos0 + 1 > c.window ? pos0 + 1 - c.window : 0;
            AttnP a{pos0, start, c.kv_heads, c.heads / c.kv_heads, c.head_dim, c.rel_extent, kv_cap[visit], 1.0f / std::sqrt(float(c.head_dim))};
            [enc setComputePipelineState:p_attend];
            [enc setBuffer:scratch offset:s_qc atIndex:0];
            [enc setBuffer:kcache[visit] offset:0 atIndex:1];
            [enc setBuffer:vcache[visit] offset:0 atIndex:2];
            [enc setBuffer:scratch offset:s_rel atIndex:3];
            [enc setBuffer:scratch offset:s_gate atIndex:4];
            [enc setBuffer:scratch offset:s_sinks atIndex:5];
            [enc setBuffer:scratch offset:s_gated atIndex:6];
            [enc setBytes:&a length:sizeof(a) atIndex:7];
            grid(c.heads, 128);
        } else {
            AttnBP a{pos0, M, c.window, w.local ? 1u : 0u, c.kv_heads, c.heads / c.kv_heads, c.head_dim, c.rel_extent, c.heads, qdim, q_stride, qdim + rdim, qdim * 2 + rdim, qdim, kv_cap[visit], 1.0f / std::sqrt(float(c.head_dim))};
            [enc setComputePipelineState:p_attend_b];
            [enc setBuffer:scratch offset:s_qc atIndex:0];
            [enc setBuffer:kcache[visit] offset:0 atIndex:1];
            [enc setBuffer:vcache[visit] offset:0 atIndex:2];
            [enc setBuffer:scratch offset:s_rel atIndex:3];
            [enc setBuffer:scratch offset:s_qbundle atIndex:4];
            [enc setBuffer:scratch offset:s_gated atIndex:5];
            [enc setBytes:&a length:sizeof(a) atIndex:6];
            grid2(c.heads, (M + 3) / 4, 128);
        }
        uint32_t out = alloc();
        if (lean_q4_ready) gemv_q4_m(w.wo4, scratch, s_gated, qdim, arena, so(out), H, M);
        else if (bf12_ready) gemv12_m(w.wo12, scratch, s_gated, qdim, arena, so(out), H, M);
        else gemv16_m(w.wo, scratch, s_gated, qdim, arena, so(out), H, M);
        return out;
    }

    struct RouteP { uint32_t experts, top; float scale; };
    struct UnionPP { uint32_t rows, top, experts, w13_groups, w2_groups; };
    struct ExpertP { uint32_t rows, cols, in_stride, out_stride, per_row_input; uint32_t pad; uint64_t expert_bytes, scale_bytes; };
    struct ExpertGP { uint32_t rows, cols, in_stride, out_stride, per_row_input, top; uint64_t expert_bytes, scale_bytes; };
    struct SwigluP { uint32_t inter, top; float gate_cap, up_cap; uint32_t round_out; };

    void experts(const LayerWeights& w, uint32_t x, uint32_t M) {
        uint32_t H = c.hidden, I = c.inter;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        if (bf12_ready) gemv12_m(w.router12, arena, so(x), H, scratch, s_router, c.experts, M, false);
        else gemv16_m(w.router, arena, so(x), H, scratch, s_router, c.experts, M, false);
        if (w.nvfp4) {
            [enc setComputePipelineState:p_qnv4];
            [enc setBuffer:arena offset:so(x) atIndex:0];
            [enc setBuffer:scratch offset:s_xq atIndex:1];
            [enc setBuffer:scratch offset:s_outer atIndex:2];
            [enc setBytes:&H length:4 atIndex:3];
            grid(M, 160);
        } else {
            [enc setComputePipelineState:p_qmxf8];
            [enc setBuffer:arena offset:so(x) atIndex:0];
            [enc setBuffer:scratch offset:s_xq atIndex:1];
            [enc setBytes:&H length:4 atIndex:2];
            grid2((H / 32 + 63) / 64, M, 64);
        }
        together = false;
        RouteP rp{c.experts, eff_top, c.route_scale};
        [enc setComputePipelineState:M == 1 ? p_route : p_route_b];
        [enc setBuffer:scratch offset:s_router atIndex:0];
        [enc setBuffer:file offset:w.router_bias atIndex:1];
        [enc setBuffer:scratch offset:s_ids atIndex:2];
        [enc setBuffer:scratch offset:s_gammas atIndex:3];
        [enc setBytes:&rp length:sizeof(rp) atIndex:4];
        grid(M, 32);
        if (M > 1) {
            UnionPP up{M, eff_top, c.experts, (2 * I + 7) / 8, (H + 7) / 8};
            [enc setComputePipelineState:p_union];
            [enc setBuffer:scratch offset:s_ids atIndex:0];
            [enc setBuffer:u_exp offset:0 atIndex:1];
            [enc setBuffer:u_cnt offset:0 atIndex:2];
            [enc setBuffer:u_pairs offset:0 atIndex:3];
            [enc setBuffer:u_ind_w13 offset:0 atIndex:4];
            [enc setBuffer:u_ind_w2 offset:0 atIndex:5];
            [enc setBytes:&up length:sizeof(up) atIndex:6];
            grid(1, 128);
        }
        auto run_expert = [&](bool is_w2, uint32_t rows, uint32_t cols, uint64_t in_off, uint64_t outer_off, uint64_t out_off, bool per_row) {
            if (M == 1) {
                ExpertP p{rows, cols, per_row ? I : 0u, per_row ? H : 2 * I, per_row ? 1u : 0u, 0, is_w2 ? w.w2_eb : w.w13_eb, is_w2 ? w.w2_sb : w.w13_sb};
                [enc setComputePipelineState:w.nvfp4 ? p_env4 : p_emxf8];
                [enc setBuffer:file offset:is_w2 ? w.w2_w : w.w13_w atIndex:0];
                [enc setBuffer:file offset:is_w2 ? w.w2_s : w.w13_s atIndex:1];
                if (w.nvfp4) {
                    [enc setBuffer:file offset:is_w2 ? w.w2_amax : w.w13_amax atIndex:2];
                    [enc setBuffer:file offset:is_w2 ? w.w2_denom : w.w13_denom atIndex:3];
                    [enc setBuffer:scratch offset:outer_off atIndex:5];
                }
                [enc setBuffer:scratch offset:in_off atIndex:4];
                [enc setBuffer:scratch offset:s_ids atIndex:6];
                [enc setBuffer:scratch offset:out_off atIndex:7];
                [enc setBytes:&p length:sizeof(p) atIndex:8];
                grid2((rows + 7) / 8, eff_top, 256);
            } else {
                ExpertGP p{rows, cols, per_row ? I : H, per_row ? H : 2 * I, per_row ? 1u : 0u, eff_top, is_w2 ? w.w2_eb : w.w13_eb, is_w2 ? w.w2_sb : w.w13_sb};
                [enc setComputePipelineState:w.nvfp4 ? p_env4_g : p_emxf8_g];
                [enc setBuffer:file offset:is_w2 ? w.w2_w : w.w13_w atIndex:0];
                [enc setBuffer:file offset:is_w2 ? w.w2_s : w.w13_s atIndex:1];
                if (w.nvfp4) {
                    [enc setBuffer:file offset:is_w2 ? w.w2_amax : w.w13_amax atIndex:2];
                    [enc setBuffer:file offset:is_w2 ? w.w2_denom : w.w13_denom atIndex:3];
                    [enc setBuffer:scratch offset:outer_off atIndex:5];
                }
                [enc setBuffer:scratch offset:in_off atIndex:4];
                [enc setBuffer:u_exp offset:0 atIndex:6];
                [enc setBuffer:u_cnt offset:0 atIndex:7];
                [enc setBuffer:u_pairs offset:0 atIndex:8];
                [enc setBuffer:scratch offset:out_off atIndex:9];
                [enc setBytes:&p length:sizeof(p) atIndex:10];
                barrier();
                [enc dispatchThreadgroupsWithIndirectBuffer:(is_w2 ? u_ind_w2 : u_ind_w13) indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        };
        run_expert(false, 2 * I, H, s_xq, s_outer, s_gu, false);
        uint32_t total_k = M * eff_top;
        if (w.nvfp4) {
            SwigluP sp{I, total_k, c.gate_cap, c.up_cap, 1u};
            [enc setComputePipelineState:p_swiglu_qnv4];
            [enc setBuffer:scratch offset:s_gu atIndex:0];
            [enc setBuffer:scratch offset:s_gammas atIndex:1];
            [enc setBuffer:scratch offset:s_hq atIndex:2];
            [enc setBuffer:scratch offset:s_houter atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            grid(total_k, 128);
        } else {
            uint32_t n = total_k * 2 * I;
            [enc setComputePipelineState:p_round];
            [enc setBuffer:scratch offset:s_gu atIndex:0];
            [enc setBytes:&n length:4 atIndex:1];
            grid((n + 255) / 256, 256);
            SwigluP sp{I, total_k, c.gate_cap, c.up_cap, 0u};
            [enc setComputePipelineState:p_swiglu];
            [enc setBuffer:scratch offset:s_gu atIndex:0];
            [enc setBuffer:scratch offset:s_gammas atIndex:1];
            [enc setBuffer:scratch offset:s_h atIndex:2];
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            grid2((I + 255) / 256, total_k, 256);
            [enc setComputePipelineState:p_qmxf8];
            [enc setBuffer:scratch offset:s_h atIndex:0];
            [enc setBuffer:scratch offset:s_hq atIndex:1];
            [enc setBytes:&I length:4 atIndex:2];
            grid2((I / 32 + 63) / 64, total_k, 64);
        }
        run_expert(true, H, I, s_hq, s_houter, s_rows, true);
    }

    struct CombineBP { uint32_t n, top, boundary, rows; };
    struct TapCopyP { uint32_t n4, tap_idx; };

    void copy_tap(uint32_t slot, uint32_t tap_idx, uint32_t M) {
        TapCopyP tp{c.hidden / 4, tap_idx};
        [enc setComputePipelineState:p_copy_tap];
        [enc setBuffer:arena offset:so(slot) atIndex:0];
        [enc setBuffer:tap_buf offset:0 atIndex:1];
        [enc setBytes:&tp length:sizeof(tp) atIndex:2];
        grid2((tp.n4 + 255) / 256, M, 256);
    }

    std::pair<std::vector<uint32_t>, uint32_t> layer(std::vector<uint32_t> blocks, uint32_t partial, uint32_t layer_idx, uint32_t pass, const std::vector<uint32_t>& bank, uint32_t pos0, uint32_t M, bool capture_taps) {
        if (capture_taps) {
            if (pass == 0 && layer_idx == 5) copy_tap(partial, 0, M);
            else if (pass == 1 && layer_idx == 11) copy_tap(partial, 1, M);
            else if (pass == 2 && layer_idx == 17) copy_tap(partial, 2, M);
        }
        const LayerWeights& w = lw[layer_idx];
        uint32_t visit = pass == 0 ? layer_idx : c.layers + (pass - 1) * (c.core_end - c.core_start) + (layer_idx - c.core_start);
        uint32_t sq, sk, sv;
        uint32_t core_layer = layer_idx - c.core_start;
        if (!bank.empty()) {
            std::vector<uint32_t> src = blocks;
            src.push_back(partial);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            together = true;
            auto streams = res(src, -1, q_attn_off[layer_idx], w.local ? 1 : 3, M);
            auto messages = res(bank, -1, q_loop_attn_off[core_layer], 3, M);
            together = false;
            uint32_t dests[3] = {streams[0], w.local ? streams[0] : streams[1], w.local ? streams[0] : streams[2]};
            uint32_t msgs[3] = {messages[0], messages[1], messages[2]};
            uint64_t noffs[3] = {w.local ? w.attn_norm : w.q_norm, w.local ? w.attn_norm : w.k_norm, w.local ? w.attn_norm : w.v_norm};
            auto norms = controllers_attn_norm(w, dests, msgs, noffs, M);
            sq = norms[0]; sk = norms[1]; sv = norms[2];
        } else if (blocks.empty()) {
            if (w.local) {
                sq = sk = sv = norm(partial, w.attn_norm, M);
            } else {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                together = true;
                sq = norm(partial, w.q_norm, M);
                sk = norm(partial, w.k_norm, M);
                sv = norm(partial, w.v_norm, M);
                together = false;
            }
        } else {
            std::vector<uint32_t> src = blocks;
            src.push_back(partial);
            if (w.local) {
                uint64_t noffs[3] = {w.attn_norm, 0, 0};
                sq = sk = sv = res(src, -1, q_attn_off[layer_idx], 1, M, noffs)[0];
            } else {
                uint64_t noffs[3] = {w.q_norm, w.k_norm, w.v_norm};
                auto norms = res(src, -1, q_attn_off[layer_idx], 3, M, noffs);
                sq = norms[0]; sk = norms[1]; sv = norms[2];
            }
        }
        uint32_t att = attention(w, visit, sq, sk, sv, pos0, M);
        uint32_t attc = alloc();
        sconv_run(arena, so(att), conv_off[visit][2], w.attn_sconv, s_dynk, arena, so(attc), c.hidden, false, M, c.hidden, 0, c.hidden);
        bool boundary = layer_idx % c.block == 0;
        std::vector<uint32_t> src = blocks;
        src.push_back(partial);
        int mlp_addend = boundary ? -1 : int(attc);
        if (boundary) src.push_back(attc);
        uint32_t x;
        if (!bank.empty()) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            together = true;
            uint32_t mlp_in = res(src, mlp_addend, q_mlp_off[layer_idx], 1, M)[0];
            uint32_t msg = res(bank, -1, q_loop_mlp_off[core_layer], 1, M)[0];
            together = false;
            x = controller_mlp_norm(w, mlp_in, msg, M);
        } else {
            uint64_t noffs[3] = {w.mlp_norm, 0, 0};
            x = res(src, mlp_addend, q_mlp_off[layer_idx], 1, M, noffs)[0];
        }
        experts(w, x, M);
        uint32_t out = alloc();
        CombineBP cp{c.hidden, eff_top, boundary ? 1u : 0u, M};
        [enc setComputePipelineState:p_combine_b];
        [enc setBuffer:scratch offset:s_rows atIndex:0];
        [enc setBuffer:scratch offset:s_ids atIndex:1];
        [enc setBuffer:arena offset:so(attc) atIndex:2];
        [enc setBuffer:arena offset:so(partial) atIndex:3];
        [enc setBuffer:conv_buf offset:conv_off[visit][3] atIndex:4];
        [enc setBuffer:file offset:w.mlp_sconv atIndex:5];
        [enc setBuffer:arena offset:so(out) atIndex:6];
        [enc setBytes:&cp length:sizeof(cp) atIndex:7];
        grid((c.hidden + 255) / 256, 256);
        if (boundary) blocks.push_back(partial);
        return {blocks, out};
    }

    struct EmbedBP { uint32_t n, rows; };
    struct CertHeadBP { uint32_t vocab, cols, ng, ntg, m_base; float cap; };

    void run_head_single(uint64_t hidden_off, uint64_t tok_out_off, bool full_logits) {
        uint32_t H = c.hidden;
        if (cert_head_active && !full_logits) {
            uint32_t V = c.vocab;
            uint32_t ng = c.hidden / 32;
            uint32_t ntg = (V + 7) / 8;
            CertHeadP chp{V, c.hidden, ng, c.softcap};
            CertTauP ctp{ntg, c.hidden, c.softcap};
            [enc setComputePipelineState:p_ch_xnorm];
            [enc setBuffer:arena offset:hidden_off atIndex:0];
            [enc setBuffer:ch_state offset:0 atIndex:1];
            [enc setBuffer:ch_count offset:0 atIndex:2];
            [enc setBytes:&H length:4 atIndex:3];
            grid(1, 576);
            [enc setComputePipelineState:p_ch_scan];
            [enc setBuffer:head_q4 offset:0 atIndex:0];
            [enc setBuffer:head_sc offset:0 atIndex:1];
            [enc setBuffer:head_err offset:0 atIndex:2];
            [enc setBuffer:arena offset:hidden_off atIndex:3];
            [enc setBuffer:ch_state offset:0 atIndex:4];
            [enc setBuffer:ch_ub offset:0 atIndex:5];
            [enc setBuffer:ch_tg_val offset:0 atIndex:6];
            [enc setBuffer:ch_tg_row offset:0 atIndex:7];
            [enc setBytes:&chp length:sizeof(chp) atIndex:8];
            grid(ntg, 256);
            [enc setComputePipelineState:p_ch_tau];
            [enc setBuffer:ch_tg_val offset:0 atIndex:0];
            [enc setBuffer:ch_tg_row offset:0 atIndex:1];
            [enc setBuffer:file offset:unembed_mat.offset atIndex:2];
            [enc setBuffer:arena offset:hidden_off atIndex:3];
            [enc setBuffer:ch_state offset:0 atIndex:4];
            [enc setBytes:&ctp length:sizeof(ctp) atIndex:5];
            grid(1, 512);
            [enc setComputePipelineState:p_ch_filter];
            [enc setBuffer:ch_ub offset:0 atIndex:0];
            [enc setBuffer:ch_state offset:0 atIndex:1];
            [enc setBuffer:ch_count offset:0 atIndex:2];
            [enc setBuffer:ch_cands offset:0 atIndex:3];
            [enc setBytes:&V length:4 atIndex:4];
            grid((V + 255) / 256, 256);
            [enc setComputePipelineState:p_ch_setup];
            [enc setBuffer:ch_count offset:0 atIndex:0];
            [enc setBuffer:ch_indirect offset:0 atIndex:1];
            grid(1, 1);
            [enc setComputePipelineState:p_ch_rescore];
            [enc setBuffer:file offset:unembed_mat.offset atIndex:0];
            [enc setBuffer:arena offset:hidden_off atIndex:1];
            [enc setBuffer:ch_cands offset:0 atIndex:2];
            [enc setBuffer:ch_count offset:0 atIndex:3];
            [enc setBuffer:ch_tg_val offset:0 atIndex:4];
            [enc setBuffer:ch_tg_row offset:0 atIndex:5];
            [enc setBytes:&ctp length:sizeof(ctp) atIndex:6];
            barrier();
            [enc dispatchThreadgroupsWithIndirectBuffer:ch_indirect indirectBufferOffset:0 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [enc setComputePipelineState:p_ch_argmax];
            [enc setBuffer:ch_tg_val offset:0 atIndex:0];
            [enc setBuffer:ch_tg_row offset:0 atIndex:1];
            [enc setBuffer:ch_indirect offset:0 atIndex:2];
            [enc setBuffer:token offset:tok_out_off atIndex:3];
            grid(1, 256);
        } else {
            gemv_ab(unembed_mat, arena, hidden_off, logits, 0, true);
            struct { uint32_t n, valid; float cap; } lp{c.padded_vocab, c.vocab, c.softcap};
            [enc setComputePipelineState:p_softcap];
            [enc setBuffer:logits offset:0 atIndex:0];
            [enc setBytes:&lp length:sizeof(lp) atIndex:1];
            grid((c.padded_vocab + 255) / 256, 256);
            [enc setComputePipelineState:p_argmax];
            [enc setBuffer:logits offset:0 atIndex:0];
            [enc setBuffer:token offset:tok_out_off atIndex:1];
            [enc setBytes:&c.padded_vocab length:4 atIndex:2];
            grid(1, 1024);
        }
    }

    void run_head_batch(uint32_t hidden, uint32_t M, LogitsMode lmode, bool full_logits) {
        if (lmode == LOGITS_NONE) return;
        uint32_t H = c.hidden;
        if (lmode == LOGITS_LAST) {
            run_head_single(so(hidden) + uint64_t(M - 1) * H * 4, 16 * 4, full_logits);
            return;
        }
        if (M == 1 || full_logits || !cert_head_active) {
            for (uint32_t m = 0; m < M; ++m) {
                run_head_single(so(hidden) + uint64_t(m) * H * 4, (16 + m) * 4, full_logits);
            }
            return;
        }
        uint32_t V = c.vocab;
        uint32_t ng = c.hidden / 32;
        uint32_t ntg = (V + 7) / 8;
        CertTauP ctp{ntg, c.hidden, c.softcap};
        [enc setComputePipelineState:p_ch_xnorm_b];
        [enc setBuffer:arena offset:so(hidden) atIndex:0];
        [enc setBuffer:ch_state offset:0 atIndex:1];
        [enc setBuffer:ch_count offset:0 atIndex:2];
        [enc setBytes:&H length:4 atIndex:3];
        grid(M, 576);
        for (uint32_t m0 = 0; m0 < M; m0 += 8) {
            uint32_t mb = std::min<uint32_t>(8, M - m0);
            CertHeadBP chb{V, c.hidden, ng, ntg, m0, c.softcap};
            [enc setComputePipelineState:p_ch_scan_rows[mb]];
            [enc setBuffer:head_q4 offset:0 atIndex:0];
            [enc setBuffer:head_sc offset:0 atIndex:1];
            [enc setBuffer:head_err offset:0 atIndex:2];
            [enc setBuffer:arena offset:so(hidden) atIndex:3];
            [enc setBuffer:ch_state offset:0 atIndex:4];
            [enc setBuffer:ch_ub offset:0 atIndex:5];
            [enc setBuffer:ch_tg_val offset:0 atIndex:6];
            [enc setBuffer:ch_tg_row offset:0 atIndex:7];
            [enc setBytes:&chb length:sizeof(chb) atIndex:8];
            grid(ntg, 256);
        }
        [enc setComputePipelineState:p_ch_tau_b];
        [enc setBuffer:ch_tg_val offset:0 atIndex:0];
        [enc setBuffer:ch_tg_row offset:0 atIndex:1];
        [enc setBuffer:file offset:unembed_mat.offset atIndex:2];
        [enc setBuffer:arena offset:so(hidden) atIndex:3];
        [enc setBuffer:ch_state offset:0 atIndex:4];
        [enc setBytes:&ctp length:sizeof(ctp) atIndex:5];
        grid(M, 512);
        [enc setComputePipelineState:p_ch_filter_b];
        [enc setBuffer:ch_ub offset:0 atIndex:0];
        [enc setBuffer:ch_state offset:0 atIndex:1];
        [enc setBuffer:ch_count offset:0 atIndex:2];
        [enc setBuffer:ch_cands offset:0 atIndex:3];
        [enc setBytes:&V length:4 atIndex:4];
        grid2((V + 255) / 256, M, 256);
        [enc setComputePipelineState:p_ch_setup_b];
        [enc setBuffer:ch_count offset:0 atIndex:0];
        [enc setBuffer:ch_indirect offset:0 atIndex:1];
        grid(M, 1);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        [enc setComputePipelineState:p_ch_rescore];
        for (uint32_t m = 0; m < M; ++m) {
            [enc setBuffer:file offset:unembed_mat.offset atIndex:0];
            [enc setBuffer:arena offset:so(hidden) + uint64_t(m) * H * 4 atIndex:1];
            [enc setBuffer:ch_cands offset:uint64_t(m) * V * 4 atIndex:2];
            [enc setBuffer:ch_count offset:uint64_t(m) * 4 atIndex:3];
            [enc setBuffer:ch_tg_val offset:uint64_t(m) * ntg * 4 atIndex:4];
            [enc setBuffer:ch_tg_row offset:uint64_t(m) * ntg * 4 atIndex:5];
            [enc setBytes:&ctp length:sizeof(ctp) atIndex:6];
            [enc dispatchThreadgroupsWithIndirectBuffer:ch_indirect indirectBufferOffset:uint64_t(m) * 16 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        [enc setComputePipelineState:p_ch_argmax];
        for (uint32_t m = 0; m < M; ++m) {
            [enc setBuffer:ch_tg_val offset:uint64_t(m) * ntg * 4 atIndex:0];
            [enc setBuffer:ch_tg_row offset:uint64_t(m) * ntg * 4 atIndex:1];
            [enc setBuffer:ch_indirect offset:uint64_t(m) * 16 atIndex:2];
            [enc setBuffer:token offset:uint64_t(16 + m) * 4 atIndex:3];
            grid(1, 256);
        }
        together = false;
    }

    double step_rows(const uint32_t* toks, uint32_t pos0, uint32_t M, LogitsMode lmode, bool auto_commit = true, bool full_logits = false, bool capture_taps = false, uint32_t ingest_pos = 0, uint32_t ingest_commit = 0) {
        next_slot = 0;
        std::memcpy(token.contents, toks, M * 4);
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        if (pending_conv_commit > 0) {
            conv_commit_encode(pending_conv_commit);
            pending_conv_commit = 0;
        }
        if (ingest_commit > 0 && mtp_ready) {
            mtp_ingest_encode(ingest_pos, ingest_commit, false);
        }
        uint32_t emb = alloc();
        uint32_t H = c.hidden;
        if (M == 1) {
            [enc setComputePipelineState:p_embed];
            [enc setBuffer:file offset:embed_off atIndex:0];
            [enc setBuffer:token offset:0 atIndex:1];
            [enc setBuffer:arena offset:so(emb) atIndex:2];
            [enc setBytes:&H length:4 atIndex:3];
            grid((H + 255) / 256, 256);
        } else {
            EmbedBP ep{H, M};
            [enc setComputePipelineState:p_embed_b];
            [enc setBuffer:file offset:embed_off atIndex:0];
            [enc setBuffer:token offset:0 atIndex:1];
            [enc setBuffer:arena offset:so(emb) atIndex:2];
            [enc setBytes:&ep length:sizeof(ep) atIndex:3];
            grid2((H + 255) / 256, M, 256);
        }
        uint32_t x = norm(emb, embed_norm_off, M);
        std::vector<uint32_t> blocks;
        uint32_t partial = x;
        for (uint32_t l = 0; l < c.core_start; ++l) std::tie(blocks, partial) = layer(blocks, partial, l, 0, {}, pos0, M, capture_taps);
        auto prelude_blocks = blocks;
        uint32_t prelude_partial = partial;
        std::vector<uint32_t> bank;
        for (uint32_t pass = 0; pass < c.passes; ++pass) {
            if (pass) {
                bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
                bank.push_back(partial);
                blocks = prelude_blocks;
                partial = prelude_partial;
            }
            for (uint32_t l = c.core_start; l < c.core_end; ++l) std::tie(blocks, partial) = layer(blocks, partial, l, pass, bank, pos0, M, capture_taps);
        }
        for (uint32_t l = c.core_end; l < c.layers; ++l) std::tie(blocks, partial) = layer(blocks, partial, l, 0, {}, pos0, M, capture_taps);
        if (lmode != LOGITS_NONE || capture_taps) {
            std::vector<uint32_t> src = blocks;
            src.push_back(partial);
            uint64_t final_noffs[3] = {final_norm_off, 0, 0};
            uint32_t hidden = res(src, -1, q_output_off, 1, M, final_noffs)[0];
            if (capture_taps) copy_tap(hidden, 3, M);
            run_head_batch(hidden, M, lmode, full_logits);
        }
        if (auto_commit) conv_commit_encode(M);
        [enc endEncoding];
        double seconds = gpu_seconds(cb);
        if (next_slot > slots) { std::fprintf(stderr, "delphi: arena overflow %u\n", next_slot); std::exit(3); }
        return seconds;
    }

    double step(uint32_t tok, uint32_t pos, bool need_logits = true, bool full_logits = false) {
        return step_rows(&tok, pos, 1, need_logits ? LOGITS_ALL : LOGITS_NONE, true, full_logits, false);
    }

    void reset_state() {
        pending_conv_commit = 0;
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
        [blit fillBuffer:conv_buf range:NSMakeRange(0, conv_buf.length) value:0];
        [blit endEncoding];
        gpu_seconds(cb);
    }

    struct MtpMixPP { uint32_t hidden, alpha_row, row_start; float eps; };
    struct MtpGemvQ4PP { uint32_t rows, cols, ng, x_stride, y_stride, round_out, has_bias; };
    struct MtpCtxKvPP { uint32_t kv_heads, head_dim, pos0, kv_cap; float eps, log_theta; };
    struct MtpEmbedPP { uint32_t hidden; float eps; };
    struct MtpQkRopePP { uint32_t q_heads, kv_heads, head_dim, qkv_stride, pos0; float eps, log_theta; };
    struct MtpAttnPP { uint32_t prefix_len, ctx_start, q_heads, kv_heads, group, qkv_stride, q_dim, kv_dim, kv_cap; float scale; };
    struct MtpBasePP { uint32_t vocab, ng, xs4; };
    struct MtpRnnInPP { uint32_t rank, hidden, step; };
    struct MtpMarkovPP { uint32_t dvocab, step; float cap, inv_temp; };
    struct MtpReducePP { uint32_t ntg, step; float inv_temp; };

    void mtp_gemv_q4(const MtpQ4Mat& m, id<MTLBuffer> xbuf, uint32_t x_stride, id<MTLBuffer> ybuf, uint32_t y_stride, uint32_t M, bool round_out, uint64_t bias_off = 0, bool has_bias = false) {
        MtpGemvQ4PP p{m.rows, m.cols, m.cols / 32, x_stride, y_stride, round_out ? 1u : 0u, has_bias ? 1u : 0u};
        [enc setComputePipelineState:p_mtp_q4_rows[M]];
        [enc setBuffer:m_q4 offset:m.q4_off atIndex:0];
        [enc setBuffer:m_sc offset:m.sc_off atIndex:1];
        [enc setBuffer:xbuf offset:0 atIndex:2];
        [enc setBuffer:mtp_file offset:has_bias ? bias_off : 0 atIndex:3];
        [enc setBuffer:ybuf offset:0 atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        grid((m.rows + 7) / 8, 256);
    }

    void mtp_ingest_encode(uint32_t pos0, uint32_t commit, bool write_f_rnn) {
        uint32_t H = mc.hidden;
        uint32_t kvw = mc.kv_heads * mc.head_dim;
        for (uint32_t off = 0; off < commit; off += 8) {
            uint32_t mb = std::min<uint32_t>(8, commit - off);
            for (uint32_t i = 0; i < mc.layers; ++i) {
                const MtpLayer& ml = mlw[i];
                MtpMixPP mp{H, i, off, mc.eps};
                [enc setComputePipelineState:p_mtp_mix];
                [enc setBuffer:tap_buf offset:0 atIndex:0];
                [enc setBuffer:m_alpha offset:0 atIndex:1];
                [enc setBuffer:mtp_file offset:ml.dflare_norm atIndex:2];
                [enc setBuffer:m_ctx_x offset:0 atIndex:3];
                [enc setBytes:&mp length:sizeof(mp) atIndex:4];
                grid(mb, 576);
                mtp_gemv_q4(ml.kv_ctx, m_ctx_x, H, m_ctx_kv, 2 * kvw, mb, false);
                MtpCtxKvPP kp{mc.kv_heads, mc.head_dim, pos0 + off, m_kv_cap[i], mc.eps, mc.log_theta};
                [enc setComputePipelineState:p_mtp_ctx_kv];
                [enc setBuffer:m_ctx_kv offset:0 atIndex:0];
                [enc setBuffer:mtp_file offset:ml.k_norm_ctx atIndex:1];
                [enc setBuffer:m_kcache[i] offset:0 atIndex:2];
                [enc setBuffer:m_vcache[i] offset:0 atIndex:3];
                [enc setBytes:&kp length:sizeof(kp) atIndex:4];
                grid2(mc.kv_heads, mb, 32);
            }
        }
        if (write_f_rnn && commit > 0) {
            MtpMixPP mp{H, mc.layers, commit - 1, mc.eps};
            [enc setComputePipelineState:p_mtp_mix];
            [enc setBuffer:tap_buf offset:0 atIndex:0];
            [enc setBuffer:m_alpha offset:0 atIndex:1];
            [enc setBuffer:mtp_file offset:m_markov_dflare atIndex:2];
            [enc setBuffer:m_f_rnn offset:0 atIndex:3];
            [enc setBytes:&mp length:sizeof(mp) atIndex:4];
            grid(1, 576);
        }
    }

    double mtp_ingest_ctx(uint32_t pos0, uint32_t commit, bool write_f_rnn) {
        if (!mtp_ready || commit == 0) return 0.0;
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        mtp_ingest_encode(pos0, commit, write_f_rnn);
        [enc endEncoding];
        return gpu_seconds(cb);
    }

    double mtp_propose(uint32_t anchor, uint32_t prefix_len, uint32_t pending_pos0, uint32_t pending_commit) {
        static_cast<uint32_t*>(m_out.contents)[0] = anchor;
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        if (pending_commit > 0) mtp_ingest_encode(pending_pos0, pending_commit, true);
        uint32_t H = mc.hidden;
        uint32_t kvw = mc.kv_heads * mc.head_dim;
        uint32_t qdim = mc.heads * mc.head_dim;
        uint32_t qkv_stride = qdim + 2 * kvw;
        MtpEmbedPP ep{H, mc.eps};
        [enc setComputePipelineState:p_mtp_embed];
        [enc setBuffer:file offset:embed_off atIndex:0];
        [enc setBuffer:file offset:embed_norm_off atIndex:1];
        [enc setBuffer:mtp_file offset:m_mask_embed atIndex:2];
        [enc setBuffer:m_out offset:0 atIndex:3];
        [enc setBuffer:m_h offset:0 atIndex:4];
        [enc setBytes:&ep length:sizeof(ep) atIndex:5];
        grid(8, 576);
        NormP np{H, 1, mc.eps};
        for (uint32_t i = 0; i < mc.layers; ++i) {
            const MtpLayer& ml = mlw[i];
            [enc setComputePipelineState:p_mtp_norm_h];
            [enc setBuffer:m_h offset:0 atIndex:0];
            [enc setBuffer:mtp_file offset:ml.in_norm atIndex:1];
            [enc setBuffer:m_hn offset:0 atIndex:2];
            [enc setBytes:&np length:sizeof(np) atIndex:3];
            grid(8, 576);
            mtp_gemv_q4(ml.qkv, m_hn, H, m_qkv, qkv_stride, 8, false);
            MtpQkRopePP rp{mc.heads, mc.kv_heads, mc.head_dim, qkv_stride, prefix_len, mc.eps, mc.log_theta};
            [enc setComputePipelineState:p_mtp_qk_rope];
            [enc setBuffer:m_qkv offset:0 atIndex:0];
            [enc setBuffer:mtp_file offset:ml.q_norm atIndex:1];
            [enc setBuffer:mtp_file offset:ml.k_norm atIndex:2];
            [enc setBytes:&rp length:sizeof(rp) atIndex:3];
            grid2(mc.heads + mc.kv_heads, 8, 32);
            uint32_t ctx_start = (ml.local && prefix_len > mc.window) ? (prefix_len - mc.window) : 0;
            MtpAttnPP ap{prefix_len, ctx_start, mc.heads, mc.kv_heads, mc.heads / mc.kv_heads, qkv_stride, qdim, kvw, m_kv_cap[i], 1.0f / std::sqrt(float(mc.head_dim))};
            [enc setComputePipelineState:p_mtp_attn];
            [enc setBuffer:m_qkv offset:0 atIndex:0];
            [enc setBuffer:m_kcache[i] offset:0 atIndex:1];
            [enc setBuffer:m_vcache[i] offset:0 atIndex:2];
            [enc setBuffer:m_attn_out offset:0 atIndex:3];
            [enc setBytes:&ap length:sizeof(ap) atIndex:4];
            grid2(mc.heads, 8, 128);
            mtp_gemv_q4(ml.wo, m_attn_out, qdim, m_h1, H, 8, true);
            [enc setComputePipelineState:p_mtp_add_norm];
            [enc setBuffer:m_h offset:0 atIndex:0];
            [enc setBuffer:m_h1 offset:0 atIndex:1];
            [enc setBuffer:mtp_file offset:ml.post_norm atIndex:2];
            [enc setBuffer:m_hn offset:0 atIndex:3];
            [enc setBytes:&np length:sizeof(np) atIndex:4];
            grid(8, 576);
            mtp_gemv_q4(ml.gu, m_hn, H, m_gu, 2 * mc.inter, 8, false);
            [enc setComputePipelineState:p_mtp_silu];
            [enc setBuffer:m_gu offset:0 atIndex:0];
            [enc setBuffer:m_mlp_h offset:0 atIndex:1];
            [enc setBytes:&mc.inter length:4 atIndex:2];
            grid2((mc.inter + 255) / 256, 8, 256);
            mtp_gemv_q4(ml.down, m_mlp_h, mc.inter, m_h1, H, 8, true);
            uint32_t n4 = H / 4;
            [enc setComputePipelineState:p_mtp_add];
            [enc setBuffer:m_h offset:0 atIndex:0];
            [enc setBuffer:m_h1 offset:0 atIndex:1];
            [enc setBytes:&n4 length:4 atIndex:2];
            grid2((n4 + 255) / 256, 8, 256);
        }
        [enc setComputePipelineState:p_mtp_norm_h];
        [enc setBuffer:m_h offset:0 atIndex:0];
        [enc setBuffer:mtp_file offset:m_final_norm atIndex:1];
        [enc setBuffer:m_hn offset:0 atIndex:2];
        [enc setBytes:&np length:sizeof(np) atIndex:3];
        grid(8, 576);
        uint32_t DV = mc.dvocab;
        MtpBasePP bp{DV, H / 32, H / 4};
        [enc setComputePipelineState:p_mtp_base7];
        [enc setBuffer:m_head_q4 offset:0 atIndex:0];
        [enc setBuffer:m_head_sc offset:0 atIndex:1];
        [enc setBuffer:m_hn offset:0 atIndex:2];
        [enc setBuffer:m_base_logits offset:0 atIndex:3];
        [enc setBytes:&bp length:sizeof(bp) atIndex:4];
        grid((DV + 7) / 8, 256);
        uint32_t mntg = (DV + 31) / 32;
        uint32_t rnn_in_dim = 2 * mc.rank + 2 * H;
        float inv_temp = 0.90f;
        for (uint32_t k = 0; k < 7; ++k) {
            MtpRnnInPP rip{mc.rank, H, k};
            [enc setComputePipelineState:p_mtp_rnn_in];
            [enc setBuffer:m_rnn_state offset:0 atIndex:0];
            [enc setBuffer:mtp_file offset:m_markov_w1 atIndex:1];
            [enc setBuffer:m_out offset:0 atIndex:2];
            [enc setBuffer:m_hn offset:0 atIndex:3];
            [enc setBuffer:m_f_rnn offset:0 atIndex:4];
            [enc setBuffer:m_rnn_in offset:0 atIndex:5];
            [enc setBytes:&rip length:sizeof(rip) atIndex:6];
            grid(1, 576);
            mtp_gemv_q4(m_rnn_proj, m_rnn_in, rnn_in_dim, m_rnn_proj_out, 3 * mc.rank, 1, false, m_rnn_bias, true);
            uint32_t rs[2] = {mc.rank, k};
            [enc setComputePipelineState:p_mtp_rnn_gates];
            [enc setBuffer:m_rnn_proj_out offset:0 atIndex:0];
            [enc setBuffer:m_rnn_state offset:0 atIndex:1];
            [enc setBuffer:m_latent offset:0 atIndex:2];
            [enc setBytes:rs length:8 atIndex:3];
            grid(1, 256);
            MtpMarkovPP mkp{DV, k, c.softcap, inv_temp};
            [enc setComputePipelineState:p_mtp_mscan];
            [enc setBuffer:m_fold_q4 offset:0 atIndex:0];
            [enc setBuffer:m_fold_sc offset:0 atIndex:1];
            [enc setBuffer:m_fold_b offset:0 atIndex:2];
            [enc setBuffer:m_base_logits offset:0 atIndex:3];
            [enc setBuffer:m_latent offset:0 atIndex:4];
            [enc setBuffer:m_tg_val offset:0 atIndex:5];
            [enc setBuffer:m_tg_row offset:0 atIndex:6];
            [enc setBuffer:m_tg_sum offset:0 atIndex:7];
            [enc setBytes:&mkp length:sizeof(mkp) atIndex:8];
            grid(mntg, 256);
            MtpReducePP rdp{mntg, k, inv_temp};
            [enc setComputePipelineState:p_mtp_mred];
            [enc setBuffer:m_tg_val offset:0 atIndex:0];
            [enc setBuffer:m_tg_row offset:0 atIndex:1];
            [enc setBuffer:m_tg_sum offset:0 atIndex:2];
            [enc setBuffer:m_vocab_ids offset:0 atIndex:3];
            [enc setBuffer:m_out offset:0 atIndex:4];
            [enc setBuffer:m_out offset:32 atIndex:5];
            [enc setBytes:&rdp length:sizeof(rdp) atIndex:6];
            grid(1, 512);
        }
        [enc endEncoding];
        return gpu_seconds(cb);
    }

    const uint32_t* mtp_draft_ids() const { return static_cast<const uint32_t*>(m_out.contents) + 1; }
    const float* mtp_draft_probs() const { return reinterpret_cast<const float*>(static_cast<const char*>(m_out.contents) + 32); }

    uint32_t chosen_at(uint32_t m) const { return static_cast<const uint32_t*>(token.contents)[16 + m]; }
    uint32_t chosen() const { return chosen_at(0); }

    std::vector<std::pair<float, uint32_t>> top5() const {
        const float* l = static_cast<const float*>(logits.contents);
        double m = -INFINITY;
        for (uint32_t i = 0; i < c.padded_vocab; ++i) m = std::max<double>(m, l[i]);
        double s = 0;
        for (uint32_t i = 0; i < c.padded_vocab; ++i) s += std::exp(double(l[i]) - m);
        double lse = m + std::log(s);
        std::vector<std::pair<float, uint32_t>> best;
        for (uint32_t i = 0; i < c.padded_vocab; ++i) {
            if (best.size() < 5 || l[i] > best.back().first) {
                best.push_back({l[i], i});
                std::sort(best.begin(), best.end(), [](auto& a, auto& b) { return a.first > b.first || (a.first == b.first && a.second < b.second); });
                if (best.size() > 5) best.pop_back();
            }
        }
        for (auto& b : best) b.first = float(b.first - lse);
        return best;
    }
};

std::optional<std::vector<uint32_t>> parse_ids(const std::string& text, std::string& error) {
    NSData* data = [NSData dataWithBytes:text.data() length:text.size()];
    NSError* err = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:data options:0 error:&err];
    if (![parsed isKindOfClass:[NSArray class]]) { error = "--input-ids must be a JSON array of non-negative integers"; return std::nullopt; }
    std::vector<uint32_t> ids;
    for (id item in static_cast<NSArray*>(parsed)) {
        if (![item isKindOfClass:[NSNumber class]] || CFGetTypeID((__bridge CFTypeRef)item) == CFBooleanGetTypeID()) {
            error = "--input-ids elements must be non-negative integers";
            return std::nullopt;
        }
        NSNumber* num = static_cast<NSNumber*>(item);
        double dv = num.doubleValue;
        if (!std::isfinite(dv) || dv < 0.0 || std::floor(dv) != dv || dv > double(UINT32_MAX)) {
            error = "--input-ids elements must be integers in [0, 2^32-1]";
            return std::nullopt;
        }
        ids.push_back(static_cast<uint32_t>(num.unsignedLongLongValue));
    }
    return ids;
}

std::vector<uint32_t> lookup_draft(const std::vector<uint32_t>& history, uint32_t max_k) {
    if (max_k == 0 || history.size() < 6) return {};
    size_t L = history.size();
    size_t best_end = 0;
    size_t best_len = 0;
    size_t scan_start = L > 16384 ? L - 16384 : 0;
    uint32_t last = history[L - 1];
    for (size_t i = L - 2; i > scan_start; --i) {
        if (history[i] != last) continue;
        size_t len = 1;
        while (len < 32 && len <= i && history[i - len] == history[L - 1 - len]) ++len;
        if (len >= 5 && len > best_len) {
            best_len = len;
            best_end = i;
            if (best_len >= 12) break;
        }
    }
    if (best_len < 5) return {};
    uint32_t cap = best_len >= 8 ? 15u : (best_len >= 6 ? 11u : 7u);
    uint32_t k = std::min<uint32_t>(max_k, cap);
    std::vector<uint32_t> out;
    out.reserve(k);
    for (uint32_t j = 1; j <= k && best_end + j < L; ++j) out.push_back(history[best_end + j]);
    return out;
}

struct DecodeStats {
    std::vector<uint32_t> out;
    double prefill_wall = 0;
    double decode_wall = 0;
    double decode_gpu = 0;
    uint64_t verify_steps = 0;
    uint64_t spec_tokens = 0;
};

DecodeStats run_decode_pass(Engine& engine, const std::vector<uint32_t>& ids, uint64_t steps, const std::string& spec, bool logprobs, std::string* steps_json) {
    DecodeStats st;
    engine.reset_state();
    bool use_mtp = engine.mtp_ready && (spec == "mtp" || spec == "auto") && !logprobs;
    bool use_batch_prefill = (spec != "off") && !logprobs;
    uint32_t pos = 0;
    auto w0 = std::chrono::steady_clock::now();
    if (use_batch_prefill && (ids.size() > 1 || use_mtp)) {
        for (size_t i = 0; i < ids.size(); ) {
            uint32_t chunk = uint32_t(std::min<size_t>(8, ids.size() - i));
            bool is_last = (i + chunk == ids.size());
            engine.step_rows(ids.data() + i, pos, chunk, is_last ? Engine::LOGITS_LAST : Engine::LOGITS_NONE, true, logprobs, use_mtp);
            if (use_mtp) engine.mtp_ingest_ctx(pos, chunk, is_last);
            pos += chunk;
            i += chunk;
        }
    } else {
        for (size_t i = 0; i < ids.size(); ++i) {
            engine.step(ids[i], pos++, i + 1 == ids.size(), logprobs);
        }
    }
    auto w1 = std::chrono::steady_clock::now();
    st.out.reserve(steps);
    std::vector<uint32_t> history = ids;
    history.reserve(ids.size() + steps);
    uint32_t anchor = engine.chosen();
    st.out.push_back(anchor);
    history.push_back(anchor);
    if (logprobs && steps_json) {
        auto top = engine.top5();
        std::ostringstream s;
        s << "{\"id\":" << anchor << ",\"top\":[";
        for (size_t k = 0; k < top.size(); ++k) s << (k ? "," : "") << "[" << top[k].first << "," << top[k].second << "]";
        s << "]}";
        *steps_json += s.str();
    }
    bool use_spec = (spec != "off") && !logprobs;
    uint32_t pending_mtp_pos = 0;
    uint32_t pending_mtp_commit = 0;
    double ema_cal = 0.90;
    double last_exp_acc = 0.0;
    bool last_was_mtp = false;
    uint32_t mtp_miss_streak = 0;
    uint32_t mtp_cooldown = 0;
    while (st.out.size() < steps) {
        uint64_t rem = steps - st.out.size();
        uint32_t ctx_room = pos < engine.max_context ? engine.max_context - pos : 0;
        if (ctx_room == 0) break;
        std::vector<uint32_t> draft;
        last_was_mtp = false;
        last_exp_acc = 0.0;
        if (use_spec && rem > 1 && ctx_room > 1) {
            uint32_t max_k = uint32_t(std::min<uint64_t>({uint64_t(Engine::MAX_BATCH - 1), rem - 1, uint64_t(ctx_room - 1)}));
            if (spec == "lookup" || spec == "auto") draft = lookup_draft(history, max_k);
            if (draft.empty() && use_mtp) {
                if (mtp_cooldown > 0) {
                    mtp_cooldown--;
                } else {
                    st.decode_gpu += engine.mtp_propose(anchor, pos, pending_mtp_pos, pending_mtp_commit);
                    pending_mtp_commit = 0;
                    const uint32_t* dids = engine.mtp_draft_ids();
                    const float* dprobs = engine.mtp_draft_probs();
                    uint32_t k_lim = std::min<uint32_t>(7u, max_k);
                    uint32_t best_k = 0;
                    double base_rate = 1.0 / engine.cost_ms[1];
                    double best_rate = base_rate;
                    double cum_p = 1.0;
                    double exp_tok = 1.0;
                    double best_exp = 0.0;
                    for (uint32_t k = 1; k <= k_lim; ++k) {
                        double pk = std::clamp(double(dprobs[k - 1]) * ema_cal, 0.0, 0.99);
                        cum_p *= pk;
                        exp_tok += cum_p;
                        double rate = exp_tok / engine.cost_ms[k + 1];
                        if (rate > best_rate) {
                            best_rate = rate;
                            best_k = k;
                            best_exp = exp_tok - 1.0;
                        }
                    }
                    if (best_k > 0) {
                        draft.assign(dids, dids + best_k);
                        last_was_mtp = true;
                        last_exp_acc = best_exp;
                    } else if (engine.lean_q4_ready) {
                        mtp_miss_streak++;
                        if (mtp_miss_streak >= 3) mtp_cooldown = std::min<uint32_t>(6u, mtp_miss_streak - 2u);
                    }
                }
            }
        }
        uint32_t ing_pos = pending_mtp_pos;
        uint32_t ing_commit = use_mtp ? pending_mtp_commit : 0;
        pending_mtp_commit = 0;
        if (draft.empty()) {
            uint32_t old_pos = pos;
            st.decode_gpu += engine.step_rows(&anchor, pos, 1, Engine::LOGITS_ALL, true, logprobs, use_mtp, ing_pos, ing_commit);
            pos++;
            st.verify_steps++;
            anchor = engine.chosen();
            st.out.push_back(anchor);
            history.push_back(anchor);
            st.spec_tokens++;
            if (use_mtp) {
                pending_mtp_pos = old_pos;
                pending_mtp_commit = 1;
            }
            if (logprobs && steps_json) {
                auto top = engine.top5();
                std::ostringstream s;
                s << ",{\"id\":" << anchor << ",\"top\":[";
                for (size_t k = 0; k < top.size(); ++k) s << (k ? "," : "") << "[" << top[k].first << "," << top[k].second << "]";
                s << "]}";
                *steps_json += s.str();
            }
        } else {
            uint32_t K = uint32_t(draft.size());
            uint32_t M = K + 1;
            std::array<uint32_t, Engine::MAX_BATCH> batch{};
            batch[0] = anchor;
            for (uint32_t j = 0; j < K; ++j) batch[1 + j] = draft[j];
            uint32_t old_pos = pos;
            st.decode_gpu += engine.step_rows(batch.data(), pos, M, Engine::LOGITS_ALL, false, false, use_mtp, ing_pos, ing_commit);
            st.verify_steps++;
            uint32_t accepted = 0;
            uint32_t next_tok = 0;
            for (uint32_t j = 0; j < K; ++j) {
                uint32_t v = engine.chosen_at(j);
                if (v == draft[j]) {
                    accepted++;
                    st.out.push_back(v);
                    history.push_back(v);
                } else {
                    next_tok = v;
                    st.out.push_back(next_tok);
                    history.push_back(next_tok);
                    break;
                }
            }
            if (accepted == K) {
                next_tok = engine.chosen_at(K);
                st.out.push_back(next_tok);
                history.push_back(next_tok);
            }
            uint32_t commit = accepted + 1;
            pos += commit;
            engine.pending_conv_commit = commit;
            anchor = next_tok;
            st.spec_tokens += commit;
            if (use_mtp) {
                pending_mtp_pos = old_pos;
                pending_mtp_commit = commit;
                if (last_was_mtp) {
                    if (accepted >= 1) {
                        mtp_miss_streak = 0;
                        mtp_cooldown = 0;
                    } else if (engine.lean_q4_ready) {
                        mtp_miss_streak++;
                        if (mtp_miss_streak >= 3) mtp_cooldown = std::min<uint32_t>(6u, mtp_miss_streak - 2u);
                    }
                    if (last_exp_acc > 0.1) {
                        double ratio = std::clamp((double(accepted) + 0.15) / (last_exp_acc + 0.15), 0.4, 1.2);
                        ema_cal = std::clamp(0.75 * ema_cal + 0.25 * (ema_cal * ratio), 0.45, 1.15);
                    }
                }
            }
        }
    }
    auto w2 = std::chrono::steady_clock::now();
    st.prefill_wall = std::chrono::duration<double>(w1 - w0).count();
    st.decode_wall = std::chrono::duration<double>(w2 - w1).count();
    return st;
}

}

int generate_command(int argc, char** argv) {
    std::string error;
    auto flags = parse_flags(argc, argv, {"--model", "--input-ids", "--steps", "--head", "--pack", "--spec", "--mode"}, {"--logprobs"}, error);
    if (!flags) return refuse(error);
    auto steps = whole_flag(*flags, "--steps", 8, 1, 4000, error);
    if (!steps) return refuse(error);
    std::string head = flags->values.count("--head") ? flags->values["--head"] : "exact";
    if (head != "exact" && head != "full") return refuse("--head must be exact or full");
    std::string pack = flags->values.count("--pack") ? flags->values["--pack"] : "bf12";
    if (pack != "bf12" && pack != "off") return refuse("--pack must be bf12 or off");
    std::string spec = flags->values.count("--spec") ? flags->values["--spec"] : "auto";
    if (spec != "off" && spec != "lookup" && spec != "mtp" && spec != "auto") return refuse("--spec must be off, lookup, mtp, or auto");
    std::string mode = flags->values.count("--mode") ? flags->values["--mode"] : "exact";
    if (mode != "exact" && mode != "lean" && mode != "turbo") return refuse("--mode must be exact, lean, or turbo");
    if (!flags->values.count("--model") || !flags->values.count("--input-ids")) return refuse("generate needs --model DIR and --input-ids JSON");
    auto ids_opt = parse_ids(flags->values["--input-ids"], error);
    if (!ids_opt) return refuse(error);
    auto& ids = *ids_opt;
    if (ids.empty()) return refuse("--input-ids must hold at least one token id");
    bool logprobs = flags->switches.count("--logprobs") > 0;
    bool use_cert = head == "exact" && !logprobs;
    bool want_mtp = (spec == "mtp" || spec == "auto") && !logprobs;
    Engine engine;
    if (!engine.load(flags->values["--model"], use_cert, pack == "bf12", want_mtp, mode, error)) {
        std::fprintf(stderr, "delphi: %s\n", error.c_str());
        return 3;
    }
    if (ids.size() > engine.max_context || (*steps - 1) > engine.max_context - ids.size()) {
        return refuse("prompt plus generated steps exceeds 32768-token context");
    }
    uint64_t embed_rows = engine.T("embed.weight").shape[0];
    for (uint32_t id : ids) {
        if (id >= embed_rows) return refuse("input token id out of embedding range");
    }
    std::string steps_json;
    DecodeStats st = run_decode_pass(engine, ids, *steps, spec, logprobs, &steps_json);
    double dec_toks = *steps > 1 ? double(*steps - 1) : 0.0;
    double acc_per_step = st.verify_steps ? double(st.spec_tokens) / double(st.verify_steps) : 1.0;
    bool is_lossy = mode != "exact";
    std::printf("{\"output_ids\": [");
    for (size_t i = 0; i < st.out.size(); ++i) std::printf(i ? ", %u" : "%u", st.out[i]);
    std::printf("], \"mode\": \"%s\", \"lossy\": %s, \"prompt_tokens\": %zu, \"prefill_tokens_per_s\": %.3f, \"decode_tokens_per_s\": %.3f, \"decode_gpu_tokens_per_s\": %.3f, \"verify_steps\": %llu, \"accepted_per_step\": %.3f%s%s%s}\n",
                mode.c_str(), is_lossy ? "true" : "false",
                ids.size(), ids.size() / st.prefill_wall, dec_toks / st.decode_wall, st.decode_gpu > 0 ? dec_toks / st.decode_gpu : 0.0,
                static_cast<unsigned long long>(st.verify_steps), acc_per_step,
                logprobs ? ", \"steps\": [" : "", steps_json.c_str(), logprobs ? "]" : "");
    return 0;
}

int bench_command(int argc, char** argv) {
    std::string error;
    auto flags = parse_flags(argc, argv, {"--model", "--input-ids", "--steps", "--repeats", "--warmup", "--head", "--pack", "--spec", "--mode"}, {}, error);
    if (!flags) return refuse(error);
    auto steps = whole_flag(*flags, "--steps", 64, 2, 4000, error);
    auto repeats = steps ? whole_flag(*flags, "--repeats", 6, 1, 100, error) : std::nullopt;
    auto warmup = repeats ? whole_flag(*flags, "--warmup", 1, 0, 10, error) : std::nullopt;
    if (!warmup) return refuse(error);
    std::string head = flags->values.count("--head") ? flags->values["--head"] : "exact";
    if (head != "exact" && head != "full") return refuse("--head must be exact or full");
    std::string pack = flags->values.count("--pack") ? flags->values["--pack"] : "bf12";
    if (pack != "bf12" && pack != "off") return refuse("--pack must be bf12 or off");
    std::string spec = flags->values.count("--spec") ? flags->values["--spec"] : "auto";
    if (spec != "off" && spec != "lookup" && spec != "mtp" && spec != "auto") return refuse("--spec must be off, lookup, mtp, or auto");
    std::string mode = flags->values.count("--mode") ? flags->values["--mode"] : "exact";
    if (mode != "exact" && mode != "lean" && mode != "turbo") return refuse("--mode must be exact, lean, or turbo");
    if (!flags->values.count("--model") || !flags->values.count("--input-ids")) return refuse("bench needs --model DIR and --input-ids JSON");
    auto ids_opt = parse_ids(flags->values["--input-ids"], error);
    if (!ids_opt) return refuse(error);
    auto& ids = *ids_opt;
    if (ids.empty()) return refuse("--input-ids must hold at least one token id");
    Engine engine;
    bool want_mtp = (spec == "mtp" || spec == "auto");
    if (!engine.load(flags->values["--model"], head == "exact", pack == "bf12", want_mtp, mode, error)) {
        std::fprintf(stderr, "delphi: %s\n", error.c_str());
        return 3;
    }
    if (ids.size() > engine.max_context || (*steps - 1) > engine.max_context - ids.size()) {
        return refuse("prompt plus generated steps exceeds 32768-token context");
    }
    uint64_t embed_rows = engine.T("embed.weight").shape[0];
    for (uint32_t id : ids) {
        if (id >= embed_rows) return refuse("input token id out of embedding range");
    }
    std::vector<double> wall_tps, gpu_tps, prefill_tps, gpu_ms_tok;
    std::vector<uint32_t> ref_out;
    uint64_t ref_vsteps = 0;
    double ref_acc = 1.0;
    for (uint64_t r = 0; r < *warmup + *repeats; ++r) {
        DecodeStats st = run_decode_pass(engine, ids, *steps, spec, false, nullptr);
        if (ref_out.empty()) {
            ref_out = st.out;
            ref_vsteps = st.verify_steps;
            ref_acc = st.verify_steps ? double(st.spec_tokens) / double(st.verify_steps) : 1.0;
        } else if (st.out != ref_out) {
            std::fprintf(stderr, "delphi: bench output changed between passes\n");
            return 3;
        }
        if (r >= *warmup) {
            double toks = double(*steps - 1);
            prefill_tps.push_back(ids.size() / st.prefill_wall);
            wall_tps.push_back(toks / st.decode_wall);
            gpu_tps.push_back(toks / st.decode_gpu);
            gpu_ms_tok.push_back(st.decode_gpu * 1000.0 / toks);
        }
    }
    std::string hash = hex64(fnv1a(ref_out.data(), ref_out.size() * sizeof(uint32_t)));
    bool is_lossy = mode != "exact";
    std::printf("{\"mode\": \"%s\", \"lossy\": %s, \"prompt_tokens\": %zu, \"steps\": %llu, \"repeats\": %llu, \"output_hash\": \"%s\", \"verify_steps\": %llu, \"accepted_per_step\": %.3f, \"prefill_tokens_per_s_median\": %.3f, \"decode_tokens_per_s_median\": %.3f, \"decode_gpu_tokens_per_s_median\": %.3f, \"gpu_ms_per_token_median\": %.3f, \"decode_gpu_runs\": [",
                mode.c_str(), is_lossy ? "true" : "false",
                ids.size(), static_cast<unsigned long long>(*steps), static_cast<unsigned long long>(*repeats), hash.c_str(),
                static_cast<unsigned long long>(ref_vsteps), ref_acc, median(prefill_tps), median(wall_tps), median(gpu_tps), median(gpu_ms_tok));
    for (size_t i = 0; i < gpu_tps.size(); ++i) std::printf(i ? ", %.3f" : "%.3f", gpu_tps[i]);
    std::printf("], \"output_ids\": [");
    for (size_t i = 0; i < ref_out.size(); ++i) std::printf(i ? ", %u" : "%u", ref_out[i]);
    std::printf("]}\n");
    return 0;
}
