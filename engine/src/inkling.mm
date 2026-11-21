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

struct CtlWeights {
    Mat down, gain, forget;
};

struct LayerWeights {
    bool local = false;
    bool nvfp4 = false;
    uint64_t attn_norm = 0, q_norm = 0, k_norm = 0, v_norm = 0, mlp_norm = 0;
    Mat wq, wr, gate_attn, sink, wk, wv, wo, dynk, dynv, router;
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
    id<MTLBuffer> arena, queries, logits, token, scratch;
    id<MTLBuffer> head_q4 = nil, head_sc = nil, head_err = nil;
    id<MTLBuffer> ch_ub = nil, ch_tg_val = nil, ch_tg_row = nil, ch_state = nil, ch_count = nil, ch_cands = nil, ch_indirect = nil;
    bool cert_head_ready = false;
    uint64_t s_q, s_qc, s_r, s_rc, s_gate, s_sinks, s_kraw, s_vraw, s_k, s_v, s_dynk, s_dynv, s_rel, s_gated;
    uint64_t s_router, s_ids, s_gammas, s_xq, s_outer, s_gu, s_h, s_hq, s_houter, s_rows;
    uint64_t s_normed[3], s_ctlh[3], s_gain[3], s_forget[3];
    std::vector<id<MTLBuffer>> kcache, vcache;
    std::vector<std::array<id<MTLBuffer>, 4>> conv;
    uint32_t max_context = 4096;
    uint32_t next_slot = 0;
    uint32_t slots = 4096;
    std::vector<uint64_t> q_attn_off, q_mlp_off, q_loop_attn_off, q_loop_mlp_off;
    uint64_t q_output_off = 0;
    id<MTLComputeCommandEncoder> enc = nil;
    bool together = false;

    const Tensor& T(const std::string& name) const { return tensors.at("model.llm." + name); }

    Mat make_mat(const std::string& name) const {
        const Tensor& t = T(name);
        return {t.offset, uint32_t(t.shape[0]), uint32_t(t.bytes / 2 / t.shape[0])};
    }

    bool load(const std::string& dir, bool use_cert_head, std::string& error) {
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
        arena = [gpu.device newBufferWithLength:uint64_t(slots) * c.hidden * 4 options:MTLResourceStorageModePrivate];
        if (!arena) { error = "arena allocation failed"; return false; }
        build_queries();
        if (!queries) { error = "queries allocation failed"; return false; }
        resolve_weights();
        logits = [gpu.device newBufferWithLength:uint64_t(c.padded_vocab) * 4 options:MTLResourceStorageModeShared];
        token = [gpu.device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        if (!logits || !token) { error = "workspace allocation failed"; return false; }
        uint32_t H = c.hidden, I = c.inter, K = c.top;
        uint64_t at = 0;
        auto slot_alloc = [&](uint64_t count) {
            uint64_t off = at;
            at += (count * 4 + 255) / 256 * 256;
            return off;
        };
        s_q = slot_alloc(c.heads * c.head_dim);
        s_qc = slot_alloc(c.heads * c.head_dim);
        s_r = slot_alloc(c.heads * c.d_rel);
        s_rc = slot_alloc(c.heads * c.d_rel);
        s_gate = slot_alloc(c.heads * c.head_dim);
        s_sinks = slot_alloc(c.heads);
        s_kraw = slot_alloc(c.kv_heads * c.head_dim);
        s_vraw = slot_alloc(c.kv_heads * c.head_dim);
        s_k = slot_alloc(c.kv_heads * c.head_dim);
        s_v = slot_alloc(c.kv_heads * c.head_dim);
        s_dynk = slot_alloc(c.kv_heads * 4);
        s_dynv = slot_alloc(c.kv_heads * 4);
        s_rel = slot_alloc(c.heads * c.rel_extent);
        s_gated = slot_alloc(c.heads * c.head_dim);
        s_router = slot_alloc(c.experts);
        s_ids = slot_alloc(16);
        s_gammas = slot_alloc(16);
        s_xq = slot_alloc(H);
        s_outer = slot_alloc(16);
        s_gu = slot_alloc(K * 2 * I);
        s_h = slot_alloc(K * I);
        s_hq = slot_alloc(K * I);
        s_houter = slot_alloc(16);
        s_rows = slot_alloc(K * H);
        for (int k = 0; k < 3; ++k) {
            s_normed[k] = slot_alloc(2 * H);
            s_ctlh[k] = slot_alloc(64);
            s_gain[k] = slot_alloc(H);
            s_forget[k] = slot_alloc(H);
        }
        scratch = [gpu.device newBufferWithLength:at options:MTLResourceStorageModePrivate];
        if (!scratch) { error = "scratch allocation failed"; return false; }
        uint32_t visits = c.layers + (c.passes - 1) * (c.core_end - c.core_start);
        conv.resize(visits);
        for (uint32_t v = 0; v < visits; ++v) {
            id<MTLBuffer> kb = [gpu.device newBufferWithLength:uint64_t(max_context) * c.kv_heads * c.head_dim * 2 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> vb = [gpu.device newBufferWithLength:uint64_t(max_context) * c.kv_heads * c.head_dim * 2 options:MTLResourceStorageModePrivate];
            if (!kb || !vb) { error = "kv cache allocation failed"; return false; }
            kcache.push_back(kb);
            vcache.push_back(vb);
            for (int s = 0; s < 4; ++s) {
                uint64_t width = s < 2 ? c.kv_heads * c.head_dim : c.hidden;
                id<MTLBuffer> b = [gpu.device newBufferWithLength:3 * width * 4 options:MTLResourceStorageModeShared];
                if (!b) { error = "conv state allocation failed"; return false; }
                conv[v][s] = b;
            }
        }
        if (use_cert_head && (c.hidden % 32 == 0) && c.vocab >= 64) {
            if (!build_cert_head(error)) return false;
        }
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
        ch_ub = [gpu.device newBufferWithLength:uint64_t(V) * 4 options:MTLResourceStorageModePrivate];
        ch_tg_val = [gpu.device newBufferWithLength:uint64_t(ntg) * 4 options:MTLResourceStorageModePrivate];
        ch_tg_row = [gpu.device newBufferWithLength:uint64_t(ntg) * 4 options:MTLResourceStorageModePrivate];
        ch_state = [gpu.device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
        ch_count = [gpu.device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        ch_cands = [gpu.device newBufferWithLength:uint64_t(V) * 4 options:MTLResourceStorageModePrivate];
        ch_indirect = [gpu.device newBufferWithLength:16 options:MTLResourceStorageModePrivate];
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
                    w.ctl[k] = {make_mat(cb + "down_weight"), make_mat(cb + "gain_weight"), make_mat(cb + "forget_weight")};
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

    uint64_t so(uint32_t slot) const { return uint64_t(slot) * c.hidden * 4; }
    uint32_t alloc() { return next_slot++; }

    void barrier() { if (!together) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
    void grid(NSUInteger groups, NSUInteger threads) { barrier(); [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }
    void grid2(NSUInteger gx, NSUInteger gy, NSUInteger threads) { barrier(); [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }

    struct GemvP { uint32_t rows, cols, round_out; float scale; };
    void gemv_ab(const Mat& m, id<MTLBuffer> in_buf, uint64_t in_off, id<MTLBuffer> out_buf, uint64_t out_off, bool round_out = true) {
        GemvP g{m.rows, m.cols, round_out ? 1u : 0u, 1.0f};
        [enc setComputePipelineState:p_gemv];
        [enc setBuffer:file offset:m.offset atIndex:0];
        [enc setBuffer:in_buf offset:in_off atIndex:1];
        [enc setBuffer:out_buf offset:out_off atIndex:2];
        [enc setBytes:&g length:sizeof(g) atIndex:3];
        grid((g.rows + 7) / 8, 256);
    }

    struct NormP { uint32_t n, has_weight; float eps; };
    uint32_t norm(uint32_t in, uint64_t weight_off) {
        uint32_t out = alloc();
        NormP p{c.hidden, 1, 1e-6f};
        [enc setComputePipelineState:p_norm];
        [enc setBuffer:arena offset:so(in) atIndex:0];
        [enc setBuffer:file offset:weight_off atIndex:1];
        [enc setBuffer:arena offset:so(out) atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid(1, 576);
        return out;
    }

    struct ResP { uint32_t n, sources; int32_t addend; uint32_t queries, normalize, has_norm, offs[8], outs[3]; uint32_t pad; uint64_t norm_offs[3]; };
    std::vector<uint32_t> res(const std::vector<uint32_t>& sources, int addend, uint64_t query_off, uint32_t nq, const uint64_t* norm_offs = nullptr) {
        ResP p{};
        p.n = c.hidden;
        p.sources = uint32_t(sources.size());
        p.addend = addend >= 0 ? int32_t(addend * c.hidden) : -1;
        p.queries = nq;
        p.normalize = 1;
        p.has_norm = norm_offs ? 1u : 0u;
        for (size_t i = 0; i < sources.size(); ++i) p.offs[i] = sources[i] * c.hidden;
        std::vector<uint32_t> outs;
        for (uint32_t k = 0; k < nq; ++k) {
            outs.push_back(alloc());
            p.outs[k] = outs.back() * c.hidden;
            if (norm_offs) p.norm_offs[k] = norm_offs[k];
        }
        [enc setComputePipelineState:p_res];
        [enc setBuffer:arena offset:0 atIndex:0];
        [enc setBuffer:queries offset:query_off atIndex:1];
        [enc setBuffer:file offset:0 atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid(1, 576);
        return outs;
    }

    std::array<uint32_t, 3> controllers_attn_norm(const LayerWeights& w, const uint32_t dests[3], const uint32_t msgs[3], const uint64_t norm_offs[3]) {
        uint32_t n = c.hidden;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        for (int k = 0; k < 3; ++k) {
            [enc setComputePipelineState:p_ctl_norm];
            [enc setBuffer:arena offset:so(dests[k]) atIndex:0];
            [enc setBuffer:arena offset:so(msgs[k]) atIndex:1];
            [enc setBuffer:scratch offset:s_normed[k] atIndex:2];
            [enc setBytes:&n length:4 atIndex:3];
            grid(1, 576);
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        for (int k = 0; k < 3; ++k) {
            gemv_ab(w.ctl[k].down, scratch, s_normed[k], scratch, s_ctlh[k]);
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        for (int k = 0; k < 3; ++k) {
            gemv_ab(w.ctl[k].gain, scratch, s_ctlh[k], scratch, s_gain[k], false);
            gemv_ab(w.ctl[k].forget, scratch, s_ctlh[k], scratch, s_forget[k], false);
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        std::array<uint32_t, 3> outs;
        for (int k = 0; k < 3; ++k) {
            outs[k] = alloc();
            [enc setComputePipelineState:p_ctl_mix_norm];
            [enc setBuffer:arena offset:so(dests[k]) atIndex:0];
            [enc setBuffer:arena offset:so(msgs[k]) atIndex:1];
            [enc setBuffer:scratch offset:s_gain[k] atIndex:2];
            [enc setBuffer:scratch offset:s_forget[k] atIndex:3];
            [enc setBuffer:file offset:norm_offs[k] atIndex:4];
            [enc setBuffer:arena offset:so(outs[k]) atIndex:5];
            [enc setBytes:&n length:4 atIndex:6];
            grid(1, 576);
        }
        together = false;
        return outs;
    }

    uint32_t controller_mlp_norm(const LayerWeights& w, uint32_t dest, uint32_t msg) {
        uint32_t n = c.hidden;
        [enc setComputePipelineState:p_ctl_norm];
        [enc setBuffer:arena offset:so(dest) atIndex:0];
        [enc setBuffer:arena offset:so(msg) atIndex:1];
        [enc setBuffer:scratch offset:s_normed[0] atIndex:2];
        [enc setBytes:&n length:4 atIndex:3];
        grid(1, 576);
        gemv_ab(w.ctl[3].down, scratch, s_normed[0], scratch, s_ctlh[0]);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        gemv_ab(w.ctl[3].gain, scratch, s_ctlh[0], scratch, s_gain[0], false);
        gemv_ab(w.ctl[3].forget, scratch, s_ctlh[0], scratch, s_forget[0], false);
        together = false;
        uint32_t out = alloc();
        [enc setComputePipelineState:p_ctl_mix_norm];
        [enc setBuffer:arena offset:so(dest) atIndex:0];
        [enc setBuffer:arena offset:so(msg) atIndex:1];
        [enc setBuffer:scratch offset:s_gain[0] atIndex:2];
        [enc setBuffer:scratch offset:s_forget[0] atIndex:3];
        [enc setBuffer:file offset:w.mlp_norm atIndex:4];
        [enc setBuffer:arena offset:so(out) atIndex:5];
        [enc setBytes:&n length:4 atIndex:6];
        grid(1, 576);
        return out;
    }

    struct ConvP { uint32_t channels, dynamic, head_dim; float scale; };
    void sconv_run(id<MTLBuffer> xb, uint64_t xoff, id<MTLBuffer> state, uint64_t woff, uint64_t dynoff, id<MTLBuffer> yb, uint64_t yoff, uint32_t channels, bool dynamic) {
        ConvP p{channels, dynamic ? 1u : 0u, c.head_dim, dynamic ? 1.0f / 3.0f : 0.4472135955f};
        [enc setComputePipelineState:p_sconv];
        [enc setBuffer:xb offset:xoff atIndex:0];
        [enc setBuffer:state offset:0 atIndex:1];
        [enc setBuffer:file offset:woff atIndex:2];
        [enc setBuffer:scratch offset:dynoff atIndex:3];
        [enc setBuffer:yb offset:yoff atIndex:4];
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        grid((channels + 255) / 256, 256);
    }

    struct ClipP { uint32_t width, has_scale; float threshold, tau; uint32_t mode; };
    void clip_run(uint64_t in_off, uint64_t scale_off, uint64_t out_off, uint32_t rows, uint32_t width, float thr, float tau, uint32_t mode) {
        ClipP p{width, scale_off ? 1u : 0u, thr, tau, mode};
        [enc setComputePipelineState:p_clip];
        [enc setBuffer:scratch offset:in_off atIndex:0];
        if (scale_off) [enc setBuffer:file offset:scale_off atIndex:1]; else [enc setBuffer:scratch offset:in_off atIndex:1];
        [enc setBuffer:scratch offset:out_off atIndex:2];
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid(rows, 32);
    }

    struct ClipKvP { uint32_t width, pos, stride, mode; float threshold; };
    void clip_kv_run(uint64_t in_off, id<MTLBuffer> cache, uint32_t rows, uint32_t width, uint32_t pos, uint32_t stride, uint32_t mode, float thr) {
        ClipKvP p{width, pos, stride, mode, thr};
        [enc setComputePipelineState:p_clip_kv];
        [enc setBuffer:scratch offset:in_off atIndex:0];
        [enc setBuffer:cache offset:0 atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        grid(rows, 32);
    }

    struct AttnP { uint32_t pos, start, kv_heads, group, head_dim, extent; float scale; };
    uint32_t attention(const LayerWeights& w, uint32_t visit, uint32_t sq, uint32_t sk, uint32_t sv, uint32_t pos) {
        uint32_t kvw = c.kv_heads * c.head_dim;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        gemv_ab(w.wq, arena, so(sq), scratch, s_q);
        gemv_ab(w.wr, arena, so(sq), scratch, s_r);
        gemv_ab(w.gate_attn, arena, so(sq), scratch, s_gate);
        gemv_ab(w.sink, arena, so(sq), scratch, s_sinks);
        gemv_ab(w.wk, arena, so(sk), scratch, s_kraw);
        gemv_ab(w.wv, arena, so(sv), scratch, s_vraw);
        if (!w.local) {
            gemv_ab(w.dynk, arena, so(sk), scratch, s_dynk);
            gemv_ab(w.dynv, arena, so(sv), scratch, s_dynv);
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        float tau = w.local ? 1.0f : 1.0f + c.alpha * std::log(std::max((pos + 1.0f) / float(c.n_floor), 1.0f));
        sconv_run(scratch, s_kraw, conv[visit][0], w.k_sconv, s_dynk, scratch, s_k, kvw, !w.local);
        sconv_run(scratch, s_vraw, conv[visit][1], w.v_sconv, s_dynv, scratch, s_v, kvw, !w.local);
        clip_run(s_q, w.q_clip, s_qc, c.heads, c.head_dim, c.clip_qk, tau, 0);
        clip_run(s_r, w.r_clip, s_rc, c.heads, c.d_rel, c.clip_rel, tau, 0);
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        clip_kv_run(s_k, kcache[visit], c.kv_heads, c.head_dim, pos, kvw, 0, c.clip_qk);
        clip_kv_run(s_v, vcache[visit], c.kv_heads, c.head_dim, pos, kvw, 1, 0.0f);
        uint32_t dims[2] = {c.d_rel, c.rel_extent};
        [enc setComputePipelineState:p_rel];
        [enc setBuffer:scratch offset:s_rc atIndex:0];
        [enc setBuffer:file offset:w.rel_proj atIndex:1];
        [enc setBuffer:scratch offset:s_rel atIndex:2];
        [enc setBytes:dims length:8 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((c.rel_extent + 255) / 256, c.heads, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        together = false;
        uint32_t start = w.local && pos + 1 > c.window ? pos + 1 - c.window : 0;
        AttnP a{pos, start, c.kv_heads, c.heads / c.kv_heads, c.head_dim, c.rel_extent, 1.0f / std::sqrt(float(c.head_dim))};
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
        uint32_t out = alloc();
        gemv_ab(w.wo, scratch, s_gated, arena, so(out));
        return out;
    }

    struct RouteP { uint32_t experts, top; float scale; };
    struct ExpertP { uint32_t rows, cols, in_stride, out_stride, per_row_input; uint32_t pad; uint64_t expert_bytes, scale_bytes; };
    struct SwigluP { uint32_t inter, top; float gate_cap, up_cap; uint32_t round_out; };

    void experts(const LayerWeights& w, uint32_t x) {
        uint32_t H = c.hidden, I = c.inter;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        gemv_ab(w.router, arena, so(x), scratch, s_router, false);
        if (w.nvfp4) {
            [enc setComputePipelineState:p_qnv4];
            [enc setBuffer:arena offset:so(x) atIndex:0];
            [enc setBuffer:scratch offset:s_xq atIndex:1];
            [enc setBuffer:scratch offset:s_outer atIndex:2];
            [enc setBytes:&H length:4 atIndex:3];
            grid(1, 160);
        } else {
            [enc setComputePipelineState:p_qmxf8];
            [enc setBuffer:arena offset:so(x) atIndex:0];
            [enc setBuffer:scratch offset:s_xq atIndex:1];
            [enc setBytes:&H length:4 atIndex:2];
            grid2((H / 32 + 63) / 64, 1, 64);
        }
        together = false;
        RouteP rp{c.experts, c.top, c.route_scale};
        [enc setComputePipelineState:p_route];
        [enc setBuffer:scratch offset:s_router atIndex:0];
        [enc setBuffer:file offset:w.router_bias atIndex:1];
        [enc setBuffer:scratch offset:s_ids atIndex:2];
        [enc setBuffer:scratch offset:s_gammas atIndex:3];
        [enc setBytes:&rp length:sizeof(rp) atIndex:4];
        grid(1, 32);
        auto run_expert = [&](bool is_w2, uint32_t rows, uint32_t cols, uint64_t in_off, uint64_t outer_off, uint64_t out_off, bool per_row) {
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
            grid2((rows + 7) / 8, c.top, 256);
        };
        run_expert(false, 2 * I, H, s_xq, s_outer, s_gu, false);
        if (w.nvfp4) {
            SwigluP sp{I, c.top, c.gate_cap, c.up_cap, 1u};
            [enc setComputePipelineState:p_swiglu_qnv4];
            [enc setBuffer:scratch offset:s_gu atIndex:0];
            [enc setBuffer:scratch offset:s_gammas atIndex:1];
            [enc setBuffer:scratch offset:s_hq atIndex:2];
            [enc setBuffer:scratch offset:s_houter atIndex:3];
            [enc setBytes:&sp length:sizeof(sp) atIndex:4];
            grid(c.top, 128);
        } else {
            uint32_t n = c.top * 2 * I;
            [enc setComputePipelineState:p_round];
            [enc setBuffer:scratch offset:s_gu atIndex:0];
            [enc setBytes:&n length:4 atIndex:1];
            grid((n + 255) / 256, 256);
            SwigluP sp{I, c.top, c.gate_cap, c.up_cap, 0u};
            [enc setComputePipelineState:p_swiglu];
            [enc setBuffer:scratch offset:s_gu atIndex:0];
            [enc setBuffer:scratch offset:s_gammas atIndex:1];
            [enc setBuffer:scratch offset:s_h atIndex:2];
            [enc setBytes:&sp length:sizeof(sp) atIndex:3];
            grid2((I + 255) / 256, c.top, 256);
            [enc setComputePipelineState:p_qmxf8];
            [enc setBuffer:scratch offset:s_h atIndex:0];
            [enc setBuffer:scratch offset:s_hq atIndex:1];
            [enc setBytes:&I length:4 atIndex:2];
            grid2((I / 32 + 63) / 64, c.top, 64);
        }
        run_expert(true, H, I, s_hq, s_houter, s_rows, true);
    }

    struct CombineP { uint32_t n, top, boundary; };

    std::pair<std::vector<uint32_t>, uint32_t> layer(std::vector<uint32_t> blocks, uint32_t partial, uint32_t layer_idx, uint32_t pass, const std::vector<uint32_t>& bank, uint32_t pos) {
        const LayerWeights& w = lw[layer_idx];
        uint32_t visit = pass == 0 ? layer_idx : c.layers + (pass - 1) * (c.core_end - c.core_start) + (layer_idx - c.core_start);
        uint32_t sq, sk, sv;
        uint32_t core_layer = layer_idx - c.core_start;
        if (!bank.empty()) {
            std::vector<uint32_t> src = blocks;
            src.push_back(partial);
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            together = true;
            auto streams = res(src, -1, q_attn_off[layer_idx], w.local ? 1 : 3);
            auto messages = res(bank, -1, q_loop_attn_off[core_layer], 3);
            together = false;
            uint32_t dests[3] = {streams[0], w.local ? streams[0] : streams[1], w.local ? streams[0] : streams[2]};
            uint32_t msgs[3] = {messages[0], messages[1], messages[2]};
            uint64_t noffs[3] = {w.local ? w.attn_norm : w.q_norm, w.local ? w.attn_norm : w.k_norm, w.local ? w.attn_norm : w.v_norm};
            auto norms = controllers_attn_norm(w, dests, msgs, noffs);
            sq = norms[0]; sk = norms[1]; sv = norms[2];
        } else if (blocks.empty()) {
            if (w.local) {
                sq = sk = sv = norm(partial, w.attn_norm);
            } else {
                [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                together = true;
                sq = norm(partial, w.q_norm);
                sk = norm(partial, w.k_norm);
                sv = norm(partial, w.v_norm);
                together = false;
            }
        } else {
            std::vector<uint32_t> src = blocks;
            src.push_back(partial);
            if (w.local) {
                uint64_t noffs[3] = {w.attn_norm, 0, 0};
                sq = sk = sv = res(src, -1, q_attn_off[layer_idx], 1, noffs)[0];
            } else {
                uint64_t noffs[3] = {w.q_norm, w.k_norm, w.v_norm};
                auto norms = res(src, -1, q_attn_off[layer_idx], 3, noffs);
                sq = norms[0]; sk = norms[1]; sv = norms[2];
            }
        }
        uint32_t att = attention(w, visit, sq, sk, sv, pos);
        uint32_t attc = alloc();
        sconv_run(arena, so(att), conv[visit][2], w.attn_sconv, s_dynk, arena, so(attc), c.hidden, false);
        bool boundary = layer_idx % c.block == 0;
        std::vector<uint32_t> src = blocks;
        src.push_back(partial);
        int mlp_addend = boundary ? -1 : int(attc);
        if (boundary) src.push_back(attc);
        uint32_t x;
        if (!bank.empty()) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            together = true;
            uint32_t mlp_in = res(src, mlp_addend, q_mlp_off[layer_idx], 1)[0];
            uint32_t msg = res(bank, -1, q_loop_mlp_off[core_layer], 1)[0];
            together = false;
            x = controller_mlp_norm(w, mlp_in, msg);
        } else {
            uint64_t noffs[3] = {w.mlp_norm, 0, 0};
            x = res(src, mlp_addend, q_mlp_off[layer_idx], 1, noffs)[0];
        }
        experts(w, x);
        uint32_t out = alloc();
        CombineP cp{c.hidden, c.top, boundary ? 1u : 0u};
        [enc setComputePipelineState:p_combine];
        [enc setBuffer:scratch offset:s_rows atIndex:0];
        [enc setBuffer:scratch offset:s_ids atIndex:1];
        [enc setBuffer:arena offset:so(attc) atIndex:2];
        [enc setBuffer:arena offset:so(partial) atIndex:3];
        [enc setBuffer:conv[visit][3] offset:0 atIndex:4];
        [enc setBuffer:file offset:w.mlp_sconv atIndex:5];
        [enc setBuffer:arena offset:so(out) atIndex:6];
        [enc setBytes:&cp length:sizeof(cp) atIndex:7];
        grid((c.hidden + 255) / 256, 256);
        if (boundary) blocks.push_back(partial);
        return {blocks, out};
    }

    double step(uint32_t tok, uint32_t pos, bool need_logits = true, bool full_logits = false) {
        next_slot = 0;
        *static_cast<uint32_t*>(token.contents) = tok;
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        uint32_t emb = alloc();
        uint32_t H = c.hidden;
        [enc setComputePipelineState:p_embed];
        [enc setBuffer:file offset:embed_off atIndex:0];
        [enc setBuffer:token offset:0 atIndex:1];
        [enc setBuffer:arena offset:so(emb) atIndex:2];
        [enc setBytes:&H length:4 atIndex:3];
        grid((H + 255) / 256, 256);
        uint32_t x = norm(emb, embed_norm_off);
        std::vector<uint32_t> blocks;
        uint32_t partial = x;
        for (uint32_t l = 0; l < c.core_start; ++l) std::tie(blocks, partial) = layer(blocks, partial, l, 0, {}, pos);
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
            for (uint32_t l = c.core_start; l < c.core_end; ++l) std::tie(blocks, partial) = layer(blocks, partial, l, pass, bank, pos);
        }
        for (uint32_t l = c.core_end; l < c.layers; ++l) std::tie(blocks, partial) = layer(blocks, partial, l, 0, {}, pos);
        if (need_logits) {
            std::vector<uint32_t> src = blocks;
            src.push_back(partial);
            uint64_t final_noffs[3] = {final_norm_off, 0, 0};
            uint32_t hidden = res(src, -1, q_output_off, 1, final_noffs)[0];
            if (cert_head_ready && !full_logits) {
                uint32_t V = c.vocab;
                uint32_t ng = c.hidden / 32;
                uint32_t ntg = (V + 7) / 8;
                CertHeadP chp{V, c.hidden, ng, c.softcap};
                CertTauP ctp{ntg, c.hidden, c.softcap};
                [enc setComputePipelineState:p_ch_xnorm];
                [enc setBuffer:arena offset:so(hidden) atIndex:0];
                [enc setBuffer:ch_state offset:0 atIndex:1];
                [enc setBuffer:ch_count offset:0 atIndex:2];
                [enc setBytes:&H length:4 atIndex:3];
                grid(1, 576);
                [enc setComputePipelineState:p_ch_scan];
                [enc setBuffer:head_q4 offset:0 atIndex:0];
                [enc setBuffer:head_sc offset:0 atIndex:1];
                [enc setBuffer:head_err offset:0 atIndex:2];
                [enc setBuffer:arena offset:so(hidden) atIndex:3];
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
                [enc setBuffer:arena offset:so(hidden) atIndex:3];
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
                [enc setBuffer:arena offset:so(hidden) atIndex:1];
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
                [enc setBuffer:token offset:4 atIndex:3];
                grid(1, 256);
            } else {
                gemv_ab(unembed_mat, arena, so(hidden), logits, 0, true);
                struct { uint32_t n, valid; float cap; } lp{c.padded_vocab, c.vocab, c.softcap};
                [enc setComputePipelineState:p_softcap];
                [enc setBuffer:logits offset:0 atIndex:0];
                [enc setBytes:&lp length:sizeof(lp) atIndex:1];
                grid((c.padded_vocab + 255) / 256, 256);
                [enc setComputePipelineState:p_argmax];
                [enc setBuffer:logits offset:0 atIndex:0];
                [enc setBuffer:token offset:4 atIndex:1];
                [enc setBytes:&c.padded_vocab length:4 atIndex:2];
                grid(1, 1024);
            }
        }
        [enc endEncoding];
        double seconds = gpu_seconds(cb);
        if (next_slot > slots) { std::fprintf(stderr, "delphi: arena overflow %u\n", next_slot); std::exit(3); }
        return seconds;
    }

    void reset_state() {
        for (auto& arr : conv) {
            for (int s = 0; s < 4; ++s) std::memset(arr[s].contents, 0, arr[s].length);
        }
    }

    uint32_t chosen() const { return static_cast<const uint32_t*>(token.contents)[1]; }

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

}

int generate_command(int argc, char** argv) {
    std::string error;
    auto flags = parse_flags(argc, argv, {"--model", "--input-ids", "--steps", "--head"}, {"--logprobs"}, error);
    if (!flags) return refuse(error);
    auto steps = whole_flag(*flags, "--steps", 8, 1, 4000, error);
    if (!steps) return refuse(error);
    std::string head = flags->values.count("--head") ? flags->values["--head"] : "exact";
    if (head != "exact" && head != "full") return refuse("--head must be exact or full");
    if (!flags->values.count("--model") || !flags->values.count("--input-ids")) return refuse("generate needs --model DIR and --input-ids JSON");
    auto ids_opt = parse_ids(flags->values["--input-ids"], error);
    if (!ids_opt) return refuse(error);
    auto& ids = *ids_opt;
    if (ids.empty()) return refuse("--input-ids must hold at least one token id");
    bool logprobs = flags->switches.count("--logprobs") > 0;
    bool use_cert = head == "exact" && !logprobs;
    Engine engine;
    if (!engine.load(flags->values["--model"], use_cert, error)) {
        std::fprintf(stderr, "delphi: %s\n", error.c_str());
        return 3;
    }
    if (ids.size() > engine.max_context || (*steps - 1) > engine.max_context - ids.size()) {
        return refuse("prompt plus generated steps exceeds 4096-token context");
    }
    uint64_t embed_rows = engine.T("embed.weight").shape[0];
    for (uint32_t id : ids) {
        if (id >= embed_rows) return refuse("input token id out of embedding range");
    }
    uint32_t pos = 0;
    auto wall0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < ids.size(); ++i) engine.step(ids[i], pos++, i + 1 == ids.size(), logprobs);
    auto wall1 = std::chrono::steady_clock::now();
    std::vector<uint32_t> out;
    std::string steps_json;
    double decode_gpu = 0;
    for (uint64_t i = 0; i < *steps; ++i) {
        uint32_t next = engine.chosen();
        out.push_back(next);
        if (logprobs) {
            auto top = engine.top5();
            std::ostringstream s;
            s << (i ? "," : "") << "{\"id\":" << next << ",\"top\":[";
            for (size_t k = 0; k < top.size(); ++k) s << (k ? "," : "") << "[" << top[k].first << "," << top[k].second << "]";
            s << "]}";
            steps_json += s.str();
        }
        if (i + 1 < *steps) decode_gpu += engine.step(next, pos++, true, logprobs);
    }
    auto wall2 = std::chrono::steady_clock::now();
    double prefill_wall = std::chrono::duration<double>(wall1 - wall0).count();
    double decode_wall = std::chrono::duration<double>(wall2 - wall1).count();
    std::printf("{\"output_ids\": [");
    for (size_t i = 0; i < out.size(); ++i) std::printf(i ? ", %u" : "%u", out[i]);
    std::printf("], \"prompt_tokens\": %zu, \"prefill_tokens_per_s\": %.3f, \"decode_tokens_per_s\": %.3f, \"decode_gpu_tokens_per_s\": %.3f%s%s%s}\n",
                ids.size(), ids.size() / prefill_wall, *steps > 1 ? (*steps - 1) / decode_wall : 0.0, *steps > 1 ? (*steps - 1) / decode_gpu : 0.0,
                logprobs ? ", \"steps\": [" : "", steps_json.c_str(), logprobs ? "]" : "");
    return 0;
}

int bench_command(int argc, char** argv) {
    std::string error;
    auto flags = parse_flags(argc, argv, {"--model", "--input-ids", "--steps", "--repeats", "--warmup", "--head"}, {}, error);
    if (!flags) return refuse(error);
    auto steps = whole_flag(*flags, "--steps", 64, 2, 4000, error);
    auto repeats = steps ? whole_flag(*flags, "--repeats", 6, 1, 100, error) : std::nullopt;
    auto warmup = repeats ? whole_flag(*flags, "--warmup", 1, 0, 10, error) : std::nullopt;
    if (!warmup) return refuse(error);
    std::string head = flags->values.count("--head") ? flags->values["--head"] : "exact";
    if (head != "exact" && head != "full") return refuse("--head must be exact or full");
    if (!flags->values.count("--model") || !flags->values.count("--input-ids")) return refuse("bench needs --model DIR and --input-ids JSON");
    auto ids_opt = parse_ids(flags->values["--input-ids"], error);
    if (!ids_opt) return refuse(error);
    auto& ids = *ids_opt;
    if (ids.empty()) return refuse("--input-ids must hold at least one token id");
    Engine engine;
    if (!engine.load(flags->values["--model"], head == "exact", error)) {
        std::fprintf(stderr, "delphi: %s\n", error.c_str());
        return 3;
    }
    if (ids.size() > engine.max_context || (*steps - 1) > engine.max_context - ids.size()) {
        return refuse("prompt plus generated steps exceeds 4096-token context");
    }
    uint64_t embed_rows = engine.T("embed.weight").shape[0];
    for (uint32_t id : ids) {
        if (id >= embed_rows) return refuse("input token id out of embedding range");
    }
    std::vector<double> wall_tps, gpu_tps, prefill_tps, gpu_ms_tok;
    std::vector<uint32_t> ref_out;
    for (uint64_t r = 0; r < *warmup + *repeats; ++r) {
        engine.reset_state();
        uint32_t pos = 0;
        auto w0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ids.size(); ++i) engine.step(ids[i], pos++, i + 1 == ids.size(), false);
        auto w1 = std::chrono::steady_clock::now();
        std::vector<uint32_t> out;
        out.reserve(*steps);
        double dgpu = 0;
        for (uint64_t i = 0; i < *steps; ++i) {
            uint32_t next = engine.chosen();
            out.push_back(next);
            if (i + 1 < *steps) dgpu += engine.step(next, pos++, true, false);
        }
        auto w2 = std::chrono::steady_clock::now();
        if (ref_out.empty()) {
            ref_out = out;
        } else if (out != ref_out) {
            std::fprintf(stderr, "delphi: bench output changed between passes\n");
            return 3;
        }
        if (r >= *warmup) {
            double pw = std::chrono::duration<double>(w1 - w0).count();
            double dw = std::chrono::duration<double>(w2 - w1).count();
            prefill_tps.push_back(ids.size() / pw);
            wall_tps.push_back((*steps - 1) / dw);
            gpu_tps.push_back((*steps - 1) / dgpu);
            gpu_ms_tok.push_back(dgpu * 1000.0 / (*steps - 1));
        }
    }
    std::string hash = hex64(fnv1a(ref_out.data(), ref_out.size() * sizeof(uint32_t)));
    std::printf("{\"prompt_tokens\": %zu, \"steps\": %llu, \"repeats\": %llu, \"output_hash\": \"%s\", \"prefill_tokens_per_s_median\": %.3f, \"decode_tokens_per_s_median\": %.3f, \"decode_gpu_tokens_per_s_median\": %.3f, \"gpu_ms_per_token_median\": %.3f, \"decode_gpu_runs\": [",
                ids.size(), static_cast<unsigned long long>(*steps), static_cast<unsigned long long>(*repeats), hash.c_str(),
                median(prefill_tps), median(wall_tps), median(gpu_tps), median(gpu_ms_tok));
    for (size_t i = 0; i < gpu_tps.size(); ++i) std::printf(i ? ", %.3f" : "%.3f", gpu_tps[i]);
    std::printf("], \"output_ids\": [");
    for (size_t i = 0; i < ref_out.size(); ++i) std::printf(i ? ", %u" : "%u", ref_out[i]);
    std::printf("]}\n");
    return 0;
}
