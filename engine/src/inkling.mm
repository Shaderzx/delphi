#include "probes.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <tuple>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
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

struct Config {
    uint32_t hidden, layers, heads, kv_heads, head_dim, d_rel, rel_extent, window, block, experts, top, inter, vocab, padded_vocab;
    uint32_t core_start, core_end, passes, n_floor;
    float clip_qk, clip_rel, alpha, gate_cap, up_cap, route_scale, softcap;
    std::vector<bool> local;
};

NSDictionary* json_object(NSData* data) {
    return [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
}

uint32_t u(NSDictionary* d, NSString* k) { return [d[k] unsignedIntValue]; }
float f(NSDictionary* d, NSString* k) { return [d[k] floatValue]; }

struct Slot { uint32_t index; };

class Engine {
public:
    Gpu gpu;
    Config c{};
    std::map<std::string, Tensor> tensors;
    id<MTLBuffer> file = nil;
    std::map<std::string, id<MTLComputePipelineState>> pipes;
    id<MTLBuffer> arena, queries, lut, logits, token, scratch;
    std::map<std::string, std::pair<uint64_t, uint64_t>> scratch_slots;
    std::vector<id<MTLBuffer>> kcache, vcache;
    std::map<std::pair<uint32_t, int>, id<MTLBuffer>> conv;
    uint32_t max_context = 4096;
    uint32_t next_slot = 0;
    uint32_t slots = 4096;
    std::map<std::string, uint64_t> query_offsets;
    id<MTLComputeCommandEncoder> enc = nil;

    bool load(const std::string& dir, std::string& error) {
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
        for (NSString* name in @[@"gemv_bf16", @"rms_norm", @"attn_res", @"sconv", @"clip_rows", @"rel_bias", @"attend", @"gate_out", @"kv_store", @"route",
                                 @"quant_nvfp4", @"quant_mxfp8", @"expert_nvfp4", @"expert_mxfp8", @"swiglu", @"combine", @"ctl_normed", @"ctl_mix",
                                 @"embed_row", @"softcap_logits", @"argmax_logits", @"round_rows"]) {
            pipes[name.UTF8String] = pipeline(gpu, name);
        }
        arena = [gpu.device newBufferWithLength:uint64_t(slots) * c.hidden * 4 options:MTLResourceStorageModePrivate];
        if (!arena) { error = "arena allocation failed"; return false; }
        build_queries();
        if (!queries) { error = "queries allocation failed"; return false; }
        std::vector<float> table(256);
        for (int b = 0; b < 256; ++b) {
            int e = (b >> 3) & 15, m = b & 7;
            float v = e == 0 ? m / 8.0f * std::ldexp(1.0f, -6) : (1 + m / 8.0f) * std::ldexp(1.0f, e - 7);
            if (e == 15 && m == 7) v = NAN;
            table[b] = (b & 0x80) ? -v : v;
        }
        lut = [gpu.device newBufferWithBytes:table.data() length:1024 options:MTLResourceStorageModeShared];
        logits = [gpu.device newBufferWithLength:uint64_t(c.padded_vocab) * 4 options:MTLResourceStorageModeShared];
        token = [gpu.device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        if (!lut || !logits || !token) { error = "workspace allocation failed"; return false; }
        uint32_t H = c.hidden, I = c.inter, K = c.top;
        std::vector<std::pair<std::string, uint64_t>> plan = {
            {"q", c.heads * c.head_dim}, {"qc", c.heads * c.head_dim}, {"r", c.heads * c.d_rel}, {"rc", c.heads * c.d_rel}, {"gate", c.heads * c.head_dim},
            {"sinks", c.heads}, {"kraw", c.kv_heads * c.head_dim}, {"vraw", c.kv_heads * c.head_dim}, {"k", c.kv_heads * c.head_dim},
            {"kc", c.kv_heads * c.head_dim}, {"v", c.kv_heads * c.head_dim}, {"vc", c.kv_heads * c.head_dim}, {"dynk", c.kv_heads * 4},
            {"dynv", c.kv_heads * 4}, {"rel", c.heads * c.rel_extent}, {"o", c.heads * c.head_dim}, {"lse", c.heads},
            {"gated", c.heads * c.head_dim}, {"router", c.experts}, {"ids", 16}, {"gammas", 16}, {"xq", H}, {"outer", 16},
            {"gu", K * 2 * I}, {"h", K * I}, {"hq", K * I}, {"houter", 16}, {"rows", K * H}, {"normed", 2 * H}, {"ctlh", 64},
            {"gain", H}, {"forget", H}};
        uint64_t at = 0;
        for (auto& [name, n] : plan) { scratch_slots[name] = {at, n}; at += (n * 4 + 255) / 256 * 256; }
        scratch = [gpu.device newBufferWithLength:at options:MTLResourceStorageModePrivate];
        if (!scratch) { error = "scratch allocation failed"; return false; }
        uint32_t visits = c.layers + (c.passes - 1) * (c.core_end - c.core_start);
        for (uint32_t v = 0; v < visits; ++v) {
            id<MTLBuffer> kb = [gpu.device newBufferWithLength:uint64_t(max_context) * c.kv_heads * c.head_dim * 4 options:MTLResourceStorageModePrivate];
            id<MTLBuffer> vb = [gpu.device newBufferWithLength:uint64_t(max_context) * c.kv_heads * c.head_dim * 4 options:MTLResourceStorageModePrivate];
            if (!kb || !vb) { error = "kv cache allocation failed"; return false; }
            kcache.push_back(kb);
            vcache.push_back(vb);
            for (int s = 0; s < 4; ++s) {
                uint64_t width = s < 2 ? c.kv_heads * c.head_dim : c.hidden;
                id<MTLBuffer> b = [gpu.device newBufferWithLength:3 * width * 4 options:MTLResourceStorageModeShared];
                if (!b) { error = "conv state allocation failed"; return false; }
                conv[{v, s}] = b;
            }
        }
        return true;
    }

    void build_queries() {
        std::vector<float> all;
        float scale = 1.0f / std::sqrt(float(c.hidden));
        auto add = [&](const std::string& key, const std::string& name, uint64_t row_offset, uint64_t rows) {
            query_offsets[key] = all.size() * 4;
            const Tensor& t = tensors.at(name);
            const float* src = reinterpret_cast<const float*>(static_cast<const char*>(file.contents) + t.offset) + row_offset * c.hidden;
            for (uint64_t i = 0; i < rows * c.hidden; ++i) all.push_back(src[i] * scale);
            while (all.size() % 64) all.push_back(0);
        };
        std::string p = "model.llm.attn_res.";
        for (uint32_t i = 1; i < c.layers; ++i) add("attn" + std::to_string(i), p + "attn_queries." + std::to_string(i - 1), 0, c.local[i] ? 1 : 3);
        for (uint32_t i = 0; i < c.layers; ++i) add("mlp" + std::to_string(i), p + "mlp_queries." + std::to_string(i), 0, 1);
        add("output", p + "output_query", 0, 1);
        uint32_t core = c.core_end - c.core_start;
        for (uint32_t l = 0; l < core; ++l) {
            add("loop_attn" + std::to_string(l) + "_0", "model.llm.middle_cycle_cross_pass_queries", 2 * l, 1);
            uint64_t a = all.size();
            add("loop_attn" + std::to_string(l) + "_1", "model.llm.middle_cycle_cross_pass_kv_queries", 2 * l, 1);
            add("loop_attn" + std::to_string(l) + "_2", "model.llm.middle_cycle_cross_pass_kv_queries", 2 * l + 1, 1);
            (void)a;
            add("loop_mlp" + std::to_string(l), "model.llm.middle_cycle_cross_pass_queries", 2 * l + 1, 1);
        }
        queries = [gpu.device newBufferWithBytes:all.data() length:all.size() * 4 options:MTLResourceStorageModeShared];
    }

    const Tensor& T(const std::string& name) { return tensors.at("model.llm." + name); }
    uint64_t so(uint32_t slot) const { return uint64_t(slot) * c.hidden * 4; }
    uint32_t alloc() { return next_slot++; }

    void bind_s(const std::string& name, int index) { [enc setBuffer:scratch offset:scratch_slots.at(name).first atIndex:index]; }
    void bind_a(uint32_t slot, int index) { [enc setBuffer:arena offset:so(slot) atIndex:index]; }
    void bind_t(const std::string& name, int index) { [enc setBuffer:file offset:T(name).offset atIndex:index]; }
    void use(const std::string& name) { [enc setComputePipelineState:pipes.at(name)]; }
    bool together = false;
    void barrier() { if (!together) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
    void grid(NSUInteger groups, NSUInteger threads) { barrier(); [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }
    void grid2(NSUInteger gx, NSUInteger gy, NSUInteger threads) { barrier(); [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)]; }

    struct GemvP { uint32_t rows, cols, round_out; float scale; };
    void gemv(const std::string& weight, std::function<void(int)> in, std::function<void(int)> out, bool round_out = true) {
        const Tensor& t = T(weight);
        GemvP g{uint32_t(t.shape[0]), uint32_t(t.bytes / 2 / t.shape[0]), round_out ? 1u : 0u, 1.0f};
        use("gemv_bf16");
        bind_t(weight, 0);
        in(1);
        out(2);
        [enc setBytes:&g length:sizeof(g) atIndex:3];
        grid((g.rows + 7) / 8, 256);
    }
    std::function<void(int)> S(const std::string& name) { return [this, name](int i) { bind_s(name, i); }; }
    std::function<void(int)> A(uint32_t slot) { return [this, slot](int i) { bind_a(slot, i); }; }

    struct NormP { uint32_t n, has_weight; float eps; };
    uint32_t norm(uint32_t in, const std::string& weight) {
        uint32_t out = alloc();
        NormP p{c.hidden, 1, 1e-6f};
        use("rms_norm");
        bind_a(in, 0);
        bind_t(weight, 1);
        bind_a(out, 2);
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid(1, 1024);
        return out;
    }

    struct ResP { uint32_t n, sources; int32_t addend; uint32_t queries, normalize, offs[8], outs[3]; };
    std::vector<uint32_t> res(const std::vector<uint32_t>& sources, int addend, const std::string& query, uint32_t nq) {
        ResP p{};
        p.n = c.hidden;
        p.sources = uint32_t(sources.size());
        p.addend = addend >= 0 ? int32_t(addend * c.hidden) : -1;
        p.queries = nq;
        p.normalize = 1;
        for (size_t i = 0; i < sources.size(); ++i) p.offs[i] = sources[i] * c.hidden;
        std::vector<uint32_t> outs;
        for (uint32_t k = 0; k < nq; ++k) { outs.push_back(alloc()); p.outs[k] = outs.back() * c.hidden; }
        use("attn_res");
        [enc setBuffer:arena offset:0 atIndex:0];
        [enc setBuffer:queries offset:query_offsets.at(query) atIndex:1];
        [enc setBytes:&p length:sizeof(p) atIndex:2];
        grid(1, 1024);
        return outs;
    }

    std::vector<uint32_t> loop_messages(const std::vector<uint32_t>& bank, uint32_t core_layer, bool attn_side) {
        std::vector<uint32_t> out;
        if (attn_side) {
            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
            together = true;
            for (int k = 0; k < 3; ++k) out.push_back(res(bank, -1, "loop_attn" + std::to_string(core_layer) + "_" + std::to_string(k), 1)[0]);
            together = false;
        } else {
            out.push_back(res(bank, -1, "loop_mlp" + std::to_string(core_layer), 1)[0]);
        }
        return out;
    }

    uint32_t controller(uint32_t layer, const std::string& name, uint32_t dest, uint32_t message) {
        std::string base = "layers." + std::to_string(layer) + ".controllers." + name + ".";
        uint32_t n = c.hidden;
        use("ctl_normed");
        bind_a(dest, 0);
        bind_a(message, 1);
        bind_s("normed", 2);
        [enc setBytes:&n length:4 atIndex:3];
        grid(1, 1024);
        gemv(base + "down_weight", S("normed"), S("ctlh"));
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        gemv(base + "gain_weight", S("ctlh"), S("gain"), false);
        gemv(base + "forget_weight", S("ctlh"), S("forget"), false);
        together = false;
        uint32_t out = alloc();
        use("ctl_mix");
        bind_a(dest, 0);
        bind_a(message, 1);
        bind_s("gain", 2);
        bind_s("forget", 3);
        bind_a(out, 4);
        [enc setBytes:&n length:4 atIndex:5];
        grid((n + 255) / 256, 256);
        return out;
    }

    struct ConvP { uint32_t channels, dynamic, head_dim; float scale; };
    void sconv(std::function<void(int)> x, id<MTLBuffer> state, const std::string& weight, std::function<void(int)> dyn, std::function<void(int)> y, uint32_t channels, bool dynamic) {
        ConvP p{channels, dynamic ? 1u : 0u, c.head_dim, dynamic ? 1.0f / 3.0f : 0.4472135955f};
        use("sconv");
        x(0);
        [enc setBuffer:state offset:0 atIndex:1];
        bind_t(weight, 2);
        if (dynamic) dyn(3); else bind_s("dynk", 3);
        y(4);
        [enc setBytes:&p length:sizeof(p) atIndex:5];
        grid((channels + 255) / 256, 256);
    }

    struct ClipP { uint32_t width, has_scale; float threshold, tau; uint32_t mode; };
    void clip(const std::string& in, const std::string& scale, const std::string& out, uint32_t rows, uint32_t width, float thr, float tau, uint32_t mode) {
        ClipP p{width, scale.empty() ? 0u : 1u, thr, tau, mode};
        use("clip_rows");
        bind_s(in, 0);
        if (scale.empty()) bind_s(in, 1); else bind_t(scale, 1);
        bind_s(out, 2);
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        grid(rows, 32);
    }

    struct AttnP { uint32_t pos, start, kv_heads, group, head_dim, extent; float scale; };
    uint32_t attention(uint32_t layer, uint32_t visit, uint32_t sq, uint32_t sk, uint32_t sv, uint32_t pos) {
        std::string L = "layers." + std::to_string(layer) + ".attn.";
        bool local = c.local[layer];
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        gemv(L + "wq_du.weight", A(sq), S("q"));
        gemv(L + "wr_du.weight", A(sq), S("r"));
        gemv(L + "gate_attn.weight", A(sq), S("gate"));
        gemv(L + "attn_sink_proj.weight", A(sq), S("sinks"));
        gemv(L + "wk_dv.weight", A(sk), S("kraw"));
        gemv(L + "wv_dv.weight", A(sv), S("vraw"));
        uint32_t kvw = c.kv_heads * c.head_dim;
        if (!local) {
            gemv(L + "k_sconv.weight_proj", A(sk), S("dynk"));
            gemv(L + "v_sconv.weight_proj", A(sv), S("dynv"));
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        if (local) {
            sconv(S("kraw"), conv[{visit, 0}], L + "k_sconv.weight", nullptr, S("k"), kvw, false);
            sconv(S("vraw"), conv[{visit, 1}], L + "v_sconv.weight", nullptr, S("v"), kvw, false);
        } else {
            sconv(S("kraw"), conv[{visit, 0}], L + "k_sconv.static_w", S("dynk"), S("k"), kvw, true);
            sconv(S("vraw"), conv[{visit, 1}], L + "v_sconv.static_w", S("dynv"), S("v"), kvw, true);
        }
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        float tau = local ? 1.0f : 1.0f + c.alpha * std::log(std::max((pos + 1.0f) / float(c.n_floor), 1.0f));
        clip("q", L + "q_l2_clip.scale", "qc", c.heads, c.head_dim, c.clip_qk, tau, 0);
        clip("k", "", "kc", c.kv_heads, c.head_dim, c.clip_qk, 1.0f, 0);
        clip("v", "", "vc", c.kv_heads, c.head_dim, 0, 1.0f, 1);
        clip("r", L + "rel_l2_clip.scale", "rc", c.heads, c.d_rel, c.clip_rel, tau, 0);
        together = false;
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        together = true;
        uint32_t dims[2] = {c.d_rel, c.rel_extent};
        use("rel_bias");
        bind_s("rc", 0);
        bind_t(L + "rel_logits_proj.proj", 1);
        bind_s("rel", 2);
        [enc setBytes:dims length:8 atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake((c.rel_extent + 255) / 256, c.heads, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        uint32_t store[2] = {pos, kvw};
        use("kv_store");
        bind_s("kc", 0);
        bind_s("vc", 1);
        [enc setBuffer:kcache[visit] offset:0 atIndex:2];
        [enc setBuffer:vcache[visit] offset:0 atIndex:3];
        [enc setBytes:store length:8 atIndex:4];
        grid((kvw + 255) / 256, 256);
        together = false;
        uint32_t start = local && pos + 1 > c.window ? pos + 1 - c.window : 0;
        AttnP a{pos, start, c.kv_heads, c.heads / c.kv_heads, c.head_dim, c.rel_extent, 1.0f / std::sqrt(float(c.head_dim))};
        use("attend");
        bind_s("qc", 0);
        [enc setBuffer:kcache[visit] offset:0 atIndex:1];
        [enc setBuffer:vcache[visit] offset:0 atIndex:2];
        bind_s("rel", 3);
        bind_s("o", 4);
        bind_s("lse", 5);
        [enc setBytes:&a length:sizeof(a) atIndex:6];
        grid(c.heads, 256);
        uint32_t gd[2] = {c.heads, c.head_dim};
        use("gate_out");
        bind_s("o", 0);
        bind_s("gate", 1);
        bind_s("lse", 2);
        bind_s("sinks", 3);
        bind_s("gated", 4);
        [enc setBytes:gd length:8 atIndex:5];
        grid((c.heads * c.head_dim + 255) / 256, 256);
        uint32_t out = alloc();
        gemv(L + "wo_ud.weight", S("gated"), A(out));
        return out;
    }

    struct RouteP { uint32_t experts, top; float scale; };
    struct ExpertP { uint32_t rows, cols, in_stride, out_stride, per_row_input; uint32_t pad; uint64_t expert_bytes, scale_bytes; };
    struct SwigluP { uint32_t inter, top; float gate_cap, up_cap; uint32_t round_out; };

    void experts(uint32_t layer, uint32_t x) {
        std::string M = "layers." + std::to_string(layer) + ".mlp.";
        gemv(M + "gate.weight", A(x), S("router"), false);
        RouteP rp{c.experts, c.top, c.route_scale};
        use("route");
        bind_s("router", 0);
        bind_t(M + "gate.bias", 1);
        bind_s("ids", 2);
        bind_s("gammas", 3);
        [enc setBytes:&rp length:sizeof(rp) atIndex:4];
        grid(1, 256);
        bool nv = T(M + "experts.w13_weight").dtype == "U8";
        uint32_t H = c.hidden, I = c.inter;
        auto run_expert = [&](const std::string& w, uint32_t rows, uint32_t cols, const std::string& in, const std::string& outer, const std::string& out, bool per_row) {
            const Tensor& wt = T(M + "experts." + w + "_weight");
            const Tensor& st = T(M + "experts." + w + "_weight_scale");
            ExpertP p{rows, cols, per_row ? I : 0u, per_row ? H : 2 * I, per_row ? 1u : 0u, 0, wt.bytes / c.experts, st.bytes / c.experts};
            use(nv ? "expert_nvfp4" : "expert_mxfp8");
            bind_t(M + "experts." + w + "_weight", 0);
            bind_t(M + "experts." + w + "_weight_scale", 1);
            if (nv) {
                bind_t(M + "experts." + w + "_weight_amax", 2);
                bind_t(M + "experts." + w + "_weight_outer_scale_denominator", 3);
                bind_s(outer, 5);
            }
            bind_s(in, 4);
            bind_s("ids", 6);
            bind_s(out, 7);
            [enc setBytes:&p length:sizeof(p) atIndex:8];
            [enc setBuffer:lut offset:0 atIndex:9];
            grid2((rows + 7) / 8, c.top, 256);
        };
        if (nv) {
            use("quant_nvfp4");
            bind_a(x, 0);
            bind_s("xq", 1);
            bind_s("outer", 2);
            [enc setBytes:&H length:4 atIndex:3];
            grid(1, 256);
        } else {
            use("quant_mxfp8");
            bind_a(x, 0);
            bind_s("xq", 1);
            [enc setBytes:&H length:4 atIndex:2];
            grid2((H / 32 + 63) / 64, 1, 64);
        }
        run_expert("w13", 2 * I, H, "xq", "outer", "gu", false);
        if (!nv) {
            uint32_t n = c.top * 2 * I;
            use("round_rows");
            bind_s("gu", 0);
            [enc setBytes:&n length:4 atIndex:1];
            grid((n + 255) / 256, 256);
        }
        SwigluP sp{I, c.top, c.gate_cap, c.up_cap, nv ? 1u : 0u};
        use("swiglu");
        bind_s("gu", 0);
        bind_s("gammas", 1);
        bind_s("h", 2);
        [enc setBytes:&sp length:sizeof(sp) atIndex:3];
        grid2((I + 255) / 256, c.top, 256);
        if (nv) {
            use("quant_nvfp4");
            bind_s("h", 0);
            bind_s("hq", 1);
            bind_s("houter", 2);
            [enc setBytes:&I length:4 atIndex:3];
            grid(c.top, 256);
        } else {
            use("quant_mxfp8");
            bind_s("h", 0);
            bind_s("hq", 1);
            [enc setBytes:&I length:4 atIndex:2];
            grid2((I / 32 + 63) / 64, c.top, 64);
        }
        run_expert("w2", H, I, "hq", "houter", "rows", true);
    }

    struct CombineP { uint32_t n, top, boundary; };

    std::pair<std::vector<uint32_t>, uint32_t> layer(std::vector<uint32_t> blocks, uint32_t partial, uint32_t layer, uint32_t pass, const std::vector<uint32_t>& bank, uint32_t pos) {
        uint32_t visit = pass == 0 ? layer : c.layers + (pass - 1) * (c.core_end - c.core_start) + (layer - c.core_start);
        bool local = c.local[layer];
        std::string L = "layers." + std::to_string(layer) + ".";
        std::vector<uint32_t> streams;
        if (blocks.empty()) {
            streams.assign(local ? 1 : 3, partial);
        } else {
            std::vector<uint32_t> src = blocks;
            src.push_back(partial);
            streams = res(src, -1, "attn" + std::to_string(layer), local ? 1 : 3);
        }
        uint32_t sq, sk, sv;
        uint32_t core_layer = layer - c.core_start;
        if (!bank.empty()) {
            auto messages = loop_messages(bank, core_layer, true);
            const char* names[3] = {"attn", "k", "v"};
            std::vector<uint32_t> mixed;
            for (int k = 0; k < 3; ++k) mixed.push_back(controller(layer, names[k], local ? streams[0] : streams[k], messages[k]));
            if (local) {
                sq = norm(mixed[0], L + "attn_norm.weight");
                sk = norm(mixed[1], L + "attn_norm.weight");
                sv = norm(mixed[2], L + "attn_norm.weight");
            } else {
                sq = norm(mixed[0], L + "q_input_norm.weight");
                sk = norm(mixed[1], L + "k_input_norm.weight");
                sv = norm(mixed[2], L + "v_input_norm.weight");
            }
        } else if (local) {
            sq = sk = sv = norm(streams[0], L + "attn_norm.weight");
        } else {
            sq = norm(streams[0], L + "q_input_norm.weight");
            sk = norm(streams[1], L + "k_input_norm.weight");
            sv = norm(streams[2], L + "v_input_norm.weight");
        }
        uint32_t att = attention(layer, visit, sq, sk, sv, pos);
        uint32_t attc = alloc();
        sconv(A(att), conv[{visit, 2}], L + "attn_sconv.weight", nullptr, A(attc), c.hidden, false);
        bool boundary = layer % c.block == 0;
        uint32_t mlp_in;
        std::vector<uint32_t> src = blocks;
        src.push_back(partial);
        if (boundary) {
            src.push_back(attc);
            mlp_in = res(src, -1, "mlp" + std::to_string(layer), 1)[0];
        } else {
            mlp_in = res(src, int(attc), "mlp" + std::to_string(layer), 1)[0];
        }
        if (!bank.empty()) {
            auto messages = loop_messages(bank, core_layer, false);
            mlp_in = controller(layer, "mlp", mlp_in, messages[0]);
        }
        uint32_t x = norm(mlp_in, L + "mlp_norm.weight");
        experts(layer, x);
        uint32_t out = alloc();
        CombineP cp{c.hidden, c.top, boundary ? 1u : 0u};
        use("combine");
        bind_s("rows", 0);
        bind_s("ids", 1);
        bind_a(attc, 2);
        bind_a(partial, 3);
        [enc setBuffer:conv[{visit, 3}] offset:0 atIndex:4];
        bind_t(L + "mlp_sconv.weight", 5);
        bind_a(out, 6);
        [enc setBytes:&cp length:sizeof(cp) atIndex:7];
        grid((c.hidden + 255) / 256, 256);
        if (boundary) blocks.push_back(partial);
        return {blocks, out};
    }

    double step(uint32_t tok, uint32_t pos) {
        next_slot = 0;
        *static_cast<uint32_t*>(token.contents) = tok;
        id<MTLCommandBuffer> cb = [gpu.queue commandBuffer];
        enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
        uint32_t emb = alloc();
        uint32_t H = c.hidden;
        use("embed_row");
        bind_t("embed.weight", 0);
        [enc setBuffer:token offset:0 atIndex:1];
        bind_a(emb, 2);
        [enc setBytes:&H length:4 atIndex:3];
        grid((H + 255) / 256, 256);
        uint32_t x = norm(emb, "embed_norm.weight");
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
        std::vector<uint32_t> src = blocks;
        src.push_back(partial);
        uint32_t final = res(src, -1, "output", 1)[0];
        uint32_t hidden = norm(final, "norm.weight");
        const Tensor& un = T("unembed.weight");
        GemvP g{uint32_t(un.shape[0]), uint32_t(un.shape[1]), 1u, 1.0f};
        use("gemv_bf16");
        bind_t("unembed.weight", 0);
        bind_a(hidden, 1);
        [enc setBuffer:logits offset:0 atIndex:2];
        [enc setBytes:&g length:sizeof(g) atIndex:3];
        grid((g.rows + 7) / 8, 256);
        struct { uint32_t n, valid; float cap; } lp{c.padded_vocab, c.vocab, c.softcap};
        use("softcap_logits");
        [enc setBuffer:logits offset:0 atIndex:0];
        [enc setBytes:&lp length:sizeof(lp) atIndex:1];
        grid((c.padded_vocab + 255) / 256, 256);
        use("argmax_logits");
        [enc setBuffer:logits offset:0 atIndex:0];
        [enc setBuffer:token offset:4 atIndex:1];
        [enc setBytes:&c.padded_vocab length:4 atIndex:2];
        grid(1, 1024);
        [enc endEncoding];
        double seconds = gpu_seconds(cb);
        if (next_slot > slots) { std::fprintf(stderr, "delphi: arena overflow %u\n", next_slot); std::exit(3); }
        return seconds;
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
    auto flags = parse_flags(argc, argv, {"--model", "--input-ids", "--steps"}, {"--logprobs"}, error);
    if (!flags) return refuse(error);
    auto steps = whole_flag(*flags, "--steps", 8, 1, 4000, error);
    if (!steps) return refuse(error);
    if (!flags->values.count("--model") || !flags->values.count("--input-ids")) return refuse("generate needs --model DIR and --input-ids JSON");
    auto ids_opt = parse_ids(flags->values["--input-ids"], error);
    if (!ids_opt) return refuse(error);
    auto& ids = *ids_opt;
    if (ids.empty()) return refuse("--input-ids must hold at least one token id");
    Engine engine;
    if (!engine.load(flags->values["--model"], error)) {
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
    bool logprobs = flags->switches.count("--logprobs") > 0;
    uint32_t pos = 0;
    auto wall0 = std::chrono::steady_clock::now();
    for (uint32_t t : ids) engine.step(t, pos++);
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
        if (i + 1 < *steps) decode_gpu += engine.step(next, pos++);
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