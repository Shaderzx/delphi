#include "probes.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "flags.h"
#include "gpu.h"

namespace {

constexpr NSUInteger kSimdsPerGroup = 8;
constexpr uint64_t kMaxRows = 32;
constexpr uint32_t kValuesExact = 0;
constexpr uint32_t kValuesRough = 1;

struct FillShape {
    uint32_t seed;
    uint32_t values;
    uint32_t base;
    uint32_t count;
};

struct GemvShape {
    uint32_t rows;
    uint32_t n;
    uint32_t k;
    float scale;
};

struct Kernels {
    id<MTLComputePipelineState> weights;
    id<MTLComputePipelineState> inputs;
    id<MTLComputePipelineState> gemv;
};

Kernels kernels(const Gpu& gpu, uint64_t rows) {
    return {pipeline(gpu, @"fill_weights"), pipeline(gpu, @"fill_inputs"), pipeline_for_rows(gpu, @"gemv_rows", static_cast<uint32_t>(rows))};
}

void fill(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> state, id<MTLBuffer> buffer, FillShape shape) {
    [encoder setComputePipelineState:state];
    [encoder setBuffer:buffer offset:0 atIndex:0];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:1];
    [encoder dispatchThreadgroups:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void gemv(id<MTLComputeCommandEncoder> encoder, const Kernels& k, id<MTLBuffer> w, NSUInteger w_offset, id<MTLBuffer> x, id<MTLBuffer> y, GemvShape shape) {
    [encoder setComputePipelineState:k.gemv];
    [encoder setBuffer:w offset:w_offset atIndex:0];
    [encoder setBuffer:x offset:0 atIndex:1];
    [encoder setBuffer:y offset:0 atIndex:2];
    [encoder setBytes:&shape length:sizeof(shape) atIndex:3];
    NSUInteger groups = (shape.n + kSimdsPerGroup - 1) / kSimdsPerGroup;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kSimdsPerGroup * 32, 1, 1)];
}

uint32_t values_code(const std::string& values) {
    return values == "exact" ? kValuesExact : kValuesRough;
}

std::optional<uint64_t> width_flag(const Flags& flags, const std::string& name, uint64_t fallback, uint64_t high, std::string& error) {
    auto value = whole_flag(flags, name, fallback, 8, high, error);
    if (value && *value % 8 != 0) {
        error = name + " must be a multiple of 8";
        return std::nullopt;
    }
    return value;
}

int gpu_failure(const std::string& error) {
    std::fprintf(stderr, "delphi: %s\n", error.c_str());
    return 3;
}

}

int probe_gemv(int argc, char** argv) {
    std::string error;
    auto flags = parse_flags(argc, argv, {"--rows", "--n", "--k", "--seed", "--values", "--first-row", "--repeats"}, {"--outputs"}, error);
    if (!flags) {
        return refuse(error);
    }
    auto rows = whole_flag(*flags, "--rows", 1, 1, kMaxRows, error);
    auto n = rows ? whole_flag(*flags, "--n", 2304, 1, 32768, error) : std::nullopt;
    auto k = n ? width_flag(*flags, "--k", 2304, 32768, error) : std::nullopt;
    auto seed = k ? whole_flag(*flags, "--seed", 1, 0, UINT32_MAX, error) : std::nullopt;
    auto first_row = seed ? whole_flag(*flags, "--first-row", 0, 0, 65535, error) : std::nullopt;
    auto repeats = first_row ? whole_flag(*flags, "--repeats", 5, 1, 1000, error) : std::nullopt;
    auto values = repeats ? choice_flag(*flags, "--values", {"exact", "rough"}, error) : std::nullopt;
    if (!values) {
        return refuse(error);
    }
    auto gpu = open_gpu(error);
    if (!gpu) {
        return gpu_failure(error);
    }
    Kernels kern = kernels(*gpu, *rows);
    uint32_t code = values_code(*values);
    uint64_t weight_count = *n * *k;
    id<MTLBuffer> w = [gpu->device newBufferWithLength:weight_count * 2 options:MTLResourceStorageModePrivate];
    id<MTLBuffer> x = [gpu->device newBufferWithLength:*rows * *k * sizeof(float) options:MTLResourceStorageModePrivate];
    id<MTLBuffer> y = [gpu->device newBufferWithLength:*rows * *n * sizeof(float) options:MTLResourceStorageModeShared];
    id<MTLCommandBuffer> setup = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [setup computeCommandEncoder];
    fill(encoder, kern.weights, w, {static_cast<uint32_t>(*seed), code, 0, static_cast<uint32_t>(weight_count)});
    fill(encoder, kern.inputs, x, {static_cast<uint32_t>(*seed), code, static_cast<uint32_t>(*first_row * *k), static_cast<uint32_t>(*rows * *k)});
    [encoder endEncoding];
    gpu_seconds(setup);
    GemvShape shape{static_cast<uint32_t>(*rows), static_cast<uint32_t>(*n), static_cast<uint32_t>(*k), 1.0f};
    std::vector<double> passes;
    for (uint64_t pass = 0; pass < *repeats; ++pass) {
        id<MTLCommandBuffer> commands = [gpu->queue commandBuffer];
        id<MTLComputeCommandEncoder> pass_encoder = [commands computeCommandEncoder];
        gemv(pass_encoder, kern, w, 0, x, y, shape);
        [pass_encoder endEncoding];
        passes.push_back(gpu_seconds(commands));
    }
    const float* out = static_cast<const float*>(y.contents);
    std::string hashes;
    std::string outputs;
    for (uint64_t r = 0; r < *rows; ++r) {
        hashes += (r ? ", \"" : "\"") + hex64(fnv1a(out + r * *n, *n * sizeof(float))) + "\"";
        outputs += r ? ", [" : "[";
        for (uint64_t i = 0; i < *n; ++i) {
            char number[32];
            std::snprintf(number, sizeof(number), i ? ", %.17g" : "%.17g", out[r * *n + i] + 0.0f);
            outputs += number;
        }
        outputs += "]";
    }
    double middle = median(passes);
    uint64_t bytes = weight_count * 2;
    std::printf("{\"rows\": %llu, \"n\": %llu, \"k\": %llu, \"seed\": %llu, \"values\": \"%s\", \"first_row\": %llu, \"row_hashes\": [%s], %s\"bytes\": %llu, \"seconds_median\": %.9f, \"gb_per_s_median\": %.6f}\n",
                static_cast<unsigned long long>(*rows), static_cast<unsigned long long>(*n), static_cast<unsigned long long>(*k),
                static_cast<unsigned long long>(*seed), values->c_str(), static_cast<unsigned long long>(*first_row), hashes.c_str(),
                flags->switches.count("--outputs") ? ("\"outputs\": [" + outputs + "], ").c_str() : "",
                static_cast<unsigned long long>(bytes), middle, bytes / middle / 1e9);
    return 0;
}

int probe_chain(int argc, char** argv) {
    std::string error;
    auto flags = parse_flags(argc, argv, {"--layers", "--n", "--rows", "--seed", "--mode", "--repeats"}, {}, error);
    if (!flags) {
        return refuse(error);
    }
    auto layers = whole_flag(*flags, "--layers", 192, 1, 1024, error);
    auto n = layers ? width_flag(*flags, "--n", 2304, 8192, error) : std::nullopt;
    auto rows = n ? whole_flag(*flags, "--rows", 1, 1, kMaxRows, error) : std::nullopt;
    auto seed = rows ? whole_flag(*flags, "--seed", 1, 0, UINT32_MAX, error) : std::nullopt;
    auto repeats = seed ? whole_flag(*flags, "--repeats", 6, 1, 1000, error) : std::nullopt;
    auto mode = repeats ? choice_flag(*flags, "--mode", {"chained", "stepwise"}, error) : std::nullopt;
    if (!mode) {
        return refuse(error);
    }
    uint64_t layer_count = *n * *n;
    if (*layers * layer_count > UINT32_MAX) {
        return refuse("--layers times --n squared must stay under 2^32 weights");
    }
    auto gpu = open_gpu(error);
    if (!gpu) {
        return gpu_failure(error);
    }
    Kernels kern = kernels(*gpu, *rows);
    uint64_t bytes = *layers * layer_count * 2;
    NSUInteger vector_bytes = *rows * *n * sizeof(float);
    id<MTLBuffer> w = [gpu->device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> start = [gpu->device newBufferWithLength:vector_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> ping = [gpu->device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> pong = [gpu->device newBufferWithLength:vector_bytes options:MTLResourceStorageModeShared];
    if (!w || !start || !ping || !pong) {
        return gpu_failure("could not allocate the chain buffers");
    }
    id<MTLCommandBuffer> setup = [gpu->queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [setup computeCommandEncoder];
    fill(encoder, kern.weights, w, {static_cast<uint32_t>(*seed), kValuesRough, 0, static_cast<uint32_t>(*layers * layer_count)});
    fill(encoder, kern.inputs, start, {static_cast<uint32_t>(*seed), kValuesRough, 0, static_cast<uint32_t>(*rows * *n)});
    [encoder endEncoding];
    gpu_seconds(setup);
    GemvShape shape{static_cast<uint32_t>(*rows), static_cast<uint32_t>(*n), static_cast<uint32_t>(*n), std::sqrt(12.0f / static_cast<float>(*n))};
    bool chained = *mode == "chained";
    std::vector<double> passes;
    std::string hash;
    for (uint64_t pass = 0; pass < *repeats; ++pass) {
        id<MTLCommandBuffer> reset = [gpu->queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [reset blitCommandEncoder];
        [blit copyFromBuffer:start sourceOffset:0 toBuffer:ping destinationOffset:0 size:vector_bytes];
        [blit endEncoding];
        gpu_seconds(reset);
        double seconds = 0;
        id<MTLCommandBuffer> commands = nil;
        id<MTLComputeCommandEncoder> step = nil;
        for (uint64_t layer = 0; layer < *layers; ++layer) {
            if (!commands) {
                commands = [gpu->queue commandBuffer];
                step = [commands computeCommandEncoder];
            }
            bool even = layer % 2 == 0;
            gemv(step, kern, w, layer * layer_count * 2, even ? ping : pong, even ? pong : ping, shape);
            if (!chained || layer + 1 == *layers) {
                [step endEncoding];
                seconds += gpu_seconds(commands);
                commands = nil;
            }
        }
        passes.push_back(seconds);
        id<MTLBuffer> last = *layers % 2 ? pong : ping;
        std::string seen = hex64(fnv1a(last.contents, vector_bytes));
        if (pass > 0 && seen != hash) {
            return gpu_failure("chain output changed between passes");
        }
        hash = seen;
    }
    double middle = median(passes);
    std::printf("{\"layers\": %llu, \"n\": %llu, \"rows\": %llu, \"mode\": \"%s\", \"bytes_per_step\": %llu, \"output_hash\": \"%s\", \"repeats\": %llu, \"seconds_median\": %.9f, \"gb_per_s_median\": %.6f}\n",
                static_cast<unsigned long long>(*layers), static_cast<unsigned long long>(*n), static_cast<unsigned long long>(*rows), mode->c_str(),
                static_cast<unsigned long long>(bytes), hash.c_str(), static_cast<unsigned long long>(*repeats), middle, bytes / middle / 1e9);
    return 0;
}