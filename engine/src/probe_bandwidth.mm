#include "probes.h"

#include <climits>
#include <cstdio>

#include "flags.h"
#include "gpu.h"

namespace {

constexpr NSUInteger kThreadsPerGroup = 1024;
constexpr NSUInteger kGroups = 160;
constexpr uint64_t kMaxMib = (uint64_t{UINT32_MAX} + 1) * 16 / (1024 * 1024) - 1;

double stream_pass(const Gpu& gpu, id<MTLComputePipelineState> state, NSArray<id<MTLBuffer>>* buffers, uint32_t count4) {
    id<MTLCommandBuffer> commands = [gpu.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:buffers[0] offset:0 atIndex:0];
    [encoder setBytes:&count4 length:sizeof(count4) atIndex:1];
    if (buffers.count > 1) {
        [encoder setBuffer:buffers[1] offset:0 atIndex:2];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(kGroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
    [encoder endEncoding];
    return gpu_seconds(commands);
}

}

int probe_bandwidth(int argc, char** argv) {
    std::string error;
    auto flags = parse_flags(argc, argv, {"--mib", "--repeats"}, {}, error);
    if (!flags) {
        return refuse(error);
    }
    auto mib = whole_flag(*flags, "--mib", 1024, 1, kMaxMib, error);
    auto repeats = mib ? whole_flag(*flags, "--repeats", 5, 1, INT_MAX, error) : std::nullopt;
    if (!repeats) {
        return refuse(error);
    }
    auto gpu = open_gpu(error);
    if (!gpu) {
        std::fprintf(stderr, "delphi: %s\n", error.c_str());
        return 3;
    }
    uint64_t bytes = *mib * 1024ull * 1024ull;
    uint32_t count4 = static_cast<uint32_t>(bytes / 16);
    id<MTLBuffer> data = [gpu->device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> total = [gpu->device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (!data || !total) {
        std::fprintf(stderr, "delphi: could not allocate the probe buffer\n");
        return 3;
    }
    stream_pass(*gpu, pipeline(*gpu, @"fill_pattern"), @[ data ], count4);
    id<MTLComputePipelineState> sum = pipeline(*gpu, @"sum_stream");
    std::vector<double> passes;
    uint32_t checksum = 0;
    for (uint64_t pass = 0; pass < *repeats; ++pass) {
        *static_cast<uint32_t*>(total.contents) = 0;
        passes.push_back(stream_pass(*gpu, sum, @[ data, total ], count4));
        uint32_t seen = *static_cast<uint32_t*>(total.contents);
        if (pass > 0 && seen != checksum) {
            std::fprintf(stderr, "delphi: checksum changed between passes\n");
            return 3;
        }
        checksum = seen;
    }
    double best = *std::min_element(passes.begin(), passes.end());
    double middle = median(passes);
    std::printf("{\"device\": \"%s\", \"bytes\": %llu, \"repeats\": %llu, \"seconds_best\": %.9f, \"gb_per_s_best\": %.6f, \"seconds_median\": %.9f, \"gb_per_s_median\": %.6f, \"checksum\": \"0x%08x\"}\n",
                gpu->device.name.UTF8String, static_cast<unsigned long long>(bytes), static_cast<unsigned long long>(*repeats),
                best, bytes / best / 1e9, middle, bytes / middle / 1e9, checksum);
    return 0;
}