#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "probe_metal.h"

namespace {

constexpr NSUInteger kThreadsPerGroup = 1024;
constexpr NSUInteger kGroups = 160;
constexpr uint64_t kMaxMib = (uint64_t{UINT32_MAX} + 1) * 16 / (1024 * 1024);

struct ProbeOptions {
    uint64_t mib = 1024;
    int repeats = 5;
};

struct ProbeResult {
    std::string device;
    uint64_t bytes = 0;
    int repeats = 0;
    double seconds_best = 0;
    double seconds_median = 0;
    uint32_t checksum = 0;
};

int refuse(const std::string& message) {
    std::fprintf(stderr, "delphi: %s\n", message.c_str());
    return 2;
}

std::optional<uint64_t> positive_integer(const char* text) {
    char* end = nullptr;
    unsigned long long value = std::strtoull(text, &end, 10);
    if (end == text || *end != '\0' || value == 0) {
        return std::nullopt;
    }
    return value;
}

std::optional<ProbeOptions> parse_probe_options(int argc, char** argv, std::string& error) {
    ProbeOptions options;
    for (int i = 0; i < argc; ++i) {
        std::string flag = argv[i];
        if (i + 1 >= argc) {
            error = flag + " needs a value";
            return std::nullopt;
        }
        auto value = positive_integer(argv[++i]);
        if (flag == "--mib" && value && *value < kMaxMib) {
            options.mib = *value;
        } else if (flag == "--repeats" && value && *value <= INT_MAX) {
            options.repeats = static_cast<int>(*value);
        } else if (flag == "--mib") {
            error = "--mib must be a whole number of MiB from 1 to " + std::to_string(kMaxMib - 1);
            return std::nullopt;
        } else if (flag == "--repeats") {
            error = "--repeats must be a whole number from 1 to " + std::to_string(INT_MAX);
            return std::nullopt;
        } else {
            error = "unknown option " + flag;
            return std::nullopt;
        }
    }
    return options;
}

id<MTLComputePipelineState> pipeline(id<MTLDevice> device, id<MTLLibrary> library, NSString* name) {
    NSError* error = nil;
    id<MTLFunction> function = [library newFunctionWithName:name];
    id<MTLComputePipelineState> state = [device newComputePipelineStateWithFunction:function error:&error];
    if (!state) {
        std::fprintf(stderr, "delphi: pipeline %s: %s\n", name.UTF8String, error.localizedDescription.UTF8String);
        std::exit(3);
    }
    return state;
}

double run_pass(id<MTLCommandQueue> queue, id<MTLComputePipelineState> state, NSArray<id<MTLBuffer>>* buffers, uint32_t count4) {
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    [encoder setComputePipelineState:state];
    [encoder setBuffer:buffers[0] offset:0 atIndex:0];
    [encoder setBytes:&count4 length:sizeof(count4) atIndex:1];
    if (buffers.count > 1) {
        [encoder setBuffer:buffers[1] offset:0 atIndex:2];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(kGroups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
    [encoder endEncoding];
    [commands commit];
    [commands waitUntilCompleted];
    if (commands.status != MTLCommandBufferStatusCompleted) {
        std::fprintf(stderr, "delphi: GPU pass failed: %s\n", commands.error.localizedDescription.UTF8String);
        std::exit(3);
    }
    return commands.GPUEndTime - commands.GPUStartTime;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    size_t mid = values.size() / 2;
    return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

std::optional<ProbeResult> probe_bandwidth(const ProbeOptions& options, std::string& error) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        error = "no Metal device";
        return std::nullopt;
    }
    uint64_t bytes = options.mib * 1024ull * 1024ull;
    uint32_t count4 = static_cast<uint32_t>(bytes / 16);
    NSError* compile_error = nil;
    NSString* source = [[NSString alloc] initWithBytes:probe_metal length:probe_metal_len encoding:NSUTF8StringEncoding];
    id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&compile_error];
    if (!library) {
        error = std::string("shader compile: ") + compile_error.localizedDescription.UTF8String;
        return std::nullopt;
    }
    id<MTLCommandQueue> queue = [device newCommandQueue];
    id<MTLBuffer> data = [device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> total = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    if (!data || !total) {
        error = "could not allocate the probe buffer";
        return std::nullopt;
    }
    run_pass(queue, pipeline(device, library, @"fill_pattern"), @[ data ], count4);
    id<MTLComputePipelineState> sum = pipeline(device, library, @"sum_stream");
    ProbeResult result{device.name.UTF8String, bytes, options.repeats, 0, 0, 0};
    std::vector<double> passes;
    passes.reserve(options.repeats);
    for (int pass = 0; pass < options.repeats; ++pass) {
        *static_cast<uint32_t*>(total.contents) = 0;
        double seconds = run_pass(queue, sum, @[ data, total ], count4);
        uint32_t checksum = *static_cast<uint32_t*>(total.contents);
        if (pass > 0 && checksum != result.checksum) {
            error = "checksum changed between passes";
            return std::nullopt;
        }
        result.checksum = checksum;
        passes.push_back(seconds);
    }
    result.seconds_best = *std::min_element(passes.begin(), passes.end());
    result.seconds_median = median(passes);
    return result;
}

void print_probe(const ProbeResult& result) {
    std::printf("{\"device\": \"%s\", \"bytes\": %llu, \"repeats\": %d, \"seconds_best\": %.9f, \"gb_per_s_best\": %.6f, \"seconds_median\": %.9f, \"gb_per_s_median\": %.6f, \"checksum\": \"0x%08x\"}\n",
                result.device.c_str(), static_cast<unsigned long long>(result.bytes), result.repeats,
                result.seconds_best, result.bytes / result.seconds_best / 1e9, result.seconds_median,
                result.bytes / result.seconds_median / 1e9, result.checksum);
}

int probe_command(int argc, char** argv) {
    if (argc < 1 || std::string(argv[0]) != "bandwidth") {
        return refuse("usage: delphi probe bandwidth [--mib N] [--repeats N]");
    }
    std::string error;
    auto options = parse_probe_options(argc - 1, argv + 1, error);
    if (!options) {
        return refuse(error);
    }
    auto result = probe_bandwidth(*options, error);
    if (!result) {
        std::fprintf(stderr, "delphi: %s\n", error.c_str());
        return 3;
    }
    print_probe(*result);
    return 0;
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc >= 2 && std::string(argv[1]) == "probe") {
            return probe_command(argc - 2, argv + 2);
        }
        return refuse("usage: delphi probe bandwidth [--mib N] [--repeats N]");
    }
}