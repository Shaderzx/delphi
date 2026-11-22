#include "gpu.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "kernels_metal.h"

std::optional<Gpu> open_gpu(std::string& error) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        error = "no Metal device";
        return std::nullopt;
    }
    MTLCompileOptions* options = [MTLCompileOptions new];
    options.mathMode = MTLMathModeSafe;
    NSError* compile_error = nil;
    NSString* source = [[NSString alloc] initWithBytes:kernels_metal length:kernels_metal_len encoding:NSUTF8StringEncoding];
    id<MTLLibrary> library = nil;
#if defined(__MAC_26_0)
    if (@available(macOS 26.0, *)) {
        options.languageVersion = MTLLanguageVersion4_0;
        library = [device newLibraryWithSource:source options:options error:&compile_error];
        if (!library) {
            options = [MTLCompileOptions new];
            options.mathMode = MTLMathModeSafe;
            options.languageVersion = MTLLanguageVersion3_2;
            compile_error = nil;
        }
    }
#endif
    if (!library) {
        library = [device newLibraryWithSource:source options:options error:&compile_error];
    }
    if (!library) {
        error = std::string("shader compile: ") + compile_error.localizedDescription.UTF8String;
        return std::nullopt;
    }
    return Gpu{device, [device newCommandQueue], library};
}

namespace {

id<MTLComputePipelineState> pipeline_from(const Gpu& gpu, id<MTLFunction> function, NSString* name, NSError* error) {
    id<MTLComputePipelineState> state = function ? [gpu.device newComputePipelineStateWithFunction:function error:&error] : nil;
    if (!state) {
        std::fprintf(stderr, "delphi: pipeline %s: %s\n", name.UTF8String, error.localizedDescription.UTF8String);
        std::exit(3);
    }
    return state;
}

}

id<MTLComputePipelineState> pipeline(const Gpu& gpu, NSString* name) {
    return pipeline_from(gpu, [gpu.library newFunctionWithName:name], name, nil);
}

id<MTLComputePipelineState> pipeline_for_rows(const Gpu& gpu, NSString* name, uint32_t rows) {
    MTLFunctionConstantValues* constants = [MTLFunctionConstantValues new];
    [constants setConstantValue:&rows type:MTLDataTypeUInt atIndex:0];
    NSError* error = nil;
    id<MTLFunction> function = [gpu.library newFunctionWithName:name constantValues:constants error:&error];
    return pipeline_from(gpu, function, name, error);
}

double gpu_seconds(id<MTLCommandBuffer> commands) {
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

uint64_t fnv1a(const void* data, size_t bytes) {
    uint64_t hash = 14695981039346656037ull;
    const auto* p = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash = (hash ^ p[i]) * 1099511628211ull;
    }
    return hash;
}

std::string hex64(uint64_t value) {
    char text[19];
    std::snprintf(text, sizeof(text), "0x%016llx", static_cast<unsigned long long>(value));
    return text;
}