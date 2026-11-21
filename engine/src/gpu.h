#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct Gpu {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> library;
};

std::optional<Gpu> open_gpu(std::string& error);
id<MTLComputePipelineState> pipeline(const Gpu& gpu, NSString* name);
double gpu_seconds(id<MTLCommandBuffer> commands);
double median(std::vector<double> values);
uint64_t fnv1a(const void* data, size_t bytes);
std::string hex64(uint64_t value);