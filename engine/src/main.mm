#import <Foundation/Foundation.h>

#include <string>

#include "flags.h"
#include "probes.h"

namespace {

constexpr const char* kUsage = "usage: delphi generate --model DIR --input-ids JSON [--steps N] | delphi probe bandwidth|gemv|chain [options]";

int probe_command(int argc, char** argv) {
    std::string name = argc >= 1 ? argv[0] : "";
    if (name == "bandwidth") {
        return probe_bandwidth(argc - 1, argv + 1);
    }
    if (name == "gemv") {
        return probe_gemv(argc - 1, argv + 1);
    }
    if (name == "chain") {
        return probe_chain(argc - 1, argv + 1);
    }
    return refuse(kUsage);
}

}

int main(int argc, char** argv) {
    @autoreleasepool {
        if (argc >= 2 && std::string(argv[1]) == "generate") {
            return generate_command(argc - 2, argv + 2);
        }
        if (argc >= 2 && std::string(argv[1]) == "probe") {
            return probe_command(argc - 2, argv + 2);
        }
        return refuse(kUsage);
    }
}