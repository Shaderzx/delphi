#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
(cd kernels && xxd -i probe.metal) > build/probe_metal.h
clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -Ibuild src/main.mm -framework Metal -framework Foundation -o build/delphi