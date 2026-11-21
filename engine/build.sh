#!/usr/bin/env bash
set -euo pipefail
engine="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "${1:-$engine/build}" && cd "${1:-$engine/build}" && pwd)"
for kernel in "$engine"/kernels/*.metal; do cat "$kernel"; echo; done > "$out/kernels.metal"
(cd "$out" && xxd -i kernels.metal) > "$out/kernels_metal.h"
clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -I"$out" "$engine"/src/*.mm -framework Metal -framework Foundation -o "$out/delphi"