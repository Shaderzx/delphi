#!/usr/bin/env bash
set -euo pipefail
engine="$(cd "$(dirname "$0")" && pwd)"
out="$(mkdir -p "${1:-$engine/build}" && cd "${1:-$engine/build}" && pwd)"
(cd "$engine/kernels" && xxd -i probe.metal) > "$out/probe_metal.h"
clang++ -std=c++20 -O2 -Wall -Wextra -Werror -fobjc-arc -I"$out" "$engine/src/main.mm" -framework Metal -framework Foundation -o "$out/delphi"