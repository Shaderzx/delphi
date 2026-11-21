# Delphi

Delphi is an inference engine for one model on one machine: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an Apple M5 Pro (20-core GPU, 64 GB, 307 GB/s spec / 266 GB/s measured sustained). It is the Mac sibling of Halo, which runs Qwen3.6-35B-A3B at 249 tok/s exact on AMD Strix Halo.

Status: working Metal C++20 engine with exact single-stream decode, certified two-stage INT4-g32 LM head, 12-bit exact lossless BF16 packing, exact row-invariant batched verify/prefill, bundled 3-layer INT4-g32 MTP + suffix-lookup speculation, 32K ring-KV long context, and opt-in lossy `--mode lean` and `--mode turbo`. Every measured number below comes from `delphi bench` on AC power and is logged in [bench/ledger.jsonl](bench/ledger.jsonl). See [docs/THEORY.md](docs/THEORY.md) for byte ceilings and [docs/PLAN.md](docs/PLAN.md) for the roadmap.

## Measured performance (Apple M5 Pro, AC power)

All rows are uncontended medians of 6 warm runs (64 decode steps) via `delphi bench` unless noted. Exact mode is 100% token-identical to `--spec off` greedy on shipped weights (verified 32/32 against SGLang G4 on `fibonacci` and 12/12 against the fp32 numpy oracle on `paris`).

| Build / Mode | Exact? | Plain (`--spec off`) GPU / wall | `story` (`--spec auto`) GPU / wall | `fibonacci` (`--spec auto`) GPU / wall | `copy_code` (`--spec auto`) GPU / wall | Prefill (`M=8`) |
|---|---|---:|---:|---:|---:|---:|
| MLX (stock / tuned) | ref | 24-25 / 26.6 | - | - | - | - |
| llama.cpp (`UD-Q4_K_XL`) | no | 28.4 | - | - | - | - |
| Delphi Wave 0 (baseline Metal) | yes | 32.1 / 30.7 | 32.1 / 30.7 | 32.1 / 30.7 | 32.1 / 30.7 | 32.1 |
| Delphi Wave 1 (fused kernels + BF16 KV) | yes | 38.5 / 37.3 | 38.5 / 37.3 | 38.5 / 37.3 | 38.5 / 37.3 | 38.5 |
| Delphi Wave 2 (+ certified exact LM head) | yes | 42.3 / 40.8 | 42.3 / 40.8 | 42.1 / 40.7 | 42.1 / 40.7 | 43.0 |
| Delphi Wave 3 (+ 12-bit exact BF16 pack) | yes | 46.1 / 44.4 | 46.0 / 44.3 | 46.1 / 44.4 | 45.4 / 43.8 | 46.9 |
| **Delphi Exact (`--mode exact`, default)** | **yes** | **46.1 / 44.5** | **49.7 / 47.9** (`1.47` acc) | **74.0 / 72.2** (`3.32` acc) | **90.1 / 88.9** (`7.00` acc) | **111 tok/s** |
| Delphi Lean (`--mode lean`, opt-in lossy) | no | 58.6 / 56.0 | 54.7 / 52.2 (`58.6` off) | 78.8 / 76.7 (`3.15` acc) | 91.1 / 89.9 (`7.00` acc) | 111 tok/s |
| Delphi Turbo (`--mode turbo`, opt-in lossy) | no | 76.0 / 71.7 | 71.7 / 67.8 (`75.9` off) | **101.4 / 98.0** (`3.00` acc) | **129.7 / 127.2** (`7.00` acc) | **167 tok/s** |

Qualification highlights (exact mode):
- **Speedup vs Wave 0 baseline:** **+44% plain decode** (32.1 -> 46.1 tok/s GPU, 72.2% of the 63.8 tok/s exact-repack streaming ceiling), **+130% on code** (74.0 tok/s GPU), **+181% on copy-code** (90.1 tok/s GPU), **3.5x prefill** (32.1 -> 111 tok/s).
- **32K Ring-KV & 16K Needle Recall:** Local attention (40 target visits + 2 MTP layers) uses collision-free 1,056-slot ring buffers while 8 global visits + 1 MTP layer allocate 32,768 slots (**1.39 GB total KV** for full 32K context). Retrieves an exact 5-digit secret buried at token 0 of a **16,305-token** prompt (`89.7 tok/s` prefill, `78.7 tok/s` GPU decode).
- **Sustained Thermal Soak:** 60 consecutive 64-step exact `fibonacci` passes (3,840 tokens immediately after 184s of 16K prefill) sustained **74.01 tok/s GPU median** (`73.93 -> 74.09 -> 73.93` across first/mid/last 10 passes), 60/60 identical hashes, nominal thermals, and **3.00 GiB** non-mmap GPU footprint (**2.40 GiB** in `lean`/`turbo`).
- **Opt-in Lossy Modes (`lossy: true` in JSON):**
  - `--mode lean`: INT4-g32 dense attention and controller projections with exact BF16 router and certified head. Preserves 64/64 exact tokens on `fibonacci` and `copy_code` and 5/5 exact 4.6K needle retrieval while cutting non-mmap footprint by 648 MiB.
  - `--mode turbo`: `lean` + top-4 expert routing (halving active expert bytes). Preserves Paris answer, 56/64 exact prefix on `fibonacci`, 64/64 on `copy_code`, and fluent prose on `story`, but drops spacing on 4.6K needle retrieval.

## Building and running

```bash
bash engine/build.sh /tmp/delphi-bin

/tmp/delphi-bin/delphi generate \
  --model ~/models/Inkling-2-Mini \
  --input-ids '[1548, 43997, 7, 77, 1546, 198]' \
  --steps 64

/tmp/delphi-bin/delphi bench \
  --model ~/models/Inkling-2-Mini \
  --input-ids '[1548, 43997, 7, 77, 1546, 198]' \
  --steps 64 --repeats 6 --mode exact --spec auto
```

Flags: `--mode exact|lean|turbo` (default `exact`), `--spec off|lookup|mtp|auto` (default `auto`), `--head exact|full` (default `exact`), `--pack bf12|off` (default `bf12`).

Exact means token-identical to Delphi's own plain greedy decoding on the shipped weights. Faster modes that change the output are opt-in and labelled (`"lossy": true` in JSON output).

## Using Delphi in your project

Delphi is licensed under [Apache-2.0](LICENSE). If you distribute Delphi or anything built from it, include the attribution notices from [NOTICE](NOTICE), as section 4(d) of the license requires: in a NOTICE file, in your source or documentation, or wherever your third-party notices appear. To cite it, use [CITATION.cff](CITATION.cff).
