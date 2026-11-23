# Delphi

Delphi is an inference engine for one model on one machine: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an Apple M5 Pro (20-core GPU, 64 GB, 307 GB/s spec / 266 GB/s measured sustained). It is the Mac sibling of Halo, which runs Qwen3.6-35B-A3B at 249 tok/s exact on AMD Strix Halo.

Status: working Metal C++20 engine with Metal 4.0 Neural Accelerator (`matmul2d`) wide prefill and verify (guarded for macOS 15 compatibility), tail-free 16/8/4-lane BF12/NVFP4/MXFP8 kernels, 8-lane `half4` certified two-stage INT4-g32 exact LM head, 12-bit exact lossless BF16 packing, 3-beam zero-sync GPU Markov tree MTP drafter + bit-exact tree verifier + `M=64` wide suffix lookup, 32K ring-KV long context, and spec-exact opt-in lossy `--mode lean` and `--mode turbo`. Every measured number below comes from `delphi bench` on AC power and is logged in [bench/ledger.jsonl](bench/ledger.jsonl). See [docs/THEORY.md](docs/THEORY.md) for byte ceilings and [docs/PLAN.md](docs/PLAN.md) for the roadmap.

## Measured performance (Apple M5 Pro, AC power)

Waves 1-8 are medians of 6 warm runs (64 decode steps, context 5-224 tokens; 512-token prompt for `Prefill 512`) via `delphi bench` with interleaved controls logged in [bench/ledger.jsonl](bench/ledger.jsonl); the Wave 0 baseline (`30.7` wall / `32.1` GPU) comes from the initial two-run `delphi generate` baseline (`30.82 / 30.60` wall) and its 6-repeat `delphi bench` controls logged in Wave 1. Exact mode is 100% token-identical to `--spec off` greedy on shipped weights (verified 32/32 against SGLang G4 on `fibonacci` and 12/12 against the fp32 numpy oracle on `paris`).

| Build / Mode | Exact? | Plain (`--spec off`) GPU / wall | `story` (`--spec auto`) GPU / wall | `fibonacci` (`--spec auto`) GPU / wall | `copy_code` (`--spec auto`) GPU / wall | `p160` (`--spec auto`) GPU / wall | Prefill 512 |
|---|---|---:|---:|---:|---:|---:|---:|
| MLX (stock / tuned) | ref | 24-25 / 26.6 | - | - | - | - | - |
| llama.cpp (`UD-Q4_K_XL`) | no | 28.4 | - | - | - | - | - |
| Delphi Wave 0 (baseline Metal) | yes | 32.1 / 30.7 | 32.1 / 30.7 | 32.1 / 30.7 | 32.1 / 30.7 | 32.1 / 30.7 | - |
| Delphi Wave 1 (fused kernels + BF16 KV) | yes | 38.5 / 37.3 | 38.5 / 37.3 | 38.5 / 37.3 | 38.5 / 37.3 | - | - |
| Delphi Wave 2 (+ certified exact LM head) | yes | 42.3 / 40.8 | 42.3 / 40.8 | 42.1 / 40.7 | 42.1 / 40.7 | - | - |
| Delphi Wave 3 (+ 12-bit exact BF16 pack) | yes | 46.1 / 44.4 | 46.0 / 44.3 | 46.1 / 44.4 | 45.4 / 43.8 | - | - |
| Delphi Wave 4-5 (PR #8, linear MTP + `M=8`) | yes | 46.1 / 44.5 | 49.7 / 47.9 | 74.0 / 72.2 | 90.1 / 88.9 | 106.5 / 105.0 | 109.1 |
| Delphi Wave 6 (+ Metal 4 prefill + tail-free kernels) | yes | 47.2 / 45.4 | 52.1 / 50.1 | 82.2 / 79.8 | 117.5 / 115.2 | 148.3 / 146.5 | **462.0** |
| **Delphi Exact (`--mode exact`, Wave 7, default)** | **yes** | **47.1 / 45.3** | **57.2 / 55.0** (`1.58` acc) | **85.3 / 82.7** (`3.15` acc) | **132.7 / 130.0** (`7.88` acc) | **345.2 / 342.5** (`63.0` acc) | **462.0** |
| Delphi Lean (`--mode lean`, Wave 8, opt-in lossy) | no | 55.2 / 52.5 | 58.4 / 55.5 (`1.34` acc) | 93.7 / 90.3 (`3.32` acc) | 143.8 / 140.5 (`7.88` acc) | 384.7 / 382.2 (`63.0` acc) | **489.2** |
| Delphi Turbo (`--mode turbo`, Wave 8, opt-in lossy) | no | 77.6 / 72.4 | 77.4 / 72.3 (`1.29` acc) | **140.1 / 133.2** (`3.32` acc) | **192.5 / 185.8** (`6.30` acc) | **575.6 / 570.0** (`63.0` acc) | **755.6** |

Qualification highlights (exact mode unless noted):
- **Speedup vs Wave 0 baseline (`exact`):** **+47% plain decode** (32.1 -> 47.1 tok/s GPU, 73.8% of the 63.8 tok/s exact-repack streaming ceiling), **+78% creative prose** (57.2 tok/s GPU via 3-beam speculative tree), **+166% Python code** (85.3 tok/s GPU), **4.1x copy-code** (132.7 tok/s GPU), **10.8x repetitive copy (`p160`)** (345.2 tok/s GPU via `M=64` Metal 4 lookup), and **4.2x 512-token prefill vs Wave 4-5** (109.1 -> 462.0 tok/s).
- **32K Ring-KV & 16K Needle Recall:** Local attention (40 target visits + 2 MTP layers) uses collision-free 1,056-slot ring buffers while 8 global visits + 1 MTP layer allocate 32,768 slots (**1.39 GB total KV** for full 32K context). Retrieves an exact 5-digit secret buried at token 0 of a **16,305-token** prompt in **58.6s total** (**295.6 tok/s** prefill, 3.30x faster than Wave 4C's 89.7 tok/s; **106.2 tok/s GPU / 105.4 wall** 16K decode, `acc=15.75`).
- **Sustained Thermal Soak:** 60 consecutive 64-step exact `fibonacci` passes (3,840 tokens immediately after 16K prefill) sustained **78.04 tok/s GPU median** (`78.37 -> 77.79 -> 77.91` across first/mid/last 10 passes), 60/60 identical hashes, nominal thermals, and **3.29 GiB** non-mmap GPU footprint (**2.68 GiB** in `lean`/`turbo`).
- **Opt-in Lossy Modes (`lossy: true` in JSON, `--spec off == --spec auto` bit-exact in both modes):**
  - `--mode lean`: Tail-free `half4` 8/2-lane INT4-g32 dense attention and controller projections + Metal 4 `bfloat` Q4 `matmul2d` prefill (`M>=24`) with exact BF16 router, all 8 experts, and certified head. Preserves 64/64 exact tokens on `fibonacci` and `copy_code` and 5/5 exact 4.6K needle retrieval (`390.9 tok/s` prefill) while cutting non-mmap footprint by 620 MiB.
  - `--mode turbo`: `lean` + top-3 expert routing. Preserves Paris answer, clean recursive Python on `fibonacci`, and fluent prose on `story` while hitting **140.1 fib / 192.5 copy_code / 575.6 p160 / 755.6 prefill tok/s** (does not guarantee long-context needle retrieval).

## Fan-out decoding (throughput: one prompt, many continuations, one weight read)

`delphi fanout` decodes up to 64 continuations of one shared prefix together. Every step feeds one token per branch through the model as a single batch, so each weight byte read from memory serves every branch at once. It suits requests that need many independent continuations of the same context: the sections of an outlined answer, several candidate completions, or a batch of short tasks over one document.

This is a throughput feature. It does not make one conversation faster. Fan-out decodes one plain greedy token per branch per step and does not use MTP speculation, so one branch runs at the `generate --spec off` rate (42.6 tok/s here), while default single-stream `generate` (speculation on) runs at 55-84 tok/s on the story and fibonacci prompts and faster on copy-heavy ones. For one or two branches, `generate` is faster. At 64 branches the 152.1 tok/s aggregate is 1.8-2.8x the default single-stream rate. Single-stream `generate`/`bench` speed, output and memory are unchanged by this feature (interleaved controls in the ledger).

Aggregate decode tok/s summed over all branches (`delphi fanout`, 35-token prefix, 128 steps per branch, median of 3 runs, raw runs in [bench/ledger.jsonl](bench/ledger.jsonl), measured at `13691c9`; engine changes since then are the fan-out refusal and reporting changes in `9bd41be`):

| Branches | `exact` (token-identical) | `--tensor-experts` (lossy) | `lean --tensor-experts` (lossy) | `turbo --tensor-experts` (lossy) |
|---:|---:|---:|---:|---:|
| 1 | 42.6 | 42.4 | 53.3 | 70.0 |
| 8 | 123.3 | 122.9 | 133.9 | 224.2 |
| 32 | 151.1 | 199.4 | 222.3 | 322.6 |
| 64 | 152.1 | **243.9** | 279.4 | 382.7 |

Per branch at 64 branches: 2.38 tok/s token-identical, 3.81 with `--tensor-experts`. Prefill of the shared prefix runs at 134-155 tok/s, because fan-out keeps the prefill on the one-row-identical kernels even when the tensor flags are on; over a 16K prefix that prefill takes longer than the decode. The JSON reports `end_to_end_tokens_per_s` (generated tokens over prefill plus decode time) next to the decode-only rate. Branches with shorter heads keep stepping until the lockstep run ends; those extra tokens are not counted, so the rates are conservative. Peak memory footprint: 1.94 GiB at one branch, 3.45 GiB at 64 (exact, 35-token prefix), 6.45 GiB at 64 branches over a 16K prefix.

| 64 branches, prefix length | 35 | 1,024 | 4,096 | 16,384 |
|---|---:|---:|---:|---:|
| `exact` | 152.1 | 130.1 | 121.0 | 97.3 |
| `exact --tensor-experts` | 243.9 | 194.8 | 176.5 | 130.5 |

These prefix-length rows are speed-only runs. Token identity over long prefixes is checked separately by the suite (4 branches over a 4,096-token prefix, and all 64 branches past the 1,024-token local window).

What is and is not exact:
- By default every branch is token-for-token identical to decoding that branch alone with `delphi generate --spec off` in the same `--mode` and `--head`, and the JSON says `"solo_identical": true`. That field states which kernels the run used (none of the lossy tensor paths), not a comparison the run performed. Identity is expected from the kernels: `attend_gated_fan` accumulates keys in the same order as the one-row `attend_gated` kernel, the dense and expert matmuls stay on the batched kernels the `probe gemv` tests prove bit-identical to one row for the matmul shapes they cover, the batched prefill uses those same kernels, and the short-convolution histories are the solo values. The e2e suite checks the tokens: all 64 branches for 128 steps; all 64 branches past the 1,024-token local window; 8 branches with 1-5 token heads past that window; 4 branches over a 4,096-token prefix; 1, 7, 25 and 23 (one below the tensor threshold) branches; 64 branches filling the 32,768-slot KV buffer (each branch sees about 550 logical tokens); `lean` at 8 and 25 branches; `turbo` at 25 branches; and `--head full`. It compares argmax tokens, not logits. CI runs only the non-model tests; the model suite runs on the Mac.
- `--tensor-experts` switches expert matmuls to the Metal 4 `matmul2d` kernels once a batch reaches 24 rows (`TENSOR_MIN_ROWS`, one constant shared by the kernels and the `solo_identical` flag; the suite checks 23 and exactly 24). The run then reports `"lossy": true, "solo_identical": false`. Tensor experts round expert products differently: in the suite, sampled branches keep the solo tokens for their first 8 steps and agree on at least 80% of 24 (measured 100%). A `--tensor-dense` flag that also moved the dense matmuls to `matmul2d` (activations rounded to BF16 first) reached 293.1 tok/s at 64 branches, but 4 sampled branches kept only 49% of their 24 solo tokens, some changing on the first token, so it was removed and the flag is refused.
- Only the lossy flags pass 200 tok/s. The token-identical path tops out near 152 tok/s because the one-row expert kernels are ALU-bound on half-precision products, which the tensor units cannot reproduce bit for bit.

How it works: the prefix is prefilled once. Branch tokens are stored interleaved after it (step `s` of branch `b` lives at slot `prefix + s * branches + b`), and `attend_gated_fan` maps each branch's logical positions onto that layout, reading each key once for the query heads that share it. Every branch attends to the prefix plus only its own tokens. The four short convolutions per layer keep a three-token history per branch in a ring inside the conv state; each step the tree-verify parent slots point at that ring and one row per branch is committed, so the batched kernels run unchanged. Branches run in lockstep: the run lasts `longest head - 1 + steps` steps (`lockstep_steps` in the JSON), and a branch with a shorter head starts and finishes earlier. The KV cache is sized to the run (`prefix + lockstep_steps * branches`, at most 32,768), and the extra conv rows are only reserved when the engine is loaded for fan-out. Sharing prefix keys across four branches per threadgroup was measured slower (718.6 vs 652.9 ms/step at a 16K prefix) and is not used.

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

```bash
/tmp/delphi-bin/delphi fanout \
  --model ~/models/Inkling-2-Mini \
  --prefix-ids '[1548, 43997, 7, 77, 1546, 198]' \
  --branch-ids '[[13], [220, 198], [1548]]' \
  --steps 64
```

`delphi fanout` flags: `--prefix-ids JSON` (shared prefix), `--branch-ids JSON` (1 to 64 non-empty token arrays, one per branch), `--steps N` (tokens generated per branch, default 64; refused when `prefix + lockstep_steps * branches` would exceed 32,768), `--tensor-experts` (opt in to the Metal 4 `matmul2d` expert kernels from 24 branches up; output is then reported `lossy`), plus `--mode`, `--head` and `--pack` as above.

Exact means token-identical to Delphi's own plain greedy decoding on the shipped weights. Faster modes that change the output are opt-in and labelled (`"lossy": true` in JSON output).

## Using Delphi in your project

Delphi is licensed under [Apache-2.0](LICENSE). If you distribute Delphi or anything built from it, include the attribution notices from [NOTICE](NOTICE), as section 4(d) of the license requires: in a NOTICE file, in your source or documentation, or wherever your third-party notices appear. To cite it, use [CITATION.cff](CITATION.cff).
