# Delphi theoretical ceilings (Apple M5 Pro + Inkling-2-Mini)

All ceilings below are grounded in direct hardware measurements on this exact machine and exact byte counts from the checkpoint safetensors header (`model-00001-of-00001.safetensors`, 1,244 tensors, 19,525,870,752 bytes).

## 1. Measured hardware truth (context: not applicable)

- Machine: Apple M5 Pro, 20-core GPU (Metal 4), 64 GB unified LPDDR5X, 307 GB/s spec.
- Sustained GPU streaming bandwidth (measurement, `delphi probe bandwidth --mib 4096/8192 --repeats 12`, context: not applicable, AC power, thermal nominal): **266 GB/s** median (86.6% of spec; best 277-283 GB/s).
- Chained dependent BF16 GEMVs (measurement, `delphi probe chain --layers 192 --n 2304 --rows M`, context: not applicable):
  - `M = 1`: 1.00x (263-272 GB/s)
  - `M = 2`: 0.98x
  - `M = 4`: 0.99x
  - `M = 8`: 1.11x (239-244 GB/s)
  - `M = 16`: 1.88x (scalar kernel compute bound; needs simdgroup matrix above 8 rows)

## 2. Exact per-token byte model (batch 1, context <= 128 tokens)

Inkling-2-Mini runs 24 physical layers with a 12-layer loop core (layers 6-17) executed 3 times per token = **48 layer visits** (44 NVFP4 expert visits + 4 MXFP8 expert visits).

| Component | Exact MB / token | Notes |
|---|---:|---|
| Attention projections (48 visits, BF16) | 2,629.18 | `wq_du`, `wk_dv`, `wv_dv`, `wr_du`, `gate_attn`, `attn_sink_proj`, `wo_ud`, `sconv`, `rel_logits_proj` |
| Loop controllers (24 visits, BF16) | 84.93 | rank-64 `down`, `gain`, `forget` on passes 1 and 2 |
| Router (48 visits, BF16) | 28.34 | `128 x 2304` per visit |
| Routed experts top-8 (48 visits) | 1,839.37 | 44 visits NVFP4 (`4.479 MB/expert`) + 4 visits MXFP8 (`8.211 MB/expert`) |
| Unembed / LM head (BF16, 201,024 rows) | 926.32 | every row has unit L2 norm (1.0000) |
| **Total shipped bytes (context <= 128 tokens)** | **5,508.14** | **20.71 ms floor -> 48.29 tok/s ceiling at 266 GB/s** |

KV cache bytes per token across all 48 visits (8 KV heads x 128 dim x K & V):
- FP32 (baseline): `48 x 8 x 128 x 2 x 4 = 393,216 B/tok` (at 2,048-token context with 1,024-token local window on 40 visits + 2,048 global on 8 visits: 472 MB/step).
- BF16 (exact, shipped in Waves 1-5 including `--mode lean` and `--mode turbo`, since K and V are already BF16-rounded): `196,608 B/tok` (at 2,048-token context: 236 MB/step).
- FP8 (hypothetical future lossy KV): `98,304 B/tok` (at 2,048-token context: 118 MB/step).

## 3. Exact lossless byte-reduction discoveries (measured on checkpoint)

1. **Certified Two-Stage LM Head (INT4-g32 + Cauchy-Schwarz L2 bound)**:
   - Unembed rows have `||w_i||_2 = 1.0000`; real post-norm hidden states have `||x||_2 ~ 128` and top-1-to-median logit gaps of `19-41`.
   - INT4-g32 coarse table + per-row `||w_i - w_hat_i||_2` = **260.1 MB**.
   - Measured on real Inkling hidden states: mean **1,220 / 200,058 survivors** (5.6 MB BF16 rescore), **8/8 exact argmax**, **265.7 MB total** vs 926.3 MB (**-660.6 MB**, 100% mathematically certified exact).
2. **12-bit Exact Lossless BF16 Packing (Attention + Controllers)**:
   - Measured BF16 exponent entropy of attention weights is **2.58 bits** (`10.58 bits` total entropy vs 16 stored).
   - Per-tensor top-15 exponents cover **99.979%** of all attention weights (**0.02102%** outliers, <0.5 per row).
   - Fixed 12-bit body (8-bit sign+mantissa + 4-bit exponent palette index, code 15 = 0.0f) + tiny CSR exact outlier patch packs `2,629 + 85 = 2,714 MB` -> **2,036 MB** (**-678 MB**, 100% bit-exact).

Combining both exact weight reductions drops plain single-token decode from **5,508 MB (48.3 tok/s)** to **4,169 MB (63.8 tok/s ceiling)** with zero numerical change.

## 4. Measured speculative decoding economics

Measured expert union `U(M)` across `M` consecutive tokens on Apple M5 Pro (`oracle/inkling.py` routing traces on `e2e/fixtures/g4-sglang-greedy.json`, 64 decode steps, context `20-84` tokens on `fibonacci` and `15-79` tokens on `story`, averaged over 48 visits; step floors are byte ceilings at 266 GB/s):

| Verify width `M` | `U(M)` code (fib) | `U(M)` prose (story) | Expert GB/step (mean) | Exact step GB (with 12-bit dense + cert head + 0.18 GB MTP) | Exact step floor (ceiling, 266 GB/s) |
|---:|---:|---:|---:|---:|---:|
| 1 | 8.00 | 8.00 | 1.84 | 4.17 | 15.7 ms |
| 2 | 12.77 | 12.12 | 2.86 | 5.37 | 20.2 ms |
| 3 | 16.56 | 15.39 | 3.67 | 6.18 | 23.2 ms |
| 4 | 19.75 | 18.13 | 4.35 | 6.86 | 25.8 ms |
| 5 | 22.54 | 20.51 | 4.95 | 7.46 | 28.0 ms |
| 6 | 25.10 | 22.69 | 5.49 | 8.00 | 30.1 ms |
| 8 | 29.52 | 26.49 | 6.44 | 8.95 | 33.6 ms |
| 12 | 36.17 | 32.36 | 7.88 | 10.40 | 39.1 ms |
| 16 | 40.92 | 36.75 | 8.93 | 11.45 | 43.0 ms |

Measured bundled 8-token MTP drafter (`mtp.safetensors`, replay harness on `fibonacci`, context `20-84` tokens, 64 decode steps): **4.14 committed tokens/step** (`depth hits = [13, 10, 7, 6, 5, 2, 1] / 14`). Optimal `M` ceiling for Fibonacci at 266 GB/s is `M = 6` (`3.93 / 30.1 ms` = **131 tok/s ceiling**); with adaptive `k*` and suffix lookup on copy-heavy code/edits (`M = 12..16`), exact ceilings reach **250-370 tok/s**.

## 5. Hypothetical future lossy ceilings (not yet shipped)

These are byte-model ceilings at 266 GB/s (Apple M5 Pro, context `20-84` tokens) for more aggressive future lossy variants beyond the Wave 5 modes shipped in Section 6:

| Hypothetical variant | Changes vs Exact | `M = 8` GB/step | Floor ms/step | Ceiling at `a = 5.0 / 6.5 / 8.0` |
|---|---|---:|---:|---|
| Future `lean-nvfp4` | NVFP4 attention + controllers (0.72 GB), NVFP4 layers 20-23 experts, FP8 KV | 6.63 GB | 24.9 ms | **201 / 261 / 321 tok/s** |
| Future `turbo-capped` | `lean-nvfp4` + union-capped expert pruning (`U <= 16`) + code-fence-gated lenient acceptance | 4.28 GB | 16.1 ms | **311 / 404 / 497 tok/s** |

## 6. Measured results vs theoretical ceilings (Apple M5 Pro, 266 GB/s sustained)

Waves 1-5 are uncontended medians of 6 warm runs (64 decode steps, context `5-116` tokens) via `delphi bench` on AC power, logged in [bench/ledger.jsonl](../bench/ledger.jsonl); the Wave 0 baseline (`32.1` GPU / `30.7` wall) comes from the initial two-run `delphi generate` baseline (`20-84` tokens) and its 6-repeat `delphi bench` controls logged in Wave 1.

| Stage | Shipped variant | Bytes / step (`M=1`) | Ceiling (`M=1`, 266 GB/s) | Measured `M=1` GPU tok/s (% ceiling) | Measured spec GPU tok/s (`story` / `fib` / `copy_code`) |
|---|---|---:|---:|---:|---:|
| Wave 0 | Shipped BF16 + FP32 KV | 5.51 GB | 48.3 | 32.1 (66.5%) | 32.1 / 32.1 / 32.1 |
| Wave 1 | Fused kernels + BF16 KV | 5.51 GB | 48.3 | 38.5 (79.8%) | 38.5 / 38.5 / 38.5 |
| Wave 2 | + Certified INT4-g32 exact head | 4.85 GB | 54.9 | 42.3 (77.1%) | 42.3 / 42.1 / 42.1 |
| Wave 3 | + 12-bit exact lossless BF16 pack | 4.17 GB | 63.8 | 46.1 (72.2%) | 46.0 / 46.1 / 45.4 |
| Wave 4B/4C | + Bundled 49K INT4-g32 MTP + lookup + 32K ring-KV | 4.17 GB | 63.8 | 46.1 (72.2%) | **49.7 / 74.0 / 90.1** (exact) |
| Wave 5 `--mode lean` | Shipped opt-in INT4-g32 dense projections + BF16 KV (lossy) | 2.81 GB | 94.7 | 58.6 (61.9%) | 54.7 (58.6 off) / 78.8 / 91.1 |
| Wave 5 `--mode turbo` | Shipped opt-in `lean` + top-4 expert routing (lossy) | 1.89 GB | 140.7 | 76.0 (54.0%) | 71.7 (75.9 off) / **101.4 / 129.7** |

Measured exact verify step latency curve (`delphi bench`, Apple M5 Pro, Wave 4A/4B, context `5-94` tokens, real expert union `U(1..8) = {8.0, 13.5, 17.0, 20.3, 23.1, 26.5, 29.6, 31.3}`):
- `M = 1..8`: `{21.6, 30.1, 35.8, 40.3, 47.9, 57.6, 63.4, 71.8} ms` (+ ~1.9 ms bundled MTP draft time for 7 proposals over the 49,152-token `markov_w1`-norm shortlist).
- Remaining gap to the 63.8 tok/s `M=1` ceiling (21.6 ms measured vs 15.7 ms streaming floor = 5.9 ms): grouped NVFP4/MXFP8 ALU dequant overhead (~2.8 ms) plus 48-visit dispatch/norm/sconv fixed overhead (~3.1 ms). Closing the exact speculation gap past 74 tok/s code / 50 tok/s prose depends primarily on higher drafter acceptance (e.g., completing the 12-token DSpine drafter) and higher expert-batch bandwidth at `M = 4..8`.
