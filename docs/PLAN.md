# Delphi plan

Council round 1 (2026-11-21): Claude Opus 5.5 (xhigh), GPT-6.2 Sol (xhigh), Gemini 4 Neon (xhigh). Two members read the real safetensors headers; the numbers below come from those headers, not from parameter counts.

How to read every speed number in this file: the machine is the M5 Pro below; decoding is one stream (batch 1) unless a row says otherwise; context is 2K tokens unless a row names another length. None of them is a measurement yet. Ceilings and projections come from the byte model in section 2: bytes read per step divided by bandwidth (307 GB/s spec, 260-275 GB/s assumed sustained), times committed tokens per step. Measured numbers will come only from `delphi bench`, logged in `ledger.jsonl`, and will say so.

## 1. The machine and the model

Machine: Apple M5 Pro MacBook Pro, 20-core GPU with a Neural Accelerator per core (FP16/INT8 matmul), Metal 4, 18-core CPU, 16-core ANE, 64 GB unified memory, 307 GB/s, 24-48 MB system-level cache (to be measured in M0).

Model: `thinkingmachines/Inkling-2-Mini`, Apache-2.0. Hidden 2304, 24 physical layers, 24 query / 8 KV heads, head_dim 128. Layers 6-17 are a loop core run three times per token, so a token makes 48 layer visits. MoE on every layer: 128 experts, top-8, no shared expert, intermediate 1152. Local sliding-window attention (1024) everywhere except layers 5, 11, 17 and 23. Learned relative position logits instead of RoPE, an input-dependent attention sink, an output gate, short convolutions (k=4) and attention-residual blocks of 6. Vocabulary 201,024. Bundled drafter: 3 dense layers proposing 8 tokens from hidden-state taps at visits 5, 23, 41 and 48, with a GRU Markov head.

What ships in the file (1,244 tensors, 19,525,870,752 data bytes):

| Part | Format | Size |
|---|---|---|
| Experts, layers 0-19 | NVFP4 (E2M1, group-16 E4M3 scales) | 4.479 MB per expert |
| Experts, layers 20-23 | MXFP8 (group-32 E8M0 scales) | 8.211 MB per expert |
| Attention projections | BF16 | 53.8 MB per layer |
| Router | BF16 | 0.59 MB per layer |
| Loop controllers (rank 64) | BF16 | 4.72 MB per core layer |
| `embed`, `unembed` (untied) | BF16 | 926 MB each |
| `mtp.safetensors` (drafter) | BF16 | 636 MB |

The surprise: only the experts are small. Attention and the output head are 16-bit, and the loop reads the core layers three times.

## 2. Speed of light

Bytes read per token at batch 1, 2K context, shipped weights:

| Component | MB/token |
|---|---|
| Attention, 48 visits | 2,583 |
| Router and loop controllers | 198 |
| Experts, top-8 (44 NVFP4 + 4 MXFP8 visits) | 1,839 |
| Output head | 926 |
| KV cache read | 235 |
| Total | ~5,780 |

| Weight variant | GB/token | Ceiling at 307 GB/s | At 260 GB/s sustained | Exact vs shipped? |
|---|---|---|---|---|
| A. Shipped, full head | 5.79 | 53 | 45 | yes |
| B. A + certified two-stage head | 5.12 | 60 | 51 | yes |
| B2. B + lossless repack of BF16 | 4.26 | 72 | 61 | yes |
| C. FP8 attention and KV | 3.63 | 85 | 72 | no, opt-in |
| D. Lean: NVFP4 attention and last-4 experts, FP8 KV | 2.94 | 104 | 88 | no, opt-in |

Caches do not help the loop: between pass p and pass p+1 of a layer, 0.45-1.1 GB streams through a cache of tens of MB. The only reuse is inside one layer visit, where one weight tile serves every verify row.

Compute is not the limit. A token costs about 10.8 GFLOP; the GPU has tens of TFLOPS. Past batch-1 decoding, every lever is about making one memory read serve more tokens.

## 3. Architecture

- C++20 with metal-cpp, one binary: `run`, `serve`, `bench`, `inspect`, `dump`. Metal 4 shaders compiled offline with fast-math off and explicit `fma()`.
- The whole decode step is one recorded Metal 4 graph: argument tables, residency sets, indirect dispatch, about 4 fused dispatches per layer visit, several steps per command buffer. Positions, expert lists, accept counts and drafts live in GPU buffers; the CPU never sits in the token loop.
- No device-wide spin megakernel by default. Apple GPUs give no forward-progress guarantee across threadgroups and the display shares the GPU. A persistent kernel is an M2 experiment behind a flag, kept only if barrier gaps cost more than 5% of a step.
- Batch-invariant numerics: every output sums over K in a fixed chunk order with fp32 accumulation, whatever the row count. No float atomics. MoE combine in router-rank order. This is what lets speculative verification stay token-identical.
- One matmul family with 16-row tiles: SIMD lanes unpack NVFP4/MXFP8 into threadgroup memory and the Neural Accelerators run `matmul2d` (FP16 in, FP32 accumulate), if M0 proves it is row-invariant; otherwise `simdgroup_matrix`. Dequantised weights are never written back to memory.
- Experts repacked once into aligned per-expert blobs, `[w13 gate/up interleaved | w2]`, codes next to scales, memory-mapped and pinned.
- Attention fuses attention-residual mixing, norm, short convolution and one Q|gate|K|V|rel|sink projection, then window/global attention with relative bias computed in threadgroup memory, then output gate and O projection. 48 per-visit KV caches (1024-token rings for local, paged for global).
- Certified output head: an INT4 table scores all live rows, then every row that could still beat the best exact score is rescored in BF16. Same answer as the full head, about 250 MB instead of 926. Halo's 1-bit head stays as a turbo option.
- Sampling: Halo's exact speculative sampling with a counter RNG, `hash(seed, position)`.
- ANE: none for text decoding (shares the same bandwidth, no gather, static graphs). Candidate for the vision tower later.

## 4. Speculation

- Drafter: the bundled 8-token block drafter, quantised for speed only (draft quality never affects exactness). Its GRU head reranks a batched coarse top-64 per position instead of running 8 full-vocabulary heads.
- Lookup drafts: suffix matching over prompt and output for copies, edits and tool calls, with longer chains when the match is long.
- Chain, not tree. Each extra verify row pulls in about g new experts per visit (g = 2-3 is the planning guess; nobody has measured it), so a row costs 1.7-2.5 ms and the best chain is about 3-6 tokens. Adaptive length: `k* = argmax (E[accepted] + 1) / cost(k)` from a measured cost table.

Round-1 projections (Opus), tok/s at g=2 / g=3; byte model of section 2 at 260 GB/s, 2K context, batch 1, drafter 1.2 ms and 0.6 ms overhead per step; projections, not measurements:

| Weights | prose | chat | code |
|---|---|---|---|
| B, shipped + certified head | 88 / 80 | 108 / 95 | 144 / 121 |
| B2, lossless repack | 101 / 91 | 124 / 108 | 162 / 136 |
| D, lean | 130 / 115 | 155 / 133 | 197 / 163 |

Copy-heavy edits with lookup drafts: about 250 tok/s on lean weights.

## 5. Correctness

- R0: the official Hugging Face code with weights dequantised exactly to fp32, run on a 128 GB machine.
- R1 (oracle): an independent fp32 PyTorch forward that keeps experts packed (about 24 GB, runs on the Mac CPU) and dumps every layer visit. R1 must match R0 at cosine >= 0.99999 per visit and give identical greedy output for 256 tokens on 20 prompts.
- E1, exact (the headline): token-identical to Delphi's own batch-1 non-speculative greedy on the same weights, for every draft length and any co-batched users; 0/200 diffs across repeated runs.
- E2, faithful (against R1): shipped weights top-1 agreement >= 99.8% and KL <= 0.002; lean weights KL <= 0.02 and HumanEval within 2 points.

## 6. Milestones

Gates are measurements on the M5 Pro through `delphi bench` (batch 1, 2K context unless the gate names another), each logged in `ledger.jsonl`.

| | Proves | Gate |
|---|---|---|
| M0 | Probes, oracle, offline economics | GPU streaming >= 260 GB/s; `matmul2d` row-invariance settled; R1 matches R0; g, acceptance and head candidate counts measured from oracle traces |
| M1 | Correct batch-1 engine on shipped weights | The 11 e2e tests below; >= 40 tok/s |
| M2 | Batch-1 speed | >= 55 tok/s exact; >= 85% of streamable bandwidth |
| M3 | Exact speculation | E1 on 40 prompts x 512 tokens x k=1..8; >= 100 chat, >= 130 code |
| M4 | Lean weights and opt-in modes | Lean exact >= 150 chat, >= 190 code; turbo >= 200, kept off inside code fences and tool calls |
| M5 | Server and agents | OpenAI-compatible API with tools and prefix cache; prefill >= 1,500 tok/s; 16 agents >= 500 tok/s total |

M1 e2e tests, each written first and seen failing, driven through the real CLI:

1. `inspect` reports 1,244 tensors, 19,525,870,752 bytes, 20 NVFP4 and 4 MXFP8 layers.
2. 64 sampled expert rows dequantise bit-identical to R1.
3. All 48 visits and the final output match R1 at cosine >= 0.9999.
4. 128 greedy tokens on 20 golden prompts match R1, except at R1 near-ties (top-2 margin < 0.01).
5. Teacher-forced on 8K tokens: top-1 >= 99.8%, KL <= 0.002.
6. A 1,100-token prompt matches across the 1,024 window edge.
7. A 9K-token prompt matches past the log-scaling floor at 8,192.
8. Prefill in chunks of 1, 7 and 512 gives bit-identical logits.
9. 50 repeated runs give identical output.
10. Generation stops at EOS 200006 and the chat template matches Hugging Face.
11. `bench` shows >= 40 tok/s at batch 1 on shipped weights.

## 7. Risks and how to kill them early

1. High g makes speculation pay little: replay oracle routing traces in M0, before any kernel.
2. Neural Accelerator results change with row count, or the compiler fuses FMAs differently: bit-probe in M0, fall back to SIMD with explicit `fma()`.
3. A misread model detail (attention residuals, loop controllers, dynamic global convolution, sink, relative logits past 1024): R0/R1 per-visit agreement before any Metal code.
4. Sustained bandwidth under 230 GB/s or barrier gaps over 3 us: replay a synthetic step with the real byte pattern and about 200 dispatches.
5. Laptop heat, display sharing the GPU, memory pressure: a 30-minute sustained run with the display on, on battery and on AC; resident memory <= 30 GB.

## 8. Decisions

- Name: Delphi.
- License: Apache-2.0 with a NOTICE file and CITATION.cff. Two of three members picked it: section 4(d) makes every distributed copy or derivative include the attribution notices, it adds a patent grant, it matches the model's license, and companies adopt it. Sol preferred AGPL-3.0 with a section 7(b) attribution term, which also reaches hosted services but costs adoption.
- Exact mode is the headline. Lossy modes are opt-in and labelled.
- Tests are end-to-end only, written first. CI pins every action by commit SHA and every job has a 20-minute budget.

## 9. Moonshot (council round 2)

Round 2 (2026-11-21) asked the same three members to aim far past 100 tok/s, using the [inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) (ceiling first, one change at a time, interleaved controls, a ledger, long-context gates). Every number is arithmetic until the ledger measures it.

### Star targets

| Workload | Sol floor / star | Opus floor / star | Neon floor / star | Delphi target |
|---|---|---|---|---|
| Exact chat | 80 / 200 | 101 / 181 | 157 / 225 | **225** |
| Exact code | 105 / 230 | 129 / 250 | 203 / 267 | **270** |
| Exact copy-edit and tool calls | 160 / 380 | 174 / 385 | 423 / 680 | **680** |
| Opt-in turbo chat | 140 / 350 | 190 / 365 | 236 / 470 | **470** |
| 16 agents, total | 135 / 380 | 385 / 501 | 405 / 707 | **700** |
| Prefill, 4K prompt | 1,000 / 2,800 | 1,150 / 1,640 | 1,830 / 4,400 | **4,400** |
| Confidence in own exact-chat star | 20% | 30% | 55% | |

All rows: projections from each member's byte model (bytes per step over 225-275 GB/s, times committed tokens per step), M5 Pro, batch 1 per stream, 2K context, not measurements. Prefill rows are compute projections at an assumed 10-35 TFLOPS. The floors are the milestone gates; the stars are what Delphi aims for. Exact means token-identical to Delphi's own non-speculative batch-1 greedy on the shipped weights.

### What gets there (all three agree)

| Lever | Exact? | Bytes before -> after | Kill test |
|---|---|---|---|
| Grouped verify: one read of each weight tile serves every verify row, fixed per-row reduction order | yes | dense 7 x 2.78 GB -> 2.78 GB per step | state drift at any row count |
| Cheap drafter: 4-bit body, pruned 8K-32K draft vocabulary, Markov head reranks a batched shortlist | yes | 1.45 GB -> 0.15-0.20 GB per block | acceptance drops more than 0.25 tokens |
| Certified two-stage head: INT4 (or Hadamard INT3) coarse scan, BF16 rerank of every row that could still win | yes | 926 -> 190-340 MB | one invalid bound, or p99 survivors over 20K rows |
| Lossless BF16 packing, decoded in threadgroup memory: measured entropy 10.57 bits per weight | yes | dense 2.73 -> 1.86-2.0 GB | under 1.25x a BF16 GEMV |
| Lossless MXFP8 and NVFP4-scale packing (NVFP4 nibbles are 3.97/4 bits: leave them) | yes | experts -130 to -350 MB per step | under 7% saved |
| Route-aware drafting: pick chains and sibling branches by accepted tokens per new expert byte | yes | g 2.5-3 -> 1.2-2 new experts per extra row | net gain under 5% |
| Lookup drafts from a CPU suffix automaton, width grown 8 -> 16 -> 32 only while it keeps paying | yes | copy-edit 1.23 GB -> 0.40-0.65 GB per token | measured union at 32 rows over 70 experts |
| Decode-style verify attention: each KV tile read once for all rows | yes | 1.64 -> 0.29 GB at 2K, 7 rows | step time grows per row with context |
| Bubble-free streaming: a low-priority dispatch warms the next visit's weights into the system cache across barriers | yes | bandwidth efficiency 0.80 -> 0.92 | chain bandwidth already over 0.9 of copy |
| Recorded Metal 4 graph with 3-4 fat dispatches per visit, not a spin megakernel | yes | dispatch overhead 5 ms -> 0.4 ms per step | graph gaps under 0.5 ms make it moot |
| Turbo weights: NVFP4 attention, NVFP4 last-4 experts, FP8 KV | no | 4.2 -> 2.9 GB per token | quality gate |
| Turbo routing: adaptive top-6/7, union-capped verify routing | no | experts -19 to -22% | quality gate |
| Lenient acceptance, chat only, off inside code, grammar and tool calls | no | +0.8-1.2 accepted tokens | repeated-line and loop count |
| Pass skip after loop pass 2 | no, high risk | -0.53 to -1.2 GB | top-1 agreement under 97% (`is_varloop` is false) |

### Rejected, with the maths

- Loop-as-drafter: one drafted token reads 1.3-1.8 GB, about 0.4 of a full forward, against 0.03 for a trained drafter.
- Pipelining across loop passes or staggering users across passes: the same core weights per token, more prelude and coda reads.
- ANE or CPU drafting for bandwidth: same DRAM, no extra GB/s, at most 1 ms of overlap per step.
- Low-rank attention: exact compression cannot beat the 10.6-bit entropy floor; the lossy version loses to NVFP4.
- Entropy-coding NVFP4 nibbles: 3.97 of 4 bits are already entropy.
- Device-wide spin megakernel: no forward-progress guarantee on Apple GPUs, saves at most about 1 ms per step.

### The two numbers that decide everything

- **g**: new experts each extra verify row pulls in per layer visit. Random routing gives 7.5; the plan needs 1.2-2.
- **alpha**: chance each drafted token is accepted. The plan needs 0.85 on chat.

Week one measures both from the fp32 oracle before any kernel is written.

### Week one, in order

1. Hardware truth: copy bandwidth, a chain of 192 dependent GEMVs with and without the cache pump, `matmul2d` FP16/INT8 at 4096^3 and 8192^3, row-invariance for M = 1..32, barrier cost. Re-anchor every ceiling if copy bandwidth is under 250 GB/s.
2. Route overlap and acceptance from the oracle: g for chains, sibling branches and lookup rows (M = 1..32); acceptance by depth for the bundled drafter at BF16 and 4-bit, full and pruned vocabulary. g >= 5 caps chains at 3 tokens and the exact-chat star falls to about 120.
3. Certified head: survivor counts and 100% greedy agreement on 2,000 real hidden states.
4. Fused lossless BF16 GEMV against BF16, bit-exact.
5. End-to-end Metal 4 verify step at M = 1..8, 2K and 32K, against the oracle.

### Ledger rules (from the skill, adapted to this Mac)

- Append-only `ledger.jsonl`: commit, config hash, lever, exact or turbo, workload bucket, context, k, acceptance by depth, step time split into attention / experts / head / KV / draft, GB/s per kernel class, union per visit, g, median of at least 6 runs, both neighbouring controls, power state, output-token hash, verdict and reason. `PLAN.md` and `THEORY.md` from day 0; negative results are kept.
- One change per config, derived from `best-config.json`; control, candidate, control. Accept only if the candidate beats both controls and no bucket (chat, code, tool, copy) drops more than 3%. Repeat any surprising win.
- The bench refuses to run off AC power, in low-power mode, with a foreign GPU process or with thermal pressure, and voids the run if any of that changed by the end. Nothing that raises a macOS permission prompt. Report minute 10 of a 10-minute soak, not the cold burst.
- Exact gate: token-identical to non-speculative batch 1 on the whole panel, plus bitwise invariance across verify widths {1, 2, 4, 8, 16, 32} and prefill chunks {1, 64, 512, 4096}.
- Long-context gate: three needles at 50% and 100% of 32K and 128K, found exactly, with tok/s at 32K.
- Tokens counted from engine counters, never streamed chunks. One GPU job at a time. Every number reported as a percentage of its ceiling.
- The skill's `ceiling.py` multiplies aggregate throughput by concurrency twice and never checks the compute limit; Delphi's ceiling script fixes both.

## 10. Bespoke 12-token drafter

The bundled drafter proposes at most 7 tokens per block. Delphi trains its own: a DSpine drafter (parallel drafting with adjacent causal injection through depth, [arXiv 2609.36173](https://arxiv.org/abs/2609.36173)) that proposes 12 tokens per block, trained on a Colab G4 (RTX PRO 6000 Blackwell, 96 GB). Drafter quality never affects exactness; it only changes how many tokens each verify step commits.

| Choice | Value | Why |
|---|---|---|
| Block | 12 proposals plus the anchor | the length asked for; at union-overlap g=2 the 12th token still pays on code and copy |
| Body | 5 DSpine layers, hidden 2304 | DSpine's depth; hidden matches the target so taps need no projection |
| Warm start | layers from the bundled `mtp.safetensors` drafter | it already reads Inkling's hidden states |
| Taps | layer visits 5, 17, 29, 41, 47: end of the prelude, end of each loop pass, last visit | one tap per stage of the loop |
| Draft vocabulary | the smallest shortlist covering at least 99% of the target's greedy tokens on the training corpus (expected 16K-32K) | the head is the drafter's biggest read |
| Mask token | an unused padding id (>= 200058) | no collision with real tokens |
| Data | prompts from chat, code, tool-call and agent traces, answered by Inkling itself (self-distillation, greedy and T=0.7) | the drafter must imitate this target, not a dataset |
| Teacher signal | taps and top-k logits from the target with BF16 activations (W4A16), matching Delphi's numerics | the Blackwell FP4-activation path differs slightly from Delphi's |
| Trainer | SpecForge DSpine objective, online, with SGLang (Inkling support, PR #51926) serving the target | SpecForge's merged DSpine objective |
| Export | 4-bit body, INT4 pruned head, INT8 Markov or transition table, loaded by Delphi's Metal drafter kernel | about 0.15-0.2 GB per block |

Gate: mean accepted length on a held-out chat / code / tool panel beats the bundled drafter at the same verify cost, measured in the same replay harness, before any Metal drafter kernel is written.