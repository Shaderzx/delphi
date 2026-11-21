# Delphi plan

Council round 1 (2026-11-21): Claude Opus 5.5 (xhigh), GPT-6.2 Sol (xhigh), Gemini 4 Neon (xhigh). Two members read the real safetensors headers; the numbers below come from those headers, not from parameter counts. Every tok/s figure here is a ceiling or a target until a ledger row measures it.

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

Round-1 projections (Opus), tok/s at g=2 / g=3:

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
- License: Apache-2.0 with a NOTICE file and CITATION.cff. Two of three members picked it: section 4(d) makes every redistribution and derivative keep the credit, it adds a patent grant, it matches the model's license, and companies adopt it. Sol preferred AGPL-3.0 with a section 7(b) attribution term, which also reaches hosted services but costs adoption.
- Exact mode is the headline. Lossy modes are opt-in and labelled.
- Tests are end-to-end only, written first. CI pins every action by commit SHA and every job has a 20-minute budget.