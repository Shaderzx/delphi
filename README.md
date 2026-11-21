# Delphi

Delphi is an inference engine for one model on one machine: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on an Apple M5 Pro (20-core GPU, 64 GB, 307 GB/s). It is the Mac sibling of Halo, which runs Qwen3.6-35B-A3B at 249 tok/s exact on AMD Strix Halo.

Status: planning. No engine code yet. Read [docs/PLAN.md](docs/PLAN.md) for the speed math, the architecture and the milestones.

## Where it starts

| | Decode tok/s, M5 Pro |
|---|---|
| MLX, stock | 24-25 |
| MLX, tuned | 26.6 |
| llama.cpp, UD-Q4_K_XL | 28.4 |
| Delphi exact, shipped weights, no speculation (ceiling) | 53 |
| Delphi exact with speculation (milestone M3 gate) | 100 chat, 130 code |
| Delphi moonshot targets, exact | 225 chat, 270 code, 680 copy-edit |
| Delphi moonshot target, opt-in turbo | 470 chat |

Exact means token-identical to Delphi's own plain greedy decoding on the shipped weights. Faster modes that change the output are opt-in and labelled.

## Using Delphi in your project

Delphi is licensed under [Apache-2.0](LICENSE). If you ship Delphi or anything built from it, keep the [NOTICE](NOTICE) file with it, as section 4(d) of the license requires. To cite it, use [CITATION.cff](CITATION.cff).