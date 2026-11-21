# Training the DSpine-12 drafter on a Colab G4

Everything here runs on a Colab G4 session (RTX PRO 6000 Blackwell, 96 GB) driven from a laptop with `google-colab-cli`.

Drive the session only through `colab exec -s <name> -f <file>`, `colab upload -s <name>` and `colab download -s <name>`. In `google-colab-cli` 0.8.0, `colab ssh -s <name>` connected to a different session than the one named.

A session whose kernel sits idle is reclaimed, and a background process does not count as activity. So the kernel runs `runner.py` in the foreground for the whole life of the session (`colab exec -s <name> --timeout 0 -f drafter/colab/runner.py`), and work is handed to it as shell scripts uploaded into `/content/queue/todo/`. The runner starts each script, logs it to `/content/<script>.log`, and writes one line per event to `/content/progress.txt`, which `colab download` can read while the kernel is busy.

## Steps

1. `setup.sh`: Rust (an SGLang build dependency has no wheel for Python 3.13), SGLang at the Inkling support commit, SpecForge, its SGLang capture plugin, Mooncake, the model, this repo, then `patch_upstream.py`.
2. `gen_shard.py <start> <end> <out>`: builds the shuffled open-perfectblend prompt shard and runs `drafter/selfdistill.py` against a plain SGLang server.
3. `drafter/to_specforge.py`: records to SpecForge's preformatted rows at `/content/data/train.jsonl`.
4. `capture.sh`: Mooncake master plus an SGLang capture server tapping layer visits 6, 17, 29, 41 and 48.
5. `train.sh`: `specforge train` with `drafter/specforge/train-g4.yaml`; extra `key=value` arguments override the config.

## Upstream patches

`patch_upstream.py` changes four things in the installed packages and fails loudly if an anchor moved:

- SGLang's capture-only check rejected tap 48, which Inkling's own capture hook requires as the last tap (the post-norm state).
- The capture plugin imported `DisaggregationMode` from its old module, and deferred responses before SGLang released them, which this SGLang refuses; it now defers from the release hook.
- SpecForge validated taps against the 24 physical layers; a loop target taps its 48 layer visits.
- SpecForge refused Inkling Mini's final logit softcap (30.0). It is monotone, so greedy drafting is unaffected, and the drafter trains on uncapped logits.