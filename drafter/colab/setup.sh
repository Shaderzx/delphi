set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" SGLANG_BUILD_RUST_EXTS=none HF_HUB_ENABLE_HF_TRANSFER=1
SGLANG_COMMIT=b736e624fec767d239a49b581578fe2da45154dc
SPECFORGE_COMMIT=232c7fd76065b012e8bf6f6a26ed3a89686c3fbf

installed_sglang_commit() {
  python3 -c "import importlib.metadata as m, json; print(json.loads(m.distribution('sglang').read_text('direct_url.json') or '{}').get('vcs_info', {}).get('commit_id', ''))" 2>/dev/null || true
}

test -x "$HOME/.cargo/bin/cargo" || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.91.1 -q
if ! python3 -c 'import sglang' 2>/dev/null; then
  uv pip install --system "sglang[all] @ git+https://github.com/sgl-project/sglang.git@$SGLANG_COMMIT#subdirectory=python"
fi
found="$(installed_sglang_commit)"
if [ "$found" != "$SGLANG_COMMIT" ]; then
  echo "setup: sglang is at '${found:-unknown}', expected $SGLANG_COMMIT; uninstall it and rerun" >&2
  exit 1
fi

test -d /content/SpecForge || git clone -q https://github.com/sgl-project/SpecForge /content/SpecForge
git -C /content/SpecForge checkout -q "$SPECFORGE_COMMIT"
if ! python3 -c 'import specforge' 2>/dev/null; then
  uv pip install --system wheel setuptools setuptools-rust setuptools-scm
  uv pip install --system --no-build-isolation --prerelease=allow -e /content/SpecForge
fi
python3 -c 'import specforge_sglang_capture' 2>/dev/null || uv pip install --system --no-deps /content/SpecForge/plugins/sglang-spec-capture
{ python3 -c 'import mooncake.store' 2>/dev/null && command -v mooncake_master >/dev/null; } || uv pip install --system mooncake-transfer-engine-cuda13

test -f /content/inkling/model-00001-of-00001.safetensors || hf download thinkingmachines/Inkling-2-Mini --local-dir /content/inkling
test -d /content/delphi || git clone -q https://github.com/Shaderzx/delphi /content/delphi
uv pip install --system -q --require-hashes -r /content/delphi/drafter/requirements.txt
python3 /content/delphi/drafter/colab/patch_upstream.py