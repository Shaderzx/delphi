set -euo pipefail
export PATH="$HOME/.cargo/bin:$PATH" SGLANG_BUILD_RUST_EXTS=none HF_HUB_ENABLE_HF_TRANSFER=1
test -x "$HOME/.cargo/bin/cargo" || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain 1.91.1 -q
python3 -c 'import sglang' 2>/dev/null || uv pip install --system "sglang[all] @ git+https://github.com/sgl-project/sglang.git@b736e624fec767d239a49b581578fe2da45154dc#subdirectory=python"
test -d /content/SpecForge || git clone -q https://github.com/sgl-project/SpecForge /content/SpecForge
git -C /content/SpecForge checkout -q 232c7fd76065b012e8bf6f6a26ed3a89686c3fbf
python3 -c 'import specforge' 2>/dev/null || {
  uv pip install --system wheel setuptools setuptools-rust setuptools-scm
  uv pip install --system --no-build-isolation --prerelease=allow -e /content/SpecForge
  uv pip install --system --no-deps /content/SpecForge/plugins/sglang-spec-capture
  uv pip install --system mooncake-transfer-engine-cuda13
}
test -f /content/inkling/model-00001-of-00001.safetensors || hf download thinkingmachines/Inkling-2-Mini --local-dir /content/inkling
test -d /content/delphi || git clone -q https://github.com/Shaderzx/delphi /content/delphi
uv pip install --system -q --require-hashes -r /content/delphi/drafter/requirements.txt
python3 /content/delphi/drafter/colab/patch_upstream.py