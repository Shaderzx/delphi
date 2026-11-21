set -euo pipefail
until curl -sf http://127.0.0.1:30000/health >/dev/null; do sleep 10; done
cd /content
exec specforge train -c /content/delphi/drafter/specforge/train-g4.yaml "$@"