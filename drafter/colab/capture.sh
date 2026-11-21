set -euo pipefail
pgrep -x mooncake_master >/dev/null || (nohup mooncake_master --rpc_port 35551 --enable_http_metadata_server --http_metadata_server_port 35880 --eviction_high_watermark_ratio 0.9 > /content/mooncake.log 2>&1 &)
export SPECFORGE_SPEC_CAPTURE=1 MOONCAKE_PROTOCOL=tcp MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 MOONCAKE_GLOBAL_SEGMENT_SIZE=8589934592
export MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:35551 MOONCAKE_METADATA_SERVER=http://127.0.0.1:35880/metadata
exec python3 -m sglang.launch_server --model-path /content/inkling --port 30000 --host 127.0.0.1 --trust-remote-code \
  --mem-fraction-static 0.45 --context-length 4608 \
  --speculative-algorithm DSPARK --enable-return-aux-hidden-states --return-aux-hidden-state-layer-ids 6 17 29 41 48 \
  --return-hidden-states-mode full --chunked-prefill-size -1