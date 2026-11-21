import json, os, random, subprocess, sys, time, urllib.request
from pathlib import Path

shard_start, shard_end, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

def healthy():
    try:
        return urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=3).status == 200
    except Exception:
        return False

subprocess.run(["bash", "-lc", "test -f /content/data/perfectblend/done || (mkdir -p /content/data/perfectblend && HF_HUB_ENABLE_HF_TRANSFER=1 hf download mlabonne/open-perfectblend --repo-type dataset --local-dir /content/data/perfectblend && touch /content/data/perfectblend/done)"], check=True)
sys.path.insert(0, "/content/delphi/drafter")
import pyarrow.parquet as pq
rows = []
for f in sorted(Path("/content/data/perfectblend").rglob("*.parquet")):
    rows.extend(pq.read_table(f).to_pylist())
role = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
prompts = []
for i, row in enumerate(rows):
    conv = row.get("conversations") or row.get("messages") or []
    msgs = []
    for turn in conv:
        r = role.get(turn.get("from") or turn.get("role"))
        v = turn.get("value") if "value" in turn else turn.get("content")
        if r is None or not isinstance(v, str):
            msgs = []
            break
        msgs.append({"role": r, "content": v})
    while msgs and msgs[-1]["role"] != "user":
        msgs.pop()
    if msgs:
        prompts.append({"id": f"pb-{i}", "messages": msgs, "reasoning_effort": ["low", "medium", "high"][i % 3]})
random.Random(20261121).shuffle(prompts)
shard = prompts[shard_start:shard_end]
pfile = f"/content/data/prompts-{shard_start}-{shard_end}.jsonl"
with open(pfile, "w") as f:
    for p in shard:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print("prompts", len(shard), "first id", shard[0]["id"], flush=True)

server = subprocess.Popen(["python3", "-m", "sglang.launch_server", "--model-path", "/content/inkling", "--port", "30000", "--host", "127.0.0.1",
    "--trust-remote-code", "--mem-fraction-static", "0.85", "--context-length", "16384", "--max-running-requests", "256"],
    stdout=open("/content/gen-server.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
while not healthy():
    if server.poll() is not None:
        raise SystemExit("server exited")
    time.sleep(10)
args = ["python3", "-I", "/content/delphi/drafter/selfdistill.py", "--server", "http://127.0.0.1:30000", "--prompts", pfile, "--out", out,
    "--template", "/content/delphi/drafter/templates/inkling-2-mini.jinja", "--expect-model", "/content/inkling",
    "--max-new-tokens", "4096", "--temperature", "0", "--concurrency", "256"] + (["--resume"] if os.path.exists(out) else [])
sys.exit(subprocess.run(args).returncode)