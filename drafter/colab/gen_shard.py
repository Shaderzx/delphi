import json, os, random, signal, subprocess, sys, time, urllib.request
from pathlib import Path

ROLE = {"human": "user", "user": "user", "gpt": "assistant", "assistant": "assistant", "system": "system"}
EFFORTS = ["low", "medium", "high"]


def healthy():
    try:
        return urllib.request.urlopen("http://127.0.0.1:30000/health", timeout=3).status == 200
    except Exception:
        return False


def ensure_dataset():
    subprocess.run(["bash", "-lc", "test -f /content/data/perfectblend/done || (mkdir -p /content/data/perfectblend && HF_HUB_ENABLE_HF_TRANSFER=1 hf download mlabonne/open-perfectblend --repo-type dataset --local-dir /content/data/perfectblend && touch /content/data/perfectblend/done)"], check=True)


def to_messages(conversation):
    messages = []
    for turn in conversation:
        role = ROLE.get(turn.get("from") or turn.get("role"))
        value = turn.get("value") if "value" in turn else turn.get("content")
        if role is None or not isinstance(value, str):
            return []
        messages.append({"role": role, "content": value})
    while messages and messages[-1]["role"] != "user":
        messages.pop()
    return messages


def shard_prompts(start, end):
    import pyarrow.parquet as pq
    rows = []
    for f in sorted(Path("/content/data/perfectblend").rglob("*.parquet")):
        rows.extend(pq.read_table(f).to_pylist())
    prompts = []
    for i, row in enumerate(rows):
        messages = to_messages(row.get("conversations") or row.get("messages") or [])
        if messages:
            prompts.append({"id": f"pb-{i}", "messages": messages, "reasoning_effort": EFFORTS[i % 3]})
    del rows
    random.Random(20261121).shuffle(prompts)
    shard = prompts[start:end]
    del prompts
    return shard


def write_prompts(shard, path):
    with open(path, "w", encoding="utf-8") as f:
        for p in shard:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


def start_server():
    server = subprocess.Popen(["python3", "-m", "sglang.launch_server", "--model-path", "/content/inkling", "--port", "30000", "--host", "127.0.0.1",
        "--trust-remote-code", "--mem-fraction-static", "0.85", "--context-length", "16384", "--max-running-requests", "256"],
        stdout=open("/content/gen-server.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
    while not healthy():
        if server.poll() is not None:
            raise SystemExit("server exited, see /content/gen-server.log")
        time.sleep(10)
    return server


def stop_server(server):
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def main(start, end, out):
    ensure_dataset()
    shard = shard_prompts(start, end)
    prompts_file = f"/content/data/prompts-{start}-{end}.jsonl"
    write_prompts(shard, prompts_file)
    print("prompts", len(shard), "first id", shard[0]["id"], flush=True)
    del shard
    server = start_server()
    try:
        args = ["python3", "-I", "/content/delphi/drafter/selfdistill.py", "--server", "http://127.0.0.1:30000", "--prompts", prompts_file, "--out", out,
            "--template", "/content/delphi/drafter/templates/inkling-2-mini.jinja", "--expect-model", "/content/inkling",
            "--max-new-tokens", "4096", "--temperature", "0", "--concurrency", "256"] + (["--resume"] if os.path.exists(out) else [])
        return subprocess.run(args).returncode
    finally:
        stop_server(server)


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]))