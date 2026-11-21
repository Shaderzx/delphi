import argparse
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment


class GenerationBlock(Extension):
    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.Scope(body, lineno=lineno)


class TemplateError(Exception):
    pass


def raise_exception(message):
    raise TemplateError(message)


def tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def load_template(path):
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[GenerationBlock, "jinja2.ext.loopcontrols"])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env.from_string(Path(path).read_text(encoding="utf-8"))


def render(template, prompt):
    variables = {"messages": prompt["messages"], "tools": prompt.get("tools")}
    if prompt.get("reasoning_effort") is not None:
        variables["reasoning_effort"] = prompt["reasoning_effort"]
    return template.render(**variables)


def http_json(url, body=None, timeout=900):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def served_model(server):
    return http_json(f"{server}/model_info", timeout=30).get("model_path")


def read_prompts(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finished_ids(path):
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["id"] for line in handle if line.strip()}


def generate(server, prompt_text, sampling):
    response = http_json(f"{server}/generate", {"text": prompt_text, "sampling_params": sampling})
    meta = response["meta_info"]
    return {
        "completion": response["text"],
        "output_ids": response["output_ids"],
        "finish_reason": meta["finish_reason"]["type"],
        "completion_tokens": meta["completion_tokens"],
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="selfdistill")
    parser.add_argument("--server", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--expect-model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def refuse(message):
    print(f"selfdistill: {message}", file=sys.stderr)
    return 2


def main(argv):
    args = parse_args(argv)
    server = args.server.rstrip("/")
    model = served_model(server)
    if model != args.expect_model:
        return refuse(f"server serves {model}, expected {args.expect_model}")
    if args.out.exists() and not args.resume:
        return refuse(f"{args.out} exists; pass --resume to continue it")

    template = load_template(args.template)
    done = finished_ids(args.out)
    pending = [p for p in read_prompts(args.prompts) if p["id"] not in done]
    sampling = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "skip_special_tokens": False}
    lock = threading.Lock()
    failures = 0

    with open(args.out, "a", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}
        for prompt in pending:
            text = render(template, prompt)
            futures[pool.submit(generate, server, text, sampling)] = (prompt["id"], text)
        for written, future in enumerate(as_completed(futures), start=1):
            prompt_id, text = futures[future]
            try:
                result = future.result()
            except Exception as error:
                failures += 1
                print(f"selfdistill: {prompt_id} failed: {error}", file=sys.stderr)
                continue
            record = {"id": prompt_id, "prompt": text, **result, "model_path": model}
            with lock:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
            if written % 100 == 0:
                print(f"{written}/{len(pending)}", flush=True)

    print(f"selfdistill: wrote {len(pending) - failures} of {len(pending)} pending, {len(done)} already done", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))