import argparse
import json
import sys
from pathlib import Path


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="to_specforge")
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--expect-model", required=True)
    return parser.parse_args(argv)


def read_records(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def usable(records):
    seen = set()
    for record in records:
        if not record["completion"] or record["id"] in seen:
            continue
        seen.add(record["id"])
        yield {"id": record["id"], "text": record["prompt"] + record["completion"]}


def main(argv):
    args = parse_args(argv)
    records = read_records(args.records)
    foreign = [r["id"] for r in records if r["model_path"] != args.expect_model]
    if foreign:
        models = sorted({r["model_path"] for r in records if r["model_path"] != args.expect_model})
        print(f"to_specforge: records {foreign[:20]} come from {models}, expected {args.expect_model}", file=sys.stderr)
        return 2
    kept = list(usable(records))
    with open(args.out, "w", encoding="utf-8") as out:
        for line in kept:
            out.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"to_specforge: kept {len(kept)} of {len(records)} records")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))