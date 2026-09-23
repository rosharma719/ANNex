#!/usr/bin/env python3
"""Read the append-only benchmark ledger for a Semantic Version."""
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=(
            "run", "sweep", "diagnostic", "uncompressed-oracle", "encoder-audit",
            "dense-baseline",
            "score-validation",
        ),
    )
    args = parser.parse_args()
    records = [json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()]
    if args.kind:
        records = [record for record in records if record["kind"] == args.kind]
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
