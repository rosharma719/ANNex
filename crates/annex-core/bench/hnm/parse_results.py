#!/usr/bin/env python3
"""
Parse hnm_recall_from_snapshot stdout into JSONL.

Reads log lines from stdin, writes one JSON object per ef_search value to stdout.
Called by run_annexdb.sh:
    cargo test ... --nocapture | python3 bench/hnm/parse_results.py > results_annexdb.jsonl
"""
import re, json, sys

recall_pat = re.compile(
    r"\[recall_stats\] ef_search=(\d+) queries=(\d+) mean=([\d.]+)"
    r" p50=([\d.]+) p90=([\d.]+) p99=([\d.]+)"
)
latency_pat = re.compile(
    r"\[query_stats\] ef_search=(\d+) ms\(p50/p90/p99\)=([\d.]+)/([\d.]+)/([\d.]+)"
)

rows: dict[int, dict] = {}
for line in sys.stdin:
    m = recall_pat.search(line)
    if m:
        ef = int(m.group(1))
        rows.setdefault(ef, {}).update({
            "ef": ef,
            "queries": int(m.group(2)),
            "recall": float(m.group(3)),
            "recall_p50": float(m.group(4)),
            "recall_p90": float(m.group(5)),
            "recall_p99": float(m.group(6)),
        })
        continue
    m = latency_pat.search(line)
    if m:
        ef = int(m.group(1))
        rows.setdefault(ef, {}).update({
            "ef": ef,
            "p50_ms": float(m.group(2)),
            "p90_ms": float(m.group(3)),
            "p99_ms": float(m.group(4)),
        })

for ef in sorted(rows):
    row = rows[ef]
    if "recall" in row and "p50_ms" in row:
        row["qps"] = round(1000 / row["p50_ms"], 1) if row.get("p50_ms") else 0
        print(json.dumps(row))
