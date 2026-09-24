#!/usr/bin/env python3
"""Run the reproducible FiQA/NFCorpus candidate-generation test matrix."""
import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dataset", default="beir/fiqa/test")
    parser.add_argument("--limit-docs", type=int)
    parser.add_argument("--limit-queries", type=int)
    parser.add_argument("--sampling", choices=["prefix", "qrels"], default="prefix")
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--centroids", type=int, default=256)
    parser.add_argument("--report-dir", type=Path, default=Path("benchmark/reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("benchmark/cache"))
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--candidates", type=int, nargs="+", default=[50, 100, 250, 500])
    parser.add_argument("--ef-search", type=int, nargs="+", default=[32, 64, 128, 256])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    common = ["--index", str(args.index), "--dataset", args.dataset, "--sampling", args.sampling, "--sample-seed", str(args.sample_seed), "--centroids", str(args.centroids), "--candidate-grid", *map(str, args.candidates), "--report-dir", str(args.report_dir), "--cache-dir", str(args.cache_dir)]
    if args.refresh_cache:
        common.append("--refresh-cache")
    if args.limit_docs is not None:
        common += ["--limit-docs", str(args.limit_docs)]
    if args.limit_queries is not None:
        common += ["--limit-queries", str(args.limit_queries)]
    subprocess.run([sys.executable, "benchmark/sweep.py", *common, "--backend", "muvera"], cwd=root, check=True)
    subprocess.run([sys.executable, "benchmark/sweep.py", *common, "--backend", "hnsw", "--ef-search-grid", *map(str, args.ef_search)], cwd=root, check=True)


if __name__ == "__main__":
    main()
