#!/usr/bin/env python3
"""Single-query hnswlib comparison on the same NYT vectors and truth as ANNex.

Install numpy and hnswlib in an isolated environment, then run this script.
Build and timing must not overlap other workloads. Use --build-only to separate
construction from measurement. This is a reference implementation comparison,
not an equal-graph comparison: hnswlib uses its standard M0=2*M storage cap.
"""
import argparse
import importlib.metadata
import json
from pathlib import Path
from time import perf_counter

import hnswlib
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/nytimes-256-angular"))
    parser.add_argument("--index", type=Path, default=Path("data/nytimes-256-angular/hnswlib-m16-efc300.bin"))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--efs", default="32,64,128,256,512")
    parser.add_argument("--offsets", default="0,1000")
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    assert args.queries > 0 and args.rounds > 0
    index = hnswlib.Index(space="cosine", dim=256)
    if args.index.exists():
        index.load_index(str(args.index))
    else:
        base = np.load(args.data / "base.npy")
        index.init_index(max_elements=len(base), ef_construction=300, M=16, random_seed=42)
        start = perf_counter()
        index.add_items(base, np.arange(len(base)), num_threads=4)
        args.index.parent.mkdir(parents=True, exist_ok=True)
        index.save_index(str(args.index))
        print(json.dumps({"build_seconds": perf_counter() - start, "version": importlib.metadata.version("hnswlib")}))
    if args.build_only:
        return
    queries = np.load(args.data / "queries.npy")
    truth = np.array(json.loads((args.data / "ground_truth.json").read_text()))
    for offset in map(int, args.offsets.split(",")):
        assert offset >= 0 and offset + args.queries <= len(queries)
        selected = queries[offset:offset + args.queries]
        for ef in map(int, args.efs.split(",")):
            index.set_ef(ef)
            # Untimed warm-up and recall, one query per call, one search thread.
            answers = [index.knn_query(q, k=20, num_threads=1)[0][0] for q in selected]
            recall = sum(len(set(a) & set(t[:20])) for a, t in zip(answers, truth[offset:offset + args.queries])) / (args.queries * 20)
            for round_id in range(args.rounds):
                times = []
                start = perf_counter()
                for query in selected:
                    query_start = perf_counter()
                    index.knn_query(query, k=20, num_threads=1)
                    times.append((perf_counter() - query_start) * 1000)
                elapsed = perf_counter() - start
                print(json.dumps({"query_offset": offset, "ef": ef, "round": round_id,
                                  "recall": recall, "qps": len(selected) / elapsed,
                                  "mean_ms": elapsed * 1000 / len(selected),
                                  "queries": len(selected), "p50_ms": float(np.percentile(times, 50)),
                                  "p99_ms": float(np.percentile(times, 99))}), flush=True)


if __name__ == "__main__":
    main()
