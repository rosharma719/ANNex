#!/usr/bin/env python3
"""Generate a clustered synthetic ANN dataset in the layout the harnesses read.

    python3 scripts/make_synthetic_dataset.py synth-128-angular --n 200000 --dim 128

Writes data/<name>/{base.npy,queries.npy,ground_truth.json}. Vectors are drawn
from a mixture of Gaussians around random unit centres and L2-normalised, which
gives HNSW real neighbourhood structure (unlike i.i.d. noise) without needing
network access. Exact top-100 ground truth is computed with a blocked matmul.
Synthetic recall says nothing about real embeddings; use it to compare kernels
and libraries on identical inputs.
"""
import argparse, json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--clusters", type=int, default=256)
    ap.add_argument("--spread", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    centers = rng.standard_normal((a.clusters, a.dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    def draw(n):
        c = rng.integers(0, a.clusters, n)
        x = centers[c] + a.spread * rng.standard_normal((n, a.dim)).astype(np.float32) / np.sqrt(a.dim)
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)

    base, queries = draw(a.n), draw(a.queries)
    truth = []
    for i in range(0, a.queries, 100):
        sims = queries[i : i + 100] @ base.T
        top = np.argpartition(-sims, 100, axis=1)[:, :100]
        order = np.take_along_axis(sims, top, axis=1).argsort(axis=1)[:, ::-1]
        truth.extend(np.take_along_axis(top, order, axis=1).tolist())

    out = Path(__file__).resolve().parent.parent / "data" / a.name
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "base.npy", base)
    np.save(out / "queries.npy", queries)
    (out / "ground_truth.json").write_text(json.dumps(truth))
    print(f"{out}: base {base.shape}, queries {queries.shape}")


if __name__ == "__main__":
    main()
