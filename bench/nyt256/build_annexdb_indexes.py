#!/usr/bin/env python3
"""
Build ANNex indexes for M=16, M=24, M=32 using the Rust binary.
Reads base.npy, writes a snapshot for each M value.

Run from repo root:
    python3 bench/nyt256/build_annexdb_indexes.py
"""
import subprocess, sys, os, json
from pathlib import Path

DATA = Path("data/nytimes-256-angular")
M_VALUES = [16, 24, 32]
EF_CONSTRUCT = 300

for M in M_VALUES:
    snap = DATA / f"annexdb_m{M}_efc{EF_CONSTRUCT}.bin"
    if snap.exists():
        print(f"M={M}: snapshot already exists at {snap}, skipping")
        continue
    print(f"M={M}: building index (ef_construct={EF_CONSTRUCT})...")
    env = os.environ.copy()
    env["VECTORDB_NYT_PERSIST_PATH"] = str(snap)
    env["VECTORDB_M"] = str(M)
    env["VECTORDB_EF_CONSTRUCT"] = str(EF_CONSTRUCT)
    result = subprocess.run(
        ["cargo", "test", "--release", "--test", "nytimes_frontier",
         "nytimes_build_pareto_index", "--", "--ignored"],
        env=env, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  FAILED: {result.stderr[-500:]}")
        sys.exit(1)
    print(f"  Done: {snap}")

print("\nAll indexes ready.")
