#!/usr/bin/env python3
"""
Download, convert, and verify ANNex benchmark datasets.

All ann-benchmarks datasets share a common HDF5 layout:
    train → base.npy  (float32)
    test  → queries.npy  (float32)
    neighbors → ground_truth.json  (list[list[int]])

Usage:
    python3 scripts/fetch_dataset.py nytimes-256-angular sift-128-euclidean
    python3 scripts/fetch_dataset.py --all
    python3 scripts/fetch_dataset.py --verify-only
    python3 scripts/fetch_dataset.py sift-128-euclidean --record

    --record     Write computed checksums back to data/registry.json (run once after first fetch).
    --verify-only  Recheck checksums of already-downloaded data without downloading anything.

Requires: numpy, h5py  (h5py only needed for conversion; not required for --verify-only)
"""
import argparse, hashlib, json, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "data" / "registry.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def download_hdf5(url: str, dest: Path) -> None:
    print(f"  ↓ {url}")
    subprocess.run(["curl", "-L", "--progress-bar", "-o", str(dest), url], check=True)
    print(f"  ✓ {dest.stat().st_size // (1 << 20)} MB")


def convert_hdf5(hdf5_path: Path, out_dir: Path) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        sys.exit(f"Conversion requires numpy and h5py: {e}")

    print(f"  Converting {hdf5_path.name}…")
    with h5py.File(hdf5_path) as f:
        base = f["train"][:].astype("float32")
        queries = f["test"][:].astype("float32")
        neighbors = f["neighbors"][:].tolist()

    np.save(out_dir / "base.npy", base)
    np.save(out_dir / "queries.npy", queries)
    with open(out_dir / "ground_truth.json", "w") as g:
        json.dump([[int(x) for x in row] for row in neighbors], g)

    print(f"  base={base.shape}  queries={queries.shape}  "
          f"ground_truth={len(neighbors)}×{len(neighbors[0])}")


def fetch_one(dataset_id: str, entry: dict, data_dir: Path,
              record: bool, verify_only: bool) -> bool:
    out = data_dir / dataset_id
    out.mkdir(parents=True, exist_ok=True)

    canonical_files = ["base.npy", "queries.npy", "ground_truth.json"]
    missing = [f for f in canonical_files if not (out / f).exists()]

    if verify_only:
        if missing:
            print(f"  ✗ missing files: {missing}")
            return False
        print(f"  files present — verifying checksums…")
    elif missing:
        with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            download_hdf5(entry["source_url"], tmp_path)
            convert_hdf5(tmp_path, out)
        finally:
            tmp_path.unlink(missing_ok=True)
    else:
        print(f"  files already present — verifying checksums…")

    # Compute checksums
    computed = {f: sha256(out / f) for f in canonical_files}

    # Verify against registry or note unrecorded
    expected = entry.get("checksums", {})
    all_match = True
    for fname, cs in computed.items():
        exp = expected.get(fname)
        if exp is None:
            print(f"  {fname}: {cs}  ← not yet recorded in registry")
        elif cs == exp:
            print(f"  {fname}: ✓")
        else:
            print(f"  {fname}: ✗ MISMATCH")
            print(f"    expected: {exp}")
            print(f"    got:      {cs}")
            all_match = False

    if not all_match:
        return False

    # Write dataset.json (local record, git-ignored)
    try:
        import numpy as np
        base_shape = tuple(int(x) for x in np.load(out / "base.npy", mmap_mode="r").shape)
        queries_shape = tuple(int(x) for x in np.load(out / "queries.npy", mmap_mode="r").shape)
    except ImportError:
        base_shape = queries_shape = None

    manifest = {
        "id": dataset_id,
        "name": entry["name"],
        "dims": entry["dims"],
        "metric": entry["metric"],
        "n_base": base_shape[0] if base_shape else entry["n_base"],
        "n_queries": queries_shape[0] if queries_shape else entry["n_queries"],
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checksums": computed,
    }
    (out / "dataset.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  → wrote {out / 'dataset.json'}")

    # --record: update registry.json checksums in-place
    if record:
        registry = json.loads(REGISTRY_PATH.read_text())
        registry[dataset_id]["checksums"] = computed
        REGISTRY_PATH.write_text(json.dumps(registry, indent=2) + "\n")
        print(f"  → updated registry.json with checksums for {dataset_id}")

    return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("names", nargs="*", help="dataset IDs (from registry.json)")
    ap.add_argument("--all", action="store_true", help="fetch all registered datasets")
    ap.add_argument("--data-dir", default=None, type=Path,
                    help="root data directory (default: data/ relative to repo root)")
    ap.add_argument("--record", action="store_true",
                    help="write computed checksums back to registry.json")
    ap.add_argument("--verify-only", action="store_true",
                    help="verify checksums without downloading")
    args = ap.parse_args()

    if not REGISTRY_PATH.exists():
        sys.exit(f"Registry not found: {REGISTRY_PATH}")

    registry = json.loads(REGISTRY_PATH.read_text())
    data_dir = args.data_dir or REPO_ROOT / "data"

    names = list(registry.keys()) if args.all else args.names
    if not names:
        ap.print_help()
        print(f"\nRegistered datasets:")
        for k, v in registry.items():
            cs = v.get("checksums", {})
            recorded = all(cs.get(f) for f in ["base.npy", "queries.npy", "ground_truth.json"])
            status = "✓ checksums recorded" if recorded else "· checksums not yet recorded"
            print(f"  {k:<30}  {v['dims']}D {v['metric']:<12} {v['n_base']:>9,} base  {status}")
        sys.exit(0)

    unknown = [n for n in names if n not in registry]
    if unknown:
        sys.exit(f"Unknown dataset(s): {unknown}\nRegistered: {list(registry.keys())}")

    failed = []
    for name in names:
        print(f"\n{'─'*60}")
        print(f"{name}  ({registry[name]['name']})")
        ok = fetch_one(name, registry[name], data_dir, args.record, args.verify_only)
        if not ok:
            failed.append(name)

    print(f"\n{'─'*60}")
    if failed:
        sys.exit(f"✗ Failed: {failed}")
    print("✓ All done.")


if __name__ == "__main__":
    main()
