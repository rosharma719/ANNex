#!/usr/bin/env python3
"""
Aggregate frontier lab JSONL outputs into CSVs and a Pareto plot.

Usage:
    python3 scripts/analyze_frontier_lab.py FILE [FILE ...] --out DIR [--thresholds 0.80,0.85,...]

Inputs: one or more JSONL files from the frontier_lab binary or benchmark_hnswlib_reference.py.
  - ANNex records: {"type":"measurement", "label":..., "ef":..., "offset":..., "round":..., ...}
  - hnswlib records: {"ef":..., "query_offset":..., "round":..., ...}  (label inferred from filename)

Outputs written to DIR/:
    all_points.csv        — one row per (label, ef, offset), rounds aggregated via median
    pareto.csv            — per-offset Pareto-frontier rows (not dominated on recall x p50_ms)
    threshold_winners.csv — fastest config per (recall threshold x offset)
    frontier.png          — recall vs p50_ms scatter with Pareto curves (requires matplotlib)
"""
import argparse, csv, json, statistics, sys
from collections import defaultdict
from pathlib import Path

FIELDS = ["label", "ef", "offset", "rounds", "recall", "p50_ms", "p99_ms", "mean_ms", "scored", "expanded"]
DEFAULT_THRESHOLDS = "0.80,0.85,0.87,0.90,0.92,0.94,0.95,0.96,0.97"


def load(path: Path) -> list[dict]:
    """Load and normalize measurement records from a JSONL file."""
    label_hint = "hnswlib" if "hnswlib" in path.stem else None
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        obj = json.loads(line)
        if obj.get("type") not in ("measurement", None):
            continue
        if "ef" not in obj:
            continue
        obj.setdefault("label", label_hint or "unknown")
        # hnswlib files use query_offset; ANNex files use offset
        if "offset" not in obj:
            obj["offset"] = obj.pop("query_offset", 0)
        obj.setdefault("scored", None)
        obj.setdefault("expanded", None)
        records.append(obj)
    return records


def aggregate(records: list[dict]) -> list[dict]:
    """Group by (label, ef, offset) and reduce rounds via median of each metric."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["label"], r["ef"], r["offset"])].append(r)

    rows = []
    for (label, ef, offset), recs in sorted(groups.items()):
        def med(field):
            vals = [r[field] for r in recs if r.get(field) is not None]
            return statistics.median(vals) if vals else None

        rows.append({
            "label": label, "ef": ef, "offset": offset, "rounds": len(recs),
            "recall": med("recall"),
            "p50_ms": med("p50_ms"),
            "p99_ms": med("p99_ms"),
            "mean_ms": med("mean_ms"),
            "scored": med("scored"),
            "expanded": med("expanded"),
        })
    return rows


def pareto_front(rows: list[dict]) -> list[dict]:
    """Rows not dominated on recall (higher) x p50_ms (lower)."""
    front = []
    for r in sorted(rows, key=lambda x: (-x["recall"], x["p50_ms"])):
        if not front or r["p50_ms"] < front[-1]["p50_ms"]:
            front.append(r)
    return front


def threshold_winners(rows: list[dict], thresholds: list[float]) -> list[dict]:
    """For each (threshold, offset), fastest row with recall >= threshold."""
    by_offset: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_offset[r["offset"]].append(r)

    winners = []
    for thr in thresholds:
        for offset, group in sorted(by_offset.items()):
            candidates = sorted(
                [r for r in group if r["recall"] >= thr],
                key=lambda x: x["p50_ms"],
            )
            if candidates:
                w = {"threshold": thr, **candidates[0]}
                winners.append(w)
    return winners


def write_csv(rows: list[dict], fields: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def plot(rows: list[dict], out: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available; skipping frontier.png)")
        return

    offsets = sorted(set(r["offset"] for r in rows))
    fig, axes = plt.subplots(1, len(offsets), figsize=(6 * len(offsets), 5), squeeze=False)

    for ax, offset in zip(axes[0], offsets):
        group = [r for r in rows if r["offset"] == offset]
        for label in sorted(set(r["label"] for r in group)):
            pts = sorted([r for r in group if r["label"] == label], key=lambda r: r["recall"])
            ax.scatter([r["recall"] for r in pts], [r["p50_ms"] for r in pts], label=label, s=20)
        front = sorted(pareto_front(group), key=lambda r: r["recall"])
        ax.plot([r["recall"] for r in front], [r["p50_ms"] for r in front],
                "k--", linewidth=1, label="Pareto")
        ax.set_xlabel("recall@20")
        ax.set_ylabel("p50 ms")
        ax.set_title(f"NYT-256  offset={offset}")
        ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  {out.name}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", type=Path, help="JSONL files from frontier_lab")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--thresholds", default=DEFAULT_THRESHOLDS,
                    help="comma-separated recall thresholds (default: %(default)s)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    thresholds = [float(t) for t in args.thresholds.split(",")]

    all_records: list[dict] = []
    for p in args.inputs:
        recs = load(p)
        print(f"  {p.name}: {len(recs)} measurement records")
        all_records.extend(recs)

    if not all_records:
        sys.exit("No measurement records found in input files.")

    points = aggregate(all_records)

    write_csv(points, FIELDS, args.out / "all_points.csv")
    print(f"all_points.csv: {len(points)} rows")

    pareto_rows: list[dict] = []
    for offset in sorted(set(r["offset"] for r in points)):
        pareto_rows.extend(pareto_front([r for r in points if r["offset"] == offset]))
    write_csv(pareto_rows, FIELDS, args.out / "pareto.csv")
    print(f"pareto.csv:     {len(pareto_rows)} rows")

    winners = threshold_winners(points, thresholds)
    write_csv(winners, ["threshold"] + FIELDS, args.out / "threshold_winners.csv")
    print(f"threshold_winners.csv: {len(winners)} rows")

    plot(points, args.out / "frontier.png")


if __name__ == "__main__":
    main()
