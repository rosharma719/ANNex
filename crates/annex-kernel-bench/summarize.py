#!/usr/bin/env python3
"""Summarise criterion output from the distance and maxsim benches.

    python3 summarize.py results/distance-*.txt [--per N]   # ns per score
    python3 summarize.py results/maxsim-*.txt               # µs per score

`--per N` divides each time by the number of scores in one iteration (the
working-set size: 256 by default, or ANNEX_BENCH_SET).
"""
import argparse, collections, re

UNIT = {"ns": 1.0, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}
ROW = re.compile(
    r"^(?P<name>[\w.+\-]+/[\w.+\-]+/[\w.+\-/]+?)\s*(?:time:\s+\[\S+ \S+ (?P<mid>\S+) (?P<unit>\S+))?",
    re.M,
)


def parse(path):
    out, pending = {}, None
    for line in open(path, encoding="utf-8"):
        line = line.rstrip()
        m = re.match(r"^(\S+/\S+/\S+)\s*$", line) or re.match(r"^(\S+/\S+/\S+)\s+time:", line)
        if m:
            pending = m.group(1)
        t = re.search(r"time:\s+\[\S+ \S+ (\S+) (\S+) \S+ \S+\]", line) or re.search(
            r"time:\s+\[\S+ \S+ (\S+) (\S+)", line
        )
        if t and pending:
            out[pending] = float(t.group(1)) * UNIT[t.group(2)]
            pending = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--per", type=int, default=1, help="scores per iteration")
    ap.add_argument("--unit", default="ns", choices=["ns", "us"])
    a = ap.parse_args()
    for path in a.files:
        rows = parse(path)
        groups = collections.defaultdict(lambda: collections.defaultdict(dict))
        for name, ns in rows.items():
            group, lib, shape = name.split("/", 2)
            groups[group][shape][lib] = ns / a.per / UNIT[a.unit]
        for group, shapes in groups.items():
            libs = sorted({l for s in shapes.values() for l in s}, key=lambda l: (l != "annex", l))
            print(f"\n{group} ({a.unit}/score) - {path}")
            print(f"{'shape':>14} " + " ".join(f"{l:>15}" for l in libs))
            for shape in sorted(shapes, key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)]):
                print(f"{shape:>14} " + " ".join(
                    f"{shapes[shape][l]:15.2f}" if l in shapes[shape] else f"{'-':>15}" for l in libs))


if __name__ == "__main__":
    main()
