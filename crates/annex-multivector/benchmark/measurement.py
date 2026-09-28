"""Durable query journal and deterministic summaries; no model dependencies."""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from contextlib import contextmanager
from pathlib import Path


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def evaluate(qrels, run, k=10):
    rows = []
    for qid, relevant in qrels.items():
        ranked = run.get(qid, [])[:k]
        # BEIR/trec_eval ndcg_cut uses the qrel grade itself as gain.
        gains = [max(0, relevant.get(doc, 0)) for doc in ranked]
        dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
        ideal = sorted((max(0, grade) for grade in relevant.values()), reverse=True)[:k]
        idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal))
        wanted = {doc for doc, gain in relevant.items() if gain > 0}
        rows.append(
            {
                "qid": qid,
                f"ndcg@{k}": dcg / idcg if idcg else 0.0,
                f"recall@{k}": len(wanted.intersection(ranked)) / len(wanted)
                if wanted
                else 0.0,
                f"mrr@{k}": next(
                    (1 / (i + 1) for i, doc in enumerate(ranked) if doc in wanted), 0.0
                ),
                **{
                    f"recall@{depth}": len(
                        wanted.intersection(run.get(qid, [])[:depth])
                    )
                    / len(wanted)
                    if wanted
                    else 0.0
                    for depth in (20, 100)
                },
            }
        )
    if not rows:
        raise ValueError("cannot evaluate an empty query set")
    result = {
        metric: statistics.fmean(row[metric] for row in rows)
        for metric in rows[0]
        if metric != "qid"
    }
    result["per_query"] = rows
    return result


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open("x", encoding="utf-8")

    def close(self):
        self.file.close()

    def write(self, event, system, **fields):
        self.file.write(
            json.dumps(
                {"event": event, "system": system, "at_unix_s": time.time(), **fields},
                allow_nan=False,
            )
            + "\n"
        )
        self.file.flush()
        os.fsync(self.file.fileno())

    @contextmanager
    def stage(self, system, name):
        self.write("stage_started", system, stage=name)
        started = time.perf_counter()
        try:
            yield
        finally:
            self.write(
                "stage_finished",
                system,
                stage=name,
                elapsed_s=time.perf_counter() - started,
            )

    def query(self, system, qid, requested_backend, call):
        self.write(
            "query_started", system, qid=qid, requested_backend=requested_backend
        )
        started = time.perf_counter()
        row = {
            "qid": qid,
            "requested_backend": requested_backend,
            "executed_backend": None,
            "ranked_ids": [],
            "scores": [],
            "error": None,
        }
        try:
            result = call()
            matches = result["matches"]
            ids = [str(hit["id"]) for hit in matches]
            scores = [float(hit["score"]) for hit in matches]
            backend = result["backend"]
            if (
                not backend
                or len(set(ids)) != len(ids)
                or not all(map(math.isfinite, scores))
            ):
                raise ValueError("invalid ranking or missing executed backend")
            row.update(
                status="ok", ranked_ids=ids, scores=scores, executed_backend=backend
            )
        except Exception as error:
            row.update(status="error", error=f"{type(error).__name__}: {error}")
        row["latency_ms"] = (time.perf_counter() - started) * 1000
        self.write("query_finished", system, **row)


def read_events(path):
    # A killed process may leave only the last line torn. Earlier corruption fails.
    with Path(path).open("rb") as source:
        for line in source:
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def summarize(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    systems = {}
    for event in read_events(directory / "events.jsonl"):
        system = systems.setdefault(
            event["system"], {"rows": {}, "stages": {}, "status": "interrupted"}
        )
        kind = event["event"]
        if kind == "query_started":
            system["rows"][event["qid"]] = {
                "qid": event["qid"],
                "status": "interrupted",
                "requested_backend": event["requested_backend"],
                "executed_backend": None,
                "ranked_ids": [],
                "scores": [],
                "latency_ms": None,
                "error": "no completed response",
            }
        elif kind == "query_finished":
            system["rows"][event["qid"]] = {
                k: v
                for k, v in event.items()
                if k not in ("event", "system", "at_unix_s")
            }
        elif kind == "stage_finished":
            system["stages"][event["stage"]] = event["elapsed_s"]
        elif kind == "system_finished":
            system.update(status=event["status"], error=event.get("error"))
        elif kind == "index_ready":
            system["index"] = event["index"]
    output = {}
    preparation = systems.pop("_preparation", {}).get("stages", {})
    for name, system in systems.items():
        rows = system.pop("rows")
        metrics = evaluate(
            manifest["qrels"], {qid: row["ranked_ids"] for qid, row in rows.items()}
        )
        for row in metrics["per_query"]:
            row.update(
                rows.get(
                    row["qid"],
                    {
                        "status": "not_attempted",
                        "ranked_ids": [],
                        "scores": [],
                        "latency_ms": None,
                        "error": "query was not attempted",
                    },
                )
            )
        latencies = [
            r["latency_ms"] for r in rows.values() if r["latency_ms"] is not None
        ]
        output[name] = {
            **system,
            **metrics,
            "queries": len(manifest["qrels"]),
            "attempted_queries": len(rows),
            "failed_queries": sum(r["status"] != "ok" for r in metrics["per_query"]),
            "build_s": sum(system["stages"].values()),
            **{f"p{p}_ms": percentile(latencies, p) for p in (50, 95, 99)},
        }
    return {
        "dataset": manifest["dataset"],
        "documents": manifest["documents"],
        "queries": len(manifest["qrels"]),
        "protocol": manifest["protocol"],
        "systems": output,
        "preparation": preparation,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rebuild a summary from a complete or interrupted run"
    )
    parser.add_argument("directory", type=Path)
    print(
        json.dumps(summarize(parser.parse_args().directory), indent=2, allow_nan=False)
    )
