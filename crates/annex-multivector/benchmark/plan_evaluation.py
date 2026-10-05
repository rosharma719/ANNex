"""Offline execution of a declared plan grid against a quiescent ANNex service."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse

from measurement import Journal, evaluate, percentile, read_events

SELECTED = "__selected__"


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def render(template, inputs):
    if isinstance(template, dict):
        if set(template) == {"$input"}:
            return copy.deepcopy(inputs[template["$input"]])
        return {key: render(value, inputs) for key, value in template.items()}
    if isinstance(template, list):
        return [render(value, inputs) for value in template]
    return template


def validate(spec):
    if spec.get("version") != 1:
        raise ValueError("unsupported evaluation spec version")
    for name in ("corpus", "configuration", "models"):
        if (
            not isinstance(spec.get("fingerprints", {}).get(name), str)
            or not spec["fingerprints"][name]
        ):
            raise ValueError(f"missing declared {name} fingerprint")
    if set(spec["selection"]) != {"request"}:
        raise ValueError("selection must contain only a request template")
    queries, plans = spec["queries"], spec["plans"]
    if not queries or not plans:
        raise ValueError("queries and plans must be nonempty")
    for rows, name in ((queries, "query"), (plans, "plan")):
        ids = [row["id"] for row in rows]
        if any(not isinstance(i, str) or not i or i == SELECTED for i in ids) or len(
            set(ids)
        ) != len(ids):
            raise ValueError(f"invalid or duplicate {name} IDs")
    for query in queries:
        if not query["qrels"] or any(
            not isinstance(doc, str)
            or not isinstance(grade, (int, float))
            or not math.isfinite(grade)
            or grade < 0
            for doc, grade in query["qrels"].items()
        ):
            raise ValueError("qrels require document IDs and finite nonnegative grades")
        selection = render(spec["selection"]["request"], query["inputs"])
        for plan in [*plans, spec["selection"]]:
            request = render(plan["request"], query["inputs"])
            if not isinstance(request, dict):
                raise ValueError("request templates must render objects")
            if request.get("filter") != selection.get("filter"):
                raise ValueError(
                    "all plans and selection must use the same query filter"
                )
            if (
                type(request.get("limit", 10)) is not int
                or request.get("limit", 10) < spec["top_k"]
            ):
                raise ValueError("result limit must be at least evaluation top_k")
    for name in ("top_k", "repeats"):
        if type(spec[name]) is not int or spec[name] <= 0:
            raise ValueError(f"{name} must be positive")
    budgets = spec["budgets_ms"]
    if not budgets or any(
        type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in budgets
    ):
        raise ValueError("budgets must be finite and positive")
    for name in ("epsilon_quality", "epsilon_latency_ms"):
        value = spec.get(name, 0)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    fingerprint(spec)  # Also reject nonfinite values anywhere in templates/inputs.


def channels(plan):
    return [
        channel
        for stage in plan["stages"]
        if stage["kind"] == "parallel"
        for channel in stage["detail"]
    ]


def execution_signature(plan):
    """Exclude estimates/reasons/calibration: retain executed semantics and budgets."""
    stages = copy.deepcopy(plan["stages"])
    for stage in stages:
        if stage["kind"] == "parallel":
            stage["detail"] = [
                {
                    key: row.get(key)
                    for key in ("index", "operator", "limit", "ef_search")
                }
                for row in stage["detail"]
            ]
    return {
        "stages": stages,
        "filter": plan["filter"],
        "eligible_documents": plan["eligible_documents"],
        "generation": plan["stats"]["generation"],
    }


def pin_request(request, plan):
    """Replay the initial decision explicitly so observations cannot reroute it."""
    request = copy.deepcopy(request)
    policy = plan.get("policy")
    if policy:
        request["prefetch"] = copy.deepcopy(policy["generated_prefetch"])
    request["planning_mode"] = "manual"
    request.pop("query", None)
    # Budget rejection was already evaluated by /plan. Replay the chosen work;
    # later cost calibration must not turn it into a different feasible decision.
    request.setdefault("objective", {}).pop("latency_budget_ms", None)
    prefetch = request["prefetch"]
    for row in channels(plan):
        channel = prefetch[row["index"]]
        channel["limit"] = row["limit"]
        operator = row["operator"]
        if channel["kind"] in ("dense", "multivector"):
            channel["backend"] = "hnsw" if operator.startswith("hnsw_") else "exact"
            if row.get("ef_search") is not None:
                channel["ef_search"] = row["ef_search"]
    return request


def json_safe(value):
    """Keep malformed numeric responses journalable without emitting invalid JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_float": repr(value)}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def http(base, route, body, timeout, token):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = None if body is None else json.dumps(body, allow_nan=False).encode()
    try:
        with urllib.request.urlopen(
            urllib.request.Request(base.rstrip("/") + route, data, headers),
            timeout=timeout,
        ) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        with error:
            detail = error.read().decode(errors="replace")
        raise RuntimeError(f"{route}: HTTP {error.code}: {detail}") from error


def run(spec, directory, base, *, timeout=60, token=None, call=None):
    validate(spec)
    parsed = urlparse(base)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.netloc
        or parsed.username
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "base URL requires HTTP(S) without credentials, query or fragment"
        )
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    call = call or (lambda route, body: http(base, route, body, timeout, token))
    manifest = {
        "spec": spec,
        "spec_sha256": fingerprint(spec),
        "base_url": base,
        "platform": platform.platform(),
        "python": sys.version,
        "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_unix_s": time.time(),
        "timeout_s": timeout,
        "counter_scope": None,
        "snapshot_guard": "document generation and service stats; graph/calibration files are caller-declared",
        "latency_statistic": "median of completed HTTP calls per query/plan",
        "selection_protocol": "all decisions compiled before retrieval, then pinned",
    }
    # Persist inputs before contacting the server, including when startup fails.
    with (directory / "manifest.json").open("x") as file:
        json.dump(manifest, file, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
    journal = Journal(directory / "observations.jsonl")
    try:
        stats = call("/v1/stats", None)
        journal.write("snapshot", "annex", stats=stats, stats_sha256=fingerprint(stats))
        frozen = {}
        # /plan does not observe execution or modify calibration. No retrieval
        # occurs until every grid and policy decision has been captured.
        for query in spec["queries"]:
            for definition in [*spec["plans"], {"id": SELECTED, **spec["selection"]}]:
                pid = definition["id"]
                request = render(definition["request"], query["inputs"])
                journal.write("plan_started", pid, qid=query["id"], request=request)
                started = time.perf_counter()
                try:
                    plan = call("/v1/plan", request)
                    if plan["stats"]["generation"] != stats["generation"]:
                        raise ValueError("generation changed during plan compilation")
                    pinned = pin_request(request, plan)
                    frozen[query["id"], pid] = (pinned, plan)
                    journal.write(
                        "plan_finished",
                        pid,
                        qid=query["id"],
                        status="ok",
                        planning_ms=(time.perf_counter() - started) * 1000,
                        plan=plan,
                        pinned_request=pinned,
                    )
                except Exception as error:
                    journal.write(
                        "plan_finished",
                        pid,
                        qid=query["id"],
                        status="error",
                        planning_ms=(time.perf_counter() - started) * 1000,
                        error=f"{type(error).__name__}: {error}",
                    )
        for repeat in range(spec["repeats"]):
            # Rotate plan order to avoid always favoring the same warmed backend.
            definitions = [*spec["plans"], {"id": SELECTED}]
            shift = repeat % len(definitions)
            definitions = definitions[shift:] + definitions[:shift]
            for query in spec["queries"]:
                for definition in definitions:
                    pid = definition["id"]
                    if (query["id"], pid) not in frozen:
                        continue
                    request, expected = frozen[query["id"], pid]
                    journal.write("query_started", pid, qid=query["id"], repeat=repeat)
                    row = {
                        "qid": query["id"],
                        "repeat": repeat,
                        "ranked_ids": [],
                        "scores": [],
                        "status": "error",
                        "quality": 0.0,
                        "spec_sha256": manifest["spec_sha256"],
                        "plan_sha256": fingerprint(expected),
                        "request_sha256": fingerprint(request),
                        "resource_metrics": {
                            "cpu_work": None,
                            "memory_bytes": None,
                            "disk_bytes": None,
                        },
                    }
                    started = time.perf_counter()
                    try:
                        response = call("/v1/retrieve", request)
                        row["latency_ms"] = (time.perf_counter() - started) * 1000
                        row["response"] = json_safe(
                            response
                        )  # Preserve malformed execution too.
                        actual = response["trace"]["plan"]
                        if response["trace"]["generation"] != stats["generation"]:
                            raise ValueError("retrieval generation changed")
                        if execution_signature(actual) != execution_signature(expected):
                            raise ValueError(
                                "executed physical plan differs from frozen plan"
                            )
                        backends = [c["backend"] for c in response["trace"]["channels"]]
                        if backends != [c["operator"] for c in channels(actual)]:
                            raise ValueError(
                                "trace backends disagree with physical operators"
                            )
                        ids = [str(hit["id"]) for hit in response["matches"]]
                        scores = [float(hit["score"]) for hit in response["matches"]]
                        if len(ids) != len(set(ids)) or not all(
                            map(math.isfinite, scores)
                        ):
                            raise ValueError("invalid ranking")
                        metrics = evaluate(
                            {query["id"]: query["qrels"]},
                            {query["id"]: ids},
                            spec["top_k"],
                        )
                        row.update(
                            status="ok",
                            ranked_ids=ids,
                            scores=scores,
                            quality=metrics[f"ndcg@{spec['top_k']}"],
                            metrics={
                                k: v for k, v in metrics.items() if k != "per_query"
                            },
                        )
                    except Exception as error:
                        row["error"] = f"{type(error).__name__}: {error}"
                        row.setdefault(
                            "latency_ms", (time.perf_counter() - started) * 1000
                        )
                    journal.write("query_finished", pid, **row)
        final_stats = call("/v1/stats", None)
        journal.write(
            "run_finished",
            "annex",
            snapshot_stable=final_stats == stats,
            final_stats=final_stats,
        )
    except Exception as error:
        journal.write("run_failed", "annex", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        journal.close()
        report = summarize(directory)
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n"
        )
    return report


def dominates(a, b, quality_epsilon=0, latency_epsilon=0):
    return (
        a["quality"] >= b["quality"] + quality_epsilon
        and a["latency_ms"] <= b["latency_ms"] - latency_epsilon
        and (a["quality"] > b["quality"] or a["latency_ms"] < b["latency_ms"])
    )


def summarize(directory):
    directory = Path(directory)
    spec = json.loads((directory / "manifest.json").read_text())["spec"]
    validate(spec)
    planned, attempts, selected_planning = {}, {}, []
    stable, finished = False, False
    for event in read_events(directory / "observations.jsonl"):
        key = (event.get("qid"), event["system"])
        if event["event"] == "plan_finished":
            planned[key] = event
            if event["system"] == SELECTED:
                selected_planning.append(event["planning_ms"])
        elif event["event"] == "query_started":
            attempts[key + (event["repeat"],)] = {
                "status": "interrupted",
                "quality": 0,
                "latency_ms": None,
                "ranked_ids": [],
            }
        elif event["event"] == "query_finished":
            attempts[key + (event["repeat"],)] = event
        elif event["event"] == "run_finished":
            stable, finished = event["snapshot_stable"], True
    rows, evaluations = [], []
    for query in spec["queries"]:
        points = []
        for pid in [p["id"] for p in spec["plans"]] + [SELECTED]:
            outcomes = [
                attempts.get(
                    (query["id"], pid, repeat),
                    {
                        "status": "not_attempted",
                        "quality": 0,
                        "latency_ms": None,
                        "ranked_ids": [],
                    },
                )
                for repeat in range(spec["repeats"])
            ]
            ok = [r for r in outcomes if r["status"] == "ok"]
            point = {
                "qid": query["id"],
                "plan_id": pid,
                "completed": len(ok),
                "failed_or_missing": len(outcomes) - len(ok),
                # Missing/failed repetitions stay in the quality denominator.
                "quality": statistics.fmean(r["quality"] for r in outcomes),
                "latency_ms": (
                    statistics.median(r["latency_ms"] for r in ok) if ok else None
                ),
                "ranking_stable": len(ok) == len(outcomes)
                and len({tuple(r["ranked_ids"]) for r in ok}) == 1,
                "planning_status": planned.get((query["id"], pid), {}).get(
                    "status", "not_attempted"
                ),
            }
            rows.append(point)
            points.append(point)
        candidates, chosen = points[:-1], points[-1]
        valid = [p for p in candidates if p["failed_or_missing"] == 0]
        frontier = [
            p["plan_id"]
            for p in valid
            if not any(dominates(other, p) for other in valid)
        ]
        for budget in spec["budgets_ms"]:
            feasible = [p for p in valid if p["latency_ms"] <= budget]
            best = max((p["quality"] for p in feasible), default=None)
            selected_ok = chosen["failed_or_missing"] == 0
            latency = chosen["latency_ms"]
            evaluations.append(
                {
                    "qid": query["id"],
                    "budget_ms": budget,
                    "frontier_plan_ids": frontier,
                    "observed_feasible_plans": len(feasible),
                    "oracle_complete": len(valid) == len(candidates),
                    "oracle_quality": best,
                    "selected_quality": chosen["quality"],
                    "selected_latency_ms": latency,
                    "quality_regret": (
                        max(0, best - chosen["quality"]) if best is not None else None
                    ),
                    "selection_failed": not selected_ok,
                    "budget_violation": selected_ok and latency > budget,
                    "budget_overrun_ms": (
                        max(0, latency - budget) if selected_ok else None
                    ),
                    "epsilon_pareto_hit": bool(feasible)
                    and selected_ok
                    and latency <= budget
                    and not any(
                        dominates(
                            p,
                            chosen,
                            spec.get("epsilon_quality", 0),
                            spec.get("epsilon_latency_ms", 0),
                        )
                        for p in feasible
                    ),
                }
            )
    budget_reports = []
    for budget in spec["budgets_ms"]:
        values = [v for v in evaluations if v["budget_ms"] == budget]
        regrets = [
            v["quality_regret"] for v in values if v["quality_regret"] is not None
        ]
        budget_reports.append(
            {
                "budget_ms": budget,
                "queries": len(values),
                "queries_with_feasible_oracle": len(regrets),
                "mean_observed_quality_regret": (
                    statistics.fmean(regrets) if regrets else None
                ),
                "epsilon_pareto_hit_rate": statistics.fmean(
                    v["epsilon_pareto_hit"] for v in values
                ),
                "budget_violation_rate": statistics.fmean(
                    v["budget_violation"] for v in values
                ),
                "selection_failure_rate": statistics.fmean(
                    v["selection_failed"] for v in values
                ),
                "p95_budget_overrun_ms": percentile(
                    [
                        v["budget_overrun_ms"]
                        for v in values
                        if v["budget_overrun_ms"] is not None
                    ],
                    95,
                ),
            }
        )
    return {
        "run_finished": finished,
        "snapshot_stable": stable,
        "valid_for_comparison": finished and stable,
        "oracle_scope": "declared grid, completed successful plans only",
        "plans": rows,
        "per_query_budgets": evaluations,
        "budgets": budget_reports,
        "planning_p50_ms": percentile(selected_planning, 50),
        "planning_p95_ms": percentile(selected_planning, 95),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        print(json.dumps(summarize(args.output), indent=2, allow_nan=False))
    else:
        if not args.base_url or not math.isfinite(args.timeout) or args.timeout <= 0:
            parser.error("--base-url and a finite positive --timeout are required")
        report = run(
            json.loads(args.spec.read_text()),
            args.output,
            args.base_url,
            timeout=args.timeout,
            token=os.environ.get("ANNEX_READ_KEY"),
        )
        print(json.dumps(report, indent=2, allow_nan=False))
        if not report["valid_for_comparison"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
