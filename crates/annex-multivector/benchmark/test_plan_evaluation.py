import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

from measurement import Journal
from plan_evaluation import (
    SELECTED,
    dominates,
    execution_signature,
    pin_request,
    json_safe,
    render,
    run,
    summarize,
    validate,
)
from server import annex_server, http


def spec():
    return {
        "version": 1,
        "fingerprints": {
            "corpus": "fixture-documents",
            "configuration": "fixture-config",
            "models": "caller-provided-fixture-vectors",
        },
        "top_k": 1,
        "repeats": 2,
        "budgets_ms": [1, 1000],
        "queries": [
            {
                "id": "q",
                "inputs": {"vector": [1, 0], "text": "alpha"},
                "qrels": {"a": 2, "b": 1},
            }
        ],
        "plans": [
            {
                "id": "exact",
                "request": {
                    "prefetch": [
                        {
                            "kind": "dense",
                            "field": "semantic",
                            "vector": {"$input": "vector"},
                            "backend": "exact",
                            "limit": 3,
                        }
                    ],
                    "limit": 1,
                },
            },
            {
                "id": "ann",
                "request": {
                    "prefetch": [
                        {
                            "kind": "dense",
                            "field": "semantic",
                            "vector": {"$input": "vector"},
                            "backend": "hnsw",
                            "ef_search": 16,
                            "limit": 3,
                        }
                    ],
                    "limit": 1,
                },
            },
        ],
        "selection": {
            "request": {
                "planning_mode": "auto",
                "query": {"dense": {"semantic": {"$input": "vector"}}},
                "limit": 1,
            }
        },
    }


def plan(operator="exact_dense", limit=3, generation=7, policy=None):
    return {
        "stats": {"generation": generation},
        "filter": "none",
        "eligible_documents": 3,
        "policy": policy,
        "stages": [
            {
                "kind": "parallel",
                "detail": [
                    {
                        "index": 0,
                        "operator": operator,
                        "limit": limit,
                        "ef_search": 16 if operator == "hnsw_dense" else None,
                        "reason": "requested_exact",
                        "estimated_cost_units": 10,
                    }
                ],
            },
            {"kind": "context", "detail": {"operator": "ranked", "result_limit": 1}},
        ],
    }


class PlanEvaluationTests(unittest.TestCase):
    def test_dominance_and_epsilon_boundaries(self):
        slow = {"quality": 0.8, "latency_ms": 10}
        fast = {"quality": 0.8, "latency_ms": 5}
        self.assertTrue(dominates(fast, slow))
        self.assertFalse(dominates(slow, slow))
        self.assertFalse(dominates(fast, slow, 0.01))
        self.assertTrue(dominates({"quality": 0.9, "latency_ms": 4}, slow, 0.05, 1))

    def test_input_validation_and_plan_pinning(self):
        value = spec()
        validate(value)
        self.assertEqual(
            render({"v": {"$input": "vector"}}, {"vector": [1, 2]}), {"v": [1, 2]}
        )
        for mutate in [
            lambda s: s["plans"].append(s["plans"][0]),
            lambda s: s.update(budgets_ms=[float("nan")]),
            lambda s: s.update(repeats=0),
            lambda s: s["fingerprints"].pop("models"),
            lambda s: s["plans"][0]["request"].update(
                filter={"op": "exists", "field": "tenant"}
            ),
        ]:
            invalid = copy.deepcopy(value)
            mutate(invalid)
            with self.assertRaises((ValueError, KeyError)):
                validate(invalid)
        request = render(value["selection"]["request"], value["queries"][0]["inputs"])
        expected = plan(
            "hnsw_dense",
            policy={
                "generated_prefetch": [
                    {
                        "kind": "dense",
                        "field": "semantic",
                        "vector": [1, 0],
                        "backend": "auto",
                    }
                ]
            },
        )
        pinned = pin_request(request, expected)
        self.assertEqual(pinned["planning_mode"], "manual")
        self.assertEqual(pinned["prefetch"][0]["backend"], "hnsw")
        self.assertEqual(pinned["prefetch"][0]["ef_search"], 16)
        changed = copy.deepcopy(expected)
        changed["stages"][0]["detail"][0]["estimated_cost_units"] = 300
        self.assertEqual(execution_signature(expected), execution_signature(changed))

    def test_freeze_before_execution_failures_and_replay(self):
        calls = []

        def call(route, request):
            calls.append(route)
            if route == "/v1/stats":
                return {"generation": 7, "documents": 3}
            if route == "/v1/plan":
                if request.get("planning_mode") == "auto":
                    return plan(
                        policy={
                            "generated_prefetch": [
                                {
                                    "kind": "dense",
                                    "field": "semantic",
                                    "vector": [1, 0],
                                    "backend": "auto",
                                }
                            ]
                        }
                    )
                return plan(
                    "hnsw_dense"
                    if request["prefetch"][0]["backend"] == "hnsw"
                    else "exact_dense"
                )
            if request["prefetch"][0]["backend"] == "hnsw":
                raise TimeoutError("deliberate failed counterfactual")
            return {
                "matches": [{"id": "a", "score": 1.0}],
                "trace": {
                    "plan": plan(),
                    "generation": 7,
                    "channels": [{"backend": "exact_dense"}],
                    "per_stage_actual_ms": [1, 0.1],
                },
            }

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            report = run(spec(), path, "http://fixture", call=call)
            self.assertEqual(
                calls[:5], ["/v1/stats"] + ["/v1/plan"] * 3 + ["/v1/retrieve"]
            )
            self.assertEqual(report, summarize(path))
            self.assertTrue(report["valid_for_comparison"])
            self.assertEqual(report["plans"][1]["failed_or_missing"], 2)
            self.assertEqual(report["plans"][1]["quality"], 0.0)
            self.assertFalse(report["per_query_budgets"][0]["oracle_complete"])
            observations = (path / "observations.jsonl").read_text()
            self.assertIn("TimeoutError", observations)
            with self.assertRaises(FileExistsError):
                run(spec(), path, "http://fixture", call=call)

    def test_interruption_and_known_frontier_regret(self):
        value = spec()
        value["repeats"] = 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            (path / "manifest.json").write_text(json.dumps({"spec": value}))
            journal = Journal(path / "observations.jsonl")
            for pid, quality, latency in [
                ("exact", 1.0, 5.0),
                ("ann", 0.7, 1.0),
                (SELECTED, 0.8, 8.0),
            ]:
                journal.write(
                    "query_finished",
                    pid,
                    qid="q",
                    repeat=0,
                    status="ok",
                    quality=quality,
                    latency_ms=latency,
                    ranked_ids=["a"],
                )
            journal.write("run_finished", "annex", snapshot_stable=True)
            journal.close()
            report = summarize(path)
            generous = report["per_query_budgets"][1]
            self.assertEqual(generous["frontier_plan_ids"], ["exact", "ann"])
            self.assertAlmostEqual(generous["quality_regret"], 0.2)
            self.assertFalse(generous["epsilon_pareto_hit"])
            self.assertTrue(report["per_query_budgets"][0]["budget_violation"])
            self.assertEqual(report["budgets"][0]["p95_budget_overrun_ms"], 7.0)
            # Replace the journal with an interrupted attempt and a torn tail.
            (path / "observations.jsonl").unlink()
            journal = Journal(path / "observations.jsonl")
            journal.write("query_started", SELECTED, qid="q", repeat=0)
            journal.close()
            with (path / "observations.jsonl").open("ab") as file:
                file.write(b'{"event":')
            report = summarize(path)
            self.assertFalse(report["valid_for_comparison"])
            self.assertEqual(report["budgets"][0]["selection_failure_rate"], 1.0)
            self.assertEqual(report["plans"][-1]["failed_or_missing"], 1)

    def test_mismatched_execution_retains_response_but_scores_zero(self):
        def call(route, request):
            if route == "/v1/stats":
                return {"generation": 7}
            if route == "/v1/plan":
                return plan(
                    policy={
                        "generated_prefetch": [
                            {"kind": "dense", "field": "semantic", "vector": [1, 0]}
                        ]
                    }
                )
            return {
                "matches": [{"id": "a", "score": 1.0}],
                "trace": {
                    "plan": plan(),
                    "generation": 7,
                    "channels": [{"backend": "hnsw_dense"}],
                },
            }

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            report = run(spec(), path, "http://fixture", call=call)
            self.assertTrue(all(p["quality"] == 0 for p in report["plans"]))
            self.assertTrue(all(p["completed"] == 0 for p in report["plans"]))
            events = [
                json.loads(line)
                for line in (path / "observations.jsonl").read_text().splitlines()
            ]
            failed = [e for e in events if e["event"] == "query_finished"]
            self.assertEqual(len(failed), 6)
            self.assertTrue(
                all(e["response"]["matches"][0]["id"] == "a" for e in failed)
            )
            self.assertTrue(all("backends disagree" in e["error"] for e in failed))

    def test_nonfinite_response_is_rejected_and_still_journalable(self):
        self.assertEqual(
            json_safe({"score": float("nan")}), {"score": {"invalid_float": "nan"}}
        )

        def call(route, request):
            if route == "/v1/stats":
                return {"generation": 7}
            if route == "/v1/plan":
                return plan(
                    policy={
                        "generated_prefetch": [
                            {"kind": "dense", "field": "semantic", "vector": [1, 0]}
                        ]
                    }
                )
            return {
                "matches": [{"id": "a", "score": float("nan")}],
                "trace": {
                    "plan": plan(),
                    "generation": 7,
                    "channels": [{"backend": "exact_dense"}],
                },
            }

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "run"
            report = run(spec(), path, "http://fixture", call=call)
            self.assertEqual(report["budgets"][0]["selection_failure_rate"], 1.0)
            events = [
                json.loads(row)
                for row in (path / "observations.jsonl").read_text().splitlines()
            ]
            responses = [e for e in events if e["event"] == "query_finished"]
            self.assertTrue(
                all(
                    e["response"]["matches"][0]["score"] == {"invalid_float": "nan"}
                    for e in responses
                )
            )

    def test_snapshot_change_invalidates_comparison(self):
        n = 0

        def call(route, request):
            nonlocal n
            if route == "/v1/stats":
                n += 1
                return {"generation": 7 if n == 1 else 8}
            if route == "/v1/plan":
                raise RuntimeError("unavailable operator")
            raise AssertionError("invalid plans must not execute")

        with tempfile.TemporaryDirectory() as temp:
            report = run(spec(), Path(temp) / "run", "http://fixture", call=call)
            self.assertFalse(report["valid_for_comparison"])
            self.assertTrue(all(p["failed_or_missing"] == 2 for p in report["plans"]))


@unittest.skipUnless(
    os.environ.get("ANNEX_TEST_BINARY"),
    "set ANNEX_TEST_BINARY for real HTTP evaluation",
)
class RealPlanEvaluationTests(unittest.TestCase):
    def test_real_operator_grid_and_auto_selection(self):
        value = spec()
        value["queries"][0]["inputs"]["tokens"] = [[1, 0]]
        value["queries"].append(
            {
                "id": "q2",
                "inputs": {"vector": [0, 1], "text": "beta", "tokens": [[0, 1]]},
                "qrels": {"b": 2, "c": 1},
            }
        )
        bm25 = {"kind": "bm25", "text": {"$input": "text"}, "limit": 3}
        dense = value["plans"][0]["request"]["prefetch"][0]
        value["plans"].extend(
            [
                {"id": "bm25", "request": {"prefetch": [bm25], "limit": 1}},
                {
                    "id": "hybrid-rerank",
                    "request": {
                        "prefetch": [bm25, dense],
                        "limit": 1,
                        "rerank": {
                            "field": "tokens",
                            "vectors": {"$input": "tokens"},
                            "limit": 3,
                        },
                    },
                },
                {
                    "id": "maxsim",
                    "request": {
                        "prefetch": [
                            {
                                "kind": "multivector",
                                "field": "tokens",
                                "vectors": {"$input": "tokens"},
                                "backend": "exact",
                                "limit": 3,
                            }
                        ],
                        "limit": 1,
                    },
                },
            ]
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with annex_server(
                Path(os.environ["ANNEX_TEST_BINARY"]).resolve(), root / "server", 2
            ) as base:
                documents = [
                    {
                        "id": id,
                        "text": text,
                        "representations": {
                            "semantic": {"kind": "dense", "vector": vector},
                            "tokens": {"kind": "multivector", "vectors": [vector]},
                        },
                    }
                    for id, text, vector in [
                        ("a", "alpha", [1, 0]),
                        ("b", "beta", [0, 1]),
                        ("c", "gamma", [-1, 0]),
                    ]
                ]
                http(base, "/v1/vectors/upsert", {"documents": documents})
                http(
                    base,
                    "/v1/dense/index",
                    {"field": "semantic", "m": 4, "ef_construct": 16},
                )
                report = run(value, root / "evaluation", base)
                self.assertTrue(report["valid_for_comparison"])
                self.assertEqual(len(report["plans"]), 12)
                self.assertTrue(
                    all(p["completed"] == 2 for p in report["plans"]), report
                )
                self.assertTrue(all(p["quality"] == 1 for p in report["plans"]), report)
                events = [
                    json.loads(row)
                    for row in (root / "evaluation/observations.jsonl")
                    .read_text()
                    .splitlines()
                ]
                actual = {
                    e["system"]: [
                        c["backend"] for c in e["response"]["trace"]["channels"]
                    ]
                    for e in events
                    if e["event"] == "query_finished"
                }
                self.assertEqual(actual["exact"], ["exact_dense"])
                self.assertEqual(actual["ann"], ["hnsw_dense"])
                self.assertEqual(actual["bm25"], ["bm25"])
                self.assertEqual(actual["maxsim"], ["exact_maxsim"])
                self.assertEqual(actual["hybrid-rerank"], ["bm25", "exact_dense"])
                self.assertEqual(report, summarize(root / "evaluation"))
