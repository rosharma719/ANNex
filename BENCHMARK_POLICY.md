# Benchmark policy

This document is the honest playbook for how ANNex measures itself and
reports results. It exists because ANN benchmark literature has a long
history of quiet tuning-on-test, cherry-picked operating points, and
comparator misconfiguration. Publishing rules up front makes it easier
to spot when we (or someone else) violate them.

## Data separation: dev vs test

- **Development slices** are used for iteration on parameters (candidate
  count, `ef_search`, FDE dimensions, quantization settings, pruning
  parameters). Any hyperparameter chosen based on a slice's numbers
  MUST be chosen against a development slice, never the test slice.
- **Test slices** are held out. Configuration is frozen before the test
  slice is evaluated. If we discover we need to re-tune, we do it on the
  dev slice and re-publish with a note about the retune.
- The `benchmark/headtohead.py` sweep at multiple candidate counts
  (100, 250, 500, 1000, 2000) is **exploratory / development** by
  definition. Freezing a single operating point per corpus and running
  it against a held-out slice is what the published number should
  reflect.

Concretely: `benchmark/reports/headtohead-*-v3-matrix.json` files are
dev-set exploration artefacts. Any single number quoted in RESULTS.md,
a blog post, or a marketing claim MUST come from a frozen-config run
against a slice that was not touched during tuning.

## Slice labelling

- Every slice used in a benchmark record must be documented with:
  - Dataset name and version (BEIR release, ir-datasets version)
  - Total document count in the source
  - Total query count in the source
  - Slice size (limit-docs, limit-queries)
  - Sampling method (`prefix`, `qrels`, other)
  - Sample seed
  - **How many queries actually had judgeable relevance in the slice**
    (the "evaluable queries" number, often much smaller than
    `limit-queries` when we take a limit-docs subset)
- Do not refer to a 41-query FiQA subset as "FiQA". Say
  "FiQA / 10K-doc prefix slice / 41 evaluable queries".

## Competitor configuration

- Comparators run in the **configuration a real user would deploy**,
  not the most convenient in-process client. If a comparator has a
  server + client split, we run the server. The convenience path is
  labelled explicitly when it is used at all (see next).
- Convenience paths (Qdrant `:memory:`, LanceDB in-process without
  ANN, etc.) may be included as **exact / brute-force references**,
  never as "Qdrant" or "LanceDB" full stop. They must be labelled
  `<vendor> <mode> <configuration>` so readers can see the mode
  isn't the production server.
- Comparator versions are pinned in the benchmark script and
  documented alongside the results.
- Comparator hardware is disclosed. If we run competitor X in a Docker
  container on the same M2 mini we ran ANNex on, we say so.

## Reporting

- **Raw per-query results are retained.** Every published headline
  number must be reproducible from a committed matrix file that
  includes per-query rankings, per-query latency, and per-query
  quality metrics.
- **Failed / timed-out queries are kept in the denominator.** No
  silent dropping of queries the system couldn't answer.
- **Unfavorable operating points are preserved.** If we ran a
  candidate sweep, every point in the sweep goes into the committed
  matrix. Publishing only the point where we look best is exactly
  the behaviour this policy exists to prevent.
- **Latency, memory, disk, throughput are reported separately.** No
  fused "cost" number that lets us hide one axis inside another.
- **Approximate-vs-approximate quality parity is stated.** If our
  quality is 3% below a comparator's, that goes in the headline, not
  the footnote.
- **Corrections and retractions are preserved publicly.** If we find
  we misconfigured a comparator or ran on a truncated slice, we
  publish the correction alongside (not in place of) the original.

## Scaling milestones

- **10K docs**: dev / debugging scale. Numbers at this scale exist for
  fast iteration and are not claims about "how ANNex performs."
- **100K docs**: serious development scale. Numbers here can be
  published if the slice labelling makes clear it isn't production
  scale.
- **1M docs**: architecture validation scale. Most architectural lies
  (RAM cliffs, quadratic-in-corpus code paths, latency variance
  explosions) show up here first.
- **>1M**: publish as such when we get there.

Any headline claim ("fastest CPU multi-vector") needs a 1M-doc data
point behind it before it's honest. Sub-1M numbers stand as
"development scale" claims.

## Pareto reporting

Long-term the benchmark story is a frontier, not a multiplier:

```
quality vs latency
quality vs RAM
quality vs disk
quality vs QPS
quality vs cost
```

Every operating point plotted. If ANNex dominates a portion of a
frontier, the graph is much harder to dispute than a hand-selected
"38× faster" number. Both forms are fine to publish, but the frontier
is the load-bearing part.

## Concurrency

- Single-query p50 is a lower bound on what production sees. Every
  serious latency claim should also include:
  - QPS at concurrency 1, 8, 32, 128
  - p50 / p95 / p99 at each concurrency
  - CPU utilization, RSS
- A system exceptional at concurrency=1 and mediocre at concurrency=32
  is a research prototype. We shouldn't publish research-prototype
  numbers as production claims.

## Mixed-workload

- Sustained mixed insert / update / delete / query loads exercise
  behaviours that a static query benchmark never touches: HNSW
  rebuild during load, compaction pressure, WAL replay after crash,
  memory reclamation.
- Any claim about being an OLTP-shaped vector DB (not a static index)
  needs a mixed-workload benchmark behind it.

## Hardware

- CPU model, memory, disk type, and kernel version disclosed for every
  headline number.
- Cross-architecture (ARM64 + x86_64) results published separately.
  Do not aggregate a claim from one architecture as though it applies
  to the other — SIMD paths, cache sizes, and thread scheduler
  behaviour all diverge.

## Enforcement

- Every RESULTS.md table cites the underlying committed matrix file.
- CI compiles the benchmark harness and reference oracle so a broken
  benchmark can't slip in unnoticed.
- New benchmark commits that violate any rule above should be either
  reverted or updated with a documented reason for the deviation.

## What "we retract" looks like

We already found and corrected one instance of tuning-on-test-adjacent
behaviour (the "0.02% gap within noise, we match Qdrant" claim from
FiQA c=500 was on a 41-query slice — that's noise floor, not a match
on a real benchmark). The pattern to follow:

1. Publish the correction with the same prominence as the original.
2. Preserve the original claim + record + reasoning in git history.
3. Add a note to RESULTS.md explaining what changed and why.

Nothing kills a benchmark reputation faster than pretending a wrong
number was never published. The right move is always to acknowledge
and correct.


## Enforced operating-point protocol

The sweep and head-to-head CLIs now default to a deterministic development
partition. Test runs require a previously frozen single operating point and
matching content/settings/source fingerprints. See
[the benchmark CLI protocol](crates/annex-multivector/benchmark/README.md#development-versus-held-out-evaluation).
This guard prevents accidental sweeps on held-out IDs; it cannot establish that
an operator has never inspected those IDs. Existing published exploratory
artifacts retain their original status and are not relabeled as held-out results.
