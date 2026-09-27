# Benchmark policy

This is the authoritative reporting contract. Commands and artifact formats live
in the [benchmark guide](crates/annex-multivector/benchmark/README.md);
launch work lives in the [implementation plan](docs/launch-verification-plan.md).

## Evaluation

- Tune on development queries. Freeze configuration before held-out evaluation.
  Previously inspected queries cannot become untouched through a new random split.
- Predeclare workloads, operating points, primary metrics, quality margins and
  aggregation. Preserve every evaluated point, including failures and losses.
- Use complete corpora for headline claims. Prefix and relevance-conditioned
  slices are diagnostics: removing distractors changes retrieval difficulty.
  Report source/slice counts, sampling, seeds and relevance coverage.
- Preserve original relevance judgments and failed queries in denominators.
  Equal aggregate scores do not prove equal rankings; lack of statistical
  significance does not establish quality parity.
- Report per-query quality and paired uncertainty. Repeated performance requests
  are not additional independent relevance judgments.

## Comparators and costs

- Identify product, version, execution mode, index/search configuration and
  durability. Embedded engines are valid comparators for embedded deployments;
  client-local emulations are not evidence about production servers.
- Give competitors comparable documented development tuning and resource budgets.
  Use identical inputs/scoring for engine comparisons. Different encoders,
  pooling or rerankers belong in explicitly named system comparisons or ablations.
- Separate encoding, training, ingestion, index readiness and recovery costs.
  Report total preparation cost as well as individual stages.
- Report quality, latency, achieved throughput, CPU, RAM and disk separately.
  Include errors/timeouts, cache state, repetitions and build variation.
- Measure scheduled-arrival latency under offered load as well as closed-loop
  concurrency. Production claims require mutation and maintenance workloads.
- Record hardware, OS, dependencies, runtime controls, model revisions, vector
  hashes, source/binary identity and resource limits. Separate architectures.

## Artifacts and claims

Raw results belong in immutable run directories or published release artifacts,
not in the source tree. Publish a durable artifact URL and digest with each claim;
retain all query outcomes, configurations and commands needed to reproduce it.
Derived tables and plots must identify those inputs.

10K documents are debugging scale; 100K is development scale. A production
performance claim requires at least one 1M+ real-document/chunk evaluation and
explicit document, chunk and token-vector counts. Synthetic fixtures verify
correctness, not competitive performance.

Keep corrections public and link the original evidence. Historical exploratory
artifacts remain recoverable through the Git revision listed in
[RESULTS.md](crates/annex-multivector/benchmark/RESULTS.md); removing generated
copies from the active tree does not upgrade their evidentiary status.

The dev/test CLI guard detects configuration changes. It cannot prove an
untouched holdout, eliminate model-training contamination, or establish launch
readiness by itself.
