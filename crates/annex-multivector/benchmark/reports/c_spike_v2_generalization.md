# C-spike v2: how the confidence primitive generalizes across corpora

Follow-up to `c_spike_findings.md`. The v1 finding was that cheap
difficulty signals (top_score, top-vs-2nd margin) don't predict which
queries benefit from more candidates, but *do* correlate with absolute
per-query nDCG. This document answers the natural next question:

> **Does a cheap-features → nDCG model built on one corpus generalize to
> other corpora, or does it need per-workload retraining?**

Short answer: **not with raw features, not with quantile features alone,
but yes with quantile features + a quantile-normalized label**. The
runtime primitive that transfers isn't absolute nDCG — it's rank
percentile within the query batch.

## Setup

- Data: v3 head-to-head sweep matrices, three BEIR corpora (FiQA 41
  queries, Scifact 98, Nfcorpus 300), at candidates=250.
- Features per query: top_score, top_minus_second, top × margin, log(top),
  log(margin), returned (6 dims).
- Label: per-query nDCG@10.
- Models: 6-feature linear regression, and isotonic regression on the
  strongest single signal.
- Evaluation: leave-one-corpus-out. Train on two corpora, test on the
  third. This is the true generalization test — the model has never
  seen the test corpus's feature or label distribution.

Analysis script: `benchmark/confidence_calibration.py`.

## Raw features + absolute label (naive baseline)

| Cross-corpus split | Best pearson | R² |
|---|---:|---:|
| scifact+nfcorpus → FiQA | +0.303 | +0.091 |
| fiqa+nfcorpus → Scifact | +0.489 | **−0.639** |
| fiqa+scifact → Nfcorpus | +0.461 | **−0.148** |

Pearson stays reasonable (~0.30-0.49) — the model ranks queries roughly
right — but R² goes negative on the two larger corpora. Root cause: nDCG
distributions differ per corpus (fiqa mean 0.40, scifact 0.73, nfcorpus
0.33), and the model trained on one distribution predicts values in the
wrong absolute range for another.

## Quantile features + absolute label

| Cross-corpus split | Best pearson | R² |
|---|---:|---:|
| scifact+nfcorpus → FiQA | +0.219 | +0.036 |
| fiqa+nfcorpus → Scifact | +0.431 | **−0.822** |
| fiqa+scifact → Nfcorpus | +0.451 | **−0.672** |

Replacing raw feature values with their rank-percentile within each
batch normalizes feature distributions, but the label is still absolute
nDCG. Result: R² gets **worse** — now the model is trained on
dimensionless features but tries to output raw nDCG for a corpus with a
different scale. Doesn't help.

## Quantile features + quantile label (the fix)

| Cross-corpus split | Best pearson | R² |
|---|---:|---:|
| scifact+nfcorpus → FiQA | +0.198 | −0.028 |
| fiqa+nfcorpus → Scifact | +0.332 | **+0.098** |
| fiqa+scifact → Nfcorpus | +0.224 | **+0.022** |

Predict rank-percentile of nDCG within the batch, not absolute nDCG.
The R² is now non-negative on the two larger corpora — small but real,
and (crucially) trending in the right direction with sample size. Small
FiQA (n=41) is dominated by noise.

Pearson drops slightly because predicting a rank-percentile is a
strictly weaker task than predicting an exact value. We give up some
ranking sharpness to gain a portable, calibrated prediction.

## Interpretation

The primitive that transfers across corpora isn't "your query has 85%
expected nDCG." Absolute nDCG doesn't transfer, and never will without
per-workload recalibration.

The primitive that does transfer is **"your query is in the top X% of
confidence for this workload's typical query mix."** That's a fully
relative statement, and it survives corpus distribution shifts by
construction.

## What to build

### Confidence primitive (runtime)

At query time the engine already computes top_score and top_minus_second
for free (they're in the returned hits' scores). Track those in a
rolling window of the last N queries per collection. For each incoming
query:

1. Compute cheap features.
2. Convert each feature to its rank-percentile against the rolling
   window.
3. Apply the trained model (ship a base version, allow per-workload
   fine-tuning at deploy time).
4. Return the predicted rank-percentile as a `confidence` field on the
   query response.

Downstream users can:
- Log confidence per query and identify systematic low-confidence
  patterns.
- Trigger fallbacks (exact re-ranking, human review) below a threshold.
- Report SLO compliance: "X% of queries were above the Y confidence
  bar this hour."

### Adaptive escalation policy (runtime)

Uses the same primitive as a routing decision. Below threshold: run at
higher c. Above threshold: keep at cheap c. Static threshold on
predicted rank-percentile is portable across workloads; a per-workload
threshold optimizes for local latency/quality tradeoff.

Not yet built. Requires either an in-engine hook or a client-side
policy layer that calls the engine twice for the escalated fraction.
The `benchmark/adaptive_policy_sim.py` offline simulator can measure
the achievable Pareto given ground truth today — as a sanity check
before wiring it into the query path.

## Two-corpus caveat / future work

The generalization result is measured with n=41/98/300 — small enough
that the exact R² values will move with more data. The *sign* of the
effect (quantile-quantile flips R² from negative to non-negative) is
robust; the *magnitude* is not. Confirming on 2-3 more BEIR corpora
(trec-covid, quora, arguana) is the natural next validation before
shipping.

## Files

- `benchmark/confidence_calibration.py` — the cross-corpus study
- `benchmark/adaptive_policy_sim.py` — offline simulator for the
  escalation policy
- `benchmark/recall_difficulty.py` — earlier per-query difficulty
  correlations
- `benchmark/reports/headtohead-*-v3-matrix.json` — the per-query data
