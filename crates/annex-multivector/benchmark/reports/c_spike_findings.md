# C-spike: does per-query difficulty predict compute allocation?

Question from the strategy roadmap: **can we predict per-query which queries
need more candidate budget, so we escalate only for the queries that will
actually improve?** If yes, that's the mechanism for the "adaptive c" /
Recall-SLO product story. If no, the whole planner thesis collapses (or
needs reformulation).

## Data

Per-query outputs from `benchmark/headtohead.py` on two BEIR corpora,
sweep over `candidates ∈ {250, 500, 1000}`, ColBERTv2 embeddings.
Signals recorded per query at each candidate level:

- `ndcg@10`: what the query actually scored at this level
- `top_score`: MaxSim of the best hit
- `top_minus_second`: MaxSim margin between best and runner-up (a proxy
  for "how confident is this top-1?")

Analysis: `benchmark/recall_difficulty.py` computes pearson/spearman
correlations plus deltas from cheap→high candidate levels.

## Result — the original thesis fails

The "predict which queries need more c" thesis, as originally scoped,
does not survive the data. Simple cheap-c difficulty signals **do not
predict** the per-query improvement from raising c.

|  | FiQA | Scifact |
|---|---:|---:|
| Queries with data at every c | 41 | 98 |
| Queries flat (Δ ≤ 0.01, c=250 → 1000) | 87.8% | 91.8% |
| Queries helped (Δ > 0.01) | 4.9% | 2.0% |
| Queries hurt (Δ < -0.01) | 7.3% | 6.1% |
| top_score → improvement Δ (pearson) | −0.02 | +0.06 |
| top-vs-2nd → improvement Δ (pearson) | −0.06 | +0.01 |

Neither pearson nor spearman correlation exceeds 0.08 in absolute value.
On this data, whether a query benefits from more candidates is essentially
independent of these cheap difficulty signals.

## Result — a stronger finding hides in the same data

Two things the data does show cleanly:

**1. ~90% of queries plateau at c=250 across both corpora.**
Spending more compute mostly does nothing (and in ~7% of cases actively
hurts, presumably from bad candidates displacing better ones after
the rescore sort). We are currently overspending on 87-92% of queries.

**2. Cheap difficulty signals DO predict absolute query quality:**

| Signal | Correlation with cheap-c nDCG@10 (pearson) | |
|---|---:|---:|
|  | FiQA | Scifact |
| top_score | +0.20 | **+0.44** |
| top-vs-2nd margin | +0.32 | **+0.54** |

That's moderate on FiQA and quite strong on Scifact. A margin-based
confidence score would meaningfully rank queries by how well they were
served, even if it can't tell us how much MORE effort would help.

## What this actually unlocks

The original planner story ("dynamic c per query") is dead in this
form. Two better product angles come from the same data:

### Angle 1: Adaptive escalation with a stable-top-k check

Start every query at low c (e.g. 100-200). Compare top-K stability
between cheap and mid c (jaccard, kendall-tau on returned IDs). Only
escalate for queries whose top-K is unstable — the ones we can't tell
from cheap signals but whose ranking is visibly noisy.

Because 90% of queries plateau early, the majority pay ~5-8 ms instead
of ~15 ms. Uncertain queries get escalated. Average latency drops
substantially at unchanged quality.

Rank-disagreement instrumentation is now in `headtohead.py`
(`ranked_ids` per query) so a follow-up sweep can measure whether
top-K disagreement between c-levels correlates with improvement Δ —
the specific hypothesis this angle rests on. First run of that
analysis is a natural next step.

### Angle 2: Confidence output (RecallGuard-style)

Return `estimated_recall` alongside each result set, computed from the
top-vs-2nd margin. On Scifact the correlation with actual nDCG is
+0.54 which is enough for a calibrated 3-4 bucket confidence label
("very high / high / medium / low") even without a full calibration
model. Sales narrative: "you can now know when to fall back to
exact re-ranking instead of guessing."

## Verdict

C thesis v1 ("predict which queries need MORE") — **fails on this data.**

C thesis v2 ("predict which queries CAN skip more") — **plausible**,
consistent with the 90% plateau finding. Rank-disagreement is the
next signal to test; the ranked_ids instrumentation is in place.

C thesis v3 ("confidence output per query") — **already works** with
existing cheap signals, modest but real correlation with per-query
nDCG on both corpora. Cheapest to ship.

## Reproducing

```bash
cd crates/annex-multivector
.venv/bin/python benchmark/headtohead.py --dataset beir/fiqa/test \
    --limit-docs 10000 --limit-queries 100 --engines annex,qdrant \
    --annex-sweep 250,500,1000 --output benchmark/results/headtohead-fiqa-v2

.venv/bin/python benchmark/headtohead.py --dataset beir/scifact/test \
    --limit-docs 5000 --limit-queries 100 --engines annex,qdrant \
    --annex-sweep 250,500,1000 --output benchmark/results/headtohead-scifact-v2

.venv/bin/python benchmark/recall_difficulty.py \
    benchmark/results/headtohead-fiqa-v2/matrix.json \
    benchmark/results/headtohead-scifact-v2/matrix.json
```
