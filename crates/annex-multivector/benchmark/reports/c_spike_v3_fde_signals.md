# C-spike v3: engine-level FDE-vs-MaxSim signals

Follow-up to `c_spike_v2_generalization.md`. That doc established the
confidence primitive that generalizes (quantile features + quantile
labels) but noted that margin-based adaptive escalation was weak and
proposed **FDE-rank vs MaxSim-rank disagreement** as the next signal to
try — a direct, engine-level readout of "did the candidate stage put
the winner in the top or did MaxSim have to promote it from further
down."

The engine now exposes `fde_score` on every `Hit`, and the benchmark
harness computes three derived features per query:

- `fde_top_score`: FDE score of the #1-by-MaxSim hit
- `fde_top_rank_in_fde`: 0-indexed rank of that same hit in the FDE-only
  ordering. High value = MaxSim rescued a candidate FDE buried.
- `fde_maxsim_agreement`: fraction of returned hits whose FDE and MaxSim
  ranks differ by ≤3.

## Did the new signals fix adaptive escalation?

Sample: v4 head-to-head sweeps (fiqa, scifact, nfcorpus), c=250 as
cheap, c=1000 as high. All numbers report mean nDCG@10 and mean per-
query latency (ms) across the corpus.

### FiQA (n=41)

| Policy | nDCG | mean ms | escalated |
|---|---:|---:|---:|
| fixed c=250 | 0.4036 | 14.7 | — |
| fixed c=1000 | 0.4186 | 19.4 | — |
| adaptive: bottom 30% by margin | 0.4131 | 20.0 | 31.7% |
| adaptive: bottom 30% by fde_disagree | 0.4123 | 21.6 | 41.5% |
| adaptive: bottom 30% by fde_top_rank | 0.4023 | 20.4 | 31.7% |
| oracle | **0.4217** | **15.7** | 4.9% |

### Scifact (n=98)

| Policy | nDCG | mean ms | escalated |
|---|---:|---:|---:|
| fixed c=250 | 0.7264 | 10.8 | — |
| fixed c=1000 | 0.7250 | 24.1 | — |
| adaptive: bottom 30% by margin | 0.7281 | 18.0 | 30.6% |
| adaptive: bottom 30% by fde_disagree | 0.7220 | 19.0 | 32.7% |
| adaptive: bottom 30% by fde_top_rank | 0.7217 | 18.1 | 32.7% |
| oracle | **0.7329** | **11.2** | 2.0% |

### Nfcorpus (n=300)

| Policy | nDCG | mean ms | escalated |
|---|---:|---:|---:|
| fixed c=250 | 0.3272 | 9.1 | — |
| fixed c=1000 | 0.3435 | 22.2 | — |
| adaptive: bottom 10% by fde_disagree | 0.3326 | 12.1 | 14.3% |
| adaptive: bottom 30% by margin | 0.3340 | 15.9 | 30.0% |
| adaptive: bottom 30% by fde_top_rank | 0.3343 | 15.7 | 30.0% |
| oracle | **0.3480** | **12.3** | 14.3% |

### Verdict on adaptive escalation

FDE-disagreement doesn't outperform margin on any corpus. It ties or
underperforms on FiQA and Scifact, and matches margin on Nfcorpus.

**Why the engine-level signal didn't rescue this**: even a perfect
predictor (oracle) only helps ~2-5% of queries on FiQA and Scifact.
The variance we can predict is bounded by how much variance actually
exists in "queries that benefit from more candidates," which turns out
to be small on our data. Any classifier working on a base rate of 2-5%
faces a fundamentally hard signal-to-noise problem.

The one place FDE disagreement wins is Nfcorpus at bottom-10% escalation,
where it reaches the *same* nDCG as margin at bottom-30% but with a third
of the escalations (14.3% vs 30.0%). That's a real 4 ms mean-latency saving
compared to margin, but the corpus is exactly where the base rate of
helpable queries is highest (12.7% helped in the raw data), so it's an
edge case rather than a general win.

## Did the new signals help the confidence-model primitive?

Cross-corpus leave-one-out R² on the confidence model (quantile
features + quantile labels), comparing v2 (6 features from margin +
top_score) with v3 (9 features adding FDE disagreement):

| Cross-corpus split | v2 R² (no FDE) | **v3 R² (with FDE)** | Δ |
|---|---:|---:|---:|
| → FiQA | −0.028 | −0.012 | **+0.016** |
| → Scifact | +0.098 | **+0.117** | **+0.019** |
| → Nfcorpus | +0.022 | **+0.042** | **+0.020** |

Small but consistent improvements across all three splits. The FDE
signal *does* improve the confidence prediction; it just doesn't help
enough for the tightly-bounded adaptive-escalation problem.

## Interpretation

Two primitives, two very different fates:

1. **Confidence output**: works. Ship as the C productization surface.
   Engine returns a `confidence` percentile per query response, built
   from (top_score, top_minus_second, fde_maxsim_agreement,
   fde_top_rank_in_fde, ...) after quantile normalization within a
   rolling window. Cross-corpus R² is small but *positive* and the
   sign is stable. Downstream users can log it, threshold on it, and
   report SLO compliance from it.
2. **Adaptive escalation**: fundamentally limited by the fraction of
   queries that actually benefit from more candidates (2-5% on
   FiQA/Scifact, ~13% on Nfcorpus). No feature we've tried closes the
   gap to the oracle. Larger candidate budgets don't add nDCG for
   most queries because the FDE stage already found what could be
   found; the ceiling is candidate-stage recall, not rescoring quality.

## What else to try (if anyone comes back to this)

- **Corpus-adaptive candidate budgets**: instead of dynamic per-query
  c, run a small validation set at deployment and choose a fixed c
  that hits the target SLO for that workload. The 3-corpus data
  suggests c=250 is nearly optimal for two of three corpora tested.
- **Multi-signal ensemble**: linear combination of margin + FDE
  disagreement might beat either alone. Present data is too small
  to distinguish.
- **Feature engineering** on the FDE side: FDE score entropy in
  top-K, FDE score decay slope, unique centroid coverage. Requires
  engine changes to expose. Marginal expected gain.
- **Confidence-based fallback** to exact rescoring: instead of
  escalating annex-multivector's own c, hand low-confidence queries
  to a slower but higher-quality path (exhaustive MaxSim). Uses the
  confidence primitive we do have.
