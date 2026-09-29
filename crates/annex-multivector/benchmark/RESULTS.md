# Benchmark evidence and corrections

No retained run currently establishes production superiority or launch readiness.

Historical development runs, plots, ledgers and investigation notes are preserved
at [revision 3e709d2](https://github.com/rosharma719/ANNex/tree/3e709d25735f1481b80a06deddb25db3c6368c43).
For example:

```sh
git show 3e709d2:crates/annex-multivector/benchmark/reports/headtohead-fiqa-v3-matrix.json > /tmp/fiqa-history.json
```

Generated copies have been removed from the active tree. This document retains
the corrections; the [benchmark policy](../../../BENCHMARK_POLICY.md) governs new claims.

| Earlier claim | Correction |
| --- | --- |
| “6–40× faster than Qdrant” and FiQA “quality parity” | These were small, explored slices against `qdrant-client :memory:`, not production Qdrant Server. They do not establish server performance or statistical equivalence. |
| Equal nDCG/Recall proves identical top-10 rankings | Aggregate equality cannot establish ranking identity. The pruning ledger did not retain sufficient rankings for that assertion. |
| Comparable build times | ANNex training was excluded; competitor readiness and configuration were not consistently verified. |
| Exact reproduction from code and free datasets | Historical records lack complete frozen model, binary, configuration and input identity. Timing also varies between runs. |
| LanceDB has no native multivector support | The old adapter mean-pooled tokens. Treat those numbers as a mean-pool ablation, not evidence about native multivector retrieval. |
| SciFact's remaining deficit was necessarily model behavior | The original encoder used a shorter document limit. Later experiments changed that limit and exposed candidate-pruning losses. |
| Unsourced latency ranges for PLAID and Vespa | Withdrawn; these were not comparable measurements on the recorded workload. |
| Early local NFCorpus hybrid numbers used standard BEIR nDCG | The initial runner used exponential gains (`2^grade−1`). BEIR/trec_eval uses the relevance grade directly. New summaries use linear gains; preserved early summaries must not be compared numerically with them. Binary-relevance SciFact/ArguAna scores are unaffected. |

Historic NYT/H&M curves and optimization notes are also exploratory. Their
original files remain in the same revision. Current commands live in
[docs/benchmarks.md](../../../docs/benchmarks.md); fresh measurements must include
their own artifacts rather than copy these tables forward.

## Local quality comparison — 2026-09-28

Implementation `54e9540`, Apple M2 / 16 GiB, full corpora, shared pinned MiniLM
vectors, exact cosine and 100 candidates per channel. Configurations were frozen
before all test-partition runs. nDCG uses BEIR/trec_eval linear gains.

| Test-partition system | NFCorpus (162 queries) | SciFact (150) | ArguAna (703) |
| --- | ---: | ---: | ---: |
| ANNex dense | 0.31485 | 0.66420 | 0.48305 |
| ANNex BM25 | 0.31335 | 0.63938 | 0.39781 |
| ANNex hybrid RRF | 0.34550 | 0.68909 | 0.48205 |
| Qdrant Server dense | 0.31494 | 0.66420 | 0.48155 |
| Qdrant Server shared BM25 | 0.31295 | 0.63938 | 0.39605 |
| Qdrant Server shared BM25 + dense RRF | 0.34411 | 0.68750 | 0.47862 |
| LanceDB dense | 0.31485 | 0.66420 | 0.48324 |
| LanceDB native BM25 | 0.32981 | 0.67069 | 0.47698 |
| LanceDB native hybrid RRF | 0.35055 | 0.72700 | 0.52349 |

ANNex hybrid improves over its dense baseline on NFCorpus and SciFact and is
slightly worse on ArguAna. LanceDB native hybrid leads all three configurations;
these results do not establish ANNex superiority. Its stemming, stop words and
ASCII folding differ from ANNex/Qdrant's shared lexical setup. Native tie orders
also differ. No engine speed ranking is claimed: ANNex/Qdrant use HTTP and
LanceDB is embedded, with serial requests and no warmup. Public-data history
means these frozen partitions cannot be described as untouched holdouts.

All 9,135 test responses and 9,126 development responses succeeded. Test nDCG
and Recall@10/20/100 agree with independent `pytrec_eval` evaluation within
1e-12. Per-query outcomes, paired intervals, MRR, latencies, ingest stages,
frozen configurations, vectors and checksums are retained in the
[draft evidence release](https://github.com/rosharma719/ANNex/releases/tag/untagged-be93740995d824df1d7f)
(repository access required until publication). The archive is outside the
source tree; SHA-256 of `annex-local-quality-20260928.tar.gz`:

```text
076ba8d319c895a2adfc6244d189027896f52a38e1ea682a5cba32ca8d7c0f75
```

Earlier development losses and the gain-definition correction are included in
that archive. Commands and comparator semantics live in [README.md](README.md).

## Lexical policy comparison — 2026-09-29

Implementation `f68f970`, on the same Apple M2 / 16 GiB host. The development
comparison selected one policy for every corpus: the persisted English analyzer,
raw query term frequency, equal channel weights and RRF `k=10`. English beat the
plain analyzer on all five development partitions. A fixed RRF grid
(`k=0,10,30,60,100,200`) selected 10 by macro-average nDCG@10. No policy was
selected per corpus or query. New frozen configurations were then written before
the test runs.

All engines used the same pinned normalized MiniLM vectors, exact cosine, 100
candidates per channel and RRF `k=10`. ANNex and Qdrant Server received identical
analyzed BM25 document/query weights; LanceDB used native default FTS. nDCG@10:

| Held-out system | NFCorpus (162) | SciFact (150) | ArguAna (703) | FiQA (324) | SciDocs (500) | Macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ANNex dense | 0.31485 | 0.66420 | 0.48305 | 0.37728 | **0.22429** | 0.41273 |
| ANNex BM25 | 0.32849 | **0.67408** | 0.46676 | **0.24350** | 0.16247 | 0.37506 |
| **ANNex hybrid** | 0.35742 | 0.72639 | 0.52384 | 0.37977 | 0.21329 | **0.44014** |
| Qdrant Server hybrid | **0.35797** | 0.72572 | 0.52074 | **0.37983** | **0.21357** | 0.43957 |
| LanceDB native hybrid | 0.35425 | **0.72762** | **0.52646** | 0.37838 | 0.21324 | 0.43999 |

ANNex has the highest numerical macro average, 0.00015 above LanceDB and 0.00057
above Qdrant. It wins two corpora against LanceDB, loses two and is effectively
tied on one. Every per-corpus paired-bootstrap 95% interval between hybrid
systems includes zero. These results demonstrate parity and a material ANNex
improvement, not a statistically established quality advantage. In particular,
SciDocs hybrid remains below dense alone, so unconditional two-channel fusion is
not universally beneficial.

All 16,551 held-out responses and 11,028 analyzer-development responses
succeeded. The frozen contracts, manifests, event journals, per-query rankings,
paired intervals and server logs are in the same
[draft evidence release](https://github.com/rosharma719/ANNex/releases/tag/untagged-be93740995d824df1d7f).
SHA-256 of `annex-lexical-quality-20260929.tar.gz`:

```text
cb7281f820eeb50c37b34308802375e83b3326e2958ed348fbd9e499b6b6f6a5
```

Latency remains descriptive only because ANNex/Qdrant use HTTP while LanceDB is
embedded. The next quality target is a globally validated query-adaptive fusion
or late-interaction reranker, evaluated on new frozen corpora rather than
retuning these now-observed test partitions.
