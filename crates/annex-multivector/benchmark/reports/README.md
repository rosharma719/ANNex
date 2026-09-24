# Committed benchmark ledgers

Each engine version has exactly one append-only JSON Lines ledger:
`v<version>.jsonl`. Every line is a complete benchmark record with the exact
`multivector` Semantic Version, Git revision/dirty state, Rust version, Python
package versions, model identifiers, dataset, configuration, and results.
Commit the ledger with the code it measures; do not commit the large indexes
under `benchmark/results/`.

Read a ledger as formatted JSON with:

```bash
python benchmark/read_reports.py --ledger benchmark/reports/v0.1.0.jsonl
```

Use `python benchmark/matrix.py ...` for the standard exact-FDE and HNSW matrix.
