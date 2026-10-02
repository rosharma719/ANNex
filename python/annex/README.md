# annex

Python client for the [ANNex](https://github.com/rosharma719/ANNex) vector search service.

## Install

```bash
pip install annex
```

## Quick start

```python
from annex import AnnexClient

client = AnnexClient("http://localhost:8080", write_key="wk-...")
client.upsert([{"id": "doc-1", "text": "hello", "metadata": {}}])
results = client.retrieve({"prefetch": [{"kind": "dense", "field": "semantic",
                                          "vector": [0.1, 0.2], "limit": 10}],
                            "limit": 5})
```
