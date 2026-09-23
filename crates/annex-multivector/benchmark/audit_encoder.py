#!/usr/bin/env python3
"""Fail-fast audit of the ColBERT checkpoint and PyLate encoding conventions."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers.utils import cached_file

from env import load_env
from provenance import write_report
from colbert_config import DOCUMENT_LENGTH, load as load_colbert

load_env()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="colbert-ir/colbertv2.0")
    parser.add_argument("--report-dir", type=Path, default=Path("benchmark/reports"))
    args = parser.parse_args()

    model = load_colbert(args.model)
    checkpoint_path = cached_file(args.model, "model.safetensors")
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        checkpoint_projection = checkpoint.get_tensor("linear.weight")
    loaded_projection = model[1].linear.weight.detach().cpu()
    projection_error = float((checkpoint_projection - loaded_projection).abs().max())

    query = model.encode(
        ["What is compound interest?"], is_query=True, output_value=None,
        show_progress_bar=False,
    )
    document = model.encode(
        ["Compound interest grows over time, with fees."], is_query=False,
        output_value=None, show_progress_bar=False,
    )
    query_tokens = model.tokenizer.convert_ids_to_tokens(query["input_ids"][0].tolist())
    document_tokens = model.tokenizer.convert_ids_to_tokens(document["input_ids"][0].tolist())
    query_vectors = np.asarray(query["token_embeddings"][0])
    document_vectors = np.asarray(document["token_embeddings"][0])
    document_mask = np.asarray(document["masks"][0])
    punctuation_positions = [
        index for index, token in enumerate(document_tokens) if token in {",", "."}
    ]

    checks = {
        "checkpoint_architecture": model[0].auto_model.config.architectures[0] == "HF_ColBERT",
        "projection_shape_128x768": tuple(loaded_projection.shape) == (128, 768),
        "projection_exact_checkpoint_match": torch.equal(
            checkpoint_projection, loaded_projection
        ),
        "query_prefix_unused0": model.query_prefix == "[unused0]",
        "document_prefix_unused1": model.document_prefix == "[unused1]",
        "query_prefix_inserted": query_tokens[1] == "[unused0]",
        "document_prefix_inserted": document_tokens[1] == "[unused1]",
        "query_length_32": model.query_length == 32 and len(query_tokens) == 32,
        "document_length_300": model.document_length == DOCUMENT_LENGTH,
        "query_mask_expansion": model.do_query_expansion
        and query_tokens.count("[MASK]") > 0,
        "expansion_attention_disabled": not model.attend_to_expansion_tokens,
        "document_punctuation_masked": bool(punctuation_positions)
        and all(not document_mask[index] for index in punctuation_positions),
        "embedding_dimension_128": query_vectors.shape[1] == 128
        and document_vectors.shape[1] == 128,
        "query_vectors_normalized": bool(
            np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=1e-5)
        ),
        "document_vectors_normalized": bool(
            np.allclose(np.linalg.norm(document_vectors, axis=1), 1.0, atol=1e-5)
        ),
    }
    report = {
        "model": args.model,
        "checkpoint_path": checkpoint_path,
        "projection_max_abs_error": projection_error,
        "query_prefix": model.query_prefix,
        "document_prefix": model.document_prefix,
        "query_length": model.query_length,
        "document_length": model.document_length,
        "do_query_expansion": model.do_query_expansion,
        "attend_to_expansion_tokens": model.attend_to_expansion_tokens,
        "checks": checks,
        "passed": all(checks.values()),
    }
    path = write_report(args.report_dir, "encoder-audit", report)
    print(path)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
