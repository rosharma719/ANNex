"""One authoritative ColBERTv2 encoding configuration for every benchmark."""
from pylate import models

MODEL_ID = "colbert-ir/colbertv2.0"
QUERY_LENGTH = 32
DOCUMENT_LENGTH = 300


def cache_config(role):
    return {
        "query_length": QUERY_LENGTH,
        "document_length": DOCUMENT_LENGTH,
        "query_expansion": True,
        "attend_to_expansion_tokens": False,
        "query_prefix": "[unused0]",
        "document_prefix": "[unused1]",
        "punctuation_skiplist": True,
        "role": role,
    }


def load(model_id=MODEL_ID):
    return models.ColBERT(
        model_name_or_path=model_id,
        query_length=QUERY_LENGTH,
        document_length=DOCUMENT_LENGTH,
        do_query_expansion=True,
        attend_to_expansion_tokens=False,
    )
