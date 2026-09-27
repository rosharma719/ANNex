"""Content-addressed caches for fixed and ragged benchmark embeddings."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from protocol import file_digest

CACHE_SCHEMA = 3
ENCODER_PACKAGES = ("numpy", "pylate", "sentence-transformers", "torch", "transformers")


@dataclass
class RaggedEmbeddings:
    values: np.ndarray
    offsets: np.ndarray

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.values[self.offsets[index] : self.offsets[index + 1]]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


def _package_versions():
    versions = {}
    for package in ENCODER_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _model_revision(model_id):
    try:
        from huggingface_hub import try_to_load_from_cache

        config = try_to_load_from_cache(model_id, "config.json")
        if isinstance(config, str):
            path = Path(config)
            if path.parent.parent.name == "snapshots":
                return path.parent.name
    except Exception:
        pass
    return "unresolved-main"


def _input_fingerprint(ids, texts):
    if len(ids) != len(texts):
        raise ValueError("embedding IDs and texts have different lengths")
    digest = hashlib.blake2b(digest_size=20)
    for item_id, text in zip(ids, texts):
        digest.update(item_id.encode() + b"\0" + text.encode() + b"\0")
    return digest.hexdigest()


def _identity(model_id, role, ids, texts, normalized, encoder_config=None):
    return {
        "schema": CACHE_SCHEMA,
        "model_id": model_id,
        "model_revision": _model_revision(model_id),
        "role": role,
        "normalized": normalized,
        "items": len(ids),
        "input_fingerprint": _input_fingerprint(ids, texts),
        "encoder_packages": _package_versions(),
        "encoder_config": encoder_config or {},
    }


def _location(root, kind, identity):
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.blake2b(encoded, digest_size=16).hexdigest()
    return root / f"{kind}-{key}", key


def _info(path, key, identity, hit):
    return {
        "hit": hit,
        "key": key,
        "path": str(path),
        "model_id": identity["model_id"],
        "model_revision": identity["model_revision"],
        "role": identity["role"],
        "files_sha256": json.loads((path / "manifest.json").read_text())[
            "files_sha256"
        ],
    }


def _verified_manifest(path, identity, files):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["identity"] != identity:
        raise RuntimeError(f"embedding cache identity mismatch: {path}")
    expected = manifest.get("files_sha256", {})
    if set(expected) != set(files) or any(
        file_digest(path / name) != expected[name] for name in files
    ):
        raise RuntimeError(f"embedding cache checksum mismatch: {path}")
    return manifest


def _write_manifest(path, manifest, files):
    if manifest["identity"]["model_revision"] == "unresolved-main":
        raise RuntimeError(
            "cannot persist embeddings with an unresolved model revision"
        )
    manifest["files_sha256"] = {name: file_digest(path / name) for name in files}
    temporary = path / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path / "manifest.json")


def ragged_fingerprint(root, model_id, role, ids, texts, encoder_config):
    """Verify bytes without loading arrays; bind them before freezing settings."""
    identity = _identity(model_id, role, ids, texts, True, encoder_config)
    if identity["model_revision"] == "unresolved-main":
        raise RuntimeError("warm a revision-resolved embedding cache before freezing")
    path, _ = _location(Path(root), "ragged", identity)
    manifest = _verified_manifest(path, identity, ["values.npy", "offsets.npy"])
    return {"identity": identity, "files_sha256": manifest["files_sha256"]}


def cached_ragged(
    root, model_id, role, ids, texts, encoder, refresh=False, encoder_config=None
):
    identity = _identity(
        model_id, role, ids, texts, normalized=True, encoder_config=encoder_config
    )
    path, key = _location(Path(root), "ragged", identity)
    manifest_path = path / "manifest.json"
    if manifest_path.exists() and not refresh:
        manifest = _verified_manifest(path, identity, ["values.npy", "offsets.npy"])
        values = np.load(path / "values.npy", mmap_mode="r", allow_pickle=False)
        offsets = np.load(path / "offsets.npy", mmap_mode="r", allow_pickle=False)
        if (
            values.dtype != np.float32
            or values.ndim != 2
            or values.shape[1] != manifest["dimension"]
            or offsets.dtype != np.int64
            or offsets.shape != (len(ids) + 1,)
            or offsets[0] != 0
            or offsets[-1] != len(values)
            or np.any(np.diff(offsets) <= 0)
        ):
            raise RuntimeError(f"invalid ragged cache shape/offsets: {path}")
        print(f"Embedding cache hit: {path}")
        return RaggedEmbeddings(values, offsets), _info(path, key, identity, True)

    encoded = [np.asarray(value, dtype=np.float32) for value in encoder()]
    if len(encoded) != len(ids) or any(
        value.ndim != 2 or not len(value) or not np.isfinite(value).all()
        for value in encoded
    ):
        raise RuntimeError("ragged encoder returned invalid embeddings")
    dimensions = {value.shape[1] for value in encoded}
    if len(dimensions) != 1:
        raise RuntimeError("ragged embeddings have inconsistent dimensions")
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(value) for value in encoded])
    values = np.concatenate(encoded, axis=0)
    # A cold encoder may have downloaded the checkpoint while encoding.
    identity["model_revision"] = _model_revision(model_id)
    path, key = _location(Path(root), "ragged", identity)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "values.npy", values, allow_pickle=False)
    np.save(path / "offsets.npy", offsets, allow_pickle=False)
    _write_manifest(
        path,
        {
            "identity": identity,
            "dimension": next(iter(dimensions)),
            "vectors": len(values),
        },
        ["values.npy", "offsets.npy"],
    )
    print(f"Embedding cache stored: {path}")
    return RaggedEmbeddings(values, offsets), _info(path, key, identity, False)


def cached_fixed(root, model_id, role, ids, texts, encoder, normalized, refresh=False):
    identity = _identity(model_id, role, ids, texts, normalized=normalized)
    path, key = _location(Path(root), "fixed", identity)
    manifest_path = path / "manifest.json"
    if manifest_path.exists() and not refresh:
        manifest = _verified_manifest(path, identity, ["values.npy"])
        values = np.load(path / "values.npy", mmap_mode="r", allow_pickle=False)
        if values.dtype != np.float32 or values.shape != (
            len(ids),
            manifest["dimension"],
        ):
            raise RuntimeError(f"invalid fixed cache shape: {path}")
        print(f"Embedding cache hit: {path}")
        return values, _info(path, key, identity, True)

    values = np.asarray(encoder(), dtype=np.float32)
    if values.ndim != 2 or len(values) != len(ids) or not np.isfinite(values).all():
        raise RuntimeError("fixed encoder returned invalid embeddings")
    # A cold encoder may have downloaded the checkpoint while encoding.
    identity["model_revision"] = _model_revision(model_id)
    path, key = _location(Path(root), "fixed", identity)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "values.npy", values, allow_pickle=False)
    _write_manifest(
        path, {"identity": identity, "dimension": values.shape[1]}, ["values.npy"]
    )
    print(f"Embedding cache stored: {path}")
    return values, _info(path, key, identity, False)
