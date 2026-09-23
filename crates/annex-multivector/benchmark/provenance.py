"""Stable provenance attached to every committed benchmark report."""
from __future__ import annotations

import importlib.metadata
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("ir-datasets", "numpy", "pylate", "sentence-transformers")


def command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version() -> str:
    manifest = (ROOT / "Cargo.toml").read_text()
    match = re.search(r'^version\s*=\s*"([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?)"', manifest, re.MULTILINE)
    if match is None:
        raise RuntimeError("Cargo.toml must contain a SemVer package version")
    return match.group(1)


def provenance() -> dict:
    versions = {}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "multivector_version": package_version(),
        "git_revision": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "rustc_version": command("rustc", "--version"),
        "python_packages": versions,
        "models": {
            "late_interaction": "colbert-ir/colbertv2.0",
            "single_vector": "sentence-transformers/all-MiniLM-L6-v2",
        },
    }


def write_report(report_dir: Path, kind: str, report: dict) -> Path:
    """Append one immutable record to the ledger for the current SemVer version."""
    version = package_version()
    path = report_dir / f"v{version}.jsonl"
    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "provenance": provenance(),
        **report,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as ledger:
        ledger.write(json.dumps(record, sort_keys=True) + "\n")
    return path
