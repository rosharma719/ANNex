"""Minimal local .env loader for benchmark credentials.

Existing environment variables always win, so CI and shell-provided secrets are
never replaced by a checked-out .env file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_env() -> None:
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))
