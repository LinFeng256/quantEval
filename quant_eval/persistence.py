"""Small durable-file helpers shared by benchmark runners."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def prepare_harness_cache(path: str | Path) -> Path:
    """Prepare a stable SQLite cache prefix for LM Evaluation Harness."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def protocol_cache_path(path: str | Path, protocol: dict[str, Any]) -> Path:
    """Derive a cache prefix that cannot be reused across model protocols."""
    base = Path(path).expanduser().resolve()
    fingerprint = hashlib.sha256(
        json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]
    return prepare_harness_cache(base.with_name(f"{base.name}.{fingerprint}"))


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
