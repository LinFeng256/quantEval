"""Helpers for the NLTK data files required by LM Evaluation Harness tasks."""

from __future__ import annotations

import hashlib
import io
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


_PUNKT_TAB_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/"
    "packages/tokenizers/punkt_tab.zip"
)
# SHA-256 published in the official nltk_data package index.
_PUNKT_TAB_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"


def _download_punkt_tab(nltk_module: Any) -> None:
    try:
        import requests
    except ImportError as exc:
        raise RuntimeError("Downloading punkt_tab requires the requests package") from exc

    try:
        response = requests.get(_PUNKT_TAB_URL, timeout=(10, 120), allow_redirects=False)
        response.raise_for_status()
        archive = response.content
    except Exception as exc:
        raise RuntimeError(
            "Could not download the official NLTK punkt_tab archive. Check network access "
            "to raw.githubusercontent.com and retry."
        ) from exc

    digest = hashlib.sha256(archive).hexdigest()
    if digest != _PUNKT_TAB_SHA256:
        raise RuntimeError(
            "The downloaded punkt_tab archive failed its SHA-256 integrity check; "
            "it was not installed."
        )

    data_paths = getattr(nltk_module.data, "path", None) or [Path.home() / "nltk_data"]
    tokenizer_root = Path(data_paths[0]).expanduser() / "tokenizers"
    tokenizer_root.mkdir(parents=True, exist_ok=True)
    destination = tokenizer_root / "punkt_tab"

    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as package:
            members = [PurePosixPath(member) for member in package.namelist()]
            if any(path.is_absolute() or ".." in path.parts for path in members):
                raise RuntimeError("The punkt_tab archive contains an unsafe file path")
            if not any(path.parts[:2] == ("punkt_tab", "english") for path in members):
                raise RuntimeError("The punkt_tab archive does not contain English tokenizer data")
            with tempfile.TemporaryDirectory(prefix="punkt_tab-", dir=tokenizer_root) as staging:
                package.extractall(staging)
                extracted = Path(staging) / "punkt_tab"
                if not (extracted / "english").is_dir():
                    raise RuntimeError("The extracted punkt_tab archive is incomplete")
                if not destination.exists():
                    os.replace(extracted, destination)
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The downloaded punkt_tab archive is invalid") from exc


def ensure_punkt_tab(nltk_module: Any = None) -> None:
    """Ensure NLTK's English word-tokenizer data exists, downloading if needed."""
    if nltk_module is None:
        try:
            import nltk as nltk_module
        except ImportError as exc:
            raise RuntimeError(
                "IFEval requires NLTK; install LM Evaluation Harness dependencies first"
            ) from exc

    resource = "tokenizers/punkt_tab/english/"
    try:
        nltk_module.data.find(resource)
        return
    except LookupError:
        pass

    _download_punkt_tab(nltk_module)
    try:
        nltk_module.data.find(resource)
    except LookupError as exc:
        raise RuntimeError(
            "punkt_tab was downloaded and verified but NLTK still cannot find its English "
            "tokenizer files. Check the NLTK_DATA directory configuration."
        ) from exc
