"""Resumable Hugging Face and ModelScope model downloader."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class ModelSource:
    provider: str
    repo_id: str


def parse_model_source(source: str) -> ModelSource:
    """Parse a model URL or a ``namespace/name`` repository id.

    ModelScope URLs use ``/models/<id>`` while Hugging Face URLs use
    ``/models/<id>`` or simply ``/<id>``.  Bare ids are treated as Hugging
    Face ids, which is the least surprising default for scripts and CI.
    """
    source = source.strip()
    if not source:
        raise ValueError("model source cannot be empty")
    if "://" not in source:
        repo_id = source.strip("/").removesuffix(".git")
        if len(repo_id.split("/")) != 2:
            raise ValueError("a bare model id must have the form namespace/name")
        return ModelSource("huggingface", repo_id)
    parsed = urlparse(source)
    host = parsed.netloc.lower().split(":", 1)[0]
    parts = [p for p in parsed.path.split("/") if p]
    if "modelscope" in host:
        if parts and parts[0] in {"models", "api", "v1"}:
            if parts[0] == "models":
                parts = parts[1:]
            elif "models" in parts:
                parts = parts[parts.index("models") + 1 :]
        if len(parts) < 2:
            raise ValueError(f"cannot find a ModelScope model id in {source!r}")
        return ModelSource("modelscope", "/".join(parts[:2]).removesuffix(".git"))
    if "huggingface" in host or host in {"hf.co", "hf-mirror.com"}:
        if parts and parts[0] in {"models", "model"}:
            parts = parts[1:]
        if len(parts) < 2:
            raise ValueError(f"cannot find a Hugging Face model id in {source!r}")
        return ModelSource("huggingface", "/".join(parts[:2]).removesuffix(".git"))
    raise ValueError(f"unsupported model host {host!r}; use Hugging Face or ModelScope")


def download_model(source: str, output_dir: str | Path, revision: str | None = None,
                   hf_endpoint: str | None = None, token: str | None = None) -> str:
    """Download a model into ``output_dir`` and return its local path.

    Both backends keep their cache and partially downloaded files, so a
    rerun resumes naturally.  ``HF_ENDPOINT`` is set before importing the
    Hugging Face client; the default is the commonly used hf-mirror mirror.
    """
    parsed = parse_model_source(source)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if parsed.provider == "huggingface":
        if hf_endpoint:
            os.environ["HF_ENDPOINT"] = hf_endpoint
        else:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from huggingface_hub import snapshot_download

        kwargs = {"repo_id": parsed.repo_id, "local_dir": str(output)}
        if revision:
            kwargs["revision"] = revision
        if token:
            kwargs["token"] = token
        # snapshot_download has supported resumable local downloads for all
        # maintained hub versions.  Existing files are retained by default.
        snapshot_download(**kwargs)
    else:
        try:
            from modelscope import snapshot_download
        except ImportError as exc:
            raise RuntimeError("ModelScope download requires: pip install modelscope") from exc
        kwargs = {"model_id": parsed.repo_id, "local_dir": str(output)}
        if revision:
            kwargs["revision"] = revision
        # Older ModelScope releases do not accept every newer keyword, so use
        # only the stable arguments and let its cache resume partial files.
        snapshot_download(**kwargs)
    return str(output)


def _worker(args: argparse.Namespace) -> int:
    path = download_model(args.source, args.output_dir, args.revision,
                          args.hf_endpoint, args.token)
    print(f"Downloaded {args.source} -> {path}", flush=True)
    return 0


def _launch_background(argv: list[str], output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download.log"
    cmd = [sys.executable, "-m", "quant_eval.download", "--_worker", *argv]
    log = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, close_fds=True)
    log.close()
    (output_dir / "download.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    print(f"Started download in background (pid={proc.pid})")
    print(f"Log: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Resumable Hugging Face/ModelScope downloader")
    p.add_argument("source", help="model URL or repository id")
    p.add_argument("--output-dir", "-o", required=True, help="destination directory")
    p.add_argument("--revision")
    p.add_argument("--hf-endpoint", default=None, help="Hugging Face endpoint (default: hf-mirror.com)")
    p.add_argument("--token", default=None)
    p.add_argument("--foreground", action="store_true", help="run in this process")
    p.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args._worker or args.foreground:
        return _worker(args)
    # Strip launcher-only flags so the worker receives a clean argument set.
    forwarded = list(argv if argv is not None else sys.argv[1:])
    forwarded = [x for x in forwarded if x != "--foreground"]
    return _launch_background(forwarded, Path(args.output_dir).expanduser().resolve())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
