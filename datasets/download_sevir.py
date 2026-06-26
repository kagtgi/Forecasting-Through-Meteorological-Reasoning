#!/usr/bin/env python
"""Thin CLI wrapper around ``asgwm.data.sevir.download_sevir_subset`` (datasets/README.md).

SEVIR is the primary train+test source (``s3://sevir``, anonymous). This wrapper lives in
``datasets/`` and keeps all downloaded data under ``datasets/sevir/`` so the raw HDF5 slice
and the cached events sit next to this tooling (the whole ``datasets/`` tree is gitignored).
It adds the repo's ``src/`` to ``sys.path``, builds a config with SEVIR-local paths, and
calls the real downloader (which transparently falls back to SyntheticSEVIR unless
``--require-real`` is set). See datasets/README.md for the full data-acquisition guide.

Usage:
    python datasets/download_sevir.py --n-events 64
    python datasets/download_sevir.py --n-events 2500 --require-real
"""
from __future__ import annotations

import argparse
import os
import sys

# This file is datasets/download_sevir.py -> repo root is its parent's parent.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from asgwm.data import sevir  # noqa: E402
from asgwm.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download/synthesize + cache the SEVIR subset.")
    ap.add_argument("--n-events", type=int, default=64, help="number of events to cache")
    ap.add_argument("--require-real", action="store_true",
                    help="forbid the SyntheticSEVIR fallback (raise on real-data failure)")
    args = ap.parse_args()

    config = os.path.join(_SRC, "configs", "default.yaml")
    overrides = [
        "data.dataset=sevir",
        f"paths.cache={os.path.join('datasets', 'sevir', 'cache')}",
        f"paths.sevir_raw={os.path.join('datasets', 'sevir', 'raw')}",
        f"data.n_train_events={args.n_events}",
        f"data.require_real={'true' if args.require_real else 'false'}",
    ]
    cfg = load_config(config, overrides)

    ids = sevir.download_sevir_subset(cfg)
    print(f"[download_sevir] cached {len(ids)} event ids")
    print(f"[download_sevir] events dir: {sevir.events_dir(cfg)}")


if __name__ == "__main__":
    main()
