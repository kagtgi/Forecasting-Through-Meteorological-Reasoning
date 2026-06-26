#!/usr/bin/env python
"""Thin CLI wrapper around ``asgwm.data.nexrad.download_nexrad_subset`` (datasets/README.md).

NEXRAD Level II is used ENTIRELY as an out-of-distribution test set (no training, no split).
Real data only: there is no synthetic fallback for an OOD set, so missing deps / network /
empty result either return ``[]`` or raise (with ``--require-real``). This wrapper adds the
repo's ``src/`` to ``sys.path`` and routes all cached events under ``datasets/nexrad/cache``.

The OOD CASES (station/date/start windows) come from the config: ``data.nexrad.cases`` in
src/configs/default.yaml (null -> built-in severe-weather cases). Edit that to add cases.
Bucket: s3://unidata-nexrad-level2 (us-east-1, anonymous). See datasets/README.md.

Usage:
    python datasets/download_nexrad.py
    python datasets/download_nexrad.py --require-real
"""
from __future__ import annotations

import argparse
import os
import sys

# This file is datasets/download_nexrad.py -> repo root is its parent's parent.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from asgwm.data import nexrad  # noqa: E402
from asgwm.data import sevir  # noqa: E402  (events_dir helper is shared)
from asgwm.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + cache the NEXRAD OOD subset.")
    ap.add_argument("--require-real", action="store_true",
                    help="raise instead of returning [] when real data is unavailable")
    args = ap.parse_args()

    config = os.path.join(_SRC, "configs", "default.yaml")
    overrides = [
        "data.dataset=nexrad",
        f"paths.cache={os.path.join('datasets', 'nexrad', 'cache')}",
        f"data.require_real={'true' if args.require_real else 'false'}",
    ]
    cfg = load_config(config, overrides)

    ids = nexrad.download_nexrad_subset(cfg)
    print(f"[download_nexrad] cached {len(ids)} event ids")
    print(f"[download_nexrad] events dir: {sevir.events_dir(cfg)}")


if __name__ == "__main__":
    main()
