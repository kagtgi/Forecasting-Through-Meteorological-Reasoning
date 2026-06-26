#!/usr/bin/env python
"""Thin CLI wrapper around ``asgwm.data.mrms.download_mrms_subset`` (datasets/README.md).

MRMS MergedReflectivityQCComposite is used ENTIRELY as an out-of-distribution test set (no
training, no split). Real data only: there is no synthetic fallback for an OOD set, so
missing deps / network / empty result either return ``[]`` or raise (with ``--require-real``).
This wrapper adds the repo's ``src/`` to ``sys.path`` and routes cached events under
``datasets/mrms/cache``.

The OOD CASES (date/start/center lat,lon) come from the config: ``data.mrms.cases`` in
src/configs/default.yaml (null -> built-in cases). The AWS archive starts 2020-10-14, so
pick cases on/after that date. Bucket: s3://noaa-mrms-pds (us-east-1, anonymous).
See datasets/README.md for the full guide.

Usage:
    python datasets/download_mrms.py
    python datasets/download_mrms.py --require-real
"""
from __future__ import annotations

import argparse
import os
import sys

# This file is datasets/download_mrms.py -> repo root is its parent's parent.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from asgwm.data import mrms  # noqa: E402
from asgwm.data import sevir  # noqa: E402  (events_dir helper is shared)
from asgwm.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download + cache the MRMS OOD subset.")
    ap.add_argument("--require-real", action="store_true",
                    help="raise instead of returning [] when real data is unavailable")
    args = ap.parse_args()

    config = os.path.join(_SRC, "configs", "default.yaml")
    overrides = [
        "data.dataset=mrms",
        f"paths.cache={os.path.join('datasets', 'mrms', 'cache')}",
        f"data.require_real={'true' if args.require_real else 'false'}",
    ]
    cfg = load_config(config, overrides)

    ids = mrms.download_mrms_subset(cfg)
    print(f"[download_mrms] cached {len(ids)} event ids")
    print(f"[download_mrms] events dir: {sevir.events_dir(cfg)}")


if __name__ == "__main__":
    main()
