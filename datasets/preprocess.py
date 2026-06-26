#!/usr/bin/env python
"""Verify the canonical representation across datasets (datasets/README.md).

Iterates the cached events for one ``--dataset`` (sevir | synthetic | nexrad | mrms) via
``asgwm.data.sevir.iter_events`` and runs ``asgwm.data.normalize.assert_canonical`` on each
event's ``vil`` array, then prints a small summary table. This PROVES that SEVIR, NEXRAD,
and MRMS all reduce to the identical ``[T,384,384]`` VIL-byte stack with values in [0,255]
(255 = missing) — the invariant that lets a SEVIR-trained model run on the OOD sets
unmodified. Pure verification: it reads the cache, never downloads.

Usage:
    python datasets/preprocess.py --dataset sevir
    python datasets/preprocess.py --dataset nexrad
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# This file is datasets/preprocess.py -> repo root is its parent's parent.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from asgwm.data import sevir  # noqa: E402
from asgwm.data import normalize as norm  # noqa: E402
from asgwm.utils.config import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify canonical [T,384,384] VIL-byte events.")
    ap.add_argument("--dataset", choices=["sevir", "synthetic", "nexrad", "mrms"],
                    default="sevir", help="which cached dataset to verify")
    args = ap.parse_args()

    config = os.path.join(_SRC, "configs", "default.yaml")
    overrides = [
        f"data.dataset={args.dataset}",
        f"paths.cache={os.path.join('datasets', args.dataset, 'cache')}",
    ]
    cfg = load_config(config, overrides)

    n = 0
    shapes = set()
    gmin, gmax, miss_frac_sum = 255.0, 0.0, 0.0
    for ev in sevir.iter_events(cfg):
        vil = np.asarray(ev["vil"], dtype=np.float32)
        norm.assert_canonical(vil)  # raises on shape/range violation
        shapes.add(tuple(vil.shape))
        finite = vil[np.isfinite(vil)]
        if finite.size:
            gmin = min(gmin, float(finite.min()))
            gmax = max(gmax, float(finite.max()))
        miss_frac_sum += float(np.mean(vil == norm.VIL_BYTE_MISSING))
        n += 1

    print(f"dataset           : {args.dataset}")
    print(f"events dir        : {sevir.events_dir(cfg)}")
    print(f"n events          : {n}")
    if n == 0:
        print("(no cached events; run the matching datasets/download_*.py first)")
        return
    print(f"shape(s)          : {sorted(shapes)}")
    print(f"byte min / max    : {gmin:.1f} / {gmax:.1f}")
    print(f"% missing (mean)  : {100.0 * miss_frac_sum / n:.3f}")
    print("OK: all events are canonical [T,384,384] VIL byte in [0,255].")


if __name__ == "__main__":
    main()
