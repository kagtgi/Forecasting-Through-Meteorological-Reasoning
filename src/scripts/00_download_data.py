#!/usr/bin/env python
"""Download or synthesize the SEVIR subset and cache it (datasource.md sections 1-3).

Idempotent: re-running skips events already cached under ``paths.cache/events``. When
real SEVIR (s3fs + h5py + network) is unavailable, this materializes a deterministic
:class:`SyntheticSEVIR` subset so the WHOLE pipeline (label -> train -> eval) runs on
CPU with no download (interface contract; training_method.md section 6).

Usage:
    python scripts/00_download_data.py --config ../configs/default.yaml \
        --override data.n_train_events=64
"""
from __future__ import annotations

import argparse
import os
import sys

# Make `import asgwm` work when run from scripts/ (coding standards).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from asgwm.data import sevir  # noqa: E402
from asgwm.data import nexrad as nexrad_mod  # noqa: E402
from asgwm.data import mrms as mrms_mod  # noqa: E402
from asgwm.utils.config import load_config  # noqa: E402


def main() -> None:
    default_config = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    ap = argparse.ArgumentParser(description="Download/synthesize + cache the SEVIR subset.")
    ap.add_argument("--config", default=default_config, help="path to YAML config")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="key.subkey=value override (repeatable)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)

    # ensure cache directories exist (idempotent)
    for key in ("paths.cache", "paths.sevir_raw", "paths.root"):
        p = cfg.get_path(key)
        if p:
            os.makedirs(p, exist_ok=True)
    events_dir = sevir.events_dir(cfg)
    sevir.asg_dir(cfg)  # create asg cache dir for the labeling pass

    dataset = str(cfg.get_path("data.dataset", "sevir")).lower()
    print(f"[00_download_data] dataset: {dataset}")
    print(f"[00_download_data] cache events dir: {events_dir}")
    print(f"[00_download_data] requested events: {cfg.get_path('data.n_train_events')}")

    # Dispatch on the dataset. SEVIR/synthetic are the train/test source (with synthetic
    # fallback); NEXRAD and MRMS are out-of-distribution test sets (real data only, no
    # fallback). Each writes to its own dataset-namespaced cache dir (see sevir.events_dir).
    if dataset in ("sevir", "synthetic", "synth"):
        ids = sevir.download_sevir_subset(cfg)
    elif dataset == "nexrad":
        ids = nexrad_mod.download_nexrad_subset(cfg)
    elif dataset == "mrms":
        ids = mrms_mod.download_mrms_subset(cfg)
    else:
        raise ValueError(
            f"unknown data.dataset={dataset!r}; expected sevir|synthetic|nexrad|mrms"
        )

    n_on_disk = len([f for f in os.listdir(events_dir) if f.endswith(".npz")])
    print(f"[00_download_data] cached {len(ids)} event ids; {n_on_disk} npz files on disk")
    if ids[:3]:
        print(f"[00_download_data] sample ids: {ids[:3]}")
    print("[00_download_data] done. Next: python scripts/01_autolabel.py")


if __name__ == "__main__":
    main()
