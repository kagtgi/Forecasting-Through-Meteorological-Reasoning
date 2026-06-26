#!/usr/bin/env python
"""Run the ASG auto-labeling pass over cached events (datasource.md section 2).

Reads cached events via ``asgwm.data.sevir.iter_events`` and writes one ASG-pair
JSON per event to ``paths.cache/asg/``. This is the "run the pysteps labeling pass
once (CPU, slow) and freeze it" step (datasource.md section 3): the produced ASGs
+ the shared future-blind motion field are cached and reused by every downstream
stage. Idempotent — events whose ASG JSON already exists are skipped unless
``--force`` is given.

Usage:
    python scripts/01_autolabel.py --config ../configs/default.yaml \
        --override data.n_train_events=64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Make `import asgwm` work when run from scripts/ (coding standards).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from asgwm.labeling import pipeline  # noqa: E402
from asgwm.utils.config import load_config  # noqa: E402


def _asg_dir(cfg) -> str:
    """Resolve paths.cache/asg, preferring sevir.asg_dir(cfg) if it exists."""
    try:
        from asgwm.data import sevir  # local import: data module may be WIP

        if hasattr(sevir, "asg_dir"):
            d = sevir.asg_dir(cfg)
            os.makedirs(d, exist_ok=True)
            return d
    except Exception:
        pass
    cache = cfg.get_path("paths.cache", "./artifacts/cache")
    d = os.path.join(cache, "asg")
    os.makedirs(d, exist_ok=True)
    return d


def _event_id(event: dict, idx: int) -> str:
    return str(event.get("id", event.get("event_id", f"event_{idx:06d}")))


def _seq_to_json(seq) -> dict:
    """Serialize an ASGSequence to a JSON-able dict (uses ASG.to_dict)."""
    return {
        "event_id": seq.event_id,
        "horizon_min": int(seq.horizon_min),
        "asg_t": seq.asg_t.to_dict(),
        "asg_th": seq.asg_th.to_dict(),
    }


def main() -> None:
    default_config = os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml")
    ap = argparse.ArgumentParser(description="ASG auto-labeling pass over cached events.")
    ap.add_argument("--config", default=default_config, help="path to YAML config")
    ap.add_argument(
        "--override",
        action="append",
        default=[],
        help="key.subkey=value override (repeatable)",
    )
    ap.add_argument("--force", action="store_true", help="re-label even if ASG JSON exists")
    ap.add_argument("--limit", type=int, default=0, help="label at most N events (0 = all)")
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)

    # Lazy import so this script imports cleanly even if the data module is WIP.
    from asgwm.data import sevir  # noqa: E402

    out_dir = _asg_dir(cfg)
    print(f"[01_autolabel] writing ASG JSON to: {out_dir}")

    n_done = 0
    n_skip = 0
    n_fail = 0
    t0 = time.time()
    for idx, event in enumerate(sevir.iter_events(cfg)):
        if args.limit and n_done + n_skip >= args.limit:
            break
        eid = _event_id(event, idx)
        out_path = os.path.join(out_dir, f"{eid}.json")
        if os.path.exists(out_path) and not args.force:
            n_skip += 1
            continue
        try:
            seq = pipeline.autolabel_event(event, cfg)
            if not seq.event_id:
                seq.event_id = eid
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(_seq_to_json(seq), f)
            n_done += 1
            if n_done <= 3 or n_done % 50 == 0:
                nt = seq.asg_t.n_objects
                nth = seq.asg_th.n_objects
                print(f"[01_autolabel] {eid}: ASG_t={nt} obj, ASG_t+h={nth} obj")
        except Exception as e:  # never let one bad event halt the freeze pass
            n_fail += 1
            print(f"[01_autolabel] WARN failed on {eid}: {type(e).__name__}: {e}")

    dt = time.time() - t0
    print(
        f"[01_autolabel] done: labeled {n_done}, skipped {n_skip}, failed {n_fail} "
        f"in {dt:.1f}s -> {out_dir}"
    )
    print("[01_autolabel] next: python scripts/10_train_tier0.py")


if __name__ == "__main__":
    main()
