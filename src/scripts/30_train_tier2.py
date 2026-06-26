#!/usr/bin/env python
"""Tier-2 end-to-end training CLI (training_method.md section 4).

Couples Stage A (VLM, stop-grad/low-LR) -> Stage B (transition) -> faithful bottleneck ->
Stage C (rectified-flow renderer), with scheduled sampling (oracle -> inferred ASG) and the
intervention-consistency loss. Checkpoint/resume safe for the <12 h A100 spot sessions.

Usage:
    python scripts/30_train_tier2.py --config ../configs/default.yaml \
        --vlm-ckpt <ph5_ckpt> --transition-ckpt <tier0_ckpt> \
        --override train.tier2.max_steps=10 --resume <tier2_ckpt>
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from asgwm.utils.config import load_config  # noqa: E402
from asgwm.train import tier2_endtoend  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-2: end-to-end A->B->bottleneck->C")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    p.add_argument("--override", action="append", default=[], help="key.subkey=value (repeatable)")
    p.add_argument("--vlm-ckpt", dest="vlm_ckpt", default=None, help="Tier-1 Ph-5 checkpoint (Stage A)")
    p.add_argument("--transition-ckpt", dest="transition_ckpt", default=None, help="Tier-0 transition checkpoint (Stage B)")
    p.add_argument("--resume", default=None, help="Tier-2 checkpoint to resume from")
    args = p.parse_args()

    cfg = load_config(args.config, args.override)

    print("[tier2] starting end-to-end training ...")
    print(f"[tier2]   vlm_ckpt={args.vlm_ckpt}  transition_ckpt={args.transition_ckpt}  resume={args.resume}")
    final = tier2_endtoend.train_tier2(
        cfg,
        vlm_ckpt=args.vlm_ckpt,
        transition_ckpt=args.transition_ckpt,
        resume=args.resume,
    )
    print(f"[tier2] Tier-2 checkpoint: {final}")
    print("[tier2] gate (run via scripts/41_eval_faithfulness.py): "
          "intervention consistency passes AND zeroed-ASG collapses to advection.")


if __name__ == "__main__":
    main()
