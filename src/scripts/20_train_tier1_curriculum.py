#!/usr/bin/env python
"""Tier-1 VLM curriculum CLI (training_method.md section 3, architecture.md section 10).

Runs the five-phase curriculum ph1_vqa -> ... -> ph5_eqcot sequentially. AFTER ph3_asg the
hard ASG-F1 gate is enforced: if F1 < cfg.train.tier1.ph3_gate_f1 the run raises (the gate
is the hard go/no-go before the unfounded downstream CoT). Each phase is checkpoint/resume
safe and resumes from the previous phase's checkpoint.

Usage:
    python scripts/20_train_tier1_curriculum.py --config ../configs/default.yaml \
        --override train.tier1.steps_per_phase.ph1_vqa=5
    # run a single phase:
    python scripts/20_train_tier1_curriculum.py --phase ph1_vqa --resume <ckpt_in>
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from asgwm.utils.config import load_config  # noqa: E402
from asgwm.train import tier1_curriculum  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-1: five-phase VLM curriculum with Ph-3 gate")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    p.add_argument("--override", action="append", default=[], help="key.subkey=value (repeatable)")
    p.add_argument("--resume", default=None, help="ckpt_in for a single-phase run")
    p.add_argument("--phase", default=None, choices=tier1_curriculum.PHASES,
                   help="run only this phase (omit to run the full curriculum)")
    args = p.parse_args()

    cfg = load_config(args.config, args.override)

    if args.phase:
        print(f"[tier1] running single phase {args.phase} ...")
        out = tier1_curriculum.run_phase(cfg, args.phase, args.resume)
        print(f"[tier1] phase {args.phase} checkpoint: {out}")
        return

    print("[tier1] running full curriculum (ph1 -> ph5) with Ph-3 gate ...")
    try:
        final = tier1_curriculum.run_curriculum(cfg)
    except RuntimeError as e:
        print(f"[tier1] CURRICULUM HALTED: {e}")
        sys.exit(2)
    print(f"[tier1] Tier-1 deliverable (Ph-5 checkpoint = Tier-2 init): {final}")


if __name__ == "__main__":
    main()
