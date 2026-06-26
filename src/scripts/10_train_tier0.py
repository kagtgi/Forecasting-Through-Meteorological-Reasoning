#!/usr/bin/env python
"""Tier-0 training CLI (training_method.md section 2).

Trains the ASG->ASG transition transformer and the deterministic renderer, then runs the
go/no-go gate (transition vs persistence AND vs future-blind advection) and prints the
result. Checkpoint/resume safe.

Usage:
    python scripts/10_train_tier0.py --config ../configs/default.yaml \
        --override train.tier0.max_steps=10 --resume <ckpt>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Make `import asgwm` work when run from scripts/.
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from asgwm.utils.config import load_config  # noqa: E402
from asgwm.train import tier0  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Tier-0: transition + deterministic renderer + gate")
    p.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    p.add_argument("--override", action="append", default=[], help="key.subkey=value (repeatable)")
    p.add_argument("--resume", default=None, help="checkpoint to resume the transition run from")
    p.add_argument("--skip-renderer", action="store_true", help="skip the deterministic renderer run")
    args = p.parse_args()

    cfg = load_config(args.config, args.override)

    print("[tier0] training transition transformer ...")
    trans_ckpt = tier0.train_transition(cfg, resume=args.resume)
    print(f"[tier0] transition checkpoint: {trans_ckpt}")

    if not args.skip_renderer:
        print("[tier0] training deterministic renderer (oracle ASG) ...")
        rend_ckpt = tier0.train_deterministic_renderer(cfg)
        print(f"[tier0] renderer checkpoint: {rend_ckpt}")

    print("[tier0] running gate check (vs persistence & advection) ...")
    gate = tier0.gate_check(cfg)
    print("[tier0] GATE RESULTS:")
    print(json.dumps(gate, indent=2))
    passed = gate["beats_persistence"] and gate["beats_advection"]
    print(f"[tier0] gate {'PASSED' if passed else 'NOT PASSED'} "
          f"(beats_persistence={gate['beats_persistence']}, "
          f"beats_advection={gate['beats_advection']})")


if __name__ == "__main__":
    main()
