"""Checkpoint / resume utilities (training_method.md section 6).

The binding constraint for the project is wall-clock on preemptible <12 h A100 spot
sessions, not VRAM. Every training loop must therefore save optimizer + step + RNG
state every N steps and resume-from-step on start so a session can drop and chain to the
next from the bucket. These helpers implement exactly that contract:

    save_ckpt(path, step, model, optim=None, rng=True, extra=None)
    load_ckpt(path, model=None, optim=None) -> dict
    latest(dirpath) -> Optional[str]

Checkpoints are plain `torch.save` payloads so they are portable across the L4 / A100
sessions and the synthetic CPU smoke test.
"""
from __future__ import annotations

import os
import glob
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch


def _model_state(model) -> Optional[dict]:
    """Return a saveable state dict for `model`.

    Supports plain nn.Module, objects exposing `.save_adapters`/peft adapters via a
    `.state_dict()`, or None. Falls back to None if the model has no state.
    """
    if model is None:
        return None
    if hasattr(model, "state_dict"):
        try:
            return model.state_dict()
        except Exception:
            return None
    return None


def _capture_rng() -> Dict[str, object]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def _restore_rng(state: Dict[str, object]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
    except Exception:
        pass
    try:
        np.random.set_state(state["numpy"])
    except Exception:
        pass
    try:
        torch.set_rng_state(_to_byte_tensor(state["torch"]))
    except Exception:
        pass
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except Exception:
            pass


def _to_byte_tensor(x) -> torch.Tensor:
    """torch RNG state must be a ByteTensor; coerce loaded payloads back to that."""
    if isinstance(x, torch.Tensor):
        return x.to(torch.uint8) if x.dtype != torch.uint8 else x
    return torch.as_tensor(x, dtype=torch.uint8)


def save_ckpt(
    path: str,
    step: int,
    model=None,
    optim=None,
    rng: bool = True,
    extra: Optional[Dict] = None,
) -> str:
    """Save step + model + optimizer (+ RNG + extra) to `path` (training_method.md s.6).

    The parent directory is created if absent. Returns the written path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload: Dict[str, object] = {
        "step": int(step),
        "model": _model_state(model),
        "optim": optim.state_dict() if optim is not None else None,
        "rng": _capture_rng() if rng else None,
        "extra": dict(extra) if extra else {},
    }
    # Write atomically: write to a temp file then replace, so a preemption mid-write does
    # not corrupt the latest good checkpoint.
    tmp = path + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return path


def load_ckpt(path: str, model=None, optim=None) -> Dict:
    """Load a checkpoint and (optionally) restore model / optimizer / RNG.

    Returns the full payload dict (so callers can read `step` and `extra`). Missing
    sub-states are tolerated so partially-saved tier checkpoints still resume.
    """
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None and payload.get("model") is not None:
        try:
            model.load_state_dict(payload["model"], strict=False)
        except Exception:
            # Tolerate architecture drift across tiers; load what matches.
            pass
    if optim is not None and payload.get("optim") is not None:
        try:
            optim.load_state_dict(payload["optim"])
        except Exception:
            pass
    if payload.get("rng"):
        _restore_rng(payload["rng"])
    return payload


_STEP_RE = re.compile(r"(?:step|ckpt)[_-]?(\d+)")


def latest(dirpath: str) -> Optional[str]:
    """Return the most recent checkpoint path in `dirpath`, or None.

    Prefers the highest embedded step number (e.g. ``ckpt_step_01000.pt``); falls back to
    mtime if no step is encoded in the filename.
    """
    if not dirpath or not os.path.isdir(dirpath):
        return None
    cands: List[str] = sorted(glob.glob(os.path.join(dirpath, "*.pt")))
    if not cands:
        return None

    def _key(p: str):
        m = _STEP_RE.search(os.path.basename(p))
        step = int(m.group(1)) if m else -1
        return (step, os.path.getmtime(p))

    return max(cands, key=_key)
