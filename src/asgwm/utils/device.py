"""Device + mixed-precision helpers (CPU-safe; uses CUDA when available).

`device: auto` in the config picks CUDA when present, else CPU. `train.precision`
(bf16/fp16) drives autocast on CUDA; on CPU autocast is a no-op (it rarely helps and can
regress), so the CPU path is unchanged.
"""
from __future__ import annotations

import contextlib

import torch


def resolve_device(cfg=None) -> torch.device:
    pref = "auto"
    if cfg is not None:
        try:
            pref = str(cfg.get_path("device", "auto")).lower()
        except Exception:
            pref = "auto"
    if pref in ("", "auto", "none", "null"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(pref)


def amp_dtype(cfg=None):
    prec = "fp32"
    if cfg is not None:
        try:
            prec = str(cfg.get_path("train.precision", "fp32")).lower()
        except Exception:
            prec = "fp32"
    if prec == "bf16":
        return torch.bfloat16
    if prec in ("fp16", "half"):
        return torch.float16
    return None


def autocast_ctx(device: torch.device, cfg=None):
    """Mixed-precision context on CUDA; no-op on CPU."""
    dt = amp_dtype(cfg)
    if dt is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=dt)
