"""Device + mixed-precision helpers (CPU-safe; uses CUDA when available).

`device: auto` in the config picks CUDA when present, else CPU. `train.precision`
(bf16/fp16) drives autocast on CUDA; on CPU autocast is a no-op (it rarely helps and can
regress), so the CPU path is unchanged.
"""
from __future__ import annotations

import contextlib

import torch


_PERF_ENABLED = False


def enable_perf() -> None:
    """Enable TF32 matmuls + cuDNN autotuning on CUDA (accuracy-neutral A100 speedups).

    TF32 uses the A100 tensor cores for fp32 matmuls/convs at ~no accuracy cost for training,
    and ``cudnn.benchmark`` autotunes conv algorithms for our fixed patch shapes. Idempotent and
    a no-op off CUDA. Called automatically by :func:`resolve_device` when a CUDA device is chosen.
    """
    global _PERF_ENABLED
    if _PERF_ENABLED or not torch.cuda.is_available():
        return
    try:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    except Exception:  # pragma: no cover - defensive on exotic builds
        pass
    _PERF_ENABLED = True


def resolve_device(cfg=None) -> torch.device:
    pref = "auto"
    if cfg is not None:
        try:
            pref = str(cfg.get_path("device", "auto")).lower()
        except Exception:
            pref = "auto"
    if pref in ("", "auto", "none", "null"):
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif pref.startswith("cuda") and not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device(pref)
    if dev.type == "cuda":
        enable_perf()
    return dev


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
