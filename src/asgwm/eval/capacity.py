"""Bottleneck capacity audit (training_method.md section 4; eval.md section 4).

The faithfulness-by-compression argument requires the ASG's channel capacity to be strictly
smaller than the raw radar input's. This module computes both, confirms ``asg_bits << input_bits``,
and provides a sweep over ``N_max`` to show how skill and capacity trade off (eval.md
``eval.capacity_sweep_nmax``).

ASG capacity (training_method.md section 4):
    N_max objects x (attribute entropy per object) x (prediction horizons / 1).
Per-object attribute entropy is the sum of the quantized fields' log2 alphabet sizes:
    - centroid (cy, cx): grid coords -> log2(grid) bits each.
    - area, peak:        coarse-quantized scalars.
    - motion (vy, vx):   quantized to ``motion_quant_kmh`` bins.
    - growth:            ``growth_sigfigs`` significant figures.
    - regime:            log2(4) categorical bits.
    - confidence:        coarse scalar.

Input capacity:
    H x W x k_frames x channels x pixel_entropy (bits/pixel for the quantized VIL range).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional

from asgwm.asg.schema import REGIMES


def _log2(x: float) -> float:
    return math.log2(x) if x > 0 else 0.0


def _attribute_bits(cfg) -> float:
    """Bits per storm object under the hard structural cap (training_method.md section 4)."""
    grid = float(cfg.get_path("data.grid", 384))
    motion_bin = float(cfg.get_path("asg.motion_quant_kmh", 8.0))
    growth_sig = int(cfg.get_path("asg.growth_sigfigs", 2))

    # Centroid: each coord lives on the grid (sub-pixel rounded to the cap -> ~grid levels).
    centroid_bits = 2.0 * _log2(grid)
    # Area: km^2 over the domain, coarse-quantized to ~64 levels (6 bits).
    area_bits = 6.0
    # Peak intensity: VIL->dBZ over a ~64 dBZ range at ~1 dBZ resolution (6 bits).
    peak_bits = 6.0
    # Motion components: speed range ~ +/-128 km/h quantized to `motion_bin` -> levels per axis.
    motion_range_kmh = 256.0
    motion_levels = max(motion_range_kmh / max(motion_bin, 1e-6), 1.0)
    motion_bits = 2.0 * _log2(motion_levels)
    # Growth: significant-figure quantization -> ~ growth_sig decimal digits of mantissa + sign.
    growth_bits = growth_sig * _log2(10.0) + 1.0
    # Regime: 4-class categorical.
    regime_bits = _log2(len(REGIMES))
    # Confidence: [0,1] at ~0.05 resolution (~20 levels, ~4.3 bits).
    conf_bits = _log2(20.0)

    return (centroid_bits + area_bits + peak_bits + motion_bits
            + growth_bits + regime_bits + conf_bits)


def capacity_bits(n_max: int, cfg) -> float:
    """Theoretical ASG channel capacity in bits for a given object budget ``n_max``.

    capacity = n_max * attribute_bits_per_object * horizons + global_bits, where ``horizons``
    is 1 (a single ASG_{t+h} per transition; the renderer reads one predicted state).

    Args:
        n_max: object budget.
        cfg:   loaded :class:`Config`.
    """
    attr_bits = _attribute_bits(cfg)
    horizons = 1.0
    # Global head: regime + n_objects counter (log2(n_max+1)).
    global_bits = _log2(len(REGIMES)) + _log2(n_max + 1)
    # Optional low-res growth field contributes a small, bounded term.
    growth_field_bits = 0.0
    if bool(cfg.get_path("bottleneck.use_growth_field", True)):
        gsz = float(cfg.get_path("asg.growth_field_size", 48))
        # Heavily quantized field: ~4 bits/cell, but it is a coarse H'xW' map.
        growth_field_bits = gsz * gsz * 4.0
    return float(n_max * attr_bits * horizons + global_bits + growth_field_bits)


def _input_bits(cfg) -> float:
    """Raw radar input channel capacity in bits (H x W x k x channels x bits/pixel)."""
    grid = float(cfg.get_path("data.grid", 384))
    k = float(cfg.get_path("data.in_frames", 13))
    channels = cfg.get_path("data.channels", ["vil", "ir069", "ir107", "glm"])
    n_ch = float(len(channels)) if channels else 1.0
    # VIL is quantized 0..255 -> 8 bits/pixel; other channels comparable.
    bits_per_pixel = 8.0
    return float(grid * grid * k * n_ch * bits_per_pixel)


def capacity_audit(cfg) -> Dict[str, object]:
    """Confirm ASG bits << raw input bits (training_method.md section 4 capacity audit).

    Returns:
        ``{"asg_bits","input_bits","ratio","ok","n_max","attr_bits_per_object"}`` where
        ``ok`` is True iff ``asg_bits < input_bits`` (the compression precondition for the
        faithfulness-by-compression argument).
    """
    n_max = int(cfg.get_path("asg.n_max", 16))
    asg_bits = capacity_bits(n_max, cfg)
    input_bits = _input_bits(cfg)
    ratio = asg_bits / input_bits if input_bits > 0 else float("inf")
    return {
        "asg_bits": float(asg_bits),
        "input_bits": float(input_bits),
        "ratio": float(ratio),
        "ok": bool(asg_bits < input_bits),
        "n_max": n_max,
        "attr_bits_per_object": float(_attribute_bits(cfg)),
    }


def capacity_sweep(
    cfg,
    train_fn: Optional[Callable] = None,
    eval_fn: Optional[Callable] = None,
) -> Dict[str, List]:
    """Sweep ``N_max`` and report capacity (and skill if a trainer/evaluator is supplied).

    For each ``N_max`` in ``cfg.eval.capacity_sweep_nmax``, compute the ASG capacity. If
    ``train_fn``/``eval_fn`` are provided they are called per-``N_max`` to produce a CSI value,
    showing the capacity/skill trade-off (eval.md section 4 capacity audit). With no callbacks,
    the CSI list is filled with NaNs (capacity-only audit).

    Args:
        cfg:      loaded :class:`Config`.
        train_fn: optional ``fn(cfg_with_nmax) -> ckpt`` to train at a given budget.
        eval_fn:  optional ``fn(cfg_with_nmax, ckpt) -> csi`` to evaluate skill.

    Returns:
        ``{"nmax": [...], "csi": [...], "bits": [...], "input_bits": float}``.
    """
    nmaxes = list(cfg.get_path("eval.capacity_sweep_nmax", [2, 4, 8, 16, 32]))
    bits, csis = [], []
    input_bits = _input_bits(cfg)
    for nm in nmaxes:
        b = capacity_bits(int(nm), cfg)
        bits.append(float(b))
        csi_val = float("nan")
        if train_fn is not None and eval_fn is not None:
            cfg.set_path("asg.n_max", int(nm))
            try:
                ckpt = train_fn(cfg)
                csi_val = float(eval_fn(cfg, ckpt))
            except Exception:
                csi_val = float("nan")
        csis.append(csi_val)
    return {"nmax": [int(n) for n in nmaxes], "csi": csis, "bits": bits,
            "input_bits": float(input_bits)}
