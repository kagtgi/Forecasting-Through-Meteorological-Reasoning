"""Structured ASG interventions for the faithfulness suite (eval.md section C-i; architecture.md section 6).

A perturbation ``delta`` is applied to ``ASG_{t+h}`` and the renderer's field response is
checked against the *predicted* effect of ``delta``. This is counterfactual simulatability
(chen2023counterfactual) made architectural: because the renderer's only future-bearing path is
the ASG, a perturbation of the state must change the rendered field in the implied direction and
location.

Intervention kinds (eval.md section C, config ``eval.intervention_types``):
    - ``translate``     : shift a cell along its motion vector by ``km`` kilometres.
    - ``regime_flip``   : flip grow<->decay and flip the sign of the growth scalar.
    - ``growth_scale``  : scale the growth scalar (and peak tendency) by ``factor``.
    - ``motion_rotate`` : rotate every motion vector by ``deg`` degrees.

This module depends only on the ASG schema + numpy, so it imports without torch.
"""
from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import numpy as np

from asgwm.asg.schema import ASG, StormObject, REGIMES


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _clone(asg: ASG) -> ASG:
    """Deep copy of an ASG (objects + growth field) so edits never alias the source."""
    objs = [copy.deepcopy(o) for o in asg.objects]
    gf = None if asg.growth_field is None else np.array(asg.growth_field, dtype=np.float32, copy=True)
    return ASG(
        objects=objs,
        global_regime=asg.global_regime,
        growth_field=gf,
        context=dict(asg.context),
        meta=dict(asg.meta),
    )


def _km_per_pixel(asg: ASG, default: float = 1.0) -> float:
    """Recover the grid resolution from ASG meta if present (SEVIR VIL is 1 km/px)."""
    v = asg.meta.get("km_per_pixel", default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# perturbations
# ---------------------------------------------------------------------------
def perturb_asg(asg: ASG, kind: str, **kw) -> ASG:
    """Apply a structured perturbation to an ASG and return a new ASG.

    Args:
        asg:  the ASG to perturb (typically ``ASG_{t+h}``).
        kind: one of {``translate``, ``regime_flip``, ``growth_scale``, ``motion_rotate``}.
        kw:
            translate     -> ``km`` (float, distance along motion), ``obj_idx`` (int, default all).
            regime_flip   -> ``obj_idx`` (int, default all).
            growth_scale  -> ``factor`` (float), ``obj_idx`` (int, default all).
            motion_rotate -> ``deg`` (float), ``obj_idx`` (int, default all).

    Returns:
        a new, perturbed :class:`ASG` (the input is not mutated).
    """
    out = _clone(asg)
    obj_idx = kw.get("obj_idx", None)
    targets = range(len(out.objects)) if obj_idx is None else [obj_idx]
    kmpp = _km_per_pixel(asg)

    if kind == "translate":
        km = float(kw.get("km", 20.0))
        for i in targets:
            if i < 0 or i >= len(out.objects):
                continue
            o = out.objects[i]
            speed = math.hypot(o.vy, o.vx)
            if speed < 1e-6:
                # No motion -> translate due East as a defined fallback direction.
                uy, ux = 0.0, 1.0
            else:
                uy, ux = o.vy / speed, o.vx / speed
            shift_px = km / max(kmpp, 1e-6)
            o.cy = o.cy + uy * shift_px
            o.cx = o.cx + ux * shift_px

    elif kind == "regime_flip":
        for i in targets:
            if i < 0 or i >= len(out.objects):
                continue
            o = out.objects[i]
            if o.regime == "grow":
                o.regime = "decay"
            elif o.regime == "decay":
                o.regime = "grow"
            elif o.regime == "init":
                o.regime = "decay"
            else:  # steady -> grow (a defined, signed flip)
                o.regime = "grow"
            o.growth = -o.growth

    elif kind == "growth_scale":
        factor = float(kw.get("factor", 2.0))
        for i in targets:
            if i < 0 or i >= len(out.objects):
                continue
            o = out.objects[i]
            o.growth = o.growth * factor
        if out.growth_field is not None:
            out.growth_field = out.growth_field * factor

    elif kind == "motion_rotate":
        deg = float(kw.get("deg", 45.0))
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        for i in targets:
            if i < 0 or i >= len(out.objects):
                continue
            o = out.objects[i]
            # Rotate (vx, vy) in image coords by `deg` (positive = counter-clockwise on screen).
            vx_new = o.vx * cos_a - o.vy * sin_a
            vy_new = o.vx * sin_a + o.vy * cos_a
            o.vx, o.vy = vx_new, vy_new

    else:
        raise ValueError(
            f"unknown intervention kind {kind!r}; expected one of "
            f"translate|regime_flip|growth_scale|motion_rotate"
        )

    return out


def expected_effect(kind: str, **kw) -> Dict[str, float]:
    """Predicted field-level effect of a perturbation, for scoring C-i.

    Returns a dict with the keys the faithfulness scorer compares against the observed
    field difference:
        - ``displacement_px``  : expected centroid shift of the affected signal (pixels).
        - ``intensity_factor`` : expected multiplicative change in local intensity.
        - ``rotation_deg``     : expected rotation of the motion-induced displacement.
        - ``sign``             : expected sign of the local intensity change (+1/-1/0).

    Args:
        kind: the intervention kind.
        kw:   the same kwargs passed to :func:`perturb_asg` (``km``, ``factor``, ``deg``)
              plus optional ``km_per_pixel`` (default 1.0) for ``translate``.
    """
    kmpp = float(kw.get("km_per_pixel", 1.0))
    if kind == "translate":
        km = float(kw.get("km", 20.0))
        return {
            "displacement_px": km / max(kmpp, 1e-6),
            "intensity_factor": 1.0,
            "rotation_deg": 0.0,
            "sign": 0.0,
        }
    if kind == "regime_flip":
        # Flipping grow<->decay flips the sign of the growth scalar, so the local intensity
        # tendency reverses: a growing cell (growth>0) weakens, a decaying cell (growth<0)
        # intensifies. ``growth_sign`` is the sign of the *original* growth (default +1).
        gs = float(kw.get("growth_sign", 1.0))
        gs = 1.0 if gs >= 0 else -1.0
        return {
            "displacement_px": 0.0,
            "intensity_factor": 1.0,
            "rotation_deg": 0.0,
            "sign": -gs,  # tendency reverses relative to the original
        }
    if kind == "growth_scale":
        factor = float(kw.get("factor", 2.0))
        gs = float(kw.get("growth_sign", 1.0))
        gs = 1.0 if gs >= 0 else -1.0
        # Scaling amplifies the existing tendency: factor>1 strengthens it (in its own sign),
        # factor<1 weakens it (change opposes the original tendency sign).
        if factor > 1.0:
            sign = gs
        elif factor < 1.0:
            sign = -gs
        else:
            sign = 0.0
        return {
            "displacement_px": 0.0,
            "intensity_factor": factor,
            "rotation_deg": 0.0,
            "sign": sign,
        }
    if kind == "motion_rotate":
        deg = float(kw.get("deg", 45.0))
        return {
            "displacement_px": 0.0,
            "intensity_factor": 1.0,
            "rotation_deg": deg,
            "sign": 0.0,
        }
    raise ValueError(f"unknown intervention kind {kind!r}")


def intervention_pairs(asg: ASG, types: List[str]) -> List[Tuple[ASG, ASG, Dict]]:
    """Build (original, perturbed, meta) triples for each requested intervention type.

    For each type, a single default-parameterized perturbation is applied to the most
    intense object (index 0 after the ASG's intensity ordering). ``meta`` records the
    ``kind``, the kwargs used, the perturbed object index, and the
    :func:`expected_effect` dict, so the C-i scorer is self-contained.

    Args:
        asg:   the ASG to perturb.
        types: subset of ``eval.intervention_types``.

    Returns:
        list of ``(orig_asg, perturbed_asg, meta)`` triples.
    """
    pairs: List[Tuple[ASG, ASG, Dict]] = []
    if asg.n_objects == 0:
        return pairs
    kmpp = _km_per_pixel(asg)
    # Default kwargs per type (drawn from sensible, spec-aligned magnitudes).
    default_kw = {
        "translate": {"km": 20.0, "obj_idx": 0},
        "regime_flip": {"obj_idx": 0},
        "growth_scale": {"factor": 2.0, "obj_idx": 0},
        "motion_rotate": {"deg": 45.0, "obj_idx": 0},
    }
    for kind in types:
        kw = dict(default_kw.get(kind, {"obj_idx": 0}))
        idx = kw.get("obj_idx", 0)
        perturbed = perturb_asg(asg, kind, **kw)
        eff_kw = dict(kw)
        eff_kw["km_per_pixel"] = kmpp
        # The expected tendency-sign of regime_flip / growth_scale depends on the targeted
        # object's original growth sign (a growing cell weakens under flip; a decaying cell
        # intensifies). Supply it so expected_effect predicts the correct direction.
        if 0 <= idx < len(asg.objects):
            eff_kw["growth_sign"] = 1.0 if asg.objects[idx].growth >= 0 else -1.0
        eff = expected_effect(kind, **eff_kw)
        meta = {
            "kind": kind,
            "kwargs": kw,
            "obj_idx": idx,
            "expected": eff,
        }
        pairs.append((_clone(asg), perturbed, meta))
    return pairs
