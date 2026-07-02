"""Knowledge-category ablation hooks (eval.md section D.1, philosophy.md section 3.1)."""
from __future__ import annotations

from typing import Dict


def apply_knowledge_ablation(context: Dict, cfg) -> Dict:
    """Zero-out context channels per ``eval.knowledge_ablation`` flags.

    Stub hook: called before forecast assembly when a Tier-2 checkpoint exists.
    Until experiments run, ablation rows remain [TBR].
    """
    flags = cfg.get_path("eval.knowledge_ablation", {}) or {}
    ctx = dict(context)
    if not flags.get("seasonal", True):
        ctx.pop("month", None)
        ctx.pop("season", None)
    if not flags.get("geographic", True):
        for k in ("dem", "coastline", "topo"):
            ctx.pop(k, None)
    if not flags.get("diurnal", True):
        ctx.pop("solar_angle", None)
        ctx.pop("hour_local", None)
    if not flags.get("synoptic", True):
        for k in ("cape", "cin", "shear", "pwat"):
            ctx.pop(k, None)
    return ctx
