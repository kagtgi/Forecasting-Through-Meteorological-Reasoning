"""Regime-stratified skill (eval.md section 1B — the contribution evidence).

Skill is broken out by ASG regime label (init / grow / decay / steady). The hypothesis
(eval.md section B) is that the gain over vision-only / physics-only baselines concentrates
on the under-constrained regimes (initiation / growth); the honest-transparency requirement
is that *every* regime is reported, including steady-advection where black-box baselines are
expected to win.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from asgwm.asg.schema import REGIMES
from asgwm.eval import metrics as M


def stratify_by_regime(records: List[dict], regimes: Sequence[str] = REGIMES) -> Dict[str, List]:
    """Group evaluation records by their regime label.

    Args:
        records: each record is a dict with at least a ``regime`` key (one of ``regimes``);
                 typically also ``pred`` and ``obs`` fields.
        regimes: the regime vocabulary to bucket into.

    Returns:
        ``{regime: [records...]}`` for every regime in ``regimes`` (empty lists kept so the
        downstream table always has all columns).
    """
    out: Dict[str, List] = {r: [] for r in regimes}
    for rec in records:
        r = rec.get("regime", "steady")
        if r not in out:
            out[r] = []
        out[r].append(rec)
    return out


def regime_skill_table(
    preds: Sequence,
    obs: Sequence,
    regime_labels: Sequence[str],
    thresholds: Sequence[float],
) -> Dict[str, Dict[str, float]]:
    """Compute CSI / HSS / POD / FAR per regime at each threshold (eval.md section 1B).

    Args:
        preds:         list of predicted fields (numpy or torch), one per sample.
        obs:           list of observed fields, aligned with ``preds``.
        regime_labels: regime string per sample, aligned with ``preds``.
        thresholds:    dBZ thresholds (e.g. config ``eval.thresholds_dbz``).

    Returns:
        Nested dict ``{regime: {"csi@<thr>": v, "hss@<thr>": v, "pod@<thr>": v,
        "far@<thr>": v, ..., "n": count}}``. A regime with no samples reports zeros.
    """
    records = [
        {"pred": p, "obs": o, "regime": r}
        for p, o, r in zip(preds, obs, regime_labels)
    ]
    by_regime = stratify_by_regime(records)
    table: Dict[str, Dict[str, float]] = {}
    for regime, recs in by_regime.items():
        row: Dict[str, float] = {"n": float(len(recs))}
        for thr in thresholds:
            csis, hsss, pods, fars = [], [], [], []
            for rec in recs:
                csis.append(M.csi(rec["pred"], rec["obs"], thr))
                hsss.append(M.hss(rec["pred"], rec["obs"], thr))
                pods.append(M.pod(rec["pred"], rec["obs"], thr))
                fars.append(M.far(rec["pred"], rec["obs"], thr))
            row[f"csi@{thr:g}"] = float(np.mean(csis)) if csis else 0.0
            row[f"hss@{thr:g}"] = float(np.mean(hsss)) if hsss else 0.0
            row[f"pod@{thr:g}"] = float(np.mean(pods)) if pods else 0.0
            row[f"far@{thr:g}"] = float(np.mean(fars)) if fars else 0.0
        table[regime] = row
    return table
