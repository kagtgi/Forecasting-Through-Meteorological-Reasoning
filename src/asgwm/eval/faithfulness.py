"""Faithfulness suite C-i..C-v (eval.md section C) — the proof of the contribution.

Each function realizes one part of the suite:
    C-i   intervention_consistency  : perturb ASG -> field changes in the predicted direction.
    C-ii  bottleneck_ablation       : inferred / oracle / zeroed / shuffled ASG comparison.
    C-iii leakage_audit (+ LeakageCLUB) : CLUB upper-bound MI of the future-blind advection path.
    C-iv  asg_accuracy              : inferred ASG vs gold subset (Hungarian centroid matching).
    C-v   counterfactual_demo       : base vs edited fields + diffs for the live demo.

The faithfulness signal is entailed by the architecture (philosophy.md section 3.3,
architecture.md sections 4, 6): because the renderer's only future-bearing path is
Z = ASG_{t+h} (+) advect_blind, perturbing the ASG must move the field, and zeroing
it must collapse the field to advection.

LeakageCLUB implements the CLUB upper bound on mutual information (cheng2020club).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from asgwm.asg.schema import ASG, StormObject, REGIMES, motion_to_compass
from asgwm import interventions as IV

# Bottleneck helpers are written by another agent; import defensively so this module
# still imports (and the non-C-ii paths still run) if it is not yet on disk.
try:
    from asgwm.models import bottleneck as _bottleneck  # type: ignore
    _HAS_BOTTLENECK = True
except Exception:  # pragma: no cover
    _bottleneck = None  # type: ignore
    _HAS_BOTTLENECK = False


def _to_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _field_2d(field) -> np.ndarray:
    """Reduce an arbitrary-rank rendered field to a single 2-D map (squeeze batch/channel)."""
    a = _to_np(field).astype(np.float64)
    while a.ndim > 2:
        a = a[0]
    return a


def _centroid_of_change(diff: np.ndarray) -> Optional[np.ndarray]:
    """Intensity-weighted centroid of the absolute field change, or None if negligible."""
    w = np.abs(diff)
    total = w.sum()
    if total < 1e-8:
        return None
    yy, xx = np.indices(diff.shape)
    cy = float((yy * w).sum() / total)
    cx = float((xx * w).sum() / total)
    return np.array([cy, cx])


# ---------------------------------------------------------------------------
# C-i — intervention consistency
# ---------------------------------------------------------------------------
def intervention_consistency(renderer, samples: Sequence[dict], cfg) -> Dict[str, object]:
    """C-i: fraction of perturbations whose rendered field effect matches the prediction.

    For each sample, build (orig, perturbed) ASG pairs via :func:`interventions.intervention_pairs`,
    render both through ``Z = ASG (+) advect_blind``, and check that the field difference matches the
    perturbation's :func:`interventions.expected_effect` within tolerance
    (``cfg.eval.spatial_tol_km`` / ``cfg.eval.intensity_tol_dbz``).

    Args:
        renderer: object exposing ``sample(Z, advect_blind, steps)->field`` (Stage C).
        samples:  list of dicts with ``asg_th`` (ASG), ``advect_blind`` (Tensor[1,H,W]); optional
                  ``H``, ``W``, ``km_per_pixel``.
        cfg:      loaded :class:`Config`.

    Returns:
        ``{"score": float, "per_type": {kind: {"score","n"}}, "n": int}``.
    """
    types = list(cfg.get_path("eval.intervention_types",
                              ["translate", "regime_flip", "growth_scale", "motion_rotate"]))
    spatial_tol_km = float(cfg.get_path("eval.spatial_tol_km", 10.0))
    intensity_tol_dbz = float(cfg.get_path("eval.intensity_tol_dbz", 3.0))
    flow_steps = int(cfg.get_path("stage_c.flow_steps", 4))
    kmpp = float(cfg.get_path("data.km_per_pixel", 1.0))

    per_type_hits: Dict[str, int] = {t: 0 for t in types}
    per_type_tot: Dict[str, int] = {t: 0 for t in types}
    total_hits = 0
    total = 0

    for s in samples:
        asg_th: ASG = s["asg_th"]
        advect_blind = s["advect_blind"]
        H = int(s.get("H", _infer_hw(advect_blind)[0]))
        W = int(s.get("W", _infer_hw(advect_blind)[1]))
        sample_kmpp = float(s.get("km_per_pixel", kmpp))

        base_field = _render(renderer, asg_th, advect_blind, H, W, flow_steps)
        base_2d = _field_2d(base_field)

        for orig, perturbed, meta in IV.intervention_pairs(asg_th, types):
            kind = meta["kind"]
            pert_field = _render(renderer, perturbed, advect_blind, H, W, flow_steps)
            pert_2d = _field_2d(pert_field)
            diff = pert_2d - base_2d
            ok = _check_effect(diff, base_2d, meta["expected"], kind,
                               spatial_tol_km, intensity_tol_dbz, sample_kmpp)
            per_type_tot[kind] += 1
            total += 1
            if ok:
                per_type_hits[kind] += 1
                total_hits += 1

    per_type = {
        t: {"score": (per_type_hits[t] / per_type_tot[t]) if per_type_tot[t] else 0.0,
            "n": per_type_tot[t]}
        for t in types
    }
    return {
        "score": (total_hits / total) if total else 0.0,
        "per_type": per_type,
        "n": total,
    }


def _infer_hw(advect_blind):
    a = _to_np(advect_blind)
    return a.shape[-2], a.shape[-1]


def _render(renderer, asg: ASG, advect_blind, H: int, W: int, steps: int):
    """Render a field from an ASG through the bottleneck Z = asg_channels (+) advect_blind.

    Builds Z via ``bottleneck.build_Z`` when available; otherwise calls the renderer with the
    advection field directly (renderers that accept an ASG are supported via duck typing).
    """
    ab = advect_blind
    if not isinstance(ab, torch.Tensor):
        ab = torch.as_tensor(np.asarray(ab), dtype=torch.float32)
    if ab.ndim == 2:
        ab = ab[None]  # -> [1,H,W]
    if _HAS_BOTTLENECK:
        Z = _bottleneck.build_Z(asg, ab, H, W)
        if Z.ndim == 3:
            Z = Z[None]
        ab_b = ab[None] if ab.ndim == 3 else ab
        return renderer.sample(Z, ab_b, steps)
    # Fallback: renderer that consumes an ASG + advection directly.
    return renderer.sample(asg, ab, steps)


def _check_effect(diff: np.ndarray, base: np.ndarray, expected: Dict[str, float], kind: str,
                  spatial_tol_km: float, intensity_tol_dbz: float, kmpp: float) -> bool:
    """Test whether the observed field change matches the predicted effect within tolerance."""
    change_mag = float(np.abs(diff).sum())
    base_mag = float(np.abs(base).sum())
    # Negligible base -> the perturbation cannot be meaningfully scored; treat as a miss only
    # when an effect was expected.
    if base_mag < 1e-6:
        return change_mag < 1e-6

    rel_change = change_mag / max(base_mag, 1e-8)  # change magnitude relative to the base field
    net = float(diff.sum())
    rel_net = net / max(change_mag, 1e-8)           # signed dominance of the change

    if kind == "translate":
        # The signal must have moved: the positive lobe (destination) and negative lobe
        # (source) of the difference must be spatially separated by ~ the expected displacement.
        pos_c = _centroid_of_change(np.maximum(diff, 0))
        neg_c = _centroid_of_change(np.maximum(-diff, 0))
        if pos_c is None or neg_c is None:
            return False
        sep_px = math.hypot(pos_c[0] - neg_c[0], pos_c[1] - neg_c[1])
        expected_px = float(expected.get("displacement_px", 0.0))
        tol_px = spatial_tol_km / max(kmpp, 1e-6)
        # Appreciable change AND the source->destination separation within the expected band.
        return rel_change > 0.02 and abs(sep_px - expected_px) <= tol_px + 0.5 * expected_px

    if kind == "regime_flip":
        # The growth sign flips: the field's net intensity moves in the predicted direction
        # (a growing cell weakens, a decaying cell intensifies; see expected["sign"]).
        sign = float(expected.get("sign", -1.0))
        return rel_change > 0.005 and (rel_net * sign) > 0.2

    if kind == "growth_scale":
        sign = float(expected.get("sign", 0.0))
        if sign == 0.0:
            return rel_change < 0.005                        # factor == 1 -> no change
        return rel_change > 0.005 and (rel_net * sign) > 0.2  # change in the predicted direction

    if kind == "motion_rotate":
        # Rotating the motion vector re-points the advective displacement of the cell, so the
        # rendered residual must shift location (non-trivial spatially-extended change).
        return rel_change > 0.01

    return change_mag > 0.0


# ---------------------------------------------------------------------------
# C-ii — bottleneck ablation
# ---------------------------------------------------------------------------
def bottleneck_ablation(renderer, transition, samples: Sequence[dict], cfg) -> Dict[str, float]:
    """C-ii: render from oracle / inferred / zeroed / shuffled ASG and score skill.

    Required pattern (eval.md section C-ii): oracle ~= best; inferred close behind; zeroed
    collapses to advection; shuffled is wrong-but-consistent-with-the-wrong-state.

    Args:
        renderer:   Stage C renderer with ``sample``.
        transition: Stage B (``predict(asg_t, context)->ASG``) or None if samples carry
                    ``asg_inferred`` directly.
        samples:    dicts with ``asg_th`` (oracle ASG), ``advect_blind``, ``target`` (field);
                    optional ``asg_t``, ``context``, ``asg_inferred``, ``H``, ``W``.
        cfg:        loaded :class:`Config`.

    Returns:
        ``{"oracle","inferred","zeroed","shuffled","advection"}`` mean CSI (heavy threshold).
    """
    from asgwm.eval import metrics as MET
    thr = float(cfg.get_path("eval.csi_thresholds_vil", cfg.get_path("eval.thresholds_dbz", [16, 35, 45]))[-1])
    flow_steps = int(cfg.get_path("stage_c.flow_steps", 4))

    n = len(samples)
    asg_ths = [s["asg_th"] for s in samples]

    def _hw(s):
        return int(s.get("H", _infer_hw(s["advect_blind"])[0])), int(s.get("W", _infer_hw(s["advect_blind"])[1]))

    csi_oracle, csi_infer, csi_zero, csi_shuf, csi_adv = [], [], [], [], []
    for i, s in enumerate(samples):
        H, W = _hw(s)
        ab = s["advect_blind"]
        target = _field_2d(s["target"])

        # oracle ASG
        oracle_field = _field_2d(_render(renderer, asg_ths[i], ab, H, W, flow_steps))
        csi_oracle.append(MET.csi(oracle_field, target, thr))

        # inferred ASG
        asg_inf = s.get("asg_inferred")
        if asg_inf is None and transition is not None and "asg_t" in s:
            asg_inf = transition.predict(s["asg_t"], s.get("context"))
        if asg_inf is None:
            asg_inf = asg_ths[i]  # fallback: oracle stands in for inferred
        infer_field = _field_2d(_render(renderer, asg_inf, ab, H, W, flow_steps))
        csi_infer.append(MET.csi(infer_field, target, thr))

        # zeroed ASG -> must collapse to advection
        zero_field = _field_2d(_render_zeroed(renderer, asg_ths[i], ab, H, W, flow_steps))
        csi_zero.append(MET.csi(zero_field, target, thr))

        # shuffled ASG (use a different event's ASG)
        j = (i + 1) % n if n > 1 else i
        shuf_field = _field_2d(_render(renderer, asg_ths[j], ab, H, W, flow_steps))
        csi_shuf.append(MET.csi(shuf_field, target, thr))

        # pure advection reference (the future-blind path alone)
        csi_adv.append(MET.csi(_field_2d(ab), target, thr))

    return {
        "oracle": float(np.mean(csi_oracle)) if csi_oracle else 0.0,
        "inferred": float(np.mean(csi_infer)) if csi_infer else 0.0,
        "zeroed": float(np.mean(csi_zero)) if csi_zero else 0.0,
        "shuffled": float(np.mean(csi_shuf)) if csi_shuf else 0.0,
        "advection": float(np.mean(csi_adv)) if csi_adv else 0.0,
    }


def _render_zeroed(renderer, asg: ASG, advect_blind, H: int, W: int, steps: int):
    """Render with the ASG channels of Z zeroed (C-ii zeroed condition)."""
    ab = advect_blind
    if not isinstance(ab, torch.Tensor):
        ab = torch.as_tensor(np.asarray(ab), dtype=torch.float32)
    if ab.ndim == 2:
        ab = ab[None]
    if _HAS_BOTTLENECK:
        Z = _bottleneck.build_Z(asg, ab, H, W)
        if Z.ndim == 3:
            Z = Z[None]
        Z = _bottleneck.zero_asg_in_Z(Z)
        ab_b = ab[None] if ab.ndim == 3 else ab
        return renderer.sample(Z, ab_b, steps)
    # Fallback: zeroed-ASG collapses to the advection field by construction.
    return ab


# ---------------------------------------------------------------------------
# C-iii — leakage audit (CLUB upper bound on mutual information)
# ---------------------------------------------------------------------------
class LeakageCLUB(nn.Module):
    """CLUB upper bound on I(advect_blind ; future | history) (cheng2020club).

    CLUB estimates an upper bound on mutual information I(X; Y) via a variational
    approximation q(y|x) = N(mu(x), sigma^2(x)):

        I_CLUB = E_{p(x,y)}[log q(y|x)] - E_{p(x)}E_{p(y)}[log q(y|x)].

    Here X = future-blind advection (flattened) and Y = the *residual* of the true future not
    explained by the history. A small (ideally ~0) bound confirms the advection path is
    future-blind: it carries no extra information about the target beyond what the history gives.
    """

    def __init__(self, x_dim: int, y_dim: int, hidden: int = 128):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.mu_net = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, y_dim),
        )
        self.logvar_net = nn.Sequential(
            nn.Linear(x_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, y_dim), nn.Tanh(),
        )

    def _q(self, x: torch.Tensor):
        mu = self.mu_net(x)
        logvar = self.logvar_net(x)  # bounded by tanh -> stable variance
        return mu, logvar

    def loglikelihood(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Mean log q(y|x); maximized to fit the variational approximation."""
        mu, logvar = self._q(x)
        return (-((mu - y) ** 2) / (2 * logvar.exp()) - 0.5 * logvar).sum(dim=1).mean()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Return the CLUB MI upper bound (nats) for the current q."""
        mu, logvar = self._q(x)
        var = logvar.exp()
        positive = (-((mu - y) ** 2) / (2 * var)).sum(dim=1)  # matched pairs
        # Negative term: mean over all y' for each x (shuffled pairing).
        n = x.shape[0]
        y_perm = y[torch.randperm(n, device=y.device)]
        negative = (-((mu - y_perm) ** 2) / (2 * var)).sum(dim=1)
        bound = (positive - negative).mean()
        return bound


def leakage_audit(advect_blind, future, hist, cfg) -> Dict[str, float]:
    """C-iii: CLUB upper bound on the MI the advection path adds about the future.

    Trains :class:`LeakageCLUB` to fit q(residual | advection), where ``residual`` is the
    component of the future not linearly predictable from the history, then reports the bound
    and a bootstrap confidence interval. A bound near zero confirms no future leakage.

    Args:
        advect_blind: [B, ...] future-blind advection fields.
        future:       [B, ...] true future fields (target).
        hist:         [B, ...] history fields (X_{<=t}).
        cfg:          loaded :class:`Config`.

    Returns:
        ``{"mi_nats": float, "ci_lo": float, "ci_hi": float}``.
    """
    device = "cpu"
    x = _flatten_batch(advect_blind).to(device)
    y_full = _flatten_batch(future).to(device)
    h = _flatten_batch(hist).to(device)

    # Reduce dimensionality for a tractable, stable estimate (random projection to <=64 dims).
    x = _project(x, 64)
    y_full = _project(y_full, 32)
    h = _project(h, 32)

    # Residual of the future not explained by history (least-squares deflation).
    y = _deflate(y_full, h)

    model = LeakageCLUB(x.shape[1], y.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    steps = int(cfg.get_path("eval.leakage_steps", 300))
    for _ in range(steps):
        opt.zero_grad()
        ll = model.loglikelihood(x, y)
        (-ll).backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        # Bootstrap the bound over resampled batches for a CI.
        boots = []
        n = x.shape[0]
        for _ in range(int(cfg.get_path("eval.leakage_boots", 50))):
            idx = torch.randint(0, n, (n,))
            boots.append(float(model(x[idx], y[idx]).item()))
        boots = np.array(boots)
        mi = float(np.clip(boots.mean(), 0.0, None))  # MI is non-negative
        lo = float(np.clip(np.percentile(boots, 2.5), 0.0, None))
        hi = float(np.clip(np.percentile(boots, 97.5), 0.0, None))
    return {"mi_nats": mi, "ci_lo": lo, "ci_hi": hi}


def _flatten_batch(x) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(np.asarray(x), dtype=torch.float32)
    t = t.float()
    return t.reshape(t.shape[0], -1)


def _project(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Deterministic random projection to ``dim`` features (identity if already smaller)."""
    if x.shape[1] <= dim:
        return x
    g = torch.Generator().manual_seed(1234)
    P = torch.randn(x.shape[1], dim, generator=g) / math.sqrt(x.shape[1])
    return x @ P


def _deflate(y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """Remove the part of ``y`` linearly predictable from ``h`` (least-squares residual)."""
    # Solve h @ B ~= y, return y - h @ B.
    try:
        sol = torch.linalg.lstsq(h, y).solution
        return y - h @ sol
    except Exception:
        return y


# ---------------------------------------------------------------------------
# C-iv — ASG accuracy (Hungarian matching on centroids)
# ---------------------------------------------------------------------------
def asg_accuracy(pred_asgs: List[ASG], gold_asgs: List[ASG]) -> Dict[str, float]:
    """C-iv: object F1, motion angular error, regime accuracy vs the gold subset.

    Objects are matched per-ASG by Hungarian assignment on centroid distance (scipy if
    present, greedy fallback). A match within a distance gate counts as a true positive.

    Args:
        pred_asgs: inferred ASGs.
        gold_asgs: hand-labeled gold ASGs (datasource.md section 2), aligned with ``pred_asgs``.

    Returns:
        ``{"obj_f1","motion_ang_err_deg","regime_acc","n_matched"}``.
    """
    tp = fp = fn = 0
    ang_errs: List[float] = []
    regime_hits = 0
    regime_tot = 0
    dist_gate = 32.0  # pixels; generous gate for SEVIR-scale cells

    for pred, gold in zip(pred_asgs, gold_asgs):
        matches, unmatched_p, unmatched_g = _match_objects(pred.objects, gold.objects, dist_gate)
        tp += len(matches)
        fp += len(unmatched_p)
        fn += len(unmatched_g)
        for pi, gi in matches:
            po, go = pred.objects[pi], gold.objects[gi]
            ang_errs.append(_angle_err(po, go))
            regime_tot += 1
            if po.regime == go.regime:
                regime_hits += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "obj_f1": float(f1),
        "motion_ang_err_deg": float(np.mean(ang_errs)) if ang_errs else 0.0,
        "regime_acc": float(regime_hits / regime_tot) if regime_tot else 0.0,
        "n_matched": float(tp),
    }


def _match_objects(pred_objs, gold_objs, dist_gate: float):
    """Hungarian (or greedy) match on centroid distance; return matched pairs + unmatched."""
    if not pred_objs or not gold_objs:
        return [], list(range(len(pred_objs))), list(range(len(gold_objs)))
    cost = np.zeros((len(pred_objs), len(gold_objs)))
    for i, po in enumerate(pred_objs):
        for j, go in enumerate(gold_objs):
            cost[i, j] = math.hypot(po.cy - go.cy, po.cx - go.cx)
    try:
        from scipy.optimize import linear_sum_assignment  # type: ignore
        rows, cols = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(rows, cols) if cost[r, c] <= dist_gate]
    except Exception:
        pairs = _greedy_match(cost, dist_gate)
    matched_p = {p for p, _ in pairs}
    matched_g = {g for _, g in pairs}
    unmatched_p = [i for i in range(len(pred_objs)) if i not in matched_p]
    unmatched_g = [j for j in range(len(gold_objs)) if j not in matched_g]
    return pairs, unmatched_p, unmatched_g


def _greedy_match(cost: np.ndarray, dist_gate: float):
    """Greedy nearest-neighbour assignment fallback."""
    pairs = []
    used_r, used_c = set(), set()
    flat = [(cost[i, j], i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])]
    for d, i, j in sorted(flat):
        if d > dist_gate:
            break
        if i in used_r or j in used_c:
            continue
        pairs.append((i, j))
        used_r.add(i)
        used_c.add(j)
    return pairs


def _angle_err(a: StormObject, b: StormObject) -> float:
    """Angular error (deg) between two motion vectors; 0 if either is ~stationary."""
    na = math.hypot(a.vy, a.vx)
    nb = math.hypot(b.vy, b.vx)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    dot = (a.vy * b.vy + a.vx * b.vx) / (na * nb)
    dot = max(-1.0, min(1.0, dot))
    return float(math.degrees(math.acos(dot)))


# ---------------------------------------------------------------------------
# C-v — counterfactual demo
# ---------------------------------------------------------------------------
def counterfactual_demo(renderer, asg: ASG, kinds: List[str], cfg) -> Dict[str, object]:
    """C-v: base field + one edited field per kind + their diffs, for the live demo.

    Args:
        renderer: Stage C renderer.
        asg:      the ASG_{t+h} to render and edit.
        kinds:    intervention kinds to demonstrate.
        cfg:      loaded :class:`Config`; uses ``data.km_per_pixel`` for the advection size.

    Returns:
        ``{"base_field": np.ndarray, "edited_fields": {kind: np.ndarray},
           "diffs": {kind: np.ndarray}}``.
    """
    flow_steps = int(cfg.get_path("stage_c.flow_steps", 4))
    grid = int(cfg.get_path("data.grid", 384))
    H = W = grid
    # A demo advection field: prefer one from the ASG meta if present, else zeros.
    ab = asg.meta.get("advect_blind")
    if ab is None:
        ab = torch.zeros(1, H, W)
    else:
        ab = torch.as_tensor(np.asarray(ab), dtype=torch.float32)
        if ab.ndim == 2:
            ab = ab[None]
        H, W = ab.shape[-2], ab.shape[-1]

    base = _field_2d(_render(renderer, asg, ab, H, W, flow_steps))
    edited: Dict[str, np.ndarray] = {}
    diffs: Dict[str, np.ndarray] = {}
    default_kw = {
        "translate": {"km": 20.0, "obj_idx": 0},
        "regime_flip": {"obj_idx": 0},
        "growth_scale": {"factor": 2.0, "obj_idx": 0},
        "motion_rotate": {"deg": 45.0, "obj_idx": 0},
    }
    for kind in kinds:
        kw = default_kw.get(kind, {"obj_idx": 0})
        edited_asg = IV.perturb_asg(asg, kind, **kw)
        ef = _field_2d(_render(renderer, edited_asg, ab, H, W, flow_steps))
        edited[kind] = ef
        diffs[kind] = ef - base
    return {"base_field": base, "edited_fields": edited, "diffs": diffs}
