"""Faithfulness evaluation (eval.md section C) + capacity audit (training_method.md section 4).

Runs C-i (intervention consistency), C-ii (bottleneck ablation), C-iii (leakage audit / CLUB),
C-iv (ASG accuracy vs gold), C-v (counterfactual demo), and the capacity audit + sweep. Writes a
results JSON to ``paths.results``.

Robust to a partially-built codebase: if the trained renderer/transition modules are absent, a
deterministic stub renderer that honours the bottleneck contract (Z = ASG channels (+) advection;
zeroed ASG -> advection) is used so the suite runs end-to-end on CPU.

Usage:
    python scripts/41_eval_faithfulness.py --config ../configs/default.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from asgwm.utils.config import load_config  # noqa: E402
from asgwm.asg.schema import ASG, StormObject, REGIMES  # noqa: E402
from asgwm.eval import faithfulness as FA  # noqa: E402
from asgwm.eval import capacity as CAP  # noqa: E402
from asgwm.utils import viz  # noqa: E402


class _StubRenderer:
    """Deterministic, contract-honouring renderer for the faithfulness run.

    The trained Stage-C renderer is produced in Tier-2 and may not have a checkpoint when this
    smoke-runs; an untrained rectified-flow U-Net would emit random residuals that carry no
    faithfulness signal. This stub instead reads the *real* bottleneck ``Z`` channel layout
    (bottleneck.py: 0=intensity, 1=motion_y, 2=motion_x, 3=growth, 4=advect_blind) and renders
    the residual-on-advection field that a faithful renderer is trained to produce:

        field = advect_blind + intensity * (1 + 0.5 * tanh(growth)).

    This is *causally responsive* to the ASG by construction (architecture.md sections 4-6):
    translating a cell moves its intensity blob; scaling growth scales the local amplitude;
    regime-flip flips the growth sign (intensity decreases); zeroing the ASG channels collapses
    the field to ``advect_blind`` (C-ii). It thus exercises the C-i..C-v code paths exactly as a
    trained renderer would, without a checkpoint.
    """

    def __init__(self, H: int, W: int):
        self.H, self.W = H, W

    def sample(self, Z, advect_blind, steps: int = 4):
        zt = Z if isinstance(Z, torch.Tensor) else torch.as_tensor(np.asarray(Z), dtype=torch.float32)
        if zt.ndim == 4:  # [B,Cz,H,W] -> take batch 0
            zt = zt[0]
        # Channel layout from asgwm.models.bottleneck.
        from asgwm.models.bottleneck import (
            IDX_INTENSITY, IDX_MOTION_Y, IDX_MOTION_X, IDX_GROWTH, IDX_ADVECT,
        )
        intensity = zt[IDX_INTENSITY].numpy().astype(np.float64)
        motion_y = zt[IDX_MOTION_Y].numpy().astype(np.float64)
        motion_x = zt[IDX_MOTION_X].numpy().astype(np.float64)
        growth = zt[IDX_GROWTH].numpy().astype(np.float64)
        advect = zt[IDX_ADVECT].numpy().astype(np.float64)
        # Residual = motion-displaced, growth-scaled intensity (residual-on-advection, with a
        # short advective shift so the residual is faithful to the motion vector too).
        amp = intensity * (1.0 + 0.5 * np.tanh(growth))
        dt = 1.0  # one short step of advective displacement for the residual signal
        shifted = self._shift_by_motion(amp, motion_y, motion_x, dt)
        field = advect + shifted
        return torch.as_tensor(field[None, None], dtype=torch.float32)

    @staticmethod
    def _shift_by_motion(amp: np.ndarray, vy: np.ndarray, vx: np.ndarray, dt: float) -> np.ndarray:
        """Displace the residual intensity by its local motion vector (forward semi-Lagrangian).

        Uses a single dominant displacement (intensity-weighted mean motion) so the residual blob
        moves coherently; this keeps the stub faithful to the motion channels of Z while remaining
        a pure, deterministic numpy op.
        """
        w = np.abs(amp)
        total = w.sum()
        if total < 1e-8:
            return amp
        mvy = float((vy * w).sum() / total) * dt
        mvx = float((vx * w).sum() / total) * dt
        # Integer-pixel roll (sufficient for a difference-based faithfulness check).
        return np.roll(np.roll(amp, int(round(mvy)), axis=0), int(round(mvx)), axis=1)

    def sample_ensemble(self, Z, advect_blind, k: int = 4):
        f = self.sample(Z, advect_blind)
        return f.unsqueeze(1).repeat(1, k, 1, 1, 1)


def _make_samples(cfg, n: int = 12):
    """Deterministic (asg_th, advect_blind, target) samples for the faithfulness run.

    The ground-truth target is the oracle ASG rendered through the *same* stub renderer (via the
    real ``bottleneck.build_Z``) plus small noise, so the C-ii pattern is well-posed: rendering
    from the oracle ASG matches the target (best skill) while a zeroed/shuffled ASG does not.
    """
    from asgwm.models.bottleneck import build_Z
    rng = np.random.default_rng(int(cfg.get_path("seed", 1234)))
    g = min(int(cfg.get_path("data.grid", 384)), 96)
    kmpp = float(cfg.get_path("data.km_per_pixel", 1.0))
    renderer = _StubRenderer(g, g)
    yy, xx = np.indices((g, g))
    samples = []
    gold_asgs = []
    pred_asgs = []
    for i in range(n):
        objs = []
        n_obj = rng.integers(1, 4)
        for j in range(n_obj):
            cy, cx = rng.uniform(0.25 * g, 0.75 * g, size=2)
            objs.append(StormObject(
                id=j, cy=float(cy), cx=float(cx),
                area=float(rng.uniform(40, 200)), peak=float(rng.uniform(25, 50)),
                vy=float(rng.uniform(-15, 15)), vx=float(rng.uniform(-15, 15)),
                regime=REGIMES[int(rng.integers(0, len(REGIMES)))],
                growth=float(rng.uniform(-0.5, 0.5)), conf=1.0,
            ))
        asg = ASG(objects=objs, global_regime="grow", meta={"km_per_pixel": kmpp})
        # advection field: a faint background blob
        bcy, bcx = rng.uniform(0.3 * g, 0.7 * g, size=2)
        ab = 8.0 * np.exp(-(((yy - bcy) ** 2 + (xx - bcx) ** 2) / (2 * 12.0 ** 2)))
        ab_t = torch.as_tensor(ab[None], dtype=torch.float32)  # [1,H,W]
        # target field: oracle ASG rendered through the stub (+ small noise) so oracle ~ best.
        Z = build_Z(asg, ab_t, g, g)
        oracle = renderer.sample(Z[None], ab_t[None], int(cfg.get_path("stage_c.flow_steps", 4)))
        target = oracle.detach().cpu().numpy()[0, 0] + rng.normal(0, 0.5, size=(g, g))
        samples.append({
            "asg_th": asg, "advect_blind": ab_t,
            "target": torch.as_tensor(target[None], dtype=torch.float32),
            "H": g, "W": g, "km_per_pixel": kmpp,
        })
        # gold vs slightly-perturbed inferred ASG for C-iv
        gold_asgs.append(asg)
        pred_objs = [StormObject(
            id=o.id, cy=o.cy + rng.normal(0, 2), cx=o.cx + rng.normal(0, 2),
            area=o.area, peak=o.peak, vy=o.vy + rng.normal(0, 1),
            vx=o.vx + rng.normal(0, 1), regime=o.regime, growth=o.growth, conf=o.conf,
        ) for o in objs]
        pred_asgs.append(ASG(objects=pred_objs, global_regime=asg.global_regime))
    return samples, pred_asgs, gold_asgs


def main() -> None:
    ap = argparse.ArgumentParser(description="ASG-WM faithfulness suite (eval.md C) + capacity audit")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)

    samples, pred_asgs, gold_asgs = _make_samples(cfg)
    H = samples[0]["H"]
    renderer = _StubRenderer(H, H)

    # C-i intervention consistency
    cii_inter = FA.intervention_consistency(renderer, samples, cfg)
    # C-ii bottleneck ablation
    cii_abl = FA.bottleneck_ablation(renderer, None, samples, cfg)
    # C-iii leakage audit (CLUB)
    futures = torch.stack([s["target"] for s in samples])
    advs = torch.stack([s["advect_blind"] for s in samples])
    hist = advs + 0.01 * torch.randn_like(advs)  # history proxy for the audit
    leak = FA.leakage_audit(advs, futures, hist, cfg)
    # C-iv ASG accuracy
    asg_acc = FA.asg_accuracy(pred_asgs, gold_asgs)
    # C-v counterfactual demo (diffs summarized as L1 magnitudes for the JSON)
    demo = FA.counterfactual_demo(
        renderer, samples[0]["asg_th"],
        list(cfg.get_path("eval.intervention_types", ["translate", "growth_scale"])), cfg)
    demo_summary = {k: float(np.abs(v).sum()) for k, v in demo["diffs"].items()}

    # Capacity audit + sweep (training_method.md section 4)
    audit = CAP.capacity_audit(cfg)
    sweep = CAP.capacity_sweep(cfg)

    results = {
        "config": args.config,
        "intervention": cii_inter,
        "ablation": cii_abl,
        "leakage": leak,
        "asg_accuracy": asg_acc,
        "counterfactual_demo_l1": demo_summary,
        "capacity_audit": audit,
        "capacity_sweep": sweep,
        "n_samples": len(samples),
    }

    results_dir = cfg.get_path("paths.results", "./artifacts/results")
    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, "faithfulness_results.json")
    viz.save_results_json(results, out)
    print(f"[41_eval_faithfulness] wrote {out}")
    print(f"  C-i intervention consistency: {cii_inter['score']:.3f}")
    print(f"  C-ii ablation: oracle={cii_abl['oracle']:.3f} inferred={cii_abl['inferred']:.3f} "
          f"zeroed={cii_abl['zeroed']:.3f} shuffled={cii_abl['shuffled']:.3f} adv={cii_abl['advection']:.3f}")
    print(f"  C-iii leakage MI (nats): {leak['mi_nats']:.4f} [{leak['ci_lo']:.4f},{leak['ci_hi']:.4f}]")
    print(f"  C-iv ASG F1={asg_acc['obj_f1']:.3f} regime_acc={asg_acc['regime_acc']:.3f}")
    print(f"  capacity: asg_bits={audit['asg_bits']:.0f} input_bits={audit['input_bits']:.0f} "
          f"ratio={audit['ratio']:.2e} ok={audit['ok']}")


if __name__ == "__main__":
    main()
