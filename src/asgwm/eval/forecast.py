"""Forecast assembly for evaluation (eval.md sections A/B/F).

`assemble(method, cfg, ...)` returns per-event evaluation samples for ANY method, so the
skill / regime / lead-time / gallery code is method-agnostic:

    sample = {event_id, regime, obs[n,H,W], pred[n,H,W], ens[K,n,H,W], context}

`method="asgwm"` runs the real Stage A -> B -> bottleneck -> C pipeline when trained
checkpoints are present, and falls back to future-blind advection when they are not (so the
pipeline runs end-to-end pre-training; numbers are TBR until Tier-2 completes). Every other
method is dispatched through the pluggable baseline registry (`asgwm.baselines`); a baseline
that is not yet implemented yields no samples and the harness writes a TBR row for it.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from asgwm.asg import ASG
from asgwm.asg.render_nl import render_NL, render_NL_delta
from asgwm import baselines as B


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cap(vil: np.ndarray, cap: int = 96) -> np.ndarray:
    """Stride-downsample [T,H,W] to <= cap on the spatial dims.

    Default cap keeps CPU/synthetic smoke cheap; the real (A100) skill run sets
    ``eval.eval_grid`` to the full canonical grid (384) so CSI matches the published
    SEVIR-VIL baselines rather than a downsampled proxy.
    """
    if vil.ndim != 3:
        vil = np.asarray(vil)
    _, h, w = vil.shape
    sy = max(1, h // cap)
    sx = max(1, w // cap)
    return np.ascontiguousarray(vil[:, ::sy, ::sx]).astype(np.float32)


def _events(cfg, n_events: int):
    """Yield up to n_events (event_id, vil[T,H,W]) from the dataset (synthetic fallback)."""
    from asgwm.data import sevir as S
    count = 0
    for ev in S.iter_events(cfg):
        vil = ev.get("vil")
        if vil is None:
            continue
        eid = str(ev.get("event_id", ev.get("id", f"event_{count:05d}")))
        yield eid, np.asarray(vil, dtype=np.float32), ev
        count += 1
        if count >= n_events:
            break


def _split(vil: np.ndarray, in_frames: int, out_frames: int):
    hist = vil[:in_frames]
    fut = vil[in_frames:in_frames + out_frames]
    return hist, fut


# ---------------------------------------------------------------------------
# ASG-WM model stack (checkpoint-gated; advection fallback otherwise)
# ---------------------------------------------------------------------------
def _load_asgwm(cfg) -> Optional[Dict]:
    """Load Stage-B + Stage-C if their checkpoints exist; else None (-> advection fallback)."""
    try:
        import torch  # noqa: F401
        from asgwm.train import checkpoint as ckpt
        cdir = cfg.get_path("paths.checkpoints", "./artifacts/ckpt")
        tr_ck = (
            ckpt.latest(os.path.join(cdir, "tier2"))
            or ckpt.latest(os.path.join(cdir, "tier2_endtoend"))
            or ckpt.latest(os.path.join(cdir, "tier0_renderer"))
        )
        b_ck = ckpt.latest(os.path.join(cdir, "tier0_transition"))
        if not (tr_ck or b_ck):
            return None  # nothing trained yet
        from asgwm.models.stage_b_transition import TransitionTransformer
        from asgwm.models.stage_c_renderer import LatentRectifiedFlowRenderer
        from asgwm.utils.device import resolve_device
        dev = resolve_device(cfg)
        models = {"device": dev}
        tt = TransitionTransformer.from_config(cfg)
        if b_ck:
            ckpt.load_ckpt(b_ck, model=tt)
        models["transition"] = tt.to(dev).eval()
        rr = LatentRectifiedFlowRenderer.from_config(cfg)
        if tr_ck:
            ckpt.load_ckpt(tr_ck, model=rr)
        models["renderer"] = rr.to(dev).eval()
        print(f"[forecast] ASG-WM models on {dev}")
        return models
    except Exception as e:  # untrained / deps missing -> graceful fallback
        print(f"[forecast] ASG-WM models unavailable ({e}); using advection fallback")
        return None


def _asgwm_pred(hist, asg_seq, cfg, models, n_out, K):
    """Return (pred[n_out,H,W], ens[K,n_out,H,W], trace_dict).

    Real path: ASG_{t+h} from Stage B, rendered per-frame on the advection sequence.
    Fallback: future-blind advection (used pre-training and on any error).

    trace_dict carries NL readouts for gallery / faithfulness demos (philosophy.md section 2.2):
        nl_t      = render_NL(ASG_t)
        nl_delta  = render_NL_delta(ASG_t, ASG_{t+h})
    """
    from asgwm.data.advection import advect_blind
    adv = np.asarray(advect_blind(hist, n_out), dtype=np.float32)  # [n_out,H,W]
    trace: Dict = {}
    if asg_seq is not None:
        trace["nl_t"] = render_NL(asg_seq.asg_t)
    if models is None:
        return adv, np.repeat(adv[None], K, axis=0), trace
    try:
        import torch
        from asgwm.models.bottleneck import build_Z
        from asgwm.utils.device import autocast_ctx
        H, W = adv.shape[-2:]
        dev = models.get("device", torch.device("cpu"))
        asg_th = models["transition"].predict(asg_seq.asg_t, None)
        if asg_seq is not None:
            trace["nl_delta"] = render_NL_delta(asg_seq.asg_t, asg_th)
        renderer = models["renderer"]
        steps = int(cfg.get_path("stage_c.flow_steps", 4))

        def _one():
            frames = []
            for k in range(n_out):
                ab_cpu = torch.from_numpy(adv[k]).float().view(1, 1, H, W)
                Z = build_Z(asg_th, ab_cpu[0], H, W).unsqueeze(0).to(dev)  # build_Z on CPU, then move
                ab = ab_cpu.to(dev)
                with torch.no_grad(), autocast_ctx(dev, cfg):
                    f = renderer.sample(Z, ab, steps)
                frames.append(f.view(H, W).float().cpu().numpy())
            return np.stack(frames, 0).astype(np.float32)

        pred = _one()
        ens = np.stack([_one() for _ in range(K)], 0)
        return pred, ens, trace
    except Exception as e:
        print(f"[forecast] ASG-WM render failed ({e}); advection fallback for this event")
        return adv, np.repeat(adv[None], K, axis=0), trace


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def assemble(method: str, cfg, n_events: int = 24, K: Optional[int] = None) -> List[Dict]:
    """Per-event evaluation samples for `method` ("asgwm" or a registered baseline name).

    Returns [] if the method is a baseline that is not yet available (-> TBR row).
    """
    in_frames = int(cfg.get_path("data.in_frames", 13))
    out_frames = int(cfg.get_path("data.out_frames", 36))
    n_out = min(out_frames, len(cfg.get_path("eval.lead_times_min", [30, 60, 90, 120, 150, 180])) * 6 or out_frames)
    n_out = out_frames if out_frames else 12
    K = int(K if K is not None else cfg.get_path("eval.ensemble_k", 10))

    is_asgwm = method.lower() in ("asgwm", "asg-wm", "ours")
    bl = None if is_asgwm else B.get(method)
    if not is_asgwm:
        if bl is None or not bl.is_available():
            return []  # TBR — not yet coded/obtained

    models = _load_asgwm(cfg) if is_asgwm else None
    from asgwm.labeling import pipeline as P

    # Evaluate at the full canonical grid for paper numbers; override eval.eval_grid (e.g. 96)
    # to keep synthetic/CPU smoke fast.
    eval_grid = int(cfg.get_path("eval.eval_grid", cfg.get_path("data.grid", 384)))

    samples: List[Dict] = []
    for eid, vil, ev in _events(cfg, n_events):
        vil = _cap(vil, eval_grid)
        hist, fut = _split(vil, in_frames, out_frames)
        if hist.shape[0] < 2 or fut.shape[0] < 1:
            continue
        n = min(n_out, fut.shape[0])
        obs = fut[:n]
        # ASG (for regime label + the ASG-WM state); cheap auto-label
        try:
            seq = P.autolabel_event(ev, cfg)
            regime = seq.asg_t.global_regime
            context = dict(seq.asg_t.context)
        except Exception:
            seq, regime, context = None, "steady", {}

        from asgwm.eval.ablation import apply_knowledge_ablation
        context = apply_knowledge_ablation(context, cfg)

        if is_asgwm:
            pred, ens, trace = _asgwm_pred(hist, seq, cfg, models, n, K)
        else:
            pred = np.asarray(bl.predict(hist, context, n), dtype=np.float32)
            ens = np.asarray(bl.predict_ensemble(hist, context, n, k=max(1, K)), dtype=np.float32)
            trace = {}
        sample = {
            "event_id": eid, "regime": regime,
            "obs": obs[:n], "pred": pred[:n], "ens": ens[:, :n] if ens.ndim == 4 else ens,
            "context": context,
        }
        if trace:
            sample.update(trace)
        samples.append(sample)
    return samples
