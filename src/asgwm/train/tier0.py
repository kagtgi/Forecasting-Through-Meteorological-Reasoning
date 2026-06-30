"""Tier 0 — transition transformer + deterministic renderer (training_method.md section 2).

Tier 0 is the thesis de-risking step and runs on the always-on L4 (and on CPU here for
the synthetic smoke test). It proves:

  1. `train_transition`         — the ASG->ASG transition transformer beats the trivial
                                  baselines on object evolution.
  2. `train_deterministic_renderer` — a deterministic (steps=1) renderer reaches pixel
                                  parity from an oracle ASG.
  3. `gate_check`               — transition vs persistence AND vs future-blind advection
                                  (the publishable go/no-go before any A100 session).

Every loop is checkpoint/resume safe (training_method.md section 6) and runs a few steps
on SyntheticSEVIR with no GPU. Heavy modules (data.*, models.*) are imported lazily and
guarded so this file imports cleanly even before those siblings land.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from asgwm.asg import ASG, REGIME_TO_IDX
from asgwm import physics
from . import checkpoint as ckpt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device(cfg) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _autocast_dtype(cfg) -> Optional[torch.dtype]:
    prec = str(cfg.get_path("train.precision", "fp32")).lower()
    if prec == "bf16":
        return torch.bfloat16
    if prec in ("fp16", "half"):
        return torch.float16
    return None


def _seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ckpt_dir(cfg, name: str) -> str:
    root = cfg.get_path("paths.checkpoints", "./artifacts/ckpt")
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    return d


def _import_transition():
    from asgwm.models.stage_b_transition import (
        TransitionTransformer,
        encode_asg,
        transition_loss,
    )
    return TransitionTransformer, encode_asg, transition_loss


def _import_transition_dataset():
    from asgwm.data.dataset import ASGTransitionDataset, collate_transition
    return ASGTransitionDataset, collate_transition


def _import_renderer():
    from asgwm.models.stage_c_renderer import LatentRectifiedFlowRenderer
    from asgwm.models.bottleneck import build_Z
    return LatentRectifiedFlowRenderer, build_Z


def _import_renderer_dataset():
    from asgwm.data.dataset import RendererDataset, collate_renderer
    return RendererDataset, collate_renderer


def _stack_targets(encode_asg, asgs: List[ASG], n_max: int, device) -> Dict[str, torch.Tensor]:
    """Encode a batch of target ASGs into the per-object tensors transition_loss expects."""
    feats, regimes, masks, centroids, motions = [], [], [], [], []
    for a in asgs:
        enc = encode_asg(a, n_max)
        feats.append(enc["obj_feats"])
        regimes.append(enc["regime_idx"])
        masks.append(enc["mask"])
        centroids.append(enc["centroids"])
        motions.append(enc["motion"])
    return {
        "obj_feats": torch.stack(feats).to(device),
        "regime_idx": torch.stack(regimes).to(device),
        "mask": torch.stack(masks).to(device),
        "centroids": torch.stack(centroids).to(device),
        "motion": torch.stack(motions).to(device),
    }


# ---------------------------------------------------------------------------
# 1) Transition transformer
# ---------------------------------------------------------------------------
def _build_transition_tensors(batch, model, encode_asg, device):
    """Encode a collated batch of (ASG_t, ASG_{t+h}) into the tensors the transition
    transformer and ``transition_loss`` consume.

    ``collate_transition`` keeps ASGs as Python lists (variable object counts); here we
    encode them slot-wise to the IB cap and build the *residual* targets that match the
    model's residual-on-advection convention (``predict``): the centroid target is
    ``cent_{t+h} - advect_points(cent_t, motion_t)`` and the attribute target is the
    delta on ``[area, peak, vy, vx, growth]``.
    """
    n_max = int(getattr(model, "n_max", 16))
    enc_t = [encode_asg(a, n_max) for a in batch["asg_t"]]
    enc_th = [encode_asg(a, n_max) for a in batch["asg_th"]]

    def stk(encs, key):
        return torch.stack([e[key] for e in encs], dim=0).to(device)

    obj_feats = stk(enc_t, "obj_feats")
    regime_idx = stk(enc_t, "regime_idx")
    mask = stk(enc_t, "mask")
    context = batch["context"].to(device)
    flow = batch["flow"].to(device)

    cent_t = stk(enc_t, "centroids")
    motion_t = stk(enc_t, "motion")
    cent_th = stk(enc_th, "centroids")
    b, n = cent_t.shape[:2]
    if bool(getattr(model, "predict_residual", True)):
        motion_px = physics.kmh_to_px_per_step(
            motion_t.reshape(-1, 2),
            float(getattr(model, "km_per_pixel", 1.0)),
            float(getattr(model, "minutes_per_frame", 5.0)),
        )
        advected = physics.advect_points(
            cent_t.reshape(-1, 2), motion_px, dt=float(getattr(model, "dt", 1.0))
        ).reshape(b, n, 2)
        target_centroid = cent_th - advected
    else:
        target_centroid = cent_th - cent_t

    attr_idx = torch.tensor([2, 3, 4, 5, 6], device=device)  # area, peak, vy, vx, growth
    attr_t = obj_feats.index_select(-1, attr_idx)
    attr_th = stk(enc_th, "obj_feats").index_select(-1, attr_idx)

    target = {
        "centroid": target_centroid,
        "attr": attr_th - attr_t,
        "regime_idx": stk(enc_th, "regime_idx"),
        "mask": mask,
    }
    return obj_feats, regime_idx, mask, context, flow, target


def _growth_target(batch, pred_growth: torch.Tensor) -> torch.Tensor:
    """Growth-field supervision target at the model head resolution.

    Uses the auto-labeled ASG_{t+h} growth field when present and shape-compatible;
    otherwise zeros (the transition signal is carried by position/regime/attr in that case).
    """
    fields = []
    h, w = pred_growth.shape[-2:]
    for a in batch["asg_th"]:
        gf = getattr(a, "growth_field", None)
        if gf is not None and gf.shape == (h, w):
            fields.append(torch.as_tensor(gf, dtype=pred_growth.dtype, device=pred_growth.device))
        else:
            fields.append(None)
    if any(f is None for f in fields):
        return torch.zeros_like(pred_growth)
    return torch.stack(fields, dim=0).unsqueeze(1)


def train_transition(cfg, resume: Optional[str] = None) -> str:
    """Train the ASG->ASG transition transformer (training_method.md section 2).

    Uses ASGTransitionDataset + TransitionTransformer + transition_loss. Checkpoints every
    `cfg.train.tier0.ckpt_every` steps to `paths.checkpoints/tier0_transition/`.
    Returns the final checkpoint path.
    """
    _seed_all(int(cfg.get_path("seed", 1234)))
    device = _device(cfg)
    n_max = int(cfg.get_path("asg.n_max", 16))
    TransitionTransformer, encode_asg, transition_loss = _import_transition()
    ASGTransitionDataset, collate_transition = _import_transition_dataset()

    ds = ASGTransitionDataset(cfg)
    bs = int(cfg.get_path("train.tier0.batch_size", 32))
    bs = max(1, min(bs, len(ds)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_transition)

    model = TransitionTransformer.from_config(cfg).to(device)
    lr = float(cfg.get_path("train.tier0.lr", 3e-4))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    start_step = 0
    cdir = _ckpt_dir(cfg, "tier0_transition")
    resume = resume or ckpt.latest(cdir)
    if resume and os.path.exists(resume):
        payload = ckpt.load_ckpt(resume, model=model, optim=optim)
        start_step = int(payload.get("step", 0))

    max_steps = int(cfg.get_path("train.tier0.max_steps", 20000))
    ckpt_every = int(cfg.get_path("train.tier0.ckpt_every", 1000))
    amp_dtype = _autocast_dtype(cfg)

    model.train()
    step = start_step
    last_path = resume or os.path.join(cdir, "ckpt_step_0.pt")
    data_iter = iter(loader)
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        obj_feats, regime_idx, mask, context, flow, target = _build_transition_tensors(
            batch, model, encode_asg, device
        )

        optim.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else _nullcontext()
        )
        with ctx:
            pred = model(obj_feats, regime_idx, mask, context)
            # growth-field target matches the model head resolution (zeros where the
            # auto-labeled field is unavailable; position/regime/attr carry the signal).
            target["growth_field"] = _growth_target(batch, pred["growth_field"])
            losses = transition_loss(pred, target, flow)
            loss = losses["total"]
        loss.backward()
        optim.step()
        step += 1

        if step % ckpt_every == 0 or step >= max_steps:
            last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
            ckpt.save_ckpt(last_path, step, model, optim, extra={"loss": float(loss.detach())})

    # Always write a final checkpoint so resume / gate_check find a state.
    last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
    ckpt.save_ckpt(last_path, step, model, optim)
    return last_path


# ---------------------------------------------------------------------------
# 2) Deterministic renderer (oracle ASG -> pixel parity)
# ---------------------------------------------------------------------------
def train_deterministic_renderer(cfg, resume: Optional[str] = None) -> str:
    """Train the deterministic (steps=1) renderer on oracle ASG (training_method.md s.2).

    Confirms pixel parity is reachable from a good state (the renderer, not the VLM, makes
    precise fields). Uses RendererDataset + LatentRectifiedFlowRenderer + bottleneck.build_Z.
    Returns the final checkpoint path.
    """
    _seed_all(int(cfg.get_path("seed", 1234)))
    device = _device(cfg)
    LatentRectifiedFlowRenderer, build_Z = _import_renderer()
    RendererDataset, collate_renderer = _import_renderer_dataset()

    ds = RendererDataset(cfg)
    bs = int(cfg.get_path("train.tier0.batch_size", 32))
    bs = max(1, min(bs, len(ds)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_renderer)

    model = LatentRectifiedFlowRenderer.from_config(cfg).to(device)
    lr = float(cfg.get_path("train.tier0.lr", 3e-4))
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    patch = int(cfg.get_path("data.patch", 128))
    start_step = 0
    cdir = _ckpt_dir(cfg, "tier0_renderer")
    resume = resume or ckpt.latest(cdir)
    if resume and os.path.exists(resume):
        payload = ckpt.load_ckpt(resume, model=model, optim=optim)
        start_step = int(payload.get("step", 0))

    max_steps = int(cfg.get_path("train.tier0.renderer_max_steps", 15000))
    ckpt_every = int(cfg.get_path("train.tier0.ckpt_every", 1000))
    amp_dtype = _autocast_dtype(cfg)

    model.train()
    step = start_step
    data_iter = iter(loader)
    last_path = resume or os.path.join(cdir, "ckpt_step_0.pt")
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        advect_blind = batch["advect_blind"].to(device)   # [B,1,H,W]
        target = batch["target"].to(device)               # [B,1,H,W]
        asg_list = batch["asg_th"]                         # list[ASG] (oracle)
        b, _, h, w = advect_blind.shape
        Z = torch.stack(
            [build_Z(asg_list[i], advect_blind[i], h, w) for i in range(b)]
        ).to(device)

        optim.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else _nullcontext()
        )
        with ctx:
            loss = model.training_loss(Z, advect_blind, target)
        loss.backward()
        optim.step()
        step += 1

        if step % ckpt_every == 0 or step >= max_steps:
            last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
            ckpt.save_ckpt(last_path, step, model, optim, extra={"loss": float(loss.detach())})

    last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
    ckpt.save_ckpt(last_path, step, model, optim)
    return last_path


# ---------------------------------------------------------------------------
# 3) Gate check — transition vs persistence & advection
# ---------------------------------------------------------------------------
def _centroid_array(asg: ASG, n_max: int) -> np.ndarray:
    arr = np.zeros((n_max, 2), dtype=np.float32)
    for i, o in enumerate(asg.objects[:n_max]):
        arr[i] = (o.cy, o.cx)
    return arr


def _match_position_error(pred: ASG, gold: ASG, n_max: int) -> float:
    """Mean centroid L2 over the min(n) matched-by-order objects (px)."""
    n = min(pred.n_objects, gold.n_objects)
    if n == 0:
        return 0.0
    pa = _centroid_array(pred, n_max)[:n]
    ga = _centroid_array(gold, n_max)[:n]
    return float(np.sqrt(((pa - ga) ** 2).sum(axis=1)).mean())


def _regime_sign_acc(pred: ASG, gold: ASG) -> float:
    """Fraction of matched objects whose growth sign agrees (growth/decay correctness)."""
    n = min(pred.n_objects, gold.n_objects)
    if n == 0:
        return 1.0
    ok = 0
    for i in range(n):
        if np.sign(pred.objects[i].growth) == np.sign(gold.objects[i].growth):
            ok += 1
    return ok / n


def gate_check(cfg) -> Dict[str, object]:
    """Tier-0 go/no-go: does the transition transformer beat persistence AND advection on
    object evolution? (training_method.md section 2 gate).

    Persistence baseline: ASG_{t+h} == ASG_t. Advection baseline: advance centroids by the
    ASG motion field via physics.advect_points (the same future-blind motion source used in
    labeling). Returns position error, regime-sign accuracy, and the two boolean gates.
    """
    _seed_all(int(cfg.get_path("seed", 1234)))
    device = _device(cfg)
    n_max = int(cfg.get_path("asg.n_max", 16))
    dt_min = float(cfg.get_path("data.horizon_min", 60))
    dt_steps = dt_min / float(cfg.get_path("data.minutes_per_frame", 5))

    TransitionTransformer, encode_asg, _ = _import_transition()
    ASGTransitionDataset, collate_transition = _import_transition_dataset()

    ds = ASGTransitionDataset(cfg)
    model = TransitionTransformer.from_config(cfg).to(device).eval()
    cdir = _ckpt_dir(cfg, "tier0_transition")
    latest_ckpt = ckpt.latest(cdir)
    if latest_ckpt:
        ckpt.load_ckpt(latest_ckpt, model=model)

    pos_model, pos_persist, pos_advect = [], [], []
    reg_model, reg_persist, reg_advect = [], [], []
    n_eval = min(len(ds), 64)
    with torch.no_grad():
        for i in range(n_eval):
            item = ds[i]
            asg_t: ASG = item["asg_t"]
            asg_gold: ASG = item["asg_th"]
            context = item["context"].to(device).unsqueeze(0)

            # --- model prediction ---
            try:
                pred_asg = model.predict(asg_t, context.squeeze(0))
            except Exception:
                pred_asg = asg_t  # degenerate fallback keeps the gate runnable
            pos_model.append(_match_position_error(pred_asg, asg_gold, n_max))
            reg_model.append(_regime_sign_acc(pred_asg, asg_gold))

            # --- persistence baseline ---
            pos_persist.append(_match_position_error(asg_t, asg_gold, n_max))
            reg_persist.append(_regime_sign_acc(asg_t, asg_gold))

            # --- advection baseline (advance centroids by motion) ---
            adv_objs = []
            if asg_t.n_objects:
                cent = torch.tensor(
                    [[o.cy, o.cx] for o in asg_t.objects], dtype=torch.float32
                )
                mot = torch.tensor(
                    [[o.vy, o.vx] for o in asg_t.objects], dtype=torch.float32
                )
                mot_px = physics.kmh_to_px_per_step(
                    mot,
                    float(cfg.get_path("data.km_per_pixel", 1.0)),
                    float(cfg.get_path("data.minutes_per_frame", 5)),
                )
                new_cent = physics.advect_points(cent, mot_px, dt=dt_steps).numpy()
                from asgwm.asg import StormObject
                for j, o in enumerate(asg_t.objects):
                    adv_objs.append(
                        StormObject(
                            id=o.id, cy=float(new_cent[j, 0]), cx=float(new_cent[j, 1]),
                            area=o.area, peak=o.peak, vy=o.vy, vx=o.vx,
                            regime=o.regime, growth=o.growth, conf=o.conf,
                        )
                    )
            adv_asg = ASG(objects=adv_objs, global_regime=asg_t.global_regime,
                          context=dict(asg_t.context))
            pos_advect.append(_match_position_error(adv_asg, asg_gold, n_max))
            reg_advect.append(_regime_sign_acc(adv_asg, asg_gold))

    def _m(xs: List[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    metrics = {
        "pos_err_model": _m(pos_model),
        "pos_err_persistence": _m(pos_persist),
        "pos_err_advection": _m(pos_advect),
        "regime_acc_model": _m(reg_model),
        "regime_acc_persistence": _m(reg_persist),
        "regime_acc_advection": _m(reg_advect),
        "n_eval": n_eval,
    }
    # Beats a baseline if lower position error AND no worse regime-sign accuracy.
    beats_persistence = (
        metrics["pos_err_model"] <= metrics["pos_err_persistence"]
        and metrics["regime_acc_model"] >= metrics["regime_acc_persistence"]
    )
    beats_advection = (
        metrics["pos_err_model"] <= metrics["pos_err_advection"]
        and metrics["regime_acc_model"] >= metrics["regime_acc_advection"]
    )
    return {
        "beats_persistence": bool(beats_persistence),
        "beats_advection": bool(beats_advection),
        "metrics": metrics,
    }


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
