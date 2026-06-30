"""Tier 2 — end-to-end coupling A -> B -> bottleneck -> C (training_method.md section 4).

Couples the perception VLM (Stage A), the transition transformer (Stage B), the faithful
bottleneck, and the rectified-flow renderer (Stage C). Training details
(training_method.md section 4, architecture.md section 6):

  - Low-LR / stop-grad on the VLM (the perception adapters are already trained in Tier 1).
  - Scheduled sampling oracle -> inferred ASG over `cfg.train.tier2.oracle_anneal_steps`,
    so renderer errors compose gracefully.
  - Intervention-consistency loss (the make-or-break faithfulness signal, C-i).
  - Ensembles for CRPS via the few-step flow.

Checkpoint/resume safe (training_method.md section 6); runs a few steps on
SyntheticSEVIR + DummyVLM with no GPU. Heavy modules imported lazily/guarded.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from asgwm.asg import ASG
from . import checkpoint as ckpt
from .losses import tier2_total_loss, intervention_consistency_loss


def _device(cfg) -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


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


def _ckpt_dir(cfg) -> str:
    root = cfg.get_path("paths.checkpoints", "./artifacts/ckpt")
    d = os.path.join(root, "tier2")
    os.makedirs(d, exist_ok=True)
    return d


def _import_modules():
    from asgwm.models.stage_a_vlm import StageAVLM
    from asgwm.models.stage_b_transition import TransitionTransformer
    from asgwm.models.stage_c_renderer import LatentRectifiedFlowRenderer
    from asgwm.models.bottleneck import build_Z, zero_asg_in_Z, asg_to_field_channels
    from asgwm.data.dataset import RendererDataset, collate_renderer
    return (
        StageAVLM, TransitionTransformer, LatentRectifiedFlowRenderer,
        build_Z, zero_asg_in_Z, asg_to_field_channels, RendererDataset, collate_renderer,
    )


def _oracle_prob(step: int, anneal_steps: int) -> float:
    """Scheduled-sampling probability of using the ORACLE ASG (vs inferred).

    Starts at 1.0 (always oracle), linearly anneals to 0.0 over `anneal_steps`
    (training_method.md section 4)."""
    if anneal_steps <= 0:
        return 0.0
    return float(max(0.0, 1.0 - step / float(anneal_steps)))


def _make_perturb_fn(asg_to_field_channels, Cz: int):
    """Build a Z-perturbation that translates the ASG intensity channel (channel 0) a few
    pixels along +x — a structured intervention whose effect is a known spatial shift.

    Returns (perturb_fn, expected_fn) for `intervention_consistency_loss`. The expected
    field change is the rendered field shifted by the same displacement minus the original
    (a first-order proxy for the translate intervention, architecture.md section 6)."""
    shift = 3  # pixels along +x (cols)

    def perturb_fn(Z: torch.Tensor) -> torch.Tensor:
        Zp = Z.clone()
        # ASG intensity channel is channel 0 by the bottleneck contract; roll it along x.
        Zp[:, 0:1] = torch.roll(Zp[:, 0:1], shifts=shift, dims=3)
        return Zp

    def expected_fn(field_orig: torch.Tensor, Z_perturbed: torch.Tensor) -> torch.Tensor:
        shifted = torch.roll(field_orig, shifts=shift, dims=3)
        return shifted - field_orig

    return perturb_fn, expected_fn


def train_tier2(
    cfg,
    vlm_ckpt: Optional[str] = None,
    transition_ckpt: Optional[str] = None,
    resume: Optional[str] = None,
) -> str:
    """End-to-end Tier-2 training (training_method.md section 4).

    Args:
        cfg:             Config.
        vlm_ckpt:        Tier-1 Ph-5 checkpoint for Stage A (loaded, then stop-grad/low-LR).
        transition_ckpt: Tier-0 transition checkpoint for Stage B.
        resume:          Tier-2 checkpoint to resume the renderer/optimizer from.
    Returns:
        Final Tier-2 checkpoint path.
    """
    _seed_all(int(cfg.get_path("seed", 1234)))
    device = _device(cfg)
    (StageAVLM, TransitionTransformer, LatentRectifiedFlowRenderer,
     build_Z, zero_asg_in_Z, asg_to_field_channels,
     RendererDataset, collate_renderer) = _import_modules()

    # --- Stage A (VLM): frozen / stop-grad, optional low-LR adapters (kept frozen here) ---
    vlm = StageAVLM.from_config(cfg)
    if vlm_ckpt:
        if hasattr(vlm, "load_adapters") and os.path.exists(str(vlm_ckpt) + ".adapters"):
            try:
                vlm.load_adapters(str(vlm_ckpt) + ".adapters")
            except Exception:
                pass
        elif os.path.exists(str(vlm_ckpt)):
            try:
                ckpt.load_ckpt(vlm_ckpt, model=vlm)
            except Exception:
                pass
    for p in getattr(vlm, "parameters", lambda: [])():
        p.requires_grad_(False)
    if hasattr(vlm, "eval"):
        vlm.eval()

    # --- Stage B (transition): loaded from Tier 0; low-LR fine-tune ---
    transition = TransitionTransformer.from_config(cfg).to(device)
    if transition_ckpt and os.path.exists(transition_ckpt):
        try:
            ckpt.load_ckpt(transition_ckpt, model=transition)
        except Exception:
            pass

    # --- Stage C (renderer): the primary trained module in Tier 2 ---
    renderer = LatentRectifiedFlowRenderer.from_config(cfg).to(device)

    lr = float(cfg.get_path("train.tier2.lr", 1e-5))
    params = list(renderer.parameters()) + [
        p for p in transition.parameters() if p.requires_grad
    ]
    optim = torch.optim.AdamW(params, lr=lr)

    ds = RendererDataset(cfg)
    bs = int(cfg.get_path("train.tier2.batch_size", 8))
    bs = max(1, min(bs, len(ds)))
    loader = DataLoader(ds, batch_size=bs, shuffle=True, collate_fn=collate_renderer)

    cdir = _ckpt_dir(cfg)
    resume = resume or ckpt.latest(cdir)
    start_step = 0
    if resume and os.path.exists(resume):
        payload = ckpt.load_ckpt(resume, model=renderer, optim=optim)
        start_step = int(payload.get("step", 0))
        try:
            if payload.get("extra", {}).get("transition") is not None:
                transition.load_state_dict(payload["extra"]["transition"], strict=False)
        except Exception:
            pass

    max_steps = int(cfg.get_path("train.tier2.max_steps", 30000))
    ckpt_every = max(1, int(cfg.get_path("train.tier2.ckpt_every", 500)))
    anneal_steps = int(cfg.get_path("train.tier2.oracle_anneal_steps", 8000))
    use_sched = bool(cfg.get_path("train.tier2.scheduled_sampling", True))
    use_intervene = bool(cfg.get_path("train.tier2.intervention_consistency", True))
    flow_steps = int(cfg.get_path("stage_c.flow_steps", 4))
    amp_dtype = _autocast_dtype(cfg)
    lam_int = float(cfg.get_path("losses.lambda_intervene", 1.0))
    # Convergence hardening (see draft FEASIBILITY analysis): ramp the intervention-
    # consistency weight 0 -> lam_int over a warmup so the renderer becomes reconstruction-
    # coherent before the high-variance, double-render causal signal is enforced; optionally
    # compute that (expensive) signal only every k steps; log every term to catch divergence
    # in minutes rather than after a paid session.
    int_warmup = int(cfg.get_path("train.tier2.intervene_warmup_steps", 2000))
    int_every = max(1, int(cfg.get_path("train.tier2.intervene_every", 1)))
    log_every = max(1, int(cfg.get_path("train.tier2.log_every", 50)))

    renderer.train()
    transition.train()
    step = start_step
    last_path = resume or os.path.join(cdir, "ckpt_step_0.pt")
    data_iter = iter(loader)
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        advect_blind = batch["advect_blind"].to(device)   # [B,1,H,W]
        target = batch["target"].to(device)               # [B,1,H,W]
        oracle_asgs: List[ASG] = batch["asg_th"]
        growth_budget = batch["growth_budget"].to(device)
        b, _, h, w = advect_blind.shape

        # --- scheduled sampling: choose oracle vs Stage-B-inferred ASG per sample ---
        p_oracle = _oracle_prob(step, anneal_steps) if use_sched else 0.0
        asgs_used: List[ASG] = []
        for i in range(b):
            if (not use_sched) or np.random.rand() < p_oracle:
                asgs_used.append(oracle_asgs[i])
            else:
                # Stage-B prediction from the oracle ASG_t available in the batch (or the
                # target ASG as a stand-in if asg_t is absent). VLM stays stop-grad.
                base = batch["asg_t"][i]
                ctx_vec = batch.get("context")
                ctx_i = ctx_vec[i].to(device) if ctx_vec is not None else None
                try:
                    asgs_used.append(transition.predict(base, ctx_i))
                except Exception:
                    asgs_used.append(oracle_asgs[i])

        Z = torch.stack(
            [build_Z(asgs_used[i], advect_blind[i], h, w) for i in range(b)]
        ).to(device)

        # Continuous ASG attributes subject to the soft IB penalty: the ASG field channels.
        asg_cont = Z[:, :-1]  # all but the advect_blind channel (bottleneck contract)
        flow = batch.get("flow")
        if flow is not None:
            flow = flow.to(device)
        else:
            flow = torch.zeros(b, 2, h, w, device=device)

        optim.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None
            else _nullcontext()
        )
        with ctx:
            # Renderer flow-matching loss is the field reconstruction term.
            render_loss = renderer.training_loss(Z, advect_blind, target)
            pred_field = renderer.sample(Z, advect_blind, steps=flow_steps)
            loss_parts = tier2_total_loss(
                pred_field, target, Z, asg_cont, flow, growth_budget, cfg
            )
            total = render_loss + (loss_parts["total"] - loss_parts["render"])

            if use_intervene and (step % int_every == 0):
                Cz = Z.shape[1]
                perturb_fn, expected_fn = _make_perturb_fn(asg_to_field_channels, Cz)
                intervene = intervention_consistency_loss(
                    renderer, Z, advect_blind, perturb_fn, expected_fn, steps=flow_steps
                )
                # linear warmup 0 -> lam_int over int_warmup steps
                w_int = lam_int * min(1.0, step / float(max(1, int_warmup)))
                total = total + w_int * intervene
            else:
                intervene = torch.zeros((), device=device)
                w_int = 0.0

        # Fail fast on a non-finite loss: abort BEFORE the optimiser step so a divergence
        # (bad lr / exploding intervention term) costs seconds, not the whole GPU session.
        if not torch.isfinite(total.detach()):
            raise RuntimeError(
                f"[tier2] non-finite loss at step {step} "
                f"(render={float(render_loss):.3e}, intervene={float(intervene):.3e}); "
                "aborting before wasting the session — lower train.tier2.lr or check inputs."
            )

        total.backward()
        optim.step()
        step += 1

        if step % log_every == 0 or step == start_step + 1:
            print(
                f"[tier2] step {step:6d}/{max_steps} total={float(total.detach()):.4f} "
                f"render={float(render_loss.detach()):.4f} "
                f"intervene={float(intervene.detach()):.4f}(w={w_int:.2f}) "
                f"mass={float(loss_parts['mass']):.2e} nonneg={float(loss_parts['nonneg']):.2e} "
                f"spec={float(loss_parts['spectral']):.2e} cont={float(loss_parts['continuity']):.2e} "
                f"ib={float(loss_parts['ib']):.2e} p_oracle={p_oracle:.2f}",
                flush=True,
            )

        if step % ckpt_every == 0 or step >= max_steps:
            last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
            ckpt.save_ckpt(
                last_path, step, renderer, optim,
                extra={
                    "transition": transition.state_dict(),
                    "render_loss": float(render_loss.detach()),
                    "intervene": float(intervene.detach()),
                    "p_oracle": p_oracle,
                },
            )

    last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
    ckpt.save_ckpt(last_path, step, renderer, optim,
                   extra={"transition": transition.state_dict()})
    return last_path


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
