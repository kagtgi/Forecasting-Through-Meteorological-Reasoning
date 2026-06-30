"""Tier-2 loss stack and the intervention-consistency term.

Implements the loss stack from training_method.md section 4:

    L = L_render + lambda_ib * L_IB + lambda_intervene * L_intervene
        + lambda_mass * L_mass + lambda_nonneg * L_nonneg
        + lambda_spectral * L_spectral + lambda_continuity * L_continuity

`L_render` is the field reconstruction term (MSE in pixel space here; the renderer's
flow-matching loss is computed separately inside Stage C). The physics / realism terms
(`L_mass`, `L_nonneg`, `L_spectral`, `L_continuity`) reuse `asgwm.physics.*`, the
compression term reuses `bottleneck.soft_ib_penalty`, and `L_intervene` is the
make-or-break faithfulness signal (architecture.md section 6).

Weights are read from `cfg.losses.*` (configs/default.yaml) using EXACT keys:
lambda_ib, lambda_intervene, lambda_mass, lambda_nonneg, lambda_spectral,
lambda_continuity.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from asgwm import physics

try:  # bottleneck is implemented by another agent; guard so this file imports cleanly.
    from asgwm.models.bottleneck import soft_ib_penalty
except Exception:  # pragma: no cover - fallback used only if bottleneck absent
    def soft_ib_penalty(asg_continuous: torch.Tensor, lambda_ib: float) -> torch.Tensor:
        """KL of `asg_continuous` to a unit Gaussian (fallback if bottleneck missing)."""
        mu = asg_continuous
        # KL(N(mu,1) || N(0,1)) = 0.5 * mu^2 ; lambda applied by caller.
        return 0.5 * (mu ** 2).mean()


def _as_float_tensor(x: object, ref: torch.Tensor) -> torch.Tensor:
    """Coerce a scalar / sequence into a 1-D float tensor on ref's device."""
    if isinstance(x, torch.Tensor):
        return x.to(device=ref.device, dtype=ref.dtype)
    return torch.as_tensor(x, device=ref.device, dtype=ref.dtype)


# ---------------------------------------------------------------------------
# Balanced reconstruction (HKO-7 B-MSE: up-weight heavy-rain pixels)
# ---------------------------------------------------------------------------
def balanced_weight_map(
    target: torch.Tensor,
    thresholds,
    weights,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-pixel weight map (HKO-7 balanced loss, Shi et al. 2017).

    Heavy-precip pixels are scarce but matter most, so weight each pixel by the bin its
    ``target`` value falls into: ``weights[0]`` below ``thresholds[0]``, then ``weights[i+1]``
    at/above ``thresholds[i]`` (thresholds ascending, ``len(weights) == len(thresholds)+1``).
    ``thresholds`` are in the SAME units as ``target`` (default config = VIL byte, aligned to
    the CSI thresholds). ``valid_mask`` (truthy = valid) zeroes invalid/clutter pixels so they
    are excluded from training, exactly as HKO-7 masks noise.
    """
    w = torch.full_like(target, float(weights[0]))
    for t, wt in zip(thresholds, weights[1:]):
        w = torch.where(target >= float(t), torch.as_tensor(float(wt), dtype=w.dtype, device=w.device), w)
    if valid_mask is not None:
        w = w * valid_mask.to(dtype=w.dtype, device=w.device)
    return w


def balanced_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Weight-normalized MSE: sum(w*(pred-target)^2) / sum(w) (commensurate with plain MSE)."""
    num = (weight * (pred - target) ** 2).sum()
    den = weight.sum().clamp_min(1e-6)
    return num / den


def tier2_total_loss(
    pred_field: torch.Tensor,
    target_field: torch.Tensor,
    Z: torch.Tensor,
    asg_cont: torch.Tensor,
    flow: torch.Tensor,
    growth_budget: torch.Tensor,
    cfg,
    valid_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Full Tier-2 loss (training_method.md section 4).

    Args:
        pred_field:    [B,1,H,W] rendered field.
        target_field:  [B,1,H,W] ground-truth field.
        Z:             [B,Cz,H,W] bottleneck tensor (ASG channels (+) advect_blind).
        asg_cont:      continuous ASG attributes subject to the soft IB penalty.
        flow:          [B,2,H,W] (vy,vx) motion field for the continuity residual.
        growth_budget: [B] target integrated content for the mass-budget term.
        cfg:           Config; reads cfg.losses.lambda_* and (optional) cfg.losses.balanced_*.
        valid_mask:    [B,1,H,W] truthy=valid; excludes clutter/no-coverage pixels from the
                       reconstruction term (HKO-7 mask handling). None = all valid.
    Returns:
        dict with keys: total, render, ib, intervene, mass, nonneg, spectral, continuity.
    """
    lam_ib = float(cfg.get_path("losses.lambda_ib", 0.01))
    lam_int = float(cfg.get_path("losses.lambda_intervene", 1.0))
    lam_mass = float(cfg.get_path("losses.lambda_mass", 0.1))
    lam_nonneg = float(cfg.get_path("losses.lambda_nonneg", 1.0))
    lam_spec = float(cfg.get_path("losses.lambda_spectral", 0.05))
    lam_cont = float(cfg.get_path("losses.lambda_continuity", 0.1))

    # Reconstruction term: optionally HKO-7 balanced (up-weight heavy-rain pixels) and/or
    # validity-masked. Composes with (does not replace) the physics terms below.
    if bool(cfg.get_path("losses.balanced_loss", False)):
        thr = list(cfg.get_path("losses.balanced_thresholds", [16, 74, 133, 181]))
        wts = list(cfg.get_path("losses.balanced_weights", [1, 2, 5, 10, 30]))
        wmap = balanced_weight_map(target_field, thr, wts, valid_mask)
        render = balanced_mse(pred_field, target_field, wmap)
    elif valid_mask is not None:
        vm = valid_mask.to(dtype=pred_field.dtype, device=pred_field.device)
        render = (vm * (pred_field - target_field) ** 2).sum() / vm.sum().clamp_min(1e-6)
    else:
        render = torch.mean((pred_field - target_field) ** 2)
    ib = soft_ib_penalty(asg_cont, lam_ib)
    mass = physics.mass_budget_residual(pred_field, _as_float_tensor(growth_budget, pred_field))
    nonneg = physics.nonneg_penalty(pred_field)
    spectral = physics.spectral_loss(pred_field, target_field)

    # Continuity residual on the field advected one step vs the target (PINN term).
    g_t = pred_field
    g_th = target_field
    continuity = physics.continuity_residual(g_t, g_th, flow, dt=1.0)

    # Intervention term is supplied by the caller through `tier2_total_loss` only when a
    # paired render is available; here it defaults to zero and is added by the loop via
    # `intervention_consistency_loss`. Kept in the dict so the key always exists.
    intervene = torch.zeros((), device=pred_field.device, dtype=pred_field.dtype)

    total = (
        render
        + ib
        + lam_int * intervene
        + lam_mass * mass
        + lam_nonneg * nonneg
        + lam_spec * spectral
        + lam_cont * continuity
    )
    return {
        "total": total,
        "render": render.detach(),
        "ib": ib.detach(),
        "intervene": intervene.detach(),
        "mass": mass.detach(),
        "nonneg": nonneg.detach(),
        "spectral": spectral.detach(),
        "continuity": continuity.detach(),
    }


def intervention_consistency_loss(
    renderer,
    Z: torch.Tensor,
    advect_blind: torch.Tensor,
    perturb_fn: Callable[[torch.Tensor], torch.Tensor],
    expected_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int = 1,
) -> torch.Tensor:
    """Paired (orig, perturbed) render whose *difference* must match the perturbation's
    predicted effect (architecture.md section 6).

    The renderer is run twice: once on `Z` and once on `perturb_fn(Z)` (a structured
    perturbation of the ASG channels, e.g. a translate / regime-flip / growth-scale).
    `expected_fn(field_orig, Z_perturbed)` returns the field change the perturbation is
    *supposed* to produce; the loss is the MSE between the observed difference and the
    expected difference. This is the C-i training signal that makes the renderer causally
    responsive to the explicit state.

    Args:
        renderer:     object exposing `.sample(Z, advect_blind, steps) -> field`.
        Z:            [B,Cz,H,W] bottleneck tensor.
        advect_blind: [B,1,H,W] future-blind advection path.
        perturb_fn:   Z -> Z' structured perturbation of the ASG channels.
        expected_fn:  (field_orig, Z') -> expected [B,1,H,W] field change.
        steps:        few-step flow integration steps.
    Returns:
        scalar consistency loss.
    """
    field_orig = renderer.sample(Z, advect_blind, steps=steps)
    Z_pert = perturb_fn(Z)
    field_pert = renderer.sample(Z_pert, advect_blind, steps=steps)
    observed_delta = field_pert - field_orig
    expected_delta = expected_fn(field_orig, Z_pert)
    expected_delta = expected_delta.to(device=observed_delta.device, dtype=observed_delta.dtype)
    return torch.mean((observed_delta - expected_delta) ** 2)
