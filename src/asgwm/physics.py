"""Differentiable physics operators (architecture.md sections 3, 5).

These are the governing-equation injection points used by Stage B (transition) and
Stage C (renderer): a semi-Lagrangian advection warp, a continuity / mass-conservation
residual, and a motion-field smoothness residual. They add loss terms + a warp, not
parameters. All operators are differentiable so they back-propagate through the
bottleneck (training_method.md section 4, L_continuity).

Conventions: image tensors are [B, C, H, W]; flow is [B, 2, H, W] with channel 0 = vy
(rows/step) and channel 1 = vx (cols/step), in *pixels per step*.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def _base_grid(h: int, w: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(0, h - 1, h, device=device, dtype=dtype)
    xs = torch.linspace(0, w - 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((gx, gy), dim=-1)  # [H, W, 2] in (x, y) for grid_sample


def semi_lagrangian_advect(field: torch.Tensor, flow: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Backward semi-Lagrangian warp: sample the source field upstream of the flow.

    Args:
        field: [B, C, H, W]
        flow:  [B, 2, H, W] (vy, vx) in pixels/step
        dt:    number of steps to advance
    Returns:
        advected field [B, C, H, W]
    """
    b, c, h, w = field.shape
    grid = _base_grid(h, w, field.device, field.dtype).unsqueeze(0).expand(b, -1, -1, -1).clone()
    vy = flow[:, 0]  # [B,H,W]
    vx = flow[:, 1]
    # backward trace: source = current - velocity*dt
    grid[..., 0] = grid[..., 0] - vx * dt   # x
    grid[..., 1] = grid[..., 1] - vy * dt   # y
    # normalize to [-1, 1]
    grid[..., 0] = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    return F.grid_sample(field, grid, mode="bilinear", padding_mode="border", align_corners=True)


def kmh_to_px_per_step(
    motion_kmh: torch.Tensor,
    km_per_pixel: float = 1.0,
    minutes_per_frame: float = 5.0,
) -> torch.Tensor:
    """Convert StormObject (vy, vx) from km/h to pixels per frame step.

    Matches the conversion used in ``labeling.pipeline._advect_objects`` and the Tier-0
    gate advection baseline: one frame step advances ``(minutes_per_frame / 60) / km_per_pixel``
    km per km/h of motion.
    """
    scale = (float(minutes_per_frame) / 60.0) / max(float(km_per_pixel), 1e-6)
    return motion_kmh * scale


def advect_points(centroids: torch.Tensor, motion: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Advance object centroids by their motion vectors.

    Args:
        centroids: [N, 2] as (cy, cx) in pixels
        motion:    [N, 2] as (vy, vx) in pixels/step
    """
    return centroids + motion * dt


def divergence(field_v: torch.Tensor) -> torch.Tensor:
    """Divergence of a 2-channel vector field [B,2,H,W] -> [B,1,H,W]."""
    vy, vx = field_v[:, 0:1], field_v[:, 1:2]
    dvx_dx = torch.gradient(vx, dim=3)[0]
    dvy_dy = torch.gradient(vy, dim=2)[0]
    return dvx_dx + dvy_dy


def continuity_residual(g_t: torch.Tensor, g_th: torch.Tensor, flow: torch.Tensor,
                        dt: float = 1.0) -> torch.Tensor:
    """Mass-conservation residual on the growth field: dg/dt + div(g v) ~= 0.

    Args:
        g_t, g_th: growth fields [B,1,H,W] at t and t+h
        flow:      [B,2,H,W] (vy,vx) pixels/step
    Returns:
        scalar mean-squared residual
    """
    dg_dt = (g_th - g_t) / dt
    flux = g_t * flow                          # [B,2,H,W]
    div_flux = divergence(flux)
    res = dg_dt + div_flux
    return (res ** 2).mean()


def motion_smoothness_residual(flow: torch.Tensor) -> torch.Tensor:
    """Penalize non-smooth motion fields (advection regularizer)."""
    dvy = torch.gradient(flow[:, 0:1], dim=(2, 3))
    dvx = torch.gradient(flow[:, 1:2], dim=(2, 3))
    terms = [t ** 2 for t in dvy] + [t ** 2 for t in dvx]
    return torch.stack([t.mean() for t in terms]).sum()


def nonneg_penalty(field: torch.Tensor) -> torch.Tensor:
    """Penalize negative precipitation values."""
    return F.relu(-field).mean()


def mass_budget_residual(field: torch.Tensor, target_total: torch.Tensor) -> torch.Tensor:
    """Match the rendered field's integrated content to an ASG-derived budget.

    The residual is normalised per-pixel so it stays commensurate with the per-pixel
    reconstruction MSE and is invariant to grid resolution. An *unnormalised* squared sum
    scales as (H*W*intensity)^2 — it would dwarf every other loss term and explode at full
    resolution (confirmed in a Tier-2 smoke: mass ~1e10 vs render ~1e3). Both ``integrated``
    and ``target_total`` are summed-scale, so dividing the difference by the pixel count
    compares mean intensities.

    Args:
        field:        [B,1,H,W] rendered field
        target_total: [B] target integrated content (from ASG growth budget)
    """
    n_pix = float(max(field.shape[-1] * field.shape[-2], 1))
    integrated = field.clamp(min=0).sum(dim=(1, 2, 3))
    return (((integrated - target_total) / n_pix) ** 2).mean()


def radial_power_spectrum(field: torch.Tensor, n_bins: int = 64) -> torch.Tensor:
    """Radially-averaged power spectral density [B, n_bins] (realism diagnostic)."""
    b, c, h, w = field.shape
    f = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))
    power = (f.abs() ** 2).mean(dim=1)  # [B,H,W]
    cy, cx = h // 2, w // 2
    ys = torch.arange(h, device=field.device) - cy
    xs = torch.arange(w, device=field.device) - cx
    rr = torch.sqrt((ys[:, None] ** 2 + xs[None, :] ** 2).float())
    rmax = rr.max()
    bins = (rr / (rmax + 1e-8) * (n_bins - 1)).long().clamp(0, n_bins - 1)  # [H,W]
    out = []
    flat_bins = bins.reshape(-1)
    for bi in range(b):
        p = power[bi].reshape(-1)
        acc = torch.zeros(n_bins, device=field.device).scatter_add(0, flat_bins, p)
        cnt = torch.zeros(n_bins, device=field.device).scatter_add(0, flat_bins, torch.ones_like(p))
        out.append(acc / cnt.clamp(min=1))
    return torch.stack(out, dim=0)


def spectral_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 between log radially-averaged power spectra of pred and target."""
    ps_p = torch.log1p(radial_power_spectrum(pred))
    ps_t = torch.log1p(radial_power_spectrum(target))
    return (ps_p - ps_t).abs().mean()
