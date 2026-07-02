"""The faithful bottleneck — the only path from the world-model state to the pixels.

This module implements the core mechanism of ASG-WM (architecture.md section 4,
philosophy.md section 3.3): the renderer input is *exactly*

    Z = [ ASG_{t+h}  ⊕  advect_blind(X_t) ],

with no raw input latents and no encoder skip connections. Because ``Z`` is the only
future-bearing path to ``X_{t+1:t+n}``, two architectural guarantees follow and are
tested in eval.md section C:

* **C-ii (zeroed):** keep only ``advect_blind`` (``zero_asg_in_Z``) ⇒ the renderer must
  collapse to pure advection.
* **C-i (perturbed):** perturbing ASG_{t+h} ⇒ the output must change in the implied
  direction (intervention consistency).

It also exposes the *soft* Information-Bottleneck penalty (training_method.md section 4,
item 2): a KL of the continuous ASG sub-fields against a unit Gaussian, weighted by
``cfg.bottleneck.lambda_ib``, which suppresses memorization of residual input detail
that would bypass the hard structural cap.

Channel layout of Z (fixed and documented — consumers rely on it verbatim):

    index 0          : object Gaussian intensity field (sum of per-object blobs)
    index 1          : motion-y field  (vy, intensity-weighted)
    index 2          : motion-x field  (vx, intensity-weighted)
    index 3          : growth field     (per-object growth, intensity-weighted)
    index 4          : advect_blind     (future-blind extrapolation of X_t)

So ``asg_to_field_channels`` returns ``C_ASG = 4`` channels and ``build_Z`` returns
``Cz = 5`` channels.  ``N_ASG_CHANNELS`` / ``N_Z_CHANNELS`` below are the public
constants for these counts.
"""
from __future__ import annotations

from typing import Optional

import torch

from asgwm.asg import ASG, intensity_class

# ---------------------------------------------------------------------------
# Fixed, documented channel budget for the bottleneck.
# ---------------------------------------------------------------------------
N_ASG_CHANNELS: int = 4   # intensity, motion-y, motion-x, growth
N_ADVECT_CHANNELS: int = 1
N_Z_CHANNELS: int = N_ASG_CHANNELS + N_ADVECT_CHANNELS  # Cz = 5

# Channel indices within Z (for zeroing / inspection).
IDX_INTENSITY: int = 0
IDX_MOTION_Y: int = 1
IDX_MOTION_X: int = 2
IDX_GROWTH: int = 3
IDX_ADVECT: int = 4

# Fraction of an object's equivalent-area radius used as the Gaussian sigma.
_SIGMA_AREA_FRAC: float = 0.5
_MIN_SIGMA: float = 1.5  # pixels — keep a tiny blob even for point-like cells


def _object_sigma(area: float) -> float:
    """Gaussian spread (pixels) for a cell of the given area (km^2 ≈ px^2).

    Uses the equivalent-circle radius r = sqrt(area / pi) scaled by a fixed fraction.
    """
    area = max(float(area), 0.0)
    r = (area / 3.141592653589793) ** 0.5
    return max(_SIGMA_AREA_FRAC * r, _MIN_SIGMA)


def asg_to_field_channels(asg: ASG, H: int, W: int) -> torch.Tensor:
    """Rasterize an ASG into the dense ASG field channels (architecture.md section 4).

    Each storm object is painted as a Gaussian intensity blob centred at its centroid;
    the blob amplitude is its (dBZ) peak.  Two motion channels (vy, vx) and one growth
    channel carry the object's vector / tendency attributes, intensity-weighted so they
    are localized to where the object actually is.

    Args:
        asg: the (predicted) Atmospheric Scene Graph to rasterize.
        H, W: target grid size in pixels.

    Returns:
        Tensor ``[N_ASG_CHANNELS, H, W]`` (= ``[4, H, W]``):
        ``[intensity, motion_y, motion_x, growth]``.
    """
    device = torch.device("cpu")
    dtype = torch.float32
    intensity = torch.zeros(H, W, dtype=dtype, device=device)
    motion_y = torch.zeros(H, W, dtype=dtype, device=device)
    motion_x = torch.zeros(H, W, dtype=dtype, device=device)
    growth = torch.zeros(H, W, dtype=dtype, device=device)
    weight = torch.zeros(H, W, dtype=dtype, device=device)  # for intensity-weighted avg

    ys = torch.arange(H, dtype=dtype, device=device).view(H, 1)
    xs = torch.arange(W, dtype=dtype, device=device).view(1, W)

    for o in asg.objects:
        sigma = _object_sigma(o.area)
        cy = float(o.cy)
        cx = float(o.cx)
        # Per-object Gaussian footprint (unit-peak).
        d2 = (ys - cy) ** 2 + (xs - cx) ** 2
        blob = torch.exp(-d2 / (2.0 * sigma * sigma))  # [H, W], peak 1.0
        amp = float(o.peak)
        intensity = intensity + amp * blob
        # Intensity-weighted accumulation of vector / tendency attributes.
        w = blob
        weight = weight + w
        motion_y = motion_y + float(o.vy) * w
        motion_x = motion_x + float(o.vx) * w
        growth = growth + float(o.growth) * w

    # Normalize the weighted attribute channels back to per-pixel attribute values.
    safe_w = weight.clamp(min=1e-6)
    motion_y = motion_y / safe_w
    motion_x = motion_x / safe_w
    growth = growth / safe_w
    # Outside any blob the attributes are zero (no spurious far-field motion/growth).
    nodata = weight <= 1e-6
    motion_y = motion_y.masked_fill(nodata, 0.0)
    motion_x = motion_x.masked_fill(nodata, 0.0)
    growth = growth.masked_fill(nodata, 0.0)

    return torch.stack([intensity, motion_y, motion_x, growth], dim=0)


def build_Z(asg_th: ASG, advect_blind: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Assemble the faithful bottleneck input ``Z = [ASG_{t+h} ⊕ advect_blind]``.

    This is the *entire* contract between Stage B and Stage C (architecture.md
    sections 4 and 8): the ASG field channels concatenated with the future-blind
    advection of the present.  Nothing else (no raw latents, no skip connections)
    is allowed onto this path.

    Args:
        asg_th: predicted ASG at horizon t+h.
        advect_blind: ``[1, H, W]`` future-blind extrapolation of X_t.
        H, W: grid size in pixels.

    Returns:
        ``Z`` of shape ``[Cz, H, W]`` with ``Cz == N_Z_CHANNELS`` (= 5).
    """
    asg_channels = asg_to_field_channels(asg_th, H, W)  # [4, H, W]
    adv = advect_blind
    if adv.dim() == 2:
        adv = adv.unsqueeze(0)  # [1, H, W]
    if adv.shape[-2:] != (H, W):
        adv = torch.nn.functional.interpolate(
            adv.unsqueeze(0).float(), size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0)
    adv = adv.to(asg_channels.dtype).to(asg_channels.device)
    z = torch.cat([asg_channels, adv[:N_ADVECT_CHANNELS]], dim=0)  # [5, H, W]
    return z


def zero_asg_in_Z(Z: torch.Tensor) -> torch.Tensor:
    """Zero the ASG channels of ``Z`` while keeping ONLY ``advect_blind`` (C-ii).

    This realizes the *zeroed* intervention of architecture.md section 4: with the
    ASG content removed, the renderer's only conditioning is the future-blind
    advection, so a faithful renderer must collapse to pure advection.

    Works on both ``[Cz, H, W]`` and batched ``[B, Cz, H, W]`` tensors.

    Args:
        Z: bottleneck tensor whose last-but-two dimension is the channel axis.

    Returns:
        A copy of ``Z`` with channels ``0..N_ASG_CHANNELS-1`` set to zero and the
        ``advect_blind`` channel untouched.
    """
    out = Z.clone()
    if Z.dim() == 3:  # [Cz, H, W]
        out[:N_ASG_CHANNELS] = 0.0
    elif Z.dim() == 4:  # [B, Cz, H, W]
        out[:, :N_ASG_CHANNELS] = 0.0
    else:
        raise ValueError(f"zero_asg_in_Z expects a 3D or 4D tensor, got {Z.dim()}D")
    return out


def soft_ib_penalty(asg_continuous: torch.Tensor, lambda_ib: float) -> torch.Tensor:
    """Soft variational Information-Bottleneck penalty (training_method.md section 4).

    KL divergence of the continuous ASG sub-fields against a unit Gaussian prior,
    treating the supplied tensor as the per-sample mean of a unit-variance posterior:

        KL( N(mu, 1) || N(0, 1) ) = 1/2 * mu^2.

    This suppresses memorization of residual input detail (sub-pixel centroid
    offsets, peak-intensity deviations) that would otherwise bypass the hard
    structural cap and leak through the bottleneck.

    Args:
        asg_continuous: any tensor of continuous ASG attributes (e.g. centroid
            offsets, peak deviations); shape is arbitrary.
        lambda_ib: penalty weight (``cfg.bottleneck.lambda_ib`` / ``cfg.losses.lambda_ib``).

    Returns:
        Scalar penalty ``lambda_ib * mean( 0.5 * mu^2 )``.
    """
    if not torch.is_tensor(asg_continuous):
        asg_continuous = torch.as_tensor(asg_continuous, dtype=torch.float32)
    asg_continuous = asg_continuous.float()
    if asg_continuous.numel() == 0:
        return asg_continuous.new_zeros(())
    kl = 0.5 * (asg_continuous ** 2)
    return float(lambda_ib) * kl.mean()
