"""Stage-C latent rectified-flow renderer (architecture.md sections 4-6).

The renderer's only window onto the future is the explicit bottleneck state
``Z = [ASG_{t+h} (+) advect_blind(X_t)]``. It produces the field as a
*residual on advection* in the VAE latent:

    field = advect_blind + decode( Delta_latent ),
    Delta_latent = rectified_flow(Z) integrated from 0.

This decomposition makes the bottleneck behaviour crisp (architecture.md s.5):
zeroed ASG channels in ``Z`` -> the flow learns ``Delta_latent -> 0`` ->
``field -> advect_blind`` (pure advection collapse, the C-ii guarantee).

Training is rectified-flow / flow-matching (`liu2023rectifiedflow`,
`lipman2023flowmatching`) on the *residual latent*
``r = encode(target) - encode(advect_blind)``: a linear interpolation path
``x_tau = (1 - tau) * noise + tau * r`` with an MSE objective on the predicted
velocity ``v = r - noise``. Sampling integrates the learned velocity with a few
Euler steps (``cfg.stage_c.flow_steps``), which is cheap to train *and* sample
(architecture.md section 7).

All heavy deps are optional; with the IdentityVAE fallback this runs on CPU.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils.config import Config
from .unet import ConditionalUNet
from .vae import VAEWrapper

# The bottleneck module may not be on disk yet (sibling agent). Import its
# zeroing helper if present; otherwise provide an equivalent local fallback.
# The number of Z channels (Cz) is read from the tensor at runtime, so the
# renderer never hardcodes the bottleneck layout.
try:  # pragma: no cover - depends on sibling module presence
    from .bottleneck import zero_asg_in_Z as _bottleneck_zero_asg
except Exception:  # pragma: no cover
    _bottleneck_zero_asg = None


def _zero_asg_in_Z(Z: torch.Tensor) -> torch.Tensor:
    """Zero the ASG channels of ``Z``, keeping only the last (advect_blind) channel.

    Mirrors ``bottleneck.zero_asg_in_Z`` (C-ii zeroed condition). The bottleneck
    builds ``Z`` as ``[asg channels ... , advect_blind]`` with the advection
    field as the final channel, so the fallback keeps the last channel and zeros
    the rest. Prefers the real bottleneck helper when importable.
    """
    if _bottleneck_zero_asg is not None:
        return _bottleneck_zero_asg(Z)
    out = torch.zeros_like(Z)
    out[:, -1:] = Z[:, -1:]
    return out


class LatentRectifiedFlowRenderer(nn.Module):
    """Latent rectified-flow renderer with residual-on-advection structure.

    Conditions on the bottleneck ``Z`` (projected to latent resolution) and
    predicts the residual latent ``Delta_latent`` such that
    ``field = advect_blind + decode(Delta_latent)`` (architecture.md section 5).
    """

    def __init__(
        self,
        vae: VAEWrapper,
        cond_ch: int,
        flow_steps: int = 4,
        unet_base: int = 128,
        ensemble_k: int = 10,
    ) -> None:
        super().__init__()
        self.vae = vae
        self.latent_channels = vae.latent_channels
        self.downscale = vae.downscale
        self.cond_ch = int(cond_ch)
        self.flow_steps = int(flow_steps)
        self.ensemble_k = int(ensemble_k)

        # Project the bottleneck Z (at field resolution) to latent-resolution
        # conditioning channels. A lightweight conv stem; spatial downscale to
        # the latent grid is handled by adaptive pooling in `_project_cond`.
        self.cond_proj = nn.Sequential(
            nn.Conv2d(self.cond_ch, self.cond_ch, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(self.cond_ch, self.cond_ch, 3, padding=1),
        )

        self.unet = ConditionalUNet(
            in_ch=self.latent_channels,
            cond_ch=self.cond_ch,
            base=int(unet_base),
        )

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Config, cond_ch: Optional[int] = None) -> "LatentRectifiedFlowRenderer":
        """Build from ``cfg.stage_c`` (+ a frozen :class:`VAEWrapper`).

        ``cond_ch`` defaults to the bottleneck ``Cz`` if discoverable, else a
        documented fallback of 4 (3 ASG field channels: intensity + 2 motion,
        plus 1 advect_blind) consistent with ``bottleneck.build_Z``. The exact
        ``cond_ch`` can also be passed explicitly by the trainer that owns Z.
        """
        vae = VAEWrapper.from_config(cfg)
        if cond_ch is None:
            cond_ch = _infer_cz(cfg)
        return cls(
            vae=vae,
            cond_ch=int(cond_ch),
            flow_steps=int(cfg.get_path("stage_c.flow_steps", 4)),
            unet_base=int(cfg.get_path("stage_c.unet_base", 128)),
            ensemble_k=int(cfg.get_path("stage_c.ensemble_k", 10)),
        )

    # ------------------------------------------------------------------ #
    def _project_cond(self, Z: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Project ``Z[B,Cz,H,W]`` to latent-resolution conditioning ``[B,Cz,h,w]``."""
        cond = self.cond_proj(Z)
        if cond.shape[-2:] != (h, w):
            cond = F.adaptive_avg_pool2d(cond, (h, w))
        return cond

    def _encode_residual_target(
        self, advect_blind: torch.Tensor, target_field: torch.Tensor
    ) -> torch.Tensor:
        """Residual latent target ``r = encode(target) - encode(advect_blind)``.

        When ``target == advect_blind`` (no ASG-driven change), ``r == 0``, so
        the flow is supervised toward a zero residual — the structural basis for
        the zeroed-ASG -> advection collapse (architecture.md sections 4-5).
        """
        lat_target = self.vae.encode(target_field)
        lat_advect = self.vae.encode(advect_blind)
        return lat_target - lat_advect

    # ------------------------------------------------------------------ #
    def training_loss(
        self,
        Z: torch.Tensor,
        advect_blind: torch.Tensor,
        target_field: torch.Tensor,
    ) -> torch.Tensor:
        """Rectified-flow loss on the residual latent (flow matching, MSE on velocity).

        Linear interpolation path ``x_tau = (1 - tau) * z0 + tau * r`` with
        ``z0 ~ N(0, I)``; the target velocity is ``r - z0`` and the U-Net
        predicts it conditioned on ``(tau, cond)`` (architecture.md section 5).

        Args:
            Z:            ``[B, Cz, H, W]`` bottleneck tensor.
            advect_blind: ``[B, 1, H, W]`` future-blind advection of X_t.
            target_field: ``[B, 1, H, W]`` ground-truth future field.
        Returns:
            scalar flow-matching loss.
        """
        r = self._encode_residual_target(advect_blind, target_field)  # [B,Cl,h,w]
        b, _, h, w = r.shape
        cond = self._project_cond(Z, h, w)

        z0 = torch.randn_like(r)
        tau = torch.rand(b, device=r.device, dtype=r.dtype)
        tau_b = tau[:, None, None, None]
        x_tau = (1.0 - tau_b) * z0 + tau_b * r
        v_target = r - z0
        v_pred = self.unet(x_tau, tau, cond)
        return F.mse_loss(v_pred, v_target)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _integrate(
        self, Z: torch.Tensor, h: int, w: int, steps: int, z0: torch.Tensor
    ) -> torch.Tensor:
        """Few-step forward Euler integration of the learned velocity 0 -> 1.

        Returns the integrated residual latent ``Delta_latent``.
        """
        cond = self._project_cond(Z, h, w)
        x = z0
        dt = 1.0 / max(steps, 1)
        for i in range(max(steps, 1)):
            tau = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            v = self.unet(x, tau, cond)
            x = x + dt * v
        return x

    def sample(
        self,
        Z: torch.Tensor,
        advect_blind: torch.Tensor,
        steps: Optional[int] = None,
        z0: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Render a field: ``advect_blind + decode(Delta_latent)``.

        Few-step Euler integration of the rectified flow (residual-on-advection).
        ``steps=1`` gives the deterministic renderer used for Tier-0 pixel
        parity; the default uses ``cfg.stage_c.flow_steps``.

        Args:
            Z:            ``[B, Cz, H, W]`` bottleneck tensor.
            advect_blind: ``[B, 1, H, W]`` future-blind advection of X_t.
            steps:        Euler steps (default ``self.flow_steps``).
            z0:           optional starting noise ``[B, Cl, h, w]`` (for ensembles).
        Returns:
            field ``[B, 1, H, W]``.
        """
        steps = self.flow_steps if steps is None else int(steps)
        b = advect_blind.shape[0]
        h = advect_blind.shape[-2] // self.downscale
        w = advect_blind.shape[-1] // self.downscale
        if z0 is None:
            z0 = torch.randn(
                b, self.latent_channels, h, w, device=advect_blind.device, dtype=advect_blind.dtype
            )
        delta_latent = self._integrate(Z, h, w, steps, z0)
        residual = self.vae.decode(delta_latent)
        # Decoded residual is at latent*downscale resolution; align to field grid.
        if residual.shape[-2:] != advect_blind.shape[-2:]:
            residual = F.interpolate(
                residual, size=advect_blind.shape[-2:], mode="bilinear", align_corners=False
            )
        return advect_blind + residual

    def sample_ensemble(
        self,
        Z: torch.Tensor,
        advect_blind: torch.Tensor,
        k: Optional[int] = None,
        steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Draw ``k`` ensemble renders by varying the flow's start noise.

        Returns ``[B, k, 1, H, W]`` for CRPS/reliability (training_method.md s.4).
        """
        k = self.ensemble_k if k is None else int(k)
        members = [self.sample(Z, advect_blind, steps=steps) for _ in range(k)]
        return torch.stack(members, dim=1)

    # ------------------------------------------------------------------ #
    def sample_zeroed(
        self, Z: torch.Tensor, advect_blind: torch.Tensor, steps: Optional[int] = None
    ) -> torch.Tensor:
        """Render with ASG channels zeroed in ``Z`` (C-ii diagnostic helper).

        With a trained renderer this collapses toward pure advection because the
        residual is supervised to 0 when no ASG signal drives change.
        """
        return self.sample(_zero_asg_in_Z(Z), advect_blind, steps=steps)


def _infer_cz(cfg: Config) -> int:
    """Best-effort discovery of the bottleneck channel count ``Cz``.

    Tries ``bottleneck.build_Z`` channel count if importable; otherwise returns
    the documented default of 4 (1 intensity + 2 motion + 1 growth ASG channels
    is 4, plus advect_blind would be 5 — but if a growth channel is disabled the
    common layout is 3 ASG + 1 advect = 4). The trainer that constructs ``Z``
    should pass ``cond_ch`` explicitly to be exact.
    """
    try:  # pragma: no cover - depends on sibling module presence
        from . import bottleneck as _bn

        cz = getattr(_bn, "Cz", None) or getattr(_bn, "CZ", None) or getattr(_bn, "Z_CHANNELS", None)
        if cz is not None:
            return int(cz)
    except Exception:
        pass
    # ASG channels (intensity + vy + vx + optional growth) + advect_blind.
    use_growth = bool(cfg.get_path("bottleneck.use_growth_field", True))
    return (4 if use_growth else 3) + 1
