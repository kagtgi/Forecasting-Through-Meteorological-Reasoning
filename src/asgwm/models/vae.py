"""VAE wrapper for the Stage-C latent renderer (architecture.md section 5).

The renderer operates in a frozen VAE latent (~8x spatial compression, SD-VAE
style, `rombach2022ldm`): a 384x384 VIL field -> ~48x48 latent; patches of
128x128 -> 16x16 latent tiles. This module wraps an optional diffusers
``AutoencoderKL`` and falls back to a parameter-free :class:`IdentityVAE`
(1->1 channels, no spatial downscale) when ``diffusers`` is unavailable, so the
whole pipeline runs end-to-end on CPU with no download.

Conventions: fields are single-channel VIL tensors ``[B, 1, H, W]``; latents are
``[B, latent_channels, h, w]`` with ``h = H // downscale``.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:  # optional heavy dependency
    from diffusers import AutoencoderKL  # type: ignore

    _HAS_DIFFUSERS = True
except Exception:  # pragma: no cover - exercised when diffusers is absent
    AutoencoderKL = None  # type: ignore
    _HAS_DIFFUSERS = False

from ..utils.config import Config


# Standard SD-VAE latent scaling factor (`rombach2022ldm`).
_SD_VAE_SCALE = 0.18215


class IdentityVAE(nn.Module):
    """Parameter-free fallback VAE: identity encode/decode, no compression.

    Maps a single-channel field to a ``latent_channels``-channel "latent" by
    repeating the channel, and back by averaging. ``downscale`` is 1 so latent
    spatial dims equal field spatial dims. This keeps the renderer's residual-on
    -advection contract intact (``decode(0) == 0``) on CPU without any model
    download (architecture.md section 5; CPU-fallback coding standard).
    """

    def __init__(self, latent_channels: int = 4) -> None:
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.downscale = 1
        self.scaling_factor = 1.0

    def encode(self, field: torch.Tensor) -> torch.Tensor:
        # [B,1,H,W] -> [B,latent_channels,H,W] by channel repeat (mean-preserving).
        return field.expand(-1, self.latent_channels, -1, -1).contiguous()

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # [B,latent_channels,h,w] -> [B,1,h,w] by channel mean (inverse of repeat).
        return latent.mean(dim=1, keepdim=True)


class VAEWrapper(nn.Module):
    """Frozen VAE encode/decode wrapper with a CPU IdentityVAE fallback.

    The wrapped autoencoder is frozen (no gradients, eval mode); only its
    encode/decode transforms are used by the Stage-C renderer. Exposes
    ``.downscale`` and ``.latent_channels`` so the renderer can size its latent
    U-Net regardless of which backend is active.
    """

    def __init__(
        self,
        backend: Optional[nn.Module] = None,
        latent_channels: int = 4,
        downscale: int = 8,
        scaling_factor: float = _SD_VAE_SCALE,
    ) -> None:
        super().__init__()
        if backend is None:
            backend = IdentityVAE(latent_channels=latent_channels)
        self.backend = backend
        self.is_identity = isinstance(backend, IdentityVAE)
        # IdentityVAE dictates its own (1x) downscale / channel count.
        self.latent_channels = int(getattr(backend, "latent_channels", latent_channels))
        self.downscale = int(getattr(backend, "downscale", downscale))
        self.scaling_factor = float(getattr(backend, "scaling_factor", scaling_factor))
        # Freeze: the VAE is never trained (architecture.md section 7).
        for p in self.backend.parameters():
            p.requires_grad_(False)
        self.backend.eval()

    # ------------------------------------------------------------------ #
    @classmethod
    def from_config(cls, cfg: Config) -> "VAEWrapper":
        """Build from ``cfg.stage_c`` (vae, vae_downscale, latent_channels).

        Loads the diffusers ``AutoencoderKL`` named ``cfg.stage_c.vae`` if
        diffusers is present; otherwise returns an IdentityVAE-backed wrapper so
        the pipeline runs on CPU (architecture.md section 5).
        """
        latent_channels = int(cfg.get_path("stage_c.latent_channels", 4))
        downscale = int(cfg.get_path("stage_c.vae_downscale", 8))
        vae_name = cfg.get_path("stage_c.vae", "stabilityai/sd-vae-ft-mse")

        if _HAS_DIFFUSERS and vae_name:
            try:
                backend = AutoencoderKL.from_pretrained(vae_name)
                return cls(
                    backend=_DiffusersVAEAdapter(backend),
                    latent_channels=latent_channels,
                    downscale=downscale,
                )
            except Exception:
                # Network/model unavailable -> fall back rather than crash.
                pass
        return cls(
            backend=IdentityVAE(latent_channels=latent_channels),
            latent_channels=latent_channels,
            downscale=downscale,
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def encode(self, field: torch.Tensor) -> torch.Tensor:
        """Encode a field ``[B,1,H,W]`` to a latent ``[B,latent_channels,h,w]``."""
        if field.dim() != 4 or field.shape[1] != 1:
            raise ValueError(f"VAEWrapper.encode expects [B,1,H,W]; got {tuple(field.shape)}")
        return self.backend.encode(field)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a latent ``[B,latent_channels,h,w]`` to a field ``[B,1,h*ds,w*ds]``."""
        return self.backend.decode(latent)


class _DiffusersVAEAdapter(nn.Module):
    """Adapts a diffusers ``AutoencoderKL`` to the (1-channel field) interface.

    SD-VAE expects 3-channel RGB input in roughly ``[-1, 1]``; VIL fields are
    single-channel. We tile the channel to 3 on encode and average the 3
    channels back to 1 on decode, applying the standard latent scaling factor.
    """

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae
        self.latent_channels = int(getattr(vae.config, "latent_channels", 4))
        self.downscale = int(2 ** (len(getattr(vae.config, "block_out_channels", [0, 0, 0, 0])) - 1))
        self.scaling_factor = float(getattr(vae.config, "scaling_factor", _SD_VAE_SCALE))

    def encode(self, field: torch.Tensor) -> torch.Tensor:
        rgb = field.expand(-1, 3, -1, -1)
        posterior = self.vae.encode(rgb).latent_dist
        return posterior.mean * self.scaling_factor

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        rgb = self.vae.decode(latent / self.scaling_factor).sample
        return rgb.mean(dim=1, keepdim=True)
