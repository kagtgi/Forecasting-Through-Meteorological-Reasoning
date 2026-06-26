"""Small conditional latent U-Net for the Stage-C rectified-flow renderer.

Predicts the flow *velocity* ``v(x_t, t, cond)`` used by the few-step rectified
flow / flow-matching renderer (architecture.md section 5, `liu2023rectifiedflow`,
`lipman2023flowmatching`). The backbone is a compact GroupNorm + SiLU U-Net with
2-3 down/up stages so it stays inside the compute envelope (architecture.md
section 7) and runs on CPU.

The conditioning tensor ``cond`` (the bottleneck ``Z`` projected to latent
resolution) is concatenated channel-wise to ``x_t`` at the input; the diffusion
timestep ``t`` is injected as a sinusoidal embedding added to every residual
block (FiLM-free additive conditioning keeps it small and self-contained).
"""
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Sinusoidal timestep embedding (`vaswani2017attention` / DDPM style).

    Args:
        t:   ``[B]`` continuous timesteps (typically in ``[0, 1]`` for rectified flow).
        dim: embedding dimension (even).
    Returns:
        ``[B, dim]`` embedding.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=t.device)
        / max(half, 1)
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:  # pad to exact dim if odd
        emb = F.pad(emb, (0, 1))
    return emb


def _gn_groups(ch: int) -> int:
    """Pick a GroupNorm group count that divides ``ch`` (<= 8 groups)."""
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    """GroupNorm-SiLU residual block with additive timestep conditioning."""

    def __init__(self, in_ch: int, out_ch: int, t_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_gn_groups(in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.temb = nn.Linear(t_dim, out_ch)
        self.norm2 = nn.GroupNorm(_gn_groups(out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.op(x)


class ConditionalUNet(nn.Module):
    """Compact conditional U-Net predicting rectified-flow velocity.

    Args:
        in_ch:   number of channels of the (noisy) latent ``x_t`` and of the
                 predicted velocity ``v``.
        cond_ch: number of conditioning channels (the projected bottleneck Z).
        base:    base feature width; channel multipliers are ``[1, 2, 4]``.
        t_dim:   timestep-embedding dimension.

    forward(x_t[B,in,h,w], t[B], cond[B,cond_ch,h,w]) -> v[B,in,h,w].
    """

    def __init__(self, in_ch: int, cond_ch: int, base: int = 128, t_dim: int = 128) -> None:
        super().__init__()
        self.in_ch = int(in_ch)
        self.cond_ch = int(cond_ch)
        self.base = int(base)
        self.t_dim = int(t_dim)

        mults: List[int] = [1, 2, 4]
        chans = [base * m for m in mults]

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 4),
            nn.SiLU(),
            nn.Linear(t_dim * 4, t_dim),
        )

        # Input projection: x_t concatenated with cond.
        self.in_conv = nn.Conv2d(self.in_ch + self.cond_ch, chans[0], 3, padding=1)

        # ---- Encoder ----
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_chs: List[int] = [chans[0]]
        prev = chans[0]
        for i, ch in enumerate(chans):
            self.down_blocks.append(ResBlock(prev, ch, t_dim))
            skip_chs.append(ch)
            prev = ch
            if i < len(chans) - 1:
                self.downsamples.append(Downsample(ch))
            else:
                self.downsamples.append(nn.Identity())

        # ---- Bottleneck ----
        self.mid1 = ResBlock(prev, prev, t_dim)
        self.mid2 = ResBlock(prev, prev, t_dim)

        # ---- Decoder ----
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for i, ch in enumerate(reversed(chans)):
            skip = skip_chs.pop()
            self.up_blocks.append(ResBlock(prev + skip, ch, t_dim))
            prev = ch
            if i < len(chans) - 1:
                self.upsamples.append(Upsample(ch))
            else:
                self.upsamples.append(nn.Identity())

        self.out_norm = nn.GroupNorm(_gn_groups(prev), prev)
        self.out_conv = nn.Conv2d(prev, self.in_ch, 3, padding=1)
        # Zero-init the output so the initial velocity (and hence the residual)
        # is ~0 -> the untrained renderer starts at pure advection.
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.dim() == 0:
            t = t.expand(x_t.shape[0])
        temb = self.t_mlp(timestep_embedding(t, self.t_dim))

        h = self.in_conv(torch.cat([x_t, cond], dim=1))
        skips = [h]
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, temb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, temb)
        h = self.mid2(h, temb)

        for block, up in zip(self.up_blocks, self.upsamples):
            skip = skips.pop()
            h = block(torch.cat([h, skip], dim=1), temb)
            h = up(h)

        return self.out_conv(F.silu(self.out_norm(h)))
