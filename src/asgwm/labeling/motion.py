"""Motion estimation for ASG auto-labeling (datasource.md section 2, step 1).

Optical-flow motion vectors drive (a) the StormObject motion attribute, (b) the
semi-Lagrangian advection field that becomes Stage-C's *future-blind* path and the
Tier-0 baseline (architecture.md sections 4-5). We prefer pysteps Lucas-Kanade /
VET; when pysteps is unavailable we fall back to a dependency-free numpy
phase-correlation estimate so the whole pipeline runs CPU-only with no install.

Convention (matches asgwm.physics): the returned flow is [2, H, W] with channel 0
= vy (rows/step) and channel 1 = vx (cols/step), in *pixels per step* (one step =
one input frame interval).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# pysteps is an optional heavy dependency; guard the import.
try:  # pragma: no cover - exercised only when pysteps is installed
    from pysteps import motion as _pysteps_motion  # type: ignore

    _HAS_PYSTEPS = True
except Exception:  # pragma: no cover
    _pysteps_motion = None
    _HAS_PYSTEPS = False


def _normalize_frames(frames: np.ndarray) -> np.ndarray:
    """Coerce a frame stack to float32 [T, H, W]."""
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:  # single frame -> degenerate stack
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"estimate_motion expects [T,H,W]; got shape {arr.shape}")
    return arr


def _phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Per-step feature displacement (dy, dx) from frame `a` to later frame `b`.

    Normalized cross-power-spectrum phase correlation with sub-pixel parabolic
    peak refinement. Robust, dependency-free fallback used when pysteps is
    missing. Returns the displacement of intensity features over the a->b step in
    pixels (positive dy = downward/south, positive dx = rightward/east), matching
    the (vy, vx) convention of `asgwm.physics`.
    """
    h, w = a.shape
    fa = np.fft.fft2(a - a.mean())
    fb = np.fft.fft2(b - b.mean())
    cross = fa * np.conj(fb)
    # Regularized (not fully whitened) phase correlation: full whitening divides by
    # |cross| and amplifies high-frequency noise, which mislocates the peak for
    # smooth single-mode blobs. We damp the normalization with epsilon*max(|cross|)
    # so strong low-frequency modes (the actual storm signal) dominate the peak.
    mag = np.abs(cross)
    eps = 0.1 * float(mag.max() + 1e-8)
    r = cross / (mag + eps)
    corr = np.fft.ifft2(r).real
    pi, pj = np.unravel_index(int(np.argmax(corr)), corr.shape)

    # Sub-pixel refinement by parabolic interpolation around the integer peak
    # (recovers fractional drift that an integer-only peak rounds away to 0).
    def _subpix(c: np.ndarray, idx: int, n: int) -> float:
        im1 = c[(idx - 1) % n]
        ip1 = c[(idx + 1) % n]
        c0 = c[idx]
        denom2 = (im1 - 2.0 * c0 + ip1)
        if abs(denom2) < 1e-9:
            return 0.0
        return float(np.clip(0.5 * (im1 - ip1) / denom2, -0.5, 0.5))

    py = pi + _subpix(corr[:, pj], pi, h)
    px = pj + _subpix(corr[pi, :], pj, w)
    # Wrap peak indices > half-dimension to the negative side.
    if py > h / 2:
        py -= h
    if px > w / 2:
        px -= w
    # The peak of ifft(Fa . conj(Fb)) sits at the NEGATIVE of the feature
    # displacement for this (a=earlier, b=later) ordering; negate so positive
    # values mean features moved down/right (south/east).
    return float(-py), float(-px)


def _global_drift(frames: np.ndarray) -> Tuple[float, float]:
    """Robust per-step global drift (vy, vx) via end-to-end phase correlation.

    The first->last displacement integrated over the window is far less noisy than
    any single consecutive-pair peak; dividing by the number of steps yields a
    stable per-step velocity. Falls back to the median consecutive-pair shift if
    the endpoints are too flat to correlate.
    """
    t = frames.shape[0]
    pairs = max(1, t - 1)
    first, last = frames[0], frames[-1]
    if first.std() > 1e-4 and last.std() > 1e-4:
        tot_dy, tot_dx = _phase_correlation_shift(first, last)
        return tot_dy / pairs, tot_dx / pairs
    g_dy: list[float] = []
    g_dx: list[float] = []
    for k in range(pairs):
        a, b = frames[k], frames[k + 1]
        if a.std() < 1e-4 or b.std() < 1e-4:
            continue
        dy, dx = _phase_correlation_shift(a, b)
        g_dy.append(dy)
        g_dx.append(dx)
    return (
        float(np.median(g_dy)) if g_dy else 0.0,
        float(np.median(g_dx)) if g_dx else 0.0,
    )


def _block_phase_flow(frames: np.ndarray, n_blocks: int = 3) -> np.ndarray:
    """Dependency-free dense flow [2, H, W] via phase correlation.

    Strategy: a robust global drift (end-to-end phase correlation) is the base
    field, since a single advecting feature is well described by a uniform vector.
    Larger tiles (n_blocks x n_blocks) are then phase-correlated and used to
    REFINE the base only where a tile both (a) carries a large share of the storm
    energy and (b) yields a shift close to the global drift (so partial-blob /
    trailing-edge tiles cannot corrupt the field). Refined tile values are
    bilinearly interpolated to full resolution. With pysteps absent this gives a
    sensible — typically near-uniform — advection field for the future-blind path.
    """
    t, h, w = frames.shape
    by = max(1, h // n_blocks)
    bx = max(1, w // n_blocks)
    ny = max(1, h // by)
    nx = max(1, w // bx)

    global_vy, global_vx = _global_drift(frames)

    block_vy = np.full((ny, nx), global_vy, dtype=np.float32)
    block_vx = np.full((ny, nx), global_vx, dtype=np.float32)
    frame_energy = float(np.mean(frames ** 2) + 1e-8)
    # A tile may only refine toward its own estimate if it is consistent with the
    # global drift (within ~1 px/step); larger deviations are treated as noise.
    tol = 1.0
    for iy in range(ny):
        for ix in range(nx):
            y0, y1 = iy * by, min(h, (iy + 1) * by)
            x0, x1 = ix * bx, min(w, (ix + 1) * bx)
            a0 = frames[0, y0:y1, x0:x1]
            aL = frames[-1, y0:y1, x0:x1]
            if a0.shape[0] < 8 or a0.shape[1] < 8:
                continue
            if np.mean(a0 ** 2) < 0.75 * frame_energy or a0.std() < 1e-3 or aL.std() < 1e-3:
                continue
            dy, dx = _phase_correlation_shift(a0, aL)
            dy /= max(1, t - 1)
            dx /= max(1, t - 1)
            if abs(dy - global_vy) <= tol and abs(dx - global_vx) <= tol:
                block_vy[iy, ix] = dy
                block_vx[iy, ix] = dx

    yy = np.linspace(0, ny - 1, h)
    xx = np.linspace(0, nx - 1, w)
    dense_vy = _bilinear_upsample(block_vy, yy, xx)
    dense_vx = _bilinear_upsample(block_vx, yy, xx)
    return np.stack([dense_vy, dense_vx], axis=0).astype(np.float32)


def _bilinear_upsample(grid: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """Bilinearly sample `grid` at fractional indices yy (rows) x xx (cols)."""
    ny, nx = grid.shape
    y0 = np.floor(yy).astype(int)
    x0 = np.floor(xx).astype(int)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y0 = np.clip(y0, 0, ny - 1)
    x0 = np.clip(x0, 0, nx - 1)
    wy = (yy - np.floor(yy))[:, None]
    wx = (xx - np.floor(xx))[None, :]
    g00 = grid[np.ix_(y0, x0)]
    g01 = grid[np.ix_(y0, x1)]
    g10 = grid[np.ix_(y1, x0)]
    g11 = grid[np.ix_(y1, x1)]
    top = g00 * (1 - wx) + g01 * wx
    bot = g10 * (1 - wx) + g11 * wx
    return top * (1 - wy) + bot * wy


def estimate_motion(frames: np.ndarray) -> np.ndarray:
    """Estimate a dense motion field from a radar frame stack.

    Args:
        frames: [T, H, W] precipitation field history (e.g. VIL), T >= 1.

    Returns:
        flow [2, H, W] = (vy, vx) in pixels per frame-step, float32. The same field
        is reused as the future-blind advection source in `asgwm.data.advection`
        (datasource.md section 2; architecture.md section 4).
    """
    arr = _normalize_frames(frames)
    t, h, w = arr.shape
    if t < 2:  # cannot estimate motion from a single frame
        return np.zeros((2, h, w), dtype=np.float32)

    if _HAS_PYSTEPS:
        try:  # pragma: no cover - requires pysteps at runtime
            oflow = _pysteps_motion.get_method("LK")
            # pysteps expects [T, H, W]; returns [2, H, W] as (u=vx, v=vy) or (vy,vx)
            # depending on version. The Lucas-Kanade method returns advection field
            # with shape (2, m, n); channel order is (i.e. (V along axis0, U along
            # axis1)) -> (vy, vx). We pass through directly.
            uv = np.asarray(oflow(arr), dtype=np.float32)
            if uv.shape == (2, h, w):
                return uv
            # Some methods return (u, v) on the last axis; normalize.
            if uv.shape[-1] == 2 and uv.shape[0] == h:
                return np.stack([uv[..., 1], uv[..., 0]], axis=0).astype(np.float32)
        except Exception:
            pass  # fall through to numpy fallback

    return _block_phase_flow(arr)


def mean_motion(flow: np.ndarray, mask: Optional[np.ndarray] = None) -> tuple[float, float]:
    """Mean (vy, vx) over an optional boolean mask; whole field if mask is None."""
    vy, vx = flow[0], flow[1]
    if mask is not None and mask.any():
        return float(vy[mask].mean()), float(vx[mask].mean())
    return float(vy.mean()), float(vx.mean())
