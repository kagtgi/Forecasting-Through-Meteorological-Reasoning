"""Future-blind advection extrapolation (datasource.md section 2; architecture.md).

This is the **future-blind** path used in two places that must stay identical:
  * Stage-C's non-ASG channel ``advect_blind`` inside the faithful bottleneck Z, and
  * the Tier-0 *advection* baseline (``future_blind_baseline``).

"Future-blind" is the load-bearing property: the extrapolation is built **only** from
the observed history ``X<=t``. It estimates a motion field from the last history frames
and then semi-Lagrangian-advects the most recent frame forward ``n_out`` steps. No
future frame is ever consulted, so any skill the renderer shows beyond this baseline is
attributable to the ASG world model — not to leaked future information (eval.md C-iii
leakage audit).

The motion source mirrors the labeling pass (pysteps Lucas-Kanade when available,
numpy phase-correlation otherwise) so the baseline and the labels share kinematics.

Public surface (interface contract):
    advect_blind(frames_hist[T,H,W], n_out, motion=None) -> np.ndarray[n_out,H,W]
    future_blind_baseline(frames_hist, n_out) -> np.ndarray   # alias / Tier-0 baseline
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# pysteps gives Lucas-Kanade optical flow + extrapolation; optional (datasource.md s.2).
try:  # pragma: no cover - exercised only when pysteps is installed
    from pysteps import motion as _pysteps_motion  # type: ignore
    from pysteps import nowcasts as _pysteps_nowcasts  # type: ignore

    _HAS_PYSTEPS = True
except Exception:  # pragma: no cover
    _pysteps_motion = None  # type: ignore
    _pysteps_nowcasts = None  # type: ignore
    _HAS_PYSTEPS = False


# ---------------------------------------------------------------------------
# Motion estimation from history only (future-blind)
# ---------------------------------------------------------------------------
def _phase_correlation_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Estimate the (dy, dx) shift mapping frame ``a`` onto frame ``b`` via FFT.

    Returns sub-integer pixel displacement using the cross-power spectrum peak. This is
    the numpy fallback for global motion when pysteps is unavailable.
    """
    a = a - a.mean()
    b = b - b.mean()
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    denom = np.abs(R)
    denom[denom == 0] = 1.0
    R /= denom
    corr = np.fft.ifft2(R).real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = a.shape
    dy = peak[0] if peak[0] <= h // 2 else peak[0] - h
    dx = peak[1] if peak[1] <= w // 2 else peak[1] - w
    return float(dy), float(dx)


def estimate_history_motion(frames_hist: np.ndarray) -> np.ndarray:
    """Estimate a dense (vy, vx) motion field [2, H, W] from history frames only.

    Uses pysteps Lucas-Kanade if present (dense flow); otherwise estimates a single
    global translation via averaged phase correlation over consecutive history frames
    and broadcasts it to a constant field. Units: pixels per frame (step).
    """
    frames_hist = np.asarray(frames_hist, dtype=np.float32)
    T, H, W = frames_hist.shape
    if T < 2:
        return np.zeros((2, H, W), dtype=np.float32)

    if _HAS_PYSTEPS:  # pragma: no cover - needs pysteps installed
        try:
            oflow = _pysteps_motion.get_method("LK")
            # pysteps expects (T, H, W); returns (2, H, W) as (vx, vy) -> reorder to (vy, vx)
            uv = oflow(frames_hist)
            vx, vy = uv[0], uv[1]
            return np.stack([vy, vx], axis=0).astype(np.float32)
        except Exception:
            pass

    # numpy fallback: average consecutive-frame phase-correlation shifts.
    dys, dxs = [], []
    for t in range(1, T):
        dy, dx = _phase_correlation_shift(frames_hist[t - 1], frames_hist[t])
        dys.append(dy)
        dxs.append(dx)
    vy = float(np.median(dys)) if dys else 0.0
    vx = float(np.median(dxs)) if dxs else 0.0
    field = np.zeros((2, H, W), dtype=np.float32)
    field[0] = vy
    field[1] = vx
    return field


# ---------------------------------------------------------------------------
# Semi-Lagrangian warp (numpy fallback mirrors physics.semi_lagrangian_advect)
# ---------------------------------------------------------------------------
def _warp_backward(field: np.ndarray, vy: np.ndarray, vx: np.ndarray, dt: float) -> np.ndarray:
    """Backward semi-Lagrangian warp of a single [H, W] frame by (vy, vx) * dt.

    Samples the source upstream of the flow with bilinear interpolation and border
    padding — the numpy mirror of :func:`asgwm.physics.semi_lagrangian_advect`.
    """
    H, W = field.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    src_y = ys - vy * dt
    src_x = xs - vx * dt
    src_y = np.clip(src_y, 0.0, H - 1.0)
    src_x = np.clip(src_x, 0.0, W - 1.0)

    y0 = np.floor(src_y).astype(np.int64)
    x0 = np.floor(src_x).astype(np.int64)
    y1 = np.clip(y0 + 1, 0, H - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    wy = src_y - y0
    wx = src_x - x0

    v00 = field[y0, x0]
    v01 = field[y0, x1]
    v10 = field[y1, x0]
    v11 = field[y1, x1]
    top = v00 * (1 - wx) + v01 * wx
    bot = v10 * (1 - wx) + v11 * wx
    return (top * (1 - wy) + bot * wy).astype(np.float32)


# ---------------------------------------------------------------------------
# Public: future-blind extrapolation
# ---------------------------------------------------------------------------
def advect_blind(
    frames_hist: np.ndarray,
    n_out: int,
    motion: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Future-blind semi-Lagrangian extrapolation of the history (datasource.md s.2).

    Args:
        frames_hist: [T, H, W] observed history frames (X<=t). Only these are used.
        n_out:       number of future frames to extrapolate.
        motion:      optional precomputed (vy, vx) field [2, H, W] in px/step; if None
                     it is estimated from ``frames_hist`` (same source as labeling).
    Returns:
        [n_out, H, W] advected frames. Built ONLY from the history -> future-blind.
    """
    frames_hist = np.asarray(frames_hist, dtype=np.float32)
    if frames_hist.ndim != 3:
        raise ValueError(f"frames_hist must be [T,H,W], got {frames_hist.shape}")
    T, H, W = frames_hist.shape
    n_out = int(n_out)
    if n_out <= 0:
        return np.zeros((0, H, W), dtype=np.float32)

    if motion is None:
        motion = estimate_history_motion(frames_hist)
    motion = np.asarray(motion, dtype=np.float32)
    vy, vx = motion[0], motion[1]

    last = frames_hist[-1]

    if _HAS_PYSTEPS:  # pragma: no cover - needs pysteps installed
        try:
            extrapolate = _pysteps_nowcasts.get_method("extrapolation")
            # pysteps wants velocity as (vx, vy)
            uv = np.stack([vx, vy], axis=0)
            out = extrapolate(last, uv, n_out)
            out = np.nan_to_num(out, nan=0.0).astype(np.float32)
            if out.shape == (n_out, H, W):
                return out
        except Exception:
            pass

    # numpy fallback: cumulative semi-Lagrangian advection of the last frame.
    out = np.empty((n_out, H, W), dtype=np.float32)
    for k in range(n_out):
        out[k] = _warp_backward(last, vy, vx, dt=float(k + 1))
    return out


def future_blind_baseline(frames_hist: np.ndarray, n_out: int) -> np.ndarray:
    """Tier-0 advection baseline = :func:`advect_blind` with motion from history.

    Alias kept distinct for call-site readability (interface contract / eval.md).
    """
    return advect_blind(frames_hist, n_out, motion=None)
