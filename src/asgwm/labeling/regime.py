"""Tendency + regime labeling for ASG auto-labeling (datasource.md section 2, steps 2-3).

From a track's peak-intensity time series we derive:
  * growth_scalar: the dVIL/dt tendency over the window (the ASG `growth` field).
  * classify_regime: a categorical regime in REGIMES = {init, grow, decay, steady}
    from morphology (track age, area trend) + tendency (growth sign).

Regime rules (morphology + tendency, datasource.md section 2 step 3):
  - init:   young track (just appeared) AND area is increasing.
  - grow:   intensity tendency clearly positive.
  - decay:  intensity tendency clearly negative.
  - steady: otherwise (advecting with little net intensity change).
"""
from __future__ import annotations

from typing import List

import numpy as np

from asgwm.asg import REGIMES

# Tendency thresholds on dVIL/dt (per minute). Tuned for VIL-like magnitudes; the
# pipeline normalizes peaks to a dBZ-like scale before calling these, so the bands
# are deliberately small.
_GROW_THRESH = 0.05
_DECAY_THRESH = -0.05
_YOUNG_FRAMES = 3  # a track <= this many samples is "young" -> candidate for init


def growth_scalar(peak_series: List[float], dt_min: float) -> float:
    """dVIL/dt tendency over the window (ASG growth scalar; datasource.md section 2).

    Fits a least-squares line to the peak-intensity series vs time (minutes) and
    returns the slope. Robust to a single sample (returns 0). Falls back to a
    simple endpoint difference when the series is degenerate.

    Args:
        peak_series: per-frame peak intensities along the track.
        dt_min: minutes between consecutive samples.

    Returns:
        Tendency (intensity units per minute), float.
    """
    y = np.asarray(peak_series, dtype=np.float64)
    n = y.size
    if n < 2 or dt_min <= 0:
        return 0.0
    times = np.arange(n, dtype=np.float64) * float(dt_min)
    t_mean = times.mean()
    y_mean = y.mean()
    denom = float(((times - t_mean) ** 2).sum())
    if denom <= 1e-12:
        return float((y[-1] - y[0]) / (times[-1] - times[0] + 1e-9))
    slope = float(((times - t_mean) * (y - y_mean)).sum() / denom)
    return slope


def _area_trend(area_series: List[float]) -> float:
    """Sign-aware normalized area change: (last - first) / max(first, eps)."""
    a = np.asarray(area_series, dtype=np.float64)
    if a.size < 2:
        return 0.0
    return float((a[-1] - a[0]) / max(a[0], 1.0))


def classify_regime(track: dict, dt_min: float) -> str:
    """Classify a track into a regime in REGIMES (datasource.md section 2 step 3).

    Args:
        track: a tracking dict with 'frames':[{'t','cy','cx','area','peak'}, ...].
        dt_min: minutes between consecutive frames.

    Returns:
        One of REGIMES = {'init','grow','decay','steady'}.
    """
    frames = track.get("frames", [])
    if not frames:
        return "steady"

    peaks = [f["peak"] for f in frames]
    areas = [f["area"] for f in frames]
    g = growth_scalar(peaks, dt_min)
    area_trend = _area_trend(areas)
    age = len(frames)
    starts_late = frames[0]["t"] > 0  # appeared after the first observed frame

    # Initiation: a young cell that recently appeared and is expanding/strengthening.
    if age <= _YOUNG_FRAMES and starts_late and (area_trend > 0.1 or g > 0.0):
        return "init"
    if g >= _GROW_THRESH:
        return "grow"
    if g <= _DECAY_THRESH:
        return "decay"
    return "steady"


def global_regime(tracks: List[dict], dt_min: float) -> str:
    """Domain-level regime: the regime of the dominant (most intense) track.

    Falls back to a vote over all tracks weighted by peak intensity. Used for the
    ASG GLOBAL line (architecture.md section 9).
    """
    if not tracks:
        return "steady"
    votes = {r: 0.0 for r in REGIMES}
    for tr in tracks:
        r = classify_regime(tr, dt_min)
        weight = max((f["peak"] for f in tr["frames"]), default=0.0)
        votes[r] += weight
    return max(votes.items(), key=lambda kv: kv[1])[0]
