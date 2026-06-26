"""Uniform-time-axis assembly for the OOD radar loaders (nexrad.py, mrms.py).

NEXRAD volume scans are irregular (~4-6 min, VCP-dependent) and MRMS is ~2 min; both must
be resampled onto the model's uniform 5-min axis, and a long window (e.g. 49 frames = ~4 h)
can cross one or more UTC midnights — and these archives are laid out **per UTC day**, so
the loader must list every day the window spans. This module centralizes that time logic so
it is shared and unit-testable WITHOUT pyart/boto3/network.

Key design choices (so a real storm case actually yields an event on "Run all"):
  * tolerance defaults to the full step ``dt`` (not dt/2): for a ~6-min NEXRAD cadence the
    nearest volume to any 5-min grid point is within ~3 min, so dt tolerance is gap-free for
    normal cadence; only a genuine outage (> dt) trips the gap.
  * a genuine gap returns None so the caller SKIPS that case (no forward-fill / duplicated
    frames) — matching the SEVIR "pct_missing==0" convention, which keeps OOD metrics honest.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np


def start_minute(start_hhmmss: str) -> int:
    """Minutes-since-midnight from an ``HHMMSS`` string.

    Right-pads to 6 digits so a user who drops trailing seconds gets the intuitive result
    (``"1200"`` -> 12:00, ``"03"`` -> 03:00), not a left-padded ``00:12``.
    """
    s = str(start_hhmmss).ljust(6, "0")[:6]
    return int(s[:2]) * 60 + int(s[2:4])


def spanned_dates(date_yyyymmdd: str, start_hhmmss: str, n_frames: int, dt_min: int) -> List[Tuple[str, int]]:
    """UTC days a window covers, as ``[(YYYYMMDD, day_offset), ...]``.

    Window = ``[start, start + (n_frames-1)*dt_min]``. ``day_offset`` is whole days after
    the start date (0 = start day), used to make per-day minute-of-day comparable across
    midnight via :func:`abs_minute`.
    """
    d0 = datetime.strptime(str(date_yyyymmdd), "%Y%m%d")
    end_min = start_minute(start_hhmmss) + (int(n_frames) - 1) * int(dt_min)
    last_off = max(0, end_min // 1440)
    return [((d0 + timedelta(days=off)).strftime("%Y%m%d"), off) for off in range(last_off + 1)]


def abs_minute(day_offset: int, minute_of_day: int) -> int:
    """Absolute minute on a continuous axis spanning multiple days (handles midnight)."""
    return int(day_offset) * 1440 + int(minute_of_day)


def select_nearest(
    avail_abs: List[int],
    start_min: int,
    n_frames: int,
    dt_min: int,
    tol_min: Optional[float] = None,
) -> Optional[List[int]]:
    """Map each uniform 5-min slot to the index of the nearest available absolute minute.

    Returns a length-``n_frames`` list of indices into ``avail_abs``, or ``None`` if any slot
    has no record within ``tol_min`` (a genuine gap -> caller skips the case). ``tol_min``
    defaults to ``dt_min``.
    """
    if not avail_abs:
        return None
    tol = float(dt_min) if tol_min is None else float(tol_min)
    avail = np.asarray(avail_abs, dtype=np.float64)
    need = [int(start_min) + i * int(dt_min) for i in range(int(n_frames))]
    idx: List[int] = []
    for tt in need:
        j = int(np.argmin(np.abs(avail - tt)))
        if abs(float(avail[j]) - tt) > tol:
            return None
        idx.append(j)
    return idx


def assemble_uniform(
    records: List[Tuple[int, np.ndarray]],
    start_min: int,
    n_frames: int,
    dt_min: int,
    tol_min: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Stack frames onto the uniform axis -> ``[n_frames, H, W]`` (or None on a gap).

    ``records`` = ``[(abs_minute, frame2d), ...]`` (any order). Convenience wrapper used by
    tests; the real loaders use :func:`select_nearest` to avoid decoding unselected volumes.
    """
    if not records:
        return None
    recs = sorted(records, key=lambda r: r[0])
    idx = select_nearest([r[0] for r in recs], start_min, n_frames, dt_min, tol_min)
    if idx is None:
        return None
    return np.stack([recs[j][1] for j in idx], axis=0).astype(np.float32)
