"""NEXRAD Level II loader — out-of-distribution generalization set (datasets/README.md).

NEXRAD WSR-88D Level II archive on AWS Open Data is single-radar reflectivity in
**polar** (azimuth/range/elevation) coordinates. To become a drop-in OOD test for a
SEVIR-trained model it is: read with Py-ART, QC'd, **gridded** to a 384x384 1 km
Cartesian tile centered on the radar, collapsed to a column-max **composite
reflectivity (dBZ)**, converted to **SEVIR VIL byte** space
(:mod:`asgwm.data.normalize`), temporally resampled to a uniform 5-min axis, and cached
with the exact same ``{vil:[T,H,W], lat, lon, time, event_id}`` schema SEVIR uses — so
labeling / datasets / eval consume it unchanged.

Bucket (verified): ``s3://unidata-nexrad-level2`` (us-east-1, anonymous). The legacy
``noaa-nexrad-level2`` was deprecated 2025-09-01 — do not use it. Key layout:
``{YYYY}/{MM}/{DD}/{STATION}/{STATION}{YYYYMMDD}_{HHMMSS}_V06``.

Heavy deps (pyart, boto3) are optional and guarded; this module imports anywhere.
Calling the downloader without them (or without network) raises a clear, actionable
error — there is no synthetic fallback because an OOD test must use real data. The
network path is integration-tested on Colab / a VM with the deps installed.

Public surface (mirrors sevir.py):
    download_nexrad_subset(cfg) -> List[str]
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np

from ..utils.config import Config
from . import sevir as _sevir
from . import normalize as _norm
from . import ood_resample as _ood

# ---- optional heavy deps ---------------------------------------------------
try:  # pragma: no cover - only when pyart is installed
    import pyart  # type: ignore

    _HAS_PYART = True
except Exception:  # pragma: no cover
    pyart = None  # type: ignore
    _HAS_PYART = False

try:  # pragma: no cover
    import boto3  # type: ignore
    from botocore import UNSIGNED  # type: ignore
    from botocore.client import Config as _BotoConfig  # type: ignore

    _HAS_BOTO = True
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore
    _HAS_BOTO = False

BUCKET = "unidata-nexrad-level2"

# A small built-in OOD case list (well-known severe-weather days) used when the config
# provides none. Each case is one radar over a multi-hour convective window.
DEFAULT_CASES: List[Dict[str, object]] = [
    {"station": "KHGX", "date": "20220322", "start": "120000"},  # SE Texas convection
    {"station": "KTLX", "date": "20220504", "start": "210000"},  # OK supercells
]


def _cases(cfg: Config) -> List[Dict[str, object]]:
    cases = cfg.get_path("data.nexrad.cases", None)
    return list(cases) if cases else DEFAULT_CASES


def _s3():  # pragma: no cover - network
    if not _HAS_BOTO:
        raise RuntimeError("boto3 required for NEXRAD: pip install boto3")
    return boto3.client("s3", config=_BotoConfig(signature_version=UNSIGNED))


def _list_volumes(s3, station: str, date: str) -> List[str]:  # pragma: no cover - network
    """Sorted _V06 object keys for one station/day (date = YYYYMMDD)."""
    prefix = f"{date[:4]}/{date[4:6]}/{date[6:8]}/{station}/"
    keys: List[str] = []
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        for o in resp.get("Contents", []):
            k = o["Key"]
            if k.endswith("_V06") or k.endswith(".ar2v") or "_V0" in k:
                keys.append(k)
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(keys)


def _vol_time_min(key: str) -> Optional[int]:
    """Minutes-since-midnight from a key's ``..._HHMMSS_V06`` timestamp."""
    base = os.path.basename(key)
    for part in base.split("_"):
        if len(part) == 6 and part.isdigit():
            hh, mm = int(part[:2]), int(part[2:4])
            return hh * 60 + mm
    return None


def _composite_from_volume(local_path: str, grid: int, km: float) -> np.ndarray:  # pragma: no cover
    """Read one Level II volume -> 384x384 1 km column-max composite reflectivity (dBZ)."""
    radar = pyart.io.read_nexrad_archive(local_path)
    # QC: drop low-correlation (non-meteorological) gates when dual-pol is present.
    gf = pyart.filters.GateFilter(radar)
    if "cross_correlation_ratio" in radar.fields:
        gf.exclude_below("cross_correlation_ratio", 0.8)
    gf.exclude_invalid("reflectivity")
    half = (grid * km / 2.0) * 1000.0 - 500.0  # meters; 384*1km/2 - 0.5km -> exact 1 km
    g = pyart.map.grid_from_radars(
        (5, grid, grid),
        ((0.0, 10000.0), (-half, half), (-half, half)),  # (z, y, x) in METERS
        radars=(radar,),
        fields=["reflectivity"],
        gatefilters=(gf,),
        weighting_function="Barnes2",
        roi_func="dist_beam",
        min_radius=1000.0,
        gridding_algo="map_gates_to_grid",
    )
    ref = np.ma.filled(g.fields["reflectivity"]["data"], np.nan)  # [z, y, x]
    comp = np.nanmax(ref, axis=0)                                  # [y, x] dBZ
    lat0 = float(g.origin_latitude["data"][0])
    lon0 = float(g.origin_longitude["data"][0])
    comp._lat0 = lat0  # type: ignore[attr-defined]
    comp._lon0 = lon0  # type: ignore[attr-defined]
    return comp


def _build_event(cfg, s3, case: Dict[str, object]) -> Optional[Dict[str, np.ndarray]]:  # pragma: no cover
    """Assemble one [T,384,384] VIL-byte event from a NEXRAD case, or None on failure.

    Lists volumes across every UTC day the window spans (handles midnight crossing), picks
    the nearest volume per uniform 5-min slot (tolerance = dt, robust to ~6-min cadence),
    and decodes ONLY the selected volumes. Returns None on a genuine gap so the caller skips.
    """
    import tempfile

    grid = int(cfg.get_path("data.grid", _norm.CANON_GRID))
    km = float(cfg.get_path("data.km_per_pixel", 1.0))
    dt = int(cfg.get_path("data.minutes_per_frame", 5))
    T = int(cfg.get_path("data.in_frames", 13)) + int(cfg.get_path("data.out_frames", 36))
    dz_eff = float(cfg.get_path("data.nexrad.dz_eff_m", _norm.DEFAULT_DZ_EFF_M))

    station = str(case["station"])
    date = str(case["date"])
    start = str(case.get("start", "000000")).ljust(6, "0")[:6]
    start_min = _ood.start_minute(start)

    # List candidate volume keys across all spanned UTC days; abs-minute handles midnight.
    keyrecs: List[tuple] = []  # (abs_minute, key)
    for day, off in _ood.spanned_dates(date, start, T, dt):
        for k in _list_volumes(s3, station, day):
            t = _vol_time_min(k)
            if t is not None:
                keyrecs.append((_ood.abs_minute(off, t), k))
    keyrecs.sort()
    if not keyrecs:
        print(f"[nexrad] no volumes for {station} {date} {start}")
        return None

    idx = _ood.select_nearest([a for a, _ in keyrecs], start_min, T, dt, tol_min=dt)
    if idx is None:
        print(f"[nexrad] {station} {date}: coverage gap > {dt}min -> skip case")
        return None

    # Decode only the distinct selected volumes.
    comp_cache: Dict[str, np.ndarray] = {}
    lat0 = lon0 = None
    for j in sorted(set(idx)):
        key = keyrecs[j][1]
        local = os.path.join(tempfile.gettempdir(), os.path.basename(key))
        if not os.path.exists(local):
            s3.download_file(BUCKET, key, local)
        comp = _composite_from_volume(local, grid, km)
        comp_cache[key] = comp
        lat0 = getattr(comp, "_lat0", lat0)
        lon0 = getattr(comp, "_lon0", lon0)

    frames = [comp_cache[keyrecs[j][1]] for j in idx]
    comp_stack = np.stack(frames, axis=0).astype(np.float32)          # [T,H,W] dBZ
    vil_byte = _norm.dbz_to_vil_byte(comp_stack, dz_eff_m=dz_eff)     # [T,H,W] byte
    vil_byte = _norm.resize_to_canonical(vil_byte, grid)
    _norm.assert_canonical(vil_byte, grid)

    z = np.zeros_like(vil_byte)
    hh, mm = divmod(start_min, 60)
    return {
        "vil": vil_byte, "ir069": z, "ir107": z.copy(), "glm": z.copy(),
        "lat": np.float32(lat0 if lat0 is not None else 0.0),
        "lon": np.float32(lon0 if lon0 is not None else 0.0),
        "time": f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh:02d}:{mm:02d}:00Z",
        "event_id": f"nexrad_{station}_{date}_{start[:4]}",
    }


def download_nexrad_subset(cfg: Config) -> List[str]:
    """Download + cache a NEXRAD OOD subset in SEVIR-compatible VIL-byte form.

    Honors ``data.require_real``: missing deps / network / empty result raise (an OOD
    test must not silently fall back). Events are cached under the dataset-namespaced
    ``events_nexrad`` dir, so ``sevir.iter_events`` / labeling / eval pick them up.
    """
    require_real = bool(cfg.get_path("data.require_real", False))
    if not (_HAS_PYART and _HAS_BOTO):
        msg = ("[nexrad] NEXRAD requires pyart + boto3 "
               "(`pip install arm-pyart boto3`); no synthetic fallback for an OOD set")
        if require_real:
            raise RuntimeError(msg + " (data.require_real=True).")
        print(msg + " -> returning no events")
        return []

    try:  # pragma: no cover - network + pyart
        s3 = _s3()
        ids: List[str] = []
        for case in _cases(cfg):
            ev = _build_event(cfg, s3, case)
            if ev is None:
                continue
            eid = str(ev["event_id"])
            _sevir._save_event_npz(_sevir._event_npz_path(cfg, eid), ev)
            ids.append(eid)
            print(f"[nexrad] cached {eid}  vil shape={ev['vil'].shape}")
        if not ids:
            raise RuntimeError("NEXRAD produced 0 events (no volumes / all gaps?)")
        _sevir._write_manifest(cfg, ids)
        print(f"[nexrad] cached {len(ids)} OOD events -> {_sevir.events_dir(cfg)}")
        return ids
    except Exception as e:  # pragma: no cover
        if require_real:
            raise RuntimeError(f"[nexrad] OOD load FAILED: {e}") from e
        print(f"[nexrad] failed ({e}) -> returning no events")
        return []
