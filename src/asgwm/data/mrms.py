"""MRMS loader — out-of-distribution generalization set (datasets/README.md).

NOAA MRMS (Multi-Radar Multi-Sensor) ``MergedReflectivityQCComposite`` on AWS Open Data
is a CONUS-wide, **already-gridded** ~1 km lat/lon composite reflectivity (dBZ) mosaic at
~2 min cadence — far easier than NEXRAD (no polar gridding). To become a drop-in OOD test
it is: downloaded (gzipped GRIB2), sentinel-masked (-99 missing, -999 no-coverage),
cropped/regridded to a 384x384 1 km tile around a chosen center, converted to **SEVIR VIL
byte** space (:mod:`asgwm.data.normalize`), subsampled to a uniform 5-min axis, and cached
with the same ``{vil:[T,H,W], lat, lon, time, event_id}`` schema as SEVIR.

Bucket (verified): ``s3://noaa-mrms-pds`` (us-east-1, anonymous). Product prefix:
``CONUS/MergedReflectivityQCComposite_00.50/<YYYYMMDD>/`` with objects
``MRMS_MergedReflectivityQCComposite_00.50_<YYYYMMDD>-<HHMMSS>.grib2.gz``. The AWS archive
begins 2020-10-14 — pick OOD cases on/after that date.

Heavy deps (boto3, xarray+cfgrib; pyresample optional) are guarded; this module imports
anywhere and the downloader raises a clear error when they / the network are absent (no
synthetic fallback for an OOD set). The native lat/lon grid is descending in latitude
(row 0 = north), matching SEVIR's north-up convention.

Public surface (mirrors sevir.py):
    download_mrms_subset(cfg) -> List[str]
"""
from __future__ import annotations

import gzip
import os
import tempfile
from typing import Dict, List, Optional

import numpy as np

from ..utils.config import Config
from . import sevir as _sevir
from . import normalize as _norm

# ---- optional heavy deps ---------------------------------------------------
try:  # pragma: no cover
    import boto3  # type: ignore
    from botocore import UNSIGNED  # type: ignore
    from botocore.client import Config as _BotoConfig  # type: ignore

    _HAS_BOTO = True
except Exception:  # pragma: no cover
    boto3 = None  # type: ignore
    _HAS_BOTO = False

try:  # pragma: no cover
    import xarray as _xr  # type: ignore

    _HAS_XR = True
except Exception:  # pragma: no cover
    _xr = None  # type: ignore
    _HAS_XR = False

BUCKET = "noaa-mrms-pds"
PRODUCT = "MergedReflectivityQCComposite_00.50"
PREFIX = f"CONUS/{PRODUCT}"

# MRMS CONUS grid geometry (cell-centered, 0.01 deg, latitude DESCENDING).
_LAT0 = 54.995   # northern-most row center
_LON0 = -129.995  # western-most col center (i.e. 230.005E)
_DLL = 0.01
_SENTINELS = (-99.0, -999.0)

# Built-in OOD cases (on/after 2020-10-14). center = (lat0, lon0) of the 384 km tile.
DEFAULT_CASES: List[Dict[str, object]] = [
    {"date": "20210510", "start": "210000", "lat": 35.5, "lon": -97.5},   # OK convection
    {"date": "20211211", "start": "030000", "lat": 36.5, "lon": -88.5},   # Dec 2021 outbreak
]


def _cases(cfg: Config) -> List[Dict[str, object]]:
    cases = cfg.get_path("data.mrms.cases", None)
    return list(cases) if cases else DEFAULT_CASES


def _s3():  # pragma: no cover - network
    if not _HAS_BOTO:
        raise RuntimeError("boto3 required for MRMS: pip install boto3")
    return boto3.client("s3", config=_BotoConfig(signature_version=UNSIGNED))


def _list_day(s3, date: str) -> List[str]:  # pragma: no cover - network
    """Sorted composite object keys for one day (date = YYYYMMDD)."""
    prefix = f"{PREFIX}/{date}/"
    keys: List[str] = []
    token = None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        resp = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".grib2.gz")]
        if not resp.get("IsTruncated"):
            break
        token = resp.get("NextContinuationToken")
    return sorted(keys)


def _key_time_min(key: str) -> Optional[int]:
    """Minutes-since-midnight from ``..._<YYYYMMDD>-<HHMMSS>.grib2.gz``."""
    base = os.path.basename(key)
    stamp = base.replace(".grib2.gz", "").split("_")[-1]  # YYYYMMDD-HHMMSS
    if "-" in stamp:
        hhmmss = stamp.split("-")[-1]
        if len(hhmmss) == 6 and hhmmss.isdigit():
            return int(hhmmss[:2]) * 60 + int(hhmmss[2:4])
    return None


def _read_grib(s3, key: str) -> np.ndarray:  # pragma: no cover - network + cfgrib
    """Download + gunzip + read one composite -> full CONUS dBZ array (lat desc, lon asc)."""
    raw = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(gzip.decompress(raw))
        path = f.name
    try:
        ds = _xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
        da = ds[list(ds.data_vars)[0]]
        arr = np.asarray(da.values, dtype=np.float32)
    finally:
        for p in (path, path + ".idx"):
            try:
                os.remove(p)
            except OSError:
                pass
    return arr


def _crop_tile(full: np.ndarray, lat0: float, lon0: float, grid: int) -> np.ndarray:
    """Index-crop a ``grid x grid`` tile centered on (lat0, lon0) from the CONUS mosaic.

    Latitude is descending, so row index increases southward. This is the fast path; for
    strict equal-area fidelity a pyresample/xesmf regrid is preferable (noted in README),
    but a center crop at ~1 km is accurate to first order.
    """
    i0 = int(round((_LAT0 - lat0) / _DLL))   # row (north->south)
    j0 = int(round((lon0 - _LON0) / _DLL))   # col (west->east)
    h, w = full.shape
    r0 = int(np.clip(i0 - grid // 2, 0, max(h - grid, 0)))
    c0 = int(np.clip(j0 - grid // 2, 0, max(w - grid, 0)))
    tile = full[r0:r0 + grid, c0:c0 + grid]
    if tile.shape != (grid, grid):  # pad if the center is near an edge
        out = np.full((grid, grid), np.nan, dtype=np.float32)
        out[: tile.shape[0], : tile.shape[1]] = tile
        tile = out
    return tile  # row 0 = north (matches SEVIR)


def _build_event(cfg, s3, case: Dict[str, object]) -> Optional[Dict[str, np.ndarray]]:  # pragma: no cover
    grid = int(cfg.get_path("data.grid", _norm.CANON_GRID))
    dt = int(cfg.get_path("data.minutes_per_frame", 5))
    T = int(cfg.get_path("data.in_frames", 13)) + int(cfg.get_path("data.out_frames", 36))
    dz_eff = float(cfg.get_path("data.mrms.dz_eff_m", _norm.DEFAULT_DZ_EFF_M))
    tol = dt / 2.0

    date = str(case["date"])
    start = str(case.get("start", "000000"))
    lat0 = float(case.get("lat", 35.5))
    lon0 = float(case.get("lon", -97.5))
    start_min = int(start[:2]) * 60 + int(start[2:4])
    need = [start_min + i * dt for i in range(T)]

    keys = _list_day(s3, date)
    keyed = [(k, _key_time_min(k)) for k in keys]
    keyed = [(k, t) for k, t in keyed if t is not None]
    if not keyed:
        print(f"[mrms] no files for {date} (archive starts 2020-10-14)")
        return None
    avail = np.array([t for _, t in keyed])

    frames = []
    for tt in need:
        j = int(np.argmin(np.abs(avail - tt)))
        if abs(int(avail[j]) - tt) > tol:
            print(f"[mrms] {date}: gap at +{tt - start_min}min -> skip case")
            return None
        full = _read_grib(s3, keyed[j][0])
        full = np.where(np.isin(full, _SENTINELS), np.nan, full)
        frames.append(_crop_tile(full, lat0, lon0, grid))

    comp = np.stack(frames, axis=0).astype(np.float32)            # [T,H,W] dBZ
    vil_byte = _norm.dbz_to_vil_byte(comp, dz_eff_m=dz_eff)       # [T,H,W] byte
    vil_byte = _norm.resize_to_canonical(vil_byte, grid)
    _norm.assert_canonical(vil_byte, grid)

    z = np.zeros_like(vil_byte)
    hh, mm = divmod(start_min, 60)
    return {
        "vil": vil_byte, "ir069": z, "ir107": z.copy(), "glm": z.copy(),
        "lat": np.float32(lat0), "lon": np.float32(lon0),
        "time": f"{date[:4]}-{date[4:6]}-{date[6:8]}T{hh:02d}:{mm:02d}:00Z",
        "event_id": f"mrms_{date}_{start[:4]}_{int(round(lat0))}_{int(round(abs(lon0)))}",
    }


def download_mrms_subset(cfg: Config) -> List[str]:
    """Download + cache an MRMS OOD subset in SEVIR-compatible VIL-byte form.

    Honors ``data.require_real``: missing deps / network / empty result raise. Events are
    cached under the dataset-namespaced ``events_mrms`` dir so the rest of the pipeline
    consumes them unchanged.
    """
    require_real = bool(cfg.get_path("data.require_real", False))
    if not (_HAS_BOTO and _HAS_XR):
        msg = ("[mrms] MRMS requires boto3 + xarray + cfgrib "
               "(`pip install boto3 xarray cfgrib eccodes`); no synthetic fallback")
        if require_real:
            raise RuntimeError(msg + " (data.require_real=True).")
        print(msg + " -> returning no events")
        return []

    try:  # pragma: no cover - network + cfgrib
        s3 = _s3()
        ids: List[str] = []
        for case in _cases(cfg):
            ev = _build_event(cfg, s3, case)
            if ev is None:
                continue
            eid = str(ev["event_id"])
            _sevir._save_event_npz(_sevir._event_npz_path(cfg, eid), ev)
            ids.append(eid)
            print(f"[mrms] cached {eid}  vil shape={ev['vil'].shape}")
        if not ids:
            raise RuntimeError("MRMS produced 0 events (no files / all gaps?)")
        _sevir._write_manifest(cfg, ids)
        print(f"[mrms] cached {len(ids)} OOD events -> {_sevir.events_dir(cfg)}")
        return ids
    except Exception as e:  # pragma: no cover
        if require_real:
            raise RuntimeError(f"[mrms] OOD load FAILED: {e}") from e
        print(f"[mrms] failed ({e}) -> returning no events")
        return []
