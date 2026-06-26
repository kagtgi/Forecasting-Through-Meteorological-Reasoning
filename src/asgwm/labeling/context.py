"""Environmental context co-location for ASG auto-labeling (datasource.md section 2, step 4).

Slice HRRR (CONUS convective context: CAPE/CIN/shear), ARCO-ERA5 (global:
CAPE/PWAT/winds), and Copernicus DEM at each event's lat/lon/time and attach the
resulting scalars to the ASG. Every loader is **best-effort**: the heavy stacks
(s3fs, xarray, herbie, rasterio) are optional and the data may simply be absent in
a synthetic / offline run. In that case we return all-zero context with
`context_available = 0.0` and never crash (per task contract).

Returned context dict keys (match cfg.asg.context_fields + an availability flag):
    cape, cin, shear, pwat, dem, context_available

The five physical scalars feed the Stage-B context vector (context_dim = 5 in
configs/default.yaml); `context_available` is metadata, not part of that vector.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from asgwm.utils.config import Config

# Optional heavy backends — guarded so import never fails offline.
try:  # pragma: no cover
    import xarray as _xr  # type: ignore

    _HAS_XARRAY = True
except Exception:  # pragma: no cover
    _xr = None
    _HAS_XARRAY = False

try:  # pragma: no cover
    import rasterio as _rasterio  # type: ignore

    _HAS_RASTERIO = True
except Exception:  # pragma: no cover
    _rasterio = None
    _HAS_RASTERIO = False

# Canonical zero context (keys mirror cfg.asg.context_fields).
_CONTEXT_KEYS = ("cape", "cin", "shear", "pwat", "dem")


def zero_context() -> Dict[str, float]:
    """All-zero context with availability flag cleared."""
    d = {k: 0.0 for k in _CONTEXT_KEYS}
    d["context_available"] = 0.0
    return d


# ---------------------------------------------------------------------------
# Best-effort source loaders. Each returns a partial dict or None on absence.
# ---------------------------------------------------------------------------
def load_dem(cfg: Config) -> Optional[np.ndarray]:
    """Load a cached DEM array if present (best-effort).

    Looks for a `.npy` DEM cache under paths.cache/dem.npy (written once by the
    download script). Returns the array, or None if rasterio/the file is absent.
    The DEM is a static field fetched once (datasource.md section 1).
    """
    cache = cfg.get_path("paths.cache", "./artifacts/cache")
    npy_path = os.path.join(cache, "dem.npy")
    if os.path.exists(npy_path):
        try:
            return np.load(npy_path)
        except Exception:
            return None
    # Optional: a GeoTIFF DEM tile if rasterio is available.
    tif_path = os.path.join(cache, "dem.tif")
    if _HAS_RASTERIO and os.path.exists(tif_path):
        try:  # pragma: no cover - requires rasterio + data
            with _rasterio.open(tif_path) as src:
                return src.read(1).astype(np.float32)
        except Exception:
            return None
    return None


def _dem_elevation(lat: Optional[float], lon: Optional[float], cfg: Config) -> Optional[float]:
    """Mean elevation near (lat, lon) from a cached DEM, or None."""
    dem = load_dem(cfg)
    if dem is None or dem.size == 0:
        return None
    # Without geotransform metadata we cannot index by lat/lon precisely; return the
    # domain-mean elevation as a stable, non-crashing proxy (documented limitation).
    return float(np.nanmean(dem))


def slice_hrrr(
    lat: Optional[float],
    lon: Optional[float],
    time_iso: Optional[str],
    cfg: Config,
) -> Optional[Dict[str, float]]:
    """Slice HRRR CAPE/CIN/shear at (lat, lon, time) — best-effort (datasource.md section 1).

    Reads from a cached per-event NetCDF/zarr under paths.cache/hrrr/<time>.nc if
    xarray and the file are present. Returns a partial dict of {cape, cin, shear}
    or None when the source is unavailable (synthetic / offline runs).
    """
    if not (_HAS_XARRAY and lat is not None and lon is not None and time_iso):
        return None
    cache = cfg.get_path("paths.cache", "./artifacts/cache")
    stamp = str(time_iso).replace(":", "").replace("-", "").replace(" ", "_")
    path = os.path.join(cache, "hrrr", f"{stamp}.nc")
    if not os.path.exists(path):
        return None
    try:  # pragma: no cover - requires xarray + data
        ds = _xr.open_dataset(path)
        out: Dict[str, float] = {}
        for key, candidates in (
            ("cape", ("cape", "CAPE", "sbcape")),
            ("cin", ("cin", "CIN", "sbcin")),
            ("shear", ("shear", "bulk_shear", "vertical_shear")),
        ):
            for c in candidates:
                if c in ds:
                    val = ds[c].sel(latitude=lat, longitude=lon, method="nearest")
                    out[key] = float(np.asarray(val).reshape(-1)[0])
                    break
        ds.close()
        return out or None
    except Exception:
        return None


def slice_era5(
    lat: Optional[float],
    lon: Optional[float],
    time_iso: Optional[str],
    cfg: Config,
) -> Optional[Dict[str, float]]:
    """Slice ARCO-ERA5 CAPE/PWAT (+ shear proxy) at (lat, lon, time) — best-effort.

    Reads a cached per-event slice under paths.cache/era5/<time>.nc via xarray.
    Global fallback for context when HRRR (CONUS only) is unavailable
    (datasource.md section 1). Returns a partial dict or None.
    """
    if not (_HAS_XARRAY and lat is not None and lon is not None and time_iso):
        return None
    cache = cfg.get_path("paths.cache", "./artifacts/cache")
    stamp = str(time_iso).replace(":", "").replace("-", "").replace(" ", "_")
    path = os.path.join(cache, "era5", f"{stamp}.nc")
    if not os.path.exists(path):
        return None
    try:  # pragma: no cover - requires xarray + data
        ds = _xr.open_dataset(path)
        out: Dict[str, float] = {}
        for key, candidates in (
            ("cape", ("cape", "CAPE")),
            ("pwat", ("tcwv", "pwat", "total_column_water_vapour")),
            ("cin", ("cin", "CIN")),
            ("shear", ("shear", "bulk_shear")),
        ):
            for c in candidates:
                if c in ds:
                    val = ds[c].sel(latitude=lat, longitude=lon, method="nearest")
                    out[key] = float(np.asarray(val).reshape(-1)[0])
                    break
        ds.close()
        return out or None
    except Exception:
        return None


def colocate_context(
    lat: Optional[float],
    lon: Optional[float],
    time_iso: Optional[str],
    cfg: Config,
) -> Dict[str, float]:
    """Attach co-located environmental scalars to an ASG (datasource.md section 2 step 4).

    Tries HRRR (CONUS) first for CAPE/CIN/shear, then ERA5 (global) to fill any
    gaps incl. PWAT, then a cached DEM for elevation. If no source yields any
    value the function returns all-zero context with `context_available = 0.0`
    and never raises (required for synthetic / offline runs).

    Args:
        lat, lon: event center coordinates (deg); may be None.
        time_iso: ISO timestamp string; may be None.
        cfg: project Config (for paths.cache).

    Returns:
        dict with keys cape, cin, shear, pwat, dem, context_available (0/1).
    """
    ctx = zero_context()
    found_any = False

    hrrr = slice_hrrr(lat, lon, time_iso, cfg)
    if hrrr:
        for k, v in hrrr.items():
            if k in ctx and np.isfinite(v):
                ctx[k] = float(v)
                found_any = True

    era5 = slice_era5(lat, lon, time_iso, cfg)
    if era5:
        for k, v in era5.items():
            # Fill only fields HRRR did not provide (HRRR preferred over CONUS).
            if k in ctx and ctx[k] == 0.0 and np.isfinite(v):
                ctx[k] = float(v)
                found_any = True

    elev = _dem_elevation(lat, lon, cfg)
    if elev is not None and np.isfinite(elev):
        ctx["dem"] = float(elev)
        found_any = True

    ctx["context_available"] = 1.0 if found_any else 0.0
    return ctx


def context_to_vector(ctx: Dict[str, float]) -> np.ndarray:
    """Pack a context dict into the fixed [5] Stage-B context vector.

    Order: (cape, cin, shear, pwat, dem) — matches cfg.asg.context_fields and the
    context_dim=5 in configs/default.yaml.
    """
    return np.array([ctx.get(k, 0.0) for k in _CONTEXT_KEYS], dtype=np.float32)
