"""Canonical cross-dataset normalization (datasource.md; datasets/README.md).

This module is the **single source of truth** that makes SEVIR, NEXRAD, and MRMS
interchangeable inputs to the same model. Every dataset is reduced to ONE canonical
representation:

    * spatial shape   : 384 x 384 at 1 km nominal spacing, time-first ``[T, 384, 384]``
    * physical channel: SEVIR VIL **byte** encoding, integer ``[0, 254]`` (255 = missing)
    * model value range: float32 in ``[0, 1]`` via ``x_byte / 255`` (Earthformer/PreDiff)

SEVIR is kept in its native VIL byte space so every published SEVIR-VIL baseline
(Earthformer, PreDiff, DiffCast, CasCast, ...) compares directly with **no variable
transform**. NEXRAD and MRMS are radar **reflectivity in dBZ**; for the out-of-distribution
generalization test they are converted *into* SEVIR's VIL byte space so a SEVIR-trained
model runs on them unmodified. The two-step bridge is:

    dBZ --(Z-M relation, effective depth)--> VIL [kg/m^2] --(invert SEVIR decode)--> byte

CAVEAT (state in the paper): composite reflectivity (a column-max) and VIL (a vertical
integral) are physically different quantities, so the dBZ->VIL map is an approximation
governed by the effective depth ``dz_eff``. OOD numbers therefore measure generalization
under an imperfect variable bridge, not a perfect like-for-like. Report ``dz_eff`` and,
as a sensitivity check, also evaluate against the MRMS-native VIL product. (Sources:
SEVIR NeurIPS-2020 generator; Greene & Clark / Marshall VIL approximation.)

Pure numpy — no heavy deps — so it imports anywhere and runs on CPU.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Canonical constants
# ---------------------------------------------------------------------------
CANON_GRID: int = 384            # canonical spatial size (H = W)
CANON_KM_PER_PIXEL: float = 1.0  # 1 km nominal spacing
VIL_BYTE_MAX: float = 255.0      # normalization divisor (Earthformer/PreDiff convention)
VIL_BYTE_MISSING: int = 255      # SEVIR missing-data sentinel
VIL_BYTE_VALID_MAX: int = 254    # highest valid encoded VIL byte

# Standard SEVIR-VIL CSI thresholds, on the raw 0-255 byte scale (Earthformer et al.).
# Use these (not kg/m^2 or a remapped dBZ scale) so CSI matches the published baselines.
SEVIR_CSI_THRESHOLDS = (16, 74, 133, 160, 181, 219)

# SEVIR official piecewise byte<->kg/m^2 decode (NeurIPS-2020 SEVIR generator).
#   R(x) = 0                         for x <= 5
#   R(x) = (x - 2) / 90.66           for 5  < x <= 18   (linear regime)
#   R(x) = exp((x - 83.9) / 38.9)    for 18 < x <= 254  (exponential regime)
_DECODE_LIN_SLOPE = 90.66
_DECODE_LIN_OFFSET = 2.0
_DECODE_EXP_SCALE = 38.9
_DECODE_EXP_OFFSET = 83.9
# Boundary R values at the linear/exp split (x = 18) used to pick the inverse branch.
_R_AT_X18_LINEAR = (18.0 - _DECODE_LIN_OFFSET) / _DECODE_LIN_SLOPE  # ~0.1765

# dBZ -> VIL (kg/m^2) bridge constants (Z in mm^6 m^-3, M in g m^-3).
#   Z = 10^(dBZ/10);  M = A * Z^B  [g/m^3];  VIL = M * dz_eff / 1000  [kg/m^2]
_ZM_A = 3.44e-3
_ZM_B = 4.0 / 7.0
DEFAULT_DZ_EFF_M: float = 4000.0   # effective composite depth (m); calibrate to SEVIR hist.
DEFAULT_DBZ_CAP: float = 56.0      # cap ice/hail contribution before conversion (dBZ)


# ---------------------------------------------------------------------------
# Sentinel masking
# ---------------------------------------------------------------------------
def mask_sentinels(arr: np.ndarray, sentinels=(VIL_BYTE_MISSING,)) -> np.ndarray:
    """Return a float copy of ``arr`` with the given sentinel values set to NaN.

    Centralizes sentinel handling so the three datasets' different no-data codes
    (SEVIR 255; MRMS -99 "missing" and -999 "no coverage"; NEXRAD masked/below-threshold)
    are all removed before any statistics or normalization.
    """
    out = np.asarray(arr, dtype=np.float32).copy()
    for s in sentinels:
        out[out == s] = np.nan
    return out


# ---------------------------------------------------------------------------
# SEVIR byte <-> physical VIL (kg/m^2)
# ---------------------------------------------------------------------------
def vil_byte_to_kgm2(byte: np.ndarray) -> np.ndarray:
    """Decode SEVIR VIL byte [0, 254] to physical VIL in kg/m^2 (255 -> NaN).

    Applies the official 3-piece SEVIR map. Vectorized and monotonic.
    """
    x = mask_sentinels(byte, (VIL_BYTE_MISSING,))
    lin = (x - _DECODE_LIN_OFFSET) / _DECODE_LIN_SLOPE
    exp = np.exp((x - _DECODE_EXP_OFFSET) / _DECODE_EXP_SCALE)
    out = np.where(x <= 5.0, 0.0, np.where(x <= 18.0, lin, exp))
    out = np.where(np.isnan(x), np.nan, out)
    return out.astype(np.float32)


def kgm2_to_vil_byte(r: np.ndarray) -> np.ndarray:
    """Inverse of :func:`vil_byte_to_kgm2`: VIL kg/m^2 -> SEVIR byte [0, 254].

    Picks the linear branch up to the x=18 boundary, the exponential branch above,
    clips to the valid byte range, and rounds to integers.
    """
    r = np.asarray(r, dtype=np.float32)
    safe_r = np.maximum(r, 1e-6)  # guard log of non-positive
    lin = _DECODE_LIN_SLOPE * r + _DECODE_LIN_OFFSET
    exp = _DECODE_EXP_SCALE * np.log(safe_r) + _DECODE_EXP_OFFSET
    byte = np.where(r <= 0.0, 0.0, np.where(r <= _R_AT_X18_LINEAR, lin, exp))
    byte = np.clip(byte, 0.0, float(VIL_BYTE_VALID_MAX))
    byte = np.where(np.isnan(r), 0.0, byte)
    return np.rint(byte).astype(np.float32)


# ---------------------------------------------------------------------------
# Reflectivity (dBZ) -> SEVIR VIL byte
# ---------------------------------------------------------------------------
def dbz_to_kgm2(
    dbz: np.ndarray,
    dz_eff_m: float = DEFAULT_DZ_EFF_M,
    dbz_cap: float = DEFAULT_DBZ_CAP,
) -> np.ndarray:
    """Approximate VIL (kg/m^2) from composite reflectivity (dBZ).

    Z = 10^(dBZ/10); M = A*Z^B [g/m^3]; VIL = M * dz_eff / 1000 [kg/m^2], with the dBZ
    capped (``dbz_cap``) to limit hail/ice inflation. NaN-safe (NaN -> NaN).
    """
    d = np.asarray(dbz, dtype=np.float32)
    d = np.where(np.isnan(d), np.nan, np.minimum(d, float(dbz_cap)))
    z = np.power(10.0, d / 10.0)                       # mm^6 / m^3
    m = _ZM_A * np.power(z, _ZM_B)                     # g / m^3
    vil = m * float(dz_eff_m) / 1000.0                 # kg / m^2
    vil = np.where(np.isnan(d) | (d <= 0.0), 0.0, vil)
    return vil.astype(np.float32)


def dbz_to_vil_byte(
    dbz: np.ndarray,
    dz_eff_m: float = DEFAULT_DZ_EFF_M,
    dbz_cap: float = DEFAULT_DBZ_CAP,
) -> np.ndarray:
    """Full NEXRAD/MRMS dBZ -> SEVIR VIL byte [0, 254] bridge (two-step).

    Composes :func:`dbz_to_kgm2` then :func:`kgm2_to_vil_byte`. This is the ONLY place
    OOD reflectivity becomes SEVIR pixel space; keep all callers routed through here so
    the conversion (and its ``dz_eff_m``) is consistent and reportable.
    """
    return kgm2_to_vil_byte(dbz_to_kgm2(dbz, dz_eff_m=dz_eff_m, dbz_cap=dbz_cap))


# ---------------------------------------------------------------------------
# Byte <-> model [0, 1]
# ---------------------------------------------------------------------------
def normalize_vil(byte: np.ndarray, fill_missing: float = 0.0) -> np.ndarray:
    """SEVIR VIL byte -> float32 model input in [0, 1] via ``x / 255``.

    Missing (255) and NaN are mapped to ``fill_missing`` (default 0) BEFORE division —
    after any statistics/QC the caller already did. This is the exact Earthformer/PreDiff
    normalization, so the headline table matches the published convention.
    """
    x = np.asarray(byte, dtype=np.float32).copy()
    x[x == VIL_BYTE_MISSING] = np.nan
    x = np.where(np.isnan(x), float(fill_missing) * VIL_BYTE_MAX, x)
    return (x / VIL_BYTE_MAX).astype(np.float32)


def denormalize_vil(norm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`normalize_vil`: [0, 1] float -> VIL byte for CSI thresholding."""
    x = np.clip(np.asarray(norm, dtype=np.float32), 0.0, 1.0) * VIL_BYTE_MAX
    return np.rint(x).astype(np.float32)


# ---------------------------------------------------------------------------
# Spatial resampling to the canonical grid (nearest-neighbour; no scipy)
# ---------------------------------------------------------------------------
def resize_to_canonical(arr: np.ndarray, size: int = CANON_GRID) -> np.ndarray:
    """Nearest-neighbour resize a ``[..., h, w]`` stack to ``[..., size, size]``.

    SEVIR is already 384x384 (returned unchanged). NEXRAD/MRMS loaders do their
    geometry-aware gridding first; this is the final shape guarantee.
    """
    a = np.asarray(arr)
    h, w = a.shape[-2:]
    if (h, w) == (size, size):
        return a
    yi = np.linspace(0, h - 1, size).round().astype(int)
    xi = np.linspace(0, w - 1, size).round().astype(int)
    return a[..., yi, :][..., :, xi]


def assert_canonical(arr: np.ndarray, size: int = CANON_GRID) -> None:
    """Raise if ``arr`` is not a ``[T, size, size]`` VIL-byte stack in [0, 255].

    Cheap invariant check the loaders call before caching — catches shape/flip/range
    bugs (e.g. a silent vertical flip or a dBZ array that skipped conversion).
    """
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[-2:] != (size, size):
        raise ValueError(f"expected [T,{size},{size}], got {a.shape}")
    finite = a[np.isfinite(a)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 255.0):
        raise ValueError(f"VIL byte out of [0,255]: min={finite.min()}, max={finite.max()}")
