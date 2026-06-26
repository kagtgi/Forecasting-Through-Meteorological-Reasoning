"""Unit tests for the canonical cross-dataset normalization (asgwm.data.normalize).

Pure numpy, no heavy deps — this whole file runs in the minimal offline env. The
asserts lock down the load-bearing invariants from datasource.md: the SEVIR byte<->kg/m^2
round-trip, the missing-data sentinel handling, the monotone dBZ->VIL-byte bridge, the
model [0,1] normalization, the canonical-grid resize/assert, and the published CSI
thresholds (which every SEVIR-VIL baseline shares).
"""
from __future__ import annotations

import numpy as np
import pytest

from asgwm.data import normalize as norm


# ---------------------------------------------------------------------------
# SEVIR byte <-> kg/m^2 round-trip
# ---------------------------------------------------------------------------
def test_vil_byte_kgm2_roundtrip_representative_bytes():
    bytes_in = np.array([6, 18, 80, 133, 181, 219, 254], dtype=np.float32)
    r = norm.vil_byte_to_kgm2(bytes_in)
    back = norm.kgm2_to_vil_byte(r)
    assert np.all(np.abs(back - bytes_in) <= 1.0), f"roundtrip drift: {back} vs {bytes_in}"


# ---------------------------------------------------------------------------
# Missing-data sentinel (255)
# ---------------------------------------------------------------------------
def test_missing_byte_maps_to_nan_in_kgm2():
    arr = np.array([10, norm.VIL_BYTE_MISSING, 200], dtype=np.float32)
    r = norm.vil_byte_to_kgm2(arr)
    assert np.isnan(r[1])
    assert np.isfinite(r[0]) and np.isfinite(r[2])


def test_missing_byte_maps_to_zero_in_normalize_vil():
    arr = np.array([10, norm.VIL_BYTE_MISSING, 200], dtype=np.float32)
    out = norm.normalize_vil(arr)
    assert out[1] == pytest.approx(0.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# dBZ -> VIL byte bridge
# ---------------------------------------------------------------------------
def test_dbz_to_vil_byte_monotonic_and_bounded():
    dbz = np.linspace(0.0, 56.0, 57).astype(np.float32)
    byte = norm.dbz_to_vil_byte(dbz)
    assert byte.dtype == np.float32
    assert np.all(byte >= 0.0) and np.all(byte <= 254.0)
    diffs = np.diff(byte)
    assert np.all(diffs >= -1e-5), f"not non-decreasing: min diff {diffs.min()}"


# ---------------------------------------------------------------------------
# Byte <-> model [0,1]
# ---------------------------------------------------------------------------
def test_normalize_denormalize_roundtrip_and_range():
    bytes_in = np.array([0, 6, 18, 80, 133, 181, 219, 254], dtype=np.float32)
    n = norm.normalize_vil(bytes_in)
    assert np.all(n >= 0.0) and np.all(n <= 1.0)
    back = norm.denormalize_vil(n)
    assert np.all(np.abs(back - bytes_in) <= 1.0)


# ---------------------------------------------------------------------------
# Canonical-grid resize
# ---------------------------------------------------------------------------
def test_resize_to_canonical_upsizes_stack():
    arr = np.zeros((4, 100, 100), dtype=np.float32)
    out = norm.resize_to_canonical(arr)
    assert out.shape == (4, norm.CANON_GRID, norm.CANON_GRID)


def test_resize_to_canonical_passthrough_when_already_canonical():
    arr = np.zeros((3, norm.CANON_GRID, norm.CANON_GRID), dtype=np.float32)
    out = norm.resize_to_canonical(arr)
    assert out.shape == arr.shape


# ---------------------------------------------------------------------------
# assert_canonical
# ---------------------------------------------------------------------------
def test_assert_canonical_passes_for_valid_stack():
    arr = np.full((2, norm.CANON_GRID, norm.CANON_GRID), 128.0, dtype=np.float32)
    norm.assert_canonical(arr)  # should not raise


def test_assert_canonical_raises_on_wrong_shape():
    with pytest.raises(ValueError):
        norm.assert_canonical(np.zeros((2, 100, 100), dtype=np.float32))


def test_assert_canonical_raises_on_out_of_range():
    low = np.full((1, norm.CANON_GRID, norm.CANON_GRID), -5.0, dtype=np.float32)
    high = np.full((1, norm.CANON_GRID, norm.CANON_GRID), 300.0, dtype=np.float32)
    with pytest.raises(ValueError):
        norm.assert_canonical(low)
    with pytest.raises(ValueError):
        norm.assert_canonical(high)


# ---------------------------------------------------------------------------
# Published CSI thresholds
# ---------------------------------------------------------------------------
def test_sevir_csi_thresholds():
    assert norm.SEVIR_CSI_THRESHOLDS == (16, 74, 133, 160, 181, 219)
