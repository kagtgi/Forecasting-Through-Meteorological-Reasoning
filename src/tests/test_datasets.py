"""Dataset dispatch + OOD graceful-degradation tests (asgwm.data.{sevir,nexrad,mrms}).

These run in the minimal offline env (no pyart/boto3/h5py, no network). They check:
  * the cache is dataset-namespaced (events / events_nexrad / events_mrms),
  * the OOD loaders degrade gracefully (return [] unless data.require_real, then RAISE)
    when their heavy deps are absent — guarded so the test is correct even where the
    deps happen to be installed,
  * the verified (non-deprecated) S3 bucket names are pinned.
"""
from __future__ import annotations

import os

import pytest

from asgwm.utils.config import load_config
from asgwm.data import sevir, nexrad, mrms

_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_CFG = os.path.join(_CODE_ROOT, "configs", "default.yaml")


def _cfg(dataset: str, tmp_path, require_real: bool = False):
    cache = os.path.join(str(tmp_path), "cache")
    return load_config(
        _DEFAULT_CFG,
        [
            f"data.dataset={dataset}",
            f"paths.cache={cache}",
            f"data.require_real={'true' if require_real else 'false'}",
        ],
    )


# ---------------------------------------------------------------------------
# Dataset-namespaced cache dirs
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dataset", ["sevir", "synthetic"])
def test_events_dir_unsuffixed_for_sevir_and_synthetic(dataset, tmp_path):
    d = sevir.events_dir(_cfg(dataset, tmp_path))
    assert os.path.basename(d.rstrip("/\\")) == "events"


def test_events_dir_nexrad_namespaced(tmp_path):
    d = sevir.events_dir(_cfg("nexrad", tmp_path))
    assert os.path.basename(d.rstrip("/\\")) == "events_nexrad"


def test_events_dir_mrms_namespaced(tmp_path):
    d = sevir.events_dir(_cfg("mrms", tmp_path))
    assert os.path.basename(d.rstrip("/\\")) == "events_mrms"


# ---------------------------------------------------------------------------
# OOD loaders degrade gracefully without deps (or raise under require_real)
# ---------------------------------------------------------------------------
def test_nexrad_download_no_deps_returns_empty_then_raises(tmp_path):
    if nexrad._HAS_PYART or nexrad._HAS_BOTO:
        pytest.skip("pyart/boto3 present in this env; offline-degradation path not exercised")
    assert nexrad.download_nexrad_subset(_cfg("nexrad", tmp_path)) == []
    with pytest.raises(RuntimeError):
        nexrad.download_nexrad_subset(_cfg("nexrad", tmp_path, require_real=True))


def test_mrms_download_no_deps_returns_empty_then_raises(tmp_path):
    if mrms._HAS_BOTO or mrms._HAS_XR:
        pytest.skip("boto3/xarray present in this env; offline-degradation path not exercised")
    assert mrms.download_mrms_subset(_cfg("mrms", tmp_path)) == []
    with pytest.raises(RuntimeError):
        mrms.download_mrms_subset(_cfg("mrms", tmp_path, require_real=True))


# ---------------------------------------------------------------------------
# Pinned (non-deprecated) bucket names
# ---------------------------------------------------------------------------
def test_mrms_bucket_is_verified():
    assert mrms.BUCKET == "noaa-mrms-pds"


def test_nexrad_bucket_is_not_the_deprecated_one():
    assert nexrad.BUCKET == "unidata-nexrad-level2"
