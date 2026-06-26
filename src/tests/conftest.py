"""Shared pytest fixtures for the ASG-WM test suite.

These tests exercise the REAL implemented APIs under ``asgwm`` (read from disk, not
mocked). The heavy/optional deps (torch, pysteps, transformers, ...) are guarded with
``pytest.importorskip`` inside the modules that need them, so the suite still collects and
runs in a minimal env (numpy + scipy + pyyaml + pytest).

Every fixture keeps tensors tiny and step counts at 1-2 so the whole suite is fast.
"""
from __future__ import annotations

import os
import sys

import pytest

# Make ``import asgwm`` resolve to the project code root regardless of CWD.
_CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _CODE_ROOT not in sys.path:
    sys.path.insert(0, _CODE_ROOT)


def _default_config_path() -> str:
    return os.path.join(_CODE_ROOT, "configs", "default.yaml")


@pytest.fixture
def tiny_cfg(tmp_path):
    """A loaded :class:`Config` shrunk to tiny tensors and a temp cache root.

    Built from the real ``configs/default.yaml`` via ``load_config`` with overrides so
    the smoke pipeline runs in well under a second on CPU. ``paths.cache`` is redirected
    into pytest's ``tmp_path`` so nothing touches the user's artifacts.
    """
    pytest.importorskip("yaml")
    from asgwm.utils.config import load_config

    cache = os.path.join(str(tmp_path), "cache")
    overrides = [
        f"paths.cache={cache}",
        f"paths.root={str(tmp_path)}",
        "data.grid=32",
        "data.in_frames=4",
        "data.out_frames=4",
        "data.n_train_events=3",
        "data.horizon_min=10",
        "data.patch=16",
        "asg.growth_field_size=8",
        "stage_b.d_model=16",
        "stage_b.n_layers=1",
        "stage_b.n_heads=2",
        "stage_b.d_ff=32",
        "stage_c.unet_base=8",
        "stage_c.flow_steps=1",
        "stage_c.ensemble_k=2",
    ]
    return load_config(_default_config_path(), overrides)


@pytest.fixture
def sample_asg():
    """A small hand-built :class:`ASG` with two objects (no optional deps)."""
    from asgwm.asg import ASG, StormObject

    objs = [
        StormObject(
            id=0, cy=10.0, cx=20.0, area=120.0, peak=48.0,
            vy=12.0, vx=-4.0, regime="grow", growth=0.30, conf=0.90,
        ),
        StormObject(
            id=1, cy=5.5, cx=8.25, area=18.0, peak=15.0,
            vy=0.0, vx=0.0, regime="steady", growth=0.0, conf=0.50,
        ),
    ]
    return ASG(objects=objs, global_regime="grow", meta={"km_per_pixel": 1.0})
