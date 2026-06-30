"""Regression tests for km/h <-> px/step motion conventions."""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from asgwm import physics
from asgwm.asg import ASG, StormObject
from asgwm.data.dataset import event_time_partition, list_asg_files
from asgwm.labeling import pipeline
from asgwm.utils.config import load_config


def test_kmh_to_px_per_step_matches_labeling_advection():
    """Stage B advection must agree with labeling.pipeline._advect_objects."""
    o = StormObject(
        id=0, cy=100.0, cx=200.0, area=64.0, peak=40.0,
        vy=60.0, vx=0.0, regime="steady", growth=0.0, conf=1.0,
    )
    horizon_min = 60
    km_per_pixel = 1.0
    dt_steps = horizon_min / 5.0  # 12 frame steps at 5 min cadence

    mot_px = physics.kmh_to_px_per_step(
        torch.tensor([[o.vy, o.vx]]), km_per_pixel, minutes_per_frame=5.0
    )
    adv = physics.advect_points(
        torch.tensor([[o.cy, o.cx]]), mot_px, dt=dt_steps
    ).numpy()[0]

    expected = pipeline._advect_objects([o], horizon_min, km_per_pixel)[0]
    assert adv[0] == pytest.approx(expected.cy, abs=1e-4)
    assert adv[1] == pytest.approx(expected.cx, abs=1e-4)


def test_temporal_split_filters_cached_asgs(tmp_path):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_config(
        os.path.join(root, "configs", "default.yaml"),
        [
            f"paths.cache={tmp_path / 'cache'}",
            "data.split=train",
        ],
    )
    asg_dir = os.path.join(str(tmp_path), "cache", "asg")
    os.makedirs(asg_dir, exist_ok=True)

    def _write(eid: str, time_iso: str) -> None:
        payload = {
            "event_id": eid,
            "horizon_min": 60,
            "asg_t": {"objects": [], "global_regime": "steady", "meta": {"time": time_iso}},
            "asg_th": {"objects": [], "global_regime": "steady", "meta": {"time": time_iso}},
        }
        with open(os.path.join(asg_dir, f"{eid}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)

    _write("train_evt", "2018-06-01T12:00:00Z")
    _write("test_evt", "2019-07-01T12:00:00Z")
    _write("synth_00001", "2019-07-01T12:00:00Z")

    train_files = list_asg_files(cfg, split="train")
    names = {os.path.splitext(os.path.basename(p))[0] for p in train_files}
    assert "train_evt" in names
    assert "synth_00001" in names
    assert "test_evt" not in names


def test_event_time_partition_boundaries():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_config(os.path.join(root, "configs", "default.yaml"), [])
    assert event_time_partition(cfg, "2018-12-31T23:59:59Z") == "train"
    assert event_time_partition(cfg, "2019-03-01T00:00:00Z") == "val"
    assert event_time_partition(cfg, "2019-06-01T00:00:00Z") == "test"


def test_tier2_ib_penalty_not_doubled():
    from asgwm.models.bottleneck import soft_ib_penalty
    from asgwm.train.losses import tier2_total_loss

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cfg = load_config(os.path.join(root, "configs", "default.yaml"), ["losses.lambda_ib=0.01"])
    z = torch.ones(1, 4, 4, 4)
    flow = torch.zeros(1, 2, 4, 4)
    parts = tier2_total_loss(z[:, :1], z[:, :1], z, z[:, :-1], flow, torch.tensor([1.0]), cfg)
    direct = soft_ib_penalty(z[:, :-1], 0.01)
    assert parts["ib"].item() == pytest.approx(direct.item())
    assert parts["ib"].item() == pytest.approx(0.01 * 0.5, rel=1e-5)
