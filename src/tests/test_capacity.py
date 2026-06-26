"""Bottleneck capacity audit tests (asgwm.eval.capacity).

The faithfulness-by-compression argument needs the ASG channel capacity to be strictly
smaller than the raw radar input's. These tests assert ``asg_bits << input_bits`` on the
real default config and that ``capacity_bits`` grows with the object budget. Pure stdlib
+ the schema, so no importorskip is needed (yaml is guarded by the ``tiny_cfg`` fixture).
"""
from __future__ import annotations

import pytest

from asgwm.eval.capacity import capacity_bits, capacity_audit, capacity_sweep


def _full_cfg():
    """The real default config (full grid / channel counts) for the bits comparison."""
    pytest.importorskip("yaml")
    import os

    from asgwm.utils.config import load_config

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return load_config(os.path.join(root, "configs", "default.yaml"), [])


def test_asg_bits_much_smaller_than_input_bits():
    cfg = _full_cfg()
    audit = capacity_audit(cfg)
    assert audit["ok"] is True
    assert audit["asg_bits"] < audit["input_bits"]
    # "<<": at least two orders of magnitude smaller on the default 384x384x13x4 input.
    assert audit["asg_bits"] * 100 < audit["input_bits"]
    assert 0.0 < audit["ratio"] < 0.01


def test_capacity_audit_reports_expected_keys():
    cfg = _full_cfg()
    audit = capacity_audit(cfg)
    for key in ("asg_bits", "input_bits", "ratio", "ok", "n_max", "attr_bits_per_object"):
        assert key in audit
    assert audit["n_max"] == int(cfg.get_path("asg.n_max", 16))
    assert audit["attr_bits_per_object"] > 0.0


def test_capacity_bits_nondecreasing_in_nmax():
    cfg = _full_cfg()
    bits = [capacity_bits(n, cfg) for n in (2, 4, 8, 16, 32)]
    assert all(b1 <= b2 for b1, b2 in zip(bits, bits[1:]))
    # strictly larger between the extremes (more objects -> more attribute bits).
    assert bits[-1] > bits[0]


def test_capacity_stays_below_input_across_the_sweep():
    cfg = _full_cfg()
    sweep = capacity_sweep(cfg)
    assert sweep["nmax"] == list(cfg.get_path("eval.capacity_sweep_nmax"))
    assert len(sweep["bits"]) == len(sweep["nmax"])
    input_bits = sweep["input_bits"]
    # every swept budget keeps the ASG capacity well under the raw input capacity.
    assert all(b < input_bits for b in sweep["bits"])


def test_capacity_sweep_csi_is_nan_without_callbacks():
    import math

    cfg = _full_cfg()
    sweep = capacity_sweep(cfg)
    assert all(math.isnan(c) for c in sweep["csi"])


def test_tiny_cfg_audit_also_compresses(tiny_cfg):
    # the shrunk config (32x32 grid) still satisfies the compression precondition.
    audit = capacity_audit(tiny_cfg)
    assert audit["ok"] is True
    assert audit["asg_bits"] < audit["input_bits"]
