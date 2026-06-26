"""Structured ASG intervention tests (asgwm.interventions).

Each ``perturb_asg`` kind must change the right field and leave the others alone, and
``expected_effect`` must predict the field-level direction the faithfulness scorer
compares against. Pure ASG-schema + numpy, so no importorskip is required.
"""
from __future__ import annotations

import math

import pytest

from asgwm.asg import ASG, StormObject
from asgwm.interventions import perturb_asg, expected_effect, intervention_pairs


@pytest.fixture
def asg_one():
    """A single moving, growing cell with a known motion vector (vy=0, vx=10)."""
    o = StormObject(
        id=0, cy=10.0, cx=10.0, area=100.0, peak=40.0,
        vy=0.0, vx=10.0, regime="grow", growth=0.5, conf=1.0,
    )
    return ASG(objects=[o], global_regime="grow", meta={"km_per_pixel": 1.0})


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------
def test_translate_shifts_along_motion(asg_one):
    out = perturb_asg(asg_one, "translate", km=20.0, obj_idx=0)
    o0, o1 = asg_one.objects[0], out.objects[0]
    # motion is purely +x; at 1 km/px a 20 km shift moves cx by +20, cy unchanged.
    assert o1.cx - o0.cx == pytest.approx(20.0, abs=1e-5)
    assert o1.cy - o0.cy == pytest.approx(0.0, abs=1e-5)
    # other attributes untouched
    assert o1.peak == o0.peak and o1.growth == o0.growth and o1.regime == o0.regime
    # original is not mutated
    assert asg_one.objects[0].cx == 10.0


def test_translate_stationary_cell_falls_back_to_east(asg_one):
    asg_one.objects[0].vx = 0.0
    asg_one.objects[0].vy = 0.0
    out = perturb_asg(asg_one, "translate", km=15.0, obj_idx=0)
    assert out.objects[0].cx - 10.0 == pytest.approx(15.0, abs=1e-5)
    assert out.objects[0].cy == pytest.approx(10.0, abs=1e-5)


# ---------------------------------------------------------------------------
# regime_flip
# ---------------------------------------------------------------------------
def test_regime_flip_grow_to_decay_negates_growth(asg_one):
    out = perturb_asg(asg_one, "regime_flip", obj_idx=0)
    assert asg_one.objects[0].regime == "grow"
    assert out.objects[0].regime == "decay"
    assert out.objects[0].growth == pytest.approx(-asg_one.objects[0].growth)
    # motion / position unchanged
    assert out.objects[0].cx == asg_one.objects[0].cx
    assert out.objects[0].vx == asg_one.objects[0].vx


def test_regime_flip_decay_to_grow(asg_one):
    asg_one.objects[0].regime = "decay"
    asg_one.objects[0].growth = -0.3
    out = perturb_asg(asg_one, "regime_flip", obj_idx=0)
    assert out.objects[0].regime == "grow"
    assert out.objects[0].growth == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# growth_scale
# ---------------------------------------------------------------------------
def test_growth_scale_scales_only_growth(asg_one):
    out = perturb_asg(asg_one, "growth_scale", factor=3.0, obj_idx=0)
    assert out.objects[0].growth == pytest.approx(3.0 * asg_one.objects[0].growth)
    # regime / motion / centroid untouched
    assert out.objects[0].regime == asg_one.objects[0].regime
    assert out.objects[0].vx == asg_one.objects[0].vx
    assert out.objects[0].cx == asg_one.objects[0].cx


# ---------------------------------------------------------------------------
# motion_rotate
# ---------------------------------------------------------------------------
def test_motion_rotate_rotates_motion_vector(asg_one):
    # rotate (vy=0, vx=10) by 90 deg -> (vy=10, vx=0) per the module's convention.
    out = perturb_asg(asg_one, "motion_rotate", deg=90.0, obj_idx=0)
    assert out.objects[0].vy == pytest.approx(10.0, abs=1e-4)
    assert out.objects[0].vx == pytest.approx(0.0, abs=1e-4)
    # speed magnitude preserved
    before = math.hypot(asg_one.objects[0].vy, asg_one.objects[0].vx)
    after = math.hypot(out.objects[0].vy, out.objects[0].vx)
    assert after == pytest.approx(before, abs=1e-4)
    # centroid unchanged
    assert out.objects[0].cx == asg_one.objects[0].cx


def test_unknown_kind_raises(asg_one):
    with pytest.raises(ValueError):
        perturb_asg(asg_one, "not_a_kind")


# ---------------------------------------------------------------------------
# expected_effect
# ---------------------------------------------------------------------------
def test_expected_effect_translate():
    eff = expected_effect("translate", km=20.0, km_per_pixel=2.0)
    assert eff["displacement_px"] == pytest.approx(10.0)
    assert eff["intensity_factor"] == pytest.approx(1.0)
    assert eff["rotation_deg"] == pytest.approx(0.0)


def test_expected_effect_regime_flip_reverses_sign():
    # a growing cell (growth_sign +1) weakens under the flip -> negative sign.
    eff = expected_effect("regime_flip", growth_sign=1.0)
    assert eff["sign"] == pytest.approx(-1.0)
    eff2 = expected_effect("regime_flip", growth_sign=-1.0)
    assert eff2["sign"] == pytest.approx(1.0)


def test_expected_effect_growth_scale_sign_follows_factor():
    # factor>1 amplifies the existing (positive) tendency; factor<1 opposes it.
    assert expected_effect("growth_scale", factor=3.0, growth_sign=1.0)["sign"] == pytest.approx(1.0)
    assert expected_effect("growth_scale", factor=0.5, growth_sign=1.0)["sign"] == pytest.approx(-1.0)
    eff = expected_effect("growth_scale", factor=3.0, growth_sign=1.0)
    assert eff["intensity_factor"] == pytest.approx(3.0)


def test_expected_effect_motion_rotate_reports_rotation():
    eff = expected_effect("motion_rotate", deg=45.0)
    assert eff["rotation_deg"] == pytest.approx(45.0)
    assert eff["displacement_px"] == pytest.approx(0.0)


def test_expected_effect_unknown_kind_raises():
    with pytest.raises(ValueError):
        expected_effect("nope")


# ---------------------------------------------------------------------------
# intervention_pairs ties perturbation + expected_effect together
# ---------------------------------------------------------------------------
def test_intervention_pairs_builds_triples_with_expected(asg_one):
    types = ["translate", "regime_flip", "growth_scale", "motion_rotate"]
    pairs = intervention_pairs(asg_one, types)
    assert len(pairs) == len(types)
    kinds = {meta["kind"] for (_, _, meta) in pairs}
    assert kinds == set(types)
    for orig, perturbed, meta in pairs:
        assert "expected" in meta
        assert isinstance(perturbed, ASG)
        # the perturbed graph keeps the same object count for these single-cell edits.
        assert perturbed.n_objects == orig.n_objects


def test_intervention_pairs_empty_for_no_objects():
    empty = ASG(objects=[], global_regime="steady")
    assert intervention_pairs(empty, ["translate"]) == []
