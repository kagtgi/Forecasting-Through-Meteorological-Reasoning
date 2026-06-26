"""ASG grammar + schema tests (asgwm.asg.grammar / schema / render_nl).

Covers the load-bearing data contract: serialize/parse roundtrip, the hard IB-cap
quantization (motion / growth) and the object-count cap, and the NL ``assertion_check``
anti-hallucination guard. Pure-stdlib + numpy, so no importorskip is needed here.
"""
from __future__ import annotations

import math

import pytest

from asgwm.asg import (
    ASG,
    StormObject,
    REGIMES,
    N_MAX,
    MOTION_QUANT_KMH,
    quantize_motion,
    intensity_class,
    motion_to_compass,
)
from asgwm.asg.grammar import (
    serialize,
    serialize_object,
    parse,
    parse_strict,
    allowed_regime_tokens,
    object_line_regex,
)
from asgwm.asg.render_nl import render_NL, render_NL_delta, assertion_check


# ---------------------------------------------------------------------------
# serialize / parse roundtrip
# ---------------------------------------------------------------------------
def test_serialize_parse_roundtrip_preserves_structure(sample_asg):
    text = serialize(sample_asg)
    # canonical: one GLOBAL line + one OBJECT line per object
    lines = text.strip().splitlines()
    assert lines[0].startswith("GLOBAL(")
    assert len(lines) == 1 + sample_asg.n_objects

    back = parse(text)
    assert back.n_objects == sample_asg.n_objects
    assert back.global_regime == sample_asg.global_regime
    for o0, o1 in zip(sample_asg.objects, back.objects):
        assert o1.id == o0.id
        assert o1.regime == o0.regime
        # serialize quantizes the print precision; values match within format tol.
        assert o1.cy == pytest.approx(o0.cy, abs=1e-2)
        assert o1.cx == pytest.approx(o0.cx, abs=1e-2)
        assert o1.peak == pytest.approx(o0.peak, abs=0.1)
        assert o1.vy == pytest.approx(o0.vy, abs=0.1)
        assert o1.vx == pytest.approx(o0.vx, abs=0.1)


def test_serialize_object_matches_object_line_regex(sample_asg):
    import re

    rx = re.compile(object_line_regex())
    for o in sample_asg.objects:
        line = serialize_object(o)
        assert rx.fullmatch(line) is not None


def test_parse_is_tolerant_of_surrounding_prose(sample_asg):
    text = serialize(sample_asg)
    noisy = "Here is the scene:\n" + text + "\nThat is all."
    back = parse(noisy)
    assert back.n_objects == sample_asg.n_objects
    assert back.global_regime == sample_asg.global_regime


def test_parse_strict_accepts_canonical_and_rejects_malformed(sample_asg):
    text = serialize(sample_asg)
    # canonical text parses strictly
    strict = parse_strict(text)
    assert strict.n_objects == sample_asg.n_objects

    bad = "GLOBAL(regime=grow, n_objects=1)\nOBJECT(this is not valid)"
    with pytest.raises(ValueError):
        parse_strict(bad)


def test_empty_asg_serializes_and_parses():
    empty = ASG(objects=[], global_regime="steady")
    text = serialize(empty)
    assert "n_objects=0" in text
    back = parse(text)
    assert back.n_objects == 0
    assert back.global_regime == "steady"


def test_allowed_regime_tokens_are_the_regime_vocabulary():
    assert list(allowed_regime_tokens()) == list(REGIMES)


# ---------------------------------------------------------------------------
# capping + quantization (the hard IB cap)
# ---------------------------------------------------------------------------
def test_quantize_motion_snaps_to_bin_grid():
    assert quantize_motion(13.0) == pytest.approx(16.0)   # nearest multiple of 8
    assert quantize_motion(3.9) == pytest.approx(0.0)
    assert quantize_motion(-13.0) == pytest.approx(-16.0)
    # always a multiple of the bin width
    for v in (0.0, 5.0, 21.0, -30.0, 100.0):
        q = quantize_motion(v)
        assert q % MOTION_QUANT_KMH == pytest.approx(0.0, abs=1e-6)


def test_storm_object_quantized_applies_motion_and_growth_caps():
    o = StormObject(
        id=0, cy=1.0, cx=2.0, area=10.0, peak=30.0,
        vy=13.0, vx=-3.0, regime="grow", growth=0.123456, conf=0.7,
    )
    q = o.quantized()
    assert q.vy == pytest.approx(16.0)
    assert q.vx == pytest.approx(0.0)
    # growth kept to GROWTH_SIGFIGS significant figures (2 sig figs -> 0.12)
    assert q.growth == pytest.approx(0.12, abs=1e-9)
    # non-quantized fields are untouched
    assert q.cy == o.cy and q.peak == o.peak and q.id == o.id


def test_capped_enforces_object_budget_and_quantizes():
    objs = [
        StormObject(
            id=i, cy=0.0, cx=0.0, area=float(i + 1), peak=float(i + 1),
            vy=13.0, vx=0.0, regime="grow", growth=0.111, conf=1.0,
        )
        for i in range(N_MAX + 6)
    ]
    asg = ASG(objects=objs, global_regime="grow")
    capped = asg.capped(N_MAX)
    assert capped.n_objects == N_MAX
    # kept the most intense cells (highest peak/area first)
    assert capped.objects[0].peak == pytest.approx(float(N_MAX + 6))
    # objects come back quantized
    assert capped.objects[0].vy == pytest.approx(16.0)


def test_capped_smaller_than_budget_is_identity_count(sample_asg):
    capped = sample_asg.capped(N_MAX)
    assert capped.n_objects == sample_asg.n_objects


def test_to_dict_from_dict_roundtrip(sample_asg):
    d = sample_asg.to_dict()
    back = ASG.from_dict(d)
    assert back.n_objects == sample_asg.n_objects
    assert back.global_regime == sample_asg.global_regime
    assert [o.id for o in back.objects] == [o.id for o in sample_asg.objects]


def test_invalid_regime_rejected():
    with pytest.raises(ValueError):
        StormObject(
            id=0, cy=0, cx=0, area=1, peak=1, vy=0, vx=0,
            regime="not_a_regime", growth=0, conf=1.0,
        )
    with pytest.raises(ValueError):
        ASG(objects=[], global_regime="bogus")


# ---------------------------------------------------------------------------
# intensity / compass helpers used by the render + assertion check
# ---------------------------------------------------------------------------
def test_intensity_class_bins():
    assert intensity_class(5.0) == "light"
    assert intensity_class(30.0) == "moderate"
    assert intensity_class(60.0) == "heavy"


def test_motion_to_compass_basic_directions():
    assert motion_to_compass(0.0, 0.0) == "stationary"
    # +x = East
    assert motion_to_compass(0.0, 5.0) == "E"
    # +y = South (rows increase downward) -> geographic South
    assert motion_to_compass(5.0, 0.0) == "S"


# ---------------------------------------------------------------------------
# assertion_check (anti-hallucination)
# ---------------------------------------------------------------------------
def test_assertion_check_passes_for_grounded_render(sample_asg):
    nl = render_NL(sample_asg)
    flagged = assertion_check(nl, sample_asg)
    assert flagged == []


def test_assertion_check_flags_ungrounded_intensity(sample_asg):
    # the sample ASG has heavy (peak 48) + light (peak 15) cells but no "moderate".
    classes = {intensity_class(o.peak) for o in sample_asg.objects}
    assert "moderate" not in classes
    nl = "The primary cell is a moderate-intensity system holding steady."
    flagged = assertion_check(nl, sample_asg)
    assert any("moderate" in s.lower() for s in flagged)


def test_render_nl_delta_runs(sample_asg):
    # move + grow the primary cell to produce a non-trivial delta sentence.
    asg_th = ASG.from_dict(sample_asg.to_dict())
    asg_th.objects[0].cy += 20.0
    asg_th.objects[0].peak += 10.0
    text = render_NL_delta(sample_asg, asg_th)
    assert isinstance(text, str) and len(text) > 0
