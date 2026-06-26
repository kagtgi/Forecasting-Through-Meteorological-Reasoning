"""Tests for the symbolic admissibility layer (asgwm.symbolic)."""
import copy

from asgwm.asg import ASG, StormObject
from asgwm.symbolic import certify_transition, ambiguity_flag, ConstraintBounds, REGIME_FSM


def _clean_pair():
    t = ASG(objects=[
        StormObject(1, 120, 100, 240, 36, -10, 8, "grow", 1.5, 0.9),
        StormObject(2, 60, 200, 90, 20, 5, -3, "steady", 0.0, 0.7),
    ], global_regime="grow", context={"cape": 1800.0, "cin": 20.0, "context_available": 1.0})
    th = ASG(objects=[
        StormObject(1, 110, 108, 300, 40, -10, 8, "grow", 1.2, 0.85),
        StormObject(2, 65, 197, 85, 19, 5, -3, "steady", 0.0, 0.7),
    ], global_regime="grow", context=t.context)
    return t, th


def test_clean_transition_is_admissible():
    t, th = _clean_pair()
    cert = certify_transition(t, th, horizon_min=30, dx_km=1.0)
    assert cert.ok, cert.diagnosis()
    assert cert.core == []


def test_teleport_caught():
    t, th = _clean_pair()
    bad = copy.deepcopy(th)
    bad.objects[0].cy += 500
    cert = certify_transition(t, bad, horizon_min=30, dx_km=1.0)
    assert not cert.ok
    assert "kinematic_speed" in cert.core or "advection_residual" in cert.core


def test_intensity_jump_caught():
    t, th = _clean_pair()
    bad = copy.deepcopy(th)
    bad.objects[0].peak += 60
    cert = certify_transition(t, bad, horizon_min=30, dx_km=1.0)
    assert not cert.ok
    assert "intensity_rate" in cert.core


def test_grow_but_weaken_caught():
    t, th = _clean_pair()
    bad = copy.deepcopy(th)
    bad.objects[0].regime = "grow"
    bad.objects[0].peak = th.objects[0].peak - 25
    cert = certify_transition(t, bad, horizon_min=30, dx_km=1.0)
    assert not cert.ok
    assert "tendency_consistency" in cert.core


def test_forbidden_regime_transition_caught():
    t, th = _clean_pair()
    t2 = copy.deepcopy(t)
    t2.objects[0].regime = "decay"
    th2 = copy.deepcopy(th)
    th2.objects[0].regime = "init"          # decay -> init is forbidden by the FSM
    cert = certify_transition(t2, th2, horizon_min=30, dx_km=1.0)
    assert not cert.ok
    assert "regime_transition" in cert.core


def test_regime_fsm_well_formed():
    for r, allowed in REGIME_FSM.items():
        assert r in allowed or r == "init"  # most regimes can persist
    assert "init" not in REGIME_FSM["decay"]   # no abrupt re-initiation


def test_dual_sat_ambiguity():
    confident = ambiguity_flag({"cape": 2800, "cin": 10})
    assert confident["initiation_admissible"] and not confident["ambiguous"]
    uncertain = ambiguity_flag({"cape": 900, "cin": 45})
    assert uncertain["ambiguous"]              # both initiation and non-initiation admissible
    nope = ambiguity_flag({"cape": 200, "cin": 120})
    assert not nope["initiation_admissible"] and not nope["ambiguous"]
