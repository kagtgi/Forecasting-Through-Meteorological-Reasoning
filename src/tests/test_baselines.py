"""Tests for the pluggable baseline registry (asgwm.baselines)."""
import numpy as np
import pytest

import asgwm.baselines as B


def test_registry_contents():
    assert set(B.all_names()) == {"pysteps", "rainnet", "nowcastnet", "langprecip", "thor"}
    assert B.available_names() == ["pysteps"]            # only pysteps implemented now
    assert B.HEADLINE[0] == "pysteps" and B.HEADLINE[-1] == "thor"


def test_display_and_family():
    assert B.display_name("nowcastnet") == "NowcastNet"
    assert "Physics" in B.family("nowcastnet")
    assert B.display_name("thor") == "ThoR"


def test_pysteps_predicts_sequence():
    ps = B.get("pysteps")
    assert ps.is_available()
    hist = np.zeros((4, 32, 32), dtype=np.float32)
    hist[:, 12:20, 12:20] = 30.0          # a blob to advect
    out = ps.predict(hist, {}, n_out=6)
    assert out.shape == (6, 32, 32)
    ens = ps.predict_ensemble(hist, {}, n_out=6, k=3)
    assert ens.shape == (3, 6, 32, 32)


def test_stub_unavailable_and_raises():
    for name in ("rainnet", "nowcastnet", "langprecip", "thor"):
        b = B.get(name)
        assert not b.is_available()
        with pytest.raises(RuntimeError):
            b.predict(np.zeros((4, 8, 8), np.float32), {}, 3)
