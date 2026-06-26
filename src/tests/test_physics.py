"""Differentiable physics operator tests (asgwm.physics).

torch is required for this module, so the whole file is skipped in a torch-free env.
Tests stay tiny (8x8 fields, single step) and assert the load-bearing invariants:
the advection warp preserves shape, advecting at zero flow is the identity, point
advection is exact, and the continuity / smoothness / spectral operators produce finite
scalars and back-propagate.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from asgwm import physics


# ---------------------------------------------------------------------------
# semi-Lagrangian advection
# ---------------------------------------------------------------------------
def test_advect_warp_preserves_shape():
    field = torch.rand(2, 3, 8, 8)
    flow = torch.randn(2, 2, 8, 8) * 0.5
    out = physics.semi_lagrangian_advect(field, flow, dt=1.0)
    assert out.shape == field.shape


def test_advect_identity_at_zero_flow():
    field = torch.rand(1, 1, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)
    out = physics.semi_lagrangian_advect(field, flow, dt=1.0)
    # backward trace with zero velocity samples each pixel exactly -> identity.
    assert torch.allclose(out, field, atol=1e-5)


def test_advect_zero_dt_is_identity():
    field = torch.rand(1, 1, 8, 8)
    flow = torch.randn(1, 2, 8, 8)
    out = physics.semi_lagrangian_advect(field, flow, dt=0.0)
    assert torch.allclose(out, field, atol=1e-5)


def test_advect_constant_shift_translates_field():
    # a single bright pixel shifted by an integer flow lands at the shifted location.
    field = torch.zeros(1, 1, 9, 9)
    field[0, 0, 4, 4] = 1.0
    flow = torch.zeros(1, 2, 9, 9)
    flow[0, 0] = 1.0  # vy = +1 (rows/step, downward)
    flow[0, 1] = 2.0  # vx = +2 (cols/step, East)
    out = physics.semi_lagrangian_advect(field, flow, dt=1.0)
    # destination of the source pixel (4,4): row 4+1=5, col 4+2=6
    assert out[0, 0, 5, 6].item() == pytest.approx(1.0, abs=1e-4)


# ---------------------------------------------------------------------------
# point advection
# ---------------------------------------------------------------------------
def test_advect_points_exact():
    centroids = torch.tensor([[2.0, 3.0], [10.0, 0.0]])
    motion = torch.tensor([[1.0, -1.0], [0.0, 2.0]])
    out = physics.advect_points(centroids, motion, dt=2.0)
    expected = torch.tensor([[4.0, 1.0], [10.0, 4.0]])
    assert torch.allclose(out, expected)


def test_advect_points_zero_motion_is_identity():
    centroids = torch.tensor([[5.0, 7.0]])
    out = physics.advect_points(centroids, torch.zeros_like(centroids), dt=3.0)
    assert torch.allclose(out, centroids)


# ---------------------------------------------------------------------------
# continuity / smoothness residuals run + are scalar + differentiable
# ---------------------------------------------------------------------------
def test_continuity_residual_is_scalar_and_finite():
    g_t = torch.rand(1, 1, 8, 8)
    g_th = torch.rand(1, 1, 8, 8)
    flow = torch.randn(1, 2, 8, 8) * 0.1
    res = physics.continuity_residual(g_t, g_th, flow, dt=1.0)
    assert res.ndim == 0
    assert torch.isfinite(res)
    assert res.item() >= 0.0


def test_continuity_residual_zero_when_steady_and_no_flow():
    g = torch.rand(1, 1, 8, 8)
    flow = torch.zeros(1, 2, 8, 8)
    res = physics.continuity_residual(g, g, flow, dt=1.0)
    assert res.item() == pytest.approx(0.0, abs=1e-6)


def test_motion_smoothness_residual_zero_for_constant_field():
    flow = torch.ones(1, 2, 8, 8) * 2.5
    res = physics.motion_smoothness_residual(flow)
    assert res.item() == pytest.approx(0.0, abs=1e-6)


def test_motion_smoothness_residual_positive_for_varying_field():
    flow = torch.randn(1, 2, 8, 8)
    res = physics.motion_smoothness_residual(flow)
    assert res.item() > 0.0


def test_divergence_shape():
    v = torch.randn(2, 2, 8, 8)
    d = physics.divergence(v)
    assert d.shape == (2, 1, 8, 8)


def test_nonneg_penalty_and_mass_budget():
    field = torch.tensor([[[[-1.0, 2.0], [3.0, -4.0]]]])
    assert physics.nonneg_penalty(field).item() > 0.0
    assert physics.nonneg_penalty(torch.ones(1, 1, 4, 4)).item() == pytest.approx(0.0)
    total = field.clamp(min=0).sum(dim=(1, 2, 3))
    assert physics.mass_budget_residual(field, total).item() == pytest.approx(0.0, abs=1e-5)


def test_continuity_residual_backpropagates():
    g_th = torch.rand(1, 1, 8, 8, requires_grad=True)
    g_t = torch.rand(1, 1, 8, 8)
    flow = torch.randn(1, 2, 8, 8) * 0.1
    res = physics.continuity_residual(g_t, g_th, flow, dt=1.0)
    res.backward()
    assert g_th.grad is not None
    assert torch.isfinite(g_th.grad).all()


# ---------------------------------------------------------------------------
# spectral diagnostics
# ---------------------------------------------------------------------------
def test_radial_power_spectrum_shape():
    field = torch.rand(2, 1, 8, 8)
    ps = physics.radial_power_spectrum(field, n_bins=4)
    assert ps.shape == (2, 4)
    assert torch.isfinite(ps).all()


def test_spectral_loss_runs_and_is_zero_for_identical_fields():
    field = torch.rand(1, 1, 8, 8)
    loss_same = physics.spectral_loss(field, field)
    assert loss_same.item() == pytest.approx(0.0, abs=1e-6)
    loss_diff = physics.spectral_loss(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8))
    assert loss_diff.ndim == 0 and torch.isfinite(loss_diff)
