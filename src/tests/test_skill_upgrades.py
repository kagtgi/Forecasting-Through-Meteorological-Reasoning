"""Tests for the related-work upgrades: HKO-7 balanced loss + SEDI + mask-aware scoring."""
import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# SEDI + mask-aware contingency (eval/metrics.py)
# --------------------------------------------------------------------------- #
from asgwm.eval import metrics as M


def test_sedi_perfect_and_noskill():
    obs = np.zeros((32, 32), np.float32)
    obs[:4, :4] = 100.0  # a rare extreme event (base rate ~1.5%)
    perfect = obs.copy()
    assert M.sedi(perfect, obs, thr=50.0) > 0.9          # near 1 for a perfect forecast
    rng = np.random.default_rng(0)
    noise = rng.uniform(0, 40, size=obs.shape).astype(np.float32)  # never exceeds thr where obs does
    assert M.sedi(noise, obs, thr=50.0) <= 0.0           # no skill / negative


def test_sedi_in_range_and_finite():
    rng = np.random.default_rng(1)
    obs = rng.uniform(0, 255, (20, 20)).astype(np.float32)
    pred = obs + rng.normal(0, 10, obs.shape).astype(np.float32)
    v = M.sedi(pred, obs, thr=181.0)
    assert np.isfinite(v) and -1.0001 <= v <= 1.0001


def test_mask_excludes_pixels():
    pred = np.zeros((10, 10), np.float32)
    obs = np.zeros((10, 10), np.float32)
    obs[0, 0] = 100.0           # real event...
    pred[0, 0] = 100.0          # ...correctly predicted (a hit)
    pred[9, 9] = 100.0          # plus one false alarm (clutter)
    mask = np.ones((10, 10), np.float32)
    mask[9, 9] = 0.0            # mark the clutter pixel invalid
    # Without mask: H=1,F=1 -> CSI 0.5, FAR 0.5. Masking the clutter -> H=1,F=0 -> CSI 1, FAR 0.
    assert M.csi(pred, obs, 50.0) == pytest.approx(0.5)
    assert M.csi(pred, obs, 50.0, mask=mask) == pytest.approx(1.0)
    assert M.far(pred, obs, 50.0, mask=mask) == 0.0


# --------------------------------------------------------------------------- #
# Balanced reconstruction loss (train/losses.py)
# --------------------------------------------------------------------------- #
def test_balanced_weight_map_and_mse():
    torch = pytest.importorskip("torch")
    from asgwm.train.losses import balanced_weight_map, balanced_mse

    thr = [16, 74, 133, 181]
    wts = [1, 2, 5, 10, 30]
    target = torch.tensor([[0.0, 20.0, 100.0, 150.0, 200.0]])  # one per bin boundary
    w = balanced_weight_map(target, thr, wts)
    assert w.flatten().tolist() == [1.0, 2.0, 5.0, 10.0, 30.0]

    # validity mask zeroes selected pixels
    mask = torch.tensor([[1.0, 1.0, 0.0, 1.0, 1.0]])
    wm = balanced_weight_map(target, thr, wts, valid_mask=mask)
    assert float(wm.flatten()[2]) == 0.0

    # balanced_mse == plain MSE when weights are uniform
    pred = target + 1.0
    uni = torch.ones_like(target)
    assert abs(float(balanced_mse(pred, target, uni)) - 1.0) < 1e-6

    # a heavy-pixel error dominates more than a light-pixel error of equal magnitude
    pred_heavy_err = target.clone(); pred_heavy_err[0, 4] += 10.0   # error on the w=30 pixel
    pred_light_err = target.clone(); pred_light_err[0, 0] += 10.0   # error on the w=1 pixel
    w_full = balanced_weight_map(target, thr, wts)
    assert float(balanced_mse(pred_heavy_err, target, w_full)) > float(balanced_mse(pred_light_err, target, w_full))


def test_tier2_balanced_loss_path_runs():
    torch = pytest.importorskip("torch")
    from asgwm.train.losses import tier2_total_loss
    from asgwm.utils.config import Config

    cfg = Config({"losses": {"balanced_loss": True,
                             "balanced_thresholds": [16, 74, 133, 181],
                             "balanced_weights": [1, 2, 5, 10, 30]}})
    B, H, W = 2, 16, 16
    pred = torch.rand(B, 1, H, W) * 255.0
    target = torch.rand(B, 1, H, W) * 255.0
    Z = torch.zeros(B, 3, H, W)
    asg_cont = torch.zeros(B, 8)
    flow = torch.zeros(B, 2, H, W)
    growth_budget = target.clamp(min=0).sum(dim=(1, 2, 3))
    out = tier2_total_loss(pred, target, Z, asg_cont, flow, growth_budget, cfg)
    assert torch.isfinite(out["total"]) and float(out["render"]) >= 0.0
