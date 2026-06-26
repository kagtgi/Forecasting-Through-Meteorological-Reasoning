"""Skill / realism metrics (eval.md sections 1A, 2).

Categorical scores (CSI/HSS/POD/FAR), scale-selective FSS and pooled CSI, probabilistic CRPS,
perceptual LPIPS (with a 1-SSIM fallback), and the radially-averaged power spectrum. All functions
accept numpy arrays or torch tensors and reduce over the batch where applicable.

References: jolliffe2012forecast (categorical), roberts2008fss (FSS), ravuri2021dgmr (pooled CSI),
gneiting2007crps / hersbach2000crps (CRPS), zhang2018lpips (LPIPS).
"""
from __future__ import annotations

from typing import Union

import numpy as np

try:  # torch is available for eval files but we still degrade gracefully
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    _HAS_TORCH = False

try:
    import lpips as _lpips_pkg  # type: ignore
    _HAS_LPIPS = True
except Exception:
    _lpips_pkg = None  # type: ignore
    _HAS_LPIPS = False

ArrayLike = Union[np.ndarray, "torch.Tensor"]

_LPIPS_NET = None  # lazily-constructed LPIPS network (cached)


def _np(x: ArrayLike) -> np.ndarray:
    """Coerce numpy array or torch tensor to a float64 numpy array."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# categorical contingency-table metrics (eval.md section 2)
# ---------------------------------------------------------------------------
def _contingency(pred: ArrayLike, obs: ArrayLike, thr: float, mask: ArrayLike = None):
    """Hits / misses / false-alarms / correct-negatives at threshold ``thr``.

    ``mask`` (optional, same shape, truthy = valid) restricts the count to valid pixels so
    radar clutter / no-coverage / below-threshold artifacts do not corrupt the score — needed
    for clean cross-region (NEXRAD / MRMS / HKO-7) evaluation (eval.md; HKO-7 mask handling).
    """
    p = _np(pred) >= thr
    o = _np(obs) >= thr
    if mask is not None:
        valid = _np(mask) > 0
        p = p & valid
        o = o & valid
        hits = float(np.sum(p & o))
        misses = float(np.sum(~p & o & valid))
        false_alarms = float(np.sum(p & ~o & valid))
        correct_neg = float(np.sum(~p & ~o & valid))
    else:
        hits = float(np.sum(p & o))
        misses = float(np.sum(~p & o))
        false_alarms = float(np.sum(p & ~o))
        correct_neg = float(np.sum(~p & ~o))
    return hits, misses, false_alarms, correct_neg


def csi(pred: ArrayLike, obs: ArrayLike, thr: float, mask: ArrayLike = None) -> float:
    """Critical Success Index = H / (H + M + F)."""
    h, m, f, _ = _contingency(pred, obs, thr, mask)
    denom = h + m + f
    return float(h / denom) if denom > 0 else 0.0


def pod(pred: ArrayLike, obs: ArrayLike, thr: float, mask: ArrayLike = None) -> float:
    """Probability Of Detection = H / (H + M)."""
    h, m, _, _ = _contingency(pred, obs, thr, mask)
    denom = h + m
    return float(h / denom) if denom > 0 else 0.0


def far(pred: ArrayLike, obs: ArrayLike, thr: float, mask: ArrayLike = None) -> float:
    """False Alarm Ratio = F / (H + F)."""
    h, _, f, _ = _contingency(pred, obs, thr, mask)
    denom = h + f
    return float(f / denom) if denom > 0 else 0.0


def hss(pred: ArrayLike, obs: ArrayLike, thr: float, mask: ArrayLike = None) -> float:
    """Heidke Skill Score against random chance."""
    h, m, f, c = _contingency(pred, obs, thr, mask)
    n = h + m + f + c
    if n == 0:
        return 0.0
    expected = ((h + m) * (h + f) + (c + m) * (c + f)) / n
    denom = n - expected
    return float((h + c - expected) / denom) if denom != 0 else 0.0


def sedi(pred: ArrayLike, obs: ArrayLike, thr: float, mask: ArrayLike = None) -> float:
    """Symmetric Extremal Dependence Index (Ferro & Stephenson 2011).

    SEDI = [ln F - ln H - ln(1-F) + ln(1-H)] / [ln F + ln H + ln(1-F) + ln(1-H)], with
    H = hit rate (POD) and F = false-alarm rate (POFD = F/(F+C)). Unlike CSI/FAR it stays
    informative as the event base-rate -> 0, so it is the better score at the high VIL
    thresholds [160,181,219] where extreme cells are rare (eval.md 1A; YingLong used SEDI for
    extreme-wind detection). Range -> 1 for a perfect forecast; 0 = no skill.
    """
    h, m, f, c = _contingency(pred, obs, thr, mask)
    H = h / (h + m) if (h + m) > 0 else 0.0   # hit rate (POD)
    F = f / (f + c) if (f + c) > 0 else 0.0   # false-alarm rate (POFD)
    eps = 1e-6
    H = min(max(H, eps), 1.0 - eps)
    F = min(max(F, eps), 1.0 - eps)
    num = np.log(F) - np.log(H) - np.log(1.0 - F) + np.log(1.0 - H)
    den = np.log(F) + np.log(H) + np.log(1.0 - F) + np.log(1.0 - H)
    return float(num / den) if den != 0 else 0.0


# ---------------------------------------------------------------------------
# scale-selective metrics (eval.md section 2)
# ---------------------------------------------------------------------------
def _pool2d(binary: np.ndarray, scale: int) -> np.ndarray:
    """Fractional coverage in non-overlapping ``scale``x``scale`` windows (mean pooling).

    Uses a uniform box convolution via cumulative sums; returns a same-shape fraction map.
    Operates on the last two dims of a 2-D array.
    """
    if scale <= 1:
        return binary.astype(np.float64)
    b = binary.astype(np.float64)
    # Integral image with a zero-padded border for O(1) window sums.
    csum = np.cumsum(np.cumsum(np.pad(b, ((1, 0), (1, 0))), axis=0), axis=1)
    h, w = b.shape
    r = scale
    out = np.zeros_like(b)
    for i in range(h):
        i0 = max(0, i - r // 2)
        i1 = min(h, i + r // 2 + (r % 2))
        for j in range(w):
            j0 = max(0, j - r // 2)
            j1 = min(w, j + r // 2 + (r % 2))
            total = csum[i1, j1] - csum[i0, j1] - csum[i1, j0] + csum[i0, j0]
            out[i, j] = total / ((i1 - i0) * (j1 - j0))
    return out


def _to_2d_list(x: np.ndarray):
    """Flatten an arbitrary-rank field into a list of 2-D maps (last two dims)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 2:
        return [a]
    flat = a.reshape((-1,) + a.shape[-2:])
    return [flat[i] for i in range(flat.shape[0])]


def fss(pred: ArrayLike, obs: ArrayLike, thr: float, scale: int) -> float:
    """Fractions Skill Score (roberts2008fss) at neighbourhood size ``scale``.

    FSS = 1 - MSE(P_frac, O_frac) / (mean(P_frac^2) + mean(O_frac^2)); 1 is perfect.
    """
    p_maps = _to_2d_list(_np(pred) >= thr)
    o_maps = _to_2d_list(_np(obs) >= thr)
    num, den = 0.0, 0.0
    for pm, om in zip(p_maps, o_maps):
        pf = _pool2d(pm, scale)
        of = _pool2d(om, scale)
        num += float(np.mean((pf - of) ** 2))
        den += float(np.mean(pf ** 2) + np.mean(of ** 2))
    if den <= 0:
        return 1.0  # both fields empty at this threshold -> perfect agreement
    return float(1.0 - num / den)


def pooled_csi(pred: ArrayLike, obs: ArrayLike, thr: float, scale: int) -> float:
    """Neighbourhood (pooled) CSI as in DGMR (ravuri2021dgmr).

    Max-pool the binary masks over ``scale``x``scale`` windows, then compute CSI: a hit is
    a predicted event within ``scale`` pixels of an observed event.
    """
    p_maps = _to_2d_list(_np(pred) >= thr)
    o_maps = _to_2d_list(_np(obs) >= thr)
    h = m = f = 0.0
    for pm, om in zip(p_maps, o_maps):
        pp = _pool2d(pm, scale) > 0
        op = _pool2d(om, scale) > 0
        h += float(np.sum(pp & op))
        m += float(np.sum(~pp & op))
        f += float(np.sum(pp & ~op))
    denom = h + m + f
    return float(h / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# probabilistic + perceptual metrics (eval.md section 2)
# ---------------------------------------------------------------------------
def crps_ensemble(ens: ArrayLike, obs: ArrayLike) -> float:
    """Ensemble CRPS (gneiting2007crps; Hersbach decomposition, hersbach2000crps).

    CRPS = E|X - y| - 0.5 E|X - X'| using the empirical ensemble distribution, averaged
    over all spatial / temporal elements.

    Args:
        ens: ensemble forecast, shape ``[K, ...]`` (K members along axis 0).
        obs: observation, shape ``[...]`` matching ``ens[0]``.
    """
    e = _np(ens)
    y = _np(obs)
    k = e.shape[0]
    # E|X - y| averaged over members.
    term1 = np.mean(np.abs(e - y[None, ...]), axis=0)
    # 0.5 E|X - X'| via the sorted-ensemble identity (O(K log K)).
    es = np.sort(e, axis=0)
    idx = np.arange(1, k + 1)
    weights = (2 * idx - k - 1).astype(np.float64)  # for E|X-X'| from order stats
    shape = (k,) + (1,) * (es.ndim - 1)
    term2 = np.sum(weights.reshape(shape) * es, axis=0) / (k * k)
    crps = term1 - term2
    return float(np.mean(crps))


def _ssim_2d(a: np.ndarray, b: np.ndarray) -> float:
    """Global SSIM (single-window) between two 2-D maps, range-normalized."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    rng = max(a.max() - a.min(), b.max() - b.min(), 1e-6)
    c1 = (0.01 * rng) ** 2
    c2 = (0.03 * rng) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / den) if den != 0 else 1.0


def lpips_metric(pred: ArrayLike, obs: ArrayLike) -> float:
    """Perceptual distance (zhang2018lpips). Falls back to ``1 - SSIM`` if lpips missing.

    Lower is better. Inputs are normalized per-pair to [0, 1] before the LPIPS net.
    """
    global _LPIPS_NET
    p = _np(pred)
    o = _np(obs)
    if _HAS_LPIPS and _HAS_TORCH:
        try:
            if _LPIPS_NET is None:
                _LPIPS_NET = _lpips_pkg.LPIPS(net="alex")
                _LPIPS_NET.eval()
            pm = _to_2d_list(p)
            om = _to_2d_list(o)
            dists = []
            for a, b in zip(pm, om):
                ta = _prep_lpips(a)
                tb = _prep_lpips(b)
                with torch.no_grad():
                    dists.append(float(_LPIPS_NET(ta, tb).item()))
            return float(np.mean(dists)) if dists else 0.0
        except Exception:
            pass  # fall through to SSIM fallback
    # Fallback: mean over 2-D maps of (1 - SSIM).
    pm = _to_2d_list(p)
    om = _to_2d_list(o)
    vals = [1.0 - _ssim_2d(a, b) for a, b in zip(pm, om)]
    return float(np.mean(vals)) if vals else 0.0


def _prep_lpips(a: np.ndarray):
    """Normalize a 2-D map to a 3-channel [-1,1] tensor for the LPIPS net."""
    lo, hi = a.min(), a.max()
    norm = (a - lo) / (hi - lo + 1e-8)
    t = torch.from_numpy(norm.astype(np.float32))[None, None]
    t = t.repeat(1, 3, 1, 1) * 2.0 - 1.0
    return t


def psd(field: ArrayLike) -> np.ndarray:
    """Radially-averaged power spectral density of a field (realism diagnostic).

    Accepts a 2-D map or a batched field (last two dims spatial); averages PSD over any
    leading dims. Returns a 1-D numpy array indexed by radial wavenumber bin.
    """
    maps = _to_2d_list(_np(field))
    spectra = []
    for m in maps:
        h, w = m.shape
        f = np.fft.fftshift(np.fft.fft2(m))
        power = np.abs(f) ** 2
        cy, cx = h // 2, w // 2
        yy, xx = np.indices((h, w))
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int64)
        n_bins = int(rr.max()) + 1
        tbin = np.bincount(rr.ravel(), power.ravel(), minlength=n_bins)
        nbin = np.bincount(rr.ravel(), minlength=n_bins)
        spectra.append(tbin / np.maximum(nbin, 1))
    max_len = max(len(s) for s in spectra)
    padded = np.array([np.pad(s, (0, max_len - len(s))) for s in spectra])
    return padded.mean(axis=0)
