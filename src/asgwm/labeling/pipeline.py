"""ASG auto-labeling pipeline (datasource.md section 2; architecture.md section 2).

Ties the classical-CV labeling steps into typed ASGs:
    motion (optical flow) -> tracking (cells -> tracks) -> tendency/regime
    -> context co-location -> StormObject set -> ASG_t and ASG_{t+h} -> ASGSequence.

Two entry points:
    build_asg_pair(...)   low-level: given history + future frame stacks + context.
    autolabel_event(...)  high-level: given a cached event dict + Config.

Design commitments from the task contract:
  * The advection (motion) field estimated here is the SAME field reused as
    Stage-C's future-blind path and the Tier-0 baseline (architecture.md
    section 4). It is exposed via `estimate_label_motion` and stashed on the
    returned ASGs' meta (`meta['flow']`) so `asgwm.data.advection` consumes the
    identical source.
  * The per-object / per-pair NL rationale uses render_NL / render_NL_delta
    (datasource.md section 5; the constrained render is the anti-hallucination
    contract). Rationales are stashed on ASG.meta so the curriculum builder reuses
    them without recomputation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from asgwm.asg import (
    ASG,
    ASGSequence,
    StormObject,
    N_MAX,
    render_NL,
    render_NL_delta,
)
from asgwm.utils.config import Config

from . import motion as motion_mod
from . import tracking as tracking_mod
from . import regime as regime_mod


# ---------------------------------------------------------------------------
# Intensity scaling. SEVIR VIL is stored on a 0..255 byte scale; the ASG peak is
# documented as "VIL -> dBZ" (architecture.md section 2). We use a monotonic
# linear map onto a dBZ-like 0..~70 range so intensity_class / eval thresholds
# (16/35/45 dBZ) are meaningful. The map is the identity-friendly default; if the
# input already looks like dBZ (<= ~80) it is passed through.
# ---------------------------------------------------------------------------
_VIL_BYTE_MAX = 255.0
_DBZ_MAX = 70.0


def vil_to_dbz(vil: np.ndarray | float) -> np.ndarray | float:
    """Map VIL (0..255 byte scale or already-dBZ) to a dBZ-like scale.

    Linear, monotonic, range-preserving for class boundaries. Values that already
    sit in a plausible dBZ range (<= 80) are returned unchanged.
    """
    arr = np.asarray(vil, dtype=np.float32)
    scalar = arr.ndim == 0
    if float(np.nanmax(arr)) <= 80.0:
        out = arr
    else:
        out = arr / _VIL_BYTE_MAX * _DBZ_MAX
    return float(out) if scalar else out


def estimate_label_motion(frames_hist: np.ndarray) -> np.ndarray:
    """The labeling/future-blind motion field [2,H,W] (px/step) from history only.

    Exposed so `asgwm.data.advection.advect_blind` reuses the IDENTICAL motion
    source (architecture.md section 4 — the future-blind path).
    """
    return motion_mod.estimate_motion(frames_hist)


def _px_per_h_to_kmh(v_px_per_step: float, dx_km: float, dt_min: float) -> float:
    """Convert a velocity in px/step to km/h."""
    steps_per_h = 60.0 / max(dt_min, 1e-6)
    return v_px_per_step * dx_km * steps_per_h


def _build_objects(
    tracks: List[dict],
    flow: np.ndarray,
    dx_km: float,
    dt_min: float,
    use_track_end: bool,
) -> List[StormObject]:
    """Turn tracks into StormObjects sampled at the window start (or end).

    Motion per object blends the per-track centroid velocity with the local
    optical-flow vector at the object's centroid; growth/regime from the track's
    intensity series. Peak intensity is mapped VIL->dBZ.
    """
    h, w = flow.shape[1], flow.shape[2]
    objs: List[StormObject] = []
    for tr in tracks:
        frames = tr["frames"]
        if not frames:
            continue
        ref = frames[-1] if use_track_end else frames[0]
        cy, cx = ref["cy"], ref["cx"]

        # Motion: prefer the track's own displacement; fall back to local flow.
        vy_px, vx_px = tracking_mod.track_motion(tr, dt_steps=1.0)
        if abs(vy_px) < 1e-6 and abs(vx_px) < 1e-6:
            iy = int(np.clip(round(cy), 0, h - 1))
            ix = int(np.clip(round(cx), 0, w - 1))
            vy_px, vx_px = float(flow[0, iy, ix]), float(flow[1, iy, ix])

        peaks = [vil_to_dbz(f["peak"]) for f in frames]
        g = regime_mod.growth_scalar(peaks, dt_min)
        reg = regime_mod.classify_regime(tr, dt_min)
        # Track-length / intensity heuristic for perception confidence in [0.3,1].
        conf = float(np.clip(0.4 + 0.1 * len(frames), 0.3, 1.0))

        objs.append(
            StormObject(
                id=int(tr["id"]),
                cy=float(cy),
                cx=float(cx),
                area=float(ref["area"]) * (dx_km ** 2),
                peak=float(vil_to_dbz(ref["peak"])),
                vy=_px_per_h_to_kmh(vy_px, dx_km, dt_min),
                vx=_px_per_h_to_kmh(vx_px, dx_km, dt_min),
                regime=reg,
                growth=float(g),
                conf=conf,
            )
        )
    return objs


def _advect_objects(
    objs: List[StormObject],
    horizon_min: int,
    dx_km: float,
) -> List[StormObject]:
    """Advect StormObjects forward by their motion vectors over the horizon.

    Used to synthesize ASG_{t+h} when no future frames are tracked for an object
    (residual-on-advection baseline; architecture.md section 3). Motion is km/h;
    convert to grid px over the horizon. Regime/growth carried forward.
    """
    out: List[StormObject] = []
    hours = horizon_min / 60.0
    for o in objs:
        dcy_px = (o.vy * hours) / max(dx_km, 1e-6)
        dcx_px = (o.vx * hours) / max(dx_km, 1e-6)
        # Apply growth tendency to peak over the horizon (mild, clamped).
        new_peak = float(np.clip(o.peak + o.growth * horizon_min, 0.0, _DBZ_MAX))
        out.append(
            StormObject(
                id=o.id,
                cy=o.cy + dcy_px,
                cx=o.cx + dcx_px,
                area=o.area,
                peak=new_peak,
                vy=o.vy,
                vx=o.vx,
                regime=o.regime,
                growth=o.growth,
                conf=o.conf,
            )
        )
    return out


def _growth_field(frames_hist: np.ndarray, size: int, dt_min: float) -> np.ndarray:
    """Low-res growth-decay field G_t (architecture.md section 2): dVIL/dt downsampled."""
    arr = vil_to_dbz(np.asarray(frames_hist, dtype=np.float32))
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.shape[0] < 2:
        diff = np.zeros_like(arr[0])
    else:
        diff = (arr[-1] - arr[0]) / max((arr.shape[0] - 1) * dt_min, 1e-6)
    # Block-average to size x size.
    return _downsample(diff, size).astype(np.float32)


def _downsample(field: np.ndarray, size: int) -> np.ndarray:
    """Mean-pool a 2D field to (size, size)."""
    h, w = field.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    out = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            block = field[ys[i]:max(ys[i] + 1, ys[i + 1]), xs[j]:max(xs[j] + 1, xs[j + 1])]
            out[i, j] = float(block.mean()) if block.size else 0.0
    return out


def build_asg_pair(
    frames_hist: np.ndarray,
    frames_future: np.ndarray,
    context: dict,
    horizon_min: int,
    dx_km: float,
    dt_min: float,
    threshold: float,
) -> ASGSequence:
    """Build an (ASG_t, ASG_{t+h}) pair from history + future frame stacks.

    Args:
        frames_hist: [T, H, W] history ending at t (X_{t-k..t}).
        frames_future: [T2, H, W] future frames at/after t+h (may be empty).
        context: co-located environmental scalars (from labeling.context).
        horizon_min: transition horizon h in minutes.
        dx_km: km per grid pixel.
        dt_min: minutes per frame step.
        threshold: VIL segmentation threshold (in input units, pre-dBZ map).

    Returns:
        ASGSequence(asg_t, asg_th, horizon_min). The shared future-blind motion
        field is stashed on both ASGs' meta['flow']; NL rationales on meta.
    """
    hist = np.asarray(frames_hist, dtype=np.float32)
    if hist.ndim == 2:
        hist = hist[None, ...]
    h, w = hist.shape[1], hist.shape[2]

    # (1) Motion field from history only — the future-blind path source.
    flow = estimate_label_motion(hist)

    # (2) Track cells over the history window.
    tracks_hist = tracking_mod.track_cells(hist, threshold)
    objs_t = _build_objects(tracks_hist, flow, dx_km, dt_min, use_track_end=True)

    g_field_t = _growth_field(hist, _growth_field_size(context), dt_min)
    global_reg_t = regime_mod.global_regime(tracks_hist, dt_min)

    asg_t = ASG(
        objects=objs_t,
        global_regime=global_reg_t,
        growth_field=g_field_t,
        context={k: float(v) for k, v in context.items()},
        meta={},
    ).capped(N_MAX)

    # (3) ASG_{t+h}: track the future stack if available, else advect ASG_t.
    fut = np.asarray(frames_future, dtype=np.float32) if frames_future is not None else None
    if fut is not None and fut.ndim == 2:
        fut = fut[None, ...]

    if fut is not None and fut.shape[0] >= 1 and fut.size > 0:
        flow_fut = motion_mod.estimate_motion(fut) if fut.shape[0] >= 2 else flow
        tracks_fut = tracking_mod.track_cells(fut, threshold)
        objs_th = _build_objects(tracks_fut, flow_fut, dx_km, dt_min, use_track_end=False)
        objs_th = _match_ids(objs_t, objs_th)
        g_field_th = _growth_field(fut, _growth_field_size(context), dt_min)
        global_reg_th = regime_mod.global_regime(tracks_fut, dt_min)
    else:
        objs_th = _advect_objects(objs_t, horizon_min, dx_km)
        g_field_th = g_field_t
        global_reg_th = global_reg_t

    asg_th = ASG(
        objects=objs_th,
        global_regime=global_reg_th,
        growth_field=g_field_th,
        context={k: float(v) for k, v in context.items()},
        meta={},
    ).capped(N_MAX)

    # (4) Constrained NL renders (datasource.md section 5).
    asg_t.meta["nl"] = render_NL(asg_t)
    asg_th.meta["nl"] = render_NL(asg_th)
    delta_nl = render_NL_delta(asg_t, asg_th)
    asg_th.meta["nl_delta"] = delta_nl

    # (5) Stash the shared future-blind flow on both ASGs (px/step). Stored as a
    # list for JSON-serializability via ASG.to_dict.
    flow_list = flow.astype(np.float32).tolist()
    asg_t.meta["flow"] = flow_list
    asg_t.meta["grid"] = [h, w]
    asg_t.meta["dx_km"] = float(dx_km)
    asg_t.meta["dt_min"] = float(dt_min)
    asg_th.meta["flow"] = flow_list
    asg_th.meta["grid"] = [h, w]
    asg_th.meta["dx_km"] = float(dx_km)
    asg_th.meta["dt_min"] = float(dt_min)

    return ASGSequence(asg_t=asg_t, asg_th=asg_th, horizon_min=int(horizon_min))


def _growth_field_size(context: dict) -> int:
    """Growth-field side length; default mirrors cfg.asg.growth_field_size=48."""
    return int(context.get("_growth_field_size", 48)) if isinstance(context, dict) else 48


def _match_ids(objs_t: List[StormObject], objs_th: List[StormObject]) -> List[StormObject]:
    """Assign each future object the id of the nearest past object (greedy).

    Preserves object identity across the (t, t+h) pair so render_NL_delta can match
    cells. Unmatched future objects (initiations) get fresh ids above the max.
    """
    if not objs_t:
        return objs_th
    used = [False] * len(objs_t)
    next_id = max((o.id for o in objs_t), default=-1) + 1
    matched: List[StormObject] = []
    # Greedy by intensity so the strongest future cells claim first.
    for o2 in sorted(objs_th, key=lambda o: o.peak, reverse=True):
        best_i = -1
        best_d = 1e18
        for i, o1 in enumerate(objs_t):
            if used[i]:
                continue
            d = (o2.cy - o1.cy) ** 2 + (o2.cx - o1.cx) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0:
            used[best_i] = True
            o2.id = objs_t[best_i].id
        else:
            o2.id = next_id
            next_id += 1
        matched.append(o2)
    return matched


def autolabel_event(event: dict, cfg: Config) -> ASGSequence:
    """Auto-label a single cached event into an ASGSequence (datasource.md section 2).

    Args:
        event: dict with 'vil' (required, [T,H,W]) and optional 'ir069','ir107',
            'glm','lat','lon','time'.
        cfg: project Config (data.*, asg.*).

    Returns:
        ASGSequence(asg_t, asg_th, horizon_min, event_id).
    """
    from .context import colocate_context

    vil = np.asarray(event["vil"], dtype=np.float32)
    if vil.ndim != 3:
        raise ValueError(f"event['vil'] must be [T,H,W]; got {vil.shape}")
    t = vil.shape[0]

    in_frames = int(cfg.get_path("data.in_frames", 13))
    dx_km = float(cfg.get_path("data.km_per_pixel", 1.0))
    dt_min = float(cfg.get_path("data.minutes_per_frame", 5))
    horizon_min = int(cfg.get_path("data.horizon_min", 60))
    gfs = int(cfg.get_path("asg.growth_field_size", 48))

    # Segmentation threshold: a fraction of the dynamic range, in input VIL units.
    threshold = _auto_threshold(vil, cfg)

    # Split into history (<= t) and future (>= t+h). Horizon in frames.
    horizon_frames = max(1, int(round(horizon_min / dt_min)))
    split = min(in_frames, t)
    hist = vil[:split]
    fut_start = min(split + horizon_frames - 1, t - 1)
    frames_future = vil[fut_start: fut_start + 1] if fut_start < t else vil[-1:]
    # If a longer future window exists, pass a couple of frames so motion can be
    # re-estimated on the future stack.
    if fut_start + 1 < t:
        frames_future = vil[fut_start: min(fut_start + 3, t)]

    # Context co-location (best-effort; zeros + context_available=0 if absent).
    lat = event.get("lat")
    lon = event.get("lon")
    time_iso = event.get("time")
    context = colocate_context(lat, lon, time_iso, cfg)
    context["_growth_field_size"] = gfs  # threaded into _growth_field_size()

    seq = build_asg_pair(
        frames_hist=hist,
        frames_future=frames_future,
        context=context,
        horizon_min=horizon_min,
        dx_km=dx_km,
        dt_min=dt_min,
        threshold=threshold,
    )
    # Drop the private hint from the persisted context so JSON stays clean.
    for asg in (seq.asg_t, seq.asg_th):
        asg.context.pop("_growth_field_size", None)

    event_id = str(event.get("id", event.get("event_id", "")))
    seq.event_id = event_id
    seq.asg_t.meta["event_id"] = event_id
    seq.asg_th.meta["event_id"] = event_id
    if time_iso is not None:
        seq.asg_t.meta["time"] = str(time_iso)
        seq.asg_th.meta["time"] = str(time_iso)
    return seq


def _auto_threshold(vil: np.ndarray, cfg: Config) -> float:
    """Pick a segmentation threshold in input VIL units.

    Uses a high percentile of nonzero pixels so only organized cells are kept,
    bounded below by a small floor to avoid segmenting clear sky.
    """
    finite = vil[np.isfinite(vil)]
    if finite.size == 0:
        return 1.0
    nz = finite[finite > 0]
    if nz.size == 0:
        return 1.0
    # ~70th percentile of rainy pixels; floor at a small absolute value.
    thr = float(np.percentile(nz, 70.0))
    vmax = float(finite.max())
    floor = 0.05 * vmax if vmax > 0 else 1.0
    return max(thr, floor)
