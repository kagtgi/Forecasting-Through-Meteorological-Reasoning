"""Stage A2 — Explicit Object Tracker (Track step in the five-step reasoning flow).

In the five-step framework (Observe → Identify → Track → Analyze → Nowcast) this
module is the Track step:
  Stage A (Identify): VLM per-frame → per-frame object set {O_{t'}}
  Stage A2 (Track):   links {O_{t'}} across history → trajectory-enriched ASG_t

Separation principle: Stage A answers *what is here now?* (spatial perception,
single frame); Stage A2 answers *where did this come from and how fast is it
moving?* (temporal association, deterministic, no learned parameters).

The tracker output is trajectory-enriched:
  - stable object IDs across the k-frame window
  - per-object velocity from multi-frame centroid displacement (more reliable
    than single-frame centroid differences, especially for slow-moving cells)
  - track age that distinguishes true initiations from advecting cells
  - trajectory-length confidence that blends with the VLM's per-attr conf

Usage (inference):
    from asgwm.models.stage_a_vlm  import StageAVLM
    from asgwm.models.stage_a2_tracker import ObjectTracker

    per_frame = vlm.generate_per_frame(images, context)   # List[List[StormObject]]
    tracker   = ObjectTracker.from_config(cfg)
    asg_t     = tracker.track(per_frame, flow=flow_np, context=context, n_max=16)
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from asgwm.asg import ASG, StormObject, N_MAX


class ObjectTracker:
    """Links per-frame VLM identifications into trajectories to produce ASG_t.

    Greedy nearest-centroid association within a gating radius, identical in
    spirit to the auto-labeling pipeline's tracking logic (labeling/tracking.py)
    but operating on already-identified StormObject instances rather than raw
    pixel frames.
    """

    def __init__(
        self,
        gate_radius: float = 32.0,
        min_track_frames: int = 1,
        dx_km: float = 1.0,
        dt_min: float = 5.0,
    ) -> None:
        self.gate_radius = float(gate_radius)
        self.min_track_frames = int(min_track_frames)
        self.dx_km = float(dx_km)
        self.dt_min = float(dt_min)

    @classmethod
    def from_config(cls, cfg=None) -> "ObjectTracker":
        gate = 32.0
        min_frames = 1
        dx_km = 1.0
        dt_min = 5.0
        if cfg is not None:
            gate = float(cfg.get_path("stage_a2.gate_radius", gate))
            min_frames = int(cfg.get_path("stage_a2.min_track_frames", min_frames))
            dx_km = float(cfg.get_path("data.km_per_pixel", dx_km))
            dt_min = float(cfg.get_path("data.minutes_per_frame", dt_min))
        return cls(gate_radius=gate, min_track_frames=min_frames,
                   dx_km=dx_km, dt_min=dt_min)

    def track(
        self,
        per_frame_objects: List[List[StormObject]],
        flow: Optional[np.ndarray] = None,
        context: Optional[Dict] = None,
        n_max: int = N_MAX,
    ) -> ASG:
        """Link per-frame object sets into trajectories and build ASG_t.

        Args:
            per_frame_objects: Outer list = frames t-k..t (oldest first).
                               Inner list = Stage-A identifications for that frame.
            flow:    [2, H, W] pixel-per-step motion field for velocity fallback
                     when a track has only one frame. None → zero fallback.
            context: environmental scalars to attach to the ASG.
            n_max:   hard object cap.

        Returns:
            Trajectory-enriched ASG at time t (last frame in per_frame_objects).
        """
        context = dict(context or {})

        if not per_frame_objects or all(len(f) == 0 for f in per_frame_objects):
            return ASG(objects=[], global_regime="steady",
                       growth_field=None, context=context,
                       meta={"source": "ObjectTracker.empty"})

        flow_h = flow.shape[1] if flow is not None else None
        flow_w = flow.shape[2] if flow is not None else None

        # Greedy nearest-centroid tracking across frames.
        finalized: List[Dict] = []
        active: List[Dict] = []
        next_id = 0

        for ti, frame_objs in enumerate(per_frame_objects):
            used = [False] * len(frame_objs)

            ordered = sorted(
                range(len(active)),
                key=lambda i: active[i]["frames"][-1]["peak"],
                reverse=True,
            )
            still_active: List[Dict] = []
            for ai in ordered:
                trk = active[ai]
                last = trk["frames"][-1]
                best_j, best_d = -1, self.gate_radius
                for j, o in enumerate(frame_objs):
                    if used[j]:
                        continue
                    d = float(np.hypot(o.cy - last["cy"], o.cx - last["cx"]))
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    used[best_j] = True
                    o = frame_objs[best_j]
                    trk["frames"].append({
                        "t": ti, "cy": o.cy, "cx": o.cx,
                        "area": o.area, "peak": o.peak,
                        "regime": o.regime, "growth": o.growth, "conf": o.conf,
                    })
                    still_active.append(trk)
                else:
                    finalized.append(trk)
            active = still_active

            for j, o in enumerate(frame_objs):
                if used[j]:
                    continue
                active.append({
                    "id": next_id,
                    "frames": [{"t": ti, "cy": o.cy, "cx": o.cx, "area": o.area,
                                "peak": o.peak, "regime": o.regime,
                                "growth": o.growth, "conf": o.conf}],
                    "obj_ref": o,
                })
                next_id += 1

        finalized.extend(active)
        finalized = [tr for tr in finalized
                     if len(tr["frames"]) >= self.min_track_frames]

        # Build StormObjects from the current-frame (last-frame) state of each track.
        last_ti = len(per_frame_objects) - 1
        objs: List[StormObject] = []

        for tr in sorted(finalized,
                         key=lambda t: max(f["peak"] for f in t["frames"]),
                         reverse=True):
            last_frame = [f for f in tr["frames"] if f["t"] == last_ti]
            if not last_frame:
                continue  # object dissipated before the last frame
            lf = last_frame[0]

            # Multi-frame velocity in px/step; convert to km/h.
            vy_px, vx_px = _track_velocity(tr["frames"])
            if len(tr["frames"]) == 1:
                # Single-frame track: fall back to optical flow, then VLM estimate.
                if flow is not None and flow_h is not None:
                    iy = int(np.clip(round(lf["cy"]), 0, flow_h - 1))
                    ix = int(np.clip(round(lf["cx"]), 0, flow_w - 1))
                    vy_px = float(flow[0, iy, ix])
                    vx_px = float(flow[1, iy, ix])
                else:
                    ref: Optional[StormObject] = tr.get("obj_ref")
                    if ref is not None:
                        # VLM already expressed velocity in km/h; skip unit conversion.
                        vy_kmh, vx_kmh = ref.vy, ref.vx
                        objs.append(StormObject(
                            id=int(tr["id"]),
                            cy=float(lf["cy"]), cx=float(lf["cx"]),
                            area=float(lf["area"]), peak=float(lf["peak"]),
                            vy=vy_kmh, vx=vx_kmh,
                            regime=_majority_regime([f["regime"] for f in tr["frames"]]),
                            growth=float(np.mean([f["growth"] for f in tr["frames"]])),
                            conf=float(lf["conf"]),
                        ))
                        continue

            vy_kmh = _px_per_h_to_kmh(vy_px, self.dx_km, self.dt_min)
            vx_kmh = _px_per_h_to_kmh(vx_px, self.dx_km, self.dt_min)

            track_age = len(tr["frames"])
            track_conf = float(np.clip(0.4 + 0.06 * track_age, 0.3, 1.0))
            base_conf = float(lf["conf"])
            conf = float(0.5 * base_conf + 0.5 * track_conf)

            objs.append(StormObject(
                id=int(tr["id"]),
                cy=float(lf["cy"]), cx=float(lf["cx"]),
                area=float(lf["area"]), peak=float(lf["peak"]),
                vy=vy_kmh, vx=vx_kmh,
                regime=_majority_regime([f["regime"] for f in tr["frames"]]),
                growth=float(np.mean([f["growth"] for f in tr["frames"]])),
                conf=conf,
            ))

        if not objs:
            return ASG(objects=[], global_regime="steady",
                       growth_field=None, context=context,
                       meta={"source": "ObjectTracker.no_surviving_tracks"})

        global_reg = _global_regime(objs)
        asg = ASG(
            objects=objs,
            global_regime=global_reg,
            growth_field=None,
            context=context,
            meta={"source": "ObjectTracker"},
        )
        return asg.capped(n_max)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track_velocity(frames: List[Dict]) -> Tuple[float, float]:
    """Mean per-step centroid velocity (vy, vx) in px/step over a track."""
    if len(frames) < 2:
        return 0.0, 0.0
    dys, dxs = [], []
    for a, b in zip(frames[:-1], frames[1:]):
        span = max(1, b["t"] - a["t"])
        dys.append((b["cy"] - a["cy"]) / span)
        dxs.append((b["cx"] - a["cx"]) / span)
    return float(np.mean(dys)), float(np.mean(dxs))


def _px_per_h_to_kmh(v_px_per_step: float, dx_km: float, dt_min: float) -> float:
    return v_px_per_step * dx_km * (60.0 / max(dt_min, 1e-6))


def _majority_regime(regimes: List[str]) -> str:
    counts: Dict[str, int] = {}
    for r in regimes:
        counts[r] = counts.get(r, 0) + 1
    return max(counts, key=counts.get) if counts else "steady"


def _global_regime(objs: List[StormObject]) -> str:
    if not objs:
        return "steady"
    counts: Dict[str, int] = {}
    for o in objs:
        counts[o.regime] = counts.get(o.regime, 0) + 1
    for r in ("grow", "init", "decay"):
        if counts.get(r, 0) >= max(1, len(objs) // 2):
            return r
    return max(counts, key=counts.get)
