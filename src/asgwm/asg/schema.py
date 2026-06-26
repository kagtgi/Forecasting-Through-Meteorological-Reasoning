"""Atmospheric Scene Graph (ASG) — the explicit, human-readable world-model state.

This module is the single source of truth for the ASG data contract used by every
stage of ASG-WM (perception -> transition -> renderer) and by the evaluation suite.
It depends only on the standard library + numpy so it is importable without torch.

Reference: architecture.md sections 2 and 9 (grammar / anti-hallucination contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple
import math

import numpy as np

# ---------------------------------------------------------------------------
# Fixed vocabulary / capacity constants (training_method.md section 4: hard cap)
# ---------------------------------------------------------------------------
REGIMES: Tuple[str, ...] = ("init", "grow", "decay", "steady")
REGIME_TO_IDX: Dict[str, int] = {r: i for i, r in enumerate(REGIMES)}
IDX_TO_REGIME: Dict[int, str] = {i: r for i, r in enumerate(REGIMES)}

N_MAX: int = 16                 # maximum storm objects per ASG (hard capacity cap)
MOTION_QUANT_KMH: float = 8.0   # motion-vector quantization bin (km/h)
GROWTH_SIGFIGS: int = 2         # significant figures kept on the growth scalar

# Intensity classes for the NL render (coarse ranges only — no raw numbers leak).
INTENSITY_BINS = (("light", 0.0, 20.0), ("moderate", 20.0, 40.0), ("heavy", 40.0, 1e9))

# 8-point compass for motion direction in the NL render.
_COMPASS = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]


def _round_sig(x: float, sig: int) -> float:
    if x == 0 or not math.isfinite(x):
        return 0.0
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


def quantize_motion(v: float, bin_kmh: float = MOTION_QUANT_KMH) -> float:
    """Quantize a motion component to the fixed km/h grid (hard IB cap)."""
    return float(round(v / bin_kmh) * bin_kmh)


def intensity_class(peak_dbz: float) -> str:
    for name, lo, hi in INTENSITY_BINS:
        if lo <= peak_dbz < hi:
            return name
    return "light"


def motion_to_compass(vy: float, vx: float) -> str:
    """Map a motion vector to an 8-point compass label.

    Convention: image coords with +x = East, +y = South (rows increase downward),
    so the geographic northward component is -vy.
    """
    if abs(vy) < 1e-6 and abs(vx) < 1e-6:
        return "stationary"
    ang = math.degrees(math.atan2(-vy, vx))  # 0 = E, 90 = N
    idx = int(((ang % 360) + 22.5) // 45) % 8
    return _COMPASS[idx]


@dataclass
class StormObject:
    """A single storm cell in the ASG (architecture.md section 2)."""

    id: int
    cy: float            # centroid row (grid coords, sub-pixel)
    cx: float            # centroid col (grid coords, sub-pixel)
    area: float          # km^2
    peak: float          # peak intensity (VIL -> dBZ)
    vy: float            # motion, rows/h in km/h
    vx: float            # motion, cols/h in km/h
    regime: str          # one of REGIMES
    growth: float        # dVIL/dt tendency scalar
    conf: float = 1.0    # perception confidence in [0, 1]
    # Optional per-attribute uncertainty spreads (probabilistic-ASG hook; paper revision).
    sigma_c: Optional[float] = None   # centroid spread (km)
    sigma_v: Optional[float] = None   # motion spread (km/h)
    sigma_g: Optional[float] = None   # growth spread
    # Radar echo morphology — filled by _region_props, propagated through the tracker.
    mean_dbz: Optional[float] = None
    min_dbz: Optional[float] = None
    aspect_ratio: Optional[float] = None   # major/minor axis ratio (≥ 1)
    orientation: Optional[float] = None    # major axis angle in radians, range [−π/2, π/2]
    solidity: Optional[float] = None       # area / convex-hull area ∈ (0, 1]
    # Convective mode — deterministic rule-based classifier (BRN from CAPE + shear).
    conv_mode: Optional[str] = None        # 'isolated'|'qlcs'|'cluster'|'airmass'
    # Topology events — set by _detect_topology in tracking.py.
    event: Optional[str] = None            # 'split'|'merge'|None
    parent_ids: Optional[List[int]] = None

    def __post_init__(self) -> None:
        if self.regime not in REGIME_TO_IDX:
            raise ValueError(f"invalid regime {self.regime!r}; expected one of {REGIMES}")

    # ---- vector views used by models -------------------------------------
    @property
    def centroid(self) -> Tuple[float, float]:
        return (self.cy, self.cx)

    @property
    def motion(self) -> Tuple[float, float]:
        return (self.vy, self.vx)

    def to_vector(self) -> np.ndarray:
        """Continuous attribute vector (for the transition transformer)."""
        return np.array(
            [self.cy, self.cx, self.area, self.peak, self.vy, self.vx, self.growth, self.conf],
            dtype=np.float32,
        )

    def quantized(self) -> "StormObject":
        """Return a copy with motion / growth quantized to the hard IB cap."""
        return StormObject(
            id=self.id,
            cy=self.cy,
            cx=self.cx,
            area=self.area,
            peak=self.peak,
            vy=quantize_motion(self.vy),
            vx=quantize_motion(self.vx),
            regime=self.regime,
            growth=_round_sig(self.growth, GROWTH_SIGFIGS),
            conf=self.conf,
            sigma_c=self.sigma_c,
            sigma_v=self.sigma_v,
            sigma_g=self.sigma_g,
            mean_dbz=self.mean_dbz,
            min_dbz=self.min_dbz,
            aspect_ratio=self.aspect_ratio,
            orientation=self.orientation,
            solidity=self.solidity,
            conv_mode=self.conv_mode,
            event=self.event,
            parent_ids=list(self.parent_ids) if self.parent_ids is not None else None,
        )


@dataclass
class ASG:
    """An Atmospheric Scene Graph: object set + global state + context.

    `growth_field` is the optional low-resolution growth-decay field G_t (H' x W').
    `context` holds co-located environmental scalars (CAPE/CIN/shear/PWAT, DEM stats).
    """

    objects: List[StormObject] = field(default_factory=list)
    global_regime: str = "steady"
    growth_field: Optional[np.ndarray] = None
    context: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)  # event_id, time, grid shape

    def __post_init__(self) -> None:
        if self.global_regime not in REGIME_TO_IDX:
            raise ValueError(f"invalid global_regime {self.global_regime!r}")

    @property
    def n_objects(self) -> int:
        return len(self.objects)

    def capped(self, n_max: int = N_MAX) -> "ASG":
        """Enforce the hard object budget by keeping the most intense N_MAX cells."""
        objs = sorted(self.objects, key=lambda o: (o.peak, o.area), reverse=True)[:n_max]
        return ASG(
            objects=[o.quantized() for o in objs],
            global_regime=self.global_regime,
            growth_field=self.growth_field,
            context=dict(self.context),
            meta=dict(self.meta),
        )

    def to_dict(self) -> Dict:
        d = asdict(self)
        if self.growth_field is not None:
            d["growth_field"] = np.asarray(self.growth_field).tolist()
        return d

    @staticmethod
    def from_dict(d: Dict) -> "ASG":
        gf = d.get("growth_field")
        objs = [StormObject(**o) for o in d.get("objects", [])]
        return ASG(
            objects=objs,
            global_regime=d.get("global_regime", "steady"),
            growth_field=np.asarray(gf, dtype=np.float32) if gf is not None else None,
            context=dict(d.get("context", {})),
            meta=dict(d.get("meta", {})),
        )


@dataclass
class ASGSequence:
    """An (ASG_t, ASG_{t+h}) pair plus the horizon, the unit of supervision."""

    asg_t: ASG
    asg_th: ASG
    horizon_min: int
    event_id: str = ""
