"""Physical-admissibility certificates for ASG transitions (prototype).

The checker verifies that a predicted ASG_{t+h} is a *physically admissible* evolution of
ASG_t. Constraints live on the low-dimensional, typed ASG state (object attributes + a
regime automaton) -- exactly the part a solver can reason about -- never on the pixel
field. Two products:

  * ``certify_transition`` -> ``Certificate`` with ``ok`` and, on failure, the
    violated-constraint *core* (the ASG-WM analogue of a Z3 UNSAT core).
  * ``ambiguity_flag`` -> a dual-SAT verdict (initiation admissible? non-initiation
    admissible? both => genuinely ambiguous), using Z3 over the context envelope when
    available, with a pure-interval fallback otherwise.

All thresholds are gathered in ``ConstraintBounds`` so the bank can be *audited for
non-vacuousness* (loose bounds => a certificate that certifies nothing -- the same trap as
the IB-capacity audit). Depends only on numpy + asgwm.asg; z3 is optional.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple

from asgwm.asg import ASG, StormObject, REGIMES

try:
    import z3  # type: ignore
    _HAS_Z3 = True
except Exception:  # pragma: no cover
    _HAS_Z3 = False


# ---------------------------------------------------------------------------
# Regime transition automaton (architecture.md section 2: init/grow/decay/steady)
# ---------------------------------------------------------------------------
REGIME_FSM: Dict[str, Set[str]] = {
    "init":   {"init", "grow", "steady"},          # a new cell may grow or settle, not vanish-decay
    "grow":   {"grow", "steady", "decay"},
    "steady": {"steady", "grow", "decay"},
    "decay":  {"decay", "steady"},                 # a decaying cell does not re-initiate / re-grow abruptly
}


@dataclass
class ConstraintBounds:
    """Physically-motivated bounds (per hour where rate-like). Tune + audit these."""

    v_max_kmh: float = 150.0           # max storm-motion speed
    advection_residual_km: float = 35.0  # how far the cell may deviate from pure advection
    max_dpeak_dbz_per_h: float = 35.0  # max intensity change rate
    area_grow_per_h: float = 6.0       # max area multiplier per hour
    area_shrink_per_h: float = 0.15    # min area multiplier per hour
    peak_min_dbz: float = 0.0
    peak_max_dbz: float = 80.0
    tendency_tol_dbz: float = 3.0      # slack for regime/tendency consistency
    dissipation_peak_dbz: float = 22.0  # a cell may only vanish if it was weak or decaying
    # instability gating (only applied when context is available)
    cape_grow_min: float = 300.0       # intensifying convection needs at least this CAPE
    cape_init_min: float = 800.0       # initiation needs this much instability
    cin_init_max: float = 50.0         # ...and little inhibition
    cape_envelope_frac: float = 0.25   # +/- uncertainty band on context for the dual-SAT check


@dataclass
class Certificate:
    """Result of certifying one transition."""

    ok: bool
    violations: List[Dict] = field(default_factory=list)  # each: {name, object_id, detail}
    n_objects_checked: int = 0
    n_constraints: int = 0
    horizon_min: int = 0

    @property
    def core(self) -> List[str]:
        """The violated-constraint core (sorted, de-duplicated)."""
        return sorted({v["name"] for v in self.violations})

    def diagnosis(self) -> str:
        if self.ok:
            return f"ADMISSIBLE: {self.n_objects_checked} objects satisfied {self.n_constraints} constraints."
        lines = ["INADMISSIBLE. Violated-constraint core: " + ", ".join(self.core)]
        for v in self.violations[:12]:
            oid = v.get("object_id")
            tag = f"obj {oid}" if oid is not None else "set"
            lines.append(f"  - [{v['name']}] ({tag}) {v['detail']}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return asdict(self)


def _dist_px(a: StormObject, b: StormObject) -> float:
    return math.hypot(a.cy - b.cy, a.cx - b.cx)


def certify_transition(
    asg_t: ASG,
    asg_th: ASG,
    horizon_min: int,
    dx_km: float = 1.0,
    bounds: Optional[ConstraintBounds] = None,
) -> Certificate:
    """Certify that ``asg_th`` is a physically-admissible evolution of ``asg_t``.

    Objects are matched by ``id``. Returns a :class:`Certificate`; on failure its
    ``core`` names the violated physical assumptions (the UNSAT-core analogue).
    """
    b = bounds or ConstraintBounds()
    dt_h = max(horizon_min, 1) / 60.0
    ctx = asg_t.context or {}
    ctx_avail = float(ctx.get("context_available", 0.0)) > 0.5
    cape = float(ctx.get("cape", 0.0))

    viol: List[Dict] = []
    n_constraints = 0

    by_t = {o.id: o for o in asg_t.objects}
    by_th = {o.id: o for o in asg_th.objects}

    def add(name: str, oid, detail: str):
        viol.append({"name": name, "object_id": oid, "detail": detail})

    # ---- set-level constraints ------------------------------------------
    n_constraints += 1
    if asg_th.n_objects > 16:
        add("object_budget", None, f"{asg_th.n_objects} > N_max=16")

    # new objects must be 'init'
    for oid, o in by_th.items():
        if oid not in by_t:
            n_constraints += 1
            if o.regime != "init":
                add("new_object_must_init", oid, f"appeared with regime={o.regime}, expected init")

    # vanished objects must have been weak or decaying
    for oid, o in by_t.items():
        if oid not in by_th:
            n_constraints += 1
            if not (o.regime == "decay" or o.peak < b.dissipation_peak_dbz):
                add("implausible_dissipation", oid,
                    f"vanished from regime={o.regime}, peak={o.peak:.1f} (>= {b.dissipation_peak_dbz})")

    # ---- per-matched-object constraints ---------------------------------
    matched = 0
    for oid in by_t.keys() & by_th.keys():
        ot, oth = by_t[oid], by_th[oid]
        matched += 1

        # 1) kinematic speed bound
        n_constraints += 1
        disp_km = _dist_px(ot, oth) * dx_km
        if disp_km > b.v_max_kmh * dt_h + b.advection_residual_km:
            add("kinematic_speed", oid,
                f"displaced {disp_km:.0f} km in {horizon_min} min (> v_max {b.v_max_kmh} km/h)")

        # 2) advection-residual bound (displacement vs the cell's own motion)
        n_constraints += 1
        adv_dy = ot.vy * dt_h / dx_km   # km/h -> px over horizon
        adv_dx = ot.vx * dt_h / dx_km
        res_km = math.hypot((oth.cy - ot.cy) - adv_dy, (oth.cx - ot.cx) - adv_dx) * dx_km
        if res_km > b.advection_residual_km:
            add("advection_residual", oid,
                f"deviates {res_km:.0f} km from its advected position (> {b.advection_residual_km} km)")

        # 3) intensity-rate bound
        n_constraints += 1
        if abs(oth.peak - ot.peak) > b.max_dpeak_dbz_per_h * dt_h:
            add("intensity_rate", oid,
                f"peak changed {oth.peak - ot.peak:+.1f} dBZ in {horizon_min} min "
                f"(> {b.max_dpeak_dbz_per_h} dBZ/h)")

        # 4) area continuity
        n_constraints += 1
        lo = ot.area * (b.area_shrink_per_h ** dt_h)
        hi = ot.area * (b.area_grow_per_h ** dt_h)
        if not (lo - 1.0 <= oth.area <= hi + 1.0):
            add("area_continuity", oid,
                f"area {ot.area:.0f}->{oth.area:.0f} km^2 outside admissible [{lo:.0f},{hi:.0f}]")

        # 5) regime/tendency consistency
        n_constraints += 1
        if oth.regime == "grow" and oth.peak < ot.peak - b.tendency_tol_dbz:
            add("tendency_consistency", oid, "labelled grow but peak decreased")
        if oth.regime == "decay" and oth.peak > ot.peak + b.tendency_tol_dbz:
            add("tendency_consistency", oid, "labelled decay but peak increased")

        # 6) regime automaton
        n_constraints += 1
        allowed = REGIME_FSM.get(ot.regime, set(REGIMES))
        if oth.regime not in allowed:
            add("regime_transition", oid, f"{ot.regime}->{oth.regime} not in FSM {sorted(allowed)}")

        # 7) range bounds
        n_constraints += 1
        if not (b.peak_min_dbz <= oth.peak <= b.peak_max_dbz) or oth.area < 0 or not (0.0 <= oth.conf <= 1.0):
            add("range_bounds", oid, f"peak/area/conf out of range (peak={oth.peak:.1f}, area={oth.area:.0f})")

        # 8) instability gating for intensification (only if context known)
        if ctx_avail and oth.regime == "grow" and (oth.peak - ot.peak) > b.tendency_tol_dbz:
            n_constraints += 1
            if cape < b.cape_grow_min:
                add("instability_gating", oid,
                    f"intensifying with CAPE={cape:.0f} J/kg (< {b.cape_grow_min}); no instability source")

    return Certificate(
        ok=(len(viol) == 0),
        violations=viol,
        n_objects_checked=matched,
        n_constraints=n_constraints,
        horizon_min=horizon_min,
    )


# ---------------------------------------------------------------------------
# Dual-SAT ambiguity: is initiation admissible? is non-initiation admissible?
# (the No/Yes/Uncertain pattern, ported to convective initiation)
# ---------------------------------------------------------------------------
def _sat_initiation(cape_lo, cape_hi, cin_lo, cin_hi, b: ConstraintBounds) -> bool:
    """Exists (cape,cin) in the envelope supporting initiation (instability, low inhibition)?"""
    if _HAS_Z3:
        s = z3.Solver()
        cape, cin = z3.Reals("cape cin")
        s.add(cape >= cape_lo, cape <= cape_hi, cin >= cin_lo, cin <= cin_hi)
        s.add(cape >= b.cape_init_min, cin <= b.cin_init_max)
        return s.check() == z3.sat
    return (cape_hi >= b.cape_init_min) and (cin_lo <= b.cin_init_max)


def _sat_no_initiation(cape_lo, cape_hi, cin_lo, cin_hi, b: ConstraintBounds) -> bool:
    """Exists (cape,cin) in the envelope where initiation is NOT supported?"""
    if _HAS_Z3:
        s = z3.Solver()
        cape, cin = z3.Reals("cape cin")
        s.add(cape >= cape_lo, cape <= cape_hi, cin >= cin_lo, cin <= cin_hi)
        s.add(z3.Or(cape < b.cape_init_min, cin > b.cin_init_max))
        return s.check() == z3.sat
    return (cape_lo < b.cape_init_min) or (cin_hi > b.cin_init_max)


def ambiguity_flag(context: Dict[str, float], bounds: Optional[ConstraintBounds] = None) -> Dict:
    """Dual-SAT verdict on convective initiation over the context envelope.

    Returns dict(initiation_admissible, no_initiation_admissible, ambiguous, confident_label).
    ``ambiguous`` True => the observations under-constrain the future and the model should
    emit an ensemble + ambiguity flag rather than a confident point forecast.
    """
    b = bounds or ConstraintBounds()
    cape = float(context.get("cape", 0.0))
    cin = float(context.get("cin", 0.0))
    f = b.cape_envelope_frac
    cape_lo, cape_hi = cape * (1 - f), cape * (1 + f)
    cin_lo, cin_hi = cin * (1 - f), cin * (1 + f) + 1e-6
    init_ok = _sat_initiation(cape_lo, cape_hi, cin_lo, cin_hi, b)
    no_init_ok = _sat_no_initiation(cape_lo, cape_hi, cin_lo, cin_hi, b)
    ambiguous = bool(init_ok and no_init_ok)
    if ambiguous:
        label = "uncertain"
    elif init_ok:
        label = "initiation-likely"
    else:
        label = "initiation-unlikely"
    return {
        "initiation_admissible": bool(init_ok),
        "no_initiation_admissible": bool(no_init_ok),
        "ambiguous": ambiguous,
        "confident_label": label,
        "solver": "z3" if _HAS_Z3 else "interval",
    }


def admissible_regimes(o: StormObject) -> Set[str]:
    """The set of next-step regimes admissible from this object's current regime (FSM)."""
    return set(REGIME_FSM.get(o.regime, set(REGIMES)))
