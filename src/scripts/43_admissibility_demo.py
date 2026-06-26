"""Prototype demo + non-vacuousness audit for the symbolic admissibility layer.

Demonstrates three things the idea must satisfy:
  1. CATCHES impossible transitions (teleport, forbidden regime jump, super-fast
     intensification, grow-but-weaken) -- with the right violated-constraint core.
  2. Is NOT VACUOUS: plausible transitions pass; impossible ones fail. Reports the
     plausible-pass-rate and impossible-catch-rate over random perturbations.
  3. Dual-SAT AMBIGUITY: high-CAPE/low-CIN -> confident initiation; moderate -> ambiguous.

Run:  python scripts/43_admissibility_demo.py [--config ../configs/default.yaml --override paths.root=...]
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import glob
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from asgwm.asg import ASG, StormObject
from asgwm.utils.config import load_config
from asgwm.symbolic import certify_transition, ambiguity_flag, ConstraintBounds


def _load_pairs(cfg):
    asg_dir = os.path.join(cfg.get_path("paths.cache", "./artifacts/cache"), "asg")
    pairs = []
    for fp in sorted(glob.glob(os.path.join(asg_dir, "*.json"))):
        with open(fp, "r", encoding="utf-8") as f:
            p = json.load(f)
        if "asg_t" in p and "asg_th" in p:
            pairs.append((ASG.from_dict(p["asg_t"]), ASG.from_dict(p["asg_th"]),
                          int(p.get("horizon_min", 60))))
    return pairs


def _synthetic_pair():
    """A clean, physically-plausible transition built from scratch (no cache needed)."""
    t = ASG(objects=[
        StormObject(1, 120, 100, 240, 36, -10, 8, "grow", 1.5, 0.9),
        StormObject(2, 60, 200, 90, 20, 5, -3, "steady", 0.0, 0.7),
    ], global_regime="grow", context={"cape": 1800.0, "cin": 20.0, "context_available": 1.0})
    # advance by ~1 step of the motion, modest growth
    th = ASG(objects=[
        StormObject(1, 110, 108, 300, 40, -10, 8, "grow", 1.2, 0.85),
        StormObject(2, 65, 197, 85, 19, 5, -3, "steady", 0.0, 0.7),
    ], global_regime="grow", context=t.context)
    return t, th, 30


def _mutate(asg: ASG, oid: int, **changes) -> ASG:
    new = copy.deepcopy(asg)
    for o in new.objects:
        if o.id == oid:
            for k, v in changes.items():
                setattr(o, k, v)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)
    dx = float(cfg.get_path("data.km_per_pixel", 1.0))
    bounds = ConstraintBounds()
    rng = random.Random(int(cfg.get_path("seed", 1234)))

    print("=" * 72)
    print("ASG-WM symbolic admissibility — prototype demo")
    print("solver:", ambiguity_flag({"cape": 0, "cin": 0})["solver"])
    print("=" * 72)

    # --- gather plausible transitions (cache if present, else one synthetic) ---
    pairs = _load_pairs(cfg)
    if not pairs:
        pairs = [_synthetic_pair()]
        print(f"[info] no cached ASG pairs found; using 1 synthetic plausible pair")
    else:
        print(f"[info] loaded {len(pairs)} cached transitions from cache")
    # always include the clean synthetic pair as a known-good control
    clean_t, clean_th, clean_h = _synthetic_pair()

    # --- 1) certify plausible transitions ---
    print("\n--- (1) certify real/plausible transitions ---")
    ok_count = 0
    for i, (t, th, h) in enumerate(pairs):
        cert = certify_transition(t, th, h, dx_km=dx, bounds=bounds)
        ok_count += int(cert.ok)
        if i < 3:
            print(f"  pair {i}: {'ADMISSIBLE' if cert.ok else 'INADMISSIBLE'} "
                  f"({cert.n_objects_checked} obj, {cert.n_constraints} checks)"
                  + ("" if cert.ok else f"  core={cert.core}"))
    cclean = certify_transition(clean_t, clean_th, clean_h, dx_km=dx, bounds=bounds)
    print(f"  clean control: {'ADMISSIBLE' if cclean.ok else 'INADMISSIBLE'}  (expect ADMISSIBLE)")
    print(f"  => {ok_count}/{len(pairs)} cached transitions admissible")

    # --- 2) catch injected-impossible transitions ---
    print("\n--- (2) catch injected-impossible transitions (expect INADMISSIBLE + correct core) ---")
    cases = {
        "teleport_500px": _mutate(clean_th, 1, cy=clean_th.objects[0].cy + 500),
        "forbidden_regime (grow->? after init)":
            _mutate(_mutate(clean_t, 1, regime="init"), 1),  # placeholder, replaced below
        "intensity_jump_+60dBZ": _mutate(clean_th, 1, peak=clean_th.objects[0].peak + 60),
        "grow_but_weaken": _mutate(clean_th, 1, regime="grow", peak=clean_th.objects[0].peak - 25),
    }
    # build a proper forbidden-regime case: t.obj1 = decay, th.obj1 = init  (decay->init forbidden)
    t_decay = _mutate(clean_t, 1, regime="decay")
    th_init = _mutate(clean_th, 1, regime="init")
    cases["forbidden_regime decay->init"] = th_init
    del cases["forbidden_regime (grow->? after init)"]

    for name, bad_th in cases.items():
        base_t = t_decay if "decay->init" in name else clean_t
        cert = certify_transition(base_t, bad_th, clean_h, dx_km=dx, bounds=bounds)
        status = "ADMISSIBLE (!! missed)" if cert.ok else "INADMISSIBLE (caught)"
        print(f"  {name:34s}: {status}  core={cert.core}")

    # --- 3) non-vacuousness audit ---
    print("\n--- (3) non-vacuousness audit (plausible should pass; impossible should fail) ---")
    N = 200
    plaus_pass = 0
    for _ in range(N):
        th2 = copy.deepcopy(clean_th)
        for o in th2.objects:  # tiny, physically-plausible jitter
            o.cy += rng.uniform(-2, 2); o.cx += rng.uniform(-2, 2)
            o.peak += rng.uniform(-1.5, 1.5); o.area *= rng.uniform(0.9, 1.1)
        plaus_pass += int(certify_transition(clean_t, th2, clean_h, dx_km=dx, bounds=bounds).ok)
    imposs_caught = 0
    for _ in range(N):
        th2 = copy.deepcopy(clean_th)
        o = th2.objects[0]
        kind = rng.choice(["teleport", "speed", "intensity", "regime", "tendency"])
        if kind == "teleport":
            o.cy += rng.uniform(200, 600)
        elif kind == "speed":
            o.cx += rng.uniform(150, 400)
        elif kind == "intensity":
            o.peak = min(80, o.peak + rng.uniform(40, 60))
        elif kind == "regime":
            o.regime = "init"  # decay/grow/steady -> init is forbidden from grow base
            base = _mutate(clean_t, 1, regime="decay")
            imposs_caught += int(not certify_transition(base, th2, clean_h, dx_km=dx, bounds=bounds).ok)
            continue
        else:  # tendency
            o.regime = "grow"; o.peak = clean_th.objects[0].peak - rng.uniform(20, 40)
        imposs_caught += int(not certify_transition(clean_t, th2, clean_h, dx_km=dx, bounds=bounds).ok)
    print(f"  plausible-pass rate : {plaus_pass}/{N} = {plaus_pass/N:.0%}   (want ~100% -> not over-rigid)")
    print(f"  impossible-catch rate: {imposs_caught}/{N} = {imposs_caught/N:.0%}   (want ~100% -> not vacuous)")

    # --- 4) dual-SAT ambiguity over context regimes ---
    print("\n--- (4) dual-SAT ambiguity (No/Yes/Uncertain analogue for initiation) ---")
    regimes = {
        "high CAPE / low CIN  (should be confident-initiation)": {"cape": 2800, "cin": 10},
        "moderate CAPE / CIN  (should be UNCERTAIN)":            {"cape": 900,  "cin": 45},
        "low CAPE / high CIN  (should be confident-no-init)":    {"cape": 200,  "cin": 120},
    }
    for desc, ctx in regimes.items():
        v = ambiguity_flag(ctx, bounds)
        print(f"  {desc:54s}: {v['confident_label']:18s} "
              f"(init={v['initiation_admissible']}, no-init={v['no_initiation_admissible']}, "
              f"ambiguous={v['ambiguous']})")

    print("\nDone.")


if __name__ == "__main__":
    main()
