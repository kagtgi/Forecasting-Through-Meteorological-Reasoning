"""Skill evaluation (eval.md A/B/F) for ASG-WM + the pluggable baseline set.

Runs `harness.evaluate_skill`, which assembles forecasts for ASG-WM (real Stage A->C path
when checkpoints exist, advection fallback otherwise) and for every registered baseline
(pysteps now; RainNet/NowcastNet/LangPrecip/ThoR slot in later -> TBR rows). Writes:

  paths.results/skill_results.json      canonical multi-method schema
  paths.results/tables/skill.tex        main-text Table 2 (TBR cells for unavailable methods)
  paths.results/tables/compute.tex      Table 4 (footprint; TBR until measured)
  paths.results/compute_results.json    footprint results (TBR placeholder)

Usage:
    python scripts/40_eval_skill.py --config ../configs/default.yaml --override eval.n_eval_events=16
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from asgwm.utils.config import load_config  # noqa: E402
from asgwm.utils import viz  # noqa: E402
from asgwm.eval import harness  # noqa: E402
from asgwm import baselines as B  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="ASG-WM skill evaluation (multi-method)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    n_events = int(cfg.get_path("eval.n_eval_events", 16))
    results = harness.evaluate_skill(cfg, n_events=n_events)

    results_dir = cfg.get_path("paths.results", "./artifacts/results")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    viz.save_results_json(results, os.path.join(results_dir, "skill_results.json"))

    with open(os.path.join(tables_dir, "skill.tex"), "w", encoding="utf-8") as f:
        f.write(harness.skill_table_tex(results))

    # computational footprint: TBR placeholder rows now (measure ASG-WM after Tier-2;
    # baseline rows fill when each baseline is implemented).
    compute = {name: {} for name in [harness.ASGWM] + list(B.HEADLINE)}
    viz.save_results_json(compute, os.path.join(results_dir, "compute_results.json"))
    with open(os.path.join(tables_dir, "compute.tex"), "w", encoding="utf-8") as f:
        f.write(harness.compute_table_tex(compute))

    # ---- console summary ----
    print(f"[40_eval_skill] methods evaluated ({n_events} events):")
    thr = results["thresholds_dbz"]
    hi = thr[-1]
    for name, md in results["methods"].items():
        tag = "ok " if md["available"] else "TBR"
        csi = md.get("aggregate", {}).get(f"csi@{hi:g}")
        csi_s = f"{csi:.3f}" if isinstance(csi, float) and csi == csi else "  -- "
        print(f"  [{tag}] {md['display']:18s} family={md['family']:22s} CSI@{hi:g}={csi_s}")
    print(f"[40_eval_skill] wrote skill_results.json + tables/skill.tex + tables/compute.tex -> {results_dir}")


if __name__ == "__main__":
    main()
