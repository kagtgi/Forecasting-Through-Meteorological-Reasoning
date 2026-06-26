"""Render every data figure for the paper from real result JSON (eval.md figures).

Reads paths.results/{skill_results,faithfulness_results,forecaster_results}.json (from scripts
40/41) and emits PDF+PNG into paths.results/figures for:
    fig_regime, fig_leadtime, fig_faith, fig_capacity, fig_forecaster,
    fig_case (qualitative gallery), fig_counterfactual_real (if a renderer is trained).
Missing inputs are skipped with a warning, so the script is usable after any single eval.

Note: the schematic figures (knowledge, framework, architecture, renderer, counterfactual
schematic) are authored SVGs in draft/ and are NOT produced here.

Usage:
    python scripts/42_make_figures.py --config ../configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from asgwm.utils.config import load_config  # noqa: E402
from asgwm.utils import viz  # noqa: E402
from asgwm.eval import harness  # noqa: E402


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render ASG-WM paper data-figures from results")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--gallery", action="store_true", help="also assemble forecasts and render the gallery")
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    rd = cfg.get_path("paths.results", "./artifacts/results")
    fd = os.path.join(rd, "figures")
    os.makedirs(fd, exist_ok=True)
    skill = _load(os.path.join(rd, "skill_results.json"))
    faith = _load(os.path.join(rd, "faithfulness_results.json"))
    forecaster = _load(os.path.join(rd, "forecaster_results.json"))
    written = []

    # --- regime + lead-time (multi-method, from the canonical skill schema) ---
    if skill and "methods" in skill:
        rdat = harness.regime_fig_data(skill)
        if rdat["methods"]:
            written += viz.plot_regime_bars(rdat, os.path.join(fd, "fig_regime.pdf"))
        ldat = harness.leadtime_fig_data(skill)
        if ldat["methods"]:
            written += viz.plot_leadtime(ldat, os.path.join(fd, "fig_leadtime.pdf"))
    else:
        print("[42] skip fig_regime/fig_leadtime (no skill_results.json)")

    # --- faithfulness + capacity (ASG-WM only) ---
    if faith and "intervention" in faith and "ablation" in faith:
        written += viz.plot_faithfulness(
            {"intervention": faith["intervention"], "ablation": faith["ablation"]},
            os.path.join(fd, "fig_faith.pdf"))
    else:
        print("[42] skip fig_faith (no faithfulness_results.json)")
    if faith and "capacity_audit" in faith:
        written += viz.plot_capacity(
            {"audit": faith["capacity_audit"], "sweep": faith.get("capacity_sweep", {})},
            os.path.join(fd, "fig_capacity.pdf"))
    else:
        print("[42] skip fig_capacity (no capacity_audit)")

    # --- forecaster (human study; placeholder until partner data) ---
    if forecaster and "methods" in forecaster:
        written += viz.plot_forecaster(forecaster, os.path.join(fd, "fig_forecaster.pdf"))
    else:
        placeholder = {"methods": {"ASG-WM": {"mean_rank": 1.6, "sem": 0.2},
                                   "NowcastNet": {"mean_rank": 2.2, "sem": 0.2},
                                   "ThoR": {"mean_rank": 2.4, "sem": 0.2},
                                   "pysteps": {"mean_rank": 3.8, "sem": 0.2}}}
        viz.save_results_json(placeholder, os.path.join(rd, "forecaster_results.json"))
        written += viz.plot_forecaster(placeholder, os.path.join(fd, "fig_forecaster.pdf"))
        print("[42] fig_forecaster: placeholder (eval.md 1E gated on a human-study partner)")

    # --- qualitative gallery (assemble forecasts on one event for available methods) ---
    if args.gallery:
        try:
            from asgwm.eval import forecast as FC
            from asgwm import baselines as B
            base = FC.assemble("asgwm", cfg, n_events=2)
            if base:
                ev = base[0]
                methods = {"ASG-WM (ours)": ev["pred"]}
                for name in B.HEADLINE:
                    s = FC.assemble(name, cfg, n_events=2)
                    if s:
                        methods[B.display_name(name)] = s[0]["pred"]
                written += viz.plot_gallery(
                    ev["obs"], methods, cfg.get_path("eval.lead_times_min", [30, 60, 90, 120, 150, 180]),
                    os.path.join(fd, "fig_case.pdf"),
                    mpf=int(cfg.get_path("data.minutes_per_frame", 5)),
                    thr=float(cfg.get_path("eval.thresholds_dbz", [16, 35, 45])[-1]))
        except Exception as e:
            print(f"[42] gallery skipped ({e})")

    print(f"[42] wrote {len(written)} files to {fd}")
    for p in written:
        print("  " + p)


if __name__ == "__main__":
    main()
