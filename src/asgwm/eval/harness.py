"""Method-agnostic skill harness + LaTeX table emission (eval.md A/B/F).

`evaluate_skill(cfg)` runs ASG-WM and every registered baseline through `forecast.assemble`,
computes the manuscript metrics, and returns one canonical schema:

    {"methods": {name: {display, family, available, aggregate{...}, regime{...}, leadtime[...]}},
     "thresholds_dbz": [...], "lead_times_min": [...], "regimes": [...]}

Unavailable baselines get ``available: False`` and TBR cells. `skill_table_tex` / `compute_table_tex`
render the exact main-text tables from this schema, so the paper stays in sync with results.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from asgwm.eval import metrics as M
from asgwm.eval import forecast as FC
from asgwm import baselines as B
from asgwm.asg import REGIMES

ASGWM = "asgwm"
ASGWM_DISPLAY = "ASG-WM (ours)"


def _seq_mean(fn, pred, obs):
    n = min(len(pred), len(obs))
    return float(np.mean([fn(pred[k], obs[k]) for k in range(n)])) if n else float("nan")


def _method_metrics(samples: List[Dict], cfg) -> Dict:
    """Aggregate / regime / lead-time metrics for one method's samples."""
    thresholds = list(cfg.get_path("eval.thresholds_dbz", [16, 35, 45]))
    fss_scales = list(cfg.get_path("eval.fss_scales", [4, 16]))
    leads = list(cfg.get_path("eval.lead_times_min", [30, 60, 90, 120, 150, 180]))
    mpf = int(cfg.get_path("data.minutes_per_frame", 5))
    thr_mid = thresholds[len(thresholds) // 2]
    thr_hi = thresholds[-1]

    agg: Dict[str, float] = {}
    for thr in thresholds:
        agg[f"csi@{thr:g}"] = float(np.mean([_seq_mean(lambda p, o: M.csi(p, o, thr), s["pred"], s["obs"]) for s in samples]))
        agg[f"hss@{thr:g}"] = float(np.mean([_seq_mean(lambda p, o: M.hss(p, o, thr), s["pred"], s["obs"]) for s in samples]))
    for sc in fss_scales:
        agg[f"fss@{thr_mid:g}_s{sc}"] = float(np.mean([_seq_mean(lambda p, o: M.fss(p, o, thr_mid, sc), s["pred"], s["obs"]) for s in samples]))
    agg[f"pooled_csi@{thr_mid:g}"] = float(np.mean([_seq_mean(lambda p, o: M.pooled_csi(p, o, thr_mid, fss_scales[-1]), s["pred"], s["obs"]) for s in samples]))
    # CRPS over the ensemble, per frame
    crps = []
    for s in samples:
        ens, obs = s["ens"], s["obs"]
        n = min(ens.shape[1] if ens.ndim == 4 else len(obs), len(obs))
        if ens.ndim == 4 and n:
            crps.append(float(np.mean([M.crps_ensemble(ens[:, k], obs[k]) for k in range(n)])))
    agg["crps"] = float(np.mean(crps)) if crps else float("nan")
    agg["lpips"] = float(np.mean([_seq_mean(M.lpips_metric, s["pred"], s["obs"]) for s in samples]))

    # regime-stratified CSI at the heavy threshold
    regime: Dict[str, float] = {}
    for r in REGIMES:
        rs = [s for s in samples if s.get("regime") == r]
        regime[r] = (float(np.mean([_seq_mean(lambda p, o: M.csi(p, o, thr_hi), s["pred"], s["obs"]) for s in rs]))
                     if rs else float("nan"))

    # lead-time CSI@mid at each requested lead time (sub-sample frames)
    lt = []
    for ltmin in leads:
        idx = max(0, ltmin // mpf - 1)
        vals = [M.csi(s["pred"][idx], s["obs"][idx], thr_mid)
                for s in samples if len(s["pred"]) > idx and len(s["obs"]) > idx]
        lt.append(float(np.mean(vals)) if vals else float("nan"))
    return {"aggregate": agg, "regime": regime, "leadtime": lt}


def evaluate_skill(cfg, n_events: int = 24) -> Dict:
    methods: Dict[str, Dict] = {}
    # ASG-WM (always)
    s = FC.assemble(ASGWM, cfg, n_events=n_events)
    m = _method_metrics(s, cfg) if s else {"aggregate": {}, "regime": {}, "leadtime": []}
    methods[ASGWM] = {"display": ASGWM_DISPLAY, "family": "Reasoning WM", "available": bool(s), **m}
    # baselines (registry order)
    for name in B.HEADLINE:
        s = FC.assemble(name, cfg, n_events=n_events)
        if s:
            m = _method_metrics(s, cfg)
            methods[name] = {"display": B.display_name(name), "family": B.family(name), "available": True, **m}
        else:
            methods[name] = {"display": B.display_name(name), "family": B.family(name),
                             "available": False, "aggregate": {}, "regime": {}, "leadtime": []}
    return {
        "methods": methods,
        "thresholds_dbz": list(cfg.get_path("eval.thresholds_dbz", [16, 35, 45])),
        "lead_times_min": list(cfg.get_path("eval.lead_times_min", [30, 60, 90, 120, 150, 180])),
        "regimes": list(REGIMES),
    }


# ---------------------------------------------------------------------------
# LaTeX tables (TBR cells for unavailable methods)
# ---------------------------------------------------------------------------
def _cell(d: Dict, key: str, fmt: str = "{:.3f}", bold: bool = False) -> str:
    v = d.get(key)
    if v is None or (isinstance(v, float) and v != v):
        s = "--"
    else:
        s = fmt.format(v)
    return f"\\textbf{{{s}}}" if bold else s


def skill_table_tex(results: Dict) -> str:
    """Main-text skill table (Table 2) as a LaTeX tabular, ordered baselines then ours."""
    thr = results["thresholds_dbz"]
    L, Mi, Hi = thr[0], thr[len(thr)//2], thr[-1]
    fss_scale = 16
    order = list(B.HEADLINE) + [ASGWM]
    rows = []
    for name in order:
        md = results["methods"].get(name)
        if not md:
            continue
        a = md.get("aggregate", {})
        bold = (name == ASGWM)
        disp = md["display"]
        if bold:
            disp = f"\\textbf{{{disp}}}"
        cells = [
            _cell(a, f"csi@{L:g}", bold=bold), _cell(a, f"csi@{Mi:g}", bold=bold), _cell(a, f"csi@{Hi:g}", bold=bold),
            _cell(a, f"hss@{L:g}", bold=bold), _cell(a, f"hss@{Mi:g}", bold=bold), _cell(a, f"hss@{Hi:g}", bold=bold),
            _cell(a, f"fss@{Mi:g}_s{fss_scale}", bold=bold),
            _cell(a, f"pooled_csi@{Mi:g}", bold=bold),
            _cell(a, "crps", bold=bold),
        ]
        rows.append(f"{disp} & {md['family']} & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{@{}llccccccccc@{}}" "\n" r"\toprule" "\n"
            r" & & \multicolumn{3}{c}{CSI ($\uparrow$)} & \multicolumn{3}{c}{HSS ($\uparrow$)} & FSS & Pooled & CRPS \\" "\n"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}" "\n"
            r"Method & Family & L & M & H & L & M & H & 16${\times}$16 & CSI & ($\downarrow$) \\" "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"


def compute_table_tex(compute: Dict) -> str:
    """Computational-footprint table (Table 4). `compute` keyed by method name -> dict."""
    order = list(B.HEADLINE) + [ASGWM]
    rows = []
    for name in order:
        md = compute.get(name, {})
        disp = B.display_name(name) if name != ASGWM else r"\textbf{ASG-WM (ours)}"
        cells = [_cell(md, k) for k in ("latency_ms", "mem_gb", "ensemble_s", "train_gpuh")]
        rows.append(f"{disp} & " + " & ".join(cells) + r" \\")
    head = (r"\begin{tabular}{@{}lcccc@{}}" "\n" r"\toprule" "\n"
            r"Method & Infer.\ latency (ms) $\downarrow$ & Peak mem.\ (GB) $\downarrow$ & "
            r"Ensemble ($K{=}10$) (s) $\downarrow$ & Train (GPU-h) $\downarrow$ \\" "\n" r"\midrule")
    return head + "\n" + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}"


def regime_fig_data(results: Dict) -> Dict:
    """Reshape the skill schema into viz.plot_regime_bars input (only available methods)."""
    regimes = results["regimes"]
    methods = {}
    order = [ASGWM] + list(B.HEADLINE)
    for name in order:
        md = results["methods"].get(name)
        if md and md.get("available"):
            methods[md["display"]] = [md["regime"].get(r, float("nan")) for r in regimes]
    return {"metric": f"csi@{results['thresholds_dbz'][-1]:g}", "regimes": regimes, "methods": methods}


def leadtime_fig_data(results: Dict) -> Dict:
    methods = {}
    order = [ASGWM] + list(B.HEADLINE)
    for name in order:
        md = results["methods"].get(name)
        if md and md.get("available") and md.get("leadtime"):
            methods[md["display"]] = md["leadtime"]
    thr = results["thresholds_dbz"]
    return {"lead_times_min": results["lead_times_min"],
            "metric": f"csi@{thr[len(thr)//2]:g}", "methods": methods}
