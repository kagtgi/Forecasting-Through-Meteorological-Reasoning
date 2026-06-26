"""Paper-figure regeneration from real result JSON (eval.md figures).

Regenerates the paper's TBR figures from result dicts written by the eval scripts:
    - fig_regime    : regime-stratified skill bars (eval.md section 1B).
    - fig_faith     : faithfulness bars — C-i intervention consistency + C-ii ablation pattern.
    - fig_leadtime  : skill vs lead time (eval.md section 1F, ``eval.lead_times_min``).
    - fig_capacity  : ASG vs input capacity + the N_max sweep (training_method.md section 4).
    - fig_forecaster: forecaster-study ranking (eval.md section 1E).

Style: Google palette (#4285F4 blue, #EA4335 red, #FBBC04 yellow, #34A853 green), thin grey
axes, sentence-case labels. matplotlib is imported lazily so the module imports without a display.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

# Google palette (blue, red, yellow, green) — fixed across all figures.
GOOGLE_BLUE = "#4285F4"
GOOGLE_RED = "#EA4335"
GOOGLE_YELLOW = "#FBBC04"
GOOGLE_GREEN = "#34A853"
_AXIS_GREY = "#9AA0A6"
# 6-method palette: ASG-WM (blue) first, then pysteps/RainNet/NowcastNet/LangPrecip/ThoR.
PALETTE = [GOOGLE_BLUE, "#7F8C8D", "#5BA3F5", GOOGLE_YELLOW, GOOGLE_GREEN, GOOGLE_RED]
# Stable per-method colours (used when a figure keys by method name).
METHOD_COLORS = {
    "asg-wm (ours)": GOOGLE_BLUE, "asg-wm": GOOGLE_BLUE,
    "pysteps": "#7F8C8D", "rainnet": "#5BA3F5",
    "nowcastnet": GOOGLE_YELLOW, "langprecip": GOOGLE_GREEN, "thor": GOOGLE_RED,
}


def method_color(name: str, i: int = 0) -> str:
    return METHOD_COLORS.get(str(name).lower(), PALETTE[i % len(PALETTE)])


def _mpl():
    """Lazily import matplotlib with a non-interactive backend and shared style."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "axes.edgecolor": _AXIS_GREY,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#ECECEC",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "figure.dpi": 120,
    })
    return plt


def _save(fig, out: str) -> List[str]:
    """Save a figure as both PDF and PNG; return the written paths."""
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    base, _ = os.path.splitext(out)
    paths = []
    for ext in (".pdf", ".png"):
        p = base + ext
        fig.savefig(p, bbox_inches="tight")
        paths.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return paths


def save_results_json(obj, out: str) -> str:
    """Write a results dict to JSON (numpy-safe), creating parent dirs. Returns the path."""
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    def _default(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:
            pass
        return str(o)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_default)
    return out


# ---------------------------------------------------------------------------
# fig_regime — regime-stratified skill bars
# ---------------------------------------------------------------------------
def plot_regime_bars(data: Dict, out: str) -> List[str]:
    """Grouped bars of CSI per regime per method (eval.md section 1B).

    Expected ``data`` shape::

        {"metric": "csi@45", "regimes": ["init","grow","decay","steady"],
         "methods": {"ASG-WM": [..], "pysteps": [..], "ConvLSTM": [..]}}

    The method order defines the palette assignment; ASG-WM should be first (blue).
    """
    plt = _mpl()
    regimes = data.get("regimes", ["init", "grow", "decay", "steady"])
    methods = data.get("methods", {})
    metric = data.get("metric", "csi")
    n_methods = max(len(methods), 1)
    width = 0.8 / n_methods

    import numpy as np
    x = np.arange(len(regimes))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, (name, vals) in enumerate(methods.items()):
        vals = [0.0 if (v != v) else v for v in list(vals)] + [0.0] * (len(regimes) - len(vals))
        ax.bar(x + i * width - 0.4 + width / 2, vals[:len(regimes)], width,
               label=name, color=method_color(name, i), edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([r.capitalize() for r in regimes])
    ax.set_ylabel(metric.replace("@", " at ").upper() if metric else "Skill")
    ax.set_xlabel("Regime")
    ax.set_title("Skill stratified by regime")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8, ncol=min(n_methods, 4))
    return _save(fig, out)


# ---------------------------------------------------------------------------
# fig_faith — faithfulness (C-i + C-ii)
# ---------------------------------------------------------------------------
def plot_faithfulness(data: Dict, out: str) -> List[str]:
    """Two-panel faithfulness figure: C-i per-type consistency + C-ii ablation pattern.

    Expected ``data`` shape::

        {"intervention": {"score": 0.9, "per_type": {"translate": {"score":..}, ...}},
         "ablation": {"oracle":.., "inferred":.., "zeroed":.., "shuffled":.., "advection":..}}
    """
    plt = _mpl()
    import numpy as np
    inter = data.get("intervention", {})
    abl = data.get("ablation", {})

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    # Panel 1: C-i intervention consistency per type.
    per_type = inter.get("per_type", {})
    types = list(per_type.keys())
    scores = [per_type[t].get("score", 0.0) if isinstance(per_type[t], dict) else float(per_type[t])
              for t in types]
    ax0 = axes[0]
    ax0.bar(np.arange(len(types)), scores, color=GOOGLE_BLUE, edgecolor="white", linewidth=0.4)
    ax0.set_xticks(np.arange(len(types)))
    ax0.set_xticklabels([t.replace("_", " ") for t in types], rotation=20, ha="right", fontsize=8)
    ax0.set_ylim(0, 1.0)
    ax0.set_ylabel("Consistency score")
    overall = inter.get("score", None)
    ax0.set_title("Intervention consistency"
                  + (f" (overall {overall:.2f})" if overall is not None else ""))

    # Panel 2: C-ii ablation pattern (oracle/inferred/zeroed/shuffled vs advection).
    order = ["oracle", "inferred", "zeroed", "shuffled", "advection"]
    labels = [k for k in order if k in abl]
    vals = [abl[k] for k in labels]
    colors = {
        "oracle": GOOGLE_GREEN, "inferred": GOOGLE_BLUE,
        "zeroed": GOOGLE_RED, "shuffled": GOOGLE_YELLOW, "advection": _AXIS_GREY,
    }
    ax1 = axes[1]
    ax1.bar(np.arange(len(labels)), vals,
            color=[colors.get(k, GOOGLE_BLUE) for k in labels],
            edgecolor="white", linewidth=0.4)
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels([k.capitalize() for k in labels], rotation=20, ha="right", fontsize=8)
    ax1.set_ylabel("CSI")
    ax1.set_title("Bottleneck ablation")
    if "advection" in abl:
        ax1.axhline(abl["advection"], color=_AXIS_GREY, linestyle="--", linewidth=0.8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# fig_leadtime — skill vs lead time
# ---------------------------------------------------------------------------
def plot_leadtime(data: Dict, out: str) -> List[str]:
    """Line plot of skill vs lead time per method (eval.md section 1F).

    Expected ``data`` shape::

        {"lead_times_min": [30,60,...], "metric": "csi@35",
         "methods": {"ASG-WM": [..], "pysteps": [..]}}
    """
    plt = _mpl()
    leads = data.get("lead_times_min", [30, 60, 90, 120, 150, 180])
    methods = data.get("methods", {})
    metric = data.get("metric", "csi")
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for i, (name, vals) in enumerate(methods.items()):
        vals = list(vals)[:len(leads)]
        lw = 2.6 if str(name).lower().startswith("asg") else 1.6
        ax.plot(leads[:len(vals)], vals, marker="o", markersize=4,
                color=method_color(name, i), linewidth=lw, label=name)
    ax.set_xlabel("Lead time (min)")
    ax.set_ylabel(metric.replace("@", " at ").upper() if metric else "Skill")
    ax.set_title("Skill versus lead time")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# fig_capacity — bottleneck capacity audit + N_max sweep
# ---------------------------------------------------------------------------
def plot_capacity(data: Dict, out: str) -> List[str]:
    """Capacity audit figure: ASG vs input bits (log) + the N_max capacity/skill sweep.

    Expected ``data`` shape::

        {"audit": {"asg_bits":.., "input_bits":..},
         "sweep": {"nmax":[..], "bits":[..], "csi":[..]}}
    """
    plt = _mpl()
    import numpy as np
    audit = data.get("audit", {})
    sweep = data.get("sweep", {})

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))

    # Panel 1: ASG bits vs input bits (log scale) — must show ASG << input.
    ax0 = axes[0]
    asg_bits = audit.get("asg_bits", 0.0)
    input_bits = audit.get("input_bits", 1.0)
    ax0.bar([0, 1], [asg_bits, input_bits],
            color=[GOOGLE_BLUE, _AXIS_GREY], edgecolor="white", linewidth=0.4)
    ax0.set_yscale("log")
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["ASG state", "Raw input"])
    ax0.set_ylabel("Channel capacity (bits, log)")
    ratio = asg_bits / input_bits if input_bits else 0.0
    ax0.set_title(f"Bottleneck capacity (ASG is {ratio:.1e}x input)")

    # Panel 2: N_max sweep — capacity (bars) + CSI (line, twin axis).
    ax1 = axes[1]
    nmax = sweep.get("nmax", [])
    bits = sweep.get("bits", [])
    csi = sweep.get("csi", [])
    x = np.arange(len(nmax))
    ax1.bar(x, bits, color=GOOGLE_YELLOW, edgecolor="white", linewidth=0.4, label="ASG bits")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in nmax])
    ax1.set_xlabel("N_max")
    ax1.set_ylabel("ASG capacity (bits)")
    csi_clean = [c for c in csi if c == c]  # drop NaNs
    if csi_clean:
        ax2 = ax1.twinx()
        ax2.plot(x, csi, marker="o", markersize=4, color=GOOGLE_GREEN, linewidth=1.8, label="CSI")
        ax2.set_ylabel("CSI")
        ax2.set_ylim(0, 1.0)
        ax2.spines["top"].set_visible(False)
    ax1.set_title("Capacity / skill versus N_max")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# fig_forecaster — forecaster study ranking (eval.md section 1E)
# ---------------------------------------------------------------------------
def plot_forecaster(data: Dict, out: str) -> List[str]:
    """Horizontal bars of mean expert preference rank per method (eval.md section 1E).

    Expected ``data`` shape::

        {"methods": {"ASG-WM": {"mean_rank":1.4, "sem":0.1}, "DGMR": {...}}}
    or ``{"methods": {"ASG-WM": 1.4, ...}}`` (mean rank only).
    """
    plt = _mpl()
    import numpy as np
    methods = data.get("methods", {})
    names, means, sems = [], [], []
    for name, v in methods.items():
        names.append(name)
        if isinstance(v, dict):
            means.append(v.get("mean_rank", 0.0))
            sems.append(v.get("sem", 0.0))
        else:
            means.append(float(v))
            sems.append(0.0)
    order = np.argsort(means)  # best (lowest rank) on top
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    sems = [sems[i] for i in order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(6.0, 0.5 * len(names) + 1.5))
    colors = [GOOGLE_BLUE if n.lower().startswith("asg") else _AXIS_GREY for n in names]
    ax.barh(y, means, xerr=sems if any(sems) else None, color=colors,
            edgecolor="white", linewidth=0.4, error_kw={"ecolor": _AXIS_GREY, "lw": 0.8})
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Mean expert rank (lower is better)")
    ax.set_title("Forecaster preference study")
    return _save(fig, out)


# ---------------------------------------------------------------------------
# fig_case — qualitative gallery (methods x lead-times, with CSI), LangPrecip Fig.5 style
# ---------------------------------------------------------------------------
def plot_gallery(obs_seq, methods: Dict, lead_times_min: Sequence[int], out: str,
                 mpf: int = 5, thr: float = 45.0, vmax: Optional[float] = None) -> List[str]:
    """Rows = Observations + each method; columns = lead times; per-cell CSI annotation.

    Args:
        obs_seq: ground-truth sequence ``[n, H, W]``.
        methods: ``{display_name: pred_seq[n,H,W]}`` (only available methods).
        lead_times_min: column lead times; frame index = ``lt // mpf - 1``.
    """
    plt = _mpl()
    import numpy as np
    from ..eval import metrics as Mmet  # local import to avoid cycle at module load

    obs = np.asarray(obs_seq)
    rows = ["Observations"] + list(methods.keys())
    cols = list(lead_times_min)
    idxs = [max(0, lt // mpf - 1) for lt in cols]
    n = obs.shape[0]
    idxs = [min(i, n - 1) for i in idxs]
    vmax = float(vmax if vmax is not None else max(1.0, np.percentile(obs, 99)))

    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.5 * len(cols) + 1, 1.5 * len(rows)),
                             squeeze=False)
    for c, (lt, fi) in enumerate(zip(cols, idxs)):
        axes[0][c].set_title(f"T+{lt} min", fontsize=9)
        axes[0][c].imshow(obs[fi], cmap="turbo", vmin=0, vmax=vmax)
        for r, name in enumerate(methods.keys(), start=1):
            pred = np.asarray(methods[name])
            fld = pred[min(fi, pred.shape[0] - 1)]
            axes[r][c].imshow(fld, cmap="turbo", vmin=0, vmax=vmax)
            csi = Mmet.csi(fld, obs[min(fi, n - 1)], thr)
            axes[r][c].text(0.5, -0.08, f"CSI {csi:.2f}", transform=axes[r][c].transAxes,
                            ha="center", va="top", fontsize=6.5, color="#333")
    for r, name in enumerate(rows):
        axes[r][0].set_ylabel(name, fontsize=8, rotation=90, va="center")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Qualitative forecast gallery (CSI at heavy threshold)", fontsize=10)
    return _save(fig, out)


# ---------------------------------------------------------------------------
# real-frame counterfactual (edit the ASG -> field responds), LangPrecip (a)/(b) style
# ---------------------------------------------------------------------------
def plot_counterfactual_real(demo: Dict, out: str, vmax: Optional[float] = None) -> List[str]:
    """Rows = edit kinds; columns = [original, edited, difference]. From counterfactual_demo()."""
    plt = _mpl()
    import numpy as np

    def _f2d(x):
        a = np.asarray(x, dtype=np.float32)
        return a[0] if a.ndim == 3 else a

    base = _f2d(demo.get("base_field"))
    edited = demo.get("edited_fields", {})
    diffs = demo.get("diffs", {})
    kinds = list(edited.keys())
    if not kinds:
        return []
    vmax = float(vmax if vmax is not None else max(1.0, np.percentile(base, 99)))
    dmax = max(1e-3, max(np.abs(_f2d(diffs[k])).max() for k in kinds))

    fig, axes = plt.subplots(len(kinds), 3, figsize=(6.5, 2.1 * len(kinds)), squeeze=False)
    for r, k in enumerate(kinds):
        axes[r][0].imshow(base, cmap="turbo", vmin=0, vmax=vmax)
        axes[r][1].imshow(_f2d(edited[k]), cmap="turbo", vmin=0, vmax=vmax)
        axes[r][2].imshow(_f2d(diffs[k]), cmap="RdBu_r", vmin=-dmax, vmax=dmax)
        axes[r][0].set_ylabel(k.replace("_", " "), fontsize=8, rotation=90, va="center")
    for c, t in enumerate(["Original", "Edited ASG", "Difference"]):
        axes[0][c].set_title(t, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Counterfactual ASG editing (rendered fields)", fontsize=10)
    return _save(fig, out)
