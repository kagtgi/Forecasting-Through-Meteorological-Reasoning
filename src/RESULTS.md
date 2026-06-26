# Results & figures pipeline — design

How the codebase turns a trained ASG-WM model into **every data figure and table** in the
paper, with the five comparison methods as pluggable slots filled later.

```
 trained ckpts ─┐
 (or synthetic) │   ┌─ scripts/40_eval_skill.py ─┐  skill_results.json ─┐
 SEVIR events ──┼──►│  harness.evaluate_skill     │  tables/skill.tex   │
 baselines ─────┘   │  (forecast.assemble × all)  │  tables/compute.tex │
                    ├─ scripts/41_eval_faithfulness ─ faithfulness_results.json ─┤─► scripts/42_make_figures.py
                    │  (C-i..C-v + capacity)         capacity in same file        │   → results/figures/*.pdf|png
                    └─ (forecaster: human study)  ─  forecaster_results.json ─────┘
```

Schematic figures (knowledge, framework, architecture, renderer, counterfactual-schematic) are
hand-authored SVGs in `paper/` and are **not** produced here. Everything **data-driven** is.

## Method abstraction — ours now, five later

`asgwm/baselines/` is a registry. A baseline implements `Baseline.predict(frames_hist, context,
n_out)` and `is_available()`. Until coded it reports unavailable and the harness writes a **TBR**
row, so figures/tables render today with the ASG-WM row real and baseline rows pending.

| name | display | family | status |
|---|---|---|---|
| `pysteps` | pysteps | Extrapolation | **implemented** (free; = future-blind advection) |
| `rainnet` | RainNet | CNN / U-Net | stub → TBR |
| `nowcastnet` | NowcastNet | Physics-generative | stub → TBR |
| `langprecip` | LangPrecip | Language-guided | stub → TBR |
| `thor` | ThoR | Physics-informed (ours-prior) | stub → TBR |

**To add a baseline later:** implement `predict` + flip `is_available` in
`asgwm/baselines/adapters.py`. Nothing else changes — it appears in every table/figure automatically.

ASG-WM forecasts come from `eval/forecast.py`: the real Stage A→B→bottleneck→C path when
Tier-0/2 checkpoints exist, else a future-blind-advection fallback (so the pipeline runs
pre-training; numbers are TBR until Tier-2 completes).

## Canonical result files (`paths.results/`)

| File | Written by | Schema (key parts) |
|---|---|---|
| `skill_results.json` | 40 | `{methods:{name:{display,family,available,aggregate{csi@/hss@/fss@/pooled_csi@/crps/lpips}, regime{init/grow/decay/steady}, leadtime[...]}}, thresholds_dbz, lead_times_min, regimes}` |
| `compute_results.json` | 40 | `{method:{latency_ms,mem_gb,ensemble_s,train_gpuh}}` (TBR until measured) |
| `faithfulness_results.json` | 41 | `{intervention, ablation, leakage, curriculum, capacity_audit, capacity_sweep}` |
| `forecaster_results.json` | 42/manual | `{methods:{name:{mean_rank,sem}}}` (human study) |
| `tables/skill.tex`, `tables/compute.tex` | 40 | `\begin{tabular}…` for paper Tables 2 & 4 |

## Figure → data → producer map

| Figure (label) | Data file | Producer |
|---|---|---|
| `fig_regime` | skill_results | `harness.regime_fig_data` → `viz.plot_regime_bars` |
| `fig_leadtime` | skill_results | `harness.leadtime_fig_data` → `viz.plot_leadtime` |
| `fig_faith` | faithfulness_results | `viz.plot_faithfulness` |
| `fig_capacity` | faithfulness_results | `viz.plot_capacity` |
| `fig_forecaster` | forecaster_results | `viz.plot_forecaster` |
| `fig_case` (gallery) | live `forecast.assemble` | `viz.plot_gallery` (`42 --gallery`) |
| `fig_counterfactual_real` | live `faithfulness.counterfactual_demo` | `viz.plot_counterfactual_real` |
| Table 2 (skill) | skill_results | `harness.skill_table_tex` |
| Table 3 (ablation) | faithfulness/ablation runs | (Group E/F/G runner — fill from §D ablations) |
| Table 4 (compute) | compute_results | `harness.compute_table_tex` |

## Run order

```bash
cd code
# 0) data + labels (once, cache to paths.root)
python scripts/00_download_data.py  --override paths.root=$ROOT
python scripts/01_autolabel.py      --override paths.root=$ROOT
# 1) train ASG-WM  (L4: tier0,tier1 ; A100: tier2) — see notebooks/
python scripts/10_train_tier0.py    --override paths.root=$ROOT
python scripts/20_train_tier1_curriculum.py --override paths.root=$ROOT
python scripts/30_train_tier2.py    --override paths.root=$ROOT
# 2) evaluate + figures + tables
python scripts/40_eval_skill.py         --override paths.root=$ROOT
python scripts/41_eval_faithfulness.py  --override paths.root=$ROOT
python scripts/42_make_figures.py --gallery --override paths.root=$ROOT
# -> results/figures/*.pdf  and  results/tables/*.tex   (copy the PDFs into paper/)
```

Everything also runs end-to-end on **synthetic data, CPU, no checkpoints** (advection fallback) for
wiring validation: same commands with `--override data.dataset=synthetic`.

## What is real now vs after training

- **Real now:** the full pipeline, schema, baseline registry, table/figure emitters, pysteps row.
- **TBR until trained:** ASG-WM metric *values* (uses advection fallback until a Tier-2 checkpoint
  exists), and the four neural-baseline rows (until their adapters are implemented).
- **Human-gated:** `fig_forecaster` (needs a forecaster-study partner).
