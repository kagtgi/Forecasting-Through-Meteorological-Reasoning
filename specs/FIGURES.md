# Figure plan — ASG-WM / FaithCast manuscript

**Build:** loose `fig_*.pdf` are build artifacts (embedded inside `paper.pdf` after compiling) and are
**not kept** in the folder. Regenerate them anytime with `python build_figs.py` (SVGs → PDF +
the matplotlib gallery), then run `pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper`.
`fig_case` (gallery) is now embedded with **illustrative** values; replace with real forecasts via
`src/scripts/42_make_figures.py --gallery`.

All figures are authored as SVG → converted to PDF (`svglib`) → `\includegraphics` in `paper.tex`.
Data figures currently carry **illustrative (TBR) values** and are **regenerated from real result
JSON** by `src/scripts/42_make_figures.py` (and `src/asgwm/utils/viz.py`) once experiments run.
Generators for the schematic figures: `scratchpad/genfigs.py` and `scratchpad/genfigs2.py`.

Headline baseline set (main text): **pysteps · RainNet · NowcastNet · LangPrecip · ThoR · ASG-WM**.
Full 12-baseline suite → Supplementary.

| # | Figure (label) | File | Type | Status | To finalize |
|---|----------------|------|------|--------|-------------|
| 1 | Physics-informed gap (`fig:knowledge`) | `fig_knowledge.pdf` | schematic | ✅ final | — |
| 2 | Conceptual 5-step framework (`fig:system`) | `fig_framework.pdf` | schematic | ✅ final | — |
| 3 | **Detailed data-flow architecture** (`fig:arch`) | `fig_architecture.pdf` | schematic | ✅ final | — |
| 4 | **Stage C renderer NN architecture** (`fig:renderer`) | `fig_renderer.pdf` | schematic | ✅ final | — |
| 5 | Regime-stratified CSI (`fig:regime`) | `fig_regime.pdf` | data (TBR) | ⏳ illustrative | fill real per-regime CSI for the 6 methods + bootstrap CIs |
| 6 | Faithfulness 4-panel (`fig:faith`) | `fig_faith.pdf` | data (TBR) | ⏳ illustrative | C-i…C-iv real values |
| 7 | Counterfactual ASG editing (`fig:counterfactual`) | `fig_counterfactual.pdf` | schematic | ✅ final (schematic) | optional: real-frame version (below) |
| 8 | Lead-time decay (`fig:leadtime`) | `fig_leadtime.pdf` | data (TBR) | ⏳ illustrative | per-step CSI curves, 5 methods × 2 regimes |
| 9 | IB capacity study (`fig:capacity`) | `fig_capacity.pdf` | data (TBR) | ⏳ illustrative | real CSI vs N_max sweep + bit audit |
| 10 | Qualitative gallery (`fig:case`) | placeholder | needs real data | ⏳ layout only | render rows × lead-times from held-out SEVIR forecasts |
| 11 | Pilot forecaster study (`fig:forecaster`) | `fig_forecaster.pdf` | data (TBR) | ⏳ illustrative | real preference + interventional-alignment scores |

## Two data-dependent figures to render once forecasts exist

- **Fig 10 — qualitative gallery** (LangPrecip Fig 5 / ThoR style): left inset = regional map +
  initiation callout (radar → reasoning readout → ASG overlay → field); right grid = rows
  {Observations, ASG-WM, pysteps, RainNet, NowcastNet, LangPrecip, ThoR} × columns
  {T+30, T+60, T+90, T+120, T+180 min}, each cell annotated with CSI@16/CSI@45.
- **Real-frame counterfactual** (LangPrecip (a)/(b) style, optional new panel): T=1,4,8,12,16 frames
  with motion-vector overlays + a dominant-direction compass, before/after an ASG edit — the
  "edit the state → watch the field respond" demonstration on real radar.

## Optional (symbolic admissibility layer, prototype)

- **Certificate / ambiguity figure**: certificate pass-rate by regime + the dual-SAT ambiguity
  map (admissible-future set). Gate on fixing object-ID stability in the labeler first.
