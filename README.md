# Forecasting Through Meteorological Reasoning — ASG-WM / FaithCast

**A precipitation nowcaster whose structured world-model state is load-bearing by construction.**
A vision–language model reads a short radar history, constructs an explicit, human-readable
**Atmospheric Scene Graph (ASG)**; a physics-constrained transition rolls that state forward;
a physics-informed renderer then materializes the future radar field **from that state and
nothing else** — so faithfulness is *architecturally entailed* and provable by intervention.

> Technical handle: **ASG-WM** (Atmospheric Scene Graph world model). Working title: *FaithCast*.
> Target venue: *Scientific Reports*. This repository is the **framework** — code and specs are
> complete; quantitative paper results are **[TBR]** until the staged Colab A100 compute plan runs.
> Method philosophy: [`specs/philosophy.md`](specs/philosophy.md).

---

## The framework in five steps

```
 ① OBSERVE      ② IDENTIFY         ③ TRACK           ④ ANALYZE             ⑤ NOWCAST
 multi-frame    VLM emits per-     trajectories,     physics transition    physics renderer
 X(t) ingest    frame objects      IDs + velocity    advection+continuity  field = advect + Δ(Z)
                → ASG_t (state)    (ASG_t complete)  → ASG_{t+h}           Z = ASG_{t+h} ⊕ advect_blind
        Stage A (perception) ───────────────► Stage B (transition) ──[faithful bottleneck]──► Stage C (renderer)
```

The renderer's **only** window onto the future is the predicted ASG plus a *future-blind*
advection of the present. Zeroing the ASG collapses the forecast to advection; perturbing it
moves the field exactly as the state dictates. That is the contribution — faithfulness by
information bottleneck, not by assertion.

---

## Repository layout

```
.
├── README.md                ← this file
├── train.ipynb              ← one-click training notebook (clone → install → download → train all tiers → train_results.zip)
├── eval.ipynb               ← one-click eval notebook (eval ours → figures/tables → paper_assets.zip)
├── paper/                   ← the manuscript + figures + figure builder
│   ├── paper.tex / paper.pdf       (Scientific Reports manuscript, 20 pp.)
│   ├── references.bib
│   ├── build_figs.py                (regenerate fig_*.svg → fig_*.pdf)
│   ├── fig_*.svg / fig_*.pdf        (figures; data-figures carry illustrative TBR values)
│   └── wlscirep.cls, naturemag-doi.bst, template/                          (LaTeX support)
├── specs/                   ← design specs (the "why" behind the code)
│   ├── philosophy.md              (method framework ontology — start here for the "why")
│   ├── idea.md architecture.md datasource.md training_method.md eval.md
│   └── FEASIBILITY.md FIGURES.md
├── datasets/                ← data tooling + downloaded data (gitignored); see datasets/README.md
└── src/                     ← the ASG-WM framework codebase
    ├── README.md                   (full code documentation — start here to run anything)
    ├── RESULTS.md                  (results-pipeline reference)
    ├── requirements.txt, pyproject.toml, configs/default.yaml
    ├── asgwm/                       (the package)
    │   ├── asg/                     ASG schema, grammar, NL render (the data contract)
    │   ├── physics.py               differentiable advection / continuity / spectral ops
    │   ├── labeling/                pysteps auto-labeling (CPU): motion, tracking, regime, context
    │   ├── data/                    SEVIR / NEXRAD / MRMS loaders, normalize, future-blind advection
    │   ├── models/                  Stage A (QLoRA VLM), Stage B (transition), Stage C (renderer), bottleneck
    │   ├── train/                   Tier 0/1/2 training loops + losses + resumable checkpoints
    │   ├── eval/                    skill metrics, regime stratification, faithfulness (C-i..C-v), capacity
    │   ├── interventions.py         structured ASG perturbations
    │   └── utils/                   config + figure regeneration
    ├── scripts/                     00_download → 01_autolabel → 10/20/30 train → 40/41/42 eval/figures
    ├── notebooks/                   run_all_colab_A100_runtime1.ipynb, ..._runtime2.ipynb
    └── tests/                       pytest (synthetic-data smoke + unit tests)
```

---

## Datasets

The canonical representation for **all** datasets is a time-first tensor `[T, 384, 384]` at
1 km, single-channel **SEVIR VIL byte** (integers `0..254`, `255` = missing); the model input
is `byte / 255` in `[0, 1]`. Skill is scored with CSI at the standard SEVIR-VIL thresholds on
the **raw byte scale**: `[16, 74, 133, 160, 181, 219]` (matching Earthformer / PreDiff /
DiffCast / CasCast). Frame configs: the headline Earthformer-matched setting is `13 → 12`
(1 h out, 5-min cadence); the extended long-lead setting is `13 → 36` (3 h).

| Dataset | Role | Notes |
|---|---|---|
| **SEVIR** (VIL) | train **+** test | Primary. Temporal split: train `< 2019-01-01`, val `2019-01-01..2019-05-31`, test `>= 2019-06-01`. Already 384×384 @ 1 km. |
| **NEXRAD Level II** | OOD test only | Real radar, no training/split. Polar → gridded to a 384×384 1 km composite (dBZ), bridged to VIL byte. |
| **MRMS** MergedReflectivityQCComposite | OOD test only | Already-gridded CONUS ~1 km (dBZ), cropped to a 384×384 tile, bridged to VIL byte. |

NEXRAD and MRMS are used **entirely** as out-of-distribution test sets (no training, no split).
Both are bridged into VIL byte via the dBZ→VIL approximation in `asgwm.data.normalize`; because
composite reflectivity (column-max) and VIL (vertical integral) are physically different, the OOD
numbers measure generalization **under an imperfect variable bridge**. See
[`datasets/README.md`](datasets/README.md) for the download tooling, bucket details, and the full
canonical-representation spec; the design rationale is in [`specs/datasource.md`](specs/datasource.md).

---

## Notebooks

Two one-click notebooks at the repo root drive the whole paid run end-to-end:

- **[`train.ipynb`](train.ipynb)** — clone → install → download data → train all tiers
  (Tier 0/1/2) → packages checkpoints into `train_results.zip`.
- **[`eval.ipynb`](eval.ipynb)** — evaluate ours → regenerate figures/tables → packages them
  into `paper_assets.zip`.

The lower-level Colab A100 notebooks (`src/notebooks/run_all_colab_A100_runtime{1,2}.ipynb`)
remain for the two-session split documented in [TUTORIAL.md](TUTORIAL.md).

---

## Compute plan (two machines)

| Tier | What it proves | Runs on |
|------|----------------|---------|
| **Tier 0** | An explicit world model beats persistence/advection on growth/decay/initiation | **L4-24 GB** ($0) |
| **Tier 1** | VLM priors + physics equations improve perception & transition (5-phase QLoRA curriculum) | **L4-24 GB** ($0) |
| **Tier 2** | Faithfulness, probabilistic skill, realism (end-to-end + rectified-flow renderer + intervention training) | **A100-40 GB** (≤2 sessions, <12 h each) |

The two Colab A100 notebooks run the compute-critical path within two ≤12 h sessions
(data → labels → capacity audit → Tier 0 → Tier 1 → Tier 2 → eval → figures), checkpointing to
Drive between/within sessions. See [`src/README.md`](src/README.md) for what remains for the
always-on L4 (full-scale auto-labeling, knowledge ablations, cross-region, gold-set labeling).

The whole pipeline also runs end-to-end on **CPU with synthetic data** (no download, no GPU) for
wiring validation — see the smoke cell in runtime 1 and `tests/test_smoke_pipeline.py`.

**VRAM / time / cost:** measured model sizes and grounded estimates are in **[COMPUTE.md](COMPUTE.md)**.
Short version: the whole real pipeline fits a single **24 GB** GPU (Tier-2 with the documented knobs),
**40 GB A100** recommended; a first time-boxed run is **~10–18 h ≈ 2 A100 sessions (~$30–60)**.

---

## Quick start

```bash
cd src
pip install -r requirements.txt
# validate the wiring on synthetic data (no download, no GPU):
python scripts/00_download_data.py --override data.dataset=synthetic
python scripts/01_autolabel.py
python scripts/10_train_tier0.py --override train.tier0.max_steps=50
pytest -q
```

For the real run, use the root **[`train.ipynb`](train.ipynb)** / **[`eval.ipynb`](eval.ipynb)**
notebooks, or follow **[TUTORIAL.md](TUTORIAL.md)** step by step on Colab A100 (two sessions)
then your L4 VM. The results→figures→tables pipeline is documented in
[src/RESULTS.md](src/RESULTS.md); the figure plan in [specs/FIGURES.md](specs/FIGURES.md).

---

## Lineage

ASG-WM extends our prior physics-informed nowcaster **ThoR** (*Scientific Reports* 15:42075,
2025) — *ThoR = physics as a PDE soft-constraint; ASG-WM = physics as a structured, inspectable
world-model state.* The design rationale lives in [`specs/idea.md`](specs/idea.md) and the four
companion specs.
