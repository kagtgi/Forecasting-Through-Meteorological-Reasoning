# ASG-WM — Atmospheric Scene Graph world model for faithful precipitation nowcasting

ASG-WM (project name *FaithCast*) is a research framework for precipitation nowcasting whose
forecasts are routed through a small, human-readable **Atmospheric Scene Graph (ASG)** state.
The pipeline is three readable stages joined by a *faithful information bottleneck*:

```
Stage A (perception)  ─▶  ASG_t  ─▶  Stage B (transition)  ─▶  ASG_{t+h}
                                            │
                                            ▼
              [ faithful bottleneck:  Z = ASG_{t+h} ⊕ advect_blind(X≤t) ]
                                            │
                                            ▼
                          Stage C (rectified-flow renderer)  ─▶  field
```

The compression precondition (`asg_bits << input_bits`) is what lets us argue that the renderer
*has* to use the ASG, and the intervention-consistency / bottleneck-ablation evals are what
verify it actually does. See `../specs/{idea,architecture,datasource,training_method,eval}.md`
for the full design.

> **Status — framework only.** Every interface in the contract is implemented and the whole
> pipeline runs end-to-end on a deterministic synthetic dataset (`SyntheticSEVIR`) with **no
> download, no GPU**. Trained weights and **baselines (NowcastNet, DGMR, pysteps, …) are
> intentionally NOT included yet** — the eval scripts emit clearly-labeled synthetic/placeholder
> comparison rows so figures regenerate, and those rows are replaced when real runs land.

---

## Repository layout

```
code/
├── asgwm/                         # the importable package
│   ├── asg/                       # FOUNDATION (do not modify) — the ASG state itself
│   │   ├── schema.py              #   ASG, StormObject, ASGSequence, REGIMES, N_MAX=16, …
│   │   ├── grammar.py             #   serialize / parse / parse_strict (the ASG text grammar)
│   │   └── render_nl.py           #   render_NL, render_NL_delta, assertion_check
│   ├── physics.py                 # FOUNDATION (torch) — semi-Lagrangian advection, continuity,
│   │                              #   divergence, mass budget, radial PSD / spectral loss
│   ├── labeling/                  # pysteps auto-labeling pass (CPU, run once, freeze)
│   │   ├── motion.py              #   estimate_motion (Lucas-Kanade / phase-corr fallback)
│   │   ├── tracking.py            #   segment_cells, track_cells (watershed / ndimage fallback)
│   │   ├── regime.py              #   growth_scalar, classify_regime
│   │   ├── context.py             #   colocate_context (CAPE/CIN/shear/pwat/DEM; best-effort)
│   │   └── pipeline.py            #   build_asg_pair, autolabel_event -> ASGSequence
│   ├── data/                      # data access + torch datasets
│   │   ├── sevir.py               #   download_sevir_subset, iter_events, SyntheticSEVIR
│   │   ├── advection.py           #   advect_blind / future_blind_baseline (FUTURE-BLIND)
│   │   └── dataset.py             #   ASGTransitionDataset, RendererDataset, VLMCurriculumDataset
│   ├── models/
│   │   ├── prompts.py             #   PHASE_PROMPTS, EQUATION_BLOCK, build_prompt
│   │   ├── stage_a_vlm.py         #   StageAVLM (QLoRA) + DummyVLM CPU fallback
│   │   ├── stage_b_transition.py  #   TransitionTransformer, encode/decode_asg, transition_loss
│   │   ├── bottleneck.py          #   asg_to_field_channels, build_Z, zero_asg_in_Z, soft_ib_penalty
│   │   ├── unet.py                #   ConditionalUNet (rectified-flow velocity net)
│   │   ├── vae.py                 #   VAEWrapper (+ IdentityVAE fallback)
│   │   └── stage_c_renderer.py    #   LatentRectifiedFlowRenderer (residual-on-advection)
│   ├── train/
│   │   ├── checkpoint.py          #   save_ckpt / load_ckpt / latest  (atomic, resume-safe)
│   │   ├── losses.py              #   tier2_total_loss, intervention_consistency_loss
│   │   ├── tier0.py               #   train_transition, train_deterministic_renderer, gate_check
│   │   ├── tier1_curriculum.py    #   run_phase, run_curriculum (Ph-3 ASG-F1 hard gate)
│   │   └── tier2_endtoend.py      #   train_tier2 (A→B→bottleneck→C, scheduled sampling)
│   ├── interventions.py           #   perturb_asg, expected_effect, intervention_pairs
│   ├── eval/
│   │   ├── metrics.py             #   csi/hss/pod/far, fss, pooled_csi, crps_ensemble, lpips, psd
│   │   ├── regime_eval.py         #   stratify_by_regime, regime_skill_table
│   │   ├── faithfulness.py        #   C-i..C-v: intervention_consistency, bottleneck_ablation,
│   │   │                          #   leakage_audit (CLUB), asg_accuracy, counterfactual_demo
│   │   └── capacity.py            #   capacity_bits, capacity_audit, capacity_sweep
│   └── utils/
│       ├── config.py              #   Config (dotted get/set) + load_config
│       └── viz.py                 #   paper-figure plotting (Google palette) + save_results_json
├── scripts/                       # thin argparse CLIs over asgwm.* (see "Running" below)
│   ├── 00_download_data.py        20_train_tier1_curriculum.py   41_eval_faithfulness.py
│   ├── 01_autolabel.py            30_train_tier2.py              42_make_figures.py
│   ├── 10_train_tier0.py          40_eval_skill.py
├── notebooks/                     # Colab A100 runbooks (two <12 h resumable runtimes)
│   ├── run_all_colab_A100_runtime1.ipynb
│   └── run_all_colab_A100_runtime2.ipynb
├── configs/
│   └── default.yaml               # the single source of truth for all hyperparameters/paths
├── tests/                         # pytest target (testpaths=tests in pyproject) — framework slot
├── pyproject.toml                 # package metadata + pytest config
├── requirements.txt
└── README.md                      # this file
```

The `asgwm/asg/` modules and `asgwm/physics.py` are the **foundation**: read them, import them,
do not modify them or any `__init__.py`.

---

## Install

`torch`/`torchvision` are preinstalled on Colab — the file lists them only for local reproduction.

```bash
# from code/
pip install -r requirements.txt
pip install -e .            # optional: installs the `asgwm` package (pyproject.toml)
```

Heavy optional dependencies (transformers/peft/bitsandbytes, diffusers, lpips, pysteps,
scikit-image, s3fs/h5py, lm-format-enforcer) are **soft**. If any is missing the code falls back
gracefully:

| Missing dependency            | Fallback behavior                                                       |
| ----------------------------- | ----------------------------------------------------------------------- |
| transformers/peft/bitsandbytes/CUDA | `StageAVLM.from_config` returns `DummyVLM` (heuristic ASG from frames) |
| diffusers                     | `VAEWrapper` falls back to `IdentityVAE` (1→1, no downscale)            |
| lpips                         | `metrics.lpips_metric` falls back to `1 − SSIM`                         |
| pysteps                       | numpy phase-correlation motion + numpy warp advection                  |
| scikit-image                  | `ndimage.label` segmentation instead of watershed                      |
| s3fs / h5py                   | `download_sevir_subset` materializes `SyntheticSEVIR` instead          |
| lm-format-enforcer            | constrained decode degrades to `grammar.parse_strict` → `parse`        |

Because of this, the **entire** label → train → eval → figures pipeline runs on CPU with the
synthetic dataset; that is the smoke path used by both notebooks before any A100 hours are spent.

---

## The 3-tier training plan, and where each tier runs

The binding constraint is **wall-clock on preemptible spot sessions**, not VRAM, so the plan is
split across hardware (`training_method.md` §2–§4):

| Tier | What it trains                                                              | Hardware (intended) | Gate / deliverable |
| ---- | --------------------------------------------------------------------------- | ------------------- | ------------------ |
| **Tier 0** | ASG→ASG transition transformer **+** deterministic (1-step) renderer  | **L4** (always-on, cheap) | `gate_check`: transition must beat **persistence** AND **future-blind advection** on object evolution — the publishable go/no-go before any A100 hour |
| **Tier 1** | Stage-A VLM (QLoRA), 5-phase curriculum ph1→ph5                       | **L4**              | **Ph-3 ASG-F1 hard gate** ≥ `train.tier1.ph3_gate_f1` (0.70); HARD-STOP (RuntimeError) below it. Ph-5 checkpoint = Tier-2 init |
| **Tier 2** | End-to-end A→B→bottleneck→C: rectified-flow renderer, scheduled sampling, intervention-consistency, ensembles | **A100-40GB** | Intervention consistency passes AND zeroed-ASG collapses to advection (verified by `41_eval_faithfulness.py`) |

The two notebooks in `notebooks/` are sized for the actual target hardware described in the task:
**Colab A100-40GB, max 2 runtimes, each < 12 h, preemptible, Drive-persistent**. Tier 0 and
Tier 1 are cheap enough that the notebooks run them on the A100 too (they would run on an L4 in a
real lab); everything checkpoints to Drive and resumes by reading `checkpoint.latest()`.

* **Runtime 1** — smoke test → download/synthesize data → autolabel → capacity audit → Tier-0 →
  Tier-1 phases ph1→ph3 (through the Ph-3 gate).
* **Runtime 2** — resume → Tier-1 ph4→ph5 → Tier-2 (chunked to fit <12 h) → skill + faithfulness
  evals → regenerate figures.

Set `paths.root` to a Google Drive folder so checkpoints/results survive a preemption; the two
notebooks chain through Drive.

---

## Running each script

Every script takes `--config` (defaults to `../configs/default.yaml` relative to the script) and a
repeatable `--override key.subkey=value`. Run from `src/` (or `src/scripts/`); each script puts
the repo root on `sys.path` itself.

```bash
# 0) data — download the SEVIR subset to paths.cache, or synthesize it (idempotent)
python scripts/00_download_data.py --config configs/default.yaml --override data.n_train_events=64

# 1) labels — pysteps ASG auto-labeling pass; one ASG-pair JSON per event -> paths.cache/asg/
python scripts/01_autolabel.py    --config configs/default.yaml [--force] [--limit N]

# 10) Tier 0 — transition + deterministic renderer + go/no-go gate
python scripts/10_train_tier0.py  --config configs/default.yaml \
      [--override train.tier0.max_steps=10] [--resume <ckpt>] [--skip-renderer]

# 20) Tier 1 — five-phase VLM curriculum with the Ph-3 ASG-F1 hard gate
python scripts/20_train_tier1_curriculum.py --config configs/default.yaml
#     run a single phase (resumes from a given checkpoint):
python scripts/20_train_tier1_curriculum.py --phase ph4_cot --resume <ph3_ckpt>
#     valid --phase values: ph1_vqa ph2_desc ph3_asg ph4_cot ph5_eqcot

# 30) Tier 2 — end-to-end A->B->bottleneck->C (checkpoint/resume aware)
python scripts/30_train_tier2.py  --config configs/default.yaml \
      --vlm-ckpt <ph5_ckpt> --transition-ckpt <tier0_transition_ckpt> [--resume <tier2_ckpt>]

# 40) skill eval  -> paths.results/skill_results.json        (A: CSI/HSS/FSS/CRPS/LPIPS/PSD; B: regime; F: lead-time)
python scripts/40_eval_skill.py        --config configs/default.yaml

# 41) faithfulness eval -> paths.results/faithfulness_results.json  (C-i..C-v + capacity audit/sweep)
python scripts/41_eval_faithfulness.py --config configs/default.yaml

# 42) figures -> paths.results/figures/*.{pdf,png}  (regenerated from the result JSON above)
python scripts/42_make_figures.py      --config configs/default.yaml
```

Recommended ordering: `00 → 01 → 10 → 20 → 30 → 40 → 41 → 42`. Scripts 40/41/42 are robust to a
partially-trained codebase (they fall back to a deterministic synthetic eval set / contract-honouring
stub renderer) so the eval + figure path runs immediately.

### Synthetic-data smoke path (no download, no GPU)

To prove the wiring end-to-end on a laptop before spending Colab hours, shrink everything via
overrides — the smoke cell in `notebooks/run_all_colab_A100_runtime1.ipynb` does exactly this:

```bash
python scripts/00_download_data.py --override data.n_train_events=16
python scripts/01_autolabel.py     --override data.n_train_events=16 --limit 16
python scripts/10_train_tier0.py   --override train.tier0.max_steps=4 \
                                   --override train.tier0.renderer_max_steps=4 \
                                   --override train.tier0.ckpt_every=2
python scripts/20_train_tier1_curriculum.py \
    --override train.tier1.steps_per_phase.ph1_vqa=2 \
    --override train.tier1.steps_per_phase.ph2_desc=2 \
    --override train.tier1.steps_per_phase.ph3_asg=2
python scripts/40_eval_skill.py
python scripts/41_eval_faithfulness.py
python scripts/42_make_figures.py
```

With s3fs/h5py absent, `00` writes a `SyntheticSEVIR` subset (deterministic moving-Gaussian VIL
blobs, seeded by `cfg.seed`); with CUDA/transformers absent, Tier-1 uses `DummyVLM`; with diffusers
absent, the renderer uses `IdentityVAE`. The Ph-3 gate runs against a synthetic gold subset when no
hand-labeled `paths.gold_subset` is present.

---

## The config system

`configs/default.yaml` is the single source of truth (read it for exact keys). `asgwm/utils/config.py`
loads it into a `Config` (a `dict` with attribute + dotted access):

```python
from asgwm.utils.config import load_config
cfg = load_config("configs/default.yaml", ["train.tier0.max_steps=10", "paths.root=/content/drive/MyDrive/asgwm"])
cfg.get_path("train.tier1.ph3_gate_f1", 0.70)   # -> 0.70
cfg.set_path("asg.n_max", 8)                     # used by the capacity sweep
```

Overrides are `key.subkey=value` strings; values are coerced (`int → float → bool → None → str`).
Nested keys like `train.tier1.steps_per_phase.ph1_vqa=2` work. Key config groups:

* `paths.*` — `root` (point at the Drive mount), `sevir_raw`, `cache`, `checkpoints`, `results`,
  `gold_subset`. Everything persists under `paths.root`.
* `data.*` — grid 384, 1 km/px, 5 min/frame, 13 history / 36 horizon frames, channels, patch 128.
* `asg.*` — `n_max=16` (the IB cap), motion/growth quantization, growth-field size 48.
* `stage_a.*` — VLM backbone (`SmolVLM-2.2B-Instruct`, alt `Qwen2.5-VL-3B`), LoRA r/alpha,
  4-bit, ASG/NL loss weights.
* `stage_b.*` — transition transformer dims, `predict_residual`, continuity/smoothness weights.
* `stage_c.*` — VAE, latent channels, U-Net base, `flow_steps=4`, `ensemble_k=10`.
* `bottleneck.*` / `losses.*` — IB penalty + Tier-2 loss weights.
* `train.tier0/1/2.*` — batch sizes, LRs, step budgets, `ckpt_every`, the Ph-3 gate, scheduled
  sampling / oracle anneal.
* `eval.*` — thresholds, FSS scales, lead times, regimes, capacity-sweep N_max grid, intervention
  types, spatial/intensity tolerances.

---

## Evaluation → script map

The eval suite mirrors `eval.md`. A/B/F live in `40_eval_skill.py`; C-i..C-v and the capacity
audit/sweep live in `41_eval_faithfulness.py`; D is the held-out generalization protocol applied
to the same metrics. Figures regenerate from the result JSON via `42_make_figures.py`.

| Eval | What it measures | Module function(s) | Script | Output |
| ---- | ---------------- | ------------------ | ------ | ------ |
| **A** Skill (aggregate) | CSI/HSS/POD/FAR, FSS, pooled-CSI, CRPS (ensemble), LPIPS realism, PSD | `eval.metrics.{csi,hss,pod,far,fss,pooled_csi,crps_ensemble,lpips_metric,psd}` | `40_eval_skill.py` | `skill_results.json → aggregate` |
| **B** Regime-stratified skill | metrics split by `init/grow/decay/steady` | `eval.regime_eval.{stratify_by_regime,regime_skill_table}` | `40_eval_skill.py` | `skill_results.json → regime_table` |
| **C-i** Intervention consistency | rendered field reacts to ASG edits within spatial/intensity tol | `eval.faithfulness.intervention_consistency` (+ `interventions.{perturb_asg,expected_effect}`) | `41_eval_faithfulness.py` | `faithfulness_results.json → intervention` |
| **C-ii** Bottleneck ablation | oracle vs inferred vs **zeroed** vs **shuffled** ASG | `eval.faithfulness.bottleneck_ablation` (uses `bottleneck.zero_asg_in_Z`) | `41_eval_faithfulness.py` | `faithfulness_results.json → ablation` |
| **C-iii** Leakage audit | CLUB MI upper bound: does `advect_blind` leak the future? | `eval.faithfulness.{LeakageCLUB,leakage_audit}` | `41_eval_faithfulness.py` | `faithfulness_results.json → leakage` |
| **C-iv** ASG accuracy | obj-F1 (Hungarian on centroids), motion angle err, regime acc vs gold | `eval.faithfulness.asg_accuracy` | `41_eval_faithfulness.py` (and the Tier-1 Ph-3 gate) | `faithfulness_results.json → asg_accuracy` |
| **C-v** Counterfactual demo | base vs edited fields + per-edit diffs | `eval.faithfulness.counterfactual_demo` | `41_eval_faithfulness.py` | `faithfulness_results.json → counterfactual_demo_l1` |
| **D** Generalization | A/B/C metrics on a held-out region/season split | same A/B/C functions on the held-out set | `40_/41_` (held-out `--override`) | same JSON keys |
| **F** Lead-time skill | metric vs lead time curve (30..180 min) | assembled from A metrics per lead time | `40_eval_skill.py` | `skill_results.json → leadtime` |
| **Capacity** | `asg_bits << input_bits` audit + N_max sweep | `eval.capacity.{capacity_bits,capacity_audit,capacity_sweep}` | `41_eval_faithfulness.py` (audit is also the Runtime-1 go/no-go) | `faithfulness_results.json → capacity_audit / capacity_sweep` |

Figure regeneration (`42_make_figures.py` → `paths.results/figures/`): `fig_regime` (from B),
`fig_leadtime` (from F), `fig_faith` (from C-i + C-ii), `fig_capacity` (from the capacity audit/sweep),
`fig_forecaster` (human forecaster study — `eval.md` §1E, gated on a partner, so a clearly-labeled
placeholder is written until real data arrives). All figures use the Google palette
(`#4285F4 #EA4335 #FBBC04 #34A853`) via `asgwm/utils/viz.py`.

> **E (forecaster study)** and the comparison **baselines** are deliberately not implemented yet —
> this repo is the framework. Eval scripts emit synthetic/placeholder comparison rows so the figure
> pipeline runs today and is swapped to real numbers when those land.
