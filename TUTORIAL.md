# ASG-WM — step-by-step run guide (Colab A100 → L4 VM)

This walks you from zero to **finalized figures, tables, and PDF**, using two Colab A100
sessions first, then your always-on L4 VM. Everything reads/writes one shared folder
(`$ROOT`) on Google Drive (or a bucket), so the VM picks up exactly what Colab produced.

```
                ┌──────────────── Google Drive / GCS bucket = $ROOT ────────────────┐
 Colab A100  ──►│  events · ASG labels · checkpoints · results · figures · tables    │◄── L4 VM
 (2× ≤12 h)     └───────────────────────────────────────────────────────────────────┘   (always-on)
   train + first eval/figures                                  full labels · ablations · baselines · final figures
```

> **Golden rule:** always pass `--override paths.root=$ROOT`. It rebases every cache /
> checkpoint / result path onto `$ROOT`, so a dropped Colab session resumes from Drive.

---

## 0. Validate the wiring first (synthetic, no real data needed)

Confirm the whole pipeline runs before spending real GPU time. **You must shrink the grid** —
at the default 384$^2$, Stage-C trains at full resolution and bogs down (minutes/step on CPU;
also slow with the identity-VAE fallback). The smoke uses a tiny grid so it's quick.

> The single biggest speedup is installing `diffusers` (real 8$\times$ SD-VAE → Stage-C runs in
> a downscaled latent) and using your GPU. With `device: auto` (default) the code picks CUDA
> automatically; `train.precision=bf16` enables mixed precision on the RTX 4060.

**GPU smoke (RTX 4060, ~1–2 min) — recommended on your laptop:**

```bash
cd src
pip install -r requirements.txt          # installs CUDA torch + diffusers (real VAE) + deps
G="--override data.dataset=synthetic --override paths.root=./_smoke --override device=cuda --override train.precision=bf16 --override data.grid=256 --override data.in_frames=6 --override data.out_frames=12 --override data.patch=128"
python scripts/00_download_data.py          $G --override data.n_train_events=8
python scripts/01_autolabel.py              $G
python scripts/10_train_tier0.py            $G --override train.tier0.max_steps=50 --override train.tier0.renderer_max_steps=50 --override train.tier0.batch_size=8
python scripts/40_eval_skill.py             $G --override eval.n_eval_events=6
python scripts/41_eval_faithfulness.py      $G --override eval.n_eval_events=6
python scripts/42_make_figures.py --gallery $G
pytest -q
```

**CPU-only smoke (no GPU, ~8–10 min) — minimal sizes:**

```bash
cd src
C="--override data.dataset=synthetic --override paths.root=./_smoke --override data.grid=64 --override data.in_frames=4 --override data.out_frames=8 --override data.patch=48"
python scripts/00_download_data.py          $C --override data.n_train_events=6
python scripts/01_autolabel.py              $C
python scripts/10_train_tier0.py            $C --override train.tier0.max_steps=6 --override train.tier0.renderer_max_steps=6 --override train.tier0.batch_size=4
python scripts/40_eval_skill.py             $C --override eval.n_eval_events=4
python scripts/41_eval_faithfulness.py      $C --override eval.n_eval_events=4
python scripts/42_make_figures.py --gallery $C
pytest -q
```

Expect `69 passed` and figures/tables under `_smoke/results/`. Pre-training, **ASG-WM ties
pysteps** (both fall back to advection until Stage-C is trained) — that is the correct,
honest result, not a bug. Delete `_smoke/` afterwards.

> **RTX 4060 VRAM note (8 GB):** great for the smoke, Tier-0, Tier-1 (QLoRA VLM in 4-bit),
> and all evaluation/inference. **Full Tier-2** end-to-end + intervention training peaks at
> ~20–28 GB — that does **not** fit 8 GB; run real Tier-2 on the Colab A100 (Part A). You can
> still smoke-test Tier-2 locally at tiny sizes.

---

## Part A — Colab A100 (two sessions, ≤12 h each)

### A0. One-time setup at the top of EACH session

```python
from google.colab import drive; drive.mount('/content/drive')
ROOT = '/content/drive/MyDrive/asgwm'        # shared with the VM; survives disconnects
%env ASGWM_ROOT={ROOT}

# get the code onto the machine (pick one):
#  (a) upload the local `src/` folder to {ROOT}/src once, then:
import sys; sys.path.insert(0, f'{ROOT}/src')
%cd {ROOT}/src
#  (b) or clone a private repo:  !git clone <your-repo> {ROOT}/repo  ; %cd {ROOT}/repo/src

!pip -q install -r requirements.txt
!nvidia-smi -L           # confirm an A100-40GB
```

The fastest route is the root **`train.ipynb`** then **`eval.ipynb`** (each is self-contained:
clone → install → download → train/eval → zip the artifacts). Otherwise you can **open the
prebuilt notebooks** (`src/notebooks/run_all_colab_A100_runtime1.ipynb`, `…runtime2.ipynb`) and
Run-All, or paste the commands below (run them from inside `src/`).

### A1. Runtime 1 — data → labels → Tier-0 → Tier-1 (gate)

```bash
R="--override paths.root=$ASGWM_ROOT"
# 1) data subset + ASG auto-labels (cache once to Drive)
python scripts/00_download_data.py $R --override data.n_train_events=1200
python scripts/01_autolabel.py     $R                 # pysteps tracking; CPU-bound
# 2) capacity audit (go/no-go: ASG bits << input bits)
python -c "from asgwm.utils.config import load_config; from asgwm.eval.capacity import capacity_audit; print(capacity_audit(load_config('configs/default.yaml',['paths.root=$ASGWM_ROOT'])))"
# 3) Tier-0: transition + deterministic renderer (+ gate vs persistence/advection)
python scripts/10_train_tier0.py   $R
# 4) Tier-1: VLM curriculum Ph-1..Ph-3 (HARD GATE: ASG F1 >= 0.70 before Ph-4)
python scripts/20_train_tier1_curriculum.py $R --override train.tier1.phases='["ph1_vqa","ph2_desc","ph3_asg"]'
```

Everything is checkpoint/resume-safe; if the session drops, just re-run the same command —
it resumes from the latest checkpoint in `$ROOT/ckpt`. **End of runtime 1:** confirm
`$ROOT/ckpt/tier1/ph3_asg/` exists and the Ph-3 gate passed.

### A2. Runtime 2 — finish Tier-1 → Tier-2 → eval → figures

```bash
R="--override paths.root=$ASGWM_ROOT"
# 5) Tier-1: Ph-4, Ph-5 (resumes from the Ph-3 checkpoint)
python scripts/20_train_tier1_curriculum.py $R --override train.tier1.phases='["ph4_cot","ph5_eqcot"]'
# 6) Tier-2: end-to-end + rectified-flow renderer + intervention-consistency (the A100 job)
python scripts/30_train_tier2.py $R
# 7) evaluation (ours; baselines still TBR) + faithfulness + capacity
python scripts/40_eval_skill.py        $R
python scripts/41_eval_faithfulness.py $R
# 8) regenerate the data figures + the gallery from real results
python scripts/42_make_figures.py --gallery $R
```

**End of runtime 2:** `$ROOT/results/figures/*.pdf` and `$ROOT/results/tables/*.tex` now hold
the real ASG-WM results. Everything is already on Drive for the VM.

> Tier-2 cumulative training may exceed one 12 h session. It checkpoints every
> `train.tier2.ckpt_every` steps — just start a third A100 session and re-run step 6; it
> resumes. Shrink `train.tier2.batch_size` / `data.patch` if you approach 40 GB.

---

## Part B — L4 VM (always-on, picks up from Drive)

The VM does the cheap-but-long and the breadth work: full-scale labels, ablations,
cross-region, the five baselines, and the final figure/table regeneration.

### B0. Setup (once)

```bash
# mount the SAME Drive folder / bucket as $ROOT, then:
cd src && pip install -r requirements.txt
export ROOT=/mnt/asgwm        # <-- your mount of the shared folder
R="--override paths.root=$ROOT"
nvidia-smi -L                 # confirm L4-24GB
```

### B1. Full-scale auto-labeling (CPU, the big one-time job)

```bash
python scripts/00_download_data.py $R --override data.n_train_events=2500
python scripts/01_autolabel.py     $R          # hours on CPU; cached to $ROOT — run once
```

### B2. Knowledge-type ablations — Group D (specs/eval.md §D)

Compare Ph-4 vs Ph-5 checkpoints and toggle context/equation components, e.g.:

```bash
python scripts/40_eval_skill.py $R --override stage_b.lambda_continuity=0     # − continuity
python scripts/40_eval_skill.py $R --override asg.context_fields='["cape","cin"]'  # − geo/seasonal ctx
# (collect the runs into Table 3; matched-benchmark splits per specs/eval.md §D)
```

### B3. Out-of-distribution generalization — Group F (real radar)

NEXRAD Level II and MRMS are used **entirely** as OOD test sets (no training, no split). Both
download **real data only** (no synthetic fallback) and are bridged into the canonical VIL byte
representation via the dBZ→VIL approximation in `asgwm.data.normalize` — see `datasets/README.md`.

```bash
python scripts/00_download_data.py $R --override data.dataset=nexrad     # or mrms
python scripts/40_eval_skill.py    $R --override data.dataset=nexrad
python scripts/41_eval_faithfulness.py $R --override data.dataset=nexrad
```

### B4. Implement + run the five baselines (ours-now → all-later)

Each baseline is a slot in `asgwm/baselines/adapters.py`. Implement `predict()` and flip
`is_available()` for **pysteps (done) · RainNet · NowcastNet · LangPrecip · ThoR**:

```python
# asgwm/baselines/adapters.py
class RainNetBaseline(Baseline):
    name, display, family = "rainnet", "RainNet", "CNN / U-Net"
    def is_available(self): return True
    def predict(self, frames_hist, context, n_out):
        ...  # load your trained RainNet, autoregress n_out frames, return [n_out,H,W]
```

Once available, they automatically appear in every figure and table — no other change.
See `src/RESULTS.md` → "To add a baseline later".

### B5. Regenerate all figures + tables with baselines filled

```bash
python scripts/40_eval_skill.py        $R       # now fills RainNet/NowcastNet/... rows
python scripts/41_eval_faithfulness.py $R
python scripts/42_make_figures.py --gallery $R
```

### B6. (optional) symbolic admissibility prototype

```bash
python scripts/43_admissibility_demo.py $R      # certificate + dual-SAT ambiguity audit
# NOTE: first fix object-ID stability in asgwm/labeling/tracking.py (see RESULTS/known issues)
```

---

## Part C — Finalize the paper

```bash
# copy the real figures/tables into the manuscript folder
cp $ROOT/results/figures/fig_regime.pdf      paper/
cp $ROOT/results/figures/fig_leadtime.pdf    paper/
cp $ROOT/results/figures/fig_faith.pdf       paper/
cp $ROOT/results/figures/fig_capacity.pdf    paper/
cp $ROOT/results/figures/fig_forecaster.pdf  paper/
cp $ROOT/results/figures/fig_case.pdf        paper/        # qualitative gallery
# tables: paste $ROOT/results/tables/skill.tex + compute.tex into the Table 2 / Table 4 bodies
# (schematic figs — knowledge/framework/architecture/renderer — are already final SVG→PDF)

cd paper
pdflatex -interaction=nonstopmode paper.tex && bibtex paper && \
pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex
```

`specs/FIGURES.md` lists which figures are data-driven (regenerated above) vs schematic
(authored once). `src/RESULTS.md` is the full results-pipeline reference.

---

## Command reference

All scripts live in `src/scripts/` and read `src/configs/default.yaml` (run them from inside `src/`).

| Script | Does | Where | Notes |
|---|---|---|---|
| `00_download_data.py` | SEVIR / synthetic + cache (NEXRAD/MRMS for OOD) | both | `data.n_train_events`, `data.dataset={sevir,synthetic,nexrad,mrms}` |
| `01_autolabel.py` | pysteps → ASG labels | VM (CPU) | run once, cache to `$ROOT` |
| `10_train_tier0.py` | transition + det. renderer + gate | A100 / L4 | de-risking go/no-go |
| `20_train_tier1_curriculum.py` | 5-phase VLM (QLoRA) | A100 / L4 | Ph-3 gate F1≥0.70; `train.tier1.phases=[...]` to split |
| `30_train_tier2.py` | end-to-end + flow + intervention | **A100** | ~20–28 GB; resumable |
| `40_eval_skill.py` | skill + regime + leadtime + tables | both | multi-method; baselines TBR until added |
| `41_eval_faithfulness.py` | C-i…C-v + capacity | both | ASG-WM only |
| `42_make_figures.py --gallery` | all data figures | both | reads `$ROOT/results/*.json` |
| `43_admissibility_demo.py` | symbolic certificate (prototype) | VM | optional |

## Resume & troubleshooting

- **Session dropped?** Re-run the same command — every trainer resumes from `$ROOT/ckpt/<job>/`.
- **Ph-3 gate fails (F1 < 0.70)?** Real signal: fix the visual projector / labels before Ph-4
  (specs/training_method.md §3). For a wiring smoke only, add `--override train.tier1.ph3_gate_f1=0.0`.
- **OOM on Tier-2?** lower `train.tier2.batch_size` or `data.patch`; keep `train.precision=bf16`.
- **Figures missing a baseline row?** That baseline isn't implemented yet (`is_available()==False`)
  — it shows as TBR by design until you add it (Part B4).
- **Nothing writes to Drive?** You forgot `--override paths.root=$ROOT`.
