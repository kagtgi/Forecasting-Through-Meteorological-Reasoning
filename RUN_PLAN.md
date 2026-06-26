# RUN_PLAN — read this before the full (paid) run

*Pre-run battle plan, 2026-06-22. Companion to `TUTORIAL.md` (which is the step-by-step how-to).
This doc is the **decision layer**: what is already proven, what to watch, and when to stop.*

The whole pipeline was just **run end-to-end on the local RTX 4060** (synthetic data + DummyVLM,
since real SEVIR/VLM need `s3fs`/`transformers` not installed locally). That run did two things:
confirmed the orchestration is sound, and **caught a real bug** (below). Treat the Colab run as
"scale up a verified pipeline", not "hope it works".

---

## 1. What is already verified (local, GPU)

| Claim | Evidence |
|---|---|
| Full script chain runs clean: `00 → 01 → 10 → 20 → 30 → 40 → 41 → 42` | ran end-to-end, all exit 0 |
| Tier-1 curriculum chains ph1→ph5, **cross-phase checkpoint seeding** works | seeds visible in log |
| Faithful bottleneck is wired: **zeroing the ASG collapses skill to 0.0** (oracle/inferred = 0.479) | `41` C-ii ablation |
| **Stage B (transition) optimiser path** is healthy | overfit one batch **309×** in 150 steps |
| **Stage C (renderer) optimiser path** is healthy; flow-matching descends | overfit one batch **65×** / 200 steps |
| Tier-2 instrumentation (warmup ramp, per-term log, NaN-abort) works on GPU | `w` ramps 0→1, `p_oracle` 1→0 |
| Results → JSON → LaTeX tables → figures pipeline produces all artifacts | `results/{tables,figures}` written |
| Baselines stay **`[TBR]`** (no fabricated numbers) | `40` skill table |
| Pre-flight = **GO** (9/9 checks) | `99_preflight.py` |

**The bug the local run caught (now fixed):** `physics.mass_budget_residual` was an *unnormalised*
squared sum, `(Σpixels − budget)²` ≈ **1e10**, while the reconstruction MSE was ≈ **1e3** — the mass
term dominated the loss by ~6 orders of magnitude and grows as resolution², so at the real 384 grid
it would be ~1e13 and the renderer would learn *total content only*, ignoring structure. Fixed to a
per-pixel residual (commensurate with the MSE, resolution-invariant). After the fix, Tier-2 `total`
dropped from ~1e9 to ~1e3 and is reconstruction-dominated. **This alone would likely have wasted a
paid session.**

## 2. What is NOT verified locally (and why)

- **Real SEVIR data loader** — needs `s3fs`; and `_download_real_sevir` does `fs.get(whole_file)`,
  and SEVIR HDF5 files are **multi-GB each** (they pack hundreds of events), so a real probe pulls
  tens of GB. Not a "quick local test." → **Validated on Colab by the pre-flight on small `n`**
  (step 2 below) *before* the big run.
- **Real VLM (SmolVLM/Qwen2.5-VL) QLoRA** — needs `transformers`/`peft`/`bitsandbytes` and >8 GB;
  the 4060 ran the `DummyVLM` fallback. → First real exercise is Tier-1 on Colab.

---

## 3. The run, in order (Colab Pro, A100-40GB)

The simplest path is the root **`train.ipynb`** (clone → install → download → train all tiers →
`train_results.zip`) followed by **`eval.ipynb`** (eval ours → figures/tables → `paper_assets.zip`).
For the explicit two-session split, open `src/notebooks/run_all_colab_A100_runtime1.ipynb`, set
runtime = **A100**, then **Runtime → Run all**. Two toggles in the Config cell:

- `FIRST_RUN = True` → time-boxed (`max_steps_per_phase=4000`, `tier2.max_steps=12000`); fits two
  A100 sessions. Set `False` only for the final publication-budget run.
- `N_EVENTS = 800` → start here; raise for the final run.

**Sequence & where the time goes:**

| Step | Cell | Cost driver | Note |
|---|---|---|---|
| Pre-flight (REAL) | RT1 §2 | catalog + a couple HDF5 files (minutes) | **must print GO** |
| Data + ASG labels | RT1 §3 | **whole SEVIR HDF5 files, GB each** — the slow/heavy step | cached to Drive; reused after |
| Capacity audit | RT1 §4 | seconds | go/no-go |
| Tier-0 | RT1 §5 | minutes | canary (see gate 3) |
| Tier-1 ph1→ph3 | RT1 §6 | bulk of RT1 | **hard F1 gate** |
| Tier-1 ph4→ph5 | RT2 §3 | | seeds from ph3 ckpt |
| Tier-2 | RT2 §4 | bulk of RT2 | watch the `[tier2]` log |
| Eval + figures | RT2 §5–6 | minutes | brings results home |

The two notebooks are designed to each finish **< 12 h**; the **data download is the wildcard** —
if Drive I/O is slow, the first RT1 run may be download-bound. It is cached, so a re-run is fast.

---

## 4. The five decision gates — do not skip

1. **Pre-flight = GO on real data.** `src/scripts/99_preflight.py … data.require_real=true`. If `data load`
   FAILs → fix S3/`s3fs`/schema; **never** let it fall back to synthetic for a paid run.
2. **Label-audit F1 ≥ 0.70 vs gold** (printed by the pre-flight). *This is the cheapest, highest-value
   check.* If it is below 0.70, **stop and fix the auto-labeller** — the Ph-3 gate is mathematically
   unreachable and Tier-1 would burn hours to fail. (No real gold set yet? Build a small hand-labeled
   one first — `paths.gold_subset`. Without it the gate runs against a weak synthetic fallback.)
3. **Tier-0 canary:** Tier-0 should beat persistence/optical-flow on the val set. If it can't, the
   data or transition is wrong — debug before spending on the VLM.
4. **Ph-3 F1 gate ≥ 0.70.** The curriculum *raises* if it fails. Do **not** lower the threshold for a
   real run (we set it to 0 locally only because DummyVLM can't produce real ASGs).
5. **Tier-2 first ~200 steps:** in the `[tier2]` log, `total` should trend down, **no NaN**, and
   `intervene(w=…)` should ramp 0→1 smoothly. If `total` explodes → lower `train.tier2.lr` (the
   NaN-abort will stop you automatically); if OOM → see triage.

---

## 5. If it fails — triage

| Symptom | Cause | Fix |
|---|---|---|
| Pre-flight `data load` FAIL | s3fs/schema/network | `pip install s3fs`; check bucket; do **not** run synthetic |
| Label-audit F1 < 0.70 | auto-labeller too noisy | tighten labeller (fewer, higher-precision objects); build/extend gold set |
| Ph-3 gate raises | perception not learning | inspect ph3 samples; check projector/LR; do not force past |
| Tier-2 OOM (A100) | 3 grad-tracked rollouts/step | `data.patch=96`, `tier2.batch_size=4`, `tier2.intervene_every=2` |
| Tier-2 NaN / explodes | lr / intervention too hot | lower `tier2.lr` to 5e-6; raise `intervene_warmup_steps` |
| Tier-2 too slow | intervention cost | `tier2.intervene_every=2` (halves the 2-render cost) |
| High-threshold CSI weak | SD-VAE radar domain gap | fine-tune VAE decoder on VIL (FEASIBILITY §5) — a *quality* lever, not a blocker |

All the knobs above already exist in `src/configs/default.yaml`.

---

## 6. Budget sketch

- **FIRST_RUN, N_EVENTS=800:** designed for ≤ 2 × A100 sessions (≤ ~24 A100-hours incl. the
  one-time data download). The download dominates RT1 the *first* time only.
- **Full publication run** (`FIRST_RUN=False`, larger N): more Tier-1/Tier-2 steps; budget a third
  session if you raise N_EVENTS substantially. The eval/figures step is cheap.
- **Cheapest insurance:** the pre-flight + label-audit cost minutes and gate the expensive hours.
  Run gate 1–2 in a *fresh free/low-cost runtime first* if you want to validate data before paying
  for the A100.

## 7. After Colab — what is left for the L4 24GB VM

- Re-run **eval + figures** (`40/41/42`) on the trained checkpoints if you tweak metrics/plots
  (no GPU-heavy training needed; L4 is plenty).
- Train/insert the **5 baselines** (pysteps real now; RainNet/NowcastNet/LangPrecip/ThoR are stubs)
  — each fits an L4; they slot into the baseline registry and the `[TBR]` rows fill in.
- Optional **VAE-decoder fine-tune** on VIL (quality lever) — fits an L4.
- **OOD generalization test** on real radar: re-run `40_eval_skill.py` / `41_eval_faithfulness.py`
  with `data.dataset=nexrad` or `data.dataset=mrms` (both download real data only, no synthetic
  fallback, and are bridged into VIL byte under the dBZ→VIL approximation — see `datasets/README.md`).
- Tier-2 itself wants ~20–28 GB peak at patch 128 → A100; on the L4 use `patch=96`, `batch_size=4`.

---

### TL;DR
Pipeline is verified end-to-end locally and a loss-scale bug is fixed. On Colab: **Run all** on
Runtime 1, **require GO + label-F1 ≥ 0.70 + Ph-3 gate**, watch the `[tier2]` log for the first 200
steps, then Runtime 2. The gates are the budget protection — respect them and the run is safe.
