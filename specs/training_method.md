# training_method.md — Tiered recipe on an always-on L4 + 2–3 spot A100-40 GB sessions

> **Principle:** decouple the stages, front-load the cheap thesis-critical experiment, and spend an A100 session only once it passes. Your envelope — **L4-24 GB always available** + **2–3 Colab A100-40 GB sessions at <12 h each** — is sufficient for the **whole** plan, because every component is engineered to ≤24–40 GB. The binding constraint is **wall-clock + ephemeral storage**, not VRAM; §6 handles it.

---

## 1. The three tiers (what each proves, where it runs)

| Tier | What it proves | Model | Where it runs | Out-of-pocket |
|---|---|---|---|---|
| **Tier 0 — Thesis** | "An explicit world model beats persistence/advection at predicting growth/decay/initiation." | pysteps perception (classical) → small **transition transformer** (ASG→ASG) → **deterministic** advection + small-U-Net renderer | **L4** (or Kaggle T4×2 / P100) | **$0** |
| **Tier 1 — VLM value** | "Vision-language priors **and** explicit physics equations improve perception + transition." | Swap in **SmolVLM-2.2B / Qwen2.5-VL-3B** (QLoRA-4-bit) for perception+rationale; add HRRR/ERA5 context + AFD grounding + physics constraints | **L4** (always-on) | **$0** |
| **Tier 2 — Full result** | Faithfulness, probabilistic skill, cross-region, realism. | **Rectified-flow** renderer (few-step), intervention-consistency training, ensembles, cross-dataset | **A100-40 GB** spot sessions (×2–3, <12 h) | **$0 on Colab credits** |

**Why Tier 0 is the win:** the scientific heart — does reasoning over an explicit state resolve the ambiguity — is a tiny transformer over scene-graph tokens with CPU-generated labels. It trains in minutes–hours on the L4 and is a clean **go/no-go before any A100 session**. If it fails, the thesis is wrong and you've spent nothing.

---

## 2. Tier 0 — transition + deterministic renderer (L4, $0)

- Train the **ASG→ASG transition transformer** on cached pysteps labels.
- **Gate:** beats persistence **and** pure-advection on object evolution (position error, regime-transition accuracy, growth/decay sign). This is the publishable de-risking result.
- Train the **deterministic renderer** (advection-warp + small U-Net residual) on **oracle ASG** to confirm pixel parity is reachable (answers "VLMs can't make precise fields" — the renderer can, from a good state).
- bf16, gradient checkpointing. Fits the L4 with room to spare; individual runs finish well inside one session.

---

## 3. Tier 1 — VLM perception + knowledge injection (L4, $0)

- **QLoRA** the VLM (`dettmers2023qlora`, `hu2022lora`) for ASG_t + NL: NF4 4-bit base, freeze the backbone, train LoRA adapters (r≈16–32) + the modality projector only. A 2–3 B VLM with QLoRA sits comfortably in 24 GB (QLoRA enables 33 B finetuning on a single 24 GB GPU; a 3 B VLM is far below that).
- **Inject knowledge of two kinds** and design the ablation (`eval.md` §D) to separate them:
  - **(a) NL meteorological priors** — the VLM's pretraining, AFD environment text, textbook context.
  - **(b) Physics equations** — advection, continuity/mass-conservation, growth–decay — entered three ways (see `architecture.md` §3): a differentiable **advection operator**, **PINN-style residual losses** (`raissi2019pinn`), and **equation-aware prompts**. All compute-cheap (loss terms + a warp, no extra parameters).
- Cache frozen visual features to the bucket so re-runs skip the vision tower.
- Expect the L4 to be ~2–4× slower than a 4090 on compute-bound steps (lower bandwidth, no NVLink); with the subset + QLoRA this still finishes in one session, occasionally two with a checkpoint.

**Five-phase curriculum** (see `architecture.md` §10 and `datasource.md` §5). Run sequentially from the previous phase's checkpoint:

| Phase | Task | Est. L4 time | Gate |
|---|---|---|---|
| Ph-1 Visual VQA | Radar grounding | 1–2 h | VQA accuracy > 80% on held-out |
| Ph-2 Object description | ASG-faithful NL | 1–2 h | — |
| Ph-3 Structured ASG | Emit ASG grammar | 2–4 h | **ASG F1 ≥ 0.70 on gold subset** |
| Ph-4 CoT reasoning | Rationale chain | 3–6 h | Transition accuracy on held-out |
| Ph-5 Equation-aware CoT | Physics-grounded reasoning | 2–4 h | Equation-grounding rate > 80% |

Total Tier-1 curriculum: ~9–18 h on the L4 across checkpoint-resumed sessions. **Ph-3 gate is the hard decision point**: if ASG F1 < 0.70, debug the visual projector or data pipeline before proceeding — the downstream CoT is unfounded without a reliable state. The Ph-5 checkpoint is the Tier-1 deliverable and the Tier-2 initialization. Run the §D ablations (±NL priors, ±physics equations) by comparing Ph-4 and Ph-5 checkpoints on the held-out validation set.

---

## 4. Tier 2 — end-to-end + faithfulness (A100-40 GB spot sessions)

Couple A→B→bottleneck→**rectified-flow renderer** (`liu2023rectifiedflow`; latent via SD-VAE `rombach2022ldm`; few-step / consistency option `song2023consistency`). Training detail:

- **Low-LR / stop-grad on the VLM** (freeze in 4-bit to save memory; the perception adapters are already trained in Tier 1).
- **Scheduled sampling oracle → inferred ASG**: start the renderer on oracle ASGs, anneal to Stage-B-inferred ASGs, so errors compose gracefully.
- **Intervention-consistency loss** (`architecture.md` §6): paired (original, perturbed) forward passes; the field *difference* must match the perturbation's predicted effect. This is the make-or-break training signal for faithfulness and the source of the C-i metric.
- **Ensembles** for CRPS/reliability: sample the few-step flow `K` times (cheap at 1–4 steps).
- **Cross-region** fine-tune/transfer for F.

**Loss stack (Tier 2):**
$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{render}}}_{\text{field recon (latent)}} + \lambda_1\,\mathcal{L}_{\text{IB}} + \lambda_2\,\mathcal{L}_{\text{intervene}} + \lambda_3\,\mathcal{L}_{\text{mass}} + \lambda_4\,\mathcal{L}_{\text{nonneg}} + \lambda_5\,\mathcal{L}_{\text{spectral}} + \lambda_6\,\mathcal{L}_{\text{continuity}}$$
where `L_IB` is the bottleneck/compression term (`tishby1999ib`, `alemi2017vib`), `L_intervene` the intervention-consistency term, `L_mass`/`L_nonneg`/`L_spectral` the field-side physics+realism terms, and `L_continuity` the Stage-B PINN residual.

**IB compression term — concrete implementation** (resolves the bottleneck-capacity audit in `eval.md` §4):

1. **Hard structural cap** (primary, no extra parameters): `N_max = 16` storm objects per ASG; motion vectors quantized to 8 km/h bins; growth scalar `g` rounded to 2 significant figures; regime is 4-class categorical. This bounds the ASG's channel capacity to a finite, inspectable value equivalent to running the IB objective in a hard-compression regime.

2. **Soft variational penalty** (secondary, `λ_IB`-weighted KL term): KL divergence on continuous attributes (sub-pixel centroid offset, peak intensity deviation from the quantized grid) against a unit Gaussian prior, weighted by `λ_IB = 0.01`. Prevents continuous sub-fields from memorizing residual input detail that bypasses the structural cap. Set `λ_IB` low enough not to distort attribute values, high enough to suppress memorization.

**Capacity audit** (run once after Tier-0, before spending any A100 time): compute the theoretical channel capacity of the hard-capped ASG — (N_max objects) × (attribute entropy per object) × (prediction horizons) — and confirm it is strictly smaller than the channel capacity of the raw radar input — (H × W × k frames × pixel entropy). If ASG capacity ≥ input capacity, reduce N_max or coarsen quantization before Tier-2. Validate the chosen cap by confirming C-ii (zeroed-ASG → advection collapse) holds.

**Gate (make-or-break):** intervention consistency passes **and** zeroed-ASG collapses to advection. If C-i fails, the interpretability claim is not earned — fix the bottleneck before scaling anything.

**Why Tier 2 goes to the A100:** end-to-end + intervention pairs peaks at ~20–28 GB (`architecture.md` §7) — comfortable at 40 GB, at the edge of 24 GB. Run the heavy end-to-end / intervention / cross-region configs in the A100 sessions; do everything else on the L4.

---

## 5. GPU assignment (your two-machine split)

- **L4 (always-on) = daily driver.** All of Tier 0; Tier 1 QLoRA; every inference / evaluation pass; demo-JSON precompute; ASG-label validation; ablations that don't need the full end-to-end graph.
- **A100-40 GB (2–3 × <12 h) = reserved bursts.** (1) End-to-end coupling + rectified-flow renderer training; (2) intervention-consistency training + the C-i/C-ii faithfulness runs + ensemble CRPS generation; (3) cross-region transfer (F) + final full-resolution evaluation. Budget the sessions to these three jobs; iterate on the L4 between them so each A100 session starts from a known-good checkpoint.

---

## 6. Compute engineering for the <12 h limit (mandatory, not optional)

The plan is feasible **only** with this discipline — the failure mode is losing an A100 session to a disconnect with no resumable checkpoint:

1. **Checkpoint/resume on every job.** Save optimizer + step + RNG state to the GCS bucket every N steps; resume-from-step on start. Colab A100 is preemptible and can drop *before* 12 h. Tier-2 cumulative training will exceed one session → **chain sessions from checkpoints**.
2. **Persistent storage, not Colab disk.** Colab disk resets between sessions; mount the **GCS bucket** (or Drive) and read cached ASG labels / context slices / frozen features / advection fields / checkpoints from there. Re-downloading SEVIR or recomputing pysteps labels per session will burn the budget — do it once (`datasource.md` §3).
3. **L4-first iteration.** Debug, ablate, and validate on the always-on L4; only promote to the A100 when a run is correct and checkpoint-clean.
4. **Session budgeting.** Three A100 jobs, ≤12 h each, each resumable. If a job won't fit 12 h even chained, shrink the patch/sequence/batch or accumulate gradients — never widen past 40 GB.

---

## 7. Efficiency tricks (minimal compute, still good enough)

- **QLoRA-4-bit** VLM; freeze backbone, train adapters + projector only.
- **Latent renderer** in a frozen 8× VAE; **128×128 patches**, short sequences.
- **Few-step rectified flow** (1–4 steps) → cheap train *and* inference.
- **Cache everything computed once** (ASG labels, context slices, frozen features, advection) → bucket; reuse across all runs and sessions.
- **bf16 + gradient checkpointing**; **rainy-oversampled** windows (≥80 % rain).
- Storage in the **tens of GB** (fits free notebooks + a cheap bucket).

---

## 8. Step-by-step (mapped to tiers)

1. **Data + ASG pipeline** *(free, CPU/L4)* — SEVIR subset; co-locate HRRR/ERA5 + DEM; pysteps tracking → ASG_t, ASG_{t+h}, advection; validate auto-labels on the hand-labeled subset; mine IEM-AFD environment + Storm-Events extremes. **Cache all to the bucket.**
2. **Tier 0** *(L4, free)* — transition transformer + deterministic renderer; gate vs persistence/advection; oracle-ASG pixel parity. **Publishable de-risking result.**
3. **Tier 1** *(L4, free)* — Five-phase curriculum: Ph-1 VQA → Ph-2 object description → Ph-3 structured ASG [**gate: F1 ≥ 0.70 on gold subset**] → Ph-4 CoT → Ph-5 equation-aware CoT; QLoRA NF4 throughout. Ph-5 checkpoint = Tier-1 deliverable and Tier-2 init. Run §D ablations (±NL priors, ±physics equations) from Ph-4 vs Ph-5 checkpoints.
4. **Tier 2** *(A100 ×2–3)* — end-to-end + bottleneck + rectified-flow renderer; low-LR/stop-grad VLM; intervention-consistency; scheduled sampling; ensembles; cross-region. **Gate:** intervention consistency passes; zeroed-ASG → advection.
5. **Evaluation + forecaster study + write-up** — Groups A–F (`eval.md`); claim-evidence audit; adversarial self-review.
