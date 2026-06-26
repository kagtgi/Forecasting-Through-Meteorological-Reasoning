# Will ASG-WM converge? Is it feasible? — a component-by-component analysis

*Pre-training feasibility review (2026-06-22). Purpose: decide whether to commit limited
GPU budget. Honest engineering assessment, grounded in the actual code, not the paper prose.*

---

## TL;DR

**Yes, it is feasible, and each stage will converge with high confidence — but for a specific,
non-obvious reason, and with two real risks that are about *quality and budget*, not divergence.**

The reason convergence is safe: **every learned module is a residual around a strong physical
prior.** The transition predicts a *correction* on top of semi-Lagrangian advection
(`predict_residual: true`); the renderer is *conditioned on the advected frame*, which is already
~90 % of the answer; Stage A is plain QLoRA SFT on a structured target. None of these starts from
a hard optimisation landscape. The one thing that *could* have made the system intractable —
back-propagating an image loss through a **discrete / text** ASG bottleneck — is **deliberately
avoided** (the VLM is stop-grad in Tier-2; the ASG is materialised and fed forward; scheduled
sampling adapts the renderer to its own upstream errors). So Tier-2 is *renderer fine-tuning under
distribution shift*, not discrete optimisation. That is tractable.

The two real risks are:

1. **Label-quality ceiling** (Stage A). The ASG-F1 gate can only reach what the *auto-labeller*
   agrees with gold on. You cannot teach the VLM to beat its teacher.
2. **Tier-2 loss balance + compute** — 7 loss terms, an intervention term (λ=1.0) that
   back-props through 2 extra sampling rollouts, enforced from step 0; and ~13 U-Net passes per
   step. This is where a paid session can be wasted on a divergence or an OOM.

Both are now **de-risked in code** (see §4). After those changes I would commit the budget.

> **Empirical update (local RTX 4060 run, 2026-06-22).** The full chain `00→01→10→20→30→40→41→42`
> was run end-to-end (synthetic + DummyVLM). It (a) **confirmed both optimiser paths**: the
> transition overfits one batch **309×** and the renderer's flow-matching loss **65×**; (b)
> confirmed the faithful bottleneck (zeroing the ASG collapses skill to 0); and (c) **caught a real
> latent bug** — `mass_budget_residual` was an unnormalised squared sum (~1e10 vs render ~1e3) that
> would dominate Tier-2 and explode at full resolution. Now normalised per-pixel; Tier-2 `total`
> fell from ~1e9 to ~1e3 and is reconstruction-dominated. See `RUN_PLAN.md` for the run sequence.

---

## 1. The convergence question, made precise

"Will it converge" conflates three different questions. Separating them is the whole analysis:

| Question | Answer |
|---|---|
| (a) Will the **training loss descend** for each module? | **Yes**, high confidence (see §2). |
| (b) Will the **semantic metric** (ASG-F1, CSI/CRPS) reach a *useful* level? | **Conditional** — bounded by label quality (Stage A) and the VAE (Stage C). |
| (c) Will the **end-to-end coupling (Tier-2)** stay stable? | **Yes, if** the loss is warmed-up and watched — now enforced in code. |

Most "it didn't work" failures in systems like this are (b) or a *silent* (c), never (a).

---

## 2. Per-component verdict

### Stage A — VLM → ASG (QLoRA SFT) · **converges: ~certain; ceiling: label-bound**
- Objective is token cross-entropy on a structured ASG + NL target. QLoRA SFT on structured
  outputs is one of the most reliable training regimes in existence — the loss *will* descend.
- Continuous attributes are **quantised** (motion → 8 km/h bins, growth → 2 sig figs): this turns
  regression into classification, which an LM does far better, and *raises* the achievable F1.
- The achievable **ASG-F1 ceiling = agreement(auto-label, gold)**. If the CV auto-labeller agrees
  with the hand-gold at < ~0.75, the **0.70 gate is unreachable** no matter how well the VLM fits.
  → This is now measured *before* training by the new **label-quality audit** (§4).
- The 5-phase curriculum (VQA → desc → ASG → CoT → eq-CoT) is a sound difficulty ramp, and the
  **hard Ph-3 gate** (raise if F1 < 0.70) stops you from spending Ph-4/5 + Tier-2 on a broken
  perceptor. *Respect the gate — it is the main budget guard.*

### Stage B — transition transformer · **converges: ~certain (lowest risk)**
- `predict_residual: true`: at init the residual ≈ 0, so the model **starts as the advection
  forecaster** — a non-trivial baseline. Loss starts *low*, gradients are well-conditioned, and
  the network only learns corrections. This is the single most convergence-friendly setup here.
- Small (6 layers, d=256), trains in minutes on an L4. The PINN continuity (0.1) and smoothness
  (0.01) penalties are small soft regularisers — they cannot destabilise.
- *Empirically verified*: the pre-flight overfit test drives a one-sample loss **104× down in 150
  steps** (grad path healthy). If that ever fails, it is a wiring bug, not a capacity limit.

### Stage C — latent rectified-flow renderer · **converges: high; quality: VAE-bound**
- The rectified-flow / flow-matching objective is an L2 velocity regression — convex-ish and far
  more stable than GANs or score-matching with a bad noise schedule. It trains reliably.
- Conditioned on `Z = ASG ⊕ advect_blind`, where `advect_blind` is already close to the target →
  again *residual/refinement* learning. Strong, spatially-aligned conditioning + flow matching =
  reliable convergence.
- **Risks are quality, not optimisation**:
  - *VAE domain gap.* `stabilityai/sd-vae-ft-mse` is trained on natural RGB images, not
    single-channel radar VIL. Encoding/decoding precipitation through it can blur the
    high-frequency structure that high-threshold CSI depends on. → mitigation in §4/§5.
  - *Ensemble calibration.* Few-step (4) flow with `ensemble_k=10` does not *guarantee* a
    calibrated spread → CRPS may be mediocre even when samples look fine. Validate early.

### Tier-0 (transition + deterministic renderer) · **converges: ~certain**
Cheap, on L4. Use it as a **canary**: if Tier-0 cannot beat persistence/optical-flow on a small
val set, stop and debug *before* spending on the VLM and Tier-2.

### Tier-1 (5-phase curriculum) · **converges: yes; gated**
Sequential SFT with cross-phase checkpoint seeding. The Ph-3 F1 gate is the decision point.

### Tier-2 (end-to-end) · **the only genuine stability risk — now hardened**
- **Why it is *not* the nightmare it looks like:** the VLM is **stop-grad** (verified in
  `tier2_endtoend.py`), so there is **no back-prop through the discrete ASG**. Scheduled sampling
  anneals oracle → Stage-B-inferred ASG over 8 000 steps, so the renderer learns to tolerate its
  own upstream errors. Tier-2 is therefore renderer + transition fine-tuning under distribution
  shift — tractable.
- **Risk R-2a — multi-loss balance.** 7 terms. The **intervention-consistency** term (λ=1.0)
  back-props through **two extra sampling rollouts** and was enforced *from step 0 at full weight*,
  before the renderer is even coherent. This is the most likely destabiliser.
  → **Fixed:** linear **warmup 0→λ** (`intervene_warmup_steps`), **per-term logging**, and a
  **non-finite-loss abort** so a divergence costs seconds, not a session.
- **Risk R-2b — compute/memory.** Each step ≈ 1 flow-matching loss + **3 grad-tracked few-step
  sample rollouts** (`pred_field` + 2 in the intervention) ≈ ~13 U-Net passes with an autograd
  graph. On A100-40 GB at patch 128 / bs 8 this is heavy (OOM + slow-step risk).
  → **Mitigated:** `intervene_every` (compute the 2-render term every *k* steps) + existing bf16 +
  grad-checkpointing. If OOM: patch 96–112, bs 4, `intervene_every=2`.
- **Risk R-2c — coupling to Stage A.** Scheduled sampling feeds *inferred* ASGs; if F1 is
  borderline the renderer learns from noisy state. → Mitigated by the Ph-3 gate (don't reach
  Tier-2 with F1 < 0.70) + the slow anneal.

---

## 3. Risks, ranked by expected budget impact

| # | Risk | Likelihood | Impact | Mitigation (status) |
|---|---|---|---|---|
| 1 | Auto-label F1 < gate → Stage A can't pass Ph-3 | Med | High (wasted VLM run) | **Label-quality audit in pre-flight** ✅ |
| 2 | Tier-2 divergence (intervention term, lr) | Med | High (wasted A100 hrs) | **Warmup + logging + NaN-abort** ✅ |
| 3 | Tier-2 OOM at patch 128 / bs 8 | Med | Med (restart) | `intervene_every`, smaller patch/bs ✅ knobs |
| 4 | VAE domain gap caps high-threshold CSI | Med | Med (quality) | Fine-tune VAE decoder on VIL (§5) |
| 5 | Ensemble mis-calibration → poor CRPS | Med | Low–Med | Validate spread-skill early (§5) |
| 6 | Intervention proxy ≠ true causality | Low–Med | Med (faithfulness claim) | Hold out intervention types (§5) |
| 7 | Wiring/grad bug | Low | High if unseen | **Overfit-tiny-batch in pre-flight** ✅ |

---

## 4. What was changed in the code to make it converge / protect budget

All landed and tested (full suite green; synthetic pre-flight = GO):

1. **Tier-2 intervention-loss warmup** (`tier2_endtoend.py`, `configs/default.yaml`):
   `intervene_warmup_steps` (default 2000) ramps the λ=1.0 causal term **0 → λ** so the renderer
   becomes reconstruction-coherent *before* the high-variance, double-render signal is enforced.
2. **Tier-2 per-term logging** (`log_every`, default 50): prints render / intervene(w) / mass /
   nonneg / spectral / continuity / ib / p_oracle every N steps → a divergence is visible in
   *minutes*, not after the session.
3. **Tier-2 non-finite-loss abort**: raises *before* the optimiser step if the loss is NaN/Inf.
4. **Tier-2 `intervene_every`** (default 1): compute the expensive 2-render intervention term
   every *k* steps to cut compute when needed.
5. **Pre-flight overfit-tiny-batch check** (`scripts/99_preflight.py`): drives a one-sample
   transition loss down (must be ≥10×) — catches grad/wiring bugs in seconds.
6. **Pre-flight label-quality audit**: auto-label vs gold ASG-F1; flags **"gate unreachable"** if
   it is already below the Ph-3 gate, so you fix the *labeller*, not waste a VLM run.
7. **Mass-term normalisation** (`physics.mass_budget_residual`): per-pixel, so it is commensurate
   with the reconstruction MSE and resolution-invariant (was an unnormalised squared sum that
   dominated Tier-2 by ~6 orders of magnitude — caught by the local run).
8. **Pre-flight renderer overfit** (check 9): the Stage-C analogue of the transition overfit —
   confirms the flow-matching objective + optimiser reduce the loss before a paid run.

→ **Run `scripts/99_preflight.py --override data.dataset=sevir --override data.require_real=true`
on Colab and require GO before launching.** On real data, *read the label-audit F1*: if it is below
0.70, fix the auto-labeller first.

---

## 5. Optional improvements (do if a first run underperforms — not blockers)

- **Fine-tune the SD-VAE decoder (or last block) on VIL** to close the natural-image → radar gap;
  the most likely lever for competitive high-threshold CSI. (Or swap in a small purpose-built AE.)
- **Validate ensemble calibration early** (rank histogram / spread-skill on a tiny val set) rather
  than discovering poor CRPS at the end.
- **Hold out intervention *types*** at train time (train on translate/growth, test on
  regime-flip/rotate) so the faithfulness claim is about *causal responsiveness*, not memorising
  the +3 px translate proxy.
- **Horizon curriculum** for the transition (short h → long h): long-horizon compounding error is
  where advection-residual models hurt most.
- **Loss auto-balancing** (uncertainty weighting / GradNorm) if manual λ + warmup proves fiddly.

---

## 6. Go / No-Go checklist (before the paid run)

1. ☐ Pre-flight = **GO** on **real** SEVIR (`require_real=true`) — all 8 checks pass.
2. ☐ Label-audit F1 **≥ 0.70** vs gold (else fix the auto-labeller first).
3. ☐ Tier-0 **beats persistence/optical-flow** on a small val set (canary).
4. ☐ Ph-3 **F1 gate ≥ 0.70** passes (do *not* force past it).
5. ☐ Tier-2 first 200 steps: **total loss descends, no NaN**, intervene ramps in smoothly
   (watch the `[tier2]` log).

If all five hold, the budget is well spent. The framework is sound; the discipline is in the gates.
