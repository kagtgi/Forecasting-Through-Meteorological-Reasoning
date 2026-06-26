# eval.md — Experiments, metrics, and the faithfulness proof

> Every experiment is **tier-tagged** (which compute tier produces it). The non-negotiable: **the faithfulness suite (C) is the proof of the contribution.** Skill numbers (A) without C reduce the paper to "another nowcaster"; C without A is un-anchored. Lead with C + B.

---

## 1. Experiment groups

### A — Skill vs SOTA *(Tier 0 / Tier 2)*
SEVIR + one cross-region (HKO-7 *or* MeteoNet). **Verified baselines only** (full citations in `references.bib`):

| Family | Baselines |
|---|---|
| Extrapolation | pysteps / S-PROG (`pulkkinen2019pysteps`) |
| Recurrent | ConvLSTM (`shi2015convlstm`), TrajGRU (`shi2017trajgru`), PredRNN / PredRNN++ (`wang2017predrnn`, `wang2018predrnnpp`) |
| CNN/U-Net | RainNet (`ayzel2020rainnet`), SmaAt-UNet (`trebing2021smaatunet`) |
| Transformer | Earthformer (`gao2022earthformer`), MetNet-1/2/3 (`sonderby2020metnet`, `espeholt2022metnet2`, `andrychowicz2023metnet3` — **public reproduction limited**) |
| Generative | DGMR (`ravuri2021dgmr`), NowcastNet (`zhang2023nowcastnet`), LDCast (`leinonen2023ldcast`), CasCast (`gong2024cascast`), PreDiff (`gao2023prediff`), DiffCast (`yu2024diffcast`), GPTCast (`franch2025gptcast`), LangPrecip (`ling2025langprecip`) |

**Metrics:** CSI / HSS at multiple thresholds incl. heavy rain (`jolliffe2012forecast`); **SEDI** (`ferro2011sedi`) reported alongside CSI/POD/FAR at the **high VIL thresholds [160, 181, 219]** for base-rate-robust extreme-cell skill — CSI degrades as the positive rate vanishes at the top thresholds, where the heavy-rain claim actually lives, so the base-rate-independent score is the honest one there (`config eval.sedi_thresholds_vil`; SEDI used for extreme detection in YingLong, `xu2025yinglong`); **FSS** (`roberts2008fss`); **pooled CSI** at 4×4 / 16×16 to 3 h (neighborhood verification as in DGMR, `ravuri2021dgmr`); **CRPS** + reliability diagrams (`gneiting2007crps`, `hersbach2000crps`); **LPIPS** (`zhang2018lpips`) + **radially-averaged power spectrum** for realism. **Target:** competitive overall, superior at **heavy-rain / long lead** — no universal-SOTA claim.

### B — Nonlinear regimes (the contribution evidence) *(Tier 0 / Tier 2)*
Skill **stratified by regime** (initiation / growth / decay / steady), using the ASG regime labels. **Hypothesis:** the gain over vision-only and physics-only baselines concentrates on **initiation / growth** — the under-constrained regimes physics+reasoning are designed to resolve. This is where the explicit-state world model earns its keep. Report **SEDI at the high thresholds [160, 181, 219]** (`ferro2011sedi`) per regime alongside CSI: extreme cells are exactly the sparse positives where base-rate-sensitive scores mislead, and the initiation/growth regimes are where they concentrate.

**Honest transparency requirement.** Report regime-specific skill for *all* baselines, including the cases where black-box baselines are expected to win. The expected pattern: ASG-WM ≈ or slightly below SOTA on **steady-advection** (the majority of frames — bounded by the bottleneck, which discards pixel detail black boxes exploit); ASG-WM > SOTA on **initiation and rapid growth** (the high-value minority — where explicit physical reasoning resolves regime ambiguity). Both halves must appear in the paper. A paper that shows only the winning regimes will be caught in review; one that predicts its failure modes and demonstrates them in data is more credible. If the pattern does not hold — if ASG-WM also loses on initiation/growth — the claim must be revised before submission, not explained away.

### C — Faithfulness (the proof) *(Tier 2)*
1. **C-i Intervention consistency.** Perturb ASG_{t+h} (translate a cell along its motion vector; flip regime grow↔decay; scale the growth scalar) and verify the rendered field changes **in the predicted direction and location**. Report an intervention-consistency score (fraction of perturbations whose field effect matches the predicted effect within tolerance). *This is counterfactual simulatability (`chen2023counterfactual`) made architectural — the answer to CoT unfaithfulness (`turpin2023cot`, `lanham2023faithfulness`, `chen2025reasoning`).*
2. **C-ii Bottleneck ablation.** Render from **inferred** vs **oracle** vs **zeroed** vs **shuffled** ASG. Required pattern: oracle ≈ best; inferred close behind; **zeroed → collapses to advection**; **shuffled → wrong field consistent with the wrong state**. This proves the state is load-bearing.
3. **C-iii Leakage audit.** Confirm the Stage-B advection path is **future-blind** (built only from `X_{≤t}`); show that removing it does not silently restore future information. Guards against the "auto-ASG = distilled flow with leakage" attack.
4. **C-iv ASG accuracy.** Inferred ASG vs the hand-labeled gold subset (`datasource.md` §2) and vs forecasters where available.

### D — Ablations / knowledge sources *(Tier 1)*
Stages and losses; VLM size; **±ERA5/HRRR context**; **±lightning / water-vapor channels**; templated vs LLM-enriched NL; and the headline split — **±NL meteorological priors** vs **±physics equations** (advection operator, continuity/mass residual, equation-aware prompting) — to isolate **which kind of knowledge drives the gain**. This is what separates "physics helps" from "a bigger prompt helps."

**Eval discipline (Nature-family rigor, cf. YingLong `xu2025yinglong`).** Every ablation table reports (a) **significance on the skill gap** — bootstrap confidence intervals or a paired test over the eval set, not bare point estimates, so a claimed gain is shown to exceed noise; and (b) **parameter count and compute cost** (params / FLOPs or latency) per row, so a gain is read against its cost rather than in isolation. A "physics helps" row that needs 2× compute for a within-interval delta is not a win, and the table must make that visible.

### E — Forecaster study (Nature-grade) *(needs partner)*
DGMR/NowcastNet-style expert ranking + decision-usefulness with operational meteorologists (cf. the 56- and 62-forecaster studies in `ravuri2021dgmr`, `zhang2023nowcastnet`). **This is the gating dependency for a flagship venue and it is people, not GPUs.** Plan it early; it does not change with compute.

### F — Generalization *(Tier 2)*
Cross-dataset transfer (SEVIR ↔ the chosen cross-region set), lead-time curves (skill vs horizon to 3 h), failure analysis, and efficiency (params / latency / few-step sampling cost). **All group-F results are [TBR].**

**Independent-benchmark discipline (`xu2025yinglong`).** YingLong scores its headline skill on a benchmark independent of its training distribution (HadISD stations vs HRRR), not on held-out splits of the training set — the discipline that makes a generalization claim credible. We adopt the same posture: **train on SEVIR, report the OOD numbers on NEXRAD / MRMS** (and HKO-7 / MeteoNet where access permits). This is not only a generalization story — it is the empirical backstop to the **C-iii leakage rebuttal**: a model that secretly distilled future-revealing flow from its training distribution does not transfer cleanly to a held-out region, so a clean independent-benchmark score is positive evidence the future-blind bottleneck holds, complementing the architectural audit in C-iii.

**HKO-7 protocol mismatch (`shi2017trajgru`) — all HKO-7 results [TBR].** HKO-7 is a deliberately *different* protocol, not a drop-in test set, and the gap must be stated, not papered over. It is **2 km CAPPI dBZ** (not VIL), **6-min cadence**, and its native task is **5-frame-in / 20-frame-out**; our canonical pipeline is **5-min VIL byte** at the **13→12** headline / **13→36** long-lead horizons. Reconciliation requires: (i) a **dBZ→VIL-byte bridge using HKO-7's own Z–R coefficients a = 58.53, b = 1.56** (not Marshall–Palmer 200/1.6), regridded to the canonical `[T, 384, 384]`; and (ii) a cadence/horizon reconciliation from 6-min 5-in-20-out to our 5-min 13→12 / 13→36. Because both the unit bridge and the cadence map introduce assumptions, **every HKO-7 number is reported [TBR] and flagged as protocol-bridged**, never compared head-to-head with native-HKO-7 baselines without the bridge caveat. Online fine-tuning on HKO-7 is **out of scope** — it mutates weights at test time and would contaminate the faithfulness ablations (C). MeteoNet (`larvor2020meteonet`, openly downloadable) is the lower-friction open alternative when HKO-7's gated access (signed undertaking, weeks lead) does not clear in time.

### Stretch (label as exploratory)
Human-augmentation (forecaster with radar-only vs radar+reasoning) and discovery (flagging merge/split/initiation before annotation). Mark clearly as exploratory.

---

## 2. Metric notes and cautions

- **Categorical (CSI/HSS/POD/FAR):** standard verification (`jolliffe2012forecast`); report at light **and** heavy thresholds — heavy rain is the target.
- **FSS** (`roberts2008fss`): scale-selective; report across neighborhood sizes to show where skill lives spatially.
- **CRPS + reliability** (`gneiting2007crps`, `hersbach2000crps`): the probabilistic skill + calibration pair; ensembles come from the few-step flow.
- **LPIPS + power spectrum** (`zhang2018lpips`; spectral diagnostic per `ravuri2021dgmr`, `leinonen2023ldcast`): realism / blur — generative nowcasters must not win CSI by over-smoothing.
- **BLEU / ROUGE vs AFDs are weak** (`papineni2002bleu`, `lin2004rouge`): **secondary only.** AFDs are synoptic-period context, not per-frame ground truth; lead with faithfulness metrics (C) + human eval (E), not text-overlap scores.
- **Mask-aware scoring (validity / noise mask):** all categorical and field metrics — **CSI / HSS / FSS / CRPS** (and POD / FAR / SEDI) — take an optional per-pixel validity mask (`asgwm.eval.metrics`, `mask=` argument; truthy = valid), so clutter, no-coverage, and out-of-domain pixels are **excluded from the score** rather than counted as misses or false alarms. This is essential for clean numbers on the radar-mosaic sources in **group F** (NEXRAD / MRMS / HKO-7), where coverage gaps and ground clutter otherwise contaminate every threshold. The same balanced-loss validity mask that zeroes a pixel's training weight also zeroes it in evaluation, so train and test agree on what counts as observed.

---

## 3. Claim → evidence audit (hard constraint)

| Claim | Evidence | Status |
|---|---|---|
| Rationale is faithful / load-bearing | C-i + C-ii | core — design complete, must pass the Tier-2 gate |
| Reasoning resolves motion ambiguity | B (regime-stratified gain on initiation/growth) | core |
| Governing equations help beyond NL priors | D (±operator, ±continuity, ±equation-prompt) | designed |
| No future leakage | C-iii | designed |
| ASG is accurate | C-iv vs gold + forecasters | designed |
| Competitive overall, superior heavy-rain/long-lead | A (CSI/HSS/FSS/pooled/CRPS/LPIPS/PSD) | designed |
| Generalizes across regions | F | designed |
| Useful to forecasters | E | **gated on partner** |

**Rule (from research-paper-writing):** any claim in the abstract/intro without a green row above is **weakened or cut**. Especially: do not claim universal SOTA; claim faithfulness (C) + regime gain (B) + competitive skill (A).

---

## 4. Adversarial self-review (resolve before submission)

- **Contribution:** Is the delta vs LangPrecip/GPTCast/Skew-T-VLM/AI-Meteorologist stated in one sentence and backed by C? *(Yes: autonomous reasoning + faithful bottleneck; backed by C-i/C-ii.)*
- **Experimental strength:** Does C-ii show the full inferred/oracle/zeroed/shuffled pattern, not just zeroed? Does B beat **both** vision-only and physics-only baselines?
- **Evaluation completeness:** Heavy-rain thresholds reported? **SEDI at the high thresholds [160, 181, 219]** shown alongside CSI so the extreme-cell claim is not resting on a base-rate-sensitive score (`ferro2011sedi`)? Reliability shown, not just CRPS scalar? LPIPS/PSD to rule out blur-wins? **Scores mask-aware** on the group-F mosaic sources (clutter/no-coverage excluded)?
- **Statistical & cost rigor (`xu2025yinglong`):** Does every skill gap carry a **significance interval/test**, not a point estimate? Does every architecture ablation list **params and compute cost** so gains are read against their price? Is at least one headline number scored on an **independent benchmark** (their HadISD-vs-HRRR precedent — our SEVIR-train / NEXRAD+MRMS-OOD-test, see group F)?
- **Method soundness:** Is the bottleneck truly the only future path (C-iii)? Is the IB compression term actually constraining capacity, or is the object budget so large the bottleneck is vacuous?
- **Writing clarity:** One message per paragraph; terminology (ASG, faithful bottleneck, future-blind advection, Stages A/B/C) stable across all five docs and the paper.

- **Honest regime loss:** Does B report steady-advection performance where vision-only baselines are expected to outperform ASG-WM? Is the per-regime breakdown shown for all baselines, not just in our favor? The regime-stratified claim requires transparency on both the winning and losing regimes.
- **Bottleneck capacity audit:** Is the IB compression term actually constraining capacity, or is the object budget so large the bottleneck is vacuous? With N_max objects and quantized attributes, is the ASG channel capacity strictly smaller than the raw radar channel capacity? Over-capacity nullifies the faithfulness-by-compression argument. See `training_method.md` §4 for the concrete capacity audit procedure.

Append the five-dimension self-review (contribution / clarity / experimental strength / evaluation completeness / method soundness) to the final draft and revise until every high-risk item is addressed.
