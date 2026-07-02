# idea.md — Faithful Reasoning Nowcasting via a Materialized Atmospheric Scene Graph

> **Working title:** *FaithCast* — a precipitation nowcaster whose structured world-model state is **load-bearing by construction**, not decorative. *(Name is a placeholder; swap freely. The technical handle used throughout these docs is **ASG-WM**: an Atmospheric-Scene-Graph world model.)*
>
> **Philosophy:** see `philosophy.md` for the unified method-framework ontology (what "meteorological reasoning" means, the three-layer model, claim boundaries).

---

## 1. One-sentence thesis

A vision-language model reads a short radar history, constructs an **explicit, human-readable world-model state** (an *Atmospheric Scene Graph*, ASG; no human-supplied text at inference), a physics-constrained transition rolls that state forward, and a physics-informed renderer materializes the future radar field **from that state and nothing else** — so faithfulness is **architecturally entailed** and provable by intervention.

**"Forecasting Through Meteorological Reasoning"** means forecasting *through* an explicit, intervenable ASG state — not through free-form chain-of-thought prose. Natural language is a downstream render of the ASG (`render_NL`, `render_NL_delta`); the load-bearing reasoning is the structured state itself (`philosophy.md` §2).

The contribution is not "a VLM that nowcasts." It is the **faithfulness-by-bottleneck** mechanism that makes interpretability real rather than cosmetic, demonstrated on nowcasting because nowcasting is the rare physical domain where an explicit state, a governing-equation transition, and an exact renderer can all be written down and checked.

---

## 2. The landscape: your pipeline is currently four papers, not one

The naive framing ("a VLM that knows weather → reasons → predicts the field") is **partly occupied**, and the headline "first language-guided nowcasting" is **already taken**. Each of the three pipeline stages — perceive, reason, render — has been done **separately**. The map:

| Existing work | What it does | Architecture | What it is **not** | Cite |
|---|---|---|---|---|
| **GPTCast** (Franch et al., GMD 2025) | GPT forecaster over radar frames tokenized by a VQ-autoencoder; ensemble nowcasts that beat extrapolation | LLM-*architecture* over visual tokens | No meteorological reasoning **in language**; the "language model" is a token sequencer, not a reasoner | `franch2025gptcast` |
| **LangPrecip** (Ling et al., arXiv 2512.22317, Dec 2025) | **"First language-guided precipitation nowcasting."** Natural-language motion descriptions as explicit semantic constraints; Rectified-Flow generator + dual-path wavelet-consistency decoder; contributes **LangPrecip-160K** (160K radar–text pairs, Swedish + MRMS) | Text **fed in** as a conditioning constraint | The language is a **supplied input**, not autonomous reasoning the model performs from radar alone | `ling2025langprecip` |
| **Skew-T VLM** (Lee, Yang & Choi, arXiv 2508.12198, 2025) | Fine-tunes SmolLM2 + SmolVLM with LoRA to **mirror a forecaster's step-by-step process** — visual grounding of instability/wind/humidity, then a CoT estimate | Small VLM, curriculum (VQA → CoT) | Output is a **4-class precipitation probability** from soundings, **not a radar sequence** | `lee2025skewt` |
| **AI-Meteorologist** (arXiv 2512.11819; Hierarchical variant 2511.23387, 2025) | LLM agents turn numerical forecast tables into transparent narratives via in-context prompting | LLM agent, no fine-tuning | It **explains** an existing forecast; it does **not generate the forecast field** | `aimeteorologist2025`, `makarov2025hierarchical` |

**The open slot.** No published system closes the **autonomous loop**: a VLM that (a) reasons in language **from the radar with no human text at inference**, and (b) where that reasoning **actually drives the field generation**. Clause (b) is the whole game.

The real motivating axis is not "physics vs no physics": NowcastNet already embeds advection equations and a continuity-style conservation constraint (`zhang2023nowcastnet`), and PreDiff injects physical knowledge post-hoc (`gao2023prediff`). The gap is **hidden state vs explicit, human-readable, manipulable state**. Existing physics-informed nowcasters remain black boxes: their internal state is a latent vector, not an inspectable, named object set. You cannot ask NowcastNet "which cell grows and why," intervene on that belief, and observe the forecast respond. The open slot is a nowcaster where this is not only possible but **architecturally guaranteed**.

---

## 3. Why faithfulness-by-bottleneck is the sharp contribution (and is unclaimed)

**The trap to avoid.** If the VLM emits a beautiful meteorological rationale *alongside* a field that is really produced from raw radar latents, the language is **post-hoc rationalization** — the exact failure mode documented for chain-of-thought: models produce plausible explanations that systematically misrepresent the true cause of their output (`turpin2023cot`), can reach the same answer after their stated reasoning is truncated or corrupted (`lanham2023faithfulness`), and silently rely on cues they never verbalize even in dedicated reasoning models (`chen2025reasoning`). An interpretability pitch built on an unfaithful rationale is decorative.

**The fix — make reasoning the only channel.** Architecturally force an **information bottleneck**: the renderer sees **only** the VLM's structured reasoning output (the ASG: cell locations, motion vectors, growth/decay flags, regime label) plus a *future-blind* extrapolation of the present, and **nothing else** from the input. Formally, the ASG is trained to be a **minimal sufficient statistic** of the radar history for predicting the future field — the Information Bottleneck objective (`tishby1999ib`; deep/variational forms `tishby2015db`, `alemi2017vib`):

$$\mathcal{L}_{\text{IB}} \;=\; \beta\, I(\text{ASG}; X_{t-k:t}) \;-\; I(\text{ASG}; X_{t+1:t+n}),$$

squeezing the input through the explicit state so the state must carry the predictive content.

**Why this is faithful, not cosmetic.** Because the state is the *only* path to the future:
- **Zero the ASG** → the forecast collapses to advection (the future-blind path), proving the state is load-bearing.
- **Perturb the ASG** (drag a cell, flip grow↔decay, rotate a motion vector) → the field changes **exactly** as the rationale says.

This is **counterfactual simulatability** (`chen2023counterfactual`) turned into an architectural guarantee and an *interventional* experiment, not a correlational hope. As far as the four works above and the broader nowcasting literature show, **the faithfulness-by-bottleneck framing for nowcasting is unclaimed** — and it is a much sharper contribution than "VLM for nowcasting."

---

## 4. The conceptual advance (venue-level framing)

Classical world models keep state **hidden** in a latent (`ha2018worldmodels`; Dreamer line `hafner2021dreamerv2`, `hafner2023dreamerv3`), which is why their forecasts are black boxes and short radar histories collapse into over-smoothed, ambiguous motion. Object-/structured world models make state explicit but in a learned latent (`kipf2020cswm`). **This project makes the state explicit *and human-readable*, with a governing-equation transition and an exact renderer.** The claim:

> Language models can construct and roll forward an **explicit, verifiable** world model of a physical system, rather than an implicit latent one. Nowcasting is the testbed; the transferable template — *interpretable world models in domains with partial physical knowledge* — is the contribution.

The state is a triple — **state, transition, observation/renderer**:
- **State** `S_t` = the **ASG**: storm objects with attributes (centroid, area, peak intensity, motion vector), a regime label (initiation / growth / decay / steady), and a growth–decay field. Compact, inspectable, manipulable.
- **Transition** `S_t → S_{t+h}`: a learned dynamics model rolling the ASG forward, **constrained by the governing equations** (advection, continuity / mass conservation, growth–decay parameterizations) and steered by physical context (CAPE/CIN, shear, PWAT, lightning) to **select the meteorologically correct future** among the many a short history permits. This is where the under-constrained motion ambiguity is resolved.
- **Renderer** `S_{t+h} → X_{t+h}`: a physics-informed generator that materializes the exact radar field **only** from the predicted state plus the future-blind advection path.

---

## 5. Contributions (numbered, each tied to evidence)

1. **A faithful-by-construction reasoning nowcaster.** The first nowcaster where an autonomous language rationale is the load-bearing forecast mechanism via an information bottleneck — distinct from GPTCast (no language reasoning), LangPrecip (text fed in), the Skew-T VLM (probability, not a field), and AI-Meteorologist (explains, does not generate). *Evidence: §C faithfulness suite in `eval.md`.*
2. **The Atmospheric Scene Graph (ASG)** as an explicit, human-readable world-model state for precipitation, with a free CPU-only auto-labeling pipeline (pysteps tracking) validated against a hand-labeled subset. *Evidence: ASG-accuracy in §C; pipeline in `datasource.md`.*
3. **A physics-constrained transition** that injects governing equations three ways — a differentiable advection operator, PINN-style continuity/mass residual losses, and equation-aware reasoning prompts — and an ablation isolating which kind of knowledge (NL priors vs. equations) drives the gain. *Evidence: §D in `eval.md`.*
4. **Interventional faithfulness as a metric**, not a claim: intervention-consistency and the inferred/oracle/zeroed/shuffled-ASG bottleneck ablation. *Evidence: §C.*
5. **Regime-stratified skill** showing the gain concentrates on the hard, nonlinear regimes (initiation/growth), where vision-only and physics-only regressors fail. *Evidence: §B.*

---

## 6. Claim → evidence map (hard constraint: no green experiment ⇒ claim is cut)

| Claim | Evidence (experiment group) | Status |
|---|---|---|
| The rationale is faithful (load-bearing) | C-i intervention consistency + C-ii bottleneck ablation (zeroed → advection; shuffled → wrong field) | core, designed |
| Reasoning resolves motion ambiguity | B: skill stratified by regime; gain on initiation/growth | core, designed |
| Governing equations help (beyond NL priors) | D: ±advection operator, ±continuity residual, ±equation-aware prompt | designed |
| Competitive overall, superior at heavy-rain / long lead | A: CSI/HSS/FSS, pooled-CSI, CRPS vs verified baselines | designed |
| Generalizes across regions | F: cross-dataset transfer (SEVIR ↔ one of HKO-7/MeteoNet) | designed |
| Useful to forecasters | E: DGMR-style ranking + decision-usefulness | **gated on a forecaster partner** |

---

## 7. Venue strategy (honest)

- **Flagship *Nature* nowcasting** (cf. DGMR `ravuri2021dgmr`, NowcastNet `zhang2023nowcastnet`) required operational skill validated by forecasters at scale **plus** extreme-event superiority. That bar = skill ≥ NowcastNet on extremes (B) **+** a forecaster study (E) **+** the faithfulness result (C). The forecaster study is the **gating dependency, and it is people, not GPUs** — minimizing device cost does not change it.
- **Without the forecaster study at scale:** target **Nature Machine Intelligence / Nature Communications / npj Climate & Atmospheric Science** on the strength of C (faithfulness) + B (regime gain) + competitive A. The faithfulness-by-bottleneck mechanism is the differentiator here and is stronger than PreDiff's *post-hoc* knowledge alignment (`gao2023prediff`).
- **AAAI-class ML venue + domain journal:** novel method + solid benchmarks + the interpretability mechanism, with modest absolute skill. This is the realistic near-term target given the compute envelope.

---

## 8. Pre-answered reviewer attacks

- *"Interpretability is decorative."* → The bottleneck makes it load-bearing; C-i/C-ii prove it interventionally. This is the central design, not an add-on.
- *"You don't beat SOTA overall."* → No universal-SOTA claim. Reframe to B (regime gain) + C (faithfulness) + E (usefulness). Heavy-rain / long-lead is the target slice.
- *"Auto-ASG is a distilled optical flow with future leakage."* → Tier-0 gate beats persistence **and** pure advection on object evolution; the Stage-B advection is **future-blind** (leakage audit in C); F shows cross-region transfer.
- *"VLMs can't make precise fields."* → They don't. The VLM emits the **state**; the physics renderer makes the field. A shows pixel parity is reachable from an oracle ASG.
- *"Single region."* → F (cross-dataset).
- *"Compute / reproducibility."* → Free Tier-0 + a single-always-on-L4 + 2–3 spot-A100-sessions recipe (see `training_method.md`) + released ASG pipeline and weights.
- *"Isn't this just LangPrecip / GPTCast?"* → No. LangPrecip *consumes* human text; we *produce* it autonomously and make it causal. GPTCast has no language reasoning at all. The novelty is the **faithful ASG→field world model**, and captioning is supervision, not the contribution.
- *"VLMs have no reliable meteorological knowledge — this is prompt engineering over a weak domain model."* → Correct that base pretraining yields shallow and unreliable domain knowledge for radar meteorology. The knowledge in this pipeline is **built** via a five-phase curriculum fine-tuning on ASG labels, AFD context, and equation-aware prompts (see `architecture.md` §10 and `training_method.md` §3). Following the curriculum principle of the Skew-T VLM (`lee2025skewt`), a small VLM learns forecaster-style structured reasoning through staged supervision — visual VQA grounding → ASG structured output → chain-of-thought reasoning → equation-aware CoT. The contribution is the **faithful-bottleneck architecture** that makes reasoning causally load-bearing; the VLM's meteorological priors are a trained component, not a pretrained assumption.

---

## 9. Honest central risk: interpretability and accuracy are in tension

Compressing the radar history into a human-readable scene graph discards pixel-level detail that a black-box encoder exploits. The naive pitch — "reasoning makes it both more interpretable and more accurate" — conflates two separate bets.

**Where ASG-WM is expected to lose.** On steady, advection-dominated cases (the majority of radar frames), the kinematic signal is simple and black-box baselines (NowcastNet, LDCast) exploit it fully from raw latents. A scene graph that discards pixel detail will score lower on raw CSI for these cases. This is expected and must be reported honestly, not hidden.

**Where ASG-WM is expected to win.** On the under-constrained regimes — initiation (no prior cell, boundary forcing) and rapid growth (cell intensification driven by instability and shear) — a short radar history admits many plausible futures and a black-box regresses toward the mean. Here, explicit physical reasoning (CAPE/CIN, boundary location, moisture, shear profile) selects the meteorologically correct future. The lost pixel detail matters less than the correct regime selection.

**The claim is therefore regime-stratified (Group B)**, not overall-SOTA. The thesis is: "On the cases where physics and reasoning matter most, explicit and faithful reasoning outperforms implicit latent regression — and we can prove the reasoning is load-bearing." This is sharper and more defensible than "beats everything everywhere," and it is the claim the bottleneck + B-evidence actually supports.

**Abstract framing:** competitive overall (Group A, no universal-SOTA claim), superior on initiation/growth (Group B), with a proved faithfulness mechanism (Group C). Acknowledge explicitly that steady-advection performance is bounded by the bottleneck. A paper that predicts its failure modes and demonstrates them is more credible than one that hides them.

---

## 10. Scope and limitations (state up front)

Not a replacement for NWP; horizon ≤ 3 h; auto-ASG labels carry noise (validated, not eliminated); AFDs are **synoptic-period context, not per-frame labels**; interpretability is over **evolution decisions**, not pixel appearance; unprecedented events stay hard; the forecaster study needs a partner. The ASG schema names are placeholders — reshape freely.
