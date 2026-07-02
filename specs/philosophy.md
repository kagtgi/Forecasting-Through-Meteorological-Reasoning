# philosophy.md — ASG-WM Method Framework Philosophy

> **Single source of truth** for the intellectual/methodological framework behind ASG-WM (FaithCast).
> Cross-referenced by `idea.md`, `architecture.md`, `eval.md`, `paper/paper.tex`, and the code package.
> Technical handle: **ASG-WM** (Atmospheric Scene Graph World Model).

---

## 1. Thesis (one sentence)

A vision–language model reads a short radar history, constructs an **explicit Atmospheric Scene Graph (ASG)** as the world-model state, a physics-constrained transition rolls that state forward, and a renderer materializes the future field **only from the predicted ASG plus future-blind advection** — so faithfulness is **architecturally entailed**, not asserted.

**"Forecasting Through Meteorological Reasoning"** means forecasting *through* an explicit, intervenable world-model state — not through free-form chain-of-thought prose.

---

## 2. What "meteorological reasoning" is (and is not)

The title uses "reasoning" in a precise, architectural sense. The following table fixes terminology across all docs:

| Concept | Role | Load-bearing? |
|---|---|---|
| **ASG** (structured state) | The *only* intervenable form of meteorological reasoning | **Yes** — sole causal path via bottleneck |
| **CoT / 5-phase curriculum** | Training device teaching spatial perception + equation grounding | **No** — not a separate inference step |
| **NL readout** (`render_NL`, `render_NL_delta`) | Human-facing projection of the ASG | **No** — deterministic template from ASG fields |
| **Explicit tracker** (Stage A2) | Inspectable temporal reasoning (IDs, motion from displacement) | Indirect — supplies motion vectors to ASG_t |
| **Symbolic admissibility** | Optional post-transition physical certificate (prototype) | Supplementary — not a core claim |

### 2.1 Reasoning = structured state transition

**Meteorological reasoning** in ASG-WM is the construction and roll-forward of a typed, enumerable state:

$$\text{Observe} \to \text{Identify} \to \text{Track} \to \text{Analyze} \to \text{Nowcast}$$

mapped to Stages A → A2 → B → bottleneck → C. The reasoning is **inspectable** because every step produces a discrete artifact (object tokens, trajectories, predicted ASG_{t+h}, rendered field) that can be read, edited, and re-rendered.

Natural language is a **downstream render** of the ASG — never a freeform channel that could drift from the load-bearing state. This is the answer to the chain-of-thought unfaithfulness problem (`turpin2023cot`, `lanham2023faithfulness`, `chen2025reasoning`): we do not test whether prose matches computation; we **remove prose from the causal path** and test whether the structured state is load-bearing.

### 2.2 NL readout contract

- **Readout #1** (after Stage A): `render_NL(ASG_t)` — describes present state.
- **Readout #2** (after Stage B): `render_NL_delta(ASG_t, ASG_{t+h})` — describes the predicted transition.

Both are computed **after** the corresponding ASG is known. Stage B does not emit freeform NL; it predicts ASG tensors, and the NL is a grammar-faithful template render. See `architecture.md` §8–9 and `asg/render_nl.py`.

---

## 3. Three-layer ontology

### 3.1 Epistemic layer (what the model knows)

Five categories of meteorological knowledge, acquired through the VLM curriculum and context encoding:

| Category | Training signal | Encoded in |
|---|---|---|
| **Governing equations** | Advection operator, PINN continuity loss, Ph-5 equation-aware prompts | Stage B dynamics + Stage A Ph-5 |
| **Seasonal** | Month-of-year encoding, AFD seasonal references | Context tokens |
| **Geographic** | DEM topography, coastline mask | Context tokens + gridded channels |
| **Diurnal** | Solar angle, time-of-day | Context tokens |
| **Synoptic** | CAPE/CIN/shear/PWAT from HRRR/ERA5 | Context tokens + gridded channels |

Knowledge is **encoded into ASG attributes and context tokens**, not retained as prose. The five-category ablation (eval.md §D, Group G) tests whether each category contributes skill on a matched held-out benchmark slice.

### 3.2 Dynamical layer (how the state evolves)

Stage B is a physics-constrained transition operator $T$:

$$ASG_{t+h} = T(ASG_t, C)$$

with three equation-injection points:
1. **Differentiable advection operator** — semi-Lagrangian warp; network predicts residual.
2. **PINN-style continuity/mass residual** — penalizes equation violation on the predicted state.
3. **Equation-aware training prompts** (Stage A Ph-5) — teaches the VLM to ground on equations during perception training.

Ambiguity resolution (which of many physically plausible futures?) happens **here**, not in the renderer.

### 3.3 Causal layer (why faithfulness holds)

The information bottleneck:

$$Z = ASG_{t+h} \oplus \texttt{advect\_blind}(X_t) \quad \Rightarrow \quad F = \mathcal{R}(Z)$$

Two architectural entailments (not empirical hopes):
- **Collapse:** zero ASG → $F = \texttt{advect\_blind}(X_t)$.
- **Intervention:** perturb ASG → field changes predictably.

Formalized as Definition "Architectural faithfulness" in `paper.tex` §Theoretical Framework. Proved by suite C-i..C-v in `eval.md` §C.

---

## 4. The five-step framework (operational meteorology mirror)

| Step | Name | Module | Output |
|---|---|---|---|
| ① | Observe | Stage A encoder | Multi-channel ingest + context |
| ② | Identify | Stage A VLM (per-frame) | Per-frame object sets |
| ③ | Track | Stage A2 explicit tracker | ASG_t with stable IDs + motion |
| ④ | Analyze | Stage B transition | ASG_{t+h} + `render_NL_delta` |
| ⑤ | Nowcast | Stage C renderer | $\hat{X}_{t+h} = \texttt{advect\_blind} + \Delta(Z)$ |

**Key separation:** Identify (spatial, single-frame) vs Track (temporal, multi-frame) vs Analyze (dynamics prediction). The VLM never does joint tracking+prediction; temporal reasoning is inspectable and deterministic.

---

## 5. Faithfulness-by-bottleneck (the sharp contribution)

### 5.1 The trap

If a VLM emits plausible meteorological prose *alongside* a field produced from raw encoder latents, the language is post-hoc rationalization. This is the documented failure mode of free-form CoT.

### 5.2 The fix

Architecturally force the renderer to see **only** the structured state plus future-blind advection. The ASG is trained as a minimal sufficient statistic via the Information Bottleneck:

$$\mathcal{L}_{\text{IB}} = \beta\, I(\text{ASG}; X_{t-k:t}) - I(\text{ASG}; X_{t+1:t+n})$$

In practice: hard structural cap (N_max=16 objects, quantized motion) + soft KL penalty on continuous sub-fields.

### 5.3 Why this is unclaimed

No published nowcaster couples (a) autonomous perception from radar with (b) causal coupling where the reasoning **actually drives** the field. LangPrecip consumes human text; GPTCast has no language reasoning; Skew-T VLM outputs probability not a field; AI-Meteorologist explains but does not generate.

---

## 6. Interpretability vs accuracy trade-off (honest claim)

| Regime | Expected outcome | Why |
|---|---|---|
| **Steady advection** | ASG-WM ≈ or below SOTA | Bottleneck discards pixel detail black-box encoders exploit |
| **Initiation / growth** | ASG-WM > SOTA | Regime selection matters more than pixel detail; physics+context disambiguate |

**The claim is regime-stratified (Group B), not universal SOTA.** A paper that predicts and demonstrates its failure modes is more credible than one that hides them.

---

## 7. Claim boundaries

| Claim | Evidence group | Status |
|---|---|---|
| State is faithful / load-bearing | C-i + C-ii | Core — designed, [TBR] until Tier-2 |
| Reasoning resolves motion ambiguity | B (regime-stratified) | Core — designed, [TBR] |
| Equations help beyond NL priors | D (±operator, ±continuity, ±prompt) | Designed, [TBR] |
| Five knowledge categories contribute | D Group G + matched benchmarks | Designed, [TBR] |
| Competitive overall skill | A | Designed, [TBR]; baselines intentionally stubbed |
| Cross-region generalization | F | Designed, [TBR] |
| Useful to forecasters | E | **Gated on partner** — do not claim |

**Hard rule:** no green experiment ⇒ claim is cut or weakened.

---

## 8. Symbolic admissibility (supplementary, not core)

`src/asgwm/symbolic/admissibility.py` is a **prototype** that certifies whether a predicted ASG transition satisfies physical FSM constraints and flags initiation ambiguity via dual-SAT over CAPE/CIN envelopes.

- **Position:** optional post-transition certificate; demonstrated in `scripts/43_admissibility_demo.py`.
- **Not wired** into Tier-2 training loss or ensemble output.
- **Not a core contribution** — do not overclaim in paper abstract or contributions list.

---

## 9. ASG schema contract (v1)

The load-bearing v1 contract is the **10-field OBJECT line** in `asg/grammar.py`:

```
OBJECT(id, cy, cx, area, peak, vy, vx, regime, growth, conf)
GLOBAL(regime, n_objects)
```

Extended fields in `schema.py` (`morphology`, `conv_mode`, topology events) are **reserved for v2** and are not part of the grammar, training targets, or bottleneck rasterization. Do not extend the grammar without a version bump.

---

## 10. Limitations (state up front)

- Not a replacement for NWP; horizon ≤ 3 h.
- Auto-ASG labels carry noise (validated via gold subset, not eliminated); VLM cannot exceed auto-label quality (Ph-3 gate F1 ≥ 0.70).
- AFDs are synoptic-period context, not per-frame ground truth.
- Interpretability is over **evolution decisions** (regime, motion, growth), not pixel appearance.
- Unprecedented events remain hard.
- Forecaster study (Group E) requires an operational partner.
- Empirical results in the paper are **[TBR]** until the staged Colab A100 compute plan runs (see `TUTORIAL.md`, `RUN_PLAN.md`).
- Neural baselines (RainNet, NowcastNet, LangPrecip, ThoR) are intentionally stubbed; only pysteps and ASG-WM are evaluated in the current framework.

---

## 11. Reviewer Q&A (extended)

- *"Reasoning is just decorative CoT."* → No. The causal path is the ASG, not prose. C-i/C-ii test ASG perturbations, not rationale edits.
- *"Isn't this LangPrecip?"* → LangPrecip *consumes* human text; we *produce* structured state autonomously and make it the sole causal channel.
- *"Stage B emits NL before ASG?"* → No. Stage B predicts ASG_{t+h}; NL is `render_NL_delta(ASG_t, ASG_{t+h})` computed downstream.
- *"Five knowledge categories are hand-wavy."* → Each has a defined training signal and matched held-out benchmark slice (eval.md §D Group G). Results are [TBR].
- *"Symbolic admissibility is the real contribution."* → It is a supplementary prototype, not wired into training. The core contribution is faithfulness-by-bottleneck.
- *"You don't beat SOTA."* → No universal-SOTA claim. Reframe to B (regime gain) + C (faithfulness).
- *"Auto-ASG is distilled optical flow."* → Tier-0 gate + C-iii leakage audit + cross-region F address this.

---

## 12. Lineage: ThoR → ASG-WM

**ThoR** (`ta2025thor`): physics as a PDE soft constraint on a hidden latent.
**ASG-WM**: physics as a **structured, inspectable state** with a governing-equation transition and exact renderer.

The progression: from "physics as penalty" to "physics as manipulable world-model state." ThoR baseline comparison is [TBR] until the adapter is implemented; the philosophical contrast is architectural, not yet empirical.
