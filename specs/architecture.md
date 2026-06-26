# architecture.md — Three readable stages + a faithful bottleneck

> Terminology is fixed across all docs: **Stage A = Perception**, **Stage B = Transition**, **Stage C = Renderer**; **ASG** = Atmospheric Scene Graph (the explicit state); **future-blind advection** = an extrapolation of the present that contains no information about the target frames.

---

## 1. Pipeline overview

```
INPUT  X_{t-k..t}  = radar VIL  +  IR069/IR107  +  water-vapor  +  GLM lightning
       context C   = CAPE/CIN/shear/PWAT/winds (HRRR/ERA5)  +  topo (DEM)  +  AFD environment text
   │
   ▼  STAGE A — PERCEPTION:  (X_{t-k..t}, C) ──► ASG_t   +   NL description        ◄ readout #1
   │            (autonomous: no human text in at inference)
   ▼  STAGE B — TRANSITION:  ASG_t ──► ASG_{t+h}   +   NL forecast rationale       ◄ readout #2
   │            advection + continuity + growth/decay constrained ── AMBIGUITY RESOLVED HERE
   ▼  ═══════════ FAITHFUL BOTTLENECK ═══════════
   │   decoder input  =  ASG_{t+h}   ⊕   FUTURE-BLIND advection(X_t)     (nothing else from X)
   ▼  STAGE C — RENDERER:  bottleneck ──► X_{t+1..t+n}   (exact field)
   │            advection-warp  +  ASG-guided residual;  mass-budget + non-neg + spectral losses
OUTPUT  forecast field (0–3 h)   +   full auditable world-model trace
```

The renderer's **only** window onto the future is the explicit state. This is the architectural commitment that makes the language faithful by construction (`idea.md` §3); it is realized as the Information-Bottleneck objective of §4.

---

## 2. Stage A — Perception → ASG_t

**Backbone.** A small VLM, QLoRA-4-bit: **SmolVLM-2.2B** (`marafioti2025smolvlm`) or **Qwen2.5-VL-3B** (`bai2025qwen25vl`). Backbone frozen; train LoRA adapters + the modality projector only (`hu2022lora`, `dettmers2023qlora`). Multi-channel radar/satellite frames are tiled to the VLM's native patch size (384×384 for SmolVLM); context `C` enters as (i) scalar fields concatenated as extra channels and (ii) **equation-aware text** in the prompt.

**Output = ASG_t**, a structured, decodable object set. Per storm object `o`:

| Field | Type | Notes |
|---|---|---|
| centroid `(y, x)` | float² (grid coords) | sub-pixel |
| area `a` | float (km²) | from connected-component / watershed on VIL |
| peak intensity `p` | float (VIL → dBZ) | |
| motion vector `v = (v_y, v_x)` | float² (km/h) | from optical flow |
| regime `r` | categorical | {initiation, growth, decay, steady} |
| growth–decay scalar `g` | float | dVIL/dt tendency |
| confidence `c` | float [0,1] | perception uncertainty |

Plus a **global** regime label and a low-resolution **growth–decay field** `G_t ∈ R^{H'×W'}`. The ASG serializes to a fixed token grammar (`ASG ↔ text`), so the NL description in readout #1 is a **constrained render of the ASG** — language cannot drift from the load-bearing state. This `ASG ↔ text` tie is the architecture-level anti-hallucination property.

---

## 3. Stage B — Transition → ASG_{t+h}

A **small transition transformer** over ASG tokens (≈10–50 M params): object tokens + context tokens → predicted object tokens at horizon `h`. It must emit predicted positions, **merge/split/initiation/decay** events, and an updated growth field, plus an NL forecast rationale (again a constrained render). Three injection points for governing equations (all compute-cheap — loss terms + a warp, no extra parameters):

1. **Differentiable advection operator.** Object centroids and the growth field are advanced by a semi-Lagrangian warp using the ASG motion field; the network predicts the *residual* on top of advection, so the linear-motion baseline is built in (mirrors the conservation idea in NowcastNet's evolution network, `zhang2023nowcastnet`).
2. **PINN-style residual losses** (`raissi2019pinn`): a **continuity / mass-conservation** residual on the predicted growth field and a smoothness/advection residual on the motion field, penalizing equation violation on the predicted *state*.
3. **Equation-aware prompting**: the governing relations (advection, continuity, growth–decay parameterization) are stated in the reasoning prompt so the VLM reasons *with* the equations, not prose alone.

This is the stage that resolves the under-constrained motion ambiguity: a short history admits many futures; physics + context **select** the meteorologically correct one.

---

## 4. The faithful bottleneck (the core mechanism)

**Definition.** The renderer input is exactly
$$Z \;=\; \big[\, \text{ASG}_{t+h}\ \oplus\ \text{advect}_{\text{blind}}(X_t)\,\big],$$
where `advect_blind` is a fixed extrapolation of the present (the same pysteps advection used to build labels) carrying **no** future information. No raw input latents, no skip connections from the encoder, reach Stage C.

**Information-theoretic justification.** Train the ASG to be a **minimal sufficient statistic** of the radar history for the future field via the Information Bottleneck (`tishby1999ib`, `tishby2015db`, `alemi2017vib`):
$$\mathcal{L}_{\text{IB}} \;=\; \beta\, I(\text{ASG}; X_{t-k:t}) \;-\; I(\text{ASG}; X_{t+1:t+n}).$$
The compression term ($\beta I(\text{ASG};X)$) forces the state to discard everything but predictive content; the prediction term forces it to keep what matters. In practice the prediction term is realized by the rendering loss through the bottleneck, and the compression term by capping ASG capacity (a fixed object budget + quantized attributes) and/or a variational penalty on continuous attributes.

**Why this makes the explanation faithful.** Because `Z` is the *only* path to `X_{t+1:t+n}`:
- **Zeroing** ASG_{t+h} (keep only `advect_blind`) ⇒ output **must** collapse to advection.
- **Perturbing** ASG_{t+h} ⇒ output **must** change in the direction the perturbation implies.

These are not hopes; they are entailed by the wiring and are tested in `eval.md` §C (intervention consistency, bottleneck ablation) and demonstrated live in the demo. This converts **counterfactual simulatability** (`chen2023counterfactual`) from a correlational probe into an architectural guarantee — the answer to the chain-of-thought unfaithfulness problem (`turpin2023cot`, `lanham2023faithfulness`, `chen2025reasoning`) for this domain.

---

## 5. Stage C — Renderer → field

**Form.** A **latent**, **few-step** generator to stay inside the compute envelope:
- Operate in a frozen **VAE latent** (≈8× spatial compression, SD-VAE style, `rombach2022ldm`); a 384×384 VIL field → ≈48×48 latent; train on **128×128 patches** → 16×16 latent tiles, short sequences.
- **Few-step rectified flow** (1–4 steps; `liu2023rectifiedflow`, alternatively flow matching `lipman2023flowmatching`, or consistency distillation `song2023consistency`) instead of 1000-step diffusion → cheap to train **and** sample. Backbone is a small latent U-Net or DiT (`peebles2023dit`).
- **Decomposition.** The field = advection-warp of `X_t` (the future-blind path) **+** an **ASG-guided residual** generated by the flow conditioned on `Z`. This residual-on-advection structure is what makes the bottleneck behavior crisp (zeroed ASG ⇒ residual ⇒ 0 ⇒ pure advection) and echoes the deterministic+stochastic split in DiffCast (`yu2024diffcast`).

**Physics-respecting losses on the field:** mass-budget consistency (rendered field's integrated water content tracks the ASG growth budget), non-negativity, and a **spectral / radially-averaged power-spectrum** term for realism (the realism diagnostic used by DGMR and LDCast, `ravuri2021dgmr`, `leinonen2023ldcast`).

---

## 6. Intervention-consistency (training + proof)

Add an **intervention-consistency loss** during Tier-2 training: sample a structured perturbation `δ` on ASG_{t+h} (translate a cell along its vector, flip its regime grow↔decay, scale `g`), render both, and require the *difference* field to match the *predicted* effect of `δ` (e.g., the cell's signal moves/intensifies as `δ` dictates). This trains the renderer to be **causally responsive** to the state and supplies the paired forward passes that the C-i experiment measures. It roughly doubles activation memory on the renderer forward — the reason Tier-2's end-to-end + intervention configuration is routed to the A100 (see `training_method.md`).

---

## 7. Concrete component budget (reproducibility)

| Component | Choice | Params | Trained? | Approx. peak VRAM (train) |
|---|---|---|---|---|
| Stage-A VLM | SmolVLM-2.2B / Qwen2.5-VL-3B, QLoRA NF4 | ~2–3 B (frozen) + adapters | adapters + projector | 8–12 GB (Tier 1) |
| Stage-B transition | ASG-token transformer | 10–50 M | full | <2 GB |
| VAE | frozen SD-VAE (8×) | ~80 M | no | ~0.3 GB |
| Stage-C renderer | latent rectified-flow U-Net/DiT, 1–4 steps | 100–400 M | full | 8–12 GB; ~16–20 GB with intervention pairs |
| Advection operator | semi-Lagrangian warp | 0 | no | negligible |

End-to-end Tier-2 peak ≈ **20–28 GB** with intervention-consistency on (VLM frozen in 4-bit, gradient checkpointing, latent + patches + short sequences) → comfortable on **A100-40 GB**, at the edge of **L4-24 GB**. Everything is engineered to ≤24–40 GB by design; nothing requires >40 GB.

---

## 8. What flows where (the contract that makes the paper)

- Stage A sees the input; Stage B sees only ASG_t (+ context tokens); Stage C sees only `Z = ASG_{t+h} ⊕ advect_blind`.
- The **only** future-bearing signal reaching the pixels is the predicted state. The future-blind path is auditable (leakage test, `eval.md` §C-iii).
- The NL at both readouts is a constrained render of the corresponding ASG, so prose is downstream of — and cannot contradict — the load-bearing state.

---

## 9. ASG ↔ text grammar (anti-hallucination contract)

The `ASG ↔ text` bijection is the architecture-level mechanism that prevents the VLM's natural-language output from drifting away from the load-bearing state.

**Serialization grammar.** Each storm object serializes to a fixed, machine-parseable token sequence:
```
OBJECT(id=<int>, cy=<float>, cx=<float>, area=<float_km2>, peak=<float_dBZ>,
       vy=<float_kmh>, vx=<float_kmh>, regime=<init|grow|decay|steady>,
       growth=<float>, conf=<float_0_1>)
```
Global context: `GLOBAL(regime=<init|grow|decay|steady>, n_objects=<int>)`. Units are fixed and labelled in the grammar tokens. The serialization is parsed deterministically into a typed ASG struct; misformed tokens are training-time errors, not inference-time fallbacks.

**Constrained decoding.** At inference, ASG token positions use constrained decoding (e.g., outlines / lm-format-enforcer) that restricts the vocabulary to valid grammar tokens. This prevents syntactic errors in the structured output without modifying model weights.

**NL as a downstream render — not a freeform channel.** The natural-language descriptions at readouts #1 and #2 are generated by a deterministic template function `render_NL(ASG)`:
- Template asserts only facts derivable from ASG fields: cell count, regime, directional motion ("moving northeast"), growth tendency sign ("intensifying" / "weakening"), dominant intensity class.
- The VLM is trained on LLM-polished versions of this template (`datasource.md` §5); it learns the template's prose style but not new semantic content, because the training signal never contains ungrounded additions.
- At inference, an automated cross-check flags any NL sentence asserting a fact absent from the parsed ASG. During evaluation these are reported; in production they are suppressed and replaced by the template sentence.

**Training loss weighting.** The VLM is supervised jointly on (ASG tokens, NL tokens): approximately 80% weight on ASG tokens (primary), 20% on NL tokens (secondary fluency). This ensures the model prioritizes structural accuracy over prose confidence.

---

## 10. VLM fine-tuning curriculum (knowledge is built, not assumed)

Out of the box, a small VLM has meteorological vocabulary from web pretraining but not forecaster judgment — domain knowledge that would be shallow and unreliable if used naively. The meteorological reasoning in this pipeline is built through a five-phase curriculum, each phase fine-tuning from the prior checkpoint, following the VQA → CoT curriculum principle established by the Skew-T VLM (`lee2025skewt`):

| Phase | Task | Input | Target output | L4 est. |
|---|---|---|---|---|
| **Ph-1** Visual VQA | Basic radar grounding | Frame + question | Short answer | 1–2 h |
| **Ph-2** Object description | ASG-faithful NL | Sequence | Grounded prose | 1–2 h |
| **Ph-3** Structured ASG output | Emit ASG grammar | Sequence + context C | ASG JSON | 2–4 h |
| **Ph-4** CoT reasoning | Full rationale chain | Sequence + context | Rationale → ASG_t → transition → ASG_{t+h} | 3–6 h |
| **Ph-5** Equation-aware CoT | Physics-grounded reasoning | Sequence + context + equations | Equation-grounded rationale → ASG | 2–4 h |

**Phase notes:**
- **Ph-1**: Procedurally generated question–answer pairs from auto-ASG (intensity, count, spatial, tendency queries). Maps radar visual patterns to language without requiring manual annotation.
- **Ph-3 gate** (hard stop): ASG F1 ≥ 0.70 on the hand-labeled gold subset. If the model cannot reliably emit the correct structured state, the downstream chain-of-thought is unfounded. Do not proceed to Ph-4 without passing the gate.
- **Ph-4**: The two-step chain (perceive → state, state → predict) is trained in a single prompt. The transition rationale is a templated render of `ASG_{t+h} − ASG_t` (change summary). The training format forces the model to produce the state *before* the rationale, enforcing causal order in the learned representation.
- **Ph-5**: Governing equations (advection, continuity/mass conservation, growth–decay parameterization) appear in the system prompt in both mathematical and verbal form. An equation-grounding check penalizes rationales that invoke equation vocabulary without referencing the correct physical quantities.

**What the curriculum gives (and doesn't give).** After Ph-5, the VLM produces reliable grounded ASG predictions and equation-referenced rationales for the SEVIR distribution. The ablation in `eval.md` §D separates NL-prior contribution from physics-equation contribution; neither is credited beyond what the ablation demonstrates.
