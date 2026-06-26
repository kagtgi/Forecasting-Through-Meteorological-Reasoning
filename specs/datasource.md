# datasource.md — Committed free data stack + ASG labeling

> **Discipline: subset surgically.** A credible benchmark needs a curated subset, not the whole archive. Storage stays in the **tens of GB** so it fits a free notebook plus a persistent bucket. Everything below is free for research.

---

## 1. Data roles and access (verified)

| Role | Source | Access (free) | Cost-minimizing use | Cite |
|---|---|---|---|---|
| **Primary radar (multimodal)** | **SEVIR** — US/CONUS: VIL radar + GOES-16 IR069/IR107 + visible + GLM lightning | AWS Open Data `s3://sevir` | Download **VIL + IR + GLM only**, ~2–3k events for train + the standard test split. *Not* the full archive. | `veillette2020sevir` |
| **Cross-region (pick one)** | **HKO-7** (Hong Kong) **or** **MeteoNet** (France) | HKO-7 via authors/GitHub; MeteoNet = Météo-France open data, Etalab 2.0 | One region for generalization evidence; **don't** train on both. | `shi2017trajgru`, `larvor2020meteonet` |
| **Convective context (CONUS)** | **HRRR** — CAPE, CIN, shear (convection-allowing, 3 km) | AWS Open Data `s3://noaa-hrrr-bdp-pds` | Best context for SEVIR; pull only needed GRIB2 fields at event time. | `noaa_hrrr` |
| **Atmospheric context (global)** | **ARCO-ERA5** (analysis-ready zarr) | Google Cloud public `gs://gcp-public-data-arco-era5` | Slice CAPE/PWAT/winds/shear at event lat/lon/time → a few MB/event. | `carver2023arco` |
| **Topography** | **Copernicus DEM (GLO-30) / SRTM** | AWS Open Data `s3://copernicus-dem-30m` | Static field; fetch once. | `copernicus_dem` |
| **Reasoning text — environment** | **NWS Area Forecast Discussions** | **Iowa Environmental Mesonet (IEM)** text-product archive + API | Queryable by WFO/time. **Caveat:** issued a few times/day → **synoptic-period context, NOT per-frame labels.** | `iem_afd` |
| **Reasoning text — extremes** | **NOAA Storm Events Database** | NCEI bulk CSV | Extreme-case grounding + validation; SEVIR events are already matched to it. | `noaa_storm_events` |
| **Radar–text prior art (relate, don't lean on)** | **LangPrecip-160K** (160K radar–text pairs, Swedish + MRMS) | arXiv 2512.22317 project release *(verify license)* | Reference/benchmark for the language-guided line; **not** our source of novelty (we *produce* text, not consume it). | `ling2025langprecip` |

**MRMS caveat (verified).** Multi-Radar Multi-Sensor (`smith2016mrms`) is real and ~1 km / 2-min over CONUS, but the **live public feed is short-retention**. Multi-year archives do exist (Iowa State `mtarchive`; AWS `noaa-mrms-pds`), so MRMS is usable for a demo or a small auxiliary slice — not the multimodal training headline, where SEVIR (which bundles the satellite + lightning channels) is the right primary.

---

## 2. ASG label construction (free, CPU)

The Atmospheric Scene Graph is **auto-labeled** with classical tools — no GPU, no manual annotation at scale:

1. **Motion + tracking** with **pysteps** (`pulkkinen2019pysteps`): Lucas–Kanade / VET optical flow → motion vectors; cell identification + tracking → object tracks; advection fields (this same advection becomes Stage-C's **future-blind** path and the Tier-0 baseline).
2. **Tendency / events from future frames**: growth–decay scalar `g = dVIL/dt`; merge/split/initiation/decay from track topology over the window.
3. **Regime label** from morphology + tendency (initiation / growth / decay / steady).
4. **Context co-location**: slice HRRR (CONUS) and ARCO-ERA5 (global) fields and DEM at each event's lat/lon/time; attach CAPE/CIN/shear/PWAT/winds/topo to each ASG.
5. **NL rationale**: templated from the ASG, fluency-polished by a free-tier LLM, **fact-anchored** to ASG attributes (so text never asserts anything absent from the state).
6. **Validation**: hand-label a **few-hundred-window** subset to validate the auto-rules; report ASG-accuracy vs this gold set (and vs forecasters where possible) in `eval.md` §C.

*Prior-art note:* LLM-captioned weather datasets already exist, so **captioning is supervision, not the novelty**; the novelty is the faithful ASG→field world model. Validate the auto-labels; do not treat them as ground truth.

---

## 3. Subsetting + caching strategy (built for Colab)

- **Subset**: VIL + IR + GLM only; ~2–3k rainy events + the standard SEVIR test split; one cross-region set. **Rainy-oversample** windows (≥80 % rain pixels) so compute isn't spent on clear sky.
- **Cache once, reuse forever** — this is the single biggest time-saver under <12 h sessions and ephemeral Colab disk: precompute and store to a **persistent GCS bucket (or Drive)**: (a) the ASG labels, (b) the ERA5/HRRR/DEM context slices, (c) frozen VLM visual features, (d) the future-blind advection fields, (e) all checkpoints. **Run the pysteps labeling pass once** (CPU, slow) and freeze it; never recompute per session.
- **Footprint**: tens of GB total — fits a free notebook plus a cheap bucket because of the subsetting above.

See `training_method.md` §6 for how this caching interacts with the L4 + spot-A100 session plan.

---

## 4. No-hallucination ledger

**Committed (stable, free, verified):** SEVIR on AWS Open Data; HRRR on AWS Open Data; ARCO-ERA5 on Google Cloud public; Copernicus DEM / SRTM; AFD archive via IEM; NOAA Storm Events via NCEI; HKO-7 / MeteoNet free for research; pysteps for labeling. Free compute: Kaggle (T4×2 / P100) + Colab (always-on L4-24 GB; on-demand A100-40 GB).

**Verify before relying:** exact SEVIR footprint/channels for your final subset; **LangPrecip-160K** release license (arXiv 2512.22317); HKO-7 current access route; any MRMS archive endpoint you depend on.

**Corrected from earlier drafts:** SEVIR is US/CONUS (not Chinese); MRMS *live feed* is short-retention (archives exist; demo/auxiliary only, not bulk training); AFDs are synoptic-period context (not per-frame labels); the "curated UK low-res radar, arXiv 2512.17924" item from earlier inputs **did not verify** — use the UK Met Office RadarNet/NIMROD composites (as in DGMR, `ravuri2021dgmr`) if a UK set is wanted; "LangPrecip-160K" **did verify** and is real (Ling et al., 2025); dropped the from-scratch large-GPU plan in favor of the tiered recipe.

---

## 5. Fine-tuning data pipeline (curriculum training pairs)

The VLM's meteorological reasoning is built from the auto-ASG pipeline through a five-phase curriculum (`architecture.md` §10). Training pairs are constructed per phase as follows. **The gold subset (~200–500 hand-labeled windows) is held out from all fine-tuning and used only for ASG-accuracy evaluation.**

### Ph-1: Visual VQA pairs
Procedurally generated (image, question, answer) triples from auto-labeled events — no manual annotation:
- **Intensity**: "What is the approximate peak VIL in the northeastern quadrant?" → from ASG `peak` field.
- **Count**: "How many storm cells are present?" → `n_objects`.
- **Spatial**: "Is precipitation concentrated in the northern or southern half of the frame?" → from centroid distribution.
- **Tendency**: "Is the dominant cell growing, decaying, or stable?" → from `regime` + `growth`.
- **Scale**: ~20–50K pairs from the SEVIR subset; procedurally unlimited given the labels.

### Ph-2: ASG-faithful object descriptions
Template → LLM-polish pipeline:
1. **Template** `render_NL(ASG)`: deterministic function emitting one sentence per object + one global summary. Asserts only facts in the ASG (directional motion, regime, growth sign, intensity class). No raw numerical values — coarse ranges only ("northwest," "moderate intensity," "intensifying").
2. **LLM polish** (free-tier): prompt = "Rewrite for natural meteorological phrasing. Do not add facts not in the template. Do not introduce specific numerical values." An automated assertion extractor checks each polished sentence against the original ASG; sentences asserting ungrounded facts are replaced by the template sentence.
3. **Scale**: ~2–3K events × multiple horizons → ~10–20K pairs.

### Ph-3: Structured ASG output pairs
(radar_sequence, context_C) → ASG_t in JSON grammar.
- Target: the auto-labeled ASG_t.
- Loss: structured token loss on ASG grammar fields only; NL tokens suppressed in this phase.
- **Gate**: ASG F1 ≥ 0.70 on the gold subset before proceeding to Ph-4.

### Ph-4: Chain-of-thought pairs
(radar_sequence, context_C) → (observation_rationale, ASG_t, transition_rationale, ASG_{t+h}).
- **Observation rationale**: `render_NL(ASG_t)` (from Ph-2 pipeline).
- **Transition rationale**: `render_NL_delta(ASG_t, ASG_{t+h})` — change summary: which cells moved, which grew/decayed, what initiated or dissipated, and the dominant physical driver (boundary, instability release, etc.) from context fields.
- Trained as a single prompt/completion; the model must produce the state before the rationale, enforcing causal order.

### Ph-5: Equation-aware CoT pairs
Identical to Ph-4, but with governing equations added to the system prompt:
- Advection equation (Lagrangian form): `∂φ/∂t + v·∇φ = 0`.
- Mass-conservation / continuity: VIL integrated tendency vs convergence.
- Growth–decay parameterization: convective tendency as a function of CAPE, CIN, and vertical wind shear.
- **Equation-grounding check**: each training pair's transition rationale is verified (rule-based) to reference the physically relevant quantities (motion vector, continuity, growth forcing) in alignment with the stated equations. Pairs that fail the check are replaced with the template rationale.

### Quality controls across all phases
- **Automated assertion check**: NL claims cross-checked against ASG using a rule-based extractor; ungrounded sentences removed before the pair enters training.
- **Gold-set isolation**: the hand-labeled subset is never seen during fine-tuning.
- **Polished NL diversity**: vary the LLM-polish prompt slightly across the dataset (e.g., different instruction phrasing) to prevent the VLM from overfitting to a single surface form.
