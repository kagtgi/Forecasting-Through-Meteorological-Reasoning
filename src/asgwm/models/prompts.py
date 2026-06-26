"""Stage-A prompt construction for the five-phase VLM curriculum.

The natural-language and structured outputs of Stage A are *constrained renders* of
the ASG (architecture.md sections 9-10); the prompts here set up each curriculum phase
so the model is asked for exactly the grounded target the data pipeline produces
(datasource.md section 5). Phase prompts are paired at training time with targets built
from ``grammar.serialize`` / ``render_NL`` / ``render_NL_delta``.

Phases (architecture.md section 10, datasource.md section 5):
    ph1_vqa   - Visual VQA: short grounded answers to procedural radar questions.
    ph2_desc  - Object description: ASG-faithful natural-language prose.
    ph3_asg   - Structured ASG output: emit the canonical grammar string.
    ph4_cot   - Chain-of-thought: observation rationale -> ASG_t -> transition
                rationale -> ASG_{t+h} (state BEFORE rationale, causal order).
    ph5_eqcot - Equation-aware CoT: identical to ph4 with the governing equations
                stated in the prompt so the model reasons *with* the physics.

This module depends only on the ASG package (no torch), so it imports cleanly anywhere.
"""
from __future__ import annotations

from typing import Dict, Optional

from asgwm.asg import REGIMES
from asgwm.asg.grammar import allowed_regime_tokens

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

# The fixed grammar contract restated for the model (architecture.md section 9).
GRAMMAR_BLOCK: str = (
    "Emit the Atmospheric Scene Graph (ASG) in this exact grammar, one line each:\n"
    "GLOBAL(regime=<init|grow|decay|steady>, n_objects=<int>)\n"
    "OBJECT(id=<int>, cy=<float>, cx=<float>, area=<float_km2>, peak=<float_dBZ>, "
    "vy=<float_kmh>, vx=<float_kmh>, regime=<init|grow|decay|steady>, "
    "growth=<float>, conf=<float_0_1>)\n"
    "One GLOBAL line, then one OBJECT line per storm cell. "
    "Units are fixed by position; emit no other text."
)

# Governing equations stated both mathematically and verbally (datasource.md section 5,
# architecture.md section 3). Injected for Ph-5 (and available to the transition rationale).
EQUATION_BLOCK: str = (
    "Governing equations (reason WITH these, not prose alone):\n"
    "1. Advection (Lagrangian form): dphi/dt + v . grad(phi) = 0  "
    "-- precipitation features are transported by the motion vector v=(vy,vx); "
    "centroids advance along v over the horizon.\n"
    "2. Continuity / mass-conservation: the integrated VIL tendency balances flux "
    "convergence -- growth requires net convergence, decay net divergence.\n"
    "3. Growth-decay parameterization: the convective tendency g is forced by the "
    "environment -- it increases with CAPE, is suppressed by CIN, and is organized by "
    "vertical wind shear. Use the context scalars (cape, cin, shear, pwat) accordingly."
)

# Context-field documentation shared by Ph-3..Ph-5 prompts.
_CONTEXT_KEYS = ("cape", "cin", "shear", "pwat", "dem")


def _format_context(context: Optional[Dict[str, float]]) -> str:
    """Render the co-located environmental scalars as a compact, fixed-order line.

    Returns an empty string when context is unavailable so prompts degrade gracefully
    (context.colocate_context returns zeros + context_available=0 when sources absent).
    """
    if not context:
        return ""
    available = context.get("context_available", 1)
    try:
        available = float(available)
    except (TypeError, ValueError):
        available = 1.0
    if available <= 0:
        return "Environmental context: unavailable."
    parts = []
    for k in _CONTEXT_KEYS:
        if k in context and context[k] is not None:
            try:
                parts.append(f"{k}={float(context[k]):.4g}")
            except (TypeError, ValueError):
                continue
    if not parts:
        return ""
    return "Environmental context (HRRR/ERA5 + DEM): " + ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# Per-phase prompt templates
# ---------------------------------------------------------------------------
_REGIME_LIST = ", ".join(allowed_regime_tokens())
assert tuple(allowed_regime_tokens()) == tuple(REGIMES)  # grammar/schema agreement

_PH1_VQA = (
    "You are a radar nowcasting assistant. Look at the provided radar/satellite frames "
    "and answer the question with a short, factual response grounded only in what is "
    "visible. Do not speculate beyond the imagery.\n"
    "{context}"
    "Question: {question}\n"
    "Answer:"
)

_PH2_DESC = (
    "You are a meteorological analyst. Describe the precipitation scene in the provided "
    "radar/satellite frames in natural language. State only facts grounded in the scene: "
    "the number of cells, the overall regime, each cell's intensity class "
    "(light/moderate/heavy), its compass direction of motion, and its growth tendency "
    "(intensifying/weakening/steady). Do not invent specific numeric values.\n"
    "{context}"
    "Description:"
)

_PH3_ASG = (
    "You are a radar nowcasting perception model. From the provided radar/satellite "
    "frames and the environmental context, identify the storm cells and emit the "
    "current Atmospheric Scene Graph.\n"
    "{context}"
    f"Valid regimes: {_REGIME_LIST}.\n"
    "{grammar}\n"
    "ASG:"
)

_PH4_COT = (
    "You are a meteorological reasoning model performing radar nowcasting. From the "
    "provided radar/satellite frames and environmental context, reason about the current "
    "scene and its evolution over the forecast horizon.\n"
    "{context}"
    f"Valid regimes: {_REGIME_LIST}.\n"
    "{grammar}\n"
    "Produce your output in this exact order:\n"
    "OBSERVATION: <one-paragraph rationale describing the present scene>\n"
    "ASG_T:\n<the current ASG in grammar>\n"
    "TRANSITION: <one-paragraph rationale for how the scene evolves over the horizon>\n"
    "ASG_TH:\n<the forecast ASG in grammar>\n"
    "Emit the ASG state lines before each rationale section as labelled above."
)

_PH5_EQCOT = (
    "You are a physics-grounded meteorological reasoning model performing radar "
    "nowcasting. Use the governing equations below explicitly in your reasoning.\n"
    "{equations}\n"
    "{context}"
    f"Valid regimes: {_REGIME_LIST}.\n"
    "{grammar}\n"
    "Produce your output in this exact order:\n"
    "OBSERVATION: <present-scene rationale>\n"
    "ASG_T:\n<the current ASG in grammar>\n"
    "TRANSITION: <forecast rationale that references the relevant physical quantities: "
    "the motion vector (advection), continuity/convergence, and the growth forcing from "
    "CAPE/CIN/shear>\n"
    "ASG_TH:\n<the forecast ASG in grammar>"
)

# Public mapping of phase name -> prompt template (architecture.md section 10).
PHASE_PROMPTS: Dict[str, str] = {
    "ph1_vqa": _PH1_VQA,
    "ph2_desc": _PH2_DESC,
    "ph3_asg": _PH3_ASG,
    "ph4_cot": _PH4_COT,
    "ph5_eqcot": _PH5_EQCOT,
}


def build_prompt(phase: str, context: Optional[Dict[str, float]] = None, **kw) -> str:
    """Build the prompt string for a curriculum ``phase``.

    Cites architecture.md section 10 / datasource.md section 5. ``context`` is the
    co-located environmental scalar dict (cape/cin/shear/pwat/dem [+ context_available]).
    Ph-1 additionally accepts ``question=<str>`` (the procedural VQA question).

    Args:
        phase: one of ``PHASE_PROMPTS`` (ph1_vqa, ph2_desc, ph3_asg, ph4_cot, ph5_eqcot).
        context: environmental scalar dict; rendered into the prompt when present.
        **kw: phase-specific fields (e.g. ``question`` for ph1_vqa).

    Returns:
        The fully formatted prompt string.
    """
    if phase not in PHASE_PROMPTS:
        raise ValueError(f"unknown phase {phase!r}; expected one of {list(PHASE_PROMPTS)}")
    template = PHASE_PROMPTS[phase]
    ctx_line = _format_context(context)
    ctx_block = (ctx_line + "\n") if ctx_line else ""
    fields: Dict[str, str] = {
        "context": ctx_block,
        "grammar": GRAMMAR_BLOCK,
        "equations": EQUATION_BLOCK,
        "question": kw.get("question", "What is happening in this radar scene?"),
    }
    # Only substitute the placeholders this template actually contains.
    return template.format(**fields)
