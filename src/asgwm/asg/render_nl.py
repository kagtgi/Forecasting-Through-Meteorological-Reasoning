"""Deterministic NL render of the ASG + an automated assertion check.

The natural-language readouts (#1 perception, #2 transition) are *constrained renders*
of the ASG, never a freeform channel (architecture.md section 9, datasource.md section 5).
`render_NL` and `render_NL_delta` produce the templated prose used to build curriculum
pairs; `assertion_check` flags any NL sentence asserting a fact absent from the parsed
ASG (used at eval time to measure / suppress hallucination).

Only coarse, ASG-derivable facts are asserted: cell count, regime, compass direction of
motion, growth-tendency sign, and intensity class. No raw numeric values are emitted.
"""
from __future__ import annotations

from typing import List

from .schema import ASG, StormObject, intensity_class, motion_to_compass

_TENDENCY = {
    "grow": "intensifying",
    "decay": "weakening",
    "init": "newly initiating",
    "steady": "holding steady",
}


def _object_sentence(o: StormObject, ordinal: str) -> str:
    direction = motion_to_compass(o.vy, o.vx)
    inten = intensity_class(o.peak)
    if o.growth > 0.05:
        tend = "intensifying"
    elif o.growth < -0.05:
        tend = "weakening"
    else:
        tend = _TENDENCY.get(o.regime, "holding steady")
    move = "nearly stationary" if direction == "stationary" else f"moving {direction}"
    return f"The {ordinal} cell is a {inten}-intensity system {move} and {tend}."


_ORDINALS = ["primary", "secondary", "third", "fourth", "fifth"]


def render_NL(asg: ASG) -> str:
    """One global summary sentence + one sentence per (top) object."""
    n = asg.n_objects
    if n == 0:
        return "No organized precipitation cells are present in the domain."
    head = (
        f"{n} precipitation cell{'s' if n != 1 else ''} present; "
        f"the overall regime is {_REGIME_PHRASE.get(asg.global_regime, asg.global_regime)}."
    )
    sents = [head]
    for i, o in enumerate(asg.objects[: len(_ORDINALS)]):
        sents.append(_object_sentence(o, _ORDINALS[i]))
    return " ".join(sents)


_REGIME_PHRASE = {
    "init": "convective initiation",
    "grow": "rapid growth",
    "decay": "decay",
    "steady": "steady advection",
}


def render_NL_delta(asg_t: ASG, asg_th: ASG) -> str:
    """Change summary between two ASGs: moved / grew / decayed / initiated / dissipated."""
    by_id_t = {o.id: o for o in asg_t.objects}
    by_id_th = {o.id: o for o in asg_th.objects}
    moved, grew, decayed, initiated, dissipated = [], [], [], [], []
    for oid, o2 in by_id_th.items():
        o1 = by_id_t.get(oid)
        if o1 is None:
            initiated.append(oid)
            continue
        if abs(o2.cy - o1.cy) + abs(o2.cx - o1.cx) > 2.0:
            moved.append(motion_to_compass(o2.cy - o1.cy, o2.cx - o1.cx))
        if o2.peak - o1.peak > 2.0:
            grew.append(oid)
        elif o1.peak - o2.peak > 2.0:
            decayed.append(oid)
    for oid in by_id_t:
        if oid not in by_id_th:
            dissipated.append(oid)

    parts: List[str] = []
    if initiated:
        parts.append(f"{len(initiated)} new cell{'s' if len(initiated) != 1 else ''} initiate")
    if grew:
        parts.append(f"{len(grew)} cell{'s' if len(grew) != 1 else ''} intensify")
    if decayed:
        parts.append(f"{len(decayed)} cell{'s' if len(decayed) != 1 else ''} weaken")
    if dissipated:
        parts.append(f"{len(dissipated)} cell{'s' if len(dissipated) != 1 else ''} dissipate")
    if moved:
        parts.append("cells advect " + "/".join(sorted(set(moved))))
    if not parts:
        return "The system evolves with little net change over the forecast horizon."
    driver = _physical_driver(asg_t)
    return "Over the horizon, " + ", ".join(parts) + f". Dominant driver: {driver}."


def _physical_driver(asg: ASG) -> str:
    cape = asg.context.get("cape")
    shear = asg.context.get("shear")
    if cape is not None and cape > 1500:
        return "instability release (high CAPE)"
    if shear is not None and shear > 20:
        return "organized shear-driven advection"
    return "kinematic advection"


# --- vocabulary the assertion checker recognizes -----------------------------
_INTENSITY_WORDS = {"light", "moderate", "heavy"}
_TENDENCY_WORDS = {"intensifying", "weakening", "growing", "decaying", "steady", "holding"}
_DIRECTION_WORDS = {"north", "south", "east", "west",
                    "northeast", "northwest", "southeast", "southwest"}


def assertion_check(nl: str, asg: ASG) -> List[str]:
    """Return sentences in `nl` asserting facts not grounded in `asg`.

    Rule-based: a sentence is flagged if it asserts an intensity / tendency /
    direction not present anywhere in the ASG object set.
    """
    asg_intens = {intensity_class(o.peak) for o in asg.objects}
    asg_dirs = {motion_to_compass(o.vy, o.vx) for o in asg.objects}
    asg_dir_words = set()
    _abbr = {"N": "north", "S": "south", "E": "east", "W": "west",
             "NE": "northeast", "NW": "northwest", "SE": "southeast", "SW": "southwest"}
    for d in asg_dirs:
        if d in _abbr:
            asg_dir_words.add(_abbr[d])
    asg_tend = set()
    for o in asg.objects:
        if o.growth > 0.05:
            asg_tend |= {"intensifying", "growing"}
        elif o.growth < -0.05:
            asg_tend |= {"weakening", "decaying"}
        else:
            asg_tend |= {"steady", "holding"}

    flagged: List[str] = []
    for sent in _split_sentences(nl):
        low = sent.lower()
        for w in _INTENSITY_WORDS:
            if w in low and w not in asg_intens and asg.n_objects > 0:
                flagged.append(sent)
                break
        else:
            for w in (_DIRECTION_WORDS):
                if w in low and w not in asg_dir_words and asg.n_objects > 0:
                    flagged.append(sent)
                    break
    return flagged


def _split_sentences(text: str) -> List[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
