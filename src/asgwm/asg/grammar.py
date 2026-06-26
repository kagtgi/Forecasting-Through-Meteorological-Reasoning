"""ASG <-> text grammar (architecture.md section 9).

A fixed, machine-parseable token grammar so the NL readout is a constrained render of
the load-bearing state. `serialize` produces the canonical string; `parse` recovers a
typed ASG; `grammar_regex`/`allowed_tokens` support constrained decoding at inference.

Grammar:
    GLOBAL(regime=<r>, n_objects=<int>)
    OBJECT(id=<int>, cy=<f>, cx=<f>, area=<f>, peak=<f>, vy=<f>, vx=<f>,
           regime=<r>, growth=<f>, conf=<f>)
one GLOBAL line followed by one OBJECT line per cell. Units are fixed by position.
"""
from __future__ import annotations

import re
from typing import List

from .schema import ASG, StormObject, REGIMES

_REGIME_ALT = "|".join(REGIMES)
_FLOAT = r"-?\d+(?:\.\d+)?"

_GLOBAL_RE = re.compile(
    rf"GLOBAL\(regime=(?P<regime>{_REGIME_ALT}),\s*n_objects=(?P<n>\d+)\)"
)
_OBJECT_RE = re.compile(
    r"OBJECT\("
    rf"id=(?P<id>\d+),\s*"
    rf"cy=(?P<cy>{_FLOAT}),\s*cx=(?P<cx>{_FLOAT}),\s*"
    rf"area=(?P<area>{_FLOAT}),\s*peak=(?P<peak>{_FLOAT}),\s*"
    rf"vy=(?P<vy>{_FLOAT}),\s*vx=(?P<vx>{_FLOAT}),\s*"
    rf"regime=(?P<regime>{_REGIME_ALT}),\s*"
    rf"growth=(?P<growth>{_FLOAT}),\s*conf=(?P<conf>{_FLOAT})\)"
)


def serialize_object(o: StormObject) -> str:
    return (
        f"OBJECT(id={o.id}, cy={o.cy:.2f}, cx={o.cx:.2f}, "
        f"area={o.area:.1f}, peak={o.peak:.1f}, "
        f"vy={o.vy:.1f}, vx={o.vx:.1f}, regime={o.regime}, "
        f"growth={o.growth:.3g}, conf={o.conf:.2f})"
    )


def serialize(asg: ASG) -> str:
    """ASG -> canonical grammar string."""
    lines = [f"GLOBAL(regime={asg.global_regime}, n_objects={asg.n_objects})"]
    for o in asg.objects:
        lines.append(serialize_object(o))
    return "\n".join(lines)


def parse(text: str) -> ASG:
    """Grammar string -> typed ASG. Tolerant of surrounding prose / whitespace.

    Misformed object lines are skipped (logged by caller if desired); during training
    a strict parser should be used so malformed tokens are training-time errors.
    """
    global_regime = "steady"
    gm = _GLOBAL_RE.search(text)
    if gm:
        global_regime = gm.group("regime")
    objects: List[StormObject] = []
    for m in _OBJECT_RE.finditer(text):
        objects.append(
            StormObject(
                id=int(m.group("id")),
                cy=float(m.group("cy")),
                cx=float(m.group("cx")),
                area=float(m.group("area")),
                peak=float(m.group("peak")),
                vy=float(m.group("vy")),
                vx=float(m.group("vx")),
                regime=m.group("regime"),
                growth=float(m.group("growth")),
                conf=float(m.group("conf")),
            )
        )
    return ASG(objects=objects, global_regime=global_regime)


def parse_strict(text: str) -> ASG:
    """Like `parse` but raises if any non-empty, non-GLOBAL line fails to parse."""
    asg = parse(text)
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("GLOBAL("):
            continue
        if not _OBJECT_RE.fullmatch(line):
            raise ValueError(f"malformed ASG line: {line!r}")
    return asg


def grammar_field_order() -> List[str]:
    return ["id", "cy", "cx", "area", "peak", "vy", "vx", "regime", "growth", "conf"]


def allowed_regime_tokens() -> List[str]:
    """Vocabulary restriction for constrained decoding at the `regime=` position."""
    return list(REGIMES)


# Compact regex describing one valid OBJECT line — usable to drive a constrained
# decoder (outlines / lm-format-enforcer) so the model can only emit valid grammar.
def object_line_regex() -> str:
    return _OBJECT_RE.pattern
