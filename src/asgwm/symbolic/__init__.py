"""Symbolic admissibility layer (prototype).

A neuro-symbolic verifier that sits between Stage B (transition) and Stage C (renderer):
given a predicted (ASG_t -> ASG_{t+h}) transition it returns a *physical-admissibility
certificate* (SAT) or the violated-constraint core (UNSAT), and a dual-SAT
*ambiguity* verdict over the context envelope. This upgrades ASG-WM's physics from a soft
PINN penalty to a checkable certificate, and defines forecast uncertainty as the set of
physically-admissible futures consistent with the observations.

Idea ported from the EXACT neuro-symbolic pipeline (LLM proposes, Z3/SymPy verifies);
here the solver checks the low-dimensional ASG state, NOT the continuous pixel field.
"""
from .admissibility import (  # noqa: F401
    ConstraintBounds,
    Certificate,
    certify_transition,
    REGIME_FSM,
    ambiguity_flag,
    admissible_regimes,
)
