"""Pluggable baseline registry (eval.md section 1A).

ASG-WM is evaluated against a focused, family-spanning baseline set. We produce ASG-WM
results now; the five comparison methods are coded / obtained later and slot in here
without touching the eval harness or the figure/table code.

A baseline implements :class:`Baseline`. Until its weights/code exist it reports
``is_available() == False`` and the harness writes a ``TBR`` row for it, so every figure
and table renders today with the ASG-WM row populated and baseline rows pending.

To add a baseline later: implement ``predict`` (and flip ``is_available``) in
``adapters.py`` — nothing else changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np


class Baseline(ABC):
    """A nowcasting baseline that maps a radar history to a forecast sequence."""

    name: str = "baseline"        # registry key (lowercase)
    display: str = "Baseline"     # name shown in tables/figures
    family: str = "?"             # family label for the skill table

    @abstractmethod
    def is_available(self) -> bool:
        """True once weights/code are present so the harness should run it."""

    @abstractmethod
    def predict(self, frames_hist: np.ndarray, context: Dict, n_out: int) -> np.ndarray:
        """Deterministic forecast.

        Args:
            frames_hist: ``[T, H, W]`` VIL history (the model's input window).
            context: co-located environmental scalars (CAPE/CIN/shear/...).
            n_out: number of future frames to predict.
        Returns:
            ``[n_out, H, W]`` forecast sequence.
        """

    def predict_ensemble(self, frames_hist: np.ndarray, context: Dict,
                         n_out: int, k: int = 1) -> np.ndarray:
        """Ensemble forecast ``[k, n_out, H, W]``. Deterministic baselines tile the mean."""
        return np.stack([self.predict(frames_hist, context, n_out) for _ in range(k)], axis=0)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Baseline] = {}


def register(b: Baseline) -> Baseline:
    _REGISTRY[b.name] = b
    return b


def get(name: str) -> Optional[Baseline]:
    return _REGISTRY.get(name)


def all_names() -> List[str]:
    """Every registered baseline, in registration order."""
    return list(_REGISTRY)


def available_names() -> List[str]:
    """Baselines whose code/weights are present (``is_available()``)."""
    return [n for n, b in _REGISTRY.items() if b.is_available()]


def display_name(name: str) -> str:
    b = _REGISTRY.get(name)
    return b.display if b else name


def family(name: str) -> str:
    b = _REGISTRY.get(name)
    return b.family if b else "?"
