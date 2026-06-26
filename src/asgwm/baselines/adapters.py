"""Concrete baseline adapters (eval.md section 1A).

Headline comparison set: pysteps (extrapolation), RainNet (CNN), NowcastNet
(physics-generative), LangPrecip (language-guided), ThoR (our prior physics-informed).

Only ``pysteps`` is implemented now — it is free and already lives in the codebase as the
future-blind advection path. The four neural baselines are stubs (``is_available()==False``)
that the harness renders as ``TBR``; each carries a docstring recipe for plugging in later.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .base import Baseline, register


class PystepsBaseline(Baseline):
    """Lagrangian extrapolation (pysteps S-PROG / semi-Lagrangian advection).

    Implemented via :func:`asgwm.data.advection.future_blind_baseline`, the exact
    future-blind advection used as Stage-C's bypass path, so this row is free.
    """

    name, display, family = "pysteps", "pysteps", "Extrapolation"

    def is_available(self) -> bool:
        return True

    def predict(self, frames_hist: np.ndarray, context: Dict, n_out: int) -> np.ndarray:
        from asgwm.data.advection import future_blind_baseline
        return np.asarray(future_blind_baseline(np.asarray(frames_hist), n_out), dtype=np.float32)


class _NotYetImplemented(Baseline):
    """Base for baselines coded/obtained later. Reports unavailable; harness writes TBR."""

    def is_available(self) -> bool:
        return False

    def predict(self, frames_hist: np.ndarray, context: Dict, n_out: int) -> np.ndarray:
        raise RuntimeError(
            f"baseline '{self.name}' is not yet implemented. "
            f"Implement predict() and is_available() in asgwm/baselines/adapters.py."
        )


class RainNetBaseline(_NotYetImplemented):
    """RainNet (Ayzel et al., 2020) — U-Net regression on radar.

    Plug-in recipe: load the trained U-Net checkpoint in ``__init__``; in ``predict``
    autoregress on ``frames_hist`` for ``n_out`` steps; return the stacked sequence.
    Set ``is_available -> True`` once the checkpoint path resolves.
    """

    name, display, family = "rainnet", "RainNet", "CNN / U-Net"


class NowcastNetBaseline(_NotYetImplemented):
    """NowcastNet (Zhang et al., 2023) — physics-informed generative nowcaster.

    Plug-in recipe: load the evolution + generative networks; ``predict`` runs the
    official sampler for ``n_out`` frames. Use ``predict_ensemble`` for CRPS.
    """

    name, display, family = "nowcastnet", "NowcastNet", "Physics-generative"


class LangPrecipBaseline(_NotYetImplemented):
    """LangPrecip (Ling et al., 2025) — language-guided rectified-flow nowcaster.

    Plug-in recipe: supply the text condition (their motion description) + radar; run
    the rectified-flow generator. Note this consumes human/derived text, unlike ASG-WM.
    """

    name, display, family = "langprecip", "LangPrecip", "Language-guided"


class ThoRBaseline(_NotYetImplemented):
    """ThoR (Ta Gia et al., 2025) — our prior physics-informed TFC framework.

    Plug-in recipe: load the ThoR generator + motion network; ``predict`` runs its
    advection-diffusion-constrained rollout for ``n_out`` frames.
    """

    name, display, family = "thor", "ThoR", "Physics-informed (ours-prior)"


# Registration order defines table/figure ordering (extrapolation -> ... -> ours-prior).
for _b in (PystepsBaseline(), RainNetBaseline(), NowcastNetBaseline(),
           LangPrecipBaseline(), ThoRBaseline()):
    register(_b)
