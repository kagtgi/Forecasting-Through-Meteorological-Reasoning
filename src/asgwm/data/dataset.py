"""Torch Datasets for all three training tiers (datasource.md sections 2, 5).

These datasets consume the **cached** artifacts produced once by the labeling pass
(``paths.cache/asg/<event>.json``) plus the cached SEVIR/SyntheticSEVIR events
(``paths.cache/events``), so every session reuses the slow CPU work
(training_method.md section 6).

Three consumers, three datasets:
  * :class:`ASGTransitionDataset` -- Tier-0/2 transition supervision: (ASG_t, context,
    ASG_{t+h}, dense flow).
  * :class:`RendererDataset` -- Tier-0/2 renderer supervision: (ASG_{t+h},
    advect_blind, target field, growth budget) on ``data.patch`` crops.
  * :class:`VLMCurriculumDataset` -- Tier-1 five-phase curriculum (Ph-1..Ph-5),
    building (images, prompt, target) text pairs from ``grammar.serialize`` and
    ``render_NL`` / ``render_NL_delta`` (datasource.md section 5).

Plus the two collate functions consumed by the trainers (interface contract):
``collate_transition`` and ``collate_renderer``.

torch is assumed for dataset/model files (coding standards) but everything else is
numpy + the foundation, so these run on CPU with no download.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..asg import (
    ASG,
    ASGSequence,
    StormObject,
    REGIME_TO_IDX,
    render_NL,
    render_NL_delta,
    serialize,
)
from ..utils.config import Config
from . import sevir as sevir_mod
from .advection import advect_blind

CONTEXT_FIELDS_DEFAULT = ["cape", "cin", "shear", "pwat", "dem"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _context_vector(asg: ASG, fields: Sequence[str]) -> torch.Tensor:
    """Pack the ASG context dict into a fixed-length [len(fields)] float tensor."""
    vec = [float(asg.context.get(f, 0.0)) for f in fields]
    return torch.tensor(vec, dtype=torch.float32)


def _load_asg_json(path: str) -> ASGSequence:
    """Load one cached ASGSequence json written by scripts/01_autolabel.py."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    asg_t = ASG.from_dict(d["asg_t"])
    asg_th = ASG.from_dict(d["asg_th"])
    return ASGSequence(
        asg_t=asg_t,
        asg_th=asg_th,
        horizon_min=int(d.get("horizon_min", 60)),
        event_id=str(d.get("event_id", os.path.splitext(os.path.basename(path))[0])),
    )


def list_asg_files(cfg: Config) -> List[str]:
    """All cached ASG json files (sorted for determinism)."""
    d = sevir_mod.asg_dir(cfg)
    return sorted(glob.glob(os.path.join(d, "*.json")))


def _grid_hw(asg: ASG, default_h: int, default_w: int) -> Tuple[int, int]:
    """Resolve the ASG's source grid as scalar ``(grid_h, grid_w)``.

    The labeling pipeline and Stage-A VLM store ``meta['grid'] = [H, W]`` (a list, for
    JSON round-tripping); some code may instead set scalar ``meta['grid_h']`` /
    ``meta['grid_w']``. Accept either form (and a legacy scalar ``meta['grid']``),
    falling back to the supplied defaults. Scalar keys take precedence, mirroring the
    original lookup order.
    """
    grid = asg.meta.get("grid")
    if isinstance(grid, (list, tuple)) and len(grid) == 2:
        grid_default_h, grid_default_w = grid[0], grid[1]
    elif grid is not None:  # legacy scalar grid applied to both dims
        grid_default_h = grid_default_w = grid
    else:
        grid_default_h, grid_default_w = default_h, default_w
    grid_h = int(asg.meta.get("grid_h", grid_default_h))
    grid_w = int(asg.meta.get("grid_w", grid_default_w))
    return grid_h, grid_w


def _flow_from_asg(asg: ASG, h: int, w: int) -> torch.Tensor:
    """Build a dense [2, h, w] flow field (px/step) by splatting per-object motion.

    Each object's quantized (vy, vx) is written into a Gaussian-weighted neighborhood
    of its centroid (scaled to the low-res grid); the field is the intensity-weighted
    average. Used as the continuity/smoothness reference in ``transition_loss``.
    """
    flow = np.zeros((2, h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    grid_h, grid_w = _grid_hw(asg, h, w)
    sy = h / max(grid_h, 1)
    sx = w / max(grid_w, 1)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for o in asg.objects:
        cy = float(o.cy) * sy
        cx = float(o.cx) * sx
        sigma = max(math.sqrt(max(o.area, 1.0)) / 2.0 * min(sy, sx), 2.0)
        g = np.exp(-(((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma ** 2)))
        flow[0] += (o.vy * sy) * g
        flow[1] += (o.vx * sx) * g
        weight += g
    weight = np.maximum(weight, 1e-6)
    flow[0] /= weight
    flow[1] /= weight
    return torch.from_numpy(flow)


# ---------------------------------------------------------------------------
# Tier-0 / Tier-2: ASG transition dataset
# ---------------------------------------------------------------------------
class ASGTransitionDataset(Dataset):
    """(ASG_t, context, ASG_{t+h}, flow) pairs for the transition transformer.

    Items: ``dict(asg_t: ASG, context: Tensor[5], asg_th: ASG, flow: Tensor[2,Hf,Wf])``
    (interface contract). ``Hf=Wf=asg.growth_field_size``.
    """

    def __init__(self, cfg: Config, files: Optional[Sequence[str]] = None) -> None:
        self.cfg = cfg
        self.files = list(files) if files is not None else list_asg_files(cfg)
        self.context_fields = list(cfg.get_path("asg.context_fields", CONTEXT_FIELDS_DEFAULT))
        self.flow_size = int(cfg.get_path("asg.growth_field_size", 48))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq = _load_asg_json(self.files[idx])
        ctx = _context_vector(seq.asg_t, self.context_fields)
        flow = _flow_from_asg(seq.asg_t, self.flow_size, self.flow_size)
        return {
            "asg_t": seq.asg_t,
            "context": ctx,
            "asg_th": seq.asg_th,
            "flow": flow,
            "event_id": seq.event_id,
        }


# ---------------------------------------------------------------------------
# Tier-0 / Tier-2: renderer dataset
# ---------------------------------------------------------------------------
class RendererDataset(Dataset):
    """(ASG_{t+h}, advect_blind, target, growth_budget) for the Stage-C renderer.

    Items: ``dict(asg_th: ASG, advect_blind: Tensor[1,H,W], target: Tensor[1,H,W],
    growth_budget: Tensor[])`` on ``data.patch`` crops (interface contract).

    Pairs each cached ASG with its source event: the target is the VIL frame at the
    transition horizon and ``advect_blind`` is the future-blind extrapolation from the
    history (datasource.md section 2). ``growth_budget`` is the target's integrated
    (clamped) content -> the mass budget the renderer must conserve.
    """

    def __init__(self, cfg: Config, files: Optional[Sequence[str]] = None) -> None:
        self.cfg = cfg
        self.files = list(files) if files is not None else list_asg_files(cfg)
        self.patch = int(cfg.get_path("data.patch", 128))
        self.in_frames = int(cfg.get_path("data.in_frames", 13))
        self.minutes_per_frame = int(cfg.get_path("data.minutes_per_frame", 5))
        self.channels = ["vil"]
        self.seed = int(cfg.get_path("seed", 1234))
        # event lookup: event_id -> cached npz path
        self._event_index = self._build_event_index()

    def _build_event_index(self) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        d = sevir_mod.events_dir(self.cfg)
        for p in glob.glob(os.path.join(d, "*.npz")):
            idx[os.path.splitext(os.path.basename(p))[0]] = p
        return idx

    def __len__(self) -> int:
        return len(self.files)

    def _load_event_frames(self, seq: ASGSequence) -> np.ndarray:
        """Return VIL frames [T, H, W] for this event (cache or synthetic fallback)."""
        eid = seq.event_id
        path = self._event_index.get(eid)
        if path is not None and os.path.exists(path):
            ev = sevir_mod.load_event(path, channels=["vil"], cfg=self.cfg)
            return np.asarray(ev["vil"], dtype=np.float32)
        # synthetic fallback: recover the index from the synth_<idx> id
        if eid.startswith("synth_"):
            try:
                gidx = int(eid.split("_")[-1])
            except ValueError:
                gidx = 0
            ev = sevir_mod.load_event(gidx, channels=["vil"], cfg=self.cfg)
            return np.asarray(ev["vil"], dtype=np.float32)
        raise FileNotFoundError(f"no event frames for {eid!r}")

    def _crop(self, *fields: np.ndarray, asg: ASG, gh: int, gw: int) -> List[np.ndarray]:
        """Deterministic crop of size ``patch`` centered on the busiest object."""
        H, W = fields[0].shape
        p = min(self.patch, H, W)
        if asg.objects:
            o = max(asg.objects, key=lambda o: (o.peak, o.area))
            cy = int(np.clip(o.cy, p // 2, H - p // 2))
            cx = int(np.clip(o.cx, p // 2, W - p // 2))
        else:
            cy, cx = H // 2, W // 2
        y0 = int(np.clip(cy - p // 2, 0, H - p))
        x0 = int(np.clip(cx - p // 2, 0, W - p))
        return [f[y0:y0 + p, x0:x0 + p] for f in fields]

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq = _load_asg_json(self.files[idx])
        frames = self._load_event_frames(seq)
        T, H, W = frames.shape

        # history = first in_frames; horizon index = in_frames-1 + horizon/dt steps
        k = min(self.in_frames, T)
        hist = frames[:k]
        steps = max(1, int(round(seq.horizon_min / max(self.minutes_per_frame, 1))))
        target_idx = min(k - 1 + steps, T - 1)
        target_full = frames[target_idx]

        ab_full = advect_blind(hist, n_out=steps)[-1]  # [H, W] at the horizon

        tgt, ab = self._crop(target_full, ab_full, asg=seq.asg_th, gh=H, gw=W)
        target = torch.from_numpy(np.ascontiguousarray(tgt)).unsqueeze(0).float()
        advect = torch.from_numpy(np.ascontiguousarray(ab)).unsqueeze(0).float()
        growth_budget = target.clamp(min=0).sum()

        return {
            "asg_th": seq.asg_th,
            "advect_blind": advect,
            "target": target,
            "growth_budget": growth_budget,
            "event_id": seq.event_id,
        }


# ---------------------------------------------------------------------------
# Tier-1: VLM five-phase curriculum dataset (datasource.md section 5)
# ---------------------------------------------------------------------------
PHASES: Tuple[str, ...] = ("ph1_vqa", "ph2_desc", "ph3_asg", "ph4_cot", "ph5_eqcot")

_PHASE_INSTRUCTION = {
    "ph1_vqa": "Answer the question about the radar sequence.",
    "ph2_desc": "Describe the precipitation cells in natural meteorological language.",
    "ph3_asg": "Emit the Atmospheric Scene Graph (ASG_t) in the fixed JSON grammar.",
    "ph4_cot": (
        "Reason step by step: first state the observed ASG, then describe the "
        "transition to the forecast horizon."
    ),
    "ph5_eqcot": (
        "Using the governing equations (advection, continuity, growth-decay), reason "
        "step by step: state the observed ASG, then the physically-driven transition."
    ),
}


def _build_prompt(phase: str, context: Dict[str, float]) -> str:
    """Build the curriculum prompt for ``phase``.

    Prefers ``asgwm.models.prompts.build_prompt`` when that module is importable; falls
    back to a self-contained template so this dataset works before Stage-A lands.
    """
    try:  # optional dependency on a sibling module (may not exist yet)
        from ..models.prompts import build_prompt  # type: ignore

        return build_prompt(phase, dict(context))
    except Exception:
        instr = _PHASE_INSTRUCTION.get(phase, _PHASE_INSTRUCTION["ph3_asg"])
        ctx_str = ", ".join(f"{k}={float(v):.0f}" for k, v in sorted(context.items()))
        ctx_line = f"\nEnvironmental context: {ctx_str}." if ctx_str else ""
        return f"{instr}{ctx_line}"


def _vqa_pair(asg: ASG, rng: np.random.Generator) -> Tuple[str, str]:
    """Procedural (question, answer) from the ASG (datasource.md section 5, Ph-1)."""
    n = asg.n_objects
    kinds = ["count", "spatial", "tendency", "intensity"]
    kind = kinds[int(rng.integers(0, len(kinds)))]
    if kind == "count" or n == 0:
        return ("How many storm cells are present?", str(n))
    dom = max(asg.objects, key=lambda o: (o.peak, o.area))
    if kind == "spatial":
        gh = float(_grid_hw(asg, 384, 384)[0])
        half = "northern" if dom.cy < gh / 2 else "southern"
        q = "Is precipitation concentrated in the northern or southern half of the frame?"
        return (q, half)
    if kind == "tendency":
        if dom.growth > 0.05:
            ans = "growing"
        elif dom.growth < -0.05:
            ans = "decaying"
        else:
            ans = "stable"
        return ("Is the dominant cell growing, decaying, or stable?", ans)
    # intensity
    from ..asg import intensity_class

    q = "What is the approximate intensity of the strongest cell?"
    return (q, intensity_class(dom.peak))


class VLMCurriculumDataset(Dataset):
    """Five-phase VLM curriculum pairs (datasource.md section 5).

    ``phase`` selects how each item's (images, prompt, target) triple is built:
      * ph1_vqa  -- procedural VQA from the ASG.
      * ph2_desc -- ``render_NL(ASG_t)`` object description.
      * ph3_asg  -- structured ``serialize(ASG_t)`` target (NL suppressed).
      * ph4_cot  -- observation rationale + ASG_t + transition rationale + ASG_{t+h}.
      * ph5_eqcot-- as ph4 but with the equation block in the prompt.

    Items: ``dict(images: Tensor[T,1,H,W], prompt: str, target: str)``. ``images`` are
    the (optionally downsampled) history VIL frames; the trainer's processor handles
    final formatting. Built on cached ASG json + cached events.
    """

    def __init__(
        self,
        cfg: Config,
        phase: str,
        files: Optional[Sequence[str]] = None,
        image_size: int = 224,
        max_hist: int = 4,
    ) -> None:
        if phase not in PHASES:
            raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
        self.cfg = cfg
        self.phase = phase
        self.files = list(files) if files is not None else list_asg_files(cfg)
        self.image_size = int(image_size)
        self.max_hist = int(max_hist)
        self.in_frames = int(cfg.get_path("data.in_frames", 13))
        self.seed = int(cfg.get_path("seed", 1234))
        self._event_index = self._build_event_index()

    def _build_event_index(self) -> Dict[str, str]:
        idx: Dict[str, str] = {}
        d = sevir_mod.events_dir(self.cfg)
        for p in glob.glob(os.path.join(d, "*.npz")):
            idx[os.path.splitext(os.path.basename(p))[0]] = p
        return idx

    def __len__(self) -> int:
        return len(self.files)

    def _images(self, seq: ASGSequence) -> torch.Tensor:
        """History VIL frames as a [n, 1, S, S] tensor (downsampled, normalized 0-1)."""
        eid = seq.event_id
        path = self._event_index.get(eid)
        if path is not None and os.path.exists(path):
            frames = np.asarray(
                sevir_mod.load_event(path, channels=["vil"], cfg=self.cfg)["vil"],
                dtype=np.float32,
            )
        elif eid.startswith("synth_"):
            gidx = int(eid.split("_")[-1]) if eid.split("_")[-1].isdigit() else 0
            frames = np.asarray(
                sevir_mod.load_event(gidx, channels=["vil"], cfg=self.cfg)["vil"],
                dtype=np.float32,
            )
        else:
            frames = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)

        k = min(self.in_frames, frames.shape[0])
        hist = frames[:k]
        # subsample up to max_hist frames evenly across the history
        if hist.shape[0] > self.max_hist:
            sel = np.linspace(0, hist.shape[0] - 1, self.max_hist).round().astype(int)
            hist = hist[sel]
        t = torch.from_numpy(np.ascontiguousarray(hist)).unsqueeze(1).float()  # [n,1,H,W]
        if t.shape[-1] != self.image_size or t.shape[-2] != self.image_size:
            t = torch.nn.functional.interpolate(
                t, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        vmax = float(t.max())
        if vmax > 0:
            t = t / vmax
        return t

    def __getitem__(self, idx: int) -> Dict[str, object]:
        seq = _load_asg_json(self.files[idx])
        images = self._images(seq)
        ctx = dict(seq.asg_t.context)
        rng = np.random.default_rng(self.seed + idx)

        if self.phase == "ph1_vqa":
            q, a = _vqa_pair(seq.asg_t, rng)
            prompt = f"{_build_prompt(self.phase, ctx)}\nQuestion: {q}"
            target = a
        elif self.phase == "ph2_desc":
            prompt = _build_prompt(self.phase, ctx)
            target = render_NL(seq.asg_t)
        elif self.phase == "ph3_asg":
            prompt = _build_prompt(self.phase, ctx)
            target = serialize(seq.asg_t)
        elif self.phase == "ph4_cot":
            prompt = _build_prompt(self.phase, ctx)
            target = (
                f"Observation: {render_NL(seq.asg_t)}\n"
                f"ASG_t:\n{serialize(seq.asg_t)}\n"
                f"Transition: {render_NL_delta(seq.asg_t, seq.asg_th)}\n"
                f"ASG_t+h:\n{serialize(seq.asg_th)}"
            )
        else:  # ph5_eqcot
            prompt = _build_prompt(self.phase, ctx)
            target = (
                f"Observation: {render_NL(seq.asg_t)}\n"
                f"ASG_t:\n{serialize(seq.asg_t)}\n"
                f"Transition (advection v.grad phi + continuity + growth(CAPE,CIN,shear)): "
                f"{render_NL_delta(seq.asg_t, seq.asg_th)}\n"
                f"ASG_t+h:\n{serialize(seq.asg_th)}"
            )

        return {"images": images, "prompt": prompt, "target": target, "event_id": seq.event_id}


# ---------------------------------------------------------------------------
# Collate functions (interface contract)
# ---------------------------------------------------------------------------
def collate_transition(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate :class:`ASGTransitionDataset` items.

    ASGs stay as Python lists (variable object counts; encoded inside Stage B);
    ``context`` and ``flow`` are stacked into batched tensors.
    """
    return {
        "asg_t": [b["asg_t"] for b in batch],
        "asg_th": [b["asg_th"] for b in batch],
        "context": torch.stack([b["context"] for b in batch], dim=0),  # [B,5]
        "flow": torch.stack([b["flow"] for b in batch], dim=0),        # [B,2,Hf,Wf]
        "event_id": [b.get("event_id", "") for b in batch],
    }


def collate_renderer(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate :class:`RendererDataset` items into batched field tensors."""
    return {
        "asg_th": [b["asg_th"] for b in batch],
        "advect_blind": torch.stack([b["advect_blind"] for b in batch], dim=0),  # [B,1,H,W]
        "target": torch.stack([b["target"] for b in batch], dim=0),              # [B,1,H,W]
        "growth_budget": torch.stack(
            [torch.as_tensor(b["growth_budget"], dtype=torch.float32) for b in batch], dim=0
        ),  # [B]
        "event_id": [b.get("event_id", "") for b in batch],
    }
