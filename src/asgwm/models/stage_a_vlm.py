"""Stage A perception: QLoRA VLM wrapper + DummyVLM CPU fallback.

Stage A maps ``(X_{t-k..t}, C) -> ASG_t + NL`` (architecture.md sections 2, 9, 10;
training_method.md section 3). The real path QLoRA-fine-tunes a small HuggingFace VLM
(``cfg.stage_a.backbone``): NF4 4-bit base via bitsandbytes, frozen backbone, trainable
LoRA adapters (peft) + the modality projector. The joint causal-LM loss weights ASG
tokens vs NL tokens by ``cfg.stage_a.asg_loss_weight`` / ``nl_loss_weight``
(architecture.md section 9, ~80/20). ``generate_asg`` constrained-decodes to the ASG
grammar (``grammar.object_line_regex`` / ``allowed_regime_tokens``, via lm-format-enforcer
when available) and then ``grammar.parse_strict`` (with a tolerant ``parse`` fallback).

Every heavy dependency (transformers / peft / bitsandbytes / CUDA / lm-format-enforcer)
is optional. If any is missing, ``StageAVLM.from_config`` returns a :class:`DummyVLM`
that exposes the identical API and derives a plausible ASG heuristically from the input
frames, so the Tier-0/Tier-1 code paths run end-to-end on CPU with no GPU and no download
(coding standards; training_method.md section 1).
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from asgwm.asg import (
    ASG,
    StormObject,
    REGIMES,
    intensity_class,
    quantize_motion,
)
from asgwm.asg.grammar import (
    serialize,
    parse,
    parse_strict,
    allowed_regime_tokens,
    object_line_regex,
)
from asgwm.models import prompts as _prompts

# ---------------------------------------------------------------------------
# Optional heavy-dependency probing (all guarded; never crash at import time).
# ---------------------------------------------------------------------------
_HAS_TRANSFORMERS = False
_HAS_PEFT = False
_HAS_BNB = False
_HAS_LMFE = False

try:  # transformers (vision tower + LM)
    import transformers  # type: ignore  # noqa: F401
    from transformers import (  # type: ignore
        AutoProcessor,
        AutoModelForVision2Seq,
    )

    _HAS_TRANSFORMERS = True
except Exception:  # pragma: no cover - depends on env
    transformers = None  # type: ignore

try:  # peft LoRA adapters
    import peft  # type: ignore  # noqa: F401
    from peft import LoraConfig, get_peft_model  # type: ignore

    _HAS_PEFT = True
except Exception:  # pragma: no cover
    peft = None  # type: ignore

try:  # bitsandbytes NF4 4-bit quantization
    import bitsandbytes  # type: ignore  # noqa: F401

    _HAS_BNB = True
except Exception:  # pragma: no cover
    bitsandbytes = None  # type: ignore

try:  # lm-format-enforcer for constrained decoding
    import lmformatenforcer  # type: ignore  # noqa: F401

    _HAS_LMFE = True
except Exception:  # pragma: no cover
    lmformatenforcer = None  # type: ignore


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


def _real_stack_available() -> bool:
    """True only when the full QLoRA stack can actually be constructed."""
    return _HAS_TRANSFORMERS and _HAS_PEFT and _HAS_BNB and _cuda_available()


# ---------------------------------------------------------------------------
# Shared frame/ASG helpers (used by both real and dummy paths).
# ---------------------------------------------------------------------------
def _to_numpy_frames(images: Any) -> Optional[np.ndarray]:
    """Best-effort coercion of an ``images`` payload to a ``[T, H, W]`` float array.

    Accepts a torch tensor, numpy array, or a list of either. Returns ``None`` if no
    frame-like array can be recovered (e.g. PIL images in the real-VLM path).
    """
    if images is None:
        return None
    if isinstance(images, (list, tuple)):
        arrs = [_to_numpy_frames(im) for im in images]
        arrs = [a for a in arrs if a is not None]
        if not arrs:
            return None
        arrs = [a.reshape(-1, a.shape[-2], a.shape[-1]) for a in arrs]
        return np.concatenate(arrs, axis=0).astype(np.float32)
    if isinstance(images, torch.Tensor):
        arr = images.detach().to(torch.float32).cpu().numpy()
    elif isinstance(images, np.ndarray):
        arr = images.astype(np.float32)
    else:
        return None
    arr = np.squeeze(arr)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim >= 3:
        arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
    else:
        return None
    return arr


def _heuristic_asg_from_frames(
    frames: Optional[np.ndarray],
    context: Optional[Dict[str, float]] = None,
    n_max: int = 16,
    threshold: float = 0.15,
) -> ASG:
    """Derive a plausible ASG from radar frames with pure numpy (no GPU).

    Connected-component blobs on the last frame give cells; centroid differencing across
    the last two frames gives motion; the VIL tendency gives the growth scalar and regime
    (architecture.md section 2). Used by :class:`DummyVLM` so Tier-1/2 code paths execute
    on CPU. Falls back to a single small plausible cell when no frames are available.
    """
    context = dict(context or {})
    if frames is None or frames.size == 0:
        return _fallback_asg(context)

    frames = np.asarray(frames, dtype=np.float32)
    last = frames[-1]
    prev = frames[-2] if frames.shape[0] >= 2 else last
    H, W = last.shape

    # Normalize to [0, 1] for a resolution-independent threshold.
    lo, hi = float(last.min()), float(last.max())
    norm = (last - lo) / (hi - lo) if hi > lo else np.zeros_like(last)
    mask = norm >= threshold
    if not mask.any():
        return _fallback_asg(context, grid=(H, W))

    labels = _label_components(mask)
    n_lab = int(labels.max())
    objs: List[StormObject] = []
    for lid in range(1, n_lab + 1):
        cell = labels == lid
        npx = int(cell.sum())
        if npx < 4:  # drop speckle
            continue
        ys, xs = np.nonzero(cell)
        cy, cx = float(ys.mean()), float(xs.mean())
        area = float(npx)  # km^2 at 1 km/pixel (data.km_per_pixel default 1.0)
        peak = float(last[cell].max())
        # Motion: shift of the local centroid between the two latest frames (px/frame
        # -> reported as km/h via the labeling convention; here a coarse heuristic).
        vy, vx = _local_motion(prev, last, cell)
        prev_mean = float(prev[cell].mean()) if prev.shape == last.shape else float(last[cell].mean())
        cur_mean = float(last[cell].mean())
        growth = cur_mean - prev_mean
        regime = _regime_from_growth(growth, area, peak)
        conf = float(np.clip(0.5 + 0.5 * (npx / max(mask.sum(), 1)), 0.3, 0.99))
        objs.append(
            StormObject(
                id=len(objs) + 1,
                cy=cy,
                cx=cx,
                area=area,
                peak=peak,
                vy=quantize_motion(vy),
                vx=quantize_motion(vx),
                regime=regime,
                growth=float(growth),
                conf=conf,
            )
        )

    if not objs:
        return _fallback_asg(context, grid=(H, W))

    global_regime = _global_regime(objs)
    asg = ASG(
        objects=objs,
        global_regime=global_regime,
        growth_field=None,
        context=context,
        meta={"grid": [H, W], "source": "DummyVLM.heuristic"},
    )
    return asg.capped(n_max)


def _label_components(mask: np.ndarray) -> np.ndarray:
    """4-connected component labeling. Uses scipy when present, else a BFS fallback."""
    try:  # pragma: no cover - exercised when scipy present
        from scipy import ndimage  # type: ignore

        labels, _ = ndimage.label(mask)
        return labels.astype(np.int32)
    except Exception:
        pass
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)
    cur = 0
    stack: List[tuple] = []
    for i in range(H):
        for j in range(W):
            if mask[i, j] and labels[i, j] == 0:
                cur += 1
                stack.append((i, j))
                labels[i, j] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and labels[ny, nx] == 0:
                            labels[ny, nx] = cur
                            stack.append((ny, nx))
    return labels


def _local_motion(prev: np.ndarray, last: np.ndarray, cell: np.ndarray) -> tuple:
    """Coarse motion of a cell as the centroid shift of its intensity between frames."""
    if prev.shape != last.shape:
        return 0.0, 0.0
    ys, xs = np.nonzero(cell)
    if ys.size == 0:
        return 0.0, 0.0
    y0, y1 = max(int(ys.min()) - 4, 0), min(int(ys.max()) + 5, last.shape[0])
    x0, x1 = max(int(xs.min()) - 4, 0), min(int(xs.max()) + 5, last.shape[1])
    a = prev[y0:y1, x0:x1].astype(np.float64)
    b = last[y0:y1, x0:x1].astype(np.float64)
    ca = _intensity_centroid(a)
    cb = _intensity_centroid(b)
    if ca is None or cb is None:
        return 0.0, 0.0
    # px/frame; scale toward km/h using a nominal 5-min frame -> 12 frames/h.
    vy = (cb[0] - ca[0]) * 12.0
    vx = (cb[1] - ca[1]) * 12.0
    return float(vy), float(vx)


def _intensity_centroid(arr: np.ndarray) -> Optional[tuple]:
    s = float(arr.sum())
    if s <= 0:
        return None
    ys = np.arange(arr.shape[0])[:, None]
    xs = np.arange(arr.shape[1])[None, :]
    return (float((arr * ys).sum() / s), float((arr * xs).sum() / s))


def _regime_from_growth(growth: float, area: float, peak: float) -> str:
    """Regime from tendency + morphology (architecture.md section 2)."""
    if growth > 1.0 and area < 50.0:
        return "init"
    if growth > 0.5:
        return "grow"
    if growth < -0.5:
        return "decay"
    return "steady"


def _global_regime(objs: Sequence[StormObject]) -> str:
    if not objs:
        return "steady"
    counts = {r: 0 for r in REGIMES}
    for o in objs:
        counts[o.regime] += 1
    # Prefer the most active non-steady regime if present.
    for r in ("grow", "init", "decay"):
        if counts[r] >= max(1, len(objs) // 2):
            return r
    return max(counts, key=counts.get)


def _fallback_asg(context: Dict[str, float], grid: tuple = (128, 128)) -> ASG:
    """A small, plausible single-cell ASG (used when no frames are recoverable)."""
    H, W = grid
    obj = StormObject(
        id=1,
        cy=H / 2.0,
        cx=W / 2.0,
        area=40.0,
        peak=35.0,
        vy=quantize_motion(8.0),
        vx=quantize_motion(8.0),
        regime="steady",
        growth=0.0,
        conf=0.5,
    )
    return ASG(
        objects=[obj],
        global_regime="steady",
        growth_field=None,
        context=dict(context),
        meta={"grid": [H, W], "source": "DummyVLM.fallback"},
    )


def _parse_asg_text(text: str) -> ASG:
    """Parse a model completion to an ASG: ``parse_strict`` first, ``parse`` as fallback.

    The real grammar lines are recovered even when surrounded by CoT prose; we slice the
    grammar block (lines starting with GLOBAL(/OBJECT() before strict-parsing.
    """
    grammar_lines = [
        ln for ln in text.splitlines()
        if ln.strip().startswith("GLOBAL(") or ln.strip().startswith("OBJECT(")
    ]
    block = "\n".join(grammar_lines) if grammar_lines else text
    try:
        return parse_strict(block)
    except Exception:
        return parse(text)


# ---------------------------------------------------------------------------
# Constrained-decoding grammar regex (full ASG completion).
# ---------------------------------------------------------------------------
def asg_completion_regex(max_objects: int = 16) -> str:
    """Regex matching a full ASG completion (one GLOBAL line + <=max_objects OBJECT lines).

    Built from ``grammar.object_line_regex`` / ``allowed_regime_tokens`` so it is driven
    by the single grammar source of truth (architecture.md section 9). Suitable for
    lm-format-enforcer ``RegexParser`` when that dependency is present.
    """
    regime_alt = "|".join(allowed_regime_tokens())
    glob = rf"GLOBAL\(regime=(?:{regime_alt}),\s*n_objects=\d+\)"
    obj = object_line_regex()
    return rf"{glob}(?:\n{obj}){{0,{int(max_objects)}}}"


# ===========================================================================
# DummyVLM — CPU fallback with the full Stage-A API.
# ===========================================================================
class DummyVLM(nn.Module):
    """CPU fallback for Stage A exposing the same API as :class:`StageAVLM`.

    Holds a single trainable scalar so optimizer/checkpoint code paths work, computes a
    real (tiny) differentiable training loss, and derives an ASG heuristically from the
    input frames in :meth:`generate_asg`. Lets Tier-0/Tier-1/Tier-2 run with no GPU
    (training_method.md section 1; coding standards).
    """

    def __init__(self, cfg: Optional[Any] = None) -> None:
        super().__init__()
        self.cfg = cfg
        sa = _sa_cfg(cfg)
        self.n_max = int(_get(sa, "n_max", None) or _asg_nmax(cfg))
        self.asg_loss_weight = float(_get(sa, "asg_loss_weight", 0.8))
        self.nl_loss_weight = float(_get(sa, "nl_loss_weight", 0.2))
        self.is_dummy = True
        # A nominal trainable parameter so DummyVLM has gradients / optimizer state.
        self._bias = nn.Parameter(torch.zeros(1))

    # -- training -----------------------------------------------------------
    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Differentiable surrogate causal-LM loss (token-length proxy on the targets).

        Mirrors the ASG/NL token weighting (architecture.md section 9) so the loss has the
        right shape and decreases as the parameter adapts; no real LM is run on CPU.
        """
        # `batch` may be a single dict, a dict-of-lists, or a list of per-item dicts
        # (the curriculum loader uses collate_fn -> list of dicts; see tier1_curriculum).
        if isinstance(batch, list):
            targets = [b.get("target", b.get("targets", "")) if isinstance(b, dict) else b
                       for b in batch]
        else:
            targets = batch.get("target", batch.get("targets", ""))
        if isinstance(targets, str):
            targets = [targets]
        device = self._bias.device
        asg_tok, nl_tok = 0, 0
        for t in targets:
            t = t or ""
            for ln in str(t).splitlines():
                s = ln.strip()
                if s.startswith("GLOBAL(") or s.startswith("OBJECT("):
                    asg_tok += max(len(s.split()), 1)
                else:
                    nl_tok += max(len(s.split()), 1)
        asg_tok = max(asg_tok, 1)
        nl_tok = max(nl_tok, 1)
        # Surrogate per-token NLL that shrinks toward 0 as _bias grows; weighted 80/20.
        scale = torch.sigmoid(-self._bias).squeeze(0)
        asg_loss = scale * float(np.log1p(asg_tok))
        nl_loss = scale * float(np.log1p(nl_tok))
        return self.asg_loss_weight * asg_loss + self.nl_loss_weight * nl_loss

    # -- inference ----------------------------------------------------------
    @torch.no_grad()
    def generate_per_frame(
        self, images: Any, context: Optional[Dict[str, float]] = None
    ) -> "List[List[StormObject]]":
        """Identify step: return per-frame object sets (oldest frame first).

        Each inner list contains the StormObjects identified in one frame.
        Velocities on the returned objects are SINGLE-FRAME estimates; the
        ObjectTracker refines them using multi-frame centroid displacement.
        """
        frames = _to_numpy_frames(images)
        if frames is None or frames.size == 0:
            return [[]]
        n_frames = frames.shape[0]
        result: List[List[StormObject]] = []
        for ti in range(n_frames):
            single = frames[ti : ti + 1]
            asg = _heuristic_asg_from_frames(single, context, n_max=self.n_max)
            result.append(list(asg.objects))
        return result

    @torch.no_grad()
    def generate_asg(self, images: Any, context: Optional[Dict[str, float]] = None) -> ASG:
        use_tracker = _use_explicit_tracker(self.cfg)
        if use_tracker:
            from asgwm.models.stage_a2_tracker import ObjectTracker
            per_frame = self.generate_per_frame(images, context)
            tracker = ObjectTracker.from_config(self.cfg)
            frames = _to_numpy_frames(images)
            flow = _optical_flow_from_frames(frames)
            return tracker.track(per_frame, flow=flow,
                                 context=context, n_max=self.n_max)
        frames = _to_numpy_frames(images)
        return _heuristic_asg_from_frames(frames, context, n_max=self.n_max)

    def generate_nl(self, images: Any, context: Optional[Dict[str, float]] = None) -> str:
        from asgwm.asg import render_NL

        return render_NL(self.generate_asg(images, context))

    # -- adapter (de)serialization -----------------------------------------
    def save_adapters(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save({"dummy": True, "state_dict": self.state_dict()}, path)

    def load_adapters(self, path: str) -> None:
        if os.path.isfile(path):
            ckpt = torch.load(path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            try:
                self.load_state_dict(sd, strict=False)
            except Exception:
                pass


# ===========================================================================
# StageAVLM — QLoRA VLM wrapper.
# ===========================================================================
class StageAVLM(nn.Module):
    """QLoRA-fine-tuned VLM for Stage-A perception (architecture.md sections 2, 9, 10).

    Construct via :meth:`from_config`, which returns a :class:`DummyVLM` whenever the
    QLoRA stack (transformers + peft + bitsandbytes + CUDA) is unavailable, so the wider
    pipeline never crashes on a CPU-only box (coding standards; training_method.md s.3).
    """

    def __init__(
        self,
        model: "nn.Module",
        processor: Any,
        cfg: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.cfg = cfg
        sa = _sa_cfg(cfg)
        self.backbone = _get(sa, "backbone", "HuggingFaceTB/SmolVLM-2.2B-Instruct")
        self.asg_loss_weight = float(_get(sa, "asg_loss_weight", 0.8))
        self.nl_loss_weight = float(_get(sa, "nl_loss_weight", 0.2))
        self.max_new_tokens = int(_get(sa, "max_new_tokens", 512))
        self.constrained_decoding = bool(_get(sa, "constrained_decoding", True))
        self.n_max = int(_asg_nmax(cfg))
        self.is_dummy = False

    # ------------------------------------------------------------------ build
    @classmethod
    def from_config(cls, cfg: Any) -> Union["StageAVLM", DummyVLM]:
        """Load the 4-bit NF4 VLM + attach LoRA, or fall back to :class:`DummyVLM`.

        QLoRA recipe (training_method.md section 3, architecture.md section 2): NF4 4-bit
        base via bitsandbytes, frozen backbone, trainable LoRA(r, alpha) adapters + the
        modality (multi-modal) projector. Returns a DummyVLM if any heavy dependency or
        CUDA is missing — the documented CPU path.
        """
        if not _real_stack_available():
            return DummyVLM(cfg)
        try:
            return cls._build_real(cfg)
        except Exception as exc:  # pragma: no cover - hardware/runtime dependent
            import warnings

            warnings.warn(f"StageAVLM real build failed ({exc!r}); using DummyVLM.")
            return DummyVLM(cfg)

    @classmethod
    def _build_real(cls, cfg: Any) -> "StageAVLM":  # pragma: no cover - needs GPU
        from transformers import BitsAndBytesConfig  # type: ignore

        sa = _sa_cfg(cfg)
        backbone = _get(sa, "backbone", "HuggingFaceTB/SmolVLM-2.2B-Instruct")
        load_4bit = bool(_get(sa, "load_in_4bit", True))
        lora_r = int(_get(sa, "lora_r", 16))
        lora_alpha = int(_get(sa, "lora_alpha", 32))
        lora_dropout = float(_get(sa, "lora_dropout", 0.05))
        precision = _get(_train_cfg(cfg), "precision", "bf16")
        compute_dtype = torch.bfloat16 if str(precision) == "bf16" else torch.float16

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_4bit,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        processor = AutoProcessor.from_pretrained(backbone, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            backbone,
            quantization_config=bnb_config if load_4bit else None,
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

        # Prepare for k-bit training (cast norms, enable input grads, grad checkpointing).
        try:
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=bool(_get(_train_cfg(cfg), "grad_checkpointing", True)),
            )
        except Exception:
            pass

        # Freeze the backbone (architecture.md section 2); LoRA + projector train.
        for p in model.parameters():
            p.requires_grad = False

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=_lora_target_modules(model),
        )
        model = get_peft_model(model, lora_cfg)

        # Also train the modality / multi-modal projector (training_method.md section 3).
        _unfreeze_projector(model)

        return cls(model=model, processor=processor, cfg=cfg)

    # --------------------------------------------------------------- training
    def training_step(self, batch: Dict[str, Any]) -> torch.Tensor:  # pragma: no cover
        """Joint causal-LM CE with ASG-token vs NL-token weighting (architecture.md s.9).

        ``batch`` provides ``images``, ``prompt`` and ``target`` (per
        VLMCurriculumDataset). The prompt span is masked out of the loss; ASG-grammar
        target tokens are up-weighted to ``asg_loss_weight`` and NL tokens to
        ``nl_loss_weight`` (~80/20).
        """
        images = batch["images"]
        prompts_b = batch.get("prompt", batch.get("prompts"))
        targets_b = batch.get("target", batch.get("targets"))
        if isinstance(prompts_b, str):
            prompts_b = [prompts_b]
        if isinstance(targets_b, str):
            targets_b = [targets_b]

        device = next(self.model.parameters()).device
        total = torch.zeros((), device=device)
        for img, prompt, target in zip(_as_list(images), prompts_b, targets_b):
            enc = self._encode_pair(img, prompt, target, device)
            logits = self.model(**enc["model_inputs"]).logits  # [1, L, V]
            # Shift for next-token prediction.
            shift_logits = logits[:, :-1, :]
            shift_labels = enc["labels"][:, 1:]
            shift_weights = enc["weights"][:, 1:]
            ce = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1).clamp_min(0),
                reduction="none",
            ).reshape(shift_labels.shape)
            valid = (shift_labels >= 0).float()
            w = shift_weights * valid
            denom = w.sum().clamp_min(1.0)
            total = total + (ce * w).sum() / denom
        return total / max(len(targets_b), 1)

    def _encode_pair(self, image: Any, prompt: str, target: str, device) -> Dict[str, Any]:  # pragma: no cover
        """Tokenize a (prompt, target) pair, masking the prompt and weighting ASG tokens."""
        full = (prompt or "") + "\n" + (target or "")
        proc = self.processor(text=full, images=_as_pil_list(image), return_tensors="pt")
        proc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in proc.items()}
        input_ids = proc["input_ids"]
        labels = input_ids.clone()

        # Mask the prompt span out of the loss.
        prompt_ids = self.processor.tokenizer(prompt or "", return_tensors="pt")["input_ids"]
        n_prompt = min(prompt_ids.shape[-1], labels.shape[-1])
        labels[:, :n_prompt] = -100

        # Per-token weights: ASG-grammar tokens up-weighted vs NL tokens.
        weights = torch.full_like(labels, fill_value=0.0, dtype=torch.float32)
        weights[labels != -100] = self.nl_loss_weight
        asg_tok_ids = self._asg_token_ids(target)
        if asg_tok_ids is not None:
            tok_set = set(asg_tok_ids.tolist())
            for j in range(labels.shape[-1]):
                if labels[0, j].item() in tok_set:
                    weights[0, j] = self.asg_loss_weight
        return {
            "model_inputs": {k: v for k, v in proc.items()},
            "labels": labels,
            "weights": weights,
        }

    def _asg_token_ids(self, target: str) -> Optional[torch.Tensor]:  # pragma: no cover
        asg_lines = [ln for ln in (target or "").splitlines()
                     if ln.strip().startswith(("GLOBAL(", "OBJECT("))]
        if not asg_lines:
            return None
        ids = self.processor.tokenizer("\n".join(asg_lines), return_tensors="pt")["input_ids"]
        return ids.reshape(-1)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def generate_per_frame(  # pragma: no cover
        self, images: Any, context: Optional[Dict[str, float]] = None
    ) -> "List[List[StormObject]]":
        """Identify step: per-frame storm-cell identification (Track step uses this).

        Runs a single VLM forward pass on the last frame of the temporal sequence
        and returns it as a single-frame identification list.  Full per-frame mode
        (one VLM call per input frame) is available by setting
        cfg.stage_a.per_frame_full=true, at the cost of k VLM passes per event.
        """
        cfg_sa = _sa_cfg(self.cfg)
        per_frame_full = bool(_get(cfg_sa, "per_frame_full", False))
        frames_np = _to_numpy_frames(images)
        n_frames = frames_np.shape[0] if frames_np is not None else 1
        if not per_frame_full:
            asg = self._generate_asg_direct(images, context)
            return [list(asg.objects)]  # treat as single-frame (last frame)
        # Full per-frame mode: run VLM on each frame individually.
        result: List[List[StormObject]] = []
        for ti in range(n_frames):
            single_frame = frames_np[ti : ti + 1] if frames_np is not None else images
            asg = self._generate_asg_direct(single_frame, context)
            result.append(list(asg.objects))
        return result

    def _generate_asg_direct(  # pragma: no cover
        self, images: Any, context: Optional[Dict[str, float]] = None
    ) -> ASG:
        """Direct VLM decode → ASG (the original single-pass path)."""
        prompt = _prompts.build_prompt("ph3_asg", context=context)
        try:
            text = self._generate_text(images, prompt, constrained=self.constrained_decoding)
            asg = _parse_asg_text(text)
            asg.context = dict(context or {})
            return asg.capped(self.n_max)
        except Exception:
            frames = _to_numpy_frames(images)
            return _heuristic_asg_from_frames(frames, context, n_max=self.n_max)

    @torch.no_grad()
    def generate_asg(self, images: Any, context: Optional[Dict[str, float]] = None) -> ASG:  # pragma: no cover
        """Constrained-decode to the ASG grammar then ``parse_strict`` (parse fallback).

        When cfg.stage_a.use_explicit_tracker=true (default), routes through the
        Identify → Track pipeline so Stage B receives a trajectory-enriched ASG.
        When false, falls back to the original single-pass VLM → ASG path.
        """
        if _use_explicit_tracker(self.cfg):
            from asgwm.models.stage_a2_tracker import ObjectTracker
            per_frame = self.generate_per_frame(images, context)
            tracker = ObjectTracker.from_config(self.cfg)
            frames_np = _to_numpy_frames(images)
            flow = _optical_flow_from_frames(frames_np)
            return tracker.track(per_frame, flow=flow,
                                 context=context, n_max=self.n_max)
        return self._generate_asg_direct(images, context)

    def _generate_text(self, images: Any, prompt: str, constrained: bool) -> str:  # pragma: no cover
        device = next(self.model.parameters()).device
        proc = self.processor(text=prompt, images=_as_pil_list(images), return_tensors="pt")
        proc = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in proc.items()}
        gen_kw: Dict[str, Any] = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        if constrained and _HAS_LMFE:
            prefix_fn = self._build_constrained_logits_processor()
            if prefix_fn is not None:
                gen_kw["prefix_allowed_tokens_fn"] = prefix_fn
        out = self.model.generate(**proc, **gen_kw)
        text = self.processor.tokenizer.decode(out[0], skip_special_tokens=True)
        # Drop the echoed prompt if present.
        return text.split(prompt, 1)[-1] if prompt in text else text

    def _build_constrained_logits_processor(self):  # pragma: no cover
        """lm-format-enforcer RegexParser -> HF prefix_allowed_tokens_fn over ASG grammar."""
        try:
            from lmformatenforcer import RegexParser  # type: ignore
            from lmformatenforcer.integrations.transformers import (  # type: ignore
                build_transformers_prefix_allowed_tokens_fn,
            )

            parser = RegexParser(asg_completion_regex(self.n_max))
            return build_transformers_prefix_allowed_tokens_fn(
                self.processor.tokenizer, parser
            )
        except Exception:
            return None

    # ----------------------------------------------------------- (de)serialize
    def save_adapters(self, path: str) -> None:  # pragma: no cover
        os.makedirs(path, exist_ok=True)
        try:
            self.model.save_pretrained(path)
        except Exception:
            torch.save(self.model.state_dict(), os.path.join(path, "adapters.pt"))

    def load_adapters(self, path: str) -> None:  # pragma: no cover
        try:
            from peft import PeftModel  # type: ignore

            if hasattr(self.model, "load_adapter"):
                self.model.load_adapter(path, adapter_name="default")
            else:
                self.model = PeftModel.from_pretrained(self.model, path)
        except Exception:
            pt = os.path.join(path, "adapters.pt")
            if os.path.isfile(pt):
                self.model.load_state_dict(torch.load(pt, map_location="cpu"), strict=False)


# ---------------------------------------------------------------------------
# Config / small utilities.
# ---------------------------------------------------------------------------
def _get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _sa_cfg(cfg: Any) -> Any:
    return _get(cfg, "stage_a", {}) or {}


def _train_cfg(cfg: Any) -> Any:
    return _get(cfg, "train", {}) or {}


def _asg_nmax(cfg: Any) -> int:
    asg = _get(cfg, "asg", {}) or {}
    from asgwm.asg import N_MAX

    return int(_get(asg, "n_max", N_MAX))


def _as_list(x: Any) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _lora_target_modules(model: "nn.Module") -> List[str]:  # pragma: no cover - GPU path
    """Find linear-projection module names to attach LoRA to (q/k/v/o + MLP)."""
    candidates = ("q_proj", "k_proj", "v_proj", "o_proj",
                  "gate_proj", "up_proj", "down_proj", "fc1", "fc2")
    found = set()
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in candidates:
            found.add(leaf)
    return sorted(found) or ["q_proj", "v_proj"]


def _unfreeze_projector(model: "nn.Module") -> None:  # pragma: no cover - GPU path
    """Make the modality/multi-modal projector trainable (training_method.md section 3)."""
    keys = ("projector", "connector", "modality", "mm_proj", "multi_modal", "image_proj")
    for name, p in model.named_parameters():
        low = name.lower()
        if any(k in low for k in keys):
            p.requires_grad = True


def _use_explicit_tracker(cfg: Any) -> bool:
    """Whether to route generate_asg through the Identify → Track pipeline."""
    cfg_sa = _sa_cfg(cfg)
    return bool(_get(cfg_sa, "use_explicit_tracker", True))


def _optical_flow_from_frames(frames: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Estimate [2, H, W] optical flow from a frame stack (px/step), or None."""
    if frames is None or frames.ndim < 3 or frames.shape[0] < 2:
        return None
    try:
        from asgwm.labeling.motion import estimate_motion
        return estimate_motion(frames)
    except Exception:
        return None


def _as_pil_list(images: Any) -> List[Any]:  # pragma: no cover - GPU path
    """Coerce frame arrays/tensors to a list of PIL images for the HF processor."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return images if isinstance(images, list) else [images]
    frames = _to_numpy_frames(images)
    if frames is None:
        return images if isinstance(images, list) else [images]
    out = []
    for fr in frames:
        a = fr.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
        out.append(Image.fromarray((a * 255).astype(np.uint8)).convert("RGB"))
    return out
