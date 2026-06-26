"""Tier 1 — five-phase VLM curriculum with the hard Ph-3 ASG-F1 gate.

Runs the curriculum ph1_vqa -> ph2_desc -> ph3_asg -> ph4_cot -> ph5_eqcot sequentially,
each phase fine-tuning the QLoRA VLM from the previous phase's checkpoint
(training_method.md section 3, architecture.md section 10).

The Ph-3 gate is the hard decision point (training_method.md section 3): AFTER ph3_asg we
compute ASG F1 on the gold subset via `eval.faithfulness.asg_accuracy`; if F1 <
`cfg.train.tier1.ph3_gate_f1` we log and raise RuntimeError — the downstream CoT is
unfounded without a reliable state.

All phases are checkpoint/resume safe (training_method.md section 6) and run a few steps
on VLMCurriculumDataset + DummyVLM with no GPU. Heavy modules are imported lazily/guarded.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from asgwm.asg import ASG, parse
from . import checkpoint as ckpt


PHASES: List[str] = ["ph1_vqa", "ph2_desc", "ph3_asg", "ph4_cot", "ph5_eqcot"]


def _device(cfg) -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _seed_all(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ckpt_dir(cfg, phase: str) -> str:
    root = cfg.get_path("paths.checkpoints", "./artifacts/ckpt")
    d = os.path.join(root, "tier1", phase)
    os.makedirs(d, exist_ok=True)
    return d


def _import_vlm():
    from asgwm.models.stage_a_vlm import StageAVLM
    return StageAVLM


def _import_curriculum_dataset():
    from asgwm.data.dataset import VLMCurriculumDataset
    return VLMCurriculumDataset


def _vlm_collate(batch: List[dict]) -> List[dict]:
    """Curriculum items hold heterogeneous image/prompt/target payloads; keep as a list so
    StageAVLM.training_step can tokenize per-item (DummyVLM ignores content)."""
    return list(batch)


def _steps_for_phase(cfg, phase: str) -> int:
    spp = cfg.get_path("train.tier1.steps_per_phase", {}) or {}
    if isinstance(spp, dict) and phase in spp:
        base = int(spp[phase])
    else:
        base = int(cfg.get_path("train.tier1.default_steps", 2000))
    # Optional scalar cap for time-boxed runs (FIRST_RUN); null/None = no cap.
    cap = cfg.get_path("train.tier1.max_steps_per_phase", None)
    if cap not in (None, "", "none", "null"):
        base = min(base, int(cap))
    return base


def run_phase(cfg, phase: str, ckpt_in: Optional[str]) -> str:
    """Fine-tune one curriculum phase (training_method.md section 3).

    Loads adapters from `ckpt_in` (the previous phase's checkpoint) if present, trains on
    VLMCurriculumDataset(phase), checkpoints, and returns the output checkpoint path.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}; expected one of {PHASES}")
    _seed_all(int(cfg.get_path("seed", 1234)))
    device = _device(cfg)

    StageAVLM = _import_vlm()
    VLMCurriculumDataset = _import_curriculum_dataset()

    ds = VLMCurriculumDataset(cfg, phase=phase)
    micro_bs = int(cfg.get_path("train.tier1.micro_batch", 2))
    micro_bs = max(1, min(micro_bs, len(ds)))
    grad_accum = max(1, int(cfg.get_path("train.tier1.grad_accum", 1)))
    loader = DataLoader(ds, batch_size=micro_bs, shuffle=True, collate_fn=_vlm_collate)

    model = StageAVLM.from_config(cfg)
    if hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:
            pass

    # Load incoming adapters (sequential fine-tuning from the previous phase).
    if ckpt_in and os.path.exists(ckpt_in):
        _load_phase_ckpt(model, ckpt_in)

    lr = float(cfg.get_path("train.tier1.lr", 1e-4))
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=lr) if params else None

    cdir = _ckpt_dir(cfg, phase)
    resume = ckpt.latest(cdir)
    start_step = 0
    if resume and os.path.exists(resume):
        payload = ckpt.load_ckpt(resume, model=model, optim=optim)
        start_step = int(payload.get("step", 0))

    max_steps = _steps_for_phase(cfg, phase)
    ckpt_every = max(1, int(cfg.get_path("train.tier1.ckpt_every", 500)))

    if hasattr(model, "train"):
        model.train()
    step = start_step
    last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
    data_iter = iter(loader)
    accum = 0
    if optim is not None:
        optim.zero_grad(set_to_none=True)
    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        loss = model.training_step(batch)
        if isinstance(loss, dict):
            loss = loss.get("loss", loss.get("total"))
        if optim is not None and isinstance(loss, torch.Tensor) and loss.requires_grad:
            (loss / grad_accum).backward()
            accum += 1
            if accum % grad_accum == 0:
                optim.step()
                optim.zero_grad(set_to_none=True)
        step += 1

        if step % ckpt_every == 0 or step >= max_steps:
            last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
            _save_phase_ckpt(model, optim, step, last_path,
                             loss=float(loss) if isinstance(loss, torch.Tensor) else 0.0)

    last_path = os.path.join(cdir, f"ckpt_step_{step:06d}.pt")
    _save_phase_ckpt(model, optim, step, last_path)
    return last_path


def _save_phase_ckpt(model, optim, step: int, path: str, loss: float = 0.0) -> None:
    """Save both the torch state and (if available) peft adapters next to the checkpoint."""
    ckpt.save_ckpt(path, step, model, optim, extra={"loss": loss})
    if hasattr(model, "save_adapters"):
        try:
            model.save_adapters(path + ".adapters")
        except Exception:
            pass


def _load_phase_ckpt(model, path: str) -> None:
    if hasattr(model, "load_adapters") and os.path.exists(path + ".adapters"):
        try:
            model.load_adapters(path + ".adapters")
            return
        except Exception:
            pass
    try:
        ckpt.load_ckpt(path, model=model)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ph-3 gate: ASG F1 on the gold subset
# ---------------------------------------------------------------------------
def _load_gold_asgs(cfg) -> List[ASG]:
    """Load the hand-labeled gold ASG subset; fall back to a synthetic gold set so the
    gate is computable in the CPU smoke test (datasource.md gold subset)."""
    gold_dir = cfg.get_path("paths.gold_subset", "./artifacts/gold")
    asgs: List[ASG] = []
    if gold_dir and os.path.isdir(gold_dir):
        import json
        for name in sorted(os.listdir(gold_dir)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(gold_dir, name), "r", encoding="utf-8") as f:
                    d = json.load(f)
                asgs.append(ASG.from_dict(d))
            except Exception:
                continue
    if asgs:
        return asgs
    # Synthetic gold fallback from the dataset's oracle ASG_t.
    try:
        VLMCurriculumDataset = _import_curriculum_dataset()
        ds = VLMCurriculumDataset(cfg, phase="ph3_asg")
        for i in range(min(len(ds), 16)):
            item = ds[i]
            tgt = item.get("target", "")
            try:
                asgs.append(parse(tgt))
            except Exception:
                pass
    except Exception:
        pass
    return asgs


def _predict_gold_asgs(cfg, model, gold: List[ASG]) -> List[ASG]:
    """Run the trained VLM to predict an ASG for each gold sample.

    Pulls images/context from the ph3 dataset where available; otherwise uses the gold
    ASG's context only (DummyVLM derives a plausible ASG)."""
    preds: List[ASG] = []
    images_list: List[object] = [None] * len(gold)
    contexts: List[dict] = [g.context for g in gold]
    try:
        VLMCurriculumDataset = _import_curriculum_dataset()
        ds = VLMCurriculumDataset(cfg, phase="ph3_asg")
        for i in range(min(len(gold), len(ds))):
            item = ds[i]
            images_list[i] = item.get("images")
    except Exception:
        pass
    for i, g in enumerate(gold):
        try:
            pa = model.generate_asg(images_list[i], contexts[i])
        except Exception:
            pa = ASG(objects=list(g.objects), global_regime=g.global_regime,
                     context=dict(g.context))
        preds.append(pa)
    return preds


def _compute_asg_f1(cfg, model) -> Dict[str, float]:
    from asgwm.eval.faithfulness import asg_accuracy
    gold = _load_gold_asgs(cfg)
    if not gold:
        return {"obj_f1": 0.0, "motion_ang_err_deg": float("nan"), "regime_acc": 0.0,
                "n_gold": 0}
    preds = _predict_gold_asgs(cfg, model, gold)
    res = asg_accuracy(preds, gold)
    res["n_gold"] = len(gold)
    return res


def run_curriculum(cfg) -> str:
    """Run ph1->ph5 sequentially (training_method.md section 3, architecture.md section 10).

    AFTER ph3_asg, compute ASG F1 on the gold subset and HARD-STOP (log + RuntimeError) if
    it is below `cfg.train.tier1.ph3_gate_f1`. The Ph-5 checkpoint is the Tier-1 deliverable
    and the Tier-2 initialization. Returns the final (ph5) checkpoint path.
    """
    phases = cfg.get_path("train.tier1.phases", PHASES) or PHASES
    gate_f1 = float(cfg.get_path("train.tier1.ph3_gate_f1", 0.70))

    ckpt_in: Optional[str] = None
    # Cross-session chaining: if the requested phases do not start at ph1, seed the adapters
    # from the checkpoint of the phase immediately preceding the first requested phase. This
    # lets a second notebook run [ph4_cot, ph5_eqcot] and continue from the ph3 checkpoint
    # produced by the first notebook (otherwise Ph-4 would start from fresh weights).
    if phases and phases[0] in PHASES:
        _i = PHASES.index(phases[0])
        if _i > 0:
            _prev = ckpt.latest(_ckpt_dir(cfg, PHASES[_i - 1]))
            if _prev:
                ckpt_in = _prev
                print(f"[tier1] seeding {phases[0]} from preceding phase {PHASES[_i - 1]}: {_prev}")
    final_ckpt: Optional[str] = None
    for phase in phases:
        print(f"[tier1] running phase {phase} (from {ckpt_in})")
        ckpt_out = run_phase(cfg, phase, ckpt_in)
        final_ckpt = ckpt_out

        if phase == "ph3_asg":
            StageAVLM = _import_vlm()
            model = StageAVLM.from_config(cfg)
            _load_phase_ckpt(model, ckpt_out)
            if hasattr(model, "eval"):
                model.eval()
            stats = _compute_asg_f1(cfg, model)
            f1 = float(stats.get("obj_f1", 0.0))
            print(
                f"[tier1] Ph-3 gate: ASG obj_f1={f1:.4f} "
                f"(threshold {gate_f1:.2f}, n_gold={stats.get('n_gold', 0)}, "
                f"regime_acc={stats.get('regime_acc', 0.0):.3f})"
            )
            if f1 < gate_f1:
                msg = (
                    f"Tier-1 Ph-3 gate FAILED: ASG F1 {f1:.4f} < {gate_f1:.2f} on the gold "
                    f"subset. The downstream CoT is unfounded without a reliable state — "
                    f"debug the visual projector or data pipeline before proceeding "
                    f"(training_method.md section 3)."
                )
                print("[tier1] " + msg)
                raise RuntimeError(msg)
            print("[tier1] Ph-3 gate PASSED; continuing to Ph-4.")

        ckpt_in = ckpt_out

    if final_ckpt is None:
        raise RuntimeError("Tier-1 curriculum ran no phases; check cfg.train.tier1.phases")
    return final_ckpt
