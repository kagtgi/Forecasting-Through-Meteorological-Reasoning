"""End-to-end CPU smoke test of the ASG-WM pipeline.

Drives the REAL stack on a handful of tiny synthetic events, exactly as the trainers do:

    SyntheticSEVIR / iter_events
        -> labeling.pipeline.autolabel_event           (ASG_t, ASG_{t+h})
        -> cache ASG json + build ASGTransitionDataset
        -> stage_b TransitionTransformer: forward + transition_loss + one backward step
        -> TransitionTransformer.predict -> ASG_{t+h}
        -> bottleneck.build_Z(asg_th, advect_blind)
        -> stage_c LatentRectifiedFlowRenderer.sample (CPU, IdentityVAE fallback)

Everything is shrunk to tiny tensors and 1 flow step so it runs in well under a second.
torch is required (Stage B/C), so the file is skipped wholesale in a torch-free env.

NOTE: ``labeling.pipeline`` writes ``meta['grid'] = [H, W]`` (a list, for JSON
round-tripping); ``ASGTransitionDataset`` consumes that form directly, so the documented
autolabel -> dataset flow runs with no caller-side massaging. See the
``test_meta_grid_is_list_from_pipeline`` guard which pins the canonical list format.
"""
from __future__ import annotations

import json
import os

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("yaml")

from asgwm.data import sevir as sevir_mod
from asgwm.labeling import pipeline
from asgwm.data.dataset import ASGTransitionDataset, collate_transition
from asgwm.data.advection import advect_blind
from asgwm.models.stage_b_transition import (
    TransitionTransformer,
    encode_asg,
    transition_loss,
)
from asgwm.models.bottleneck import build_Z, N_Z_CHANNELS
from asgwm.models.stage_c_renderer import LatentRectifiedFlowRenderer


_N_ITEMS = 3


def _autolabel_and_cache(cfg, n_items):
    """Label ``n_items`` synthetic events and write their ASG json to the cache."""
    asg_dir = sevir_mod.asg_dir(cfg)
    events = []
    for i, ev in enumerate(sevir_mod.iter_events(cfg)):
        if i >= n_items:
            break
        events.append(ev)
        seq = pipeline.autolabel_event(ev, cfg)
        eid = seq.event_id or f"event_{i:06d}"
        seq.event_id = eid
        payload = {
            "event_id": eid,
            "horizon_min": int(seq.horizon_min),
            "asg_t": seq.asg_t.to_dict(),
            "asg_th": seq.asg_th.to_dict(),
        }
        with open(os.path.join(asg_dir, f"{eid}.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f)
    return events


# ---------------------------------------------------------------------------
# Building blocks (each a small, independently-failing assertion)
# ---------------------------------------------------------------------------
def test_synthetic_sevir_generates_vil(tiny_cfg):
    synth = sevir_mod.SyntheticSEVIR(tiny_cfg)
    ev = synth.generate(0)
    assert ev["vil"].ndim == 3
    assert ev["vil"].shape[0] == synth.n_frames
    # deterministic in cfg.seed + index
    ev_again = sevir_mod.SyntheticSEVIR(tiny_cfg).generate(0)
    assert np.allclose(ev["vil"], ev_again["vil"])


def test_autolabel_event_produces_asg_sequence(tiny_cfg):
    ev = next(iter(sevir_mod.iter_events(tiny_cfg)))
    seq = pipeline.autolabel_event(ev, tiny_cfg)
    assert seq.asg_t is not None and seq.asg_th is not None
    assert seq.horizon_min == int(tiny_cfg.get_path("data.horizon_min"))
    # pipeline stashes the shared future-blind flow + grid on meta.
    assert "flow" in seq.asg_t.meta
    assert "grid" in seq.asg_t.meta


def test_meta_grid_is_list_from_pipeline(tiny_cfg):
    """Pin the canonical ``meta['grid'] = [H, W]`` list format the dataset consumes."""
    ev = next(iter(sevir_mod.iter_events(tiny_cfg)))
    seq = pipeline.autolabel_event(ev, tiny_cfg)
    assert isinstance(seq.asg_t.meta.get("grid"), list)


def test_transition_dataset_items_have_contract_shapes(tiny_cfg):
    _autolabel_and_cache(tiny_cfg, _N_ITEMS)
    ds = ASGTransitionDataset(tiny_cfg)
    assert len(ds) == _N_ITEMS
    item = ds[0]
    assert set(["asg_t", "context", "asg_th", "flow"]).issubset(item.keys())
    assert tuple(item["context"].shape) == (5,)
    flow_sz = int(tiny_cfg.get_path("asg.growth_field_size"))
    assert tuple(item["flow"].shape) == (2, flow_sz, flow_sz)


# ---------------------------------------------------------------------------
# Full chain: dataset -> transition -> Z -> renderer
# ---------------------------------------------------------------------------
def test_full_pipeline_one_step(tiny_cfg):
    events = _autolabel_and_cache(tiny_cfg, _N_ITEMS)
    ds = ASGTransitionDataset(tiny_cfg)
    batch = collate_transition([ds[i] for i in range(min(2, len(ds)))])

    model = TransitionTransformer.from_config(tiny_cfg)

    # encode the batch of ASGs into padded object tensors.
    encs = [encode_asg(a, n_max=model.n_max) for a in batch["asg_t"]]
    obj_feats = torch.stack([e["obj_feats"] for e in encs])
    regime_idx = torch.stack([e["regime_idx"] for e in encs])
    mask = torch.stack([e["mask"] for e in encs])

    out = model(obj_feats, regime_idx, mask, batch["context"])
    assert set(["d_centroid", "d_attr", "regime_logits", "growth_field"]).issubset(out.keys())
    b, n = obj_feats.shape[:2]
    assert tuple(out["d_centroid"].shape) == (b, n, 2)
    assert tuple(out["d_attr"].shape) == (b, n, 5)
    assert tuple(out["regime_logits"].shape) == (b, n, 4)

    # one transition_loss + backward step.
    tgt_encs = [encode_asg(a, n_max=model.n_max) for a in batch["asg_th"]]
    target = {
        "centroid": out["d_centroid"].detach(),
        "attr": torch.zeros_like(out["d_attr"]),
        "regime_idx": torch.stack([e["regime_idx"] for e in tgt_encs]),
        "growth_field": torch.zeros_like(out["growth_field"]),
        "mask": mask,
    }
    losses = transition_loss(out, target, batch["flow"])
    assert set(["total", "pos", "regime", "growth", "continuity", "smooth"]).issubset(losses.keys())
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0

    # inference: predict an ASG_{t+h}.
    pred_asg = model.predict(batch["asg_t"][0], batch["context"][0])
    assert pred_asg.n_objects == batch["asg_t"][0].n_objects

    # build the faithful bottleneck Z and render on CPU.
    H = W = 16
    vil = np.asarray(events[0]["vil"], dtype=np.float32)
    k = int(tiny_cfg.get_path("data.in_frames"))
    ab = advect_blind(vil[:k], n_out=1)[-1][:H, :W]
    adv = torch.from_numpy(np.ascontiguousarray(ab)).unsqueeze(0).float()  # [1,H,W]

    Z = build_Z(pred_asg, adv, H, W)
    assert tuple(Z.shape) == (N_Z_CHANNELS, H, W)

    renderer = LatentRectifiedFlowRenderer.from_config(tiny_cfg, cond_ch=N_Z_CHANNELS)
    field = renderer.sample(Z.unsqueeze(0), adv.unsqueeze(0), steps=1)
    assert tuple(field.shape) == (1, 1, H, W)
    assert torch.isfinite(field).all()


def test_build_Z_zeroed_keeps_advection(tiny_cfg):
    """The C-ii zeroed condition: zeroing ASG channels keeps only advect_blind."""
    from asgwm.models.bottleneck import zero_asg_in_Z, N_ASG_CHANNELS

    events = _autolabel_and_cache(tiny_cfg, 1)
    ds = ASGTransitionDataset(tiny_cfg)
    asg_th = ds[0]["asg_th"]

    H = W = 16
    vil = np.asarray(events[0]["vil"], dtype=np.float32)
    k = int(tiny_cfg.get_path("data.in_frames"))
    ab = advect_blind(vil[:k], n_out=1)[-1][:H, :W]
    adv = torch.from_numpy(np.ascontiguousarray(ab)).unsqueeze(0).float()

    Z = build_Z(asg_th, adv, H, W)
    Z0 = zero_asg_in_Z(Z)
    # ASG channels zeroed, advect_blind channel preserved.
    assert torch.all(Z0[:N_ASG_CHANNELS] == 0)
    assert torch.allclose(Z0[N_ASG_CHANNELS:], Z[N_ASG_CHANNELS:])
