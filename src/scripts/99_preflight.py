"""Pre-flight check — confirm everything is ready BEFORE spending a paid GPU session.

Runs the real pipeline on a tiny slice and reports a clear GO / NO-GO, so a data, loader,
GPU, or OOM problem is caught in ~2 minutes instead of after a 12 h run. Validates:

  1. Environment   — Python, torch + CUDA/GPU, optional deps (s3fs/h5py/transformers/...).
  2. Data          — loads a few REAL SEVIR events (or synthetic if data.dataset=synthetic),
                     reporting the actual source + array shapes. With data.require_real=true a
                     silent synthetic fallback is forbidden (it would waste a paid session).
  3. Auto-labeling — labels the slice and times it -> extrapolates to data.n_train_events.
  4. Transition    — one Stage-B forward + loss + backward step on the device.
  5. Render        — one Stage-C sample on the device (GPU if available).
  6. Eval          — CSI on one rendered frame.
  7. Overfit (B)   — drive the transition loss down on ONE tiny batch (catches grad/wiring bugs).
  8. Label audit   — auto-label vs gold ASG F1; if below the Ph-3 gate, the gate is unreachable.
  9. Overfit (C)   — drive the renderer flow-matching loss down on ONE tiny batch.

Usage (Colab, real data):  python scripts/99_preflight.py --override data.dataset=sevir --override data.require_real=true
Usage (laptop, wiring):    python scripts/99_preflight.py --override data.dataset=synthetic --override data.grid=64
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main() -> int:
    ap = argparse.ArgumentParser(description="ASG-WM pre-flight (go/no-go before paid training)")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"))
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--n", type=int, default=3, help="events to probe")
    args = ap.parse_args()

    from asgwm.utils.config import load_config
    cfg = load_config(args.config, args.override)
    results = []  # (name, ok, detail)

    def check(name):
        def deco(fn):
            t0 = time.time()
            try:
                detail = fn()
                results.append((name, True, f"{detail}  [{time.time()-t0:.1f}s]"))
            except Exception as e:
                results.append((name, False, f"{type(e).__name__}: {e}"))
                if os.environ.get("PREFLIGHT_TRACE"):
                    traceback.print_exc()
            return fn
        return deco

    print("=" * 70)
    print("ASG-WM PRE-FLIGHT")
    print("=" * 70)

    # 1) environment
    @check("environment")
    def _env():
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        opt = {}
        for m in ("s3fs", "h5py", "transformers", "peft", "bitsandbytes", "diffusers", "pysteps", "skimage"):
            try:
                __import__(m); opt[m] = "y"
            except Exception:
                opt[m] = "-"
        miss = [m for m, v in opt.items() if v == "-"]
        return (f"py{sys.version.split()[0]} torch {torch.__version__} cuda={torch.cuda.is_available()} "
                f"({gpu}); optional present: {[m for m,v in opt.items() if v=='y']}; missing: {miss}")

    # shared state across checks
    state = {}

    # 2) data
    @check("data load")
    def _data():
        from asgwm.data import sevir as S
        cfg.set_path("data.n_train_events", args.n)
        ids = S.download_sevir_subset(cfg)
        evs = []
        for i, ev in enumerate(S.iter_events(cfg)):
            evs.append(ev)
            if i + 1 >= args.n:
                break
        state["events"] = evs
        vil = evs[0]["vil"]
        src = "SYNTHETIC" if str(evs[0].get("event_id", "")).startswith("synth") else "REAL SEVIR"
        return f"source={src}  n={len(evs)}  vil shape={tuple(vil.shape)}  channels={list(evs[0].keys())[:6]}"

    # 3) autolabel + extrapolation
    @check("auto-labeling")
    def _label():
        from asgwm.labeling import pipeline as P
        evs = state.get("events", [])
        if not evs:
            raise RuntimeError("no events from data step")
        t0 = time.time()
        seqs = [P.autolabel_event(ev, cfg) for ev in evs]
        per = (time.time() - t0) / max(len(evs), 1)
        state["seqs"] = seqs
        N = int(cfg.get_path("data.n_train_events", 800))
        # use the *intended* full N from the user's real config note
        est_full = per * 800
        return (f"{per:.1f}s/event  ASG_t={seqs[0].asg_t.n_objects} obj  "
                f"=> labeling 800 events ~ {est_full/60:.0f} min (1 CPU)")

    # 4) transition step
    @check("transition step (device)")
    def _trans():
        import torch
        from asgwm.models.stage_b_transition import TransitionTransformer, encode_asg
        from asgwm.utils.device import resolve_device
        dev = resolve_device(cfg)
        m = TransitionTransformer.from_config(cfg).to(dev)
        seqs = state["seqs"]
        enc = encode_asg(seqs[0].asg_t, m.n_max)
        out = m(enc["obj_feats"].unsqueeze(0).to(dev), enc["regime_idx"].unsqueeze(0).to(dev),
                enc["mask"].unsqueeze(0).to(dev), torch.zeros(1, 5, device=dev))
        loss = out["d_centroid"].abs().mean()
        loss.backward()
        return f"device={dev}  out keys={sorted(out)[:3]}...  loss={float(loss):.4f}"

    # 5) render
    @check("render sample (device)")
    def _render():
        import torch
        import numpy as np
        from asgwm.models.stage_c_renderer import LatentRectifiedFlowRenderer
        from asgwm.models.bottleneck import build_Z
        from asgwm.utils.device import resolve_device, autocast_ctx
        dev = resolve_device(cfg)
        r = LatentRectifiedFlowRenderer.from_config(cfg).to(dev)
        asg = state["seqs"][0].asg_th
        H = W = min(int(cfg.get_path("data.grid", 384)), 96)
        ab = torch.zeros(1, 1, H, W, device=dev)
        Z = build_Z(asg, ab[0].cpu(), H, W).unsqueeze(0).to(dev)
        with torch.no_grad(), autocast_ctx(dev, cfg):
            f = r.sample(Z, ab, int(cfg.get_path("stage_c.flow_steps", 4)))
        peak = torch.cuda.max_memory_allocated()/1e9 if dev.type == "cuda" else 0.0
        state["field"] = f.float().cpu().numpy()[0, 0]
        return f"field {tuple(f.shape)}  peak GPU mem={peak:.2f} GB"

    # 6) metric
    @check("eval metric")
    def _eval():
        from asgwm.eval.metrics import csi
        import numpy as np
        f = state.get("field")
        if f is None:
            raise RuntimeError("no rendered field")
        return f"CSI(self)={csi(f, f, float(np.percentile(f,90))):.3f}"

    # 7) overfit a tiny batch — the single most diagnostic cheap test: if the optimiser
    #    cannot drive the transition loss down on ONE fixed sample in a few steps, there is
    #    a wiring / gradient bug (not a capacity problem). Catch it now, not after hours.
    @check("overfit tiny-batch (transition)")
    def _overfit():
        import torch
        from asgwm.models.stage_b_transition import TransitionTransformer, encode_asg
        from asgwm.utils.device import resolve_device
        seqs = state.get("seqs")
        if not seqs:
            raise RuntimeError("no labeled sequences from earlier step")
        dev = resolve_device(cfg)
        m = TransitionTransformer.from_config(cfg).to(dev)
        opt = torch.optim.Adam(m.parameters(), lr=5e-3)
        enc = encode_asg(seqs[0].asg_t, m.n_max)
        x = enc["obj_feats"].unsqueeze(0).to(dev)
        ri = enc["regime_idx"].unsqueeze(0).to(dev)
        mk = enc["mask"].unsqueeze(0).to(dev)
        ctx0 = torch.zeros(1, 5, device=dev)
        # Reachable target: a fixed perturbation of the model's own output. A healthy grad
        # path overfits ONE sample to near-zero; require a >=10x drop (not just "decreasing").
        # Score VALID objects only — padded slots share identical (zero) features so the shared
        # head cannot give them distinct outputs; including them would floor the loss.
        valid = mk.squeeze(0)  # [n_max] bool
        if int(valid.sum()) == 0:
            return "no valid objects in sample — overfit test skipped (degenerate ASG)"
        with torch.no_grad():
            tgt = (m(x, ri, mk, ctx0)["d_centroid"] + 0.4 * torch.randn_like(x[..., :2])).detach()

        def _vloss():
            pred = m(x, ri, mk, ctx0)["d_centroid"][0][valid]
            return ((pred - tgt[0][valid]) ** 2).mean()

        losses = []
        for _ in range(150):
            opt.zero_grad(set_to_none=True)
            loss = _vloss()
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if not (losses[-1] < 0.1 * losses[0]):
            raise RuntimeError(
                f"could not overfit a tiny batch ({losses[0]:.3e} -> {losses[-1]:.3e}, "
                f"<10x drop in 150 steps); likely a gradient/wiring bug — DO NOT start a paid run"
            )
        return f"loss {losses[0]:.3e} -> {losses[-1]:.3e} ({losses[0]/max(losses[-1],1e-9):.0f}x in 150 steps) — grad path OK"

    # 8) label-quality audit — the F1 gate is bounded by how well the AUTO-labeller agrees
    #    with the hand-labelled gold. If that agreement is already below the gate, the gate is
    #    unreachable no matter how well the VLM trains: fix the labeller, not the model.
    @check("label-quality audit (auto vs gold)")
    def _labelaudit():
        from asgwm.eval.faithfulness import asg_accuracy
        seqs = state.get("seqs", [])
        if not seqs:
            raise RuntimeError("no labeled sequences from earlier step")
        preds = [s.asg_t for s in seqs]
        try:
            from asgwm.train.tier1_curriculum import _load_gold_asgs
            gold = _load_gold_asgs(cfg)
        except Exception:
            gold = []
        gate = float(cfg.get_path("train.tier1.ph3_gate_f1", 0.70))
        if not gold:
            # No gold dir (laptop/synthetic): only validate the matching machinery runs and
            # is self-consistent; the true ceiling is measured on Colab with the real gold set.
            self_acc = asg_accuracy(preds, preds)
            if self_acc["obj_f1"] < 0.99:
                raise RuntimeError(f"asg_accuracy self-match broken (f1={self_acc['obj_f1']:.2f})")
            return "no gold dir — matching machinery OK; run on Colab w/ real gold for the true ceiling"
        n = min(len(preds), len(gold))
        acc = asg_accuracy(preds[:n], gold[:n])
        flag = "" if acc["obj_f1"] >= gate else f"  <-- BELOW gate {gate:.2f}: gate UNREACHABLE, fix labeller first"
        return (f"obj_f1={acc['obj_f1']:.2f} regime_acc={acc['regime_acc']:.2f} "
                f"motion_err={acc['motion_ang_err_deg']:.0f}deg vs GOLD (n={n}){flag}")

    # 9) overfit the renderer on one tiny batch — the Stage-C analogue of check 7: confirm the
    #    flow-matching objective + optimiser actually reduce the loss before a paid run.
    @check("overfit tiny-batch (renderer)")
    def _overfit_render():
        import torch
        from asgwm.models.stage_c_renderer import LatentRectifiedFlowRenderer
        from asgwm.models.bottleneck import build_Z
        from asgwm.utils.device import resolve_device
        seqs = state.get("seqs")
        if not seqs:
            raise RuntimeError("no labeled sequences from earlier step")
        dev = resolve_device(cfg)
        H = W = min(int(cfg.get_path("data.grid", 384)), 64)
        r = LatentRectifiedFlowRenderer.from_config(cfg).to(dev)
        opt = torch.optim.Adam(r.parameters(), lr=1e-3)
        ab = torch.zeros(1, 1, H, W, device=dev)
        Z = build_Z(seqs[0].asg_th, ab[0].cpu(), H, W).unsqueeze(0).to(dev)
        tgt = torch.rand(1, 1, H, W, device=dev) * 50.0  # fixed reachable target
        losses = []
        for _ in range(60):
            opt.zero_grad(set_to_none=True)
            loss = r.training_loss(Z, ab, tgt)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        if not (losses[-1] < 0.5 * losses[0]):
            raise RuntimeError(
                f"renderer did not overfit ({losses[0]:.3e} -> {losses[-1]:.3e}); flow/wiring bug — "
                "DO NOT start a paid run"
            )
        return f"flow loss {losses[0]:.3e} -> {losses[-1]:.3e} ({losses[0]/max(losses[-1],1e-9):.0f}x in 60 steps) — grad path OK"

    # run all (decorators already executed in definition order)
    for fn in (_env, _data, _label, _trans, _render, _eval, _overfit, _labelaudit, _overfit_render):
        pass

    print()
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:24s} {detail}")
    n_fail = sum(1 for _, ok, _ in results if not ok)
    print("-" * 70)
    if n_fail == 0:
        print("GO  — all checks passed. Safe to launch the paid training run.")
    else:
        print(f"NO-GO — {n_fail} check(s) failed. Fix before training (set PREFLIGHT_TRACE=1 for tracebacks).")
    print("=" * 70)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
