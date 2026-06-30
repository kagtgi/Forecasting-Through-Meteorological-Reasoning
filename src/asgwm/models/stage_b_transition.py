"""Stage B — the transition transformer ASG_t -> ASG_{t+h} (architecture.md section 3).

A small transformer over ASG object tokens (+ a context token) predicts the evolved
scene graph at horizon ``h``.  Three governing-equation injection points (all
compute-cheap: loss terms + a warp, no extra parameters):

1. **Differentiable advection.** Object centroids and the growth field are advanced by
   the semi-Lagrangian point warp (``physics.advect_points``); the network predicts the
   *residual* on top of advection, so the linear-motion baseline is built in
   (``cfg.stage_b.predict_residual``).
2. **PINN-style residual losses** (``transition_loss``): a continuity / mass-conservation
   residual on the predicted growth field (``physics.continuity_residual``) and a
   smoothness residual on the motion field (``physics.motion_smoothness_residual``).
3. Equation-aware prompting lives in Stage A; this module owns (1) and (2).

The transition unit of supervision is the ``ASGSequence`` (asg_t, asg_th, horizon).
Object tokens carry the 8-dim ``StormObject.to_vector()`` features plus a learned
regime embedding; a context token injects the 5 environmental scalars
(CAPE/CIN/shear/PWAT/DEM).  Outputs are *residuals*: centroid displacement, attribute
deltas, updated regime logits, and a low-res growth field.

References: architecture.md sections 3, 4, 8; training_method.md section 4 (IB cap).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from asgwm.asg import (
    ASG,
    StormObject,
    REGIMES,
    REGIME_TO_IDX,
    IDX_TO_REGIME,
    N_MAX,
)
from asgwm import physics
from asgwm.utils.config import Config

# Continuous attribute residuals the network predicts per object, in addition to the
# 2-d centroid displacement: [d_area, d_peak, d_vy, d_vx, d_growth]  -> k = 5.
N_ATTR_RESID: int = 5


# ---------------------------------------------------------------------------
# ASG <-> tensor encoding (binding signatures used by the dataset + trainer).
# ---------------------------------------------------------------------------
def encode_asg(asg: ASG, n_max: int = N_MAX) -> Dict[str, torch.Tensor]:
    """Encode an ASG into padded tensors for the transition transformer.

    Pads / truncates to ``n_max`` objects (the hard IB cap, training_method.md
    section 4).  The 8-dim object feature vector is exactly ``StormObject.to_vector()``
    = ``[cy, cx, area, peak, vy, vx, growth, conf]``.

    Args:
        asg: the source scene graph (typically ASG_t).
        n_max: padded object slot count.

    Returns:
        dict with:
          ``obj_feats``  : ``[N, 8]`` float
          ``regime_idx`` : ``[N]``   long  (REGIME_TO_IDX; 0 for padding)
          ``mask``       : ``[N]``   bool  (True for real objects)
          ``centroids``  : ``[N, 2]`` float (cy, cx)
          ``motion``     : ``[N, 2]`` float (vy, vx)
    """
    objs = asg.objects[:n_max]
    obj_feats = torch.zeros(n_max, 8, dtype=torch.float32)
    regime_idx = torch.zeros(n_max, dtype=torch.long)
    mask = torch.zeros(n_max, dtype=torch.bool)
    centroids = torch.zeros(n_max, 2, dtype=torch.float32)
    motion = torch.zeros(n_max, 2, dtype=torch.float32)

    for i, o in enumerate(objs):
        obj_feats[i] = torch.from_numpy(o.to_vector())
        regime_idx[i] = REGIME_TO_IDX.get(o.regime, REGIME_TO_IDX["steady"])
        mask[i] = True
        centroids[i, 0] = float(o.cy)
        centroids[i, 1] = float(o.cx)
        motion[i, 0] = float(o.vy)
        motion[i, 1] = float(o.vx)

    return {
        "obj_feats": obj_feats,
        "regime_idx": regime_idx,
        "mask": mask,
        "centroids": centroids,
        "motion": motion,
    }


def decode_asg(out: Dict[str, torch.Tensor], base_asg: ASG) -> ASG:
    """Apply predicted residuals (``out``) to ``base_asg`` to produce ASG_{t+h}.

    The residuals are interpreted on top of the *advected* base state:
    ``out`` must contain ``centroid`` (absolute predicted [N,2]), ``attr`` ([N,5] =
    d_area,d_peak,d_vy,d_vx,d_growth deltas relative to the base object), and
    ``regime_idx`` ([N] long).  Padded slots are dropped via ``mask`` if present.

    Args:
        out: residual / prediction dict (single sample; tensors with a leading object
            dim, no batch dim).
        base_asg: the ASG_t whose objects are evolved.

    Returns:
        A new ASG with evolved objects, capped to the IB budget.
    """
    centroid = out["centroid"]
    attr = out["attr"]
    regime_idx = out.get("regime_idx")
    mask = out.get("mask")

    centroid = centroid.detach().cpu()
    attr = attr.detach().cpu()
    if regime_idx is not None:
        regime_idx = regime_idx.detach().cpu()

    new_objects: List[StormObject] = []
    n = min(len(base_asg.objects), centroid.shape[0])
    for i in range(n):
        if mask is not None and not bool(mask[i]):
            continue
        b = base_asg.objects[i]
        cy = float(centroid[i, 0])
        cx = float(centroid[i, 1])
        d_area, d_peak, d_vy, d_vx, d_growth = (float(x) for x in attr[i].tolist())
        if regime_idx is not None:
            reg = IDX_TO_REGIME.get(int(regime_idx[i]), b.regime)
        else:
            reg = b.regime
        new_objects.append(
            StormObject(
                id=b.id,
                cy=cy,
                cx=cx,
                area=max(b.area + d_area, 0.0),
                peak=max(b.peak + d_peak, 0.0),
                vy=b.vy + d_vy,
                vx=b.vx + d_vx,
                regime=reg,
                growth=b.growth + d_growth,
                conf=b.conf,
                sigma_c=b.sigma_c,
                sigma_v=b.sigma_v,
                sigma_g=b.sigma_g,
            )
        )

    new_regime = base_asg.global_regime
    if "global_regime_idx" in out:
        gi = int(out["global_regime_idx"])
        new_regime = IDX_TO_REGIME.get(gi, base_asg.global_regime)

    evolved = ASG(
        objects=new_objects,
        global_regime=new_regime,
        growth_field=out.get("growth_field_np", base_asg.growth_field),
        context=dict(base_asg.context),
        meta=dict(base_asg.meta),
    )
    return evolved.capped(N_MAX)


# ---------------------------------------------------------------------------
# The transition transformer.
# ---------------------------------------------------------------------------
class TransitionTransformer(nn.Module):
    """ASG-token transformer that predicts the residual evolution to horizon ``h``.

    Architecture (≈10–50 M params, architecture.md section 7):
      * linear embed of the 8-dim object features + learned regime embedding,
      * a context token from the 5 environmental scalars,
      * ``n_layers`` standard pre-norm Transformer encoder blocks with padding mask,
      * residual heads: centroid displacement, attribute deltas, regime logits, and a
        low-resolution growth field decoded from the pooled scene token.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        obj_feat_dim: int = 8,
        context_dim: int = 5,
        n_max: int = N_MAX,
        growth_field_size: int = 48,
        predict_residual: bool = True,
        dt: float = 1.0,
        km_per_pixel: float = 1.0,
        minutes_per_frame: float = 5.0,
        lambda_continuity: float = 0.1,
        lambda_smooth: float = 0.01,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_max = n_max
        self.growth_field_size = growth_field_size
        self.predict_residual = predict_residual
        self.dt = dt
        self.km_per_pixel = km_per_pixel
        self.minutes_per_frame = minutes_per_frame
        self.lambda_continuity = lambda_continuity
        self.lambda_smooth = lambda_smooth
        self.n_regimes = len(REGIMES)

        self.obj_proj = nn.Linear(obj_feat_dim, d_model)
        self.regime_emb = nn.Embedding(self.n_regimes, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_max, d_model))
        nn.init.normal_(self.pos_emb, std=0.02)
        self.ctx_proj = nn.Linear(context_dim, d_model)
        self.ctx_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.ctx_token, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        # Residual heads (per-object).
        self.head_centroid = nn.Linear(d_model, 2)            # d_centroid (dy, dx)
        self.head_attr = nn.Linear(d_model, N_ATTR_RESID)     # d_area,d_peak,d_vy,d_vx,d_growth
        self.head_regime = nn.Linear(d_model, self.n_regimes)  # regime logits

        # Growth-field head (from the pooled context/scene token).
        self.gf_hidden = 64
        self.gf_fc = nn.Linear(d_model, self.gf_hidden * 6 * 6)
        self.gf_deconv = nn.Sequential(
            nn.GroupNorm(8, self.gf_hidden),
            nn.SiLU(),
            nn.ConvTranspose2d(self.gf_hidden, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
        )  # 6 -> 48 (3x stride-2)

    # ---- construction ----------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config) -> "TransitionTransformer":
        """Build from ``cfg.stage_b`` (+ ``cfg.asg.growth_field_size``)."""
        sb = cfg.get("stage_b", {}) if hasattr(cfg, "get") else cfg["stage_b"]
        gf = int(cfg.get_path("asg.growth_field_size", 48)) if hasattr(cfg, "get_path") else 48

        def _g(key, default):
            try:
                return sb.get(key, default)
            except AttributeError:
                return sb[key] if key in sb else default

        return cls(
            d_model=int(_g("d_model", 256)),
            n_layers=int(_g("n_layers", 6)),
            n_heads=int(_g("n_heads", 8)),
            d_ff=int(_g("d_ff", 1024)),
            dropout=float(_g("dropout", 0.1)),
            obj_feat_dim=int(_g("obj_feat_dim", 8)),
            context_dim=int(_g("context_dim", 5)),
            n_max=int(cfg.get_path("asg.n_max", N_MAX)) if hasattr(cfg, "get_path") else N_MAX,
            growth_field_size=gf,
            predict_residual=bool(_g("predict_residual", True)),
            dt=float(cfg.get_path("data.horizon_min", 60.0) / max(cfg.get_path("data.minutes_per_frame", 5.0), 1.0))
            if hasattr(cfg, "get_path") else 1.0,
            km_per_pixel=float(cfg.get_path("data.km_per_pixel", 1.0)) if hasattr(cfg, "get_path") else 1.0,
            minutes_per_frame=float(cfg.get_path("data.minutes_per_frame", 5.0))
            if hasattr(cfg, "get_path") else 5.0,
            lambda_continuity=float(_g("lambda_continuity", 0.1)),
            lambda_smooth=float(_g("lambda_smooth", 0.01)),
        )

    # ---- forward ---------------------------------------------------------
    def forward(
        self,
        obj_feats: torch.Tensor,
        regime_idx: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict residual evolution for a batch of ASGs.

        Args:
            obj_feats: ``[B, N, 8]`` object features.
            regime_idx: ``[B, N]`` long regime indices.
            mask: ``[B, N]`` bool (True = real object).
            context: ``[B, 5]`` environmental scalars.

        Returns:
            dict with ``d_centroid`` ``[B,N,2]``, ``d_attr`` ``[B,N,5]``,
            ``regime_logits`` ``[B,N,4]``, ``growth_field`` ``[B,1,h,w]``.
        """
        b, n, _ = obj_feats.shape
        tok = self.obj_proj(obj_feats) + self.regime_emb(regime_idx) + self.pos_emb[:, :n]
        ctx = self.ctx_proj(context).unsqueeze(1) + self.ctx_token  # [B,1,d]
        seq = torch.cat([ctx, tok], dim=1)  # [B, 1+N, d]

        # key_padding_mask: True where to ignore. Context token is always valid.
        pad = ~mask  # [B, N]
        ctx_pad = torch.zeros(b, 1, dtype=torch.bool, device=mask.device)
        key_padding_mask = torch.cat([ctx_pad, pad], dim=1)  # [B, 1+N]

        h = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        h = self.out_norm(h)
        ctx_h = h[:, 0]        # [B, d] pooled scene/context token
        obj_h = h[:, 1:]       # [B, N, d]

        d_centroid = self.head_centroid(obj_h)              # [B,N,2]
        d_attr = self.head_attr(obj_h)                      # [B,N,5]
        regime_logits = self.head_regime(obj_h)             # [B,N,4]

        gf = self.gf_fc(ctx_h).view(b, self.gf_hidden, 6, 6)
        gf = self.gf_deconv(gf)                             # [B,1,48,48]
        if gf.shape[-1] != self.growth_field_size:
            gf = F.interpolate(
                gf, size=(self.growth_field_size, self.growth_field_size),
                mode="bilinear", align_corners=False,
            )

        return {
            "d_centroid": d_centroid,
            "d_attr": d_attr,
            "regime_logits": regime_logits,
            "growth_field": gf,
        }

    # ---- inference: residual-on-advection -------------------------------
    @torch.no_grad()
    def predict(self, asg_t: ASG, context_vec: Optional[torch.Tensor] = None) -> ASG:
        """Predict ASG_{t+h} from ASG_t as a residual on the advected base state.

        The base future state is the present advected by each object's own motion
        (``physics.advect_points``); the network's centroid head then adds a learned
        *residual* on top (``cfg.stage_b.predict_residual``).  With ``predict_residual``
        off, the centroid head predicts the displacement directly from the present.

        Args:
            asg_t: current scene graph.
            context_vec: ``[5]`` or ``[1,5]`` environmental scalars; zeros if None.

        Returns:
            Predicted ASG_{t+h}.
        """
        self.eval()
        device = self.pos_emb.device
        enc = encode_asg(asg_t, self.n_max)
        obj_feats = enc["obj_feats"].unsqueeze(0).to(device)
        regime_idx = enc["regime_idx"].unsqueeze(0).to(device)
        mask = enc["mask"].unsqueeze(0).to(device)
        if context_vec is None:
            context = torch.zeros(1, self.ctx_proj.in_features, device=device)
        else:
            context = torch.as_tensor(context_vec, dtype=torch.float32, device=device).view(1, -1)

        out = self.forward(obj_feats, regime_idx, mask, context)

        centroids = enc["centroids"].to(device)   # [N,2]
        motion = enc["motion"].to(device)         # [N,2]
        d_centroid = out["d_centroid"][0]          # [N,2]

        if self.predict_residual:
            motion_px = physics.kmh_to_px_per_step(
                motion, self.km_per_pixel, self.minutes_per_frame
            )
            advected = physics.advect_points(centroids, motion_px, dt=self.dt)  # [N,2]
            new_centroid = advected + d_centroid
        else:
            new_centroid = centroids + d_centroid

        regime_pred = out["regime_logits"][0].argmax(dim=-1)  # [N]

        # Map the predicted low-res growth field back to a numpy array on the ASG.
        gf_np = out["growth_field"][0, 0].detach().cpu().numpy()

        decoded = decode_asg(
            {
                "centroid": new_centroid,
                "attr": out["d_attr"][0],
                "regime_idx": regime_pred,
                "mask": enc["mask"],
                "growth_field_np": gf_np,
            },
            asg_t,
        )
        return decoded


# ---------------------------------------------------------------------------
# PINN-style transition loss (architecture.md section 3.2; weights cfg.stage_b.lambda_*).
# ---------------------------------------------------------------------------
def transition_loss(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    flow: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Supervised + physics-residual loss for the transition transformer.

    Combines object-level supervision (position, attributes, regime, growth) with the
    two governing-equation residuals: a continuity / mass-conservation residual on the
    predicted growth field (``physics.continuity_residual``) and a motion-field
    smoothness residual (``physics.motion_smoothness_residual``).

    Expected ``pred`` keys: ``d_centroid``/``new_centroid`` ``[B,N,2]``, ``d_attr``
    ``[B,N,5]``, ``regime_logits`` ``[B,N,4]``, ``growth_field`` ``[B,1,h,w]``.
    Expected ``target`` keys: ``centroid`` ``[B,N,2]`` (residual target matching pred),
    ``attr`` ``[B,N,5]``, ``regime_idx`` ``[B,N]`` long, ``growth_field`` ``[B,1,h,w]``,
    ``mask`` ``[B,N]`` bool, and optionally ``growth_field_prev`` ``[B,1,h,w]`` for the
    continuity residual (defaults to zeros).

    Args:
        pred: model outputs.
        target: ground-truth tensors.
        flow: motion field ``[B,2,h,w]`` (px/step) used by the physics residuals,
            sampled at the growth-field resolution.

    Returns:
        dict(total, pos, regime, growth, continuity, smooth) of scalar tensors.
        Component weights: continuity = ``lambda_continuity`` (default 0.1),
        smooth = ``lambda_smooth`` (default 0.01).
    """
    lambda_continuity = float(target.get("lambda_continuity", 0.1)) if isinstance(target, dict) else 0.1
    lambda_smooth = float(target.get("lambda_smooth", 0.01)) if isinstance(target, dict) else 0.01

    pred_centroid = pred.get("d_centroid")
    if pred_centroid is None:
        pred_centroid = pred["new_centroid"]
    tgt_centroid = target["centroid"]
    mask = target.get("mask")
    if mask is None:
        mask = torch.ones(pred_centroid.shape[:2], dtype=torch.bool, device=pred_centroid.device)
    m = mask.unsqueeze(-1).float()  # [B,N,1]
    denom = m.sum().clamp(min=1.0)

    # Position (centroid) regression — Smooth-L1 over real objects only.
    pos = (F.smooth_l1_loss(pred_centroid, tgt_centroid, reduction="none") * m).sum() / (denom * 2)

    # Attribute deltas.
    attr_loss = pred_centroid.new_zeros(())
    if "d_attr" in pred and "attr" in target:
        a = (F.smooth_l1_loss(pred["d_attr"], target["attr"], reduction="none") * m).sum() / (
            denom * pred["d_attr"].shape[-1]
        )
        attr_loss = a

    # Regime classification — cross-entropy over real objects.
    regime = pred_centroid.new_zeros(())
    if "regime_logits" in pred and "regime_idx" in target:
        logits = pred["regime_logits"]
        b, n, c = logits.shape
        ce = F.cross_entropy(
            logits.reshape(b * n, c),
            target["regime_idx"].reshape(b * n),
            reduction="none",
        ).reshape(b, n)
        regime = (ce * mask.float()).sum() / denom

    # Growth-field regression.
    growth = pred_centroid.new_zeros(())
    pred_gf = pred.get("growth_field")
    tgt_gf = target.get("growth_field")
    if pred_gf is not None and tgt_gf is not None:
        growth = F.mse_loss(pred_gf, tgt_gf)

    # --- Physics residuals (PINN, architecture.md section 3.2) ---
    continuity = pred_centroid.new_zeros(())
    if pred_gf is not None:
        g_prev = target.get("growth_field_prev")
        if g_prev is None:
            g_prev = torch.zeros_like(pred_gf)
        f = flow
        if f.shape[-2:] != pred_gf.shape[-2:]:
            f = F.interpolate(f, size=pred_gf.shape[-2:], mode="bilinear", align_corners=False)
        continuity = physics.continuity_residual(g_prev, pred_gf, f, dt=1.0)

    smooth = physics.motion_smoothness_residual(flow)

    total = (
        pos
        + attr_loss
        + regime
        + growth
        + lambda_continuity * continuity
        + lambda_smooth * smooth
    )

    return {
        "total": total,                 # carries grad — used for .backward()
        "pos": pos.detach(),            # the rest are detached scalars for logging
        "regime": regime.detach(),
        "growth": growth.detach(),
        "continuity": continuity.detach(),
        "smooth": smooth.detach(),
    }
