# COMPUTE — VRAM, time, and cost to run ASG-WM end-to-end

*Engineering estimates (not experimental results). Model sizes are **measured** from the code;
VRAM/time/cost are derived from those sizes, the step budgets in `src/configs/default.yaml`, and
typical A100/L4 throughput. Skill numbers stay **[TBR]** until the run — these are resource
estimates only. Companion to `RUN_PLAN.md` (decision gates) and `TUTORIAL.md` (how-to).*

## Measured model sizes

| Component | Params | Notes |
|---|---|---|
| Stage A — VLM (perception) | **2.2 B (SmolVLM) / 3 B (Qwen2.5-VL)** | loaded in **4-bit (QLoRA)**; only LoRA adapters (~10–40 M) train |
| Stage B — transition transformer | **5.4 M** | d_model 256, 6 layers; trivial |
| Stage C — renderer U-Net | **29.3 M** | rectified-flow velocity net |
| Stage C — SD-VAE (frozen) | **~83 M** | `stabilityai/sd-vae-ft-mse`; inference-only, no grads |

Everything except the VLM is small; the VLM dominates memory, and **Tier-2 (end-to-end, with the
2-render intervention-consistency step) is the peak**.

## GPU VRAM (training)

| Phase | What runs | VRAM (typical) | Fits on |
|---|---|---|---|
| **Tier-0** (transition + deterministic renderer) | Stage B + Stage C U-Net + frozen VAE, bf16 | **~6–10 GB** | 12 GB; Stage B alone fits 8 GB / CPU |
| **Tier-1** (VLM 5-phase QLoRA curriculum) | 4-bit VLM + LoRA + multi-frame vision activations, grad-checkpointing, micro-batch 2 | **~10–16 GB** | **L4-24 GB** comfortably; tight on 16 GB |
| **Tier-2** (A→B→bottleneck→C end-to-end + intervention) | 4-bit VLM (low-LR) + B + C + 2–3 grad-tracked renders/step, patch 128, batch 8 | **~20–28 GB** | **A100-40 GB** (recommended) |
| **Tier-2 on a 24 GB card** | same, with `data.patch=96 train.tier2.batch_size=4 train.tier2.intervene_every=2` | **~16–22 GB** | **L4 / RTX 4090 / A10-24 GB** |
| **Eval / inference** | ensemble_k=10 renders, no grads | **~6–12 GB** | any 12 GB+ |

**Minimum to run the whole real pipeline:** a single **24 GB** GPU (L4, A10, RTX 4090) with the
Tier-2 knobs above. **Recommended:** one **40 GB A100** so Tier-2 runs at full patch/batch.
**Smoke/wiring only:** the 8 GB RTX 4060 (or CPU) runs the full chain on **synthetic data + DummyVLM**
(no real VLM, no download) — already validated, 102/102 tests green.

## Time (A100-40 GB, `FIRST_RUN=True` time-boxed config)

| Step | Driver | Wall-clock |
|---|---|---|
| Data download + auto-label (one-time) | SEVIR HDF5 GB-scale pull (the wildcard); pysteps labeling on CPU | **1–4 h** (cached; reused after) |
| Tier-0 | 4 k transition + ~4 k renderer steps | **0.5–1 h** |
| Tier-1 | 5 phases × 4 k = 20 k QLoRA steps + Ph-3 F1 gate | **3–5 h** |
| Tier-2 | 12 k steps × (2-render intervention) | **4–8 h** |
| Eval + figures | skill + faithfulness + gallery | **0.5–1 h** |
| **Total (FIRST_RUN)** | | **~10–18 h ≈ 2 A100 sessions (≤ 24 A100-hr incl. download)** |
| **Full publication run** (`FIRST_RUN=False`, larger `N_EVENTS`, more steps) | | **~24–40 A100-hr (budget a 3rd session)** |

The two `src/notebooks/run_all_colab_A100_runtime{1,2}.ipynb` are each designed to finish **< 12 h**;
the first-time data download is the only thing that can blow the budget (it is cached, so re-runs are fast).

## Cost

| Platform | Rate (A100-40 GB) | FIRST_RUN (~24 A100-hr) | Full run (~24–40 A100-hr) |
|---|---|---|---|
| **Colab Pro+** ($49.99/mo, 500 compute units) | A100 ≈ 12 CU/hr → ~42 A100-hr/mo | **fits in one month (~$50)** | may need 2 months / top-up |
| **Cloud on-demand** (Lambda ~$1.3, RunPod/Vast ~$1.6–2.5/hr) | ~$1.3–2.5/hr | **~$30–60** | **~$50–100** |
| **L4-24 GB** (Colab/cloud ~$0.6–1.0/hr) | cheaper, ~1.5–2× slower on Tier-2 | **~$30–60** (longer wall-clock) | **~$50–90** |

**Cheapest safe path:** run the pre-flight + label-F1 gates (minutes) on a free/L4 runtime first
(`RUN_PLAN.md` gates 1–2), then spend the A100 only once the gates are GREEN.

## Performance optimizations (enabled by default; accuracy-neutral)

Tuned for one A100-40GB finishing in budget without sacrificing the paper metrics:

- **bf16 autocast** (`train.precision: bf16`) on all three tiers + eval — ~½ the VRAM and faster
  matmuls than fp32 at no meaningful accuracy cost on the A100.
- **TF32 matmuls + cuDNN autotune** — auto-enabled whenever a CUDA device is resolved
  (`device.enable_perf`): free tensor-core speedup for the fp32/conv ops bf16 doesn't cover.
- **Gradient checkpointing** on the Stage-A VLM (trades ~20% compute for a large activation-memory
  saving so QLoRA fits comfortably).
- **QLoRA Stage A**: 4-bit NF4 + double-quant backbone, bf16 compute, frozen weights + LoRA — the
  2.2-3B VLM trains in ~10-16 GB.
- **VAE frozen + `@torch.no_grad`** encode/decode — no autograd graph through the 83M-param SD-VAE.
- **DataLoader workers** (`train.num_workers: 4`) overlap the CPU auto-labelling/IO with GPU compute
  (Linux/A100; auto-clamped to 0 on Windows). Set 0 if a loader stalls.
- **`tier2.intervene_every: 2`** — the intervention-consistency term needs 2 extra renders/step; doing
  it every other step ~halves the Tier-2 step cost. The faithfulness guarantee is architectural, so
  C-i stays high; set 1 for maximum C-i fidelity if time allows.

**VRAM safety valves** (if OOM on 40GB): `data.patch=96`, `train.tier2.batch_size=4`, keep
`intervene_every=2`. **Further speedups not enabled** (higher risk / one-time): `torch.compile` on the
renderer U-Net (opt-in once validated on the box), batching the eval ensemble render across lead
frames, and multiprocessing the one-time `01_autolabel` pass (cached after the first run, so amortized).

## Assumptions / caveats
- VLM memory/throughput are estimated for SmolVLM-2.2B/Qwen2.5-VL-3B QLoRA with grad-checkpointing; the
  exact peak depends on the chosen backbone, `image_size`, and how many history frames the processor packs.
- Tier-2's 2–3 grad-tracked renders/step is the dominant cost; `intervene_every=2` roughly halves it.
- Times scale with `n_train_events`, `max_steps_per_phase`, and `tier2.max_steps` — all in the config.
- Real-data extras must be installed for a real run (`s3fs h5py` SEVIR; `arm-pyart boto3` NEXRAD;
  `boto3 xarray cfgrib eccodes` MRMS); the notebooks install them per dataset.
