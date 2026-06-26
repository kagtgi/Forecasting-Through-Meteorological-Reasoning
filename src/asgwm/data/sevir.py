"""SEVIR loader + deterministic SyntheticSEVIR fallback (datasource.md sections 1-3).

The primary radar source is **SEVIR** (VIL radar + GOES-16 IR069/IR107 + GLM lightning)
on AWS Open Data ``s3://sevir`` (``veillette2020sevir``). Real SEVIR access goes through
:func:`download_sevir_subset` (s3fs + h5py); both deps are optional, so when they are
absent — or no network/bucket is configured — we transparently fall back to
:class:`SyntheticSEVIR`, a deterministic moving-Gaussian-blob VIL generator.

SyntheticSEVIR is what lets the WHOLE pipeline (label -> train -> eval) run on a laptop
CPU with **no S3 download and no GPU**: it produces seeded, physically-plausible VIL
sequences (moving, growing/decaying blobs) plus best-effort IR/GLM channels and
lat/lon/time metadata so every downstream consumer (labeling, datasets, eval) has the
fields it expects. The same events are cached to ``paths.cache`` so the costly pass is
run once and reused (datasource.md section 3; training_method.md section 6).

Public surface (interface contract):
    download_sevir_subset(cfg) -> List[str]
    load_event(path_or_idx, channels) -> Dict[str, np.ndarray]   # arrays [T, H, W]
    iter_events(cfg) -> Iterator[dict]
    SyntheticSEVIR(cfg)
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from ..utils.config import Config

# ---- optional heavy deps (datasource.md section 1): real SEVIR I/O ----------
try:  # pragma: no cover - exercised only when the bucket is reachable
    import s3fs  # type: ignore

    _HAS_S3FS = True
except Exception:  # pragma: no cover
    s3fs = None  # type: ignore
    _HAS_S3FS = False

try:  # pragma: no cover
    import h5py  # type: ignore

    _HAS_H5PY = True
except Exception:  # pragma: no cover
    h5py = None  # type: ignore
    _HAS_H5PY = False

# Channel name -> SEVIR HDF5 image type. We only pull VIL + IR + GLM
# (datasource.md section 3: "Download VIL + IR + GLM only").
SEVIR_IMG_TYPES: Dict[str, str] = {
    "vil": "vil",
    "ir069": "ir069",
    "ir107": "ir107",
    "glm": "lght",
}

DEFAULT_CHANNELS: List[str] = ["vil", "ir069", "ir107", "glm"]


# ---------------------------------------------------------------------------
# Cache layout helpers (datasource.md section 3 / training_method.md section 6)
# ---------------------------------------------------------------------------
def _cache_root(cfg: Config) -> str:
    return str(cfg.get_path("paths.cache", "./artifacts/cache"))


def _dataset_tag(cfg: Config) -> str:
    """Cache-namespace suffix derived from ``data.dataset``.

    SEVIR/synthetic share the historical un-suffixed dirs (``events`` / ``asg``) so
    existing caches and tests are untouched. The OOD sets (nexrad, mrms) get their own
    namespaced dirs (``events_nexrad`` / ``asg_mrms`` ...) so a SEVIR run and an OOD run
    never mix events or labels — the whole pipeline (labeling, datasets, eval) routes
    automatically because it all goes through :func:`events_dir` / :func:`asg_dir`.
    """
    ds = str(cfg.get_path("data.dataset", "sevir")).lower()
    return "" if ds in ("sevir", "synthetic", "synth") else f"_{ds}"


def events_dir(cfg: Config) -> str:
    """Directory holding one ``<event_id>.npz`` per cached event (dataset-namespaced)."""
    d = os.path.join(_cache_root(cfg), "events" + _dataset_tag(cfg))
    os.makedirs(d, exist_ok=True)
    return d


def asg_dir(cfg: Config) -> str:
    """Directory the labeling pass writes per-event ASG json into (dataset-namespaced)."""
    d = os.path.join(_cache_root(cfg), "asg" + _dataset_tag(cfg))
    os.makedirs(d, exist_ok=True)
    return d


def _manifest_path(cfg: Config) -> str:
    return os.path.join(events_dir(cfg), "manifest.json")


def _read_manifest(cfg: Config) -> List[str]:
    p = _manifest_path(cfg)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return list(json.load(f))
        except Exception:
            return []
    return []


def _write_manifest(cfg: Config, ids: Sequence[str]) -> None:
    with open(_manifest_path(cfg), "w", encoding="utf-8") as f:
        json.dump(list(ids), f, indent=2)


def _event_npz_path(cfg: Config, event_id: str) -> str:
    return os.path.join(events_dir(cfg), f"{event_id}.npz")


# ---------------------------------------------------------------------------
# SyntheticSEVIR — deterministic, dependency-free VIL generator
# ---------------------------------------------------------------------------
class SyntheticSEVIR:
    """Deterministic moving-Gaussian-blob VIL sequences (datasource.md section 2).

    Each "event" is a sequence of ``T = in_frames + out_frames`` VIL frames containing
    a small number of convective cells that translate at a constant velocity and
    grow/decay over the window, so classical motion/tracking/regime labeling recovers a
    meaningful ASG. IR069/IR107 are synthesized as smooth anti-correlated brightness
    fields and GLM as a sparse lightning proxy near intense cells. All randomness is
    seeded from ``cfg.seed`` and the event index, so the dataset is fully reproducible
    and requires no download or GPU.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.seed = int(cfg.get_path("seed", 1234))
        self.grid = int(cfg.get_path("data.grid", 384))
        self.in_frames = int(cfg.get_path("data.in_frames", 13))
        self.out_frames = int(cfg.get_path("data.out_frames", 36))
        self.n_events = int(cfg.get_path("data.n_train_events", 2500))
        self.vil_clip = list(cfg.get_path("data.vil_clip", [0.0, 255.0]))
        self.km_per_pixel = float(cfg.get_path("data.km_per_pixel", 1.0))
        self.minutes_per_frame = int(cfg.get_path("data.minutes_per_frame", 5))

    # -- public -------------------------------------------------------------
    @property
    def n_frames(self) -> int:
        return self.in_frames + self.out_frames

    def event_ids(self) -> List[str]:
        return [f"synth_{i:05d}" for i in range(self.n_events)]

    def generate(self, index: int) -> Dict[str, np.ndarray]:
        """Generate the full multi-channel event at ``index`` as arrays [T, H, W]."""
        rng = np.random.default_rng(self.seed + 1000 + index)
        H = W = self.grid
        T = self.n_frames
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)

        n_cells = int(rng.integers(1, 5))
        vil = np.zeros((T, H, W), dtype=np.float32)

        for _ in range(n_cells):
            # initial centroid biased toward interior so cells stay in-domain
            cy0 = float(rng.uniform(0.2 * H, 0.8 * H))
            cx0 = float(rng.uniform(0.2 * W, 0.8 * W))
            # velocity in pixels/frame (a few px/frame ~ realistic storm motion)
            vy = float(rng.uniform(-2.5, 2.5))
            vx = float(rng.uniform(-2.5, 2.5))
            sigma0 = float(rng.uniform(6.0, 18.0))
            peak0 = float(rng.uniform(40.0, 200.0))
            # growth/decay: linear amplitude ramp over the window
            growth_rate = float(rng.uniform(-0.6, 0.9))  # fraction of peak per full window
            sigma_rate = float(rng.uniform(-0.2, 0.4))

            for t in range(T):
                frac = t / max(T - 1, 1)
                cy = cy0 + vy * t
                cx = cx0 + vx * t
                amp = max(peak0 * (1.0 + growth_rate * frac), 0.0)
                sigma = max(sigma0 * (1.0 + sigma_rate * frac), 2.0)
                blob = amp * np.exp(
                    -(((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma ** 2))
                )
                vil[t] += blob.astype(np.float32)

        # mild correlated noise for texture (seeded), then clip to the VIL range
        noise = rng.normal(0.0, 1.5, size=(T, H, W)).astype(np.float32)
        vil = vil + noise
        vil = np.clip(vil, self.vil_clip[0], self.vil_clip[1]).astype(np.float32)

        # IR brightness temperatures: anti-correlated with VIL (cold tops over storms)
        ir_base = 280.0
        ir069 = (ir_base - 0.25 * vil).astype(np.float32)
        ir107 = (ir_base - 0.20 * vil + 2.0).astype(np.float32)

        # GLM lightning proxy: sparse flashes where VIL is intense
        glm = np.zeros((T, H, W), dtype=np.float32)
        thr = 0.7 * float(self.vil_clip[1])
        mask = vil > thr
        glm[mask] = rng.uniform(0.0, 1.0, size=int(mask.sum())).astype(np.float32)

        # synthetic geolocation/time so context co-location has something to slice
        lat = float(35.0 + rng.uniform(-5.0, 5.0))
        lon = float(-97.0 + rng.uniform(-8.0, 8.0))
        # deterministic timestamp in the warm season (convective)
        hour = int(18 + (index % 6))
        time_iso = f"2019-06-{1 + (index % 28):02d}T{hour % 24:02d}:00:00Z"

        return {
            "vil": vil,
            "ir069": ir069,
            "ir107": ir107,
            "glm": glm,
            "lat": np.float32(lat),
            "lon": np.float32(lon),
            "time": time_iso,
            "event_id": f"synth_{index:05d}",
        }

    def iter_events(self) -> Iterator[Dict[str, np.ndarray]]:
        for i in range(self.n_events):
            yield self.generate(i)


# ---------------------------------------------------------------------------
# Caching: write synthetic (or downloaded) events to npz under paths.cache
# ---------------------------------------------------------------------------
def _save_event_npz(path: str, event: Dict[str, np.ndarray]) -> None:
    payload: Dict[str, np.ndarray] = {}
    meta: Dict[str, object] = {}
    for k, v in event.items():
        if isinstance(v, np.ndarray) and v.ndim == 3:
            payload[k] = v.astype(np.float32)
        else:
            meta[k] = v.item() if isinstance(v, np.generic) else v
    np.savez_compressed(path, _meta=json.dumps(meta), **payload)


def _load_event_npz(path: str, channels: Sequence[str]) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["_meta"])) if "_meta" in z.files else {}
        keys = [k for k in z.files if k != "_meta"]
        shape = None
        for k in keys:
            shape = z[k].shape
            break
        out: Dict[str, np.ndarray] = {}
        for ch in channels:
            if ch in z.files:
                out[ch] = z[ch].astype(np.float32)
            elif shape is not None:
                out[ch] = np.zeros(shape, dtype=np.float32)  # tolerate missing channel
    out.update(meta)  # lat/lon/time/event_id
    out["event_id"] = meta.get("event_id", os.path.splitext(os.path.basename(path))[0])
    return out


def build_synthetic_cache(cfg: Config, limit: Optional[int] = None) -> List[str]:
    """Materialize SyntheticSEVIR events to ``paths.cache/events`` (idempotent)."""
    synth = SyntheticSEVIR(cfg)
    n = synth.n_events if limit is None else min(limit, synth.n_events)
    ids: List[str] = []
    for i in range(n):
        ev = synth.generate(i)
        eid = str(ev["event_id"])
        path = _event_npz_path(cfg, eid)
        if not os.path.exists(path):
            _save_event_npz(path, ev)
        ids.append(eid)
    _write_manifest(cfg, ids)
    return ids


# ---------------------------------------------------------------------------
# Real SEVIR subset download (best-effort; falls back to synthetic)
# ---------------------------------------------------------------------------
def download_sevir_subset(cfg: Config) -> List[str]:
    """Download a curated SEVIR subset from ``s3://sevir`` to the cache.

    Pulls VIL + IR + GLM only for up to ``data.n_train_events`` rainy events
    (datasource.md section 3). If ``s3fs``/``h5py`` are missing or the bucket is
    unreachable, build :class:`SyntheticSEVIR` instead and return its event ids so the
    pipeline still runs end-to-end with no download (interface contract).

    Returns the list of cached event ids.
    """
    limit = int(cfg.get_path("data.n_train_events", 2500))
    dataset = str(cfg.get_path("data.dataset", "sevir")).lower()
    require_real = bool(cfg.get_path("data.require_real", False))

    # Explicit synthetic request — always honoured, clearly logged.
    if dataset in ("synthetic", "synth"):
        print("[sevir] data.dataset=synthetic -> SyntheticSEVIR (NOT real data)")
        return build_synthetic_cache(cfg, limit=limit)

    # Real SEVIR requested. NEVER silently train on synthetic when real was asked for and
    # data.require_real=True — that would waste a paid GPU session. Fail loudly instead.
    if not (_HAS_S3FS and _HAS_H5PY):
        msg = "[sevir] real SEVIR requested but s3fs/h5py are not installed"
        if require_real:
            raise RuntimeError(msg + "; `pip install s3fs h5py` (data.require_real=True).")
        print(msg + " -> falling back to SyntheticSEVIR (set data.require_real=true to forbid this)")
        return build_synthetic_cache(cfg, limit=limit)
    try:  # pragma: no cover - requires network + anon S3
        ids = _download_real_sevir(cfg, limit=limit)
        if not ids:
            raise RuntimeError("real SEVIR returned 0 events (catalog/schema mismatch?)")
        print(f"[sevir] REAL SEVIR: cached {len(ids)} events -> {events_dir(cfg)}")
        return ids
    except Exception as e:
        if require_real:
            raise RuntimeError(
                f"[sevir] REAL SEVIR load FAILED: {e}. Fix the data source before training "
                f"(data.require_real=True forbids the synthetic fallback)."
            ) from e
        print(f"[sevir] REAL SEVIR failed ({e}) -> SyntheticSEVIR fallback")
        return build_synthetic_cache(cfg, limit=limit)


def _resize_nn(arr: np.ndarray, H: int, W: int) -> np.ndarray:
    """Nearest-neighbour resize of a [T, h, w] stack to [T, H, W] (no scipy dependency)."""
    if arr.shape[-2:] == (H, W):
        return arr
    h, w = arr.shape[-2:]
    yi = np.linspace(0, h - 1, H).round().astype(int)
    xi = np.linspace(0, w - 1, W).round().astype(int)
    return arr[:, yi][:, :, xi]


def _download_real_sevir(cfg: Config, limit: int) -> List[str]:  # pragma: no cover
    """Real SEVIR pull using the documented catalog/HDF5 schema (veillette2020sevir).

    CATALOG.csv maps (id, img_type) -> (file_name, file_index) plus geolocation and time.
    Each HDF5 file stores a dataset named by img_type with shape [N, h, w, T]; this event
    is row ``file_index``. VIL is 384x384; IR069/IR107 are 192x192 (upsampled to the grid);
    GLM 'lght' is variable-length flash-list data (not a grid) and is left as zeros here
    (a documented limitation; the model tolerates a zero channel). Each event is logged.
    """
    fs = s3fs.S3FileSystem(anon=True)  # type: ignore[union-attr]
    raw_root = str(cfg.get_path("paths.sevir_raw", "./artifacts/sevir_raw"))
    os.makedirs(raw_root, exist_ok=True)
    grid = int(cfg.get_path("data.grid", 384))
    channels = list(cfg.get_path("data.channels", DEFAULT_CHANNELS))

    catalog_local = os.path.join(raw_root, "CATALOG.csv")
    if not os.path.exists(catalog_local):
        fs.get("sevir/CATALOG.csv", catalog_local)

    import csv

    # group catalog rows by event id: img_type -> (file_name, file_index); plus geo/time
    by_event: Dict[str, Dict[str, object]] = {}
    with open(catalog_local, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            eid = row.get("id") or ""
            it = row.get("img_type", "")
            fname = row.get("file_name", "")
            if not eid or not it or not fname:
                continue
            rec = by_event.setdefault(eid, {"_geo": None, "_time": None})
            try:
                fidx = int(float(row.get("file_index", 0)))
            except (TypeError, ValueError):
                fidx = 0
            rec[it] = (fname, fidx)
            if not rec["_time"]:
                rec["_time"] = row.get("time_utc") or row.get("time") or ""
            if rec["_geo"] is None:
                try:
                    rec["_geo"] = (0.5 * (float(row["llcrnrlat"]) + float(row["urcrnrlat"])),
                                   0.5 * (float(row["llcrnrlon"]) + float(row["urcrnrlon"])))
                except (KeyError, TypeError, ValueError):
                    rec["_geo"] = None

    warned_lght = False
    ids: List[str] = []
    for eid, rec in by_event.items():
        if len(ids) >= limit:
            break
        if SEVIR_IMG_TYPES["vil"] not in rec:        # need VIL for this event
            continue
        event: Dict[str, np.ndarray] = {}
        ref_shape = None
        for ch in channels:
            it = SEVIR_IMG_TYPES.get(ch, ch)
            if it == "lght":
                if not warned_lght:
                    print("[sevir] note: GLM 'lght' is flash-list data; stored as zeros (model tolerates it)")
                    warned_lght = True
                continue
            ent = rec.get(it)
            if not isinstance(ent, tuple):
                continue
            fname, fidx = ent
            local = os.path.join(raw_root, os.path.basename(str(fname)))
            if not os.path.exists(local):
                # SEVIR HDF5 files live under the bucket's data/ prefix, but CATALOG.csv's
                # file_name column omits it (e.g. "vil/2019/SEVIR_VIL_*.h5"). Prepend data/
                # so the key resolves (verified live: s3://sevir/data/vil/2019/...).
                key = str(fname) if str(fname).startswith("data/") else f"data/{fname}"
                fs.get(f"sevir/{key}", local)
            with h5py.File(local, "r") as hf:  # type: ignore[union-attr]
                if it not in hf:
                    continue
                arr = np.asarray(hf[it][int(fidx)])                 # [h, w, T]
                if arr.ndim != 3:
                    continue
                arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)  # -> [T, h, w]
                arr = _resize_nn(arr, grid, grid)
                event[ch] = arr
                ref_shape = arr.shape
        if "vil" not in event:
            continue
        for ch in channels:                          # zero-fill channels we couldn't load
            if ch not in event and ref_shape is not None:
                event[ch] = np.zeros(ref_shape, dtype=np.float32)
        event["event_id"] = eid
        if rec.get("_geo"):
            event["lat"] = np.float32(rec["_geo"][0])
            event["lon"] = np.float32(rec["_geo"][1])
        if rec.get("_time"):
            event["time"] = str(rec["_time"])
        _save_event_npz(_event_npz_path(cfg, eid), event)
        ids.append(eid)
        if len(ids) % 50 == 0:
            print(f"[sevir]   cached {len(ids)} real events...")

    _write_manifest(cfg, ids)
    return ids


# ---------------------------------------------------------------------------
# Loading single events
# ---------------------------------------------------------------------------
def load_event(
    path_or_idx: Union[str, int],
    channels: Optional[Sequence[str]] = None,
    cfg: Optional[Config] = None,
) -> Dict[str, np.ndarray]:
    """Load one event as a dict of arrays [T, H, W] (interface contract).

    ``path_or_idx`` may be a path to a cached ``.npz`` file, or an integer index into
    the synthetic generator (requires ``cfg``). Missing channels are returned as zeros.
    """
    if channels is None:
        channels = DEFAULT_CHANNELS
    channels = list(channels)

    if isinstance(path_or_idx, (int, np.integer)):
        if cfg is None:
            raise ValueError("load_event(idx) requires cfg to drive SyntheticSEVIR")
        ev = SyntheticSEVIR(cfg).generate(int(path_or_idx))
        # ensure requested channels exist (zeros if absent)
        ref = ev["vil"]
        out: Dict[str, np.ndarray] = {}
        for ch in channels:
            v = ev.get(ch)
            out[ch] = np.asarray(v, dtype=np.float32) if isinstance(v, np.ndarray) else np.zeros_like(ref)
        for k in ("lat", "lon", "time", "event_id"):
            if k in ev:
                out[k] = ev[k]
        return out

    return _load_event_npz(str(path_or_idx), channels)


# ---------------------------------------------------------------------------
# Iterating events (cache-first, synthetic fallback)
# ---------------------------------------------------------------------------
def iter_events(cfg: Config) -> Iterator[dict]:
    """Yield events (``vil`` + optional channels + lat/lon/time) for labeling.

    Prefers cached npz events under ``paths.cache/events``; if none exist, streams
    deterministically from :class:`SyntheticSEVIR` without writing to disk (so a
    notebook smoke run needs no prior download step). datasource.md section 3.
    """
    channels = list(cfg.get_path("data.channels", DEFAULT_CHANNELS))
    ids = _read_manifest(cfg)
    cached = [eid for eid in ids if os.path.exists(_event_npz_path(cfg, eid))]

    if cached:
        for eid in cached:
            yield _load_event_npz(_event_npz_path(cfg, eid), channels)
        return

    # No cache present -> stream synthetic events directly.
    synth = SyntheticSEVIR(cfg)
    for ev in synth.iter_events():
        out: Dict[str, np.ndarray] = {}
        ref = ev["vil"]
        for ch in channels:
            v = ev.get(ch)
            out[ch] = np.asarray(v, dtype=np.float32) if isinstance(v, np.ndarray) else np.zeros_like(ref)
        for k in ("lat", "lon", "time", "event_id"):
            if k in ev:
                out[k] = ev[k]
        yield out
