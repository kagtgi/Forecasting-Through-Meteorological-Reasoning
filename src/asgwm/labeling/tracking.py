"""Cell identification + tracking for ASG auto-labeling (datasource.md section 2).

Step 1 of the auto-label pipeline: identify storm cells per frame (connected
components / watershed on the VIL field) and link them across frames into object
tracks by nearest-centroid association. Each track yields one StormObject in the
ASG. Prefer skimage watershed for splitting merged cells; fall back to
scipy.ndimage.label; fall back again to a pure-numpy flood fill so the pipeline
runs with no optional deps.

Conventions: frames are [T, H, W]; a segmentation is an int label image [H, W]
with 0 = background. Centroids are (cy, cx) in grid coords; area is pixel count
(callers convert to km^2 with dx_km). Tracks are linked greedily by centroid
proximity within a gating radius.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

# Optional segmentation backends.
try:  # pragma: no cover
    from scipy import ndimage as _ndimage  # type: ignore

    _HAS_NDIMAGE = True
except Exception:  # pragma: no cover
    _ndimage = None
    _HAS_NDIMAGE = False

try:  # pragma: no cover
    from skimage.feature import peak_local_max as _peak_local_max  # type: ignore
    from skimage.segmentation import watershed as _watershed  # type: ignore

    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover
    _peak_local_max = None
    _watershed = None
    _HAS_SKIMAGE = False


# ---------------------------------------------------------------------------
# Per-frame segmentation
# ---------------------------------------------------------------------------
def _numpy_label(mask: np.ndarray) -> np.ndarray:
    """Pure-numpy 4-connected connected-component labeling (BFS flood fill)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    cur = 0
    stack: List[tuple[int, int]] = []
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            cur += 1
            stack.append((sy, sx))
            labels[sy, sx] = cur
            while stack:
                y, x = stack.pop()
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = cur
                        stack.append((ny, nx))
    return labels


def segment_cells(frame: np.ndarray, threshold: float) -> np.ndarray:
    """Segment storm cells in a single frame above `threshold`.

    Args:
        frame: [H, W] intensity field (e.g. VIL).
        threshold: intensity threshold separating storm from background.

    Returns:
        int label image [H, W]; 0 = background, 1..K = cells. Uses skimage
        watershed (splits touching cells at intensity peaks) when available,
        else scipy.ndimage.label, else a numpy flood fill.
    """
    field = np.asarray(frame, dtype=np.float32)
    mask = field >= float(threshold)
    if not mask.any():
        return np.zeros_like(field, dtype=np.int32)

    if _HAS_SKIMAGE and _HAS_NDIMAGE:
        try:  # pragma: no cover - requires skimage at runtime
            # Markers at local intensity maxima; watershed on the negated field so
            # basins grow downhill from the peaks, splitting merged storms.
            coords = _peak_local_max(
                field, labels=mask.astype(int), min_distance=3, exclude_border=False
            )
            markers = np.zeros_like(field, dtype=np.int32)
            for i, (yy, xx) in enumerate(coords, start=1):
                markers[yy, xx] = i
            if markers.max() == 0:
                markers, _ = _ndimage.label(mask)
            labels = _watershed(-field, markers, mask=mask)
            return labels.astype(np.int32)
        except Exception:
            pass

    if _HAS_NDIMAGE:
        try:  # pragma: no cover
            labels, _ = _ndimage.label(mask)
            return labels.astype(np.int32)
        except Exception:
            pass

    return _numpy_label(mask)


def _region_props(labels: np.ndarray, field: np.ndarray) -> List[Dict[str, float]]:
    """Centroid / area / peak per labeled region."""
    props: List[Dict[str, float]] = []
    ids = np.unique(labels)
    for lab in ids:
        if lab == 0:
            continue
        ys, xs = np.where(labels == lab)
        if ys.size == 0:
            continue
        vals = field[ys, xs]
        props.append(
            {
                "label": int(lab),
                "cy": float(ys.mean()),
                "cx": float(xs.mean()),
                "area": float(ys.size),
                "peak": float(vals.max()),
            }
        )
    return props


# ---------------------------------------------------------------------------
# Multi-frame tracking (greedy nearest-centroid association)
# ---------------------------------------------------------------------------
def track_cells(
    frames: np.ndarray,
    threshold: float,
    gate_radius: Optional[float] = None,
    min_area: float = 4.0,
) -> List[dict]:
    """Link per-frame cells into tracks across a frame stack.

    Args:
        frames: [T, H, W] intensity history.
        threshold: segmentation threshold.
        gate_radius: max centroid distance (px) to associate frame t->t+1.
            Defaults to ~6% of the domain diagonal.
        min_area: drop cells smaller than this (px) before association.

    Returns:
        List of track dicts, each:
            {'id': int, 'frames': [{'t', 'cy', 'cx', 'area', 'peak'}, ...]}
        sorted by descending peak intensity (most intense first).
    """
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    t, h, w = arr.shape
    if gate_radius is None:
        gate_radius = 0.06 * float(np.hypot(h, w))

    # Per-frame detections.
    detections: List[List[Dict[str, float]]] = []
    for ti in range(t):
        labels = segment_cells(arr[ti], threshold)
        props = [p for p in _region_props(labels, arr[ti]) if p["area"] >= min_area]
        detections.append(props)

    tracks: List[dict] = []
    active: List[dict] = []  # tracks open for extension (each has last cy/cx)
    next_id = 0

    for ti in range(t):
        dets = detections[ti]
        used = [False] * len(dets)

        # Greedy association: for each active track, claim the nearest detection.
        ordered = sorted(
            range(len(active)),
            key=lambda i: active[i]["frames"][-1]["peak"],
            reverse=True,
        )
        still_active: List[dict] = []
        for ai in ordered:
            trk = active[ai]
            last = trk["frames"][-1]
            best_j = -1
            best_d = gate_radius
            for j, d in enumerate(dets):
                if used[j]:
                    continue
                dist = float(np.hypot(d["cy"] - last["cy"], d["cx"] - last["cx"]))
                if dist < best_d:
                    best_d = dist
                    best_j = j
            if best_j >= 0:
                used[best_j] = True
                d = dets[best_j]
                trk["frames"].append(
                    {
                        "t": ti,
                        "cy": d["cy"],
                        "cx": d["cx"],
                        "area": d["area"],
                        "peak": d["peak"],
                    }
                )
                still_active.append(trk)
            else:
                # Track ends; finalize it.
                tracks.append(trk)
        # Maintain ordering stability by index for tracks not in `ordered` loop.
        active = still_active

        # Unmatched detections start new tracks.
        for j, d in enumerate(dets):
            if used[j]:
                continue
            trk = {
                "id": next_id,
                "frames": [
                    {
                        "t": ti,
                        "cy": d["cy"],
                        "cx": d["cx"],
                        "area": d["area"],
                        "peak": d["peak"],
                    }
                ],
            }
            next_id += 1
            active.append(trk)

    tracks.extend(active)

    # Sort by peak intensity over the track (most intense first) for N_MAX capping.
    tracks.sort(key=lambda tr: max(f["peak"] for f in tr["frames"]), reverse=True)
    # Re-assign stable ids in sorted order.
    for new_id, tr in enumerate(tracks):
        tr["id"] = new_id
    return tracks


def track_motion(track: dict, dt_steps: float = 1.0) -> tuple[float, float]:
    """Mean per-step centroid velocity (vy, vx) in px/step over a track."""
    fr = track["frames"]
    if len(fr) < 2:
        return 0.0, 0.0
    dys = []
    dxs = []
    for a, b in zip(fr[:-1], fr[1:]):
        span = max(1, b["t"] - a["t"])
        dys.append((b["cy"] - a["cy"]) / span)
        dxs.append((b["cx"] - a["cx"]) / span)
    return float(np.mean(dys)) * dt_steps, float(np.mean(dxs)) * dt_steps
