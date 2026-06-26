# Datasets — data acquisition & canonical representation

This folder holds the data **tooling** and the **downloaded data** for the radar-nowcasting
pipeline. The downloaded data (and the per-dataset `cache/` and `raw/` dirs) are gitignored;
only the scripts and this guide are tracked.

Three radar sources feed the same model through **one canonical representation**:

| Dataset | Role | Bucket | Split |
| --- | --- | --- | --- |
| **SEVIR** | train **+** test (primary) | `s3://sevir` | temporal (see below) |
| **NEXRAD** Level II | **OOD test only** | `s3://unidata-nexrad-level2` | none (all test) |
| **MRMS** Composite | **OOD test only** | `s3://noaa-mrms-pds` | none (all test) |

NEXRAD and MRMS are used **entirely as out-of-distribution test sets** — no training, no
train/val/test split. They exist to measure how a SEVIR-trained model generalizes to other
radar products.

The loaders live in `src/asgwm/data/{sevir,nexrad,mrms}.py`; the single normalization /
conversion module is `src/asgwm/data/normalize.py`.

---

## Canonical representation

Every event from every dataset is reduced to **one** representation:

- **Shape:** `[T, 384, 384]`, time-first, at **1 km** nominal spacing.
- **Channel:** the SEVIR **VIL byte** encoding — integers `0..254`, with `255 = missing`.
- **Model input:** `x = byte / 255`, a float32 in `[0, 1]` (the Earthformer/PreDiff
  convention). Convert back with `denormalize_vil` before thresholding.
- **CSI thresholds (raw byte scale):** `[16, 74, 133, 160, 181, 219]`. Threshold in **byte
  space**, not in kg/m² or a remapped dBZ scale, so CSI matches the published SEVIR-VIL
  baselines (Earthformer / PreDiff / DiffCast / CasCast).
- **Frame configs (SEVIR 5-min cadence):**
  - **Headline** (Earthformer-matched): **13 input → 12 output** (1 h horizon).
  - **Extended** (long-lead): **13 input → 36 output** (3 h horizon).
- **SEVIR temporal split** (Earthformer/PreDiff): `train < 2019-01-01`;
  `val = 2019-01-01 .. 2019-05-31`; `test >= 2019-06-01`.

The invariant that all three datasets share this exact shape *and* value range is verified
by `datasets/preprocess.py` (it runs `normalize.assert_canonical` on every cached event).

Key constants in `asgwm.data.normalize`: `SEVIR_CSI_THRESHOLDS=(16,74,133,160,181,219)`,
`CANON_GRID=384`, `VIL_BYTE_MISSING=255`, `DEFAULT_DZ_EFF_M=4000.0`.

### dBZ → VIL caveat (state this honestly)

NEXRAD and MRMS are radar **reflectivity in dBZ**; SEVIR is **VIL**. To run a SEVIR-trained
model on them, dBZ is bridged into SEVIR VIL byte space (`normalize.dbz_to_vil_byte`):

```
Z = 10^(dBZ/10)                          # mm^6 / m^3   (dBZ capped at 56)
M = 3.44e-3 * Z^(4/7)                     # g / m^3
VIL = M * dz_eff / 1000                   # kg / m^2     (dz_eff default 4000 m)
byte = invert_SEVIR_decode(VIL)           # -> 0..254
```

`dz_eff` is configured per dataset (`data.nexrad.dz_eff_m` / `data.mrms.dz_eff_m`).
**Composite reflectivity (a column-max) and VIL (a vertical integral) are physically
different quantities.** This bridge is therefore an *approximation* governed by `dz_eff`, so
OOD numbers measure generalization **under an imperfect variable bridge**, not a perfect
like-for-like. Report `dz_eff` alongside any OOD result.

---

## Pip extras (per dataset)

```bash
pip install s3fs h5py                      # SEVIR
pip install arm-pyart boto3               # NEXRAD Level II
pip install boto3 xarray cfgrib eccodes   # MRMS
```

All buckets are accessed **anonymously** (no AWS credentials needed):
`--no-sign-request` for the AWS CLI; `anon=True` (s3fs) / `botocore UNSIGNED` (boto3) in code.

---

## SEVIR (primary — train + test)

- **Bucket:** `s3://sevir` (region `us-west-2`, anonymous).
- **VIL files:** `s3://sevir/data/vil/<YEAR>/SEVIR_VIL_STORMEVENTS_<YEAR>_<MMDD>_<MMDD>.h5`
  (1–8 GB each).
- **Catalog:** `s3://sevir/CATALOG.csv` — filter `img_type == 'vil'`; use `file_name` +
  `file_index` to locate each event.
- **Format:** HDF5, dataset key `'vil'`. In-file array is
  `[N_events, 384, 384, 49]` = **TIME-LAST**; one event is `arr[file_index]`.
- **Variable / units / grid / cadence:** VIL, byte-encoded; 384×384 @ 1 km (no spatial
  resampling needed); 5-min cadence, 49 frames per event.
- **Value range / sentinels:** `uint8` `0..254` valid, `255 = missing` (mask before stats).
  Official decode is the 3-piece SEVIR map, handled in `asgwm.data.normalize`.
- **To canonical:** transpose time-last → time-first `[T,384,384]`, keep the VIL byte
  channel, `x = byte/255` for the model.

**Download (AWS CLI):**
```bash
aws s3 cp s3://sevir/CATALOG.csv ./datasets/sevir/raw/ --no-sign-request
aws s3 cp s3://sevir/data/vil/2019/ ./datasets/sevir/raw/ --recursive --no-sign-request
```
**Download (python CLI):**
```bash
python datasets/download_sevir.py --n-events 64
python datasets/download_sevir.py --n-events 2500 --require-real   # forbid synthetic fallback
```
If `s3fs`/`h5py` or the network are unavailable, the SEVIR downloader falls back to a
deterministic **SyntheticSEVIR** generator (unless `--require-real`), so the whole pipeline
runs offline.

---

## NEXRAD Level II (OOD test)

- **Bucket:** `s3://unidata-nexrad-level2` (region `us-east-1`, anonymous via botocore
  `UNSIGNED`). The legacy `s3://noaa-nexrad-level2` was **deprecated 2025-09-01 — do not
  use it.**
- **Key layout:** `{YYYY}/{MM}/{DD}/{STATION}/{STATION}{YYYYMMDD}_{HHMMSS}_V06`
  (e.g. `2022/03/22/KHGX/KHGX20220322_120125_V06`).
- **Format / variable / grid / cadence:** Level II volume scans, **polar** (azimuth / range
  / elevation) reflectivity in dBZ; irregular ~4–6 min cadence; single radar.
- **To canonical:** read with `pyart.io.read_nexrad_archive`, QC with a `GateFilter`
  (drop `RhoHV < 0.8`), grid to a 384×384 1 km tile with `pyart.map.grid_from_radars`
  (limits in **meters**, axis order `(z, y, x)`), composite = `np.nanmax` over `z` → dBZ,
  resample to a uniform 5-min axis, then `dbz_to_vil_byte` → canonical VIL byte.
- **Sentinels:** masked / below-threshold gates dropped by the gate filter before gridding.

**Download (AWS CLI — inspect a station/day):**
```bash
aws s3 ls s3://unidata-nexrad-level2/2022/03/22/KHGX/ --no-sign-request
aws s3 cp s3://unidata-nexrad-level2/2022/03/22/KHGX/ ./datasets/nexrad/raw/ \
    --recursive --no-sign-request
```
**Download (python CLI):**
```bash
python datasets/download_nexrad.py                 # built-in cases, [] if deps/network absent
python datasets/download_nexrad.py --require-real  # raise instead of returning []
```
OOD **cases** (station / date / start window) come from `data.nexrad.cases` in
`src/configs/default.yaml` (`null` → built-in severe-weather cases). No synthetic fallback —
an OOD test must use real data.

---

## MRMS MergedReflectivityQCComposite (OOD test)

- **Bucket:** `s3://noaa-mrms-pds` (region `us-east-1`, anonymous). **Archive starts
  2020-10-14** — pick cases on/after that date.
- **Key layout:** prefix
  `CONUS/MergedReflectivityQCComposite_00.50/<YYYYMMDD>/`; objects
  `MRMS_MergedReflectivityQCComposite_00.50_<YYYYMMDD>-<HHMMSS>.grib2.gz`
  (gzipped GRIB2, ~1.4 MB each).
- **Format / variable / grid / cadence:** already-gridded CONUS composite reflectivity
  (dBZ) at ~1 km lat/lon (3500×7000, 0.01°, latitude **descending** = north-up); ~2 min
  cadence.
- **Value range / sentinels:** dBZ; `-99 = missing`, `-999 = no coverage` (mask **both**).
- **To canonical:** `gzip.decompress` + `xarray` `engine='cfgrib'` (or pygrib), mask
  sentinels, crop a 384×384 tile around a center `(lat, lon)`, resample to a uniform 5-min
  axis, then `dbz_to_vil_byte` → canonical VIL byte.

**Download (AWS CLI — inspect a day):**
```bash
aws s3 ls s3://noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/20210510/ \
    --no-sign-request
aws s3 cp s3://noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/20210510/ \
    ./datasets/mrms/raw/ --recursive --no-sign-request
```
**Download (python CLI):**
```bash
python datasets/download_mrms.py                 # built-in cases, [] if deps/network absent
python datasets/download_mrms.py --require-real  # raise instead of returning []
```
OOD **cases** (date / start / center lat,lon) come from `data.mrms.cases` in
`src/configs/default.yaml` (`null` → built-in cases). No synthetic fallback.

---

## Verify the canonical representation

After downloading, prove every cached event is the identical canonical stack:

```bash
python datasets/preprocess.py --dataset sevir
python datasets/preprocess.py --dataset nexrad
python datasets/preprocess.py --dataset mrms
```

This runs `normalize.assert_canonical` on each event's `vil` array and prints a summary
(n events, shape, byte min/max, % missing) — confirming identical shape **and** value range
across all three datasets.

---

## Access preflight (run this first)

Before any heavy download, confirm every bucket is reachable **and** that each OOD case has
gap-free coverage for the requested window — using only the Python stdlib over anonymous
HTTPS (no boto3 / pyart / cfgrib / AWS account):

```bash
python datasets/check_access.py                          # all three, config defaults
python datasets/check_access.py --override data.out_frames=12   # OOD at the 120-min window
python datasets/check_access.py --datasets nexrad mrms   # just the OOD sets
```

It lists the real object keys across every UTC day the window spans and runs the loaders' own
time-selection (`ood_resample.select_nearest`, day-spanning + tolerance = `dt`), so a **GO**
means the case will actually yield an event once decoded — not merely that the bucket exists.
A dataset is GO when **≥1** configured case is covered (the loader caches every covered case and
skips gappy ones). `train.ipynb` / `eval.ipynb` call this automatically before downloading.

> Coverage note: single-radar NEXRAD VCP switches can open >5-min gaps over a long window, so
> OOD downloads/evals use the **120-min headline horizon** (`out_frames=12`, T=25), which is the
> standard cross-dataset comparison window and is gap-free for the built-in cases. The 245-min
> long-lead window (`out_frames=36`) is used for SEVIR in-distribution analysis.

---

## Layout

```
datasets/
  README.md            # this guide
  check_access.py      # no-deps HTTPS preflight: are all 3 reachable + OOD windows covered?
  download_sevir.py    # SEVIR CLI  -> datasets/sevir/cache (+ raw)
  download_nexrad.py   # NEXRAD CLI -> datasets/nexrad/cache
  download_mrms.py     # MRMS CLI   -> datasets/mrms/cache
  preprocess.py        # verify canonical [T,384,384] VIL byte across datasets
  sevir/ nexrad/ mrms/ # downloaded data + caches (gitignored; .gitkeep tracked)
```
