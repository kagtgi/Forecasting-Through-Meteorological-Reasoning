#!/usr/bin/env python
"""Preflight: confirm SEVIR / NEXRAD / MRMS are downloadable BEFORE any heavy work.

Uses ONLY the Python standard library (urllib) over anonymous HTTPS — no boto3, pyart,
cfgrib, h5py, or AWS account needed — so it runs instantly at the top of train.ipynb /
eval.ipynb and tells you, per dataset, GO / NO-GO.

For the OOD sets it does more than ping the bucket: it lists the real object keys across every
UTC day the requested window spans and runs the loaders' OWN time-selection
(``ood_resample.select_nearest`` with the loaders' day-spanning + tolerance=dt). So a PASS
means each configured case actually has gap-free 5-min coverage for the full T-frame window and
WILL yield an event once pyart/cfgrib decode it — not merely that the bucket exists.

Usage:
    python datasets/check_access.py                 # checks all three with config defaults
    python datasets/check_access.py --datasets nexrad mrms
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import urllib.error
import urllib.request

# datasets/check_access.py -> repo root is the parent dir; package lives in src/.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from asgwm.utils.config import load_config            # noqa: E402
from asgwm.data import nexrad as nexrad_mod           # noqa: E402
from asgwm.data import mrms as mrms_mod               # noqa: E402
from asgwm.data import ood_resample as ood            # noqa: E402

_TIMEOUT = 30


def _list_keys(bucket: str, prefix: str, max_keys: int = 1000) -> list:
    """Anonymous S3 REST v2 listing -> list of object keys (no JS, unlike index.html)."""
    url = f"https://{bucket}.s3.amazonaws.com/?list-type=2&prefix={prefix}&max-keys={max_keys}"
    with urllib.request.urlopen(url, timeout=_TIMEOUT) as r:
        body = r.read().decode("utf-8", "replace")
    return re.findall(r"<Key>([^<]+)</Key>", body)


def _head_ok(url: str) -> bool:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
            return 200 <= r.status < 300
    except urllib.error.URLError:
        return False


def _frames(cfg) -> tuple:
    T = int(cfg.get_path("data.in_frames", 13)) + int(cfg.get_path("data.out_frames", 36))
    dt = int(cfg.get_path("data.minutes_per_frame", 5))
    return T, dt


def check_sevir(cfg) -> bool:
    print("\n[SEVIR]  s3://sevir  (train+test)")
    ok_cat = _head_ok("https://sevir.s3.amazonaws.com/CATALOG.csv")
    print(f"  CATALOG.csv reachable : {'yes' if ok_cat else 'NO'}")
    vil = _list_keys("sevir", "data/vil/2019/", max_keys=5)
    storm = [k for k in vil if "STORMEVENTS" in k]
    print(f"  VIL files (2019)      : {len(vil)} found e.g. {os.path.basename(storm[0]) if storm else '-'}")
    ok = ok_cat and bool(vil)
    print(f"  => {'GO' if ok else 'NO-GO'}")
    return ok


def _check_ood(name, bucket, cases, prefix_fn, time_fn, T, dt) -> bool:
    print(f"\n[{name}]  s3://{bucket}  (OOD test)   T={T} frames x {dt}min")
    if not cases:
        print("  no cases configured"); return False
    covered = 0
    for c in cases:
        start = str(c.get("start", "000000")).ljust(6, "0")[:6]
        date = str(c["date"])
        label = c.get("station", f"{c.get('lat','?')},{c.get('lon','?')}")
        recs = []
        for day, off in ood.spanned_dates(date, start, T, dt):
            try:
                for k in _list_keys(bucket, prefix_fn(c, day), max_keys=2000):
                    t = time_fn(k)
                    if t is not None:
                        recs.append(ood.abs_minute(off, t))
            except urllib.error.URLError as e:
                print(f"  {date} {label}: list FAILED ({e})")
        recs.sort()
        idx = ood.select_nearest(recs, ood.start_minute(start), T, dt, tol_min=dt)
        ok = idx is not None
        covered += int(ok)
        days = "+".join(d for d, _ in ood.spanned_dates(date, start, T, dt))
        print(f"  {date} {start[:4]} {label:>14}: {len(recs):4d} files over {days}  "
              f"window {'COVERED (gap-free)' if ok else 'HAS A GAP -> case skipped'}")
    # The loader caches every covered case and only fails if ZERO events result, so the
    # dataset is GO as long as at least one configured case has gap-free coverage.
    ok = covered >= 1
    print(f"  => {'GO' if ok else 'NO-GO'}  ({covered}/{len(cases)} cases covered)")
    return ok


def check_nexrad(cfg) -> bool:
    T, dt = _frames(cfg)
    cases = cfg.get_path("data.nexrad.cases", None) or nexrad_mod.DEFAULT_CASES
    return _check_ood(
        "NEXRAD", nexrad_mod.BUCKET, cases,
        lambda c, day: f"{day[:4]}/{day[4:6]}/{day[6:8]}/{c['station']}/",
        nexrad_mod._vol_time_min, T, dt)


def check_mrms(cfg) -> bool:
    T, dt = _frames(cfg)
    cases = cfg.get_path("data.mrms.cases", None) or mrms_mod.DEFAULT_CASES
    return _check_ood(
        "MRMS", mrms_mod.BUCKET, cases,
        lambda c, day: f"{mrms_mod.PREFIX}/{day}/",
        mrms_mod._key_time_min, T, dt)


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight check: are SEVIR/NEXRAD/MRMS downloadable?")
    ap.add_argument("--config", default=os.path.join(_SRC, "configs", "default.yaml"))
    ap.add_argument("--datasets", nargs="+", default=["sevir", "nexrad", "mrms"],
                    choices=["sevir", "nexrad", "mrms"])
    ap.add_argument("--override", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.override)

    print("=" * 64)
    print("DATA ACCESS PRE-FLIGHT (anonymous HTTPS, no AWS / heavy deps)")
    print("=" * 64)
    checks = {"sevir": check_sevir, "nexrad": check_nexrad, "mrms": check_mrms}
    results = {d: checks[d](cfg) for d in args.datasets}

    print("\n" + "-" * 64)
    for d, ok in results.items():
        print(f"  {d:8s}: {'GO' if ok else 'NO-GO'}")
    n_fail = sum(1 for ok in results.values() if not ok)
    print("=" * 64)
    print("ALL GO — datasets are reachable and OOD windows are covered."
          if n_fail == 0 else f"{n_fail} dataset(s) NO-GO — see above before a paid run.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
