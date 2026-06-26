"""Unit tests for asgwm.data.ood_resample (OOD time-axis assembly).

Pure logic — no network/pyart/cfgrib — covering the two bugs the loaders had to fix:
multi-day windows (midnight crossing) and gap detection vs. irregular volume cadence.
"""
import numpy as np

from asgwm.data import ood_resample as ood


def test_start_minute():
    assert ood.start_minute("000000") == 0
    assert ood.start_minute("1200") == 720
    assert ood.start_minute("210000") == 21 * 60


def test_spanned_dates_single_day():
    # 12:00 + (25-1)*5min = +120min -> 14:00, same day
    span = ood.spanned_dates("20220322", "120000", 25, 5)
    assert span == [("20220322", 0)]


def test_spanned_dates_crosses_midnight():
    # 21:00 + (49-1)*5min = +240min -> 01:00 next day -> two UTC days
    span = ood.spanned_dates("20210510", "210000", 49, 5)
    assert span == [("20210510", 0), ("20210511", 1)]


def test_spanned_dates_multi_day():
    # a very long window spans 3 days
    span = ood.spanned_dates("20211231", "230000", 49 * 12, 5)  # ~49h
    assert [s[0] for s in span] == ["20211231", "20220101", "20220102"]
    assert [s[1] for s in span] == [0, 1, 2]


def test_abs_minute_orders_across_midnight():
    # 23:50 day0 < 00:10 day1 on the absolute axis
    assert ood.abs_minute(0, 23 * 60 + 50) < ood.abs_minute(1, 10)


def test_select_nearest_gapfree_6min_cadence():
    # volumes every ~6 min (NEXRAD-like); 5-min grid, tol = dt = 5 -> fully covered
    dt, T = 5, 25
    avail = [i * 6 for i in range(60)]  # 0,6,12,... up to 354 min
    idx = ood.select_nearest(avail, 0, T, dt, tol_min=dt)
    assert idx is not None and len(idx) == T
    # every chosen volume is within tol of its 5-min slot
    for i, j in enumerate(idx):
        assert abs(avail[j] - i * dt) <= dt


def test_select_nearest_detects_real_gap():
    dt, T = 5, 25
    # drop everything between 40 and 90 min -> a >dt hole the 5-min grid cannot fill
    avail = [t for t in range(0, 360, 5) if not (40 < t < 90)]
    assert ood.select_nearest(avail, 0, T, dt, tol_min=dt) is None


def test_select_nearest_empty():
    assert ood.select_nearest([], 0, 10, 5) is None


def test_assemble_uniform_stacks_and_gaps():
    dt, T = 5, 5
    H = W = 8
    records = [(i * dt, np.full((H, W), float(i), np.float32)) for i in range(T)]
    out = ood.assemble_uniform(records, 0, T, dt, tol_min=dt)
    assert out is not None and out.shape == (T, H, W)
    # frame i should carry value i (nearest exact match)
    assert [float(out[i].mean()) for i in range(T)] == [0.0, 1.0, 2.0, 3.0, 4.0]
    # a record set with a hole returns None
    sparse = [records[0], records[4]]
    assert ood.assemble_uniform(sparse, 0, T, dt, tol_min=dt) is None
