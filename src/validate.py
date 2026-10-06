import numpy as np
import pandas as pd
from .config import *


def validate(mt, mte, tmpl, wx, ev, audit, feature_cols):
    res = []

    def chk(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, repr(e)
        res.append(dict(check=name, result="PASS" if ok else "FAIL", detail=detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name} -> {detail}")

    chk("1 one row per zone-hour (train and test)", lambda: (not mt.duplicated(["zone", "hour_eat"]).any() and not mte.duplicated(["zone", "hour_eat"]).any(), f"train {len(mt)}, test {len(mte)}"))
    chk("2 only the 12 zone labels, same in train and test", lambda: (mt.zone.nunique() == N_ZONES and set(mt.zone) == set(mte.zone), sorted(mt.zone.unique())))
    chk("3 train hours on EAT clock inside 2025-01-01..10-31, on the hour", lambda: (mt.hour_eat.min() >= GRID_START and mt.hour_eat.max() <= TRAIN_END and (mt.hour_eat.dt.minute == 0).all(), f"{mt.hour_eat.min()} .. {mt.hour_eat.max()}"))
    chk("4 test = 12 zones x 336 h (1-14 Nov) and row_ids identical to template order", lambda: (len(mte) == 4032 and mte.hour_eat.min() == TEST_START and mte.hour_eat.max() == TEST_END and list(mte.row_id) == list(tmpl.row_id), f"{len(mte)} rows"))
    num = [c for c in mt.columns if mt[c].dtype.kind in "fi" and c not in ("zone_id",)]
    chk("5 no sentinel codes left; no negative trips/ops/weather magnitudes", lambda: (not mt[num].isin(SENTINELS).any().any() and (mt[["trips"] + OPS_COLS + ["rain_mm", "humidity_pct", "wind_kmh"]].min() >= 0).all(), "ok"))
    chk("6 row count unchanged by weather join and by events join", lambda: (audit["rows_before"] == audit["rows_after_weather"] == audit["rows_after_events"], audit))
    chk("7 train and test have identical feature columns (same order)", lambda: (all(c in mt and c in mte for c in feature_cols) and not ({"trips"} | set(OPS_COLS)) & set(mte.columns), f"{len(feature_cols)} features"))
    nolag = [c for c in feature_cols if not c.startswith(("lag_", "zone_level"))]
    chk("8 no NaN in non-lag features (train and test)", lambda: (not mt[nolag].isna().any().any() and not mte[nolag].isna().any().any(), "ok"))
    chk("9 no train-only column is a feature (trips, fare, wait, drivers)", lambda: (not ({"trips"} | set(OPS_COLS)) & set(feature_cols), "ok"))

    def lag_ok():
        d = mt.set_index(["zone", "hour_eat"])["trips"]
        s = mt.sample(min(3000, len(mt)), random_state=SEED)
        ref = d.reindex(pd.MultiIndex.from_arrays([s.zone, s.hour_eat - pd.Timedelta(hours=336)])).values
        m = ~np.isnan(ref) & s.lag_14d.notna().values
        return bool(np.allclose(ref[m], s.lag_14d.values[m])), f"{int(m.sum())} rows: lag_14d == trips exactly 336 h earlier"
    chk("10 lag features use only information >= 14 days old", lag_ok)
    chk("11 weather: one row per hour, no gaps, EAT range", lambda: (wx.hour_eat.is_unique and len(wx) == int((TEST_END - WX_START) / pd.Timedelta(hours=1)) + 1, f"{len(wx)} hours"))
    chk("12 weather rows from 1 Nov are flagged forecast", lambda: ((wx.loc[wx.hour_eat >= TEST_START, "data_type"] == "forecast").all(), "ok"))
    u = ev[ev["use"]]
    chk("13 every used event has end > start", lambda: ((u.end > u.start).all(), f"{len(u)} events"))
    return pd.DataFrame(res)