import numpy as np
import pandas as pd
from .config import *

EV_COLS = [f"ev_{t}_{p}" for t in POINT_TYPES for p in ("pre", "during", "post")] + ["ev_road_closure_during"]
WX_COLS = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh", "rain_last_3h", "rain_last_6h", "rain_last_24h", "rain_class", "is_raining"]
CAL_COLS = ["hour", "dow", "day_of_month", "month", "is_weekend", "hour_sin", "hour_cos", "days_to_month_end",
            "is_payday_window", "is_public_holiday", "is_holiday_eve", "is_day_after_holiday", "is_school_break"]
ZONE_COLS = ["zone_id", "zone_age_days", "t_days"]
LAG_COLS = ["lag_14d", "lag_21d", "lag_28d", "lag_same_hour_mean_3w", "zone_level_28d"]
EVT_COLS = EV_COLS + ["ev_n_active", "ev_att_log_max", "hours_to_next_event", "hours_since_last_event"]
FEATURE_COLS = CAL_COLS + ZONE_COLS + LAG_COLS + WX_COLS + EVT_COLS


def explode_events(ev, zones, windows=None, types=None):
    """One row per (event, zone, hour) with phase pre/during/post. Event reaches pre hours before and post hours after."""
    windows = windows or EVENT_WINDOWS
    parts = []
    for e in ev[ev["use"]].itertuples():
        if types is not None and e.event_type not in types:
            continue
        pre, post = windows.get(e.event_type, (0, 0))
        hrs = pd.date_range((e.start - pd.Timedelta(hours=pre)).floor("h"), (e.end + pd.Timedelta(hours=post)).ceil("h"), freq="h", inclusive="left")
        if len(hrs) == 0:
            continue
        during = (hrs < e.end) & (hrs + pd.Timedelta(hours=1) > e.start)
        before = hrs + pd.Timedelta(hours=1) <= e.start
        phase = np.where(during, "during", np.where(before, "pre", "post"))
        zs = list(zones) if e.citywide else list(e.zones)
        parts.append(pd.DataFrame({"zone": np.repeat(zs, len(hrs)), "hour_eat": np.tile(hrs.values, len(zs)),
                                   "phase": np.tile(phase, len(zs)), "event_id": e.event_id, "event_type": e.event_type,
                                   "att": e.att_filled, "status": e.status_clean}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["zone", "hour_eat", "phase", "event_id", "event_type", "att", "status"])


def _date_set(ev, etype):
    s = set()
    for e in ev[ev["use"] & ev["is_active"] & (ev["event_type"] == etype)].itertuples():
        s |= set(pd.date_range(e.start.normalize(), (e.end - pd.Timedelta(minutes=1)).normalize()))
    return s


def calendar_features(df, ev):
    t = df["hour_eat"]
    df["hour"], df["dow"], df["day_of_month"], df["month"] = t.dt.hour, t.dt.dayofweek, t.dt.day, t.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df["hour"] / 24), np.cos(2 * np.pi * df["hour"] / 24)
    df["days_to_month_end"] = t.dt.days_in_month - df["day_of_month"]
    df["is_payday_window"] = ((df["day_of_month"] >= PAYDAY_START_DAY) | (df["day_of_month"] <= PAYDAY_END_DAY)).astype(int)
    d = t.dt.normalize()
    hol, brk = _date_set(ev, "public_holiday"), _date_set(ev, "school_break")
    one = pd.Timedelta(days=1)
    df["is_public_holiday"] = d.isin(hol).astype(int)
    df["is_holiday_eve"] = (d + one).isin(hol).astype(int)
    df["is_day_after_holiday"] = (d - one).isin(hol).astype(int)
    df["is_school_break"] = d.isin(brk).astype(int)
    df["t_days"] = (t - GRID_START) / pd.Timedelta(days=1)
    return df


def lag_features(df):
    """Every lag is >= 336 h (14 days) so it is known when forecasting a 14-day horizon."""
    df = df.sort_values(["zone", "hour_eat"]).reset_index(drop=True)
    g = df.groupby("zone")["trips"]
    for k, n in [(336, "lag_14d"), (504, "lag_21d"), (672, "lag_28d")]:
        df[n] = g.shift(k)
    df["lag_same_hour_mean_3w"] = df[["lag_14d", "lag_21d", "lag_28d"]].mean(axis=1)
    df["zone_level_28d"] = g.transform(lambda s: s.shift(336).rolling(672, min_periods=168).mean())
    return df


def weather_features(wx):
    w = wx.sort_values("hour_eat").reset_index(drop=True).copy()
    r = w["rain_mm"]
    w["rain_last_3h"], w["rain_last_6h"], w["rain_last_24h"] = [r.rolling(k, min_periods=1).sum() for k in (3, 6, 24)]
    w["rain_class"] = np.select([r <= 0, r <= 2.5, r <= 7.6], [0, 1, 2], 3)
    w["is_raining"] = (r > 0).astype(int)
    return w


def event_features(keys, ev, zones):
    use = ev[ev["use"] & ev["is_active"]]
    L = explode_events(use, zones, types=POINT_TYPES + ["road_closure"])
    out = keys[["zone", "hour_eat"]].copy()
    if len(L):
        L = L[~((L["event_type"] == "road_closure") & (L["phase"] != "during"))]
        wide = L.assign(col="ev_" + L["event_type"] + "_" + L["phase"], v=1).pivot_table(index=["zone", "hour_eat"], columns="col", values="v", aggfunc="max")
        out = out.merge(wide.reindex(columns=EV_COLS).reset_index(), on=["zone", "hour_eat"], how="left", validate="1:1")
        out = out.merge(L.groupby(["zone", "hour_eat"])["event_id"].nunique().rename("ev_n_active").reset_index(), on=["zone", "hour_eat"], how="left", validate="1:1")
        att = np.log1p(L[L["event_type"].isin(POINT_TYPES)].groupby(["zone", "hour_eat"])["att"].max()).rename("ev_att_log_max").reset_index()
        out = out.merge(att, on=["zone", "hour_eat"], how="left", validate="1:1")
    for c in EV_COLS + ["ev_n_active", "ev_att_log_max"]:
        if c not in out:
            out[c] = 0
        out[c] = out[c].fillna(0)
    out[EV_COLS + ["ev_n_active"]] = out[EV_COLS + ["ev_n_active"]].astype(int)
    P = use[use["event_type"].isin(POINT_TYPES)]
    Lp = explode_events(P, zones, windows={t: (0, 0) for t in POINT_TYPES}, types=POINT_TYPES)
    k = out[["zone", "hour_eat"]].sort_values("hour_eat")
    if len(Lp):
        se = P.assign(zl=P.apply(lambda r: list(zones) if r["citywide"] else r["zones"], axis=1)).explode("zl")
        se = se.assign(zone=se["zl"], s=se["start"].dt.floor("h"), e=se["end"].dt.ceil("h"))
        nxt = pd.merge_asof(k, se[["zone", "s"]].sort_values("s"), left_on="hour_eat", right_on="s", by="zone", direction="forward")
        prv = pd.merge_asof(k, se[["zone", "e"]].sort_values("e"), left_on="hour_eat", right_on="e", by="zone", direction="backward")
        k["hours_to_next_event"] = ((nxt["s"] - nxt["hour_eat"]) / pd.Timedelta(hours=1)).clip(0, 72).fillna(72).values
        k["hours_since_last_event"] = ((prv["hour_eat"] - prv["e"]) / pd.Timedelta(hours=1)).clip(0, 72).fillna(72).values
    else:
        k["hours_to_next_event"], k["hours_since_last_event"] = 72.0, 72.0
    return out.merge(k, on=["zone", "hour_eat"], how="left", validate="1:1")


def build_master(train_grid, test, wx, ev, zones, launch_eff):
    cols = ["record_id", "zone", "hour_eat", "trips", "gap_type"] + OPS_COLS
    tr = train_grid[cols].assign(split="train")
    te = test[["row_id", "zone", "hour_eat", "_order"]].assign(split="test", trips=np.nan, gap_type="")
    a = pd.concat([tr, te], ignore_index=True)
    a = lag_features(a)
    audit = {"rows_before": len(a)}
    a = calendar_features(a, ev)
    a["zone_id"] = a["zone"].map({z: i for i, z in enumerate(zones)})
    a["zone_age_days"] = (a["hour_eat"] - a["zone"].map(launch_eff)) / pd.Timedelta(days=1)
    wf = weather_features(wx)
    a = a.merge(wf[["hour_eat"] + WX_COLS + ["wx_missing_hour", "wx_imputed"]], on="hour_eat", how="left", validate="m:1")
    audit["rows_after_weather"] = len(a)
    a = a.merge(event_features(a[["zone", "hour_eat"]].drop_duplicates(), ev, zones), on=["zone", "hour_eat"], how="left", validate="m:1")
    audit["rows_after_events"] = len(a)
    return a, audit


# ---------------------------------------------------------------- A6 feature table / A8 data dictionary
_SPEC = """hour|calendar|hour of day 0-23|hour_eat|daily demand cycle|yes
dow|calendar|day of week, Mon=0|hour_eat|weekly cycle, differs by zone type|yes
day_of_month|calendar|day number 1-31|hour_eat|month-end / payday pattern|yes
month|calendar|month 1-12|hour_eat|seasonality (rainy season)|yes
is_weekend|calendar|dow>=5|hour_eat|weekend shape differs per zone|yes
hour_sin|calendar|sin(2*pi*hour/24)|hour_eat|midnight continuity for linear/NN models|yes
hour_cos|calendar|cos(2*pi*hour/24)|hour_eat|midnight continuity for linear/NN models|yes
days_to_month_end|calendar|days_in_month - day|hour_eat|distance to payday|yes
is_payday_window|calendar|day>=25 or day<=3|hour_eat|pay-period effect (tested in B4.3)|yes
is_public_holiday|calendar/events|date inside an active public_holiday event|events: event_type,start,end,status|holidays shift demand|yes
is_holiday_eve|calendar/events|next day is a holiday|events|pre-holiday travel|yes
is_day_after_holiday|calendar/events|previous day was a holiday|events|return travel|yes
is_school_break|calendar/events|date inside an active school_break|events|school-run demand disappears|yes
zone_id|zone|integer code of canonical zone (sorted)|zone|zone-specific level and profile|yes
zone_age_days|trend|days since zone launch (1 Jan for zones live from the start)|first observed hour per zone (train)|late-launch zone ramps up|yes
t_days|trend|days since 2025-01-01|hour_eat|growth trend (B1.4)|yes
lag_14d|lag|trips of same zone 336 h earlier|trips|same hour-of-week two weeks ago; 336 h is the shortest lag known for a 14-day horizon|yes
lag_21d|lag|trips 504 h earlier|trips|same hour-of-week three weeks ago|yes
lag_28d|lag|trips 672 h earlier|trips|same hour-of-week four weeks ago|yes
lag_same_hour_mean_3w|lag|mean(lag_14d,lag_21d,lag_28d)|trips|smooths noise in single lags|yes
zone_level_28d|rolling|28-day rolling mean of zone trips, shifted 14 days|trips|current demand level / growth|yes
temp_c|weather|hourly temperature (EAT-aligned)|weather: temp_c|temperature affects trip mood|yes (forecast for Nov)
rain_mm|weather|rain in previous hour|weather: rain_mm|rain shifts demand (B2)|yes (forecast for Nov)
humidity_pct|weather|relative humidity|weather: humidity_pct|weak proxy for weather state|yes (forecast for Nov)
wind_kmh|weather|wind speed|weather: wind_kmh|weak proxy for weather state|yes (forecast for Nov)
rain_last_3h|weather|rain_mm summed over current+previous 2 h|rain_mm|demand reacts to recent rain, not only this hour|yes
rain_last_6h|weather|rain_mm summed over 6 h|rain_mm|longer rain memory|yes
rain_last_24h|weather|rain_mm summed over 24 h|rain_mm|wet-day indicator|yes
rain_class|weather|0 none, 1 light <=2.5, 2 moderate <=7.6, 3 heavy|rain_mm|non-linear dose-response (B2.3)|yes
is_raining|weather|rain_mm>0|rain_mm|simple rain flag|yes
ev_n_active|events|distinct active events whose pre/during/post window covers the zone-hour|events|crowd pressure|yes (calendar published in advance; cancellations are a risk)
ev_att_log_max|events|log1p(max attendance, imputed) of point events in window|events: expected_attendance|bigger events, bigger spike|yes
hours_to_next_event|events|hours until next point-event start in the zone, cap 72|events: start|build-up before events|yes
hours_since_last_event|events|hours since last point-event end in the zone, cap 72|events: end|post-event exit surge|yes"""


def feature_spec():
    rows = [l.split("|") for l in _SPEC.splitlines()]
    for t in POINT_TYPES:
        for p, why in (("pre", "arrivals before"), ("during", "demand while it runs"), ("post", "exit surge after")):
            rows.append([f"ev_{t}_{p}", "events", f"1 if zone-hour is in the {p} phase of an active {t} (window {EVENT_WINDOWS[t]} h)",
                         "events: event_type,zone,start,end,status", f"{why} the event", "yes"])
    rows.append(["ev_road_closure_during", "events", "1 if an active road_closure covers the zone-hour", "events", "closures disrupt trips", "yes"])
    spec = pd.DataFrame(rows, columns=["feature", "group", "formula", "source_columns", "why_it_should_help", "known_at_forecast_time"])
    assert set(spec["feature"]) == set(FEATURE_COLS), set(FEATURE_COLS) ^ set(spec["feature"])
    return spec.set_index("feature").loc[FEATURE_COLS].reset_index()


def data_dictionary(mt, mte):
    spec = feature_spec().set_index("feature")
    base = {"record_id": ("id", "ride_demand_train", "unique trip-record id"), "row_id": ("id", "ride_demand_test", "row id used for scoring"),
            "zone": ("id", "all tables", "canonical zone label (12)"), "hour_eat": ("id", "pickup_hour", "start of hour, Addis Ababa local time (EAT, UTC+3)"),
            "trips": ("target", "ride_demand_train", "trips requested in the zone-hour"),
            "avg_fare_birr": ("analysis_only", "ride_demand_train", "average fare, train only - NOT a model input"),
            "avg_wait_min": ("analysis_only", "ride_demand_train", "average wait, train only - NOT a model input"),
            "active_drivers": ("analysis_only", "ride_demand_train", "active drivers, train only - NOT a model input")}
    rows = []
    for c in dict.fromkeys(list(mt.columns) + list(mte.columns)):
        ser = mt[c] if c in mt else mte[c]
        if c in base:
            role, src, desc = base[c]; how = "cleaned in src/cleaning.py"
        elif c in spec.index:
            role, src, desc, how = "feature", spec.loc[c, "source_columns"], spec.loc[c, "why_it_should_help"], spec.loc[c, "formula"]
        else:
            continue
        rows.append(dict(column=c, role=role, dtype=str(ser.dtype), source=src, description=desc, derivation=how,
                         in_master_train=c in mt, in_master_test=c in mte))
    return pd.DataFrame(rows)