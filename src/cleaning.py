import re, difflib
import numpy as np
import pandas as pd
from .config import *


class CleaningLog:
    def __init__(self):
        self.rows, self.parse = [], {}

    def add(self, file, columns, issue, n, total, fix, why):
        self.rows.append(dict(file=file, columns=columns, issue=issue, rows_affected=int(n),
                              pct_of_rows=round(100 * n / total, 3) if total else np.nan, fix=fix, why=why))

    def df(self):
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------- generic helpers
def _blank_to_nan(s):
    s = s.astype(str).str.strip()
    return s.mask(s.str.lower().isin(NA_TOKENS))


def to_number(s, col, log, fname):
    s = _blank_to_nan(s)
    num = pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")
    bad = num.isna() & s.notna()
    rec = 0
    if bad.any():
        ext = pd.to_numeric(s[bad].str.replace(",", "", regex=False).str.extract(r"(-?\d+\.?\d*)")[0], errors="coerce")
        num.loc[bad] = ext
        rec = int(ext.notna().sum())
    log.add(fname, col, "text inside numeric column (units, words, thousands separators)", bad.sum(), len(s),
            f"extracted number where possible ({rec} recovered), else NaN", "keeps real values, never invents one")
    log.add(fname, col, "blank / NA-token values", s.isna().sum(), len(s), "kept as NaN (imputed later only where documented)", "missing is not zero")
    return num


def drop_sentinels(x, col, fname, log):
    m = x.isin(SENTINELS)
    log.add(fname, col, "sentinel codes meaning 'no reading' (-999, -9999, 999, 9999 ...)", m.sum(), len(x), "set to NaN", "sentinels would distort every statistic")
    return x.mask(m)


def drop_range(x, col, fname, log):
    lo, hi = RANGES[col]
    m = (x < lo) | (x > hi)
    log.add(fname, col, f"impossible values outside [{lo},{hi}]", m.sum(), len(x), "set to NaN", "physically / operationally impossible")
    return x.mask(m)


FORMATS = ["%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M",
           "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S", "%d-%m-%Y %H:%M", "%d.%m.%Y %H:%M",
           "%b %d, %Y %I:%M %p", "%B %d, %Y %I:%M %p", "%d %b %Y %H:%M", "%Y/%m/%d %H:%M",
           "%Y-%m-%d", "%d/%m/%Y", "%b %d, %Y", "%d %b %Y"]


def parse_datetimes(s, split_tz=False):
    """Explicit-format parsing (no guessing). Returns (naive datetimes, report DataFrame, tz offset hours or NaN)."""
    s = s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    s = s.mask(s.str.lower().isin(NA_TOKENS))
    rep = []
    # slash dates: dd/mm is the default; a second field > 12 proves month-first for that row
    sl = s.str.extract(r"^(\d{1,2})/(\d{1,2})/(\d{4})(.*)$")
    is_sl = sl[0].notna()
    a, b = pd.to_numeric(sl[0]), pd.to_numeric(sl[1])
    mf = is_sl & (b > 12) & (a <= 12)
    rep += [("slash rows", int(is_sl.sum())), ("slash: day-first PROVEN (first field>12)", int((is_sl & (a > 12)).sum())),
            ("slash: month-first PROVEN (second field>12) -> swapped", int(mf.sum())),
            ("slash: ambiguous (both<=12) -> read day-first", int((is_sl & (a <= 12) & (b <= 12)).sum()))]
    dd = pd.Series(np.where(mf, sl[1], sl[0]), index=s.index).astype(str)
    mm = pd.Series(np.where(mf, sl[0], sl[1]), index=s.index).astype(str)
    s = s.mask(is_sl, dd.str.zfill(2) + "/" + mm.str.zfill(2) + "/" + sl[2].astype(str) + sl[3].astype(str))
    off = pd.Series(np.nan, index=s.index)
    if split_tz:
        tz = s.str.extract(r"(Z|[+-]\d{2}:?\d{2})$")[0]
        off = tz.map(lambda x: np.nan if pd.isna(x) else (0.0 if x == "Z" else (1 if x[0] == "+" else -1) * (int(x[1:3]) + int(x[-2:]) / 60)))
        s = s.str.replace(r"(Z|[+-]\d{2}:?\d{2})$", "", regex=True)
        rep += [("tz marker present (Z / +hh:mm)", int(off.notna().sum())), ("no tz marker (clock unknown)", int(off.isna().sum()))]
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for f in FORMATS:
        todo = out.isna() & s.notna()
        if not todo.any():
            break
        p = pd.to_datetime(s[todo], format=f, errors="coerce")
        ok = p.notna()
        if ok.any():
            out.loc[p.index[ok]] = p[ok]
            rep.append((f"format {f}", int(ok.sum())))
    rep.append(("UNPARSED (non-blank)", int((out.isna() & s.notna()).sum())))
    return out, pd.DataFrame(rep, columns=["format_or_check", "rows"]), off


# ---------------------------------------------------------------- zones
GENERIC = r"\b(zone|area|district|sub ?city|woreda|kebele)\b"


def zkey(x):
    x = re.sub(r"\(.*?\)|\[.*?\]", " ", str(x).lower())
    x = re.sub(GENERIC, " ", x)
    return re.sub(r"[^a-z0-9]+", "", x)


class ZoneMapper:
    """Learns the 12 canonical zones from the TRAIN zone column only."""

    def fit(self, raw):
        raw = _blank_to_nan(raw).dropna()
        keys = raw.map(zkey)
        self.key_counts = keys.value_counts()
        self.keys = [zkey(z) for z in CANONICAL_ZONES] if CANONICAL_ZONES else list(self.key_counts.index[:N_ZONES])
        self.labels = {}
        for i, k in enumerate(self.keys):
            if CANONICAL_ZONES:
                self.labels[k] = CANONICAL_ZONES[i]
            else:
                spell = raw[keys == k].str.replace(r"\(.*?\)", "", regex=True).str.strip().value_counts().index[0]
                self.labels[k] = spell.upper() if len(k) <= 3 else spell.title()
        assert len(self.labels) == N_ZONES, f"expected {N_ZONES} zones, got {len(self.labels)}"
        return self

    def key_to_label(self, k):
        k = ZONE_ALIASES.get(k, k)
        if k in self.labels:
            return self.labels[k]
        hit = difflib.get_close_matches(k, self.keys, n=1, cutoff=0.8)
        return self.labels[hit[0]] if hit else np.nan

    def map_one(self, x):
        return np.nan if pd.isna(x) or str(x).strip().lower() in NA_TOKENS else self.key_to_label(zkey(x))

    def map(self, raw):
        keys = _blank_to_nan(raw).map(lambda x: np.nan if pd.isna(x) else zkey(x))
        cache = {k: self.key_to_label(k) for k in keys.dropna().unique()}
        return keys.map(cache)

    def map_cell(self, text):
        """events zone cell -> (list of zone labels, citywide flag). Handles extra text, several zones, 'Citywide'."""
        if pd.isna(text) or str(text).strip().lower() in NA_TOKENS:
            return [], False
        t = str(text).lower()
        if re.search(r"city\s*-?\s*wide|all zones|whole city|entire city|all of addis|\ball\b", t):
            return list(self.labels.values()), True
        t = re.sub(r"\(.*?\)", " ", t)
        out = []
        for p in [p for p in re.split(r",|;|/|&|\+|\||\band\b|\bor\b", t) if p.strip()]:
            k = zkey(p)
            lab = self.key_to_label(k)
            if pd.isna(lab):
                out += [self.labels[c] for c in self.keys if len(c) >= 3 and c in k]
            else:
                out.append(lab)
        return list(dict.fromkeys(out)), False


# ---------------------------------------------------------------- trips
def clean_trips(raw, zm, log, fname, is_test=False):
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    n0 = len(df)
    idc = "row_id" if is_test else "record_id"
    df["_order"] = np.arange(n0)
    if not is_test:
        ex = df.drop(columns="_order").duplicated(keep="first")
        log.add(fname, "all", "exact duplicate rows", ex.sum(), n0, "dropped copies", "double counting")
        df = df[~ex]
        d2 = df.duplicated(idc, keep="first")
        log.add(fname, idc, "same record_id twice with different content", d2.sum(), n0, "kept first", "record_id must be unique")
        df = df[~d2]
    df["zone_raw"] = df["zone"]
    df["zone"] = zm.map(df["zone"])
    chg = df["zone"].notna() & (df["zone_raw"].astype(str).str.strip() != df["zone"])
    log.add(fname, "zone", "inconsistent spelling / case / whitespace / suffix (e.g. 'PIASSA', 'Kazanchis (Kirkos)')", chg.sum(), n0,
            "normalised key + fuzzy match to the 12 canonical labels", "one label per zone")
    um = df["zone"].isna()
    log.add(fname, "zone", "blank or unmatched zone", um.sum(), n0, "dropped (train) / error (test)", "cannot be assigned to a zone")
    if is_test and um.any():
        raise ValueError("test rows with unmatched zone - extend ZONE_ALIASES")
    df = df[~um]
    ts, rep, _ = parse_datetimes(df["pickup_hour"])
    log.parse[f"{fname}.pickup_hour"] = rep
    df["hour_eat"] = ts
    bad = df["hour_eat"].isna()
    log.add(fname, "pickup_hour", "unparseable timestamp", bad.sum(), n0, "dropped (train) / error (test)", "cannot be placed on the time axis")
    if is_test and bad.any():
        raise ValueError("unparseable test timestamps")
    df = df[~bad]
    off = df["hour_eat"].dt.minute != 0
    log.add(fname, "pickup_hour", "timestamp not on the hour", off.sum(), n0, "floored to hour", "hourly grain")
    df["hour_eat"] = df["hour_eat"].dt.floor("h")
    lo, hi = (TEST_START, TEST_END) if is_test else (GRID_START, TRAIN_END)
    oor = (df["hour_eat"] < lo) | (df["hour_eat"] > hi)
    log.add(fname, "pickup_hour", f"timestamp outside {lo.date()}..{hi.date()}", oor.sum(), n0, "dropped (train) / error (test)", "wrong year or typo")
    if is_test and oor.any():
        raise ValueError("test timestamps outside the forecast fortnight")
    df = df[~oor]
    if is_test:
        assert not df.duplicated(["zone", "hour_eat"]).any(), "duplicate zone-hours in test"
        assert len(df) == n0, "test rows were lost"
        return df[[idc, "zone", "hour_eat", "_order"]].sort_values("_order").reset_index(drop=True)
    for c in ["trips"] + OPS_COLS:
        x = to_number(df[c], c, log, fname)
        x = drop_range(drop_sentinels(x, c, fname, log), c, fname, log)
        df[c] = x
    ni = (df["trips"].notna() & (df["trips"] % 1 != 0)).sum()
    log.add(fname, "trips", "non-integer trip counts", ni, n0, "rounded", "trips are counts")
    df["trips"] = df["trips"].round()
    df["active_drivers"] = df["active_drivers"].round()
    key = ["zone", "hour_eat"]
    d = df.duplicated(key, keep=False)
    conf = int(df[d].groupby(key)["trips"].nunique().gt(1).sum())
    log.add(fname, "zone,pickup_hour", f"same zone-hour appears several times ({conf} groups disagree on trips)", d.sum(), n0,
            "merged to one row (median of numeric columns)", "one row per zone-hour")
    df = df.groupby(key, as_index=False).agg(record_id=("record_id", "first"), trips=("trips", "median"),
                                              avg_fare_birr=("avg_fare_birr", "median"), avg_wait_min=("avg_wait_min", "median"),
                                              active_drivers=("active_drivers", "median"))
    df["trips"] = df["trips"].round()
    cap = df.groupby("zone")["trips"].transform(lambda s: s.quantile(0.995))
    err = df["trips"] > TRIPS_ERROR_MULT * cap
    log.add(fname, "trips", f"absurd spikes > {TRIPS_ERROR_MULT} x zone 99.5th percentile (data-entry errors)", err.sum(), n0,
            "set to NaN (genuine event spikes below this are KEPT)", "outliers would dominate RMSE; real spikes are signal")
    df.loc[err, "trips"] = np.nan
    log.add(fname, "trips", "target missing after cleaning", df["trips"].isna().sum(), n0, "excluded from training, never imputed", "target must be real")
    return df[["record_id", "zone", "hour_eat", "trips"] + OPS_COLS]


def build_grid(tr):
    zones = sorted(tr["zone"].unique())
    first = tr.groupby("zone")["hour_eat"].min()
    late = first > GRID_START + pd.Timedelta(days=LATE_LAUNCH_AFTER_DAYS)
    launch_eff = first.where(late, GRID_START)
    parts = [pd.DataFrame({"zone": z, "hour_eat": pd.date_range(launch_eff[z], TRAIN_END, freq="h")}) for z in zones]
    g = pd.concat(parts, ignore_index=True).merge(tr, on=["zone", "hour_eat"], how="left", validate="1:1")
    had = g["record_id"].notna()
    miss = (~had).groupby(g["hour_eat"]).mean()
    out_h = miss.index[miss >= OUTAGE_MISSING_FRAC]
    g["gap_type"] = np.select([g["trips"].notna(), had & g["trips"].isna(), g["hour_eat"].isin(out_h)],
                              ["observed", "invalid_value", "outage"], "random_missing")
    return g, first, late, launch_eff


# ---------------------------------------------------------------- weather
def _circ(a, b):
    d = abs(a - b) % 24
    return min(d, 24 - d)


def clean_weather(raw, log, fname="weather_hourly.csv"):
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    n0 = len(df)
    num = ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]
    df["raw_ts"] = df["timestamp"].astype(str)
    naive, rep, off = parse_datetimes(df["timestamp"], split_tz=True)
    log.parse["weather.timestamp"] = rep
    bad = naive.isna()
    log.add(fname, "timestamp", "unparseable / blank timestamp", bad.sum(), n0, "dropped", "cannot be placed on the time axis")
    df, naive, off = df[~bad].copy(), naive[~bad], off[~bad]
    df["naive"], df["tz_off"] = naive, off
    log.add(fname, "timestamp", "mixed timestamp conventions (some rows with Z/offset, some without)",
            min(off.notna().sum(), off.isna().sum()), n0, "clock decided per group with temperature evidence (below)", "a join is only as good as the clock")
    for c in num:
        x = to_number(df[c], c, log, fname)
        df[c] = drop_sentinels(x, c, fname, log)
    day, mon = df["naive"].dt.floor("D"), df["naive"].dt.month
    diag = df.groupby(mon).agg(temp_median=("temp_c", "median"), humidity_q99=("humidity_pct", lambda s: s.quantile(.99)),
                               wind_median=("wind_kmh", "median"), rain_wet_mean=("rain_mm", lambda s: s[s > 0].mean())).round(3)
    # --- unit problems for part of the year
    f_rows = df.groupby(day)["temp_c"].transform("median") > 45
    rng = f"{df.loc[f_rows, 'naive'].min()} .. {df.loc[f_rows, 'naive'].max()}" if f_rows.any() else "n/a"
    df.loc[f_rows, "temp_c"] = (df.loc[f_rows, "temp_c"] - 32) * 5 / 9
    log.add(fname, "temp_c", f"part of the year in Fahrenheit (daily median > 45) [{rng}]", f_rows.sum(), n0, "converted (F-32)*5/9", "mixed units")
    h_rows = df.groupby(mon)["humidity_pct"].transform(lambda s: s.quantile(.99)) <= 1.5
    df.loc[h_rows, "humidity_pct"] *= 100
    log.add(fname, "humidity_pct", "months stored as fraction 0-1 instead of %", h_rows.sum(), n0, "x100", "mixed units")
    w_ratio = df.groupby(mon)["wind_kmh"].transform("median") / df.groupby(mon)["wind_kmh"].median().median()
    w_rows = w_ratio.between(0.2, 0.36)
    df.loc[w_rows, "wind_kmh"] *= 3.6
    log.add(fname, "wind_kmh", "months that look like m/s (median ~0.28x of other months)", w_rows.sum(), n0, "x3.6 to km/h", "mixed units (check unit_diag)")
    wet = df["rain_mm"].where(df["rain_mm"] > 0)
    r_ratio = wet.groupby(mon).transform("mean") / wet.groupby(mon).mean().median()
    r_rows = r_ratio.between(0.02, 0.08)
    df.loc[r_rows, "rain_mm"] *= 25.4
    log.add(fname, "rain_mm", "months that look like inches (wet-hour mean ~1/25 of other months)", r_rows.sum(), n0, "x25.4 to mm", "mixed units (check unit_diag)")
    for c in num:
        df[c] = drop_range(df[c], c, fname, log)
    # --- clock evidence
    z = df["tz_off"].notna()
    eat = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    eat[z] = df.loc[z, "naive"] - pd.to_timedelta(df.loc[z, "tz_off"], unit="h") + pd.Timedelta(hours=UTC_OFFSET_H)
    nz = ~z
    ev = {"decision": "all rows carry a tz marker"}
    prof = {}
    if z.any():
        pz = df[z].groupby(eat[z].dt.hour)["temp_c"].mean()
        prof["tz_marked_rows__EAT_hour"] = pz
        p_loc = int(pz.idxmax())
    if nz.any():
        pn = df[nz].groupby(df.loc[nz, "naive"].dt.hour)["temp_c"].mean()
        prof["unmarked_rows__raw_hour"] = pn
        p_n = int(pn.idxmax())
        if z.any():
            p_utc = (p_loc - UTC_OFFSET_H) % 24
            clock = "EAT" if _circ(p_n, p_loc) <= _circ(p_n, p_utc) else "UTC"
        else:
            clock = "EAT" if _circ(p_n, 14) <= _circ((p_n + UTC_OFFSET_H) % 24, 14) else "UTC"
        eat[nz] = df.loc[nz, "naive"] + pd.Timedelta(hours=0 if clock == "EAT" else UTC_OFFSET_H)
        ev["decision"] = f"unmarked rows are on {clock} (temp peak raw hour {p_n}" + (f"; marked rows peak at {p_loc} EAT)" if z.any() else ")")
    ev["profiles"] = pd.DataFrame(prof).round(2)
    ev["unit_diag"] = diag
    df["hour_eat"] = eat.dt.floor("h")
    # --- data_type consistency
    dt_raw = df["data_type"].astype(str).str.strip().str.lower()
    expected = np.where(df["hour_eat"] >= TEST_START, "forecast", "observed")
    got = np.where(dt_raw.str[:1] == "f", "forecast", np.where(dt_raw.str[:1].isin(["o", "a"]), "observed", ""))
    log.add(fname, "data_type", "label inconsistent with date (forecast must be >= 1 Nov) or misspelled", (got != expected).sum(), n0, "re-derived from corrected EAT time", "forecast/observed flag must be reliable")
    df["data_type"] = expected
    oor = (df["hour_eat"] < WX_START) | (df["hour_eat"] > TEST_END)
    log.add(fname, "timestamp", "hours outside modelling range after clock correction", oor.sum(), n0, "dropped", "not needed")
    df = df[~oor]
    g = df.groupby("hour_eat")
    dup = int((g.size() > 1).sum())
    conf = int((g[num].nunique() > 1).any(axis=1).sum())
    log.add(fname, "timestamp", f"duplicate hours ({conf} with conflicting values)", dup, n0, "averaged per hour", "one weather row per hour is required for a many-to-one join")
    agg = g.agg({**{c: "mean" for c in num}, "data_type": "first", "raw_ts": lambda s: "|".join(s)}).rename(columns={"raw_ts": "raw_timestamps"})
    full = pd.date_range(WX_START, TEST_END, freq="h")
    w = agg.reindex(full)
    w.index.name = "hour_eat"
    w["wx_missing_hour"] = ~w.index.isin(agg.index)
    log.add(fname, "timestamp", "hours with no weather row at all (gaps)", w["wx_missing_hour"].sum(), len(w), "re-indexed to full hourly grid, then imputed", "every zone-hour needs weather")
    w["wx_imputed"] = w[num].isna().any(axis=1)
    w["data_type"] = np.where(w.index >= TEST_START, "forecast", "observed")
    for c in ["temp_c", "humidity_pct", "wind_kmh"]:
        w[c] = w[c].interpolate(limit=6, limit_area="inside")
    obs = w[(w.index < TEST_START) & ~w["wx_imputed"]]
    mm = np.where(w.index.month == 11, 10, w.index.month)
    idx = pd.MultiIndex.from_arrays([mm, w.index.hour])
    for c in ["temp_c", "humidity_pct", "wind_kmh"]:   # climatology learned from TRAIN-period observed rows only
        clim = obs.groupby([obs.index.month, obs.index.hour])[c].mean()
        w[c] = w[c].fillna(pd.Series(clim.reindex(idx).values, index=w.index))
        w[c] = w[c].fillna(pd.Series(w.index.hour, index=w.index).map(obs.groupby(obs.index.hour)[c].mean()))
    w["rain_mm"] = w["rain_mm"].fillna(0)
    w["humidity_pct"] = w["humidity_pct"].clip(0, 100)
    log.add(fname, "temp_c,rain_mm,humidity_pct,wind_kmh", "remaining NaN after cleaning", w["wx_imputed"].sum(), len(w),
            "<=6h linear interpolation; longer: month x hour climatology from train period; rain -> 0 (flagged wx_imputed)", "no NaN features at forecast time")
    return w.reset_index(), ev


# ---------------------------------------------------------------- events
SYN = {"holiday": "public_holiday", "public": "public_holiday", "school": "school_break", "break": "school_break",
       "football": "football_match", "match": "football_match", "soccer": "football_match", "concert": "concert",
       "music": "concert", "conference": "conference", "summit": "conference", "expo": "exhibition",
       "exhibition": "exhibition", "closure": "road_closure", "road": "road_closure", "run": "sports_run",
       "marathon": "sports_run", "race": "sports_run"}


def norm_event_type(x):
    if pd.isna(x):
        return np.nan
    k = re.sub(r"[\s\-]+", "_", str(x).strip().lower())
    if k in EVENT_TYPES:
        return k
    hit = difflib.get_close_matches(k, EVENT_TYPES, n=1, cutoff=0.75)
    if hit:
        return hit[0]
    for w, t in SYN.items():
        if w in k:
            return t
    return np.nan


def norm_status(x):
    if pd.isna(x):
        return "unknown"
    t = str(x).strip().lower()
    if re.search(r"cancel|called off|abandon", t):
        return "cancelled"
    if re.search(r"postpon|delay|resched", t):
        return "postponed"
    if re.search(r"confirm|held|went ahead|happened|complete|yes", t):
        return "confirmed"
    return "unknown"


def parse_attendance(x):
    if pd.isna(x):
        return np.nan
    t = str(x).lower().replace(",", "").replace(" ", "")
    nums = re.findall(r"(\d+\.?\d*)(k|m)?", t)
    if not nums:
        return np.nan
    vals = [float(n) * (1e3 if u == "k" else 1e6 if u == "m" else 1) for n, u in nums]
    return float(np.mean(vals))


def clean_events(raw, zm, log, fname="events_calendar.csv"):
    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    n0 = len(df)
    df["raw_index"] = np.arange(n0)
    for c in ["event_id", "event_name", "venue", "zone", "status", "expected_attendance"]:
        df[c] = _blank_to_nan(df[c])
    df["event_type_raw"] = df["event_type"]
    df["event_type"] = df["event_type"].map(norm_event_type)
    log.add(fname, "event_type", "inconsistent spelling/case (e.g. 'Road closure' vs 'road_closure')",
            (df["event_type_raw"].astype(str).str.strip() != df["event_type"]).sum(), n0, "mapped to 8 canonical types", "consistent categories")
    df["start"], rs, _ = parse_datetimes(df["start_datetime"])
    df["end"], rend, _ = parse_datetimes(df["end_datetime"])
    log.parse["events.start_datetime"], log.parse["events.end_datetime"] = rs, rend
    log.add(fname, "start_datetime,end_datetime", "dates in several formats (ISO, dd/mm/yyyy, 'Feb 06, 2025 08:00 PM')",
            int(rs.loc[rs.format_or_check.str.startswith("format"), "rows"].sum()), n0, "explicit-format parsing, day-first for slash dates", "no silent misreads")
    dur = ((df["end"] - df["start"]).dt.total_seconds() / 3600).where(lambda x: x > 0)
    med = dur[df["start"] < TEST_START].groupby(df["event_type"]).median()
    default = {"public_holiday": 24, "school_break": 168}
    fill = df["event_type"].map(med).fillna(df["event_type"].map(default)).fillna(3)
    m_end, e_lt = df["end"].isna(), df["end"].notna() & (df["end"] <= df["start"])
    log.add(fname, "end_datetime", "end missing", m_end.sum(), n0, "end = start + median duration of that event type (learned from valid rows)", "window needs an end")
    log.add(fname, "end_datetime", "end earlier than (or equal to) start", e_lt.sum(), n0, "same rule as missing end", "swap/overnight cannot be told apart reliably")
    bad_end = (m_end | e_lt) & df["start"].notna()
    df["end_imputed"] = bad_end
    df.loc[bad_end, "end"] = df["start"] + pd.to_timedelta(fill, unit="h")
    df["duration_h"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    df["attendance"] = df["expected_attendance"].map(parse_attendance)
    free = df["expected_attendance"].notna() & ~df["expected_attendance"].astype(str).str.fullmatch(r"\d+")
    log.add(fname, "expected_attendance", "free text ('approx 34000', '20k', ranges)", free.sum(), n0, "parsed to a number (k/m suffix, mean of ranges)", "numeric feature")
    log.add(fname, "expected_attendance", "blank attendance", df["attendance"].isna().sum(), n0, "median of its event type (events before Nov only) + flag", "avoid NaN features")
    df["attendance_imputed"] = df["attendance"].isna()
    tmed = df[df["start"] < TEST_START].groupby("event_type")["attendance"].median()
    df["att_filled"] = df["attendance"].fillna(df["event_type"].map(tmed)).fillna(df["attendance"].median()).fillna(0)
    parsed = df["zone"].map(zm.map_cell)
    df["zones"] = parsed.map(lambda t: t[0])
    df["citywide"] = parsed.map(lambda t: t[1])
    log.add(fname, "zone", "zone spelled differently / extra text / several zones / 'Citywide'", df["zone"].notna().sum(), n0, "split, normalised, fuzzy-matched to the 12 zone labels", "joinable to trips")
    one = df[(df["zones"].map(len) == 1) & df["venue"].notna()]
    v2z = one.groupby("venue")["zones"].agg(lambda s: pd.Series([z[0] for z in s]).mode().iloc[0]).to_dict()
    nz = df["zones"].map(len) == 0
    by_v = nz & df["venue"].isin(v2z)
    df.loc[by_v, "zones"] = df.loc[by_v, "venue"].map(lambda v: [v2z[v]])
    by_t = (df["zones"].map(len) == 0) & df["event_type"].isin(["public_holiday", "school_break"])
    df.loc[by_t, "zones"] = df.loc[by_t, "zones"].map(lambda _: list(zm.labels.values()))
    df.loc[by_t, "citywide"] = True
    log.add(fname, "zone", "zone blank/unresolved", nz.sum(), n0, f"venue->zone lookup learned from the table ({int(by_v.sum())} filled); holidays/breaks -> citywide ({int(by_t.sum())})", "recover location without guessing")
    st = df["status"].map(norm_status)
    log.add(fname, "status", "inconsistent status spelling / blank", (df["status"].astype(str).str.strip().str.lower() != st).sum(), n0, "normalised to confirmed/cancelled/postponed/unknown", "consistent categories")
    df["status_clean"] = st
    df["is_active"] = st.isin(ACTIVE_STATUSES)
    df["_k"] = df["event_type"].astype(str) + "|" + df["start"].astype(str) + "|" + df["zones"].map(lambda z: ";".join(sorted(z)))
    df = df.sort_values("attendance", na_position="last", kind="stable")
    dup = df.duplicated("_k", keep="first")
    log.add(fname, "event_id", "duplicate events (same type, start, zones)", dup.sum(), n0, "kept the row with attendance", "double counting")
    df = df[~dup].sort_values("raw_index")
    r = pd.Series("", index=df.index)
    r[df["start"].isna()] = "start missing/unparseable"
    r[(r == "") & df["event_type"].isna()] = "event_type unrecognised"
    r[(r == "") & (df["zones"].map(len) == 0)] = "zone unresolved"
    r[(r == "") & ((df["end"] < GRID_START - pd.Timedelta(days=3)) | (df["start"] > TEST_END + pd.Timedelta(days=3)))] = "outside modelling period"
    df["excluded_reason"] = r
    df["use"] = r == ""
    df["zones_str"] = df["zones"].map(lambda z: ";".join(z))
    return df.drop(columns=["_k"]).reset_index(drop=True)
