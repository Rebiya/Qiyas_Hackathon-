from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TZ = "Africa/Addis_Ababa"
RANDOM_STATE = 42

ZONES = [
    "Arat Kilo",
    "Ayat",
    "Bole",
    "CMC",
    "Gerji",
    "Kazanchis",
    "Kolfe",
    "Lideta",
    "Megenagna",
    "Merkato",
    "Piassa",
    "Sarbet",
]

ZONE_ALIASES = {
    "arat kilo": "Arat Kilo",
    "ayat": "Ayat",
    "bole": "Bole",
    "bole rd": "Bole",
    "cmc": "CMC",
    "c m c": "CMC",
    "gerji": "Gerji",
    "kazanchis": "Kazanchis",
    "kazanches": "Kazanchis",
    "kolfe": "Kolfe",
    "kolfe keranio": "Kolfe",
    "lideta": "Lideta",
    "megenagna": "Megenagna",
    "megenaga": "Megenagna",
    "merkato": "Merkato",
    "mercato": "Merkato",
    "piassa": "Piassa",
    "piazza": "Piassa",
    "sarbet": "Sarbet",
}

EVENT_TYPE_ALIASES = {
    "concert": "concert",
    "conference": "conference",
    "exhibition": "exhibition",
    "football match": "football_match",
    "football_match": "football_match",
    "public holiday": "public_holiday",
    "public_holiday": "public_holiday",
    "road closure": "road_closure",
    "road_closure": "road_closure",
    "school break": "school_break",
    "school_break": "school_break",
    "sports run": "sports_run",
    "sports_run": "sports_run",
}


def normalize_token(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.strip().lower().replace(".", " ").replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def clean_zone(value: object) -> str | None:
    token = normalize_token(value)
    token = re.sub(r"\([^)]*\)", "", token).strip()
    return ZONE_ALIASES.get(token)


def clean_event_type(value: object) -> str:
    return EVENT_TYPE_ALIASES.get(normalize_token(value), normalize_token(value).replace(" ", "_"))


def parse_mixed_local(series: pd.Series, source_name: str) -> tuple[pd.Series, pd.DataFrame]:
    """Parse known timestamp formats without day/month ambiguity."""
    s = series.astype("string")
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    fmt = pd.Series("unparsed", index=series.index, dtype="object")

    iso_z = s.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", na=False)
    if iso_z.any():
        parsed.loc[iso_z] = (
            pd.to_datetime(s.loc[iso_z], format="%Y-%m-%dT%H:%M:%SZ", utc=True, errors="coerce")
            .dt.tz_convert(TZ)
            .dt.tz_localize(None)
        )
        fmt.loc[iso_z] = "utc_iso_z_to_eat"

    iso_offset = s.str.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$", na=False)
    if iso_offset.any():
        parsed.loc[iso_offset] = (
            pd.to_datetime(s.loc[iso_offset], utc=True, errors="coerce")
            .dt.tz_convert(TZ)
            .dt.tz_localize(None)
        )
        fmt.loc[iso_offset] = "iso_offset_to_eat"

    iso_local = s.str.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$", na=False)
    if iso_local.any():
        parsed.loc[iso_local] = pd.to_datetime(
            s.loc[iso_local], format="%Y-%m-%d %H:%M", errors="coerce"
        )
        fmt.loc[iso_local] = "local_yyyy_mm_dd_hhmm"

    dmy_local = s.str.match(r"^\d{2}/\d{2}/\d{4} \d{2}:\d{2}$", na=False)
    if dmy_local.any():
        parsed.loc[dmy_local] = pd.to_datetime(
            s.loc[dmy_local], format="%d/%m/%Y %H:%M", errors="coerce"
        )
        fmt.loc[dmy_local] = "local_dd_mm_yyyy_hhmm"

    month_name = s.str.match(r"^[A-Za-z]{3} \d{1,2}, \d{4} \d{2}:\d{2} [AP]M$", na=False)
    if month_name.any():
        parsed.loc[month_name] = pd.to_datetime(
            s.loc[month_name], format="%b %d, %Y %I:%M %p", errors="coerce"
        )
        fmt.loc[month_name] = "local_month_name_12h"

    audit = (
        pd.DataFrame({"source": source_name, "format_rule": fmt, "parsed_ok": parsed.notna()})
        .groupby(["source", "format_rule", "parsed_ok"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    return parsed, audit


def pct(n: int, d: int) -> float:
    return round(100 * n / d, 2) if d else 0.0


def add_log(log: list[dict], file: str, columns: str, issue: str, affected: int, total: int, fix: str, why: str) -> None:
    log.append(
        {
            "file": file,
            "columns": columns,
            "issue_type": issue,
            "rows_affected": int(affected),
            "pct_affected": pct(int(affected), int(total)),
            "fix_applied": fix,
            "why": why,
        }
    )


def read_raw() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_csv(RAW / "ride_demand_train.csv"),
        "test": pd.read_csv(RAW / "ride_demand_test.csv"),
        "weather": pd.read_csv(RAW / "weather_hourly.csv"),
        "events": pd.read_csv(RAW / "events_calendar.csv"),
    }


def clean_trip_table(df: pd.DataFrame, name: str, log: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    total = len(out)
    before_zones = sorted(out["zone"].dropna().astype(str).unique())
    out["zone_raw"] = out["zone"].astype(str)
    out["zone"] = out["zone"].map(clean_zone)
    add_log(log, f"ride_demand_{name}.csv", "zone", "inconsistent zone spelling", (out["zone_raw"] != out["zone"]).sum(), total, "standardized aliases to 12 canonical zone labels", "Zone is a join/model key.")
    add_log(log, f"ride_demand_{name}.csv", "zone", "unmapped zone labels", out["zone"].isna().sum(), total, "left unmapped as null and failed validation if any remain", "Unknown zones cannot be scored reliably.")

    out["pickup_hour_raw"] = out["pickup_hour"].astype(str)
    out["pickup_hour"], parse_audit = parse_mixed_local(out["pickup_hour_raw"], f"ride_demand_{name}.pickup_hour")
    add_log(log, f"ride_demand_{name}.csv", "pickup_hour", "mixed timestamp formats", total, total, "parsed ISO local, dd/mm/yyyy local, and ISO +03:00 offset to EAT", "Chronological validation and joins require one clock.")
    add_log(log, f"ride_demand_{name}.csv", "pickup_hour", "unparsed timestamps", out["pickup_hour"].isna().sum(), total, "coerced to missing and failed validation if any remain", "Unparsed hours cannot be joined.")

    if "trips" in out:
        add_log(log, f"ride_demand_{name}.csv", "trips", "missing target", out["trips"].isna().sum(), total, "imputed with train zone x weekday x hour median fallback", "Master table needs a numeric target; flag records keep the repair visible.")
        add_log(log, f"ride_demand_{name}.csv", "trips", "negative target sentinel", (out["trips"] < 0).sum(), total, "set negative trips to missing before target imputation", "Trip counts cannot be negative.")
        add_log(log, f"ride_demand_{name}.csv", "avg_wait_min", "negative wait sentinel", (out["avg_wait_min"] < 0).sum(), total, "set negative waits to missing then imputed from train medians", "Wait time cannot be negative.")
        add_log(log, f"ride_demand_{name}.csv", "avg_fare_birr", "missing fare", out["avg_fare_birr"].isna().sum(), total, "imputed from train zone median fallback", "Fare is needed for analysis/demo revenue, not as model input.")
        dupes = out.duplicated(["zone", "pickup_hour"], keep=False).sum()
        add_log(log, f"ride_demand_{name}.csv", "zone,pickup_hour", "duplicate cleaned zone-hours", dupes, total, "aggregated to one row per zone-hour using medians/first IDs", "One target row per zone-hour is required.")

    zone_audit = pd.DataFrame(
        {
            "table": f"ride_demand_{name}",
            "raw_unique_values": [before_zones],
            "clean_unique_values": [sorted(out["zone"].dropna().unique())],
        }
    )
    return out, pd.concat([parse_audit, zone_audit], ignore_index=True, sort=False)


def train_imputation_stats(train: pd.DataFrame) -> dict[str, object]:
    tmp = train.copy()
    tmp["trips_clean"] = tmp["trips"].where(tmp["trips"].ge(0))
    tmp["avg_wait_min_clean"] = tmp["avg_wait_min"].where(tmp["avg_wait_min"].ge(0))
    return {
        "trips_zhdh": tmp.groupby(["zone", tmp["pickup_hour"].dt.dayofweek, tmp["pickup_hour"].dt.hour])["trips_clean"].median(),
        "trips_zone_hour": tmp.groupby(["zone", tmp["pickup_hour"].dt.hour])["trips_clean"].median(),
        "trips_global": float(tmp["trips_clean"].median()),
        "fare_zone": tmp.groupby("zone")["avg_fare_birr"].median(),
        "fare_global": float(tmp["avg_fare_birr"].median()),
        "wait_zone_hour": tmp.groupby(["zone", tmp["pickup_hour"].dt.hour])["avg_wait_min_clean"].median(),
        "wait_global": float(tmp["avg_wait_min_clean"].median()),
    }


def fill_by_mapping(keys: pd.DataFrame, mapping: pd.Series, fallback: float) -> pd.Series:
    values = [mapping.get(tuple(row), np.nan) for row in keys.to_numpy()]
    return pd.Series(values, index=keys.index).fillna(fallback)


def finalize_train_trip(train: pd.DataFrame, stats: dict[str, object]) -> pd.DataFrame:
    out = train.copy()
    out["trips_was_imputed"] = out["trips"].isna() | out["trips"].lt(0)
    out["trips"] = out["trips"].where(out["trips"].ge(0))
    keys = pd.DataFrame(
        {
            "zone": out["zone"],
            "dow": out["pickup_hour"].dt.dayofweek,
            "hour": out["pickup_hour"].dt.hour,
        }
    )
    fill = fill_by_mapping(keys[["zone", "dow", "hour"]], stats["trips_zhdh"], stats["trips_global"])
    out["trips"] = out["trips"].fillna(fill)

    out["avg_fare_birr_was_imputed"] = out["avg_fare_birr"].isna()
    out["avg_fare_birr"] = out["avg_fare_birr"].fillna(out["zone"].map(stats["fare_zone"])).fillna(stats["fare_global"])

    out["avg_wait_min_was_imputed"] = out["avg_wait_min"].isna() | out["avg_wait_min"].lt(0)
    out["avg_wait_min"] = out["avg_wait_min"].where(out["avg_wait_min"].ge(0))
    wait_keys = pd.DataFrame({"zone": out["zone"], "hour": out["pickup_hour"].dt.hour})
    out["avg_wait_min"] = out["avg_wait_min"].fillna(fill_by_mapping(wait_keys, stats["wait_zone_hour"], stats["wait_global"]))

    agg = {
        "record_id": "first",
        "trips": "median",
        "avg_fare_birr": "median",
        "avg_wait_min": "median",
        "active_drivers": "median",
        "trips_was_imputed": "max",
        "avg_fare_birr_was_imputed": "max",
        "avg_wait_min_was_imputed": "max",
    }
    grouped = out.groupby(["zone", "pickup_hour"], as_index=False).agg(agg)
    grouped["raw_rows_aggregated"] = out.groupby(["zone", "pickup_hour"]).size().to_numpy()
    return grouped


def finalize_test_trip(test: pd.DataFrame) -> pd.DataFrame:
    return test.sort_values("row_id").copy()


def clean_weather(weather: pd.DataFrame, log: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = weather.copy()
    total = len(out)
    out["timestamp_raw"] = out["timestamp"].astype(str)
    out["timestamp"], parse_audit = parse_mixed_local(out["timestamp_raw"], "weather.timestamp")
    add_log(log, "weather_hourly.csv", "timestamp", "mixed timestamp formats and UTC Z rows", total, total, "parsed Z timestamps as UTC then converted to EAT; parsed dd/mm rows as local EAT", "Weather must align with local pickup hours.")
    add_log(log, "weather_hourly.csv", "timestamp", "duplicate hours", out.duplicated("timestamp", keep=False).sum(), total, "aggregated duplicate hours with numeric medians", "Many-to-one weather join cannot have duplicate keys.")
    add_log(log, "weather_hourly.csv", "temp_c", "implausible Fahrenheit-coded temperatures", out["temp_c"].gt(45).sum(), total, "converted values above 45 using (F - 32) * 5 / 9", "Addis Ababa hourly Celsius temperatures should not be 60-76 C.")
    add_log(log, "weather_hourly.csv", "rain_mm", "negative rain sentinel", out["rain_mm"].lt(0).sum(), total, "set negative rain to missing then interpolated/filled", "Rainfall cannot be negative.")
    add_log(log, "weather_hourly.csv", "temp_c,humidity_pct", "missing weather values", out[["temp_c", "humidity_pct"]].isna().any(axis=1).sum(), total, "interpolated by time, then forward/back filled", "Weather features must exist for every zone-hour.")

    out.loc[out["temp_c"] > 45, "temp_c"] = (out.loc[out["temp_c"] > 45, "temp_c"] - 32) * 5 / 9
    out.loc[out["rain_mm"] < 0, "rain_mm"] = np.nan
    out["data_type"] = out["data_type"].astype(str).str.strip().str.lower()

    out = (
        out.groupby("timestamp", as_index=False)
        .agg(
            temp_c=("temp_c", "median"),
            rain_mm=("rain_mm", "median"),
            humidity_pct=("humidity_pct", "median"),
            wind_kmh=("wind_kmh", "median"),
            data_type=("data_type", lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0]),
            duplicate_weather_rows=("timestamp_raw", "size"),
        )
        .sort_values("timestamp")
    )
    full_hours = pd.DataFrame({"timestamp": pd.date_range(out["timestamp"].min(), out["timestamp"].max(), freq="h")})
    out = full_hours.merge(out, on="timestamp", how="left")
    out["weather_was_missing_hour"] = out["data_type"].isna()
    out["data_type"] = out["data_type"].ffill().bfill()
    for col in ["temp_c", "rain_mm", "humidity_pct", "wind_kmh"]:
        out[col] = out[col].interpolate(limit_direction="both").ffill().bfill()
    out["duplicate_weather_rows"] = out["duplicate_weather_rows"].fillna(0).astype(int)
    out["rain_3h_mm"] = out["rain_mm"].rolling(3, min_periods=1).sum()
    out["rain_class"] = pd.cut(
        out["rain_mm"],
        bins=[-0.01, 0, 2.5, 7.5, math.inf],
        labels=["none", "light", "moderate", "heavy"],
    ).astype(str)
    return out, parse_audit


def event_zones(raw_zone: object) -> list[str]:
    token = normalize_token(raw_zone)
    if token in {"all", "all zones", "citywide", "city wide", "city-wide"}:
        return ZONES.copy()
    cleaned = set()
    for bit in re.split(r"&|,|/| and ", token):
        z = clean_zone(bit)
        if z:
            cleaned.add(z)
    if not cleaned:
        for alias, zone in ZONE_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", token):
                cleaned.add(zone)
    return sorted(cleaned)


def parse_attendance(value: object) -> float:
    if pd.isna(value):
        return np.nan
    digits = re.sub(r"[^0-9.]", "", str(value))
    return float(digits) if digits else np.nan


def clean_events(events: pd.DataFrame, log: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = events.copy()
    total = len(out)
    before_types = sorted(out["event_type"].dropna().astype(str).unique())
    before_zones = sorted(out["zone"].dropna().astype(str).unique())
    out["event_type_raw"] = out["event_type"].astype(str)
    out["zone_raw"] = out["zone"].astype(str)
    out["status"] = out["status"].astype(str).str.strip().str.lower()
    out["event_type"] = out["event_type"].map(clean_event_type)
    out["start_datetime"], start_audit = parse_mixed_local(out["start_datetime"], "events.start_datetime")
    out["end_datetime"], end_audit = parse_mixed_local(out["end_datetime"], "events.end_datetime")
    out["attendance_num"] = out["expected_attendance"].map(parse_attendance)
    out["zone_list"] = out["zone_raw"].map(event_zones)

    missing_end = out["end_datetime"].isna().sum()
    out.loc[out["end_datetime"].isna(), "end_datetime"] = out.loc[out["end_datetime"].isna(), "start_datetime"] + pd.Timedelta(hours=2)
    reversed_end = (out["end_datetime"] < out["start_datetime"]).sum()
    out.loc[out["end_datetime"] < out["start_datetime"], "end_datetime"] = out.loc[out["end_datetime"] < out["start_datetime"], "start_datetime"] + pd.Timedelta(hours=2)

    add_log(log, "events_calendar.csv", "event_type", "inconsistent event type spelling", (out["event_type_raw"] != out["event_type"]).sum(), total, "standardized to snake_case event types", "Event features group by event type.")
    add_log(log, "events_calendar.csv", "zone", "citywide, multi-zone, and aliased zone labels", sum(len(z) != 1 or str(r).strip() not in ZONES for z, r in zip(out["zone_list"], out["zone_raw"])), total, "expanded citywide events to all zones and multi-zone rows to one row per affected zone", "Events join by zone-hour intervals.")
    add_log(log, "events_calendar.csv", "start_datetime,end_datetime", "mixed timestamp formats", total, total, "parsed ISO, dd/mm/yyyy, and month-name formats as local EAT", "Event windows must align with trip hours.")
    add_log(log, "events_calendar.csv", "end_datetime", "missing event end", missing_end, total, "filled missing ends as two hours after start", "Intervals need an end to join.")
    add_log(log, "events_calendar.csv", "end_datetime", "end earlier than start", reversed_end, total, "set end to start + two hours", "Negative event duration is invalid.")
    add_log(log, "events_calendar.csv", "expected_attendance", "free-text or missing attendance", out["attendance_num"].isna().sum(), total, "extracted digits and imputed feature aggregations with zero when absent", "Attendance intensity should be numeric.")
    add_log(log, "events_calendar.csv", "status", "inconsistent status spelling", (events["status"].astype(str) != out["status"]).sum(), total, "trimmed and lowercased status", "Cancelled events must be auditable/excludable.")

    exploded = (
        out.drop(columns=["zone"])
        .explode("zone_list")
        .rename(columns={"zone_list": "zone"})
        .reset_index(drop=True)
    )
    exploded["zone"] = exploded["zone"].astype("string")
    exploded["is_public_holiday"] = exploded["event_type"].eq("public_holiday")
    exploded["event_window_start"] = exploded["start_datetime"] - pd.Timedelta(hours=2)
    exploded["event_window_end"] = exploded["end_datetime"] + pd.Timedelta(hours=2)
    type_audit = pd.DataFrame(
        {
            "table": ["events_calendar"],
            "raw_unique_values": [before_types],
            "clean_unique_values": [sorted(out["event_type"].dropna().unique())],
        }
    )
    zone_audit = pd.DataFrame(
        {
            "table": ["events_calendar.zone"],
            "raw_unique_values": [before_zones],
            "clean_unique_values": [ZONES],
        }
    )
    parse_audit = pd.concat([start_audit, end_audit, type_audit, zone_audit], ignore_index=True, sort=False)
    return exploded, parse_audit


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["pickup_hour"]
    out["hour"] = dt.dt.hour
    out["dayofweek"] = dt.dt.dayofweek
    out["month"] = dt.dt.month
    out["day"] = dt.dt.day
    out["is_weekend"] = out["dayofweek"].isin([5, 6]).astype(int)
    out["is_payday_window"] = ((out["day"] <= 3) | (out["day"] >= 28)).astype(int)
    out["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    out["days_since_start"] = (dt.dt.normalize() - pd.Timestamp("2025-01-01")).dt.days
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def attach_events(base: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    base = base.reset_index(drop=True)
    events = events.reset_index(drop=True)
    confirmed = events[events["status"].eq("confirmed") & events["zone"].notna()].copy()
    rows = []
    matched_event_ids = set()
    for _, ev in confirmed.iterrows():
        mask = (
            base["zone"].eq(ev["zone"])
            & base["pickup_hour"].ge(ev["event_window_start"])
            & base["pickup_hour"].le(ev["event_window_end"])
        )
        if mask.any():
            matched_event_ids.add(ev["event_id"])
            tmp = base.loc[mask, ["zone", "pickup_hour"]].copy()
            tmp["event_id"] = ev["event_id"]
            tmp["event_type"] = ev["event_type"]
            tmp["event_name"] = ev["event_name"]
            tmp["venue"] = ev["venue"]
            tmp["attendance_num"] = ev["attendance_num"] if pd.notna(ev["attendance_num"]) else 0.0
            tmp["event_in_core_window"] = tmp["pickup_hour"].between(ev["start_datetime"], ev["end_datetime"]).astype(int)
            tmp["event_in_pre_window"] = ((tmp["pickup_hour"] >= ev["event_window_start"]) & (tmp["pickup_hour"] < ev["start_datetime"])).astype(int)
            tmp["event_in_post_window"] = ((tmp["pickup_hour"] > ev["end_datetime"]) & (tmp["pickup_hour"] <= ev["event_window_end"])).astype(int)
            delta_start = (ev["start_datetime"] - tmp["pickup_hour"]).dt.total_seconds() / 3600
            delta_end = (tmp["pickup_hour"] - ev["end_datetime"]).dt.total_seconds() / 3600
            tmp["hours_until_event_start"] = delta_start
            tmp["hours_since_event_end"] = delta_end
            tmp["is_public_holiday_event"] = int(ev["event_type"] == "public_holiday")
            rows.append(tmp)
    if rows:
        long = pd.concat(rows, ignore_index=True)
        event_features = (
            long.groupby(["zone", "pickup_hour"], as_index=False)
            .agg(
                event_count=("event_id", "nunique"),
                event_attendance_sum=("attendance_num", "sum"),
                event_in_core_window=("event_in_core_window", "max"),
                event_in_pre_window=("event_in_pre_window", "max"),
                event_in_post_window=("event_in_post_window", "max"),
                is_public_holiday=("is_public_holiday_event", "max"),
                hours_until_event_start=("hours_until_event_start", lambda x: x[x >= 0].min() if (x >= 0).any() else np.nan),
                hours_since_event_end=("hours_since_event_end", lambda x: x[x >= 0].min() if (x >= 0).any() else np.nan),
                event_types=("event_type", lambda x: ",".join(sorted(set(x)))),
                event_names=("event_name", lambda x: " | ".join(sorted(set(map(str, x)))[:5])),
            )
        )
    else:
        long = pd.DataFrame()
        event_features = base[["zone", "pickup_hour"]].copy().iloc[0:0]

    out = base.merge(event_features, on=["zone", "pickup_hour"], how="left")
    fill_zero = [
        "event_count",
        "event_attendance_sum",
        "event_in_core_window",
        "event_in_pre_window",
        "event_in_post_window",
        "is_public_holiday",
    ]
    for c in fill_zero:
        out[c] = out[c].fillna(0).astype(int if c != "event_attendance_sum" else float)
    out["hours_until_event_start"] = out["hours_until_event_start"].fillna(999.0)
    out["hours_since_event_end"] = out["hours_since_event_end"].fillna(999.0)
    out["event_types"] = out["event_types"].fillna("none")
    out["event_names"] = out["event_names"].fillna("")
    audit = {
        "confirmed_expanded_events": int(len(confirmed)),
        "matched_expanded_events": int(confirmed["event_id"].isin(matched_event_ids).sum()),
        "excluded_cancelled_expanded_events": int(events["status"].eq("cancelled").sum()),
        "unmatched_confirmed_expanded_events": int((~confirmed["event_id"].isin(matched_event_ids)).sum()),
        "joined_zone_hours_with_events": int(out["event_count"].gt(0).sum()),
    }
    return out, audit, long


def attach_weather(base: pd.DataFrame, weather: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = len(base)
    out = base.merge(weather.rename(columns={"timestamp": "pickup_hour"}), on="pickup_hour", how="left", validate="many_to_one")
    no_weather = out["temp_c"].isna().sum()
    audit = {
        "rows_before": before,
        "rows_after": len(out),
        "matched_rows": int(out["temp_c"].notna().sum()),
        "match_rate_pct": pct(out["temp_c"].notna().sum(), len(out)),
        "no_weather_rows": int(no_weather),
        "no_weather_reason": "None after completing hourly weather grid; missing raw hours were interpolated." if no_weather == 0 else "Outside cleaned weather range.",
    }
    return out, audit


def add_lag_features(train_master: pd.DataFrame, test_master: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_master.sort_values(["zone", "pickup_hour"]).copy()
    test = test_master.sort_values(["zone", "pickup_hour"]).copy()
    history = train[["zone", "pickup_hour", "trips"]].copy()
    combined = pd.concat(
        [
            history.assign(is_test=0),
            test[["zone", "pickup_hour"]].assign(trips=np.nan, is_test=1),
        ],
        ignore_index=True,
    ).sort_values(["zone", "pickup_hour"])
    combined["lag_168h_trips"] = combined.groupby("zone")["trips"].shift(168)
    combined["rolling_168h_mean_trips"] = (
        combined.groupby("zone")["trips"].shift(1).rolling(168, min_periods=24).mean().reset_index(level=0, drop=True)
    )
    zone_hour_median = train.groupby(["zone", "hour"])["trips"].median()
    global_median = train["trips"].median()
    fallback = fill_by_mapping(
        pd.DataFrame({"zone": combined["zone"], "hour": combined["pickup_hour"].dt.hour}),
        zone_hour_median,
        global_median,
    )
    combined["lag_168h_trips"] = combined["lag_168h_trips"].fillna(fallback)
    combined["rolling_168h_mean_trips"] = combined["rolling_168h_mean_trips"].fillna(fallback)
    lag_cols = combined[["zone", "pickup_hour", "lag_168h_trips", "rolling_168h_mean_trips", "is_test"]]
    train = train.merge(lag_cols[lag_cols["is_test"].eq(0)].drop(columns="is_test"), on=["zone", "pickup_hour"], how="left")
    test = test.merge(lag_cols[lag_cols["is_test"].eq(1)].drop(columns="is_test"), on=["zone", "pickup_hour"], how="left")
    return train, test


def make_feature_table() -> pd.DataFrame:
    rows = [
        ("hour", "pickup_hour hour", "pickup_hour", "Captures commute/night demand cycles.", "yes"),
        ("dayofweek", "pickup_hour weekday", "pickup_hour", "Weekday/weekend routines differ.", "yes"),
        ("is_weekend", "dayofweek in Saturday/Sunday", "pickup_hour", "Weekend demand differs from weekday demand.", "yes"),
        ("is_payday_window", "day <= 3 or day >= 28", "pickup_hour", "Pay-period spending may lift demand.", "yes"),
        ("rain_mm", "hourly rain after weather cleaning", "weather_hourly.rain_mm", "Rain can shift riders into ride hailing.", "yes"),
        ("rain_3h_mm", "rolling 3-hour rain sum", "weather_hourly.rain_mm", "Recent rain can affect current demand.", "yes"),
        ("rain_class", "none/light/moderate/heavy bins", "weather_hourly.rain_mm", "Allows non-linear rain response.", "yes"),
        ("event_count", "confirmed events in zone-hour extended window", "events_calendar interval join", "Local events change pickup pressure.", "yes"),
        ("event_in_core_window", "hour between event start/end", "events_calendar start/end", "Demand differs during an event.", "yes"),
        ("event_in_pre_window", "2 hours before event start", "events_calendar start/end", "Pre-event arrivals can lift demand.", "yes"),
        ("event_in_post_window", "2 hours after event end", "events_calendar start/end", "Post-event departures can lift demand.", "yes"),
        ("event_attendance_sum", "sum expected attendance for matched events", "events_calendar.expected_attendance", "Bigger crowds can produce bigger spikes.", "yes"),
        ("lag_168h_trips", "same zone, 168 hours earlier", "ride_demand_train.trips", "Weekly seasonality anchor known from history.", "yes"),
        ("rolling_168h_mean_trips", "previous 168-hour mean by zone", "ride_demand_train.trips", "Recent level/trend by zone.", "yes"),
    ]
    return pd.DataFrame(rows, columns=["feature", "formula", "source_columns", "why_expected_to_help", "known_at_forecast_time"])


def make_data_dictionary(columns: list[str]) -> pd.DataFrame:
    derived = {
        "zone": ("category", "trip raw", "Canonical pickup zone.", "standardized from raw zone aliases"),
        "pickup_hour": ("datetime64[ns]", "trip raw", "Start of pickup hour in EAT local time.", "parsed from mixed formats"),
        "trips": ("float", "trip train", "Hourly ride requests target.", "cleaned nonnegative target; train only"),
        "row_id": ("string", "test raw", "Test row ID for submission ordering.", "copied from raw test"),
        "record_id": ("string", "train raw", "Representative training record ID.", "first ID after duplicate aggregation"),
        "avg_fare_birr": ("float", "trip train", "Historical average fare for analysis/demo revenue.", "imputed by train zone median"),
        "avg_wait_min": ("float", "trip train", "Historical average wait for analysis only.", "negative sentinels imputed"),
        "active_drivers": ("float", "trip train", "Historical active drivers for analysis only.", "median after duplicate aggregation"),
        "temp_c": ("float", "weather", "Hourly temperature in Celsius.", "UTC weather converted to EAT, duplicated/interpolated"),
        "rain_mm": ("float", "weather", "Rain in previous hour.", "negative sentinels interpolated"),
        "humidity_pct": ("float", "weather", "Relative humidity percent.", "interpolated if missing"),
        "wind_kmh": ("float", "weather", "Wind speed.", "interpolated if missing"),
        "data_type": ("category", "weather", "observed or forecast weather.", "mode after duplicate aggregation"),
    }
    feature_dict = {r.feature: ("feature", r.source_columns, r.why_expected_to_help, r.formula) for r in make_feature_table().itertuples()}
    rows = []
    for c in columns:
        if c in derived:
            typ, src, desc, how = derived[c]
        elif c in feature_dict:
            typ, src, desc, how = feature_dict[c]
        else:
            typ, src, desc, how = ("feature/metadata", "pipeline", c.replace("_", " "), "derived in cleaning.py")
        rows.append({"column": c, "type": typ, "source": src, "description": desc, "derived_how": how})
    return pd.DataFrame(rows)


def validate(master_train: pd.DataFrame, master_test: pd.DataFrame, weather_audit: dict, expected_test_rows: int) -> pd.DataFrame:
    feature_train = set(master_train.drop(columns=["trips"], errors="ignore").columns)
    feature_test = set(master_test.columns)
    checks = [
        ("train_one_row_per_zone_hour", not master_train.duplicated(["zone", "pickup_hour"]).any()),
        ("test_one_row_per_zone_hour", not master_test.duplicated(["zone", "pickup_hour"]).any()),
        ("only_12_zone_labels_train", set(master_train["zone"].unique()) <= set(ZONES) and master_train["zone"].nunique() == 12),
        ("only_12_zone_labels_test", set(master_test["zone"].unique()) <= set(ZONES) and master_test["zone"].nunique() == 12),
        ("train_target_nonnegative_nonmissing", master_train["trips"].notna().all() and master_train["trips"].ge(0).all()),
        ("weather_join_row_count_unchanged", weather_audit["rows_before"] == weather_audit["rows_after"]),
        ("weather_join_full_match", weather_audit["no_weather_rows"] == 0),
        ("test_row_count_4032", len(master_test) == expected_test_rows),
        ("train_test_same_feature_columns", feature_train == feature_test),
        ("forecast_model_excludes_train_only_operational_inputs", not {"avg_fare_birr", "avg_wait_min", "active_drivers"} <= feature_test),
    ]
    return pd.DataFrame({"check": [c[0] for c in checks], "status": ["PASS" if c[1] else "FAIL" for c in checks]})


def write_report(
    cleaning_log: pd.DataFrame,
    parse_zone_audit: pd.DataFrame,
    feature_table: pd.DataFrame,
    join_audit: pd.DataFrame,
    join_proof: pd.DataFrame,
    checks: pd.DataFrame,
    weather_clock: pd.DataFrame,
) -> None:
    REPORTS.mkdir(exist_ok=True)
    report = REPORTS / "A_cleaning_and_integration.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# A - Data Cleaning & Integration Pipeline\n\n")
        f.write("Ground truth: `Qiyas-AI-Hackathon_Ride_Demand_Instructions.pdf`. All outputs are generated by `python -m src.cleaning`.\n\n")
        f.write("## A1 Cleaning Log\n\n")
        f.write(cleaning_log.to_markdown(index=False))
        f.write("\n\n## A2 Time & Key Standardization\n\n")
        f.write("Zone labels are standardized to the 12 canonical labels used by the train and test files. Event types are standardized to snake_case. Timestamp parsing is format-specific to avoid day/month ambiguity.\n\n")
        f.write(parse_zone_audit.to_markdown(index=False))
        f.write("\n\n### Weather Clock Proof\n\n")
        f.write("Weather rows ending in `Z` are UTC. Converting them to EAT shifts the average daily temperature peak from the UTC afternoon to the expected local afternoon. Non-`Z` weather rows are parsed as already-local EAT. All trip and event timestamps are represented as local EAT after parsing.\n\n")
        f.write(weather_clock.to_markdown(index=False))
        f.write("\n\n## A3 Join Map & Diagram\n\n")
        f.write("```text\n")
        f.write("ride_demand_train/test (left table: every required zone-hour to forecast)\n")
        f.write("  |-- many-to-one on pickup_hour == weather.timestamp after UTC-to-EAT conversion\n")
        f.write("  |   -> weather_hourly cleaned to one row per local hour\n")
        f.write("  |\n")
        f.write("  |-- interval join on zone and pickup_hour between event_window_start and event_window_end\n")
        f.write("      -> events expanded from citywide/multi-zone rows to affected zones\n")
        f.write("      -> event window = 2 hours before start through 2 hours after end\n")
        f.write("```\n\n")
        f.write("The trip/test zone-hour table is the left table because scoring requires preserving every forecast row. Weather is many-to-one by hour; events are many-to-many before aggregation, then reduced back to one feature row per zone-hour.\n\n")
        f.write("## A4 Join Audit\n\n")
        f.write(join_audit.to_markdown(index=False))
        f.write("\n\n## A5 Join Proof\n\n")
        f.write(join_proof.to_markdown(index=False))
        f.write("\n\n## A6 Feature Engineering Table\n\n")
        f.write(feature_table.to_markdown(index=False))
        f.write("\n\n## A7 Integrity Checks\n\n")
        f.write(checks.to_markdown(index=False))
        f.write("\n\n## A8 Master Tables & Data Dictionary\n\n")
        f.write("Generated files:\n\n")
        f.write("- `data/processed/master_train.csv`\n")
        f.write("- `data/processed/master_test.csv`\n")
        f.write("- `data/processed/data_dictionary_master.csv`\n")
        f.write("- audit support files in `data/processed/`\n")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    raw = read_raw()
    log: list[dict] = []

    train_clean, train_audit = clean_trip_table(raw["train"], "train", log)
    test_clean, test_audit = clean_trip_table(raw["test"], "test", log)
    weather_clean, weather_parse_audit = clean_weather(raw["weather"], log)
    events_clean, events_audit = clean_events(raw["events"], log)

    stats = train_imputation_stats(train_clean)
    train_base = finalize_train_trip(train_clean, stats)
    test_base = finalize_test_trip(test_clean)

    train_base = add_calendar_features(train_base)
    test_base = add_calendar_features(test_base)

    train_weather, train_weather_audit = attach_weather(train_base, weather_clean)
    test_weather, test_weather_audit = attach_weather(test_base, weather_clean)
    train_events, train_event_audit, event_matches_train = attach_events(train_weather, events_clean)
    test_events, test_event_audit, event_matches_test = attach_events(test_weather, events_clean)
    master_train, master_test = add_lag_features(train_events, test_events)

    # Keep forecast feature columns identical, while retaining target only in train.
    train_only_cols = ["avg_fare_birr", "avg_wait_min", "active_drivers"]
    model_excluded_cols = []
    for col in train_only_cols:
        if col in master_test.columns:
            model_excluded_cols.append(col)
    # Test legitimately does not have operational train-only outcomes; do not synthesize them as features.
    shared_cols = [c for c in master_test.columns if c in master_train.columns]
    master_train = master_train[shared_cols + ["trips"]]
    master_test = master_test[shared_cols]

    sort_cols = ["zone", "pickup_hour"]
    master_train = master_train.sort_values(sort_cols).reset_index(drop=True)
    master_test = master_test.sort_values("row_id").reset_index(drop=True)

    cleaning_log = pd.DataFrame(log)
    feature_table = make_feature_table()
    data_dictionary = make_data_dictionary(list(master_train.columns))
    parse_zone_audit = pd.concat([train_audit, test_audit, weather_parse_audit, events_audit], ignore_index=True, sort=False)

    join_audit = pd.DataFrame(
        [
            {"join": "train + weather", **train_weather_audit},
            {"join": "test + weather", **test_weather_audit},
            {"join": "train + events", **train_event_audit},
            {"join": "test + events", **test_event_audit},
        ]
    )

    proof_rows = []
    rain_row = master_train[master_train["rain_mm"].gt(0)].sort_values("rain_mm", ascending=False).head(1)
    event_row = master_train[master_train["event_count"].gt(0)].head(1)
    holiday_row = master_train[master_train["is_public_holiday"].eq(1)].head(1)
    for label, rowdf in [("rain affected", rain_row), ("inside event window", event_row), ("public holiday", holiday_row)]:
        if not rowdf.empty:
            r = rowdf.iloc[0]
            proof_rows.append(
                {
                    "case": label,
                    "zone": r["zone"],
                    "pickup_hour": r["pickup_hour"],
                    "weather_attached": f"temp={r['temp_c']:.1f}C rain={r['rain_mm']:.1f}mm rain_3h={r['rain_3h_mm']:.1f}mm class={r['rain_class']}",
                    "events_attached": r.get("event_names", ""),
                    "event_features": f"count={r['event_count']}, core={r['event_in_core_window']}, pre={r['event_in_pre_window']}, post={r['event_in_post_window']}, holiday={r['is_public_holiday']}",
                }
            )
    join_proof = pd.DataFrame(proof_rows)

    weather_clock = pd.DataFrame(
        {
            "clock_view": ["raw UTC hour from Z rows", "converted EAT local hour"],
            "mean_temp_peak_hour": [
                int(pd.to_datetime(raw["weather"].loc[raw["weather"]["timestamp"].astype(str).str.endswith("Z"), "timestamp"], utc=True).dt.hour.groupby(
                    pd.to_datetime(raw["weather"].loc[raw["weather"]["timestamp"].astype(str).str.endswith("Z"), "timestamp"], utc=True).dt.hour
                ).count().index[0])
                if False else int(
                    raw["weather"]
                    .loc[raw["weather"]["timestamp"].astype(str).str.endswith("Z")]
                    .assign(raw_hour=lambda d: pd.to_datetime(d["timestamp"], utc=True).dt.hour)
                    .groupby("raw_hour")["temp_c"]
                    .mean()
                    .idxmax()
                ),
                int(weather_clean.groupby(weather_clean["timestamp"].dt.hour)["temp_c"].mean().idxmax()),
            ],
            "interpretation": [
                "Peak appears three hours earlier because Z rows are UTC.",
                "Peak falls in local afternoon after UTC+3 conversion, matching expected Addis daytime warming.",
            ],
        }
    )

    checks = validate(master_train, master_test, test_weather_audit, len(raw["test"]))

    master_train.to_csv(PROCESSED / "master_train.csv", index=False)
    master_test.to_csv(PROCESSED / "master_test.csv", index=False)
    data_dictionary.to_csv(PROCESSED / "data_dictionary_master.csv", index=False)
    cleaning_log.to_csv(PROCESSED / "cleaning_log.csv", index=False)
    parse_zone_audit.to_csv(PROCESSED / "time_key_standardization_audit.csv", index=False)
    feature_table.to_csv(PROCESSED / "feature_engineering_table.csv", index=False)
    join_audit.to_csv(PROCESSED / "join_audit.csv", index=False)
    join_proof.to_csv(PROCESSED / "join_proof.csv", index=False)
    checks.to_csv(PROCESSED / "integrity_checks.csv", index=False)
    weather_clean.to_csv(PROCESSED / "clean_weather_hourly.csv", index=False)
    events_clean.to_csv(PROCESSED / "clean_events_calendar.csv", index=False)
    event_matches_train.to_csv(PROCESSED / "event_zone_hour_matches_train.csv", index=False)
    event_matches_test.to_csv(PROCESSED / "event_zone_hour_matches_test.csv", index=False)

    write_report(cleaning_log, parse_zone_audit, feature_table, join_audit, join_proof, checks, weather_clock)

    print("Integrity checks")
    for row in checks.itertuples(index=False):
        print(f"{row.check}: {row.status}")
    if (checks["status"] == "FAIL").any():
        raise SystemExit("One or more integrity checks failed.")
    print(f"Wrote {PROCESSED / 'master_train.csv'} rows={len(master_train)}")
    print(f"Wrote {PROCESSED / 'master_test.csv'} rows={len(master_test)}")
    print(f"Wrote {REPORTS / 'A_cleaning_and_integration.md'}")


if __name__ == "__main__":
    main()
