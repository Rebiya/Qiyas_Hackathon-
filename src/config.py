from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
REPORTS, TABLES, FIGS = ROOT / "reports", ROOT / "reports" / "tables", ROOT / "figures"
for _p in (PROC, REPORTS, TABLES, FIGS):
    _p.mkdir(parents=True, exist_ok=True)

FILES = {"train": "ride_demand_train.csv", "test": "ride_demand_test.csv",
         "template": "submission_template.csv", "weather": "weather_hourly.csv",
         "events": "events_calendar.csv"}

SEED = 42
UTC_OFFSET_H = 3                       # Africa/Addis_Ababa = UTC+3, no DST
N_ZONES = 12
GRID_START = pd.Timestamp("2025-01-01 00:00")
TRAIN_END = pd.Timestamp("2025-10-31 23:00")
TEST_START = pd.Timestamp("2025-11-01 00:00")
TEST_END = pd.Timestamp("2025-11-14 23:00")
WX_START = GRID_START - pd.Timedelta(days=1)   # extra day so 24h rain sums exist on 1 Jan

NA_TOKENS = {"", "nan", "na", "n/a", "null", "none", "-", "--", "?", "missing", "unknown", "tbd", "nil", "#n/a"}
SENTINELS = {-999, -9999, -99, 999, 9999, 99999}
RANGES = {"trips": (0, 2000), "avg_fare_birr": (10, 3000), "avg_wait_min": (0, 60),
          "active_drivers": (0, 500), "temp_c": (-5, 40), "rain_mm": (0, 150),
          "humidity_pct": (0, 100), "wind_kmh": (0, 150)}
TRIPS_ERROR_MULT = 5          # trips > 5 x zone's 99.5th pct = data error (real event spikes are kept)
LATE_LAUNCH_AFTER_DAYS = 7    # zone whose first row is > 7 days after 1 Jan = late launch
OUTAGE_MISSING_FRAC = 0.9     # >=90% of operating zones missing in an hour = platform outage

# After you inspect the printed zone counts you may pin these (normalised keys, lowercase, no spaces)
CANONICAL_ZONES = None        # e.g. ["Bole","Piassa",...]  -> forces the 12 labels
ZONE_ALIASES = {}             # e.g. {"kazanchs": "kazanchis"}

EVENT_TYPES = ["public_holiday", "school_break", "football_match", "concert",
               "conference", "exhibition", "road_closure", "sports_run"]
POINT_TYPES = ["football_match", "concert", "conference", "exhibition", "sports_run"]
# (hours before start, hours after end) that an event reaches into. Revisit after B3.3.
EVENT_WINDOWS = {"football_match": (2, 3), "concert": (2, 3), "conference": (1, 1),
                 "exhibition": (1, 1), "sports_run": (1, 2), "road_closure": (0, 0),
                 "public_holiday": (0, 0), "school_break": (0, 0)}
ACTIVE_STATUSES = {"confirmed", "unknown"}       # cancelled / postponed are not treated as happening
PAYDAY_START_DAY, PAYDAY_END_DAY = 25, 3         # tested in B4.3
OPS_COLS = ["avg_fare_birr", "avg_wait_min", "active_drivers"]   # train-only -> never model inputs


def save_table(df, name):
    df.to_csv(TABLES / f"{name}.csv", index=False)
    return df