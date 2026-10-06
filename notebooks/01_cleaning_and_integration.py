# %% [markdown]

# %%
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Markdown
from src.config import *
from src.cleaning import *
from src.features import *
from src.validate import validate
pd.set_option("display.max_columns", 60, "display.width", 200, "display.max_colwidth", 80)

def rd(k): return pd.read_csv(RAW / FILES[k], dtype=str, keep_default_na=False)
raw_train, raw_test, raw_tmpl, raw_wx, raw_ev = [rd(k) for k in ["train", "test", "template", "weather", "events"]]
print({k: v.shape for k, v in dict(train=raw_train, test=raw_test, template=raw_tmpl, weather=raw_wx, events=raw_ev).items()})

# %% [markdown]
# ## Raw health check (feeds fig01 later)
# %%
def raw_missing(df, name):
    nm = df.apply(lambda s: s.astype(str).str.strip().str.lower().isin(NA_TOKENS))
    return pd.DataFrame({"table": name, "column": df.columns, "blank_or_NA": nm.sum().values, "pct": (100 * nm.mean()).round(2).values})
raw_miss = pd.concat([raw_missing(d, n) for d, n in [(raw_train, "train"), (raw_wx, "weather"), (raw_ev, "events")]])
display(save_table(raw_miss, "A_raw_missingness"))

# %% [markdown]
# ## A1 – Cleaning log
# %%
log = CleaningLog()
zm = ZoneMapper().fit(raw_train["zone"])
print("Normalised zone keys by row count (the 12 real zones should clearly outnumber typos):")
display(zm.key_counts.head(16))
trips = clean_trips(raw_train, zm, log, "ride_demand_train.csv")
test = clean_trips(raw_test, zm, log, "ride_demand_test.csv", is_test=True)
wx, wx_evidence = clean_weather(raw_wx, log)
ev = clean_events(raw_ev, zm, log)
grid, first_obs, late, launch_eff = build_grid(trips)
zones = sorted(grid["zone"].unique()); assert len(zones) == N_ZONES
cl = log.df()
display(save_table(cl, "A1_cleaning_log"))
print(cl.groupby("file").apply(lambda d: (d.rows_affected > 0).sum(), include_groups=False).rename("issues with >0 rows"))

# %% [markdown]
# ## A2 – Time & key standardisation
# ### A2a – Unique values before / after
# %%
def ub(raw, mapper, name):
    t = raw.value_counts().rename_axis("raw_value").reset_index(name="rows")
    t["cleaned"] = t["raw_value"].map(mapper); t.insert(0, "table", name); return t
cell = lambda v: ", ".join(zm.map_cell(v)[0]) + (" [citywide]" if zm.map_cell(v)[1] else "")
zt = pd.concat([ub(raw_train["zone"], zm.map_one, "trips_train"), ub(raw_test["zone"], zm.map_one, "trips_test"), ub(raw_ev["zone"], cell, "events")])
display(save_table(zt, "A2a_zone_before_after"))
print("unique zone values before:", zt.groupby("table").raw_value.nunique().to_dict())
print("unique zone labels after :", sorted(zm.labels.values()))
et = ub(raw_ev["event_type"], norm_event_type, "events")
display(save_table(et, "A2a_event_type_before_after"))

# %% [markdown]
# ### A2b – How every timestamp format was parsed
# %%
for k, v in log.parse.items():
    print("\n==", k); display(v)
sl = raw_train["pickup_hour"].str.contains("/")
chk = pd.DataFrame({"ISO rows": trips_month if False else None}, index=[0]) if False else None
t_all = pd.to_datetime(raw_train["pickup_hour"].where(~sl), errors="coerce")
t_sl = trips.loc[trips.record_id.isin(raw_train.loc[sl, "record_id"])]
t_iso = trips.loc[trips.record_id.isin(raw_train.loc[~sl, "record_id"])]
mdist = pd.DataFrame({"slash_rows_by_month": t_sl.hour_eat.dt.month.value_counts(normalize=True).sort_index(),
                      "iso_rows_by_month": t_iso.hour_eat.dt.month.value_counts(normalize=True).sort_index()}).round(3)
print("Evidence that dd/mm was read correctly: slash rows must spread over Jan-Oct like ISO rows (not pile up on days<=12 of wrong months)")
display(save_table(mdist.reset_index(), "A2b_slash_vs_iso_month_distribution"))
print("events: rows with end < start after parsing ->", int((ev.end < ev.start).sum()), "(must be 0)")

# %% [markdown]
# ### A2c – Which clock is each table on?
# %%
print("TRIPS: stated as EAT. Evidence – demand should bottom out at night and peak in the morning/evening LOCAL hours:")
hp = trips.groupby(trips.hour_eat.dt.hour)["trips"].mean().round(2).rename("mean_trips_by_EAT_hour")
display(hp.to_frame().T)
print("WEATHER:", wx_evidence["decision"])
print("Mean temperature by hour (peak = mid-afternoon local; compare tz-marked vs unmarked rows):")
display(wx_evidence["profiles"].T)
save_table(wx_evidence["profiles"].reset_index(), "A2c_weather_clock_evidence")
print("Conversion applied: tz-marked rows: UTC -> EAT (+3 h). Unmarked rows: see decision above. EVENTS: local time, no conversion.")
print("Unit diagnostics per month (look for a month that is ~1/3, x3.6, x25 or x1/100 off):")
display(wx_evidence["unit_diag"])

# %% [markdown]
# ## A3 – Join map & diagram
# %%
fig, ax = plt.subplots(figsize=(14, 6)); ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 6)
def box(x, y, w, h, t, c): ax.add_patch(plt.Rectangle((x, y), w, h, fc=c, ec="k")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=9)
def arr(a, b, t): ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.5)); ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + .15, t, ha="center", fontsize=8)
box(5, 2.2, 4, 1.8, "LEFT: zone-hour grid\n(train trips + test rows)\nkey = (zone, hour_eat)\n12 zones x hourly", "#cfe3f3")
box(5, 4.7, 4, 1.1, "weather_hourly (cleaned, EAT)\nkey = hour_eat, 1 row/hour", "#fbe3b6")
box(5, 0.2, 4, 1.4, "events_calendar (cleaned)\ninterval [start-pre, end+post]\nexploded to (zone, hour)", "#cde8d8")
box(10.3, 2.2, 3.4, 1.8, "master_train / master_test\n(one row per zone-hour)", "#e6d6ee")
arr((7, 4.7), (7, 4.0), "MANY-to-ONE on hour_eat\n(validate='m:1')")
arr((7, 1.6), (7, 2.2), "INTERVAL join -> (zone,hour) flags\n(validate='m:1' after aggregation)")
arr((9, 3.1), (10.3, 3.1), "")
ax.text(0.2, 3.9, "Why LEFT = trip grid:\nwe must forecast every\nzone-hour; weather/events\nare context only.\n\nEvent window rule:\npre/post hours per type\n(config.EVENT_WINDOWS),\nonly status confirmed/unknown.", fontsize=9, va="top")
plt.title("A3 – Join map"); plt.savefig(REPORTS / "A_join_map.png", dpi=150, bbox_inches="tight"); plt.show()
display(Markdown("Event window per type (hours before start, hours after end): `%s`" % EVENT_WINDOWS))

# %% [markdown]
# ## Build the master table (joins + features)
# %%
master_all, audit = build_master(grid, test, wx, ev, zones, launch_eff)
print(audit)

# %% [markdown]
# ## A4 – Join audit
# %%
tr_rows = master_all[master_all.split == "train"]
a4 = pd.DataFrame([
 dict(join="trips x weather (m:1 on hour_eat)", rows_before=audit["rows_before"], rows_after=audit["rows_after_weather"],
      match_rate_pct=round(100 * (~master_all.wx_missing_hour).mean(), 3), unmatched_zone_hours=int(master_all.wx_missing_hour.sum()),
      note="zone-hours whose hour had no weather row in the raw export (gap hours) -> weather imputed (<=6h interpolation, else train climatology; rain=0) and flagged wx_imputed"),
 dict(join="grid x events (m:1 after aggregation)", rows_before=audit["rows_after_weather"], rows_after=audit["rows_after_events"],
      match_rate_pct=round(100 * (master_all[EV_COLS + ["ev_n_active"]].sum(axis=1) > 0).mean(), 3), unmatched_zone_hours=int((master_all.ev_n_active == 0).sum()),
      note="match = zone-hour inside some event's pre/during/post window; the rest legitimately have no event")])
display(save_table(a4, "A4_join_audit"))
print("zone-hours with any imputed weather value:", int(master_all.wx_imputed.sum()))
print("Duplicate keys in weather after cleaning:", int(wx.hour_eat.duplicated().sum()), "(must be 0)")
L = explode_events(ev[ev.use & ev.is_active], zones).merge(master_all[["zone", "hour_eat"]].drop_duplicates(), on=["zone", "hour_eat"])
ev_audit = pd.DataFrame({"metric": ["events in table (after dedupe)", "events used", "matched >=1 zone-hour", "excluded (any reason)", "inactive (cancelled/postponed)"],
                         "count": [len(ev), int(ev.use.sum()), L.event_id.nunique(), int((~ev.use).sum()), int((ev.use & ~ev.is_active).sum())]})
display(save_table(ev_audit, "A4_event_audit"))
print("Exclusion reasons:"); display(ev.loc[~ev.use, ["event_id", "event_type", "excluded_reason"]])
print("Status counts:"); display(ev.status_clean.value_counts())

# %% [markdown]
# ## A5 – Join proof (3 zone-hours)
# %%
def show_zone_hour(row, tag):
    print(f"\n######## {tag}: zone={row.zone} hour_eat={row.hour_eat}")
    hr = wx.loc[wx.hour_eat == row.hour_eat].iloc[0]
    print("Weather row attached (raw timestamps that produced it):", hr.raw_timestamps)
    display(raw_wx[raw_wx.timestamp.isin(hr.raw_timestamps.split("|"))])
    evs = L[(L.zone == row.zone) & (L.hour_eat == row.hour_eat)]
    if len(evs):
        ids = evs.event_id.unique(); rows = ev[ev.event_id.isin(ids)]
        print("Event rows attached:"); display(raw_ev.iloc[rows.raw_index.values]); display(evs[["event_id", "event_type", "phase"]])
    cols = ["temp_c", "rain_mm", "rain_last_3h", "rain_class", "is_public_holiday", "is_holiday_eve", "ev_n_active", "ev_att_log_max", "hours_to_next_event", "hours_since_last_event"] + [c for c in EV_COLS if row[c] == 1]
    display(row[cols].to_frame().T)
T = master_all[master_all.split == "train"]
show_zone_hour(T[(T.rain_mm >= 5)].iloc[0], "RAIN-affected")
fb = T[(T.ev_football_match_during == 1)]
show_zone_hour((fb if len(fb) else T[T.ev_n_active > 0]).iloc[0], "INSIDE an event window")
ph = T[T.is_public_holiday == 1]
r = ph.iloc[0]; show_zone_hour(r, "PUBLIC HOLIDAY")
display(ev[(ev.event_type == "public_holiday") & (ev.start <= r.hour_eat) & (ev.end > r.hour_eat)][["event_id", "event_name", "start", "end", "status_clean"]])

# %% [markdown]
# ## A6 – Feature engineering table
# %%
spec = feature_spec()
display(save_table(spec, "A6_feature_table")); print(len(spec), "features;", spec.group.value_counts().to_dict())

# %% [markdown]
# ## A7 – Integrity checks
# %%
mt = master_all[(master_all.split == "train") & (master_all.gap_type == "observed")]
mte = master_all[master_all.split == "test"].sort_values("_order")
ID_TR, ID_TE = ["record_id", "zone", "hour_eat"], ["row_id", "zone", "hour_eat"]
master_train = mt[ID_TR + ["trips"] + OPS_COLS + FEATURE_COLS].reset_index(drop=True)
master_test = mte[ID_TE + FEATURE_COLS].reset_index(drop=True)
checks = validate(master_train, master_test, raw_tmpl, wx, ev, audit, FEATURE_COLS)
save_table(checks, "A7_integrity_checks")
assert (checks.result == "PASS").all(), "fix failing checks before exporting"

# %% [markdown]
# ## A8 – Export master tables & data dictionary
# %%
master_train.to_csv(PROC / "master_train.csv", index=False)
master_test.to_csv(PROC / "master_test.csv", index=False)
dd = data_dictionary(master_train, master_test); dd.to_csv(PROC / "data_dictionary_master.csv", index=False)
# supporting files for notebook 02 and the demo app
wx.to_csv(PROC / "weather_clean_eat.csv", index=False)
ev.drop(columns=["zones"]).to_csv(PROC / "events_clean.csv", index=False)
grid.to_csv(PROC / "grid_with_gaps.csv", index=False)
pd.DataFrame({"zone": first_obs.index, "first_obs": first_obs.values, "late": late.values, "launch_eff": launch_eff.values}).to_csv(PROC / "zone_launch.csv", index=False)
print(master_train.shape, master_test.shape, "| test has the same", len(FEATURE_COLS), "feature columns, no target")
display(dd.head(12))