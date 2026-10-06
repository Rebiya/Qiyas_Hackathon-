# %% [markdown]
# # 02 – Data Analysis Report (Deliverable B)
# %%
import sys
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from IPython.display import display, Markdown
from src.config import *
from src.features import explode_events
pd.set_option("display.max_columns", 50, "display.width", 200)
say = lambda tid, txt: display(Markdown(f"**{tid} – interpretation:** {txt}"))

mt = pd.read_csv(PROC / "master_train.csv", parse_dates=["hour_eat"])
grid = pd.read_csv(PROC / "grid_with_gaps.csv", parse_dates=["hour_eat"])
wx = pd.read_csv(PROC / "weather_clean_eat.csv", parse_dates=["hour_eat"])
ev = pd.read_csv(PROC / "events_clean.csv", parse_dates=["start", "end"])
ev["zones"] = ev.zones_str.fillna("").str.split(";")
launch = pd.read_csv(PROC / "zone_launch.csv", parse_dates=["first_obs", "launch_eff"]).set_index("zone")
zones = sorted(mt.zone.unique())

# ---- shared baseline: "expected" demand for a zone-hour with no rain / event / holiday, detrended by zone-week level
EVF = [c for c in mt.columns if c.startswith("ev_") and c not in ("ev_n_active", "ev_att_log_max")]
mt["any_event"] = mt[EVF].sum(axis=1) > 0
mt["holiday_any"] = mt[["is_public_holiday", "is_holiday_eve", "is_day_after_holiday"]].sum(axis=1) > 0
mt["wk"] = (mt.t_days // 7).astype(int)
mt["is_clean"] = (~mt.any_event) & (~mt.holiday_any) & (mt.rain_mm == 0)
lvl = mt[mt.is_clean].groupby(["zone", "wk"]).trips.mean().rename("lvl").reset_index()
mt = mt.merge(lvl, on=["zone", "wk"], how="left")
mt["rel"] = mt.trips / mt.lvl
prof = mt[mt.is_clean].groupby(["zone", "dow", "hour"]).rel.mean().rename("prof").reset_index()
mt = mt.merge(prof, on=["zone", "dow", "hour"], how="left")
mt["expected"] = (mt.lvl * mt.prof).clip(lower=0.5)
mt["ratio"] = mt.trips / mt.expected
mt["date"] = mt.hour_eat.dt.normalize()

def agg_ratio(d, by, ci=False, reps=300):
    by = [by] if isinstance(by, str) else list(by)
    d = d.dropna(subset=["expected"]); out = []
    for key, g in d.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        day = g.groupby("date").agg(t=("trips", "sum"), e=("expected", "sum"))
        row = dict(zip(by, key)); row.update(ratio=day.t.sum() / day.e.sum(), n_hours=len(g), n_days=len(day))
        if ci and len(day) > 2:
            idx = np.random.default_rng(SEED).integers(0, len(day), (reps, len(day)))
            r = day.t.values[idx].sum(1) / day.e.values[idx].sum(1)
            row.update(ci_lo=np.percentile(r, 2.5), ci_hi=np.percentile(r, 97.5))
        out.append(row)
    return pd.DataFrame(out)

# %% [markdown]
# ## B1.1 – Volume by zone
# %%
b11 = mt.groupby("zone").agg(total_trips=("trips", "sum"), mean_trips_per_zone_hour=("trips", "mean"), observed_zone_hours=("trips", "size"), first_obs=("hour_eat", "min"))
b11["share_pct"] = (100 * b11.total_trips / b11.total_trips.sum()).round(2)
b11["late_launch"] = launch.late; b11 = b11.sort_values("total_trips", ascending=False)
display(save_table(b11.reset_index(), "B1_1_volume_by_zone"))
lt = b11[b11.late_launch]
say("B1.1", f"{', '.join(b11.index[:3])} carry {b11.share_pct.iloc[:3].sum():.1f}% of all trips. "
    + (f"{', '.join(lt.index)} started late (first row {lt.first_obs.min().date()}), so its total share understates it; compare mean trips per zone-hour instead." if len(lt) else "All zones operated from the start."))

# %% [markdown]
# ## B1.2 – Hour-of-day profile by zone type
# %%
wd = mt[mt.dow < 5].groupby(["zone", "hour"]).trips.mean().unstack()
share = wd.div(wd.sum(1), axis=0)
best = max(((silhouette_score(share, KMeans(k, n_init=10, random_state=SEED).fit(share).labels_), k) for k in range(3, 7)))
km = KMeans(best[1], n_init=10, random_state=SEED).fit(share)
ZONE_TYPE_OVERRIDE = {}     # e.g. {"Bole": "airport/nightlife"} if you disagree with the auto label
names = {}
for c in range(best[1]):
    p = share[km.labels_ == c].mean().values
    night, am, pm, mid = p[[22, 23, 0, 1, 2, 3, 4]].sum(), p[6:10].sum(), p[16:20].sum(), p[10:16].sum()
    names[c] = "nightlife/airport" if night > 0.2 else "market/midday" if mid > 0.5 else "commuter (AM+PM peaks)" if am > 0.18 and pm > 0.18 else "evening-leisure" if pm >= am else "morning-peak"
    if list(names.values()).count(names[c]) > 1: names[c] += f"_{c}"
ztype = {z: ZONE_TYPE_OVERRIDE.get(z, names[l]) for z, l in zip(share.index, km.labels_)}
mt["ztype"] = mt.zone.map(ztype)
display(pd.Series(ztype, name="zone_type").to_frame())
pt = mt[mt.dow < 5].groupby(["ztype", "hour"]).trips.mean().unstack(0)
b12 = pd.DataFrame({"peak_hour": pt.idxmax(), "peak_mean_trips": pt.max().round(1), "quietest_hour": pt.idxmin(), "quiet_mean_trips": pt.min().round(1)})
display(save_table(b12.reset_index(), "B1_2_zone_type_peaks"))
say("B1.2", f"Zones clustered into {best[1]} types by weekday hourly shape (silhouette {best[0]:.2f}). " + "; ".join(f"{t}: peaks {int(r.peak_hour):02d}:00, quietest {int(r.quietest_hour):02d}:00" for t, r in b12.iterrows()) + ".")

# %% [markdown]
# ## B1.3 – Weekday vs weekend
# %%
we = mt.groupby(["zone", "is_weekend"]).trips.mean().unstack(); we.columns = ["weekday_mean", "weekend_mean"]
we["weekend_to_weekday"] = (we.weekend_mean / we.weekday_mean).round(3); we = we.sort_values("weekend_to_weekday", ascending=False)
display(save_table(we.reset_index(), "B1_3_weekend_ratio"))
star = (np.log(we.weekend_to_weekday)).abs().idxmax()
dow_star = mt[mt.zone == star].groupby("dow").trips.mean().round(2).rename(index=dict(enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])))
display(dow_star.to_frame(f"{star} mean trips"))
say("B1.3", f"{we.index[0]} is busiest on weekends (x{we.weekend_to_weekday.iloc[0]:.2f}); {we.index[-1]} collapses (x{we.weekend_to_weekday.iloc[-1]:.2f}). Most distinctive day-of-week shape: {star} (table above).")

# %% [markdown]
# ## B1.4 – Trend
# %%
wk = mt.groupby("wk").agg(total_trips=("trips", "sum"), mean_per_zone_hour=("trips", "mean"), n=("trips", "size"))
full = wk[wk.n >= 0.8 * wk.n.max()].copy()
full["std_weekly_trips"] = full.mean_per_zone_hour * len(zones) * 168
stable = [z for z in zones if not launch.loc[z, "late"]]
st = mt[mt.zone.isin(stable)].groupby("wk").trips.mean().reindex(full.index) * len(stable) * 168
full["stable_zones_weekly_trips"] = st
slope, _ = np.polyfit(full.index, full.stable_zones_weekly_trips, 1)
g = 100 * (full.stable_zones_weekly_trips.iloc[-4:].mean() / full.stable_zones_weekly_trips.iloc[:4].mean() - 1)
display(save_table(full.reset_index(), "B1_4_weekly_trend"))
say("B1.4", f"For zones live all year, weekly trips grow ~{slope:,.0f} trips per week per week; last 4 vs first 4 full weeks = {g:+.1f}%. A tree model cannot extrapolate a rising trend, so November must be forecast with trend/lag-level features (t_days, zone_level_28d) or the model will under-forecast.")

# %% [markdown]
# ## B2.1 – Timezone check of the weather clock
# %%
obs = wx[wx.hour_eat < TEST_START]
tp = obs.groupby(obs.hour_eat.dt.hour).temp_c.mean().round(2)
print("Mean temp by EAT hour (cleaned clock). Peak hour:", int(tp.idxmax())); display(tp.to_frame().T)
core = mt[(~mt.any_event) & (~mt.holiday_any)]
byt = core.groupby(["ztype", "hour_eat"]).ratio.mean().unstack(0).dropna()
rain = wx.set_index("hour_eat").rain_mm; rows = []
for s in range(-6, 7):
    r = rain.reindex(byt.index - pd.Timedelta(hours=s)).values
    rows.append(dict(shift_h=s, **{t: np.corrcoef(r, byt[t].values)[0, 1] for t in byt.columns}))
sh = pd.DataFrame(rows).set_index("shift_h"); sh["mean_abs_corr"] = sh.abs().mean(axis=1)
display(save_table(sh.round(4).reset_index(), "B2_1_rain_demand_corr_vs_shift"))
pk = int(sh.mean_abs_corr.idxmax())
say("B2.1", f"Cleaned weather peaks in temperature at {int(tp.idxmax())}:00 EAT. Rain–demand correlation is strongest at shift {pk} h (|r|={sh.mean_abs_corr.max():.3f}) and weakens to {sh.mean_abs_corr.get(3, np.nan):.3f} at +3 h and {sh.mean_abs_corr.get(-3, np.nan):.3f} at -3 h – a UTC/EAT mix-up would cost this signal. "
    + ("Shift 0 is the best alignment, so the clock fix is confirmed." if pk == 0 else "The best shift is not 0: re-check the weather clock decision in A2c."))

# %% [markdown]
# ## B2.2 – Rain effect by zone type
# %%
base = mt[(~mt.any_event) & (~mt.holiday_any)]
rainy = agg_ratio(base[base.rain_mm > 0], ["ztype"], ci=True).round(3)
display(save_table(rainy, "B2_2_rain_effect_by_zone_type"))
say("B2.2", "; ".join(f"{r.ztype}: x{r.ratio:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]" for r in rainy.itertuples()) + ". "
    + ("Rain does NOT raise demand everywhere." if (rainy.ratio < 1).any() and (rainy.ratio > 1).any() else "The sign of the effect is the same across types; only its size differs.") + " (ratio = trips / expected for dry, event-free hours of the same zone, weekday, hour, detrended.)")

# %% [markdown]
# ## B2.3 – Rain dose-response
# %%
cls = {0: "none", 1: "light (<=2.5)", 2: "moderate (<=7.6)", 3: "heavy (>7.6)"}
b = base.assign(rain_cls=base.rain_class.map(cls))
dose = agg_ratio(b, ["ztype", "rain_cls"], ci=True).round(3)
dp = dose.pivot(index="ztype", columns="rain_cls", values="ratio")[list(cls.values())]
display(save_table(dose, "B2_3_rain_dose_response")); display(dp)
inc = dp.diff(axis=1).iloc[:, 1:]
say("B2.3", "Ratios by rain class are above. Step changes light->moderate->heavy are " + ", ".join(f"{t}: {', '.join(f'{v:+.2f}' for v in inc.loc[t])}" for t in inc.index) + ". If the second/third steps are much smaller than the first, the response saturates (proportional only for light rain).")

# %% [markdown]
# ## B3.1 – Public holidays
# %%
hol = ev[ev.use & ev.is_active & (ev.event_type == "public_holiday")]
hd = pd.DataFrame([(d.normalize(), e.event_name) for e in hol.itertuples() for d in pd.date_range(e.start.normalize(), (e.end - pd.Timedelta(minutes=1)).normalize())], columns=["date", "holiday"]).drop_duplicates("date")
brk = set(d for e in ev[ev.use & ev.is_active & (ev.event_type == "school_break")].itertuples() for d in pd.date_range(e.start.normalize(), (e.end - pd.Timedelta(minutes=1)).normalize()))
daily = mt.groupby("date").trips.mean(); cov = mt.groupby("date").size()
zday = mt.groupby(["zone", "date"]).trips.mean().unstack(0)
one = pd.Timedelta(days=1); hset = set(hd.date)
bad = hset | {d + one for d in hset} | {d - one for d in hset} | brk | set(cov.index[cov < 0.8 * cov.median()])
def same_weekday_index(d):
    comps = [d + k * 7 * one for k in (-3, -2, -1, 1, 2, 3)]; comps = [c for c in comps if c in daily.index and c not in bad]
    if d not in daily.index or not comps: return None
    return daily[d] / daily[comps].mean(), zday.loc[d] / zday.loc[comps].mean()
rows, zrows = [], {}
for r in hd.itertuples():
    out = same_weekday_index(r.date)
    if out: rows.append(dict(date=r.date.date(), holiday=r.holiday, weekday=r.date.day_name(), city_index=round(out[0], 3))); zrows[r.date] = out[1]
b31 = pd.DataFrame(rows).sort_values("city_index"); display(save_table(b31, "B3_1_holiday_index"))
local = [(pd.Timestamp(r.date), z, v) for r in b31.itertuples() for z, v in zrows[pd.Timestamp(r.date)].items() if abs(v - r.city_index) > 0.15]
loc_df = pd.DataFrame(local, columns=["date", "zone", "zone_index"]).round(3); display(save_table(loc_df, "B3_1_local_holiday_reactions"))
say("B3.1", f"{(b31.city_index < 1).sum()} of {len(b31)} holidays reduce city demand (range {b31.city_index.min():.2f}–{b31.city_index.max():.2f}). "
    + ("Not every holiday reduces demand. " if (b31.city_index >= 1).any() else "") + f"{loc_df.zone.nunique() if len(loc_df) else 0} zones deviate locally by >0.15 from the city index (table above).")

# %% [markdown]
# ## B3.2 – Football event-window study
# %%
def event_ratios(sub, windows, types=None):
    L = explode_events(sub, zones, windows=windows, types=types).merge(mt[["zone", "hour_eat", "trips", "expected"]], on=["zone", "hour_eat"]).dropna(subset=["expected"])
    per = L.groupby(["event_type", "event_id", "phase"]).agg(t=("trips", "sum"), e=("expected", "sum")).reset_index(); per["ratio"] = per.t / per.e
    s = per.groupby(["event_type", "phase"]).ratio.agg(mean="mean", sd="std", n_events="count").reset_index()
    s["se"] = s.sd / np.sqrt(s.n_events); s["ci_lo"], s["ci_hi"] = s["mean"] - 1.96 * s.se, s["mean"] + 1.96 * s.se
    s["p_vs_1"] = [stats.ttest_1samp(per[(per.event_type == r.event_type) & (per.phase == r.phase)].ratio, 1).pvalue if r.n_events > 2 else np.nan for r in s.itertuples()]
    return s.round(3)
fb = ev[ev.use & ev.is_active & (ev.event_type == "football_match") & (ev.status_clean == "confirmed")]
b32 = event_ratios(fb, {"football_match": (2, 2)}, ["football_match"]); b32["phase"] = pd.Categorical(b32.phase, ["pre", "during", "post"])
display(save_table(b32.sort_values("phase"), "B3_2_football_windows"))
top = b32.sort_values("mean").iloc[-1] if len(b32) else None
say("B3.2", f"{len(fb)} confirmed matches. " + (f"Largest uplift in the '{top.phase}' window (x{top['mean']:.2f}, CI [{top.ci_lo:.2f}, {top.ci_hi:.2f}])." if top is not None else "No usable matches."))

# %% [markdown]
# ## B3.3 – Event-type ranking
# %%
W = {**{t: (2, 2) for t in POINT_TYPES}, "road_closure": (0, 0)}
act = ev[ev.use & ev.is_active & ev.event_type.isin(list(W))]
b33 = event_ratios(act, W)
b33["effect"] = b33["mean"] - 1
rank = b33.loc[b33.groupby("event_type").effect.apply(lambda s: s.abs().idxmax())].sort_values("effect", key=abs, ascending=False)
rank["measurable"] = rank.p_vs_1 < 0.05
# holidays & school breaks use the same-weekday method (weekly-level baseline would absorb them)
extra = [dict(event_type="public_holiday", phase="day", mean=b31.city_index.mean(), n_events=len(b31), effect=b31.city_index.mean() - 1,
              p_vs_1=stats.ttest_1samp(b31.city_index, 1).pvalue if len(b31) > 2 else np.nan)]
bd = sorted(brk); idx = [same_weekday_index(d) for d in bd]; idx = [i[0] for i in idx if i]
if idx: extra.append(dict(event_type="school_break", phase="day", mean=np.mean(idx), n_events=len(idx), effect=np.mean(idx) - 1, p_vs_1=stats.ttest_1samp(idx, 1).pvalue if len(idx) > 2 else np.nan))
rank = pd.concat([rank, pd.DataFrame(extra)], ignore_index=True); rank["measurable"] = rank.p_vs_1 < 0.05
rank = rank.sort_values("effect", key=abs, ascending=False)
display(save_table(rank, "B3_3_event_type_ranking"))
say("B3.3", "Ranked by the largest |effect| across pre/during/post windows: " + ", ".join(f"{r.event_type} ({r.effect:+.2f})" for r in rank.itertuples()) + ". No measurable effect (p>=0.05): " + (", ".join(rank[~rank.measurable].event_type) or "none") + ".")

# %% [markdown]
# ## B3.4 – Cancelled and unlisted events
# %%
canc = ev[ev.use & (ev.status_clean == "cancelled")]
if len(canc):
    W_all = {**W, "public_holiday": (0, 0), "school_break": (0, 0)}
    b34a = event_ratios(canc, W_all)
    display(save_table(b34a, "B3_4a_cancelled_footprint"))
    say("B3.4a", f"{len(canc)} cancelled events. Their windows show ratios " + ", ".join(f"{r.event_type}/{r.phase} x{r['mean']:.2f}" for _, r in b34a.iterrows() if r.n_events > 0) + " vs confirmed events in B3.3: a cancelled event with ratio ~1 left no footprint, so treating cancelled events as inactive is right.")
else:
    say("B3.4a", "No events with status 'cancelled' after cleaning, so no footprint can be measured.")
sp = mt[(~mt.any_event) & (~mt.holiday_any) & (mt.expected >= 3)].copy()
sp["excess"] = sp.trips - sp.expected
c = sp[(sp.ratio >= 2) & (sp.excess >= 15)].sort_values(["zone", "hour_eat"]).copy()
c["episode"] = (c.groupby("zone").hour_eat.diff().dt.total_seconds().div(3600).fillna(99) > 2).cumsum()
epi = c.groupby("episode").agg(zone=("zone", "first"), start=("hour_eat", "min"), end=("hour_eat", "max"), hours=("trips", "size"),
                               peak_ratio=("ratio", "max"), excess_trips=("excess", "sum"), rain_mm=("rain_mm", "max")).sort_values("excess_trips", ascending=False).head(10)
def hyp(r):
    if r.rain_mm >= 2.5: return "rain-driven surge"
    if r.start.dayofweek >= 5 and r.start.hour >= 20: return "weekend night: private event/wedding/nightlife not in calendar"
    if r.start.hour in range(6, 10): return "morning surge: transit disruption / school or office start"
    if r.start.dayofweek == 6: return "Sunday gathering (religious/market) not in calendar"
    return "unlisted event / promotion / app incentive"
epi["hypothesis"] = epi.apply(hyp, axis=1)
display(save_table(epi.reset_index(), "B3_4b_unexplained_spikes"))
say("B3.4b", f"{c.episode.nunique()} spike episodes (>=2x expected, >=15 excess trips, no listed event). Top 10 above with rule-based hypotheses – edit them after looking at the dates/zones.")

# %% [markdown]
# ## B4.1 – Operational variables vs demand
# %%
rows = []
dev = mt[["trips"] + OPS_COLS] - mt.groupby(["zone", "dow", "hour"])[["trips"] + OPS_COLS].transform("mean")
for cname in OPS_COLS:
    rows.append(dict(variable=cname, pearson=mt.trips.corr(mt[cname]), spearman=mt.trips.corr(mt[cname], method="spearman"), pearson_after_removing_zone_hour_of_week_mean=dev.trips.corr(dev[cname])))
b41 = pd.DataFrame(rows).round(3); display(save_table(b41, "B4_1_ops_correlations"))
say("B4.1", "active_drivers moves with trips because drivers go where demand is (and demand needs supply) – correlation, not cause; avg_wait_min is an OUTCOME (long waits when demand exceeds supply, so sign can flip with context); avg_fare_birr reflects surge/zone mix. All three are consequences measured after demand happens and do not exist for 1–14 Nov, so they cannot be forecast-time inputs.")

# %% [markdown]
# ## B4.2 – Gaps and outages
# %%
oh = grid[grid.gap_type == "outage"].hour_eat.drop_duplicates().sort_values()
runs = oh.groupby((oh.diff() != pd.Timedelta(hours=1)).cumsum()).agg(start="min", end="max", hours="size").reset_index(drop=True)
display(save_table(runs, "B4_2_outages"))
pre = launch[launch.late].assign(prelaunch_hours=lambda d: ((d.first_obs - GRID_START) / pd.Timedelta(hours=1)).astype(int))
display(save_table(pre.reset_index(), "B4_2_late_launch"))
rm = grid[grid.gap_type == "random_missing"]
def longest(s):
    h = s.sort_values(); return int(h.groupby((h.diff() != pd.Timedelta(hours=1)).cumsum()).size().max()) if len(h) else 0
rz = grid.groupby("zone").agg(grid_hours=("hour_eat", "size"), random_missing=("gap_type", lambda s: (s == "random_missing").sum()), invalid=("gap_type", lambda s: (s == "invalid_value").sum()))
rz["longest_random_run_h"] = rm.groupby("zone").hour_eat.apply(longest).reindex(rz.index).fillna(0).astype(int)
rz["random_missing_pct"] = (100 * rz.random_missing / rz.grid_hours).round(2); display(save_table(rz.reset_index(), "B4_2_random_missing"))
by_h = grid[grid.gap_type.isin(["observed", "random_missing"])].groupby(grid.hour_eat.dt.hour).apply(lambda d: (d.gap_type == "random_missing").mean(), include_groups=False)
cr = by_h.corr(mt.groupby("hour").trips.mean())
say("B4.2", f"{len(runs)} platform outage run(s) ({int(runs.hours.sum()) if len(runs) else 0} h): treated as unknown demand, excluded from training (NOT zeros). Late-launch zone(s): {', '.join(pre.index) or 'none'} – hours before launch are not in the grid. Random missing zone-hours: {len(rm)} ({100*len(rm)/len(grid):.2f}%), excluded, not imputed. Correlation between hour-of-day missingness and mean demand = {cr:.2f}: "
    + ("strongly negative, so missing rows at quiet hours may really be zero-trip hours (consider treating them as 0)." if cr < -0.5 else "no strong link, so missing rows are not simply unreported zero-trip hours."))

# %% [markdown]
# ## B4.3 – Pay-period effect
# %%
d = mt[~mt.holiday_any].groupby("date").trips.mean()
d = d[mt.groupby("date").size().reindex(d.index) >= 0.8 * mt.groupby("date").size().median()]
det = d / d.rolling(15, center=True, min_periods=8).mean()
dw = det.groupby(det.index.dayofweek).transform("mean"); adj = det / dw
pay = (adj.index.day >= PAYDAY_START_DAY) | (adj.index.day <= PAYDAY_END_DAY)
eff = 100 * (adj[pay].mean() / adj[~pay].mean() - 1)
t = stats.ttest_ind(adj[pay], adj[~pay], equal_var=False); u = stats.mannwhitneyu(adj[pay], adj[~pay])
dom = adj.groupby(adj.index.day).mean().round(3).rename("detrended_dow_adjusted_index")
display(save_table(dom.reset_index(), "B4_3_day_of_month_profile"))
res = pd.DataFrame([dict(payday_days=int(pay.sum()), other_days=int((~pay).sum()), effect_pct=round(eff, 2), welch_p=round(t.pvalue, 4), mannwhitney_p=round(u.pvalue, 4))])
display(save_table(res, "B4_3_payday_test"))
say("B4.3", f"Payday window (day>={PAYDAY_START_DAY} or <={PAYDAY_END_DAY}) days are {eff:+.1f}% vs ordinary days after removing the 15-day trend and day-of-week (Welch p={t.pvalue:.3f}). "
    + ("Keep is_payday_window: the effect is both statistically and practically non-trivial." if abs(eff) >= 2 and t.pvalue < 0.05 else "Too small/uncertain to matter much; the feature is cheap to keep but expect little gain (confirm in the D5 ablation)."))