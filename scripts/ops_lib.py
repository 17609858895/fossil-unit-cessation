"""Shared definitions: feature groups, landmark loading, prospective-emulation splits."""
from __future__ import annotations

from paths import DP, RES

import numpy as np
import pandas as pd



CENS_QI = (2026 * 4 + 2) - 8

# ---- feature groups (ablation units) -------------------------------------------------
STATIC = ["log_cap", "yrs_obs", "age_known", "dual_fuel", "scrubber", "dsi", "scr", "sncr", "lnb",
          "baghouse", "esp", "hg_aci", "rggi", "n_prog"]
LEVEL = ["cf", "util", "lf_on", "starts_kh", "cold_share", "log_run_len", "ramp_rel", "load_cv", "hr",
         "pl_pen", "lo_share", "hi_hours_share", "co2_rate", "so2_rate", "nox_rate", "night_share",
         "peak_share", "zero_load_share", "summer_share", "month_hhi", "months_idle", "sub_share", "rep_frac"]
DYNAMIC = ["cf_d1", "cf_d3", "util_d1", "util_d3", "starts_kh_d1", "starts_kh_d3", "hr_d1", "hr_d3",
           "lf_on_d1", "lf_on_d3", "night_share_d1", "night_share_d3", "so2_rate_d3", "nox_rate_d3",
           "cf_rel_peak", "cf_slope5", "util_slope5", "hr_drift", "cf_sd5", "log_cum_op", "log_cum_starts",
           "log_cum_cold", "scrub_new3", "scr_new3"]
CONTEXT = ["fac_units", "fac_cf", "cap_share_fac", "sib_ceased3", "cf_rank_state", "hr_rank_state",
           "cap_rank_state", "cf_rank_tech", "hr_rank_tech", "st_gas_share", "st_coal_cf",
           "st_gas_share_d3", "st_coal_cf_d3", "st_mwh_g3", "st_ret3_share", "nat_fuel_cf", "nat_ret3_share"]
CATS = ["fuel_g", "tech_g"]
GROUPS = {"static": STATIC, "level": LEVEL, "dynamic": DYNAMIC, "context": CONTEXT}

PRETTY = {
    "log_cap": "Capacity (log MW)", "yrs_obs": "Years in CEMS record", "cf": "Capacity factor",
    "util": "Operating-hour share", "lf_on": "Load factor when on", "starts_kh": "Starts per 1000 h",
    "cold_share": "Cold-start share", "log_run_len": "Run length (log h)", "ramp_rel": "Hourly ramp / capacity",
    "load_cv": "Load variability (CV)", "hr": "Heat rate", "pl_pen": "Part-load heat-rate penalty",
    "lo_share": "Low-load energy share", "hi_hours_share": "High-load hour share", "co2_rate": "CO2 intensity",
    "so2_rate": "SO2 rate", "nox_rate": "NOx rate", "night_share": "Night energy share",
    "peak_share": "Afternoon-peak energy share", "zero_load_share": "Zero-load operating share",
    "summer_share": "Summer energy share", "month_hhi": "Monthly concentration (HHI)",
    "months_idle": "Idle months", "sub_share": "Substitute-data share", "rep_frac": "Reporting coverage",
    "cf_d1": "CF change (1 y)", "cf_d3": "CF change vs 3-y mean", "util_d1": "Operating share change (1 y)",
    "util_d3": "Operating share change vs 3-y mean", "starts_kh_d1": "Start-rate change (1 y)",
    "starts_kh_d3": "Start-rate change vs 3-y mean", "hr_d1": "Heat-rate change (1 y)",
    "hr_d3": "Heat-rate change vs 3-y mean", "lf_on_d1": "Load-factor change (1 y)",
    "lf_on_d3": "Load-factor change vs 3-y mean", "night_share_d1": "Night share change (1 y)",
    "night_share_d3": "Night share change vs 3-y mean", "so2_rate_d3": "SO2 rate change vs 3-y mean",
    "nox_rate_d3": "NOx rate change vs 3-y mean", "cf_rel_peak": "CF relative to own peak",
    "cf_slope5": "CF slope (5 y)", "util_slope5": "Operating share slope (5 y)", "hr_drift": "Heat-rate drift",
    "cf_sd5": "CF volatility (5 y)", "log_cum_op": "Cumulative operating hours (log)",
    "log_cum_starts": "Cumulative starts (log)", "log_cum_cold": "Cumulative cold starts (log)",
    "scrub_new3": "Scrubber added (3 y)", "scr_new3": "SCR added (3 y)", "fac_units": "Units at plant",
    "fac_cf": "Plant capacity factor", "cap_share_fac": "Share of plant capacity",
    "sib_ceased3": "Sibling units ceased (3 y)", "cf_rank_state": "CF rank in state fleet",
    "hr_rank_state": "Heat-rate rank in state fleet", "cap_rank_state": "Size rank in state fleet",
    "cf_rank_tech": "CF rank in technology fleet", "hr_rank_tech": "Heat-rate rank in technology fleet",
    "st_gas_share": "State gas share", "st_coal_cf": "State coal CF", "st_gas_share_d3": "State gas share change (3 y)",
    "st_coal_cf_d3": "State coal CF change (3 y)", "st_mwh_g3": "State output growth (3 y)",
    "st_ret3_share": "State same-fuel capacity ceased (3 y)", "nat_fuel_cf": "National fuel-fleet CF",
    "nat_ret3_share": "National same-fuel capacity ceased (3 y)",
    "scrubber": "Scrubber", "scr": "SCR", "sncr": "SNCR", "dsi": "Sorbent injection", "lnb": "Low-NOx burner",
    "baghouse": "Baghouse", "esp": "ESP", "hg_aci": "Hg sorbent", "rggi": "RGGI", "n_prog": "Programs",
    "dual_fuel": "Dual fuel", "age_known": "Entered after 1998",
}


def load_landmark(arp_only: bool = True, power_only: bool = True) -> pd.DataFrame:
    d = pd.read_parquet(DP / "landmark.parquet")
    d["log_cap"] = np.log10(d["cap"].clip(lower=1))
    d["log_run_len"] = np.log10(d["run_len"].clip(lower=1))
    d["log_cum_op"] = np.log10(d["cum_op_kh"].clip(lower=0.001))
    d["log_cum_starts"] = np.log10(d["cum_starts"].clip(lower=1))
    d["log_cum_cold"] = np.log10(d["cum_cold"].clip(lower=1))
    if arp_only:
        d = d[d["arp"] == 1]
    if power_only:
        d = d[d["fuel_g"].isin(["coal", "gas", "oil"])]
    return d.reset_index(drop=True)


# Period-level covariates: one value per (state or nation) x fuel x year. In a pooled model they
# act as calendar-period identifiers and transfer poorly into a new retirement wave.
PERIOD = ["st_gas_share", "st_coal_cf", "st_gas_share_d3", "st_coal_cf_d3", "st_mwh_g3", "st_ret3_share",
          "nat_fuel_cf", "nat_ret3_share"]
LOCAL_CONTEXT = [c for c in CONTEXT if c not in PERIOD]


def fleet_relative(d: pd.DataFrame, cols, by=("year", "fuel_g", "tech_g")) -> pd.DataFrame:
    """Percentile rank of each unit among the units active in the same year (and fuel/technology).

    Uses only same-year cross-sectional information, so it is available at the landmark date.
    Missing values stay missing. Groups with fewer than 5 units fall back to year x fuel.
    """
    out = {}
    g_full = d.groupby(list(by))
    n_full = g_full["uid"].transform("size")
    g_fuel = d.groupby(["year", "fuel_g"])
    for c in cols:
        r = g_full[c].rank(pct=True)
        r2 = g_fuel[c].rank(pct=True)
        out[f"rk_{c}"] = np.where(n_full >= 5, r, r2).astype("float32")
    return pd.DataFrame(out, index=d.index)


def period_relative(d: pd.DataFrame, cols) -> pd.DataFrame:
    """Within-year rank of period-level covariates across units (removes the calendar level)."""
    g = d.groupby("year")
    return pd.DataFrame({f"rk_{c}": g[c].rank(pct=True).astype("float32") for c in cols}, index=d.index)


def design(d: pd.DataFrame, groups=("static", "level", "dynamic", "context"), cats: bool = True,
           encoding: str = "absolute"):
    """encoding:
      'absolute'          features as measured (reference)
      'no_period'         absolute, period-level context removed
      'relative'          level/dynamic fingerprints and period context as within-year fleet ranks;
                          static and unit-local context kept as measured
      'absolute+relative' both representations side by side
    """
    if encoding == "absolute":
        cols = [c for g in groups for c in GROUPS[g]]
        X = d[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    else:
        base = [c for g in groups if g in ("static",) for c in GROUPS[g]]
        if "context" in groups:
            base += LOCAL_CONTEXT
        X = d[base].astype("float32").replace([np.inf, -np.inf], np.nan)
        fp = [c for g in groups if g in ("level", "dynamic") for c in GROUPS[g]]
        if encoding == "no_period":
            X = pd.concat([X, d[fp].astype("float32").replace([np.inf, -np.inf], np.nan)], axis=1)
        elif encoding == "relative":
            parts = [X, fleet_relative(d.replace([np.inf, -np.inf], np.nan), fp)]
            if "context" in groups:
                parts.append(period_relative(d, PERIOD))
            X = pd.concat(parts, axis=1)
        elif encoding == "absolute+relative":
            parts = [X, d[fp].astype("float32").replace([np.inf, -np.inf], np.nan),
                     fleet_relative(d.replace([np.inf, -np.inf], np.nan), fp)]
            if "context" in groups:
                parts.append(period_relative(d, PERIOD))
            X = pd.concat(parts, axis=1)
        else:
            raise ValueError(encoding)
    if cats:
        dm = pd.get_dummies(d[CATS], prefix=CATS, dtype="float32")
        X = pd.concat([X, dm], axis=1)
    return X


END_QI = 2026 * 4 + 2
BUFFER_Q = 8
_ONQ = None


def _on_quarters() -> pd.DataFrame:
    global _ONQ
    if _ONQ is None:
        _ONQ = pd.read_parquet(DP / "unit_on_quarters.parquet")
    return _ONQ


def outcomes(d: pd.DataFrame, info_qi: int, buffer_q: int = BUFFER_Q) -> pd.DataFrame:
    """Time-to-cessation outcomes using only operating quarters observed up to `info_qi`.

    A unit is a known cessation when its last operating quarter (as seen at info_qi) is at least
    `buffer_q` quarters before info_qi. Time is counted in years from the landmark year end, so
    annual interval k corresponds to calendar year t+k; a unit whose last operation fell inside
    the landmark year gets time 0.25. All other units are administratively censored at
    info_qi - buffer_q: censoring them at their own last operation would make censoring
    informative, because units that stop inside the confirmation window are disproportionately
    future cessations.
    """
    onq = _on_quarters()
    last_known = onq[onq.qi <= info_qi].groupby("uid")["qi"].max()
    lk = d["uid"].map(last_known).astype(float).values
    cq = info_qi - buffer_q
    ev = (lk <= cq).astype(int)
    t = np.maximum(np.minimum(lk, cq) - d["lm_qi"].values, 1) / 4.0
    out = pd.DataFrame({"time": t, "event": ev}, index=d.index)
    for H in (1, 2, 3, 5):
        known = (ev == 1) | (t >= H)
        out[f"y{H}"] = np.where(known, ((ev == 1) & (t <= H)).astype(float), np.nan)
    return out


def emulate_origin(d: pd.DataFrame, origin: int):
    """Prospective emulation at year-end `origin`.

    Training rows: landmarks t < origin with outcomes rebuilt from data observed by the origin
    (no later restarts, no later cessations). Test rows: the landmark-`origin` risk set scored
    against the final archive (2026Q3, two-year confirmation buffer).
    """
    oq = origin * 4 + 3
    tr = d[d["year"] < origin].copy()
    o = outcomes(tr, oq)
    tr["time_tr"], tr["event_tr"] = o["time"].values, o["event"].values
    for H in (1, 2, 3, 5):
        tr[f"ytr{H}"] = o[f"y{H}"].values
    te = d[d["year"] == origin].copy()
    o = outcomes(te, END_QI)
    for c in o.columns:
        te[c] = o[c].values
    return tr, te


def horizon_label(tr: pd.DataFrame, origin: int, H: int):
    """Binary 'ceases within H years' for training cohorts that are FULLY resolved at the origin.

    For landmarks later than origin - H - 2, cessations inside the horizon can already be
    confirmed but survival through the horizon cannot, so keeping their non-missing labels would
    select rows on the outcome (recent positives in, recent negatives out). Only cohorts with
    landmark year <= origin - H - 2 are used; every label in them is resolved.
    """
    ok = (tr["year"] <= origin - H - 2).values
    v = tr[f"ytr{H}"].values
    assert not np.isnan(v[ok]).any()
    return ok, np.nan_to_num(v).astype(int)
