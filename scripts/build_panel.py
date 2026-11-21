"""Build the unit-year operating-fingerprint panel, retirement events and the
landmark (dynamic-prediction) dataset from the unit-quarter / unit-month parts.

Outputs (data_processed/):
  unit_quarter.parquet   all quarterly summaries, 1995Q1-2026Q3
  unit_year.parquet      unit x calendar-year fingerprints (features use year <= t only)
  units.parquet          one row per unit: entry, last operation, event flags, static attributes
  landmark.parquet       unit x landmark-year rows with features known at year-end t and
                         time-to-last-operation outcome under administrative censoring
"""
from __future__ import annotations

import glob
import re
from paths import DP

import numpy as np
import pandas as pd

PARTS = DP / "cems_parts"

END_QI = 2026 * 4 + 2          # 2026Q3 = last quarter in the archive
BUFFER_Q = 8                   # a unit must be silent >= 2 years to count as ceased
CENS_QI = END_QI - BUFFER_Q    # administrative censoring quarter (2024Q3)
FIRST_LM, LAST_LM = 1999, 2025
LOAD_CEILING_MW = 2000.0


def qi(tag: pd.Series) -> pd.Series:
    return tag.str[:4].astype(int) * 4 + tag.str[5].astype(int) - 1


def fuel_group(s: pd.Series) -> pd.Series:
    x = s.fillna("").str.lower()
    out = np.select(
        [x.str.startswith("coal") | x.str.contains("petroleum coke") | x.str.contains("coal refuse"),
         x.str.contains("natural gas") | x.str.contains("pipeline"),
         x.str.contains("oil"),
         x.str.contains("gas")],
        ["coal", "gas", "oil", "other_gas"], default="other")
    return pd.Series(out, index=s.index)


def tech_group(s: pd.Series) -> pd.Series:
    x = s.fillna("").str.lower()
    out = np.select(
        [x.str.contains("combined cycle"), x.str.contains("combustion turbine"),
         x.str.contains("tangential"), x.str.contains("wall-fired"), x.str.contains("cyclone"),
         x.str.contains("fluidized"), x.str.contains("cell burner"),
         x.str.contains("boiler") | x.str.contains("stoker") | x.str.contains("fired")],
        ["combined_cycle", "combustion_turbine", "tangential", "wall_fired", "cyclone",
         "fluidized_bed", "cell_burner", "other_boiler"], default="other")
    return pd.Series(out, index=s.index)


def has(s: pd.Series, pat: str) -> pd.Series:
    return s.fillna("").str.contains(pat, case=False, regex=True).astype("int8")


def slope(a: np.ndarray) -> float:
    m = np.isfinite(a)
    if m.sum() < 3:
        return np.nan
    x = np.arange(len(a))[m]
    return float(np.polyfit(x, a[m], 1)[0])


def main():
    expected = {f"{y}q{q}" for y in range(1995, 2027) for q in range(1, 5) if (y, q) <= (2026, 3)}
    for table in ("unit_quarter", "unit_month"):
        available = {p.stem.rsplit("_", 1)[-1] for p in PARTS.glob(f"{table}_*.parquet")}
        if available != expected:
            raise ValueError(f"{table}: expected 127 quarters; missing {sorted(expected - available)}, "
                             f"unexpected {sorted(available - expected)}")
    # ------------------------------------------------------------------ quarters
    q = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PARTS / "unit_quarter_*.parquet")))],
                  ignore_index=True)
    q["qi"] = qi(q["quarter"])
    q["year"] = q["quarter"].str[:4].astype(int)
    for c in ["state", "facility_name", "fuel1", "fuel2", "unit_type", "so2_ctrl", "nox_ctrl",
              "pm_ctrl", "hg_ctrl", "program", "fuel1_first", "so2_ctrl_first", "nox_ctrl_first"]:
        q[c] = q[c].replace({"nan": None, "None": None})
    q["facility_id"] = q["facility_id"].astype(int)
    # Physical plausibility: no US unit exceeds ~1,400 MW gross. Quarters with gross-load records
    # above 2,000 MW are reporting errors; their load-derived quantities are set to missing.
    bad = q["load_max"] > LOAD_CEILING_MW
    load_cols = [c for c in q.columns if c.startswith(("load_", "mwh", "ramp_"))] + ["co2_zero_load", "op_zero_load"]
    q.loc[bad, load_cols] = np.nan
    q["load_qc_flag"] = bad.astype(int)
    print("quarters with implausible gross load set to missing:", int(bad.sum()), "units:", q.loc[bad, "uid"].nunique())
    q = q.sort_values(["uid", "qi"]).reset_index(drop=True)
    q.to_parquet(DP / "unit_quarter.parquet", index=False)
    print("unit-quarter rows", len(q), "units", q.uid.nunique(), "quarters", q.qi.nunique())

    m = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(PARTS / "unit_month_*.parquet")))],
                  ignore_index=True)
    m["year"] = m["quarter"].str[:4].astype(int)

    # ------------------------------------------------------------------ unit-year
    add = ["n_rows", "op_hours", "n_on", "mwh", "starts", "starts_cold", "starts_warm", "starts_hot",
           "heat", "co2", "so2", "nox", "steam", "n_co2", "mwh_night", "mwh_peak", "co2_zero_load",
           "op_zero_load", "hours_hi", "heat_hi", "mwh_hi", "co2_hi", "nox_hi", "hours_lo", "heat_lo",
           "mwh_lo", "co2_lo", "nox_lo"]
    sub = [c for c in q.columns if c.endswith("_substitute") or c.endswith("_measuredands")]
    q["_ramp_w"] = q["ramp_mean"] * q["n_on"]
    q["_lstd_w"] = q["load_std"] * q["n_on"]
    q["_sub_co2"] = q[[c for c in sub if c.startswith("co2_ind")]].sum(axis=1)
    g = q.groupby(["uid", "year"], sort=True)
    y = g[add + ["_ramp_w", "_lstd_w", "_sub_co2"]].sum()
    y["peak_q"] = g["load_p99"].max()
    y["load_max"] = g["load_max"].max()
    y["n_quarters"] = g["qi"].nunique()
    y["q_last_on"] = q[q.n_on > 0].groupby(["uid", "year"])["qi"].max()
    last = g[["state", "facility_id", "facility_name", "fuel1", "fuel2", "unit_type", "so2_ctrl",
              "nox_ctrl", "pm_ctrl", "hg_ctrl", "program"]].last()
    y = y.join(last).reset_index()

    mm = m.groupby(["uid", "year", "month"], as_index=False)[["op_hours", "mwh", "n_rows"]].sum()
    mm["summer"] = mm["month"].isin([6, 7, 8]) * mm["mwh"]
    mg = mm.groupby(["uid", "year"])
    tot = mg["mwh"].transform("sum")
    mm["sh2"] = np.where(tot > 0, (mm["mwh"] / tot) ** 2, np.nan)
    ms = mg.agg(summer_mwh=("summer", "sum"), month_hhi=("sh2", "sum"), months_rep=("month", "nunique"),
                months_idle=("op_hours", lambda s: int((s <= 0).sum()))).reset_index()
    y = y.merge(ms, on=["uid", "year"], how="left")

    y = y.sort_values(["uid", "year"]).reset_index(drop=True)
    gy = y.groupby("uid", sort=False)
    # capacity proxy: highest quarterly p99 gross load seen in the trailing three years (past only)
    y["cap"] = gy["peak_q"].transform(lambda s: s.rolling(3, min_periods=1).max())
    y["cap"] = gy["cap"].ffill()
    safe = lambda a, b: np.where((b > 0) & np.isfinite(b), a / b, np.nan)
    y["cf"] = np.clip(safe(y.mwh, y.cap * y.n_rows), 0, 1.2)
    y["util"] = safe(y.op_hours, y.n_rows)
    y["lf_on"] = np.clip(safe(safe(y.mwh, y.op_hours), y.cap), 0, 1.5)
    y["starts_kh"] = safe(y.starts * 1000.0, y.op_hours)
    y["cold_share"] = safe(y.starts_cold, y.starts)
    y["run_len"] = safe(y.op_hours, y.starts.clip(lower=1))
    y["ramp_rel"] = safe(safe(y._ramp_w, y.n_on), y.cap)
    y["load_cv"] = safe(safe(y._lstd_w, y.n_on), safe(y.mwh, y.op_hours))
    y["hr"] = safe(y.heat, y.mwh)
    y["hr_hi"] = safe(y.heat_hi, y.mwh_hi)
    y["hr_lo"] = safe(y.heat_lo, y.mwh_lo)
    y["pl_pen"] = safe(y.hr_lo, y.hr_hi)
    y["lo_share"] = safe(y.mwh_lo, y.mwh)
    y["hi_hours_share"] = safe(y.hours_hi, y.n_on)
    y["co2_rate"] = safe(y.co2, y.mwh)
    y["so2_rate"] = safe(y.so2, y.heat)
    y["nox_rate"] = safe(y.nox, y.heat)
    y["night_share"] = safe(y.mwh_night, y.mwh)
    y["peak_share"] = safe(y.mwh_peak, y.mwh)
    y["zero_load_share"] = safe(y.op_zero_load, y.op_hours)
    y["summer_share"] = safe(y.summer_mwh, y.mwh)
    y["sub_share"] = safe(y._sub_co2, y.n_co2)
    y["rep_frac"] = y.n_rows / np.where(y.year % 4 == 0, 8784, 8760)
    for c in ("hr", "hr_hi", "hr_lo"):
        y.loc[(y[c] < 3) | (y[c] > 40), c] = np.nan
    y.loc[(y.pl_pen < 0.5) | (y.pl_pen > 3), "pl_pen"] = np.nan
    y.loc[y.co2_rate > 5, "co2_rate"] = np.nan

    y["fuel_g"] = fuel_group(y.fuel1)
    y["tech_g"] = tech_group(y.unit_type)
    y["dual_fuel"] = y.fuel2.notna().astype("int8")
    y["scrubber"] = has(y.so2_ctrl, r"wet lime|dry lime|limestone|dual alkali|magnesium|sodium")
    y["dsi"] = has(y.so2_ctrl, r"sorbent")
    y["scr"] = has(y.nox_ctrl, r"selective catalytic")
    y["sncr"] = has(y.nox_ctrl, r"non-catalytic")
    y["lnb"] = has(y.nox_ctrl, r"low nox|overfire")
    y["baghouse"] = has(y.pm_ctrl, r"baghouse")
    y["esp"] = has(y.pm_ctrl, r"electrostatic")
    y["hg_aci"] = has(y.hg_ctrl, r"sorbent|additive")
    y["arp"] = has(y.program, r"\bARP\b")
    y["rggi"] = has(y.program, r"RGGI")
    y["n_prog"] = y.program.fillna("").str.count(",") + y.program.notna().astype(int)

    # ---- history / trend features (strictly past + current year)
    gy = y.groupby("uid", sort=False)
    y["first_year"] = gy["year"].transform("min")
    y["yrs_obs"] = y.year - y.first_year + 1
    y["age_known"] = (y.first_year >= 1998).astype("int8")   # entered after full ARP coverage began
    y["cum_op_kh"] = gy["op_hours"].cumsum() / 1000.0
    y["cum_starts"] = gy["starts"].cumsum()
    y["cum_cold"] = gy["starts_cold"].cumsum()
    for v in ("cf", "util", "starts_kh", "hr", "lf_on", "night_share", "so2_rate", "nox_rate"):
        prev = gy[v].shift(1)
        base = gy[v].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
        y[f"{v}_d1"] = y[v] - prev
        y[f"{v}_d3"] = y[v] - base
    y["cf_max_past"] = gy["cf"].cummax()
    y["cf_rel_peak"] = safe(y.cf, y.cf_max_past)
    y["cf_slope5"] = gy["cf"].transform(lambda s: s.rolling(5, min_periods=3).apply(slope, raw=True))
    y["util_slope5"] = gy["util"].transform(lambda s: s.rolling(5, min_periods=3).apply(slope, raw=True))
    y["hr_base"] = gy["hr"].transform(lambda s: s.expanding(1).median().shift(1))
    y["hr_drift"] = safe(y.hr, y.hr_base)
    y["cf_sd5"] = gy["cf"].transform(lambda s: s.rolling(5, min_periods=3).std())
    y["scrub_new3"] = (y.scrubber - gy["scrubber"].shift(3).fillna(y.scrubber)).clip(lower=0)
    y["scr_new3"] = (y.scr - gy["scr"].shift(3).fillna(y.scr)).clip(lower=0)

    # ------------------------------------------------------------------ units / events
    on = q[q.n_on > 0].groupby("uid")["qi"].agg(first_on_qi="min", last_on_qi="max")
    u = q.groupby("uid").agg(first_qi=("qi", "min"), last_rep_qi=("qi", "max"),
                             facility_id=("facility_id", "last"), state=("state", "last"),
                             facility_name=("facility_name", "last")).join(on)
    yl = y.groupby("uid").agg(fuel_first=("fuel_g", "first"), fuel_last=("fuel_g", "last"),
                              tech=("tech_g", "last"), cap_max=("cap", "max"), ever_arp=("arp", "max"),
                              mwh_total=("mwh", "sum"), heat_total=("heat", "sum"))
    u = u.join(yl).reset_index()
    u["ceased"] = (u.last_on_qi <= CENS_QI).astype(int)          # silent >= 2 years before archive end
    u["last_on_year"] = u.last_on_qi // 4
    u["gap_after_q"] = u.last_rep_qi - u.last_on_qi               # quarters still reporting after last operation
    # longest silent spell followed by a restart (mothball-and-return check for the event definition)
    onq = q.loc[q.n_on > 0, ["uid", "qi"]].sort_values(["uid", "qi"])
    onq.to_parquet(DP / "unit_on_quarters.parquet", index=False)
    gaps = onq.groupby("uid")["qi"].diff()
    u = u.merge((gaps - 1).groupby(onq["uid"]).max().rename("max_restart_gap_q").reset_index(), on="uid", how="left")
    # coal -> gas conversion year (competing fate for coal units)
    fy = y[y.op_hours > 0][["uid", "year", "fuel_g"]]
    coal_seen = fy.assign(c=(fy.fuel_g == "coal").astype(int)).groupby("uid")["c"].cummax()
    conv = fy[(coal_seen == 1) & (fy.fuel_g == "gas")].groupby("uid")["year"].min().rename("conv_year")
    u = u.merge(conv.reset_index(), on="uid", how="left")
    u.to_parquet(DP / "units.parquet", index=False)

    # ---- context features (fleet information knowable at the end of year t only)
    act = y[(y.op_hours > 0) & (y.cap > 0)].copy()
    fac = act.groupby(["facility_id", "year"]).agg(fac_units=("uid", "nunique"), fac_cap=("cap", "sum"),
                                                   fac_mwh=("mwh", "sum"), fac_hours=("n_rows", "max"))
    fac["fac_cf"] = fac.fac_mwh / (fac.fac_cap * fac.fac_hours)
    y = y.merge(fac[["fac_units", "fac_cap", "fac_cf"]].reset_index(), on=["facility_id", "year"], how="left")
    y["cap_share_fac"] = safe(y.cap, y.fac_cap)
    # units that were active in t-3..t-1 but silent in t (observable at year end, no future status used)
    yrs = np.arange(1995, 2027)
    A = act.pivot_table(index="uid", columns="year", values="op_hours", aggfunc="sum").reindex(columns=yrs).fillna(0) > 0
    last_act = pd.DataFrame(np.where(A.values, yrs[None, :], -1), index=A.index, columns=yrs).cummax(axis=1)
    sil = last_act.stack().rename("last_act").reset_index().rename(columns={"level_1": "year"})
    sil = sil[(sil.last_act >= sil.year - 3) & (sil.last_act <= sil.year - 1)]
    meta = y.groupby("uid")[["facility_id", "state"]].last()
    capu = y.groupby("uid")["cap"].max()
    fuelu = y.groupby("uid")["fuel_g"].last()
    sil = sil.join(meta, on="uid").assign(cap=lambda s: s.uid.map(capu), fuel_g=lambda s: s.uid.map(fuelu))
    sib = sil.groupby(["facility_id", "year"]).size().rename("sib_ceased3").reset_index()
    y = y.merge(sib, on=["facility_id", "year"], how="left")
    y["sib_ceased3"] = y.sib_ceased3.fillna(0)

    sf = act.groupby(["state", "fuel_g", "year"])
    act["cf_rank_state"] = sf["cf"].rank(pct=True)
    act["hr_rank_state"] = sf["hr"].rank(pct=True)
    act["cap_rank_state"] = sf["cap"].rank(pct=True)
    nf = act.groupby(["fuel_g", "tech_g", "year"])
    act["cf_rank_tech"] = nf["cf"].rank(pct=True)
    act["hr_rank_tech"] = nf["hr"].rank(pct=True)
    y = y.merge(act[["uid", "year", "cf_rank_state", "hr_rank_state", "cap_rank_state", "cf_rank_tech",
                     "hr_rank_tech"]], on=["uid", "year"], how="left")
    st = act.groupby(["state", "year"]).apply(
        lambda d: pd.Series({
            "st_gas_share": d.loc[d.fuel_g == "gas", "mwh"].sum() / max(d.mwh.sum(), 1e-9),
            "st_coal_cf": d.loc[d.fuel_g == "coal", "mwh"].sum() / max((d.loc[d.fuel_g == "coal", "cap"] * d.loc[d.fuel_g == "coal", "n_rows"]).sum(), 1e-9),
            "st_mwh": d.mwh.sum()}), include_groups=False).reset_index()
    st = st.sort_values(["state", "year"])
    gs = st.groupby("state")
    st["st_gas_share_d3"] = st.st_gas_share - gs["st_gas_share"].shift(3)
    st["st_coal_cf_d3"] = st.st_coal_cf - gs["st_coal_cf"].shift(3)
    st["st_mwh_g3"] = st.st_mwh / gs["st_mwh"].shift(3) - 1
    y = y.merge(st.drop(columns="st_mwh"), on=["state", "year"], how="left")
    # share of same-fuel state capacity that went silent in the previous three years
    cs = sil.groupby(["state", "fuel_g", "year"])["cap"].sum().rename("capret").reset_index()
    stcap = act.groupby(["state", "fuel_g", "year"])["cap"].sum().rename("st_fuel_cap").reset_index()
    y = y.merge(cs, on=["state", "fuel_g", "year"], how="left").merge(stcap, on=["state", "fuel_g", "year"], how="left")
    y["st_ret3_share"] = safe(y.capret.fillna(0), y.st_fuel_cap + y.capret.fillna(0))
    ns = sil.groupby(["fuel_g", "year"])["cap"].sum().rename("natret").reset_index()
    nat = act.groupby(["fuel_g", "year"]).apply(
        lambda d: pd.Series({"nat_fuel_cf": d.mwh.sum() / max((d.cap * d.n_rows).sum(), 1e-9), "nat_cap": d.cap.sum()}),
        include_groups=False).reset_index().merge(ns, on=["fuel_g", "year"], how="left")
    nat["nat_ret3_share"] = nat.natret.fillna(0) / (nat.nat_cap + nat.natret.fillna(0))
    y = y.merge(nat[["fuel_g", "year", "nat_fuel_cf", "nat_ret3_share"]], on=["fuel_g", "year"], how="left")
    y.drop(columns=["_ramp_w", "_lstd_w", "_sub_co2", "capret"], inplace=True)
    y.to_parquet(DP / "unit_year.parquet", index=False)
    print("unit-year rows", len(y))

    # ------------------------------------------------------------------ landmark data
    # Risk set at landmark t = units with any operation during calendar year t (known at year end).
    # Outcomes are NOT stored here: they depend on the information date and are built by
    # ops_lib.outcomes() from unit_on_quarters.parquet for each forecast origin.
    lm = y[(y.year >= FIRST_LM) & (y.year <= LAST_LM) & (y.op_hours > 0) & (y.cap > 0) & (y.yrs_obs >= 2)].copy()
    lm["lm_qi"] = lm.year * 4 + 3
    q4 = q.loc[(q.qi % 4 == 3) & (q.n_on > 0), ["uid", "year"]].drop_duplicates().assign(op_q4=1)
    lm = lm.merge(q4, on=["uid", "year"], how="left")
    lm["op_q4"] = lm.op_q4.fillna(0).astype(int)
    lm.to_parquet(DP / "landmark.parquet", index=False)
    print("landmark rows", len(lm), "units", lm.uid.nunique(), "units ceased (final)", int(u.ceased.sum()))
    print("units with a >=8-quarter silent spell followed by restart:", int((u.max_restart_gap_q >= 8).sum()))
    print("coal units converted to gas:", int(u.conv_year.notna().sum()))


if __name__ == "__main__":
    main()
