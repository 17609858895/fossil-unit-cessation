"""Event-time study of operating fingerprints before operating cessation (descriptive).

For each ARP coal/gas/oil unit that ceased operating (confirmed by a two-year silence), metrics
in years L-12..L-1 (L = final operating year) are expressed as excess over the median of
matched survivors: units of the same fuel group and technology, operating in the same calendar
year, that did not cease within the following 5 years. Unit-clustered bootstrap intervals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ops_lib import DP, RES

SEED = 20261119
METRICS = ["cf", "util", "lf_on", "starts_kh", "cold_share", "hr", "pl_pen", "night_share", "peak_share",
           "summer_share", "ramp_rel", "months_idle", "load_cv", "sub_share"]


def main():
    y = pd.read_parquet(DP / "unit_year.parquet")
    u = pd.read_parquet(DP / "units.parquet")[["uid", "ceased", "last_on_year", "conv_year"]]
    y = y.merge(u, on="uid", how="left")
    y = y[(y.arp == 1) & y.fuel_g.isin(["coal", "gas", "oil"]) & (y.op_hours > 0) & (y.cap > 0)]
    y["rel"] = y.year - y.last_on_year
    surv = y[(y.ceased == 0) | (y.last_on_year > y.year + 5)]
    ctrl = surv.groupby(["fuel_g", "tech_g", "year"])[METRICS].median().add_suffix("_ctrl")
    cnt = surv.groupby(["fuel_g", "tech_g", "year"]).size().rename("n_ctrl")
    ev = y[(y.ceased == 1) & (y.rel >= -12) & (y.rel <= -1) & (y.last_on_year >= 2000)].join(ctrl, on=["fuel_g", "tech_g", "year"]).join(cnt, on=["fuel_g", "tech_g", "year"])
    ev = ev[ev.n_ctrl >= 5]
    for m in METRICS:
        ev[f"{m}_ex"] = ev[m] - ev[f"{m}_ctrl"]
    rng = np.random.default_rng(SEED)
    rows = []
    for fuel in ["all", "coal", "gas", "oil"]:
        e = ev if fuel == "all" else ev[ev.fuel_g == fuel]
        uids = e.uid.unique()
        piv = {m: e.pivot_table(index="uid", columns="rel", values=f"{m}_ex", aggfunc="mean") for m in METRICS}
        for m in METRICS:
            P = piv[m].reindex(uids)
            med = P.median()
            boots = []
            for _ in range(300):
                b = P.iloc[rng.integers(0, len(P), len(P))]
                boots.append(b.median().values)
            boots = np.array(boots)
            for j, rel in enumerate(P.columns):
                rows.append({"fuel": fuel, "metric": m, "rel": int(rel), "median_excess": float(med[rel]),
                             "lo": float(np.nanquantile(boots[:, j], 0.025)), "hi": float(np.nanquantile(boots[:, j], 0.975)),
                             "n_units": int(P[rel].notna().sum()),
                             "ctrl_median": float(e.loc[e.rel == rel, f"{m}_ctrl"].median())})
    r = pd.DataFrame(rows)
    r.to_csv(RES / "pre_retirement_trajectories.csv", index=False)
    print("ceased units in study:", ev.uid.nunique(), ev.groupby("fuel_g").uid.nunique().to_dict())
    show = r[(r.fuel == "all") & r.rel.isin([-10, -7, -5, -3, -2, -1])]
    print(show.pivot_table(index="metric", columns="rel", values="median_excess").round(3).to_string())
    # first year (counting back) from which the excess is consistently significant through L-1
    lead = []
    for (fuel, m), g in r.groupby(["fuel", "metric"]):
        g = g.sort_values("rel", ascending=False)
        sig = ((g.lo > 0) | (g.hi < 0)).values
        k = 0
        while k < len(sig) and sig[k]:
            k += 1
        lead.append({"fuel": fuel, "metric": m, "years_of_consistent_signal": k,
                     "excess_at_L-1": float(g.median_excess.iloc[0]), "excess_at_L-5": float(g.loc[g.rel == -5, "median_excess"].iloc[0]) if (g.rel == -5).any() else np.nan})
    lead = pd.DataFrame(lead)
    lead.to_csv(RES / "pre_retirement_lead_times.csv", index=False)
    print(lead[lead.fuel != "all"].pivot_table(index="metric", columns="fuel", values="years_of_consistent_signal").to_string())


if __name__ == "__main__":
    main()
