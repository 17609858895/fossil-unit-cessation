"""Link WRI Global Power Plant Database v1.3.0 (USA rows) to EPA CEMS facilities.

WRI USA identifiers are 'USA' + zero-padded EIA plant code; the EIA plant code equals the
ORIS / CAMPD Facility ID. Outputs:
  results/wri_cems_link.csv        facility-level match table
  results/wri_cems_generation_qc.csv   CEMS gross MWh vs EIA-reported net generation (2013-2019)
  results/wri_global_context.csv   global fossil capacity by country with / without reported generation

WRI v1.3.0 lists plants operating around 2019-2020, so plants that fully closed earlier are absent.
WRI fields are therefore used only for external QC and context, not as model features.
"""
from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd

from paths import DP, RES, WRI_ARCHIVE


def main():
    z = zipfile.ZipFile(WRI_ARCHIVE)
    w = pd.read_csv(z.open("global_power_plant_database.csv"), low_memory=False)
    us = w[w.country == "USA"].copy()
    us["facility_id"] = pd.to_numeric(us.gppd_idnr.str[3:], errors="coerce").astype("Int64")
    uy = pd.read_parquet(DP / "unit_year.parquet")
    units = pd.read_parquet(DP / "units.parquet")

    fac = units.groupby("facility_id").agg(n_units=("uid", "nunique"), any_active_2019=("last_on_year", lambda s: int((s >= 2019).any())),
                                           all_ceased=("ceased", "min"), last_on=("last_on_year", "max"),
                                           fuel=("fuel_last", lambda s: s.mode().iat[0])).reset_index()
    link = fac.merge(us[["facility_id", "name", "primary_fuel", "capacity_mw", "commissioning_year", "latitude", "longitude"]],
                     on="facility_id", how="left")
    link["matched"] = link.name.notna().astype(int)
    link.to_csv(RES / "wri_cems_link.csv", index=False)
    print("CEMS facilities", len(link), "matched to WRI", int(link.matched.sum()))
    print(link.groupby("any_active_2019")["matched"].agg(["mean", "count"]).rename(index={0: "inactive by 2019", 1: "active in/after 2019"}))

    # plant-level generation QC (calendar year, ARP units summed)
    g = uy.groupby(["facility_id", "year"])["mwh"].sum().rename("cems_gross_mwh").reset_index()
    long = us.melt(id_vars=["facility_id", "primary_fuel", "capacity_mw"],
                   value_vars=[f"generation_gwh_{y}" for y in range(2013, 2020)], var_name="yr", value_name="eia_net_gwh")
    long["year"] = long.yr.str[-4:].astype(int)
    qc = long.dropna(subset=["eia_net_gwh"]).merge(g, on=["facility_id", "year"], how="inner")
    qc = qc[(qc.eia_net_gwh > 1) & (qc.cems_gross_mwh > 1000)]
    qc["ratio_gross_net"] = qc.cems_gross_mwh / 1000 / qc.eia_net_gwh
    qc.to_csv(RES / "wri_cems_generation_qc.csv", index=False)
    lg = np.log10(qc[["cems_gross_mwh", "eia_net_gwh"]].assign(cems_gross_mwh=lambda s: s.cems_gross_mwh / 1000))
    r = np.corrcoef(lg.cems_gross_mwh, lg.eia_net_gwh)[0, 1]
    print(f"plant-years {len(qc)}  log-log r = {r:.3f}  median gross/net = {qc.ratio_gross_net.median():.3f}")
    print(qc.groupby("primary_fuel")["ratio_gross_net"].describe(percentiles=[0.1, 0.5, 0.9]).round(3))

    # global context: fossil capacity with any reported (not estimated) generation
    fos = w[w.primary_fuel.isin(["Coal", "Gas", "Oil", "Petcoke"])].copy()
    fos["has_rep"] = fos[[f"generation_gwh_{y}" for y in range(2013, 2020)]].notna().any(axis=1)
    ctx = fos.groupby("country_long").agg(cap_gw=("capacity_mw", lambda s: s.sum() / 1000), plants=("gppd_idnr", "size"),
                                          rep_cap_gw=("capacity_mw", lambda s: s[fos.loc[s.index, "has_rep"]].sum() / 1000))
    ctx["rep_share"] = ctx.rep_cap_gw / ctx.cap_gw
    ctx = ctx.sort_values("cap_gw", ascending=False)
    ctx.to_csv(RES / "wri_global_context.csv")
    tot = ctx.cap_gw.sum()
    print(f"global fossil capacity in WRI: {tot:.0f} GW; with any reported plant generation: {ctx.rep_cap_gw.sum():.0f} GW "
          f"({ctx.rep_cap_gw.sum() / tot:.1%}); USA share of reported: {ctx.loc['United States of America', 'rep_cap_gw'] / ctx.rep_cap_gw.sum():.1%}")
    print(ctx.head(12).round(2).to_string())


if __name__ == "__main__":
    main()
