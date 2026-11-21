"""Stream the PUDL raw EPA CEMS hourly archive (1995Q1-2026Q3) into compact
unit-quarter and unit-month tables without unpacking the ZIP files to disk.

Raw files are never modified. One hourly CSV member is parsed in memory at a
time per worker, reduced to per-unit operating/emission summaries, and written
as Parquet under data_processed/cems_parts/.

Usage
-----
python scripts/extract_hourly.py --workers 5
python scripts/extract_hourly.py --only 2010q3      # single-member test
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv

from paths import DP, RAW

OUT = DP / "cems_parts"

NUM = {
    "Operating Time": "op",
    "Gross Load (MW)": "load",
    "Steam Load (1000 lb/hr)": "steam",
    "SO2 Mass (lbs)": "so2",
    "CO2 Mass (short tons)": "co2",
    "NOx Mass (lbs)": "nox",
    "Heat Input (mmBtu)": "heat",
}
CAT = {
    "State": "state",
    "Facility Name": "facility_name",
    "Unit ID": "unit_id",
    "SO2 Mass Measure Indicator": "so2_ind",
    "CO2 Mass Measure Indicator": "co2_ind",
    "NOx Rate Measure Indicator": "noxr_ind",
    "Heat Input Measure Indicator": "heat_ind",
    "Primary Fuel Type": "fuel1",
    "Secondary Fuel Type": "fuel2",
    "Unit Type": "unit_type",
    "SO2 Controls": "so2_ctrl",
    "NOx Controls": "nox_ctrl",
    "PM Controls": "pm_ctrl",
    "Hg Controls": "hg_ctrl",
    "Program Code": "program",
}
OTHER = {"Facility ID": "facility_id", "Date": "date", "Hour": "hour"}
ATTRS = ["state", "facility_name", "fuel1", "fuel2", "unit_type",
         "so2_ctrl", "nox_ctrl", "pm_ctrl", "hg_ctrl", "program"]


def read_member(zpath: Path, member: str) -> pd.DataFrame:
    """Parse one quarterly CSV member by column name (header differs by year)."""
    cols = {**NUM, **CAT, **OTHER}
    types = {k: pa.float32() for k in NUM}
    types.update({k: pa.dictionary(pa.int32(), pa.string()) for k in CAT})
    types.update({"Facility ID": pa.int32(), "Date": pa.date32(), "Hour": pa.int8()})
    with zipfile.ZipFile(zpath) as z, z.open(member) as fh:
        tbl = pacsv.read_csv(
            fh,
            read_options=pacsv.ReadOptions(block_size=64 << 20),
            convert_options=pacsv.ConvertOptions(
                include_columns=list(cols), column_types=types, strings_can_be_null=True),
        )
    tbl = tbl.rename_columns([cols[c] for c in tbl.column_names])
    df = tbl.to_pandas(self_destruct=True, split_blocks=True)
    del tbl
    return df


def reduce_quarter(df: pd.DataFrame, tag: str):
    df["ts"] = (df["date"].astype("datetime64[s]").astype("int64") // 3600 + df["hour"]).astype("int64")
    df["month"] = pd.DatetimeIndex(df["date"].astype("datetime64[s]")).month.astype("int8")
    uid = df["facility_id"].astype(str) + "_" + df["unit_id"].astype(str)
    df["uid"] = pd.Categorical(uid)
    df["ucode"] = df["uid"].cat.codes.astype("int32")
    df.sort_values(["ucode", "ts"], inplace=True, kind="stable")
    df.reset_index(drop=True, inplace=True)

    op = df["op"].fillna(0).to_numpy()
    on = op > 0
    load = df["load"].to_numpy()
    load0 = np.nan_to_num(load, nan=0.0)
    code = df["ucode"].to_numpy()
    ts = df["ts"].to_numpy()
    same = np.r_[False, code[1:] == code[:-1]]
    consec = same & np.r_[False, ts[1:] - ts[:-1] == 1]

    # starts: off -> on inside a unit; off-duration before the start (hours)
    prev_on = np.r_[False, on[:-1]]
    start = on & same & ~prev_on
    last_on_ts = pd.Series(np.where(on, ts, np.nan)).groupby(code).ffill().shift(1).to_numpy()
    last_on_ts[~same] = np.nan
    off_dur = ts - last_on_ts
    cold = start & (np.isnan(off_dur) | (off_dur > 48))
    warm = start & (off_dur > 8) & (off_dur <= 48)
    hot = start & (off_dur <= 8)

    # hour-to-hour ramp while continuously operating
    ramp = np.where(consec & on & prev_on, np.abs(load0 - np.r_[0.0, load0[:-1]]), np.nan)

    df["on"] = on
    df["mwh"] = load0 * op
    df["start"], df["cold"], df["warm"], df["hot"] = start, cold, warm, hot
    df["ramp"] = ramp
    df["load_on"] = np.where(on & (load0 > 0), load0, np.nan)
    hr = df["hour"].to_numpy()
    df["mwh_night"] = np.where(hr <= 5, df["mwh"], 0.0)
    df["mwh_peak"] = np.where((hr >= 14) & (hr <= 19), df["mwh"], 0.0)
    for k in ("so2", "co2", "nox", "heat", "steam"):
        df[k] = df[k].astype("float64")
    df["co2_zero_load"] = np.where(on & (load0 <= 0), df["co2"].fillna(0), 0.0)
    df["op_zero_load"] = np.where(on & (load0 <= 0), op, 0.0)

    g = df.groupby("ucode", sort=True, observed=True)
    q = g.agg(
        n_rows=("op", "size"), op_hours=("op", "sum"), n_on=("on", "sum"),
        mwh=("mwh", "sum"), load_max=("load_on", "max"), load_mean=("load_on", "mean"),
        load_std=("load_on", "std"), ramp_mean=("ramp", "mean"), ramp_p95=("ramp", lambda s: s.quantile(0.95)),
        starts=("start", "sum"), starts_cold=("cold", "sum"), starts_warm=("warm", "sum"),
        starts_hot=("hot", "sum"), heat=("heat", "sum"), co2=("co2", "sum"), so2=("so2", "sum"),
        nox=("nox", "sum"), steam=("steam", "sum"), n_co2=("co2", "count"), n_so2=("so2", "count"),
        n_nox=("nox", "count"), n_heat=("heat", "count"), n_load=("load_on", "count"),
        mwh_night=("mwh_night", "sum"), mwh_peak=("mwh_peak", "sum"),
        co2_zero_load=("co2_zero_load", "sum"), op_zero_load=("op_zero_load", "sum"),
        ts_first=("ts", "min"), ts_last=("ts", "max"),
    )
    qs = g["load_on"].quantile([0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).unstack()
    qs.columns = [f"load_p{int(round(c * 100)):02d}" for c in qs.columns]
    q = q.join(qs)

    # part-load vs high-load efficiency relative to the quarter p99 load
    p99 = q["load_p99"].reindex(code).to_numpy()
    rel = np.where(p99 > 0, load0 / p99, np.nan)
    valid = on & (load0 > 0) & df["heat"].notna().to_numpy()
    for name, mask in (("hi", valid & (rel >= 0.8)), ("lo", valid & (rel < 0.5))):
        df["_m"] = mask
        df["_h"] = np.where(mask, df["heat"], 0.0)
        df["_w"] = np.where(mask, df["mwh"], 0.0)
        df["_c"] = np.where(mask, df["co2"].fillna(0), 0.0)
        df["_x"] = np.where(mask, df["nox"].fillna(0), 0.0)
        a = df.groupby("ucode", observed=True)[["_m", "_h", "_w", "_c", "_x"]].sum()
        a.columns = [f"{c}_{name}" for c in ("hours", "heat", "mwh", "co2", "nox")]
        q = q.join(a)
    on_ts = df.loc[df["on"], ["ucode", "ts"]].groupby("ucode", observed=True)["ts"].agg(["min", "max"])
    q["ts_first_on"], q["ts_last_on"] = on_ts["min"], on_ts["max"]

    # measurement-method shares (kept by name; values differ across years)
    inds = {}
    for col in ("co2_ind", "so2_ind", "noxr_ind", "heat_ind"):
        ct = pd.crosstab(df["ucode"], df[col].astype(object).fillna("NA"))
        inds[col] = {str(k): int(v) for k, v in ct.sum().items()}
        for lab in ct.columns:
            key = re.sub(r"[^a-z]+", "", str(lab).lower())[:12] or "na"
            if key in ("measured", "calculated", "substitute", "measuredands", "lme", "other"):
                q[f"{col}_{key}"] = ct[lab]
    last = df.groupby("ucode", observed=True)[ATTRS + ["uid", "facility_id", "unit_id"]].last()
    first = df.groupby("ucode", observed=True)[["fuel1", "so2_ctrl", "nox_ctrl"]].first().add_suffix("_first")
    q = q.join(last.astype(str)).join(first.astype(str))
    q["quarter"] = tag

    m = df.groupby(["ucode", "month"], observed=True).agg(
        n_rows=("op", "size"), op_hours=("op", "sum"), mwh=("mwh", "sum"), heat=("heat", "sum"),
        co2=("co2", "sum"), so2=("so2", "sum"), nox=("nox", "sum"), starts=("start", "sum"),
        load_max=("load_on", "max")).reset_index()
    m["uid"] = df["uid"].cat.categories[m["ucode"]].astype(str)
    m["quarter"] = tag
    return q.reset_index(drop=True), m.drop(columns="ucode"), inds


def run_member(zname: str, member: str) -> dict:
    t0 = time.time()
    tag = re.search(r"(\d{4}q\d)", member).group(1)
    qf, mf = OUT / f"unit_quarter_{tag}.parquet", OUT / f"unit_month_{tag}.parquet"
    if qf.exists() and mf.exists() and (OUT / f"indicators_{tag}.json").exists():
        return {"tag": tag, "skipped": True}
    df = read_member(RAW / zname, member)
    n = len(df)
    q, m, inds = reduce_quarter(df, tag)
    del df
    q.to_parquet(qf, index=False)
    m.to_parquet(mf, index=False)
    (OUT / f"indicators_{tag}.json").write_text(json.dumps(inds, ensure_ascii=False), encoding="utf-8")
    return {"tag": tag, "rows": n, "units": len(q), "sec": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for zp in sorted(RAW.glob("epacems-*.zip")):
        with zipfile.ZipFile(zp) as z:
            jobs += [(zp.name, i.filename, i.file_size) for i in z.infolist() if i.filename.endswith(".csv")]
    if a.only:
        jobs = [j for j in jobs if a.only in j[1]]
    if not jobs:
        ap.error(f"No matching quarterly CSV members found in {RAW}")
    jobs.sort(key=lambda j: -j[2])  # big files first
    log = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(run_member, j[0], j[1]): j for j in jobs}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # keep going; report at the end
                r = {"tag": futs[f][1], "error": repr(e)}
            log.append(r)
            print(json.dumps(r), flush=True)
    (OUT / "_extract_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    bad = [r for r in log if "error" in r]
    print(f"done: {len(log)} members, {len(bad)} errors", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
