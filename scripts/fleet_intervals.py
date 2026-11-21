"""Calibrated prediction intervals for fleet-level retiring capacity.

Unit-level probabilities imply a Poisson-binomial total if units retire independently, but
retirements arrive in correlated waves (shared prices, rules, owners). Candidate interval rules
for retiring capacity within H years at origin O:
  (i)   independent : Monte Carlo of independent Bernoulli draws
  (ii)  backtest-ratio : (i) x empirical obs/exp ratios from ALL other origins (uses future origins -
        not deployable, shown as the ad-hoc reference)
  (iii) sequential wave-calibrated (proposed): (i) x a log-normal wave multiplier whose scale is
        estimated only from earlier origins whose H-year outcome is resolved at O (O' <= O-H-2),
        with a weak prior when few are available.
Scored by empirical coverage of nominal 90 % intervals and mean interval width across origins.
"""
from __future__ import annotations

import argparse
import hazard as lath
from pathlib import Path

import numpy as np
import pandas as pd

from ops_lib import BUFFER_Q, DP, RES, design, emulate_origin, load_landmark


SEED = 20261119
HS = (1, 3, 5)
ORIGINS = list(range(2003, 2023))


def prior_outcomes_at_origin(landmarks, operating_quarters, calibration, current_origin, horizon):
    # Once the horizon and buffer have elapsed, the earlier cohort is complete. Its
    # apparent terminal events can still be revised by later restarts.
    resolved = calibration.origin * 4 + 3 + 4 * horizon + BUFFER_Q <= current_origin * 4 + 3
    past = calibration[(calibration.H == horizon) & resolved].copy()
    last = operating_quarters.loc[operating_quarters.qi <= current_origin*4+3].groupby("uid").qi.max()
    totals = []
    for previous in past.origin:
        cohort = landmarks[landmarks.year == previous]
        known_last = cohort.uid.map(last)
        is_event = known_last.le(cohort.lm_qi + 4*horizon)
        totals.append(cohort.loc[is_event, "cap"].sum() / 1000)
    past["asof_gw"] = totals
    return past.assign(logratio=np.log(past.asof_gw.clip(lower=0.05) / past.exp_gw.clip(lower=0.05)))


def main(output_dir=RES, threads=4):
    rng = np.random.default_rng(SEED)
    d = load_landmark()
    operating_quarters = pd.read_parquet(DP / "unit_on_quarters.parquet")
    X = design(d, encoding="no_period")
    rows = []
    for O in ORIGINS:
        tr, te = emulate_origin(d, O)
        if tr.event_tr.sum() < 30:
            continue
        xx, yy, _ = lath.person_period(X.loc[tr.index].values, tr.time_tr.values, tr.event_tr.values, tr.year.values)
        m = lath.lgbm().set_params(n_jobs=threads).fit(xx, yy, sample_weight=np.ones(len(yy)))
        S = lath.surv_curve(m, X.loc[te.index].values)
        cap = te.cap.values / 1000
        for H in HS:
            y = te[f"y{H}"].values
            if np.isnan(y).mean() > 0.05:
                continue
            ok = ~np.isnan(y)
            p = 1 - S[ok, H]
            draws = (rng.random((3000, ok.sum())) < p) @ cap[ok]
            rows.append({"origin": O, "H": H, "obs_gw": float((cap[ok] * y[ok]).sum()), "exp_gw": float((cap[ok] * p).sum()),
                         "ind_lo": float(np.quantile(draws, 0.05)), "ind_hi": float(np.quantile(draws, 0.95)),
                         "draws": draws})
        print("origin", O, flush=True)
    r = pd.DataFrame(rows)
    r["logratio"] = np.log(r.obs_gw.clip(lower=0.05) / r.exp_gw.clip(lower=0.05))
    out = []
    prior_comparisons = []
    for i, row in r.iterrows():
        H, O = row.H, row.origin
        others = r[(r.H == H) & (r.origin != O)]
        mult_bt = rng.choice(np.exp(others.logratio.values), size=len(row.draws))
        prior = prior_outcomes_at_origin(d, operating_quarters, r, O, H)
        prior_comparisons.append(prior[["origin","H","obs_gw","asof_gw","exp_gw"]].assign(calibration_origin=O))
        past = prior.logratio.values
        # weak prior: sigma0 = 0.3 worth 2 pseudo-origins; posterior-mean variance of log wave multipliers
        n0, s0 = 2, 0.3
        sig = np.sqrt((n0 * s0 ** 2 + np.sum(past ** 2)) / (n0 + len(past)))
        mult_seq = np.exp(rng.normal(0, sig, size=len(row.draws)))
        bt = row.draws * mult_bt
        sq = row.draws * mult_seq
        # robust recent variant: median-centred, MAD scale over the 8 most recent resolved origins
        rec = prior.sort_values("origin").logratio.values[-8:]
        if len(rec) >= 3:
            mu_r = float(np.median(rec))
            sig_r = max(1.4826 * float(np.median(np.abs(rec - mu_r))), 0.15)
        else:
            mu_r, sig_r = 0.0, s0 * 2
        rb = row.draws * np.exp(rng.normal(mu_r, sig_r, size=len(row.draws)))
        out.append({"origin": O, "H": H, "obs_gw": row.obs_gw, "exp_gw": row.exp_gw, "n_past": len(past), "sigma": sig,
                    "ind_lo": row.ind_lo, "ind_hi": row.ind_hi,
                    "bt_lo": float(np.quantile(bt, 0.05)), "bt_hi": float(np.quantile(bt, 0.95)),
                    "seq_lo": float(np.quantile(sq, 0.05)), "seq_hi": float(np.quantile(sq, 0.95)),
                    "rob_lo": float(np.quantile(rb, 0.05)), "rob_hi": float(np.quantile(rb, 0.95)), "rob_mu": mu_r, "rob_sigma": sig_r})
    o = pd.DataFrame(out)
    output_dir.mkdir(parents=True, exist_ok=True)
    o.to_csv(output_dir / "fleet_interval_calibration.csv", index=False)
    pd.concat(prior_comparisons, ignore_index=True).to_csv(output_dir / "prior_capacity_at_each_origin.csv", index=False)
    ev = o[o.origin >= 2008]
    summ = []
    for H, g in ev.groupby("H"):
        for k, lab in [("ind", "Independent units"), ("bt", "Back-test ratio (uses future origins)"),
                       ("seq", "Sequential wave-calibrated"), ("rob", "Sequential robust (recent, MAD)")]:
            cov = ((g.obs_gw >= g[f"{k}_lo"]) & (g.obs_gw <= g[f"{k}_hi"])).mean()
            summ.append({"H": H, "rule": lab, "coverage_90": float(cov), "mean_width_gw": float((g[f"{k}_hi"] - g[f"{k}_lo"]).mean()),
                         "n_origins": int(len(g))})
    s = pd.DataFrame(summ)
    s.to_csv(output_dir / "fleet_interval_calibration_summary.csv", index=False)
    print(s.round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RES)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    main(args.output, args.threads)
