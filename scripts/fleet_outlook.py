"""Apply the pooled hazard to the 2025 cohort with labels updated through 2026 Q3."""
import numpy as np
import pandas as pd

import hazard as lath
from ops_lib import BUFFER_Q, END_QI, RES, design, load_landmark, outcomes


def fit_predict(d, X, train_years_lt, info_qi, cohort_year, tau=None):
    tr = d[d.year < train_years_lt].copy()
    o = outcomes(tr, info_qi)
    tr["time_tr"], tr["event_tr"] = o["time"].values, o["event"].values
    origin = info_qi // 4
    tau = np.inf if tau is None else tau          # final LATCH spec: no recency weighting, no recalibration
    m = lath.fit_hazard(X.loc[tr.index].values, tr, origin, tau)
    te = d[d.year == cohort_year].copy()
    S = lath.surv_curve(m, X.loc[te.index].values, shift=0.0)
    for k in (1, 3, 5):
        te[f"p{k}"] = 1 - S[:, k]
    return te


def scenario_intervals():
    units = pd.read_csv(RES / "forecast_2026_2030_units.csv")
    previous = pd.read_csv(RES / "fleet_interval_calibration.csv")
    capacity = units.cap.to_numpy() / 1000
    rng = np.random.default_rng(20261119)
    rows = []
    for h in (1, 3, 5):
        # Each prior origin is at Q4. Require both the horizon and the confirmation period to finish.
        eligible = previous[(previous.H == h) & (previous.origin * 4 + 3 + 4*h + BUFFER_Q <= END_QI)]
        log_ratio = np.log(eligible.obs_gw.clip(lower=0.05) / eligible.exp_gw.clip(lower=0.05))
        sigma = np.sqrt((2*0.3**2 + (log_ratio**2).sum()) / (2 + len(eligible)))
        probabilities = units[f"p{h}"].to_numpy()
        draws = (rng.random((3000, len(units))) < probabilities) @ capacity
        wave = draws * np.exp(rng.normal(0, sigma, len(draws)))
        ind_lo, ind_hi = np.quantile(draws, [0.05, 0.95])
        seq_lo, seq_hi = np.quantile(wave, [0.05, 0.95])
        rows.append(dict(H=h, expected_gw=probabilities @ capacity, ind_lo=ind_lo, ind_hi=ind_hi,
                         seq_lo=seq_lo, seq_hi=seq_hi, prior_origins=len(eligible), log_sd=sigma))
    pd.DataFrame(rows).to_csv(RES / "fleet_scenario_intervals.csv", index=False)


def main():
    d = load_landmark()
    design_matrix = design(d, encoding="no_period")
    units = fit_predict(d, design_matrix, 2025, END_QI, 2025)
    fields = ["uid", "facility_id", "facility_name", "state", "fuel_g", "tech_g", "cap",
              "cf", "util", "starts_kh", "hr", "co2", "so2", "nox", "mwh", "p1", "p3", "p5"]
    units[fields].sort_values("p5", ascending=False).to_csv(RES / "forecast_2026_2030_units.csv", index=False)
    scenario_intervals()


if __name__ == "__main__":
    main()
