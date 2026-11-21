"""Compare prevalent scrubber status under complete five-year follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from ops_lib import BUFFER_Q, CONTEXT, DP, DYNAMIC, LEVEL, RES, END_QI, load_landmark, outcomes

warnings.filterwarnings("ignore")
SEED = 20261119
THREADS = 2
H = 5
TRIALS = [year for year in range(2008, END_QI // 4)
          if year * 4 + 3 + 4 * H + BUFFER_Q <= END_QI]
# Two regional and national summaries incorporate full-record unit capacity and fuel.
# Retain only adjustment variables measured by the landmark.
DROP = {"so2_rate", "so2_rate_d3", "scrubber", "dsi", "scrub_new3", "st_ret3_share", "nat_ret3_share"}
CONF = [c for c in ["log_cap", "yrs_obs", "age_known", "dual_fuel", "scr", "sncr", "lnb", "baghouse", "esp",
                    "hg_aci", "rggi", "n_prog"] + LEVEL + DYNAMIC + CONTEXT if c not in DROP]


def lgb_clf():
    return lgb.LGBMClassifier(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=30,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0,
                              n_jobs=THREADS, random_state=SEED, verbose=-1)


def lgb_reg():
    return lgb.LGBMRegressor(n_estimators=300, num_leaves=15, learning_rate=0.03, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0,
                             n_jobs=THREADS, random_state=SEED, verbose=-1)


def crossfit_nuisance(X, A, Y, groups, k=5):
    e, m1, m0 = np.zeros(len(Y)), np.zeros(len(Y)), np.zeros(len(Y))
    for tr, te in GroupKFold(k).split(X, Y, groups):
        pe = lgb_clf().fit(X.iloc[tr], A[tr])
        e[te] = pe.predict_proba(X.iloc[te])[:, 1]
        XA = X.iloc[tr].assign(_A=A[tr])
        mo = lgb_clf().fit(XA, Y[tr])
        m1[te] = mo.predict_proba(X.iloc[te].assign(_A=1))[:, 1]
        m0[te] = mo.predict_proba(X.iloc[te].assign(_A=0))[:, 1]
    return e, m1, m0


def clustered_se(psi, groups):
    df = pd.DataFrame({"p": psi - psi.mean(), "g": groups})
    s = df.groupby("g")["p"].sum()
    return float(np.sqrt((s ** 2).sum()) / len(psi))


def aipw(X, A, Y, groups, trim=0.02, return_parts=False):
    e, m1, m0 = crossfit_nuisance(X, A, Y, groups)
    e = np.clip(e, trim, 1 - trim)
    psi = m1 - m0 + A * (Y - m1) / e - (1 - A) * (Y - m0) / (1 - e)
    ate = float(psi.mean())
    se = clustered_se(psi, groups)
    out = {"ate": ate, "se": se, "lo": ate - 1.96 * se, "hi": ate + 1.96 * se,
           "ipw": float(np.mean(A * Y / e) / np.mean(A / e) - np.mean((1 - A) * Y / (1 - e)) / np.mean((1 - A) / (1 - e))),
           "or": float(np.mean(m1 - m0)), "e_min": float(e.min()), "e_max": float(e.max()),
           "frac_trim": float(np.mean((e <= trim + 1e-9) | (e >= 1 - trim - 1e-9)))}
    if return_parts:
        return out, psi, e, m1, m0
    return out


def evalue_rr(rr):
    rr = max(rr, 1 / rr)
    return rr + np.sqrt(rr * (rr - 1))


def main(output_dir=RES, annual_from=None):
    output_dir.mkdir(exist_ok=True, parents=True)
    d = load_landmark()
    units = pd.read_parquet(DP / "units.parquet")[["uid", "conv_year"]]
    d = d.merge(units, on="uid", how="left")
    rows = []
    pooled = []
    for t in TRIALS:
        c = d[(d.year == t) & (d.fuel_g == "coal")].copy()
        o = outcomes(c, END_QI)
        c["Y"] = o[f"y{H}"].values
        c["Yexit"] = np.where(c.conv_year.notna() & (c.conv_year <= t + H) & (c.conv_year > t), 1.0, c["Y"])
        if c.Y.isna().any():
            raise ValueError(f"Incomplete five-year outcome in the {t} cohort")
        c["trial"] = t
        pooled.append(c)
    P = pd.concat(pooled, ignore_index=True)
    P["A"] = P["scrubber"].astype(int)
    print("pooled person-trials", len(P), "units", P.uid.nunique(), "treated share", round(P.A.mean(), 3),
          "Y rate", round(P.Y.mean(), 3))
    X = P[CONF].astype("float32").replace([np.inf, -np.inf], np.nan)
    X = pd.concat([X, pd.get_dummies(P[["tech_g"]], dtype="float32"), pd.get_dummies(P["trial"], prefix="trial", dtype="float32")], axis=1)
    G = P["facility_id"].values

    res = {}
    # The eligibility rule and the adjustment variables determine whether intermediate
    # estimates can be reused after a long fit. Each run keeps its own output directory.
    signature = {"trials": TRIALS, "n": len(P), "covariates": list(X.columns), "seed": SEED}
    stage = output_dir / "scrubber_progress.json"
    parts = output_dir / "scrubber_pooled_parts.parquet"
    state = json.loads(stage.read_text()) if stage.exists() else {}
    cached = state.get("signature") == signature
    if cached and parts.exists():
        pp = pd.read_parquet(parts)
        if not P[["uid", "trial"]].reset_index(drop=True).equals(pp[["uid","trial"]].reset_index(drop=True)):
            raise ValueError("Cohort rows differ from the intermediate scrubber fit")
        for col in ["psi", "e_hat", "m1", "m0"]:
            P[col] = pp[col].values
        res = state.get("results", {})
    def checkpoint():
        stage.write_text(json.dumps({"signature": signature, "results": res}), encoding="utf-8")
    for yname in ("Y", "Yexit"):
        if yname in res:
            continue
        Yv, Av = P[yname].values.astype(int), P.A.values
        crude = Yv[Av == 1].mean() - Yv[Av == 0].mean()
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(
            pd.concat([X.fillna(X.median()), pd.Series(Av, name="_A")], axis=1), Yv)
        r, psi, e, m1, m0 = aipw(X, Av, Yv, G, return_parts=True)
        r.update(outcome=yname, crude=float(crude), n=int(len(Yv)), treated=int(Av.sum()),
                 risk_treated=float(np.mean(m1)), risk_control=float(np.mean(m0)), logit_coef_A=float(lr.coef_[0][-1]))
        rr = max(np.mean(m1), 1e-6) / max(np.mean(m0), 1e-6)
        r["risk_ratio"] = float(rr)
        r["e_value"] = float(evalue_rr(rr))
        res[yname] = r
        print(yname, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}))
        if yname == "Y":
            P["psi"], P["e_hat"], P["m1"], P["m0"] = psi, e, m1, m0
            P[["uid","trial","psi","e_hat","m1","m0"]].to_parquet(parts, index=False)
        checkpoint()

    # ---- per-trial ATEs (sequence of trials)
    annual = output_dir / "causal_scrubber_by_trial.csv"
    if annual_from is not None:
        completed = pd.read_csv(annual_from)
        cohort_sizes = P.groupby("trial").size()
        if not completed.set_index("trial").n.sort_index().eq(cohort_sizes).all() or len(completed) != len(cohort_sizes):
            raise ValueError("Year-specific sample counts do not match the eligible cohorts")
        rows = completed.to_dict("records")
    elif cached and annual.exists():
        rows = pd.read_csv(annual).to_dict("records")
    done_years = {int(r["trial"]) for r in rows}
    for t in TRIALS:
        if t in done_years:
            continue
        s = P.trial == t
        if s.sum() < 150:
            continue
        r = aipw(X[s], P.A.values[s], P.Y.values[s].astype(int), G[s])
        r.update(trial=t, n=int(s.sum()), y_rate=float(P.Y[s].mean()), treated_share=float(P.A[s].mean()))
        rows.append(r)
        pd.DataFrame(rows).to_csv(annual, index=False)
        print("trial", t, round(r["ate"], 3), "±", round(1.96 * r["se"], 3), "n", r["n"])
    pd.DataFrame(rows).to_csv(output_dir / "causal_scrubber_by_trial.csv", index=False)

    # ---- DR-learner CATE (cross-fitted regression of the AIPW pseudo-outcome)
    cate = np.zeros(len(P))
    for tr, te in GroupKFold(5).split(X, P.psi, G):
        cate[te] = lgb_reg().fit(X.iloc[tr], P.psi.values[tr]).predict(X.iloc[te])
    P["cate"] = cate
    het = {}
    for v in ["log_cap", "cf", "hr", "st_gas_share", "starts_kh", "cf_d3"]:
        qv = pd.qcut(P[v].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        g = P.groupby(qv, observed=True)
        het[v] = {str(k): {"cate_dr": float(grp.psi.mean()), "se": clustered_se(grp.psi.values, grp.facility_id.values),
                           "n": int(len(grp)), "x_median": float(grp[v].median())} for k, grp in g}
    # best linear projection of the pseudo-outcome on standardized moderators
    Z = P[["log_cap", "cf", "hr", "st_gas_share", "starts_kh"]].copy()
    Z = (Z - Z.mean()) / Z.std()
    Z = Z.fillna(0)
    Zc = np.column_stack([np.ones(len(Z)), Z.values])
    beta = np.linalg.lstsq(Zc, P.psi.values, rcond=None)[0]
    res["blp"] = dict(zip(["intercept"] + list(Z.columns), map(float, beta)))
    res["heterogeneity"] = het

    # ---- refutations
    rng = np.random.default_rng(SEED)
    Yv, Av = P.Y.values.astype(int), P.A.values
    # placebo treatment: permute A within trial x technology strata
    Ap = P.groupby(["trial", "tech_g"])["A"].transform(lambda s: rng.permutation(s.values)).values
    if "refute_placebo" not in res:
        res["refute_placebo"] = aipw(X, Ap, Yv, G)
        checkpoint()
    # Sensitivity to an independent random covariate
    random_covariate = rng.normal(size=len(X))
    if "refute_random_cause" not in res:
        res["refute_random_cause"] = aipw(X.assign(_rand=random_covariate), Av, Yv, G)
        checkpoint()
    # subset stability (5 x 80 % plant subsamples)
    subs = res.get("subset_differences", [])
    fac = np.unique(G)
    for j in range(5):
        keep = np.isin(G, rng.choice(fac, int(0.8 * len(fac)), replace=False))
        if j >= len(subs):
            subs.append(aipw(X[keep], Av[keep], Yv[keep], G[keep])["ate"])
            res["subset_differences"] = subs
            checkpoint()
    res["refute_subset"] = {"ates": subs, "mean": float(np.mean(subs)), "sd": float(np.std(subs))}
    # Association with a prior capacity-factor change
    nco = (P["cf_d3"] < P["cf_d3"].median()).astype(int).values
    Xn = X.drop(columns=[c for c in X.columns if c.startswith("cf_d3") or c == "cf_d1"])
    res["refute_negative_control_past_cf_drop"] = aipw(Xn, Av, nco, G)
    # unadjusted vs adjusted for technology only
    res["overlap"] = {"e_q01": float(np.quantile(P.e_hat, 0.01)), "e_q99": float(np.quantile(P.e_hat, 0.99)),
                      "treated_e_mean": float(P.e_hat[P.A == 1].mean()), "control_e_mean": float(P.e_hat[P.A == 0].mean())}
    (output_dir / "causal_scrubber_summary.json").write_text(json.dumps(res, indent=1))
    P[["uid", "facility_id", "trial", "A", "Y", "Yexit", "e_hat", "m1", "m0", "psi", "cate", "log_cap", "cf", "hr",
       "st_gas_share", "starts_kh", "cf_d3", "tech_g", "state"]].to_parquet(output_dir / "causal_scrubber_units.parquet", index=False)
    for k in ("refute_placebo", "refute_random_cause", "refute_subset", "refute_negative_control_past_cf_drop", "blp"):
        v = res[k]
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RES)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--annual-from", type=Path)
    args = parser.parse_args()
    THREADS = args.threads
    main(args.output, args.annual_from)
