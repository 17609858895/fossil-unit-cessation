"""Backtest the four primary learners and their equal-weight log-odds average."""
import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import hazard as lath
from ops_lib import RES, design, emulate_origin, horizon_label, load_landmark

HS = (1, 3, 5)
BASES = ["Hazard (pooled)", "Hazard (recency tau=2)", "Classifier LightGBM", "Classifier elastic-net"]


def lin_clf():
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.05, max_iter=3000, random_state=20261119))


def base_predictions(d, X, origin):
    """Fit every base learner with information at `origin`; return {name: {H: p}} for the origin cohort."""
    tr, te = emulate_origin(d, origin)
    Xtr, Xte = X.loc[tr.index].values, X.loc[te.index].values
    P = {}
    for name, tau in [("Hazard (pooled)", np.inf), ("Hazard (recency tau=2)", 2.0)]:
        m = lath.fit_hazard(Xtr, tr, origin, tau)
        S = lath.surv_curve(m, Xte)
        P[name] = {H: 1 - S[:, H] for H in HS}
    for H in HS:
        ok, ylab = horizon_label(tr, origin, H)
        y = ylab[ok]
        if ok.sum() < 500 or y.sum() < 20 or y.sum() == len(y):
            continue
        P.setdefault("Classifier LightGBM", {})[H] = lath.lgbm().fit(Xtr[ok], y).predict_proba(Xte)[:, 1]
        P.setdefault("Classifier elastic-net", {})[H] = lin_clf().fit(Xtr[ok], y).predict_proba(Xte)[:, 1]
    return P, te


def logit(p):
    p = np.clip(p, 1e-5, 1 - 1e-5)
    return np.log(p / (1 - p))


def score(y, p, cap):
    ok = ~np.isnan(y)
    y, p, c = y[ok].astype(int), np.clip(p[ok], 1e-6, 1 - 1e-6), cap[ok]
    return {"n": int(len(y)), "events": int(y.sum()), "auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "obs": int(y.sum()), "exp": float(p.sum()),
            "obs_gw": float((c * y).sum() / 1000), "exp_gw": float((c * p).sum() / 1000)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--origins", type=int, nargs="+", default=list(range(2008, 2023)))
    args = parser.parse_args()
    data = load_landmark()
    features = design(data, encoding="no_period")
    rows = []
    for origin in args.origins:
        print(f"Forecast origin {origin}", flush=True)
        components, cohort = base_predictions(data, features, origin)
        for h in HS:
            observed = cohort[f"y{h}"].to_numpy()
            if np.isnan(observed).mean() > 0.05 or np.nansum(observed) < 5:
                continue
            predictions = {m: components[m][h] for m in BASES if h in components.get(m, {})}
            if len(predictions) == len(BASES):
                logits = np.column_stack([logit(predictions[m]) for m in BASES])
                predictions["Equal-weight logit pool"] = 1 / (1 + np.exp(-logits.mean(axis=1)))
            for population, use in [("Any operation", np.ones(len(cohort), dtype=bool)),
                                    ("Fourth-quarter operation", cohort.op_q4.eq(1).to_numpy())]:
                for model, probabilities in predictions.items():
                    row = score(observed[use], probabilities[use], cohort.cap.to_numpy()[use])
                    rows.append(dict(origin=origin, H=h, population=population, model=model, **row))
    if not rows:
        raise ValueError("No evaluable origin/horizon combinations. Use the 2008–2022 origins.")
    pd.DataFrame(rows).to_csv(RES / "backtest.csv", index=False)


if __name__ == "__main__":
    main()
