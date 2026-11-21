"""Estimate annual conditional hazards and cumulative cessation risk."""
import os

import lightgbm as lgb
import numpy as np

SEED = 20261119
KMAX = 6


def lgbm(n=350):
    return lgb.LGBMClassifier(n_estimators=n, num_leaves=15, learning_rate=0.04, subsample=0.7, subsample_freq=1,
                              colsample_bytree=0.6, min_child_samples=60, reg_lambda=2.0, n_jobs=int(os.getenv("CESSATION_THREADS", "4")),
                              random_state=SEED, verbose=-1)


def person_period(X, time_, event, year, kmax=KMAX):
    Xs, ks, ys, cy = [], [], [], []
    for k in range(1, kmax + 1):
        hit = (event == 1) & (time_ > k - 1) & (time_ <= k)
        use = (time_ > k - 1) & (hit | (time_ >= k))
        Xs.append(X[use]); ks.append(np.full(use.sum(), k, "float32")); ys.append(hit[use]); cy.append(year[use] + k)
    return (np.column_stack([np.vstack(Xs), np.concatenate(ks)]), np.concatenate(ys).astype(int), np.concatenate(cy))


def surv_curve(model, X, kmax=KMAX, shift=0.0):
    S = np.ones((len(X), kmax + 1))
    for k in range(1, kmax + 1):
        z = model.predict(np.column_stack([X, np.full(len(X), k, "float32")]), raw_score=True) + shift
        S[:, k] = S[:, k - 1] * (1 - 1 / (1 + np.exp(-z)))
    return S


def fit_hazard(Xtr, tr, origin, tau):
    Xpp, ypp, cy = person_period(Xtr, tr.time_tr.values, tr.event_tr.values, tr.year.values)
    # an interval is resolved only if it ended >= 2 years before the origin (confirmation buffer)
    w = np.ones(len(ypp)) if np.isinf(tau) else np.exp(-(origin - cy) / tau)
    m = lgbm().fit(Xpp, ypp, sample_weight=w)
    return m
