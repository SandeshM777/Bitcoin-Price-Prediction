"""
Feature experiments on development data only (predictions up to 2025-09-26).

Validation: walk-forward. Each quarter from 2020-01-01 is predicted by a model trained
only on rows whose target date is before that quarter (an expanding window), then the model
is refit for the next quarter. That gives about 23 refits and about 2,100 validation days.

Two tasks
1. Direction: will tomorrow's return be positive? Scored by AUC (does the model rank up days
   above down days, regardless of the overall up-trend), accuracy against the always-up
   rate, and log-loss skill against predicting the training up-rate every day.
2. Volatility: how large will tomorrow's move be (the absolute log return)? Scored by MAE,
   R² and rank correlation against a trailing 20-day average of absolute returns.

Decision rule, written down before any run
- Direction: the configuration with the highest validation AUC is the candidate. It counts as
  an edge only if its bootstrap 95% interval for AUC is entirely above 0.5 AND its accuracy
  beats the always-up rate. Otherwise the conclusion is "no reliable directional edge".
- Volatility: the model with the lowest validation MAE is the candidate. It counts as skill
  only if it beats the 20-day baseline on MAE and on rank correlation.
- Model settings are fixed below and were not tuned on validation results.
- Every configuration tried is reported, including the ones that lose.

Run:  python experiments.py
Writes: experiments/*.csv
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from extra_features import GROUPS, add_all
from features import FEATURES
from holdout import HOLDOUT_START, load_development

warnings.filterwarnings("ignore")
VAL_START = "2020-01-01"
SEED = 42
OUT = "experiments"

FEATURE_SETS = {
    "price": FEATURES,
    "price+volume": FEATURES + GROUPS["volume"],
    "price+eth": FEATURES + GROUPS["eth"],
    "price+macro": FEATURES + GROUPS["macro"],
    "price+sentiment": FEATURES + GROUPS["sentiment"],
    "price+funding": FEATURES + GROUPS["funding"],
    "all": FEATURES + sum(GROUPS.values(), []),
}


def direction_models():
    return {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(C=0.05, max_iter=2000)),
        "xgboost": XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.02, subsample=0.8,
                                 colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0,
                                 random_state=SEED, n_jobs=-1, eval_metric="logloss"),
    }


def quarters(dates):
    starts = pd.date_range(VAL_START, dates.max() + pd.Timedelta(days=1), freq="QS")
    return list(zip(starts, list(starts[1:]) + [dates.max() + pd.Timedelta(days=1)]))


def walk_forward(data, cols, make_model, target, kind):
    preds = pd.Series(np.nan, index=data.index)
    for q_start, q_end in quarters(data["target_date"]):
        train = data[data["target_date"] < q_start]
        test = data[(data["target_date"] >= q_start) & (data["target_date"] < q_end)]
        if test.empty:
            continue
        model = make_model()
        model.fit(train[cols], train[target])
        preds.loc[test.index] = (model.predict_proba(test[cols])[:, 1] if kind == "clf"
                                 else model.predict(test[cols]))
    return preds


def bootstrap_auc(y, p, n=1000):
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(y))
    stats = []
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        if y[s].min() != y[s].max():
            stats.append(roc_auc_score(y[s], p[s]))
    return np.percentile(stats, [2.5, 97.5])


def main():
    os.makedirs(OUT, exist_ok=True)
    data = add_all(load_development(), cutoff=HOLDOUT_START)
    assert data["target_date"].max() < pd.Timestamp(HOLDOUT_START)
    data["up"] = (data["target"] > 0).astype(int)
    data["abs_target"] = data["target"].abs()
    data["absret_ma5"] = data["ret_1"].abs().rolling(5).mean()
    data["absret_ma20"] = data["ret_1"].abs().rolling(20).mean()
    val = data["target_date"] >= VAL_START
    print(f"Validation: {data.loc[val, 'target_date'].min().date()} to "
          f"{data.loc[val, 'target_date'].max().date()} ({val.sum()} days), refit every quarter\n")

    # ---- Direction ----
    y = data.loc[val, "up"].to_numpy()
    up_rate = y.mean()
    rows, yearly = [], []
    # Baseline log loss: predict each quarter's training up-rate every day.
    base_p = walk_forward(data, ["ret_1"], lambda: _ConstantRate(), "up", "clf")[val].to_numpy()
    for set_name, cols in FEATURE_SETS.items():
        for model_name in direction_models():
            p = walk_forward(data, cols, lambda m=model_name: direction_models()[m], "up", "clf")[val].to_numpy()
            lo, hi = bootstrap_auc(y, p)
            rows.append({
                "features": set_name, "model": model_name, "n_features": len(cols),
                "auc": roc_auc_score(y, p), "auc_ci_low": lo, "auc_ci_high": hi,
                "accuracy": ((p > 0.5) == y).mean(), "always_up_rate": up_rate,
                "share_predicted_up": (p > 0.5).mean(),
                "logloss_skill": 1 - log_loss(y, p) / log_loss(y, base_p),
            })
            years = data.loc[val, "target_date"].dt.year.to_numpy()
            for yr in np.unique(years):
                m = years == yr
                yearly.append({"features": set_name, "model": model_name, "year": int(yr),
                               "auc": roc_auc_score(y[m], p[m])})
            print(f"direction  {set_name:<16} {model_name:<9} AUC {rows[-1]['auc']:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]  acc {rows[-1]['accuracy']:.3f}")
    direction = pd.DataFrame(rows).sort_values("auc", ascending=False)
    direction.to_csv(f"{OUT}/direction_results.csv", index=False)
    pd.DataFrame(yearly).to_csv(f"{OUT}/direction_auc_by_year.csv", index=False)

    # ---- Volatility ----
    yv = data.loc[val, "abs_target"].to_numpy()
    har_cols = ["absret_1", "absret_ma5", "absret_ma20"]
    data["absret_1"] = data["ret_1"].abs()
    vol_models = {
        "baseline: 20-day average": None,
        "HAR linear (1, 5, 20-day averages)": (har_cols, lambda: make_pipeline(SimpleImputer(strategy="median"), LinearRegression())),
        "xgboost price+volume": (FEATURE_SETS["price+volume"] + har_cols, lambda: _xgb_reg()),
        "xgboost all": (FEATURE_SETS["all"] + har_cols, lambda: _xgb_reg()),
    }
    vrows = []
    for name, spec in vol_models.items():
        if spec is None:
            pv = data.loc[val, "absret_ma20"].to_numpy()
        else:
            cols, make = spec
            pv = walk_forward(data, cols, make, "abs_target", "reg")[val].to_numpy()
        vrows.append({"model": name, "mae": mean_absolute_error(yv, pv), "r2": r2_score(yv, pv),
                      "rank_corr": spearmanr(yv, pv).statistic})
        print(f"volatility {name:<36} MAE {vrows[-1]['mae']:.5f}  R² {vrows[-1]['r2']:.3f}  "
              f"rank corr {vrows[-1]['rank_corr']:.3f}")
    pd.DataFrame(vrows).to_csv(f"{OUT}/volatility_results.csv", index=False)
    print(f"\nAlways-up rate in validation: {up_rate:.4f}. Saved results to {OUT}/")


class _ConstantRate:
    """Predicts the training up-rate every day (the log-loss baseline)."""
    def fit(self, X, y):
        self.rate = float(np.mean(y))
        return self

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.rate), np.full(len(X), self.rate)])


def _xgb_reg():
    return XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=20, reg_lambda=5.0,
                        random_state=SEED, n_jobs=-1)


if __name__ == "__main__":
    main()
