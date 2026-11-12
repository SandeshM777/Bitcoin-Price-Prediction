"""
Bitcoin next-day prediction with a leakage-free, time-ordered evaluation.

What it predicts: the next day's log return, from features known at today's close.
How it is tested: train on 2010-2022, test once on 2023-2025, never shuffled.
Hyperparameters: chosen by walk-forward cross-validation inside the training years.
Compared with: a random-walk baseline (tomorrow's close = today's close) and an
"always up" baseline for direction.

Run:  python sideproject.py
Writes: btc_model.pkl (best model by cross-validation) and results.json
"""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

from features import FEATURES, build_features, load_prices

warnings.filterwarnings("ignore")
TRAIN_END = "2022-12-31"
SEED = 42


def evaluate(name, test, pred_ret):
    """Score next-day predictions in return space and in dollars."""
    true_ret = test["target"].to_numpy()
    close_today = test["close"].to_numpy()
    actual_next = close_today * np.exp(true_ret)
    pred_next = close_today * np.exp(pred_ret)
    up = true_ret > 0
    hits = int(((pred_ret > 0) == up).sum())
    return {
        "model": name,
        "directional_accuracy": hits / len(test),
        "direction_hits": hits,
        "share_predicted_up": float((pred_ret > 0).mean()),
        "mae_usd": mean_absolute_error(actual_next, pred_next),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual_next, pred_next))),
        "mape_pct": mean_absolute_percentage_error(actual_next, pred_next) * 100,
        "r2_returns": r2_score(true_ret, pred_ret),
        "r2_price_level": r2_score(actual_next, pred_next),
    }


def main():
    data = build_features(load_prices()).dropna(subset=FEATURES + ["target"])
    train = data[data["date"] <= TRAIN_END]
    test = data[data["date"] > TRAIN_END]
    x_train, y_train = train[FEATURES], train["target"]
    x_test = test[FEATURES]

    print(f"Train: {train.date.min().date()} to {train.date.max().date()} ({len(train)} days)")
    print(f"Test:  {test.date.min().date()} to {test.date.max().date()} ({len(test)} days)")

    # Walk-forward CV: each fold trains on earlier days and validates on the days after.
    cv = TimeSeriesSplit(n_splits=5)
    searches = {
        "Lasso": GridSearchCV(
            make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Lasso(max_iter=20000)),
            {"lasso__alpha": [1e-5, 1e-4, 1e-3, 1e-2]},
            scoring="neg_mean_absolute_error", cv=cv, n_jobs=1),
        "Random Forest": GridSearchCV(
            RandomForestRegressor(random_state=SEED, n_jobs=-1),
            {"n_estimators": [200], "max_depth": [3, 5, 7], "min_samples_leaf": [20, 50]},
            scoring="neg_mean_absolute_error", cv=cv, n_jobs=1),
        "XGBoost": GridSearchCV(
            XGBRegressor(random_state=SEED, subsample=0.8, n_jobs=-1),
            {"n_estimators": [100, 300], "max_depth": [2, 3, 4], "learning_rate": [0.01, 0.05]},
            scoring="neg_mean_absolute_error", cv=cv, n_jobs=1),
    }

    results, cv_scores = [], {}
    for name, search in searches.items():
        search.fit(x_train, y_train)
        cv_scores[name] = -search.best_score_
        print(f"{name}: best params {search.best_params_}, CV MAE (log return) {cv_scores[name]:.5f}")
        results.append(evaluate(name, test, search.predict(x_test)))

    # Baselines
    results.append(evaluate("Random walk (no change)", test, np.zeros(len(test))))
    always_up = evaluate("Always up", test, np.full(len(test), 1e-9))

    # Pick the model by cross-validation, not by the test set.
    best_name = min(cv_scores, key=cv_scores.get)
    best = next(r for r in results if r["model"] == best_name)
    up_rate = always_up["directional_accuracy"]
    p_value = binomtest(best["direction_hits"], len(test), up_rate, alternative="greater").pvalue

    print("\nOut-of-sample results on", len(test), "test days")
    cols = ["model", "directional_accuracy", "share_predicted_up", "mae_usd", "rmse_usd", "mape_pct",
            "r2_returns", "r2_price_level"]
    print(pd.DataFrame(results)[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"\nShare of up days in the test period (always-up baseline): {up_rate:.4f}")
    print(f"Best model by CV: {best_name}. One-sided binomial p-value of its direction hits vs the "
          f"always-up rate: {p_value:.3f}")

    pickle.dump(searches[best_name].best_estimator_, open("btc_model.pkl", "wb"))
    summary = {
        "train_period": [str(train.date.min().date()), str(train.date.max().date()), len(train)],
        "test_period": [str(test.date.min().date()), str(test.date.max().date()), len(test)],
        "features": FEATURES,
        "cv_mae_log_return": cv_scores,
        "best_model_by_cv": best_name,
        "always_up_rate": up_rate,
        "best_direction_p_value_vs_always_up": p_value,
        "results": results,
    }
    json.dump(summary, open("results.json", "w"), indent=2)
    print("\nSaved btc_model.pkl and results.json")


if __name__ == "__main__":
    main()
