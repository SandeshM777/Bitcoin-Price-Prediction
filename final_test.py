"""
Score the chosen configuration once on the fresh holdout (target dates from 2025-09-27).

Run this only after every development choice is final, and commit first so the log
records exactly which code was scored:

    python final_test.py --features price+eth --model xgboost --note "why this one" --confirm

It repeats the validation procedure: at the start of each period the model is refit on
every row whose target date is before that period, then predicts it. A finished holdout
quarter becomes training data for the next one, just as it would have in real time.
Each run is appended to holdout_log.csv, and the result is saved to holdout_results.json.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from experiments import FEATURE_SETS, _ConstantRate, bootstrap_auc, direction_models
from extra_features import add_all
from holdout import HOLDOUT_START, holdout_looks, load_all_for_final_test


def periods(last_target):
    starts = [pd.Timestamp(HOLDOUT_START)] + list(
        pd.date_range(HOLDOUT_START, last_target, freq="QS"))
    starts = sorted(set(starts))
    ends = starts[1:] + [last_target + pd.Timedelta(days=1)]
    return list(zip(starts, ends))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, choices=list(FEATURE_SETS))
    ap.add_argument("--model", required=True, choices=["logistic", "xgboost"])
    ap.add_argument("--note", required=True, help="what is being scored, for the log")
    ap.add_argument("--confirm", action="store_true", help="required: acknowledge this uses up a look")
    args = ap.parse_args()

    looks = holdout_looks()
    if not args.confirm:
        sys.exit(f"Refusing to run without --confirm. The holdout has been used {looks} time(s).")
    if looks:
        print(f"WARNING: the holdout has already been used {looks} time(s). "
              "Any result from here on is less trustworthy. Report every look.")

    data = add_all(load_all_for_final_test(f"{args.features} {args.model}: {args.note}"), cutoff=None)
    data["up"] = (data["target"] > 0).astype(int)
    cols = FEATURE_SETS[args.features]
    hold = data["target_date"] >= HOLDOUT_START

    prob = pd.Series(np.nan, index=data.index)
    base = pd.Series(np.nan, index=data.index)
    for start, end in periods(data.loc[hold, "target_date"].max()):
        train = data[data["target_date"] < start]
        test = data[(data["target_date"] >= start) & (data["target_date"] < end)]
        model = direction_models()[args.model].fit(train[cols], train["up"])
        prob.loc[test.index] = model.predict_proba(test[cols])[:, 1]
        base.loc[test.index] = _ConstantRate().fit(None, train["up"]).predict_proba(test[cols])[:, 1]

    y, p, b = data.loc[hold, "up"].to_numpy(), prob[hold].to_numpy(), base[hold].to_numpy()
    lo, hi = bootstrap_auc(y, p)
    out = {
        "features": args.features, "model": args.model,
        "first_target": str(data.loc[hold, "target_date"].min().date()),
        "last_target": str(data.loc[hold, "target_date"].max().date()),
        "days": int(hold.sum()),
        "auc": roc_auc_score(y, p), "auc_ci_95": [float(lo), float(hi)],
        "accuracy": float(((p > 0.5) == y).mean()),
        "always_up_rate": float(y.mean()),
        "share_predicted_up": float((p > 0.5).mean()),
        "logloss_skill": float(1 - log_loss(y, p) / log_loss(y, b)),
        "look_number": looks + 1,
    }
    for k, v in out.items():
        print(f"{k}: {v}")
    json.dump(out, open("holdout_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
