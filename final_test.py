"""
Score a frozen model once on the fresh holdout (2025-09-27 onward).

Run this only after every development choice is final, and commit first so the log
records exactly which code was scored:

    python final_test.py --model final_model.pkl --note "what this model is" --confirm

The model file is a pickle of {"model": fitted estimator, "features": [column names],
"description": text}. Each run is appended to holdout_log.csv, and the result to
holdout_results.json.
"""

import argparse
import json
import pickle
import sys

import numpy as np
import pandas as pd

from holdout import HOLDOUT_START, holdout_looks, load_holdout_for_final_test
from sideproject import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--note", required=True, help="what is being scored, for the log")
    ap.add_argument("--confirm", action="store_true", help="required: acknowledge this uses up a look")
    args = ap.parse_args()

    looks = holdout_looks()
    if not args.confirm:
        sys.exit(f"Refusing to run without --confirm. The holdout has been used {looks} time(s).")
    if looks:
        print(f"WARNING: the holdout has already been used {looks} time(s). "
              "Any result from here on is less trustworthy. Report every look.")

    bundle = pickle.load(open(args.model, "rb"))
    test = load_holdout_for_final_test(args.note)
    pred = bundle["model"].predict(test[bundle["features"]])

    results = [
        evaluate(bundle.get("description", "model"), test, pred),
        evaluate("Random walk (no change)", test, np.zeros(len(test))),
    ]
    up_rate = float((test["target"] > 0).mean())
    out = {
        "holdout_start": HOLDOUT_START,
        "first_target": str(test["target_date"].min().date()),
        "last_target": str(test["target_date"].max().date()),
        "days": len(test),
        "always_up_rate": up_rate,
        "results": results,
        "look_number": looks + 1,
    }
    cols = ["model", "directional_accuracy", "share_predicted_up", "mae_usd", "mape_pct", "r2_returns"]
    print(f"Holdout {out['first_target']} to {out['last_target']} ({len(test)} days)")
    print(pd.DataFrame(results)[cols].to_string(index=False, float_format=lambda v: f"{v:,.4f}"))
    print(f"Always-up baseline (share of up days): {up_rate:.4f}")
    json.dump(out, open("holdout_results.json", "w"), indent=2)


if __name__ == "__main__":
    main()
