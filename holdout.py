"""
The fresh holdout: every day from 2025-09-27 on, which no earlier experiment has seen
(BTC.csv, used for all earlier work, ends on 2025-09-26).

Rules
1. Development code (feature ideas, model choice, tuning) calls load_development() only.
   It never receives a price from the holdout period, so it can't learn from it by accident.
2. Choices are made on walk-forward validation inside the development data, never on
   holdout results.
3. Only final_test.py reads the holdout (through load_all_for_final_test). Every run is appended to holdout_log.csv with the
   git commit, so the number of looks at the holdout is on record.

Known peek: while checking the download on 2026-11-12, the last two closes (2026-11-10
and 2026-11-11) were printed. They show the price is lower than in September 2025. No
returns, model results or other holdout values were looked at.
"""

import csv
import os
import subprocess
from datetime import datetime, timezone

import pandas as pd

from features import FEATURES, build_features

DATA = "data/btc_usd_daily_coinbase.csv"
HOLDOUT_START = "2025-09-27"   # first target date in the holdout
LOG = "holdout_log.csv"


def _prices():
    return pd.read_csv(DATA, parse_dates=["date"]).sort_values("date").reset_index(drop=True)


def load_development():
    """Feature rows for development. Prices from the holdout are cut off before features
    are built, so the last usable row predicts 2025-09-26."""
    prices = _prices()
    prices = prices[prices["date"] < HOLDOUT_START]
    return build_features(prices).dropna(subset=FEATURES + ["target"]).reset_index(drop=True)


def load_all_for_final_test(note):
    """Every feature row, development and holdout together, so final_test.py can refit each
    period on everything before it. Logs the access. Call only from final_test.py."""
    feats = build_features(_prices()).dropna(subset=FEATURES + ["target"]).reset_index(drop=True)
    _log_access(note, feats[feats["target_date"] >= HOLDOUT_START])
    return feats


def holdout_looks():
    if not os.path.exists(LOG):
        return 0
    with open(LOG, newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def _log_access(note, rows):
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", True
    new_file = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["accessed_at_utc", "git_commit", "uncommitted_changes", "first_target",
                        "last_target", "rows", "note"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), commit, dirty,
                    rows["target_date"].min().date(), rows["target_date"].max().date(), len(rows), note])
