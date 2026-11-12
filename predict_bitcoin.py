"""
Predict tomorrow's Bitcoin close from recent daily prices.

Needs btc_model.pkl (run sideproject.py first) and a CSV of daily prices with
date, open, high, low, close columns covering at least the last 51 consecutive days.

Run:  python predict_bitcoin.py            (uses BTC.csv)
      python predict_bitcoin.py my.csv
"""

import pickle
import sys

import numpy as np

from features import FEATURES, WARMUP_DAYS, build_features, load_prices


def predict_next_close(prices, model_path="btc_model.pkl"):
    """Return (date of last close, last close, predicted next-day close)."""
    feats = build_features(prices)
    last = feats.iloc[-1]
    if feats[FEATURES].iloc[-1].isna().any():
        raise ValueError(f"Need at least {WARMUP_DAYS + 1} consecutive days of prices.")
    model = pickle.load(open(model_path, "rb"))
    pred_ret = model.predict(feats[FEATURES].iloc[[-1]])[0]
    return last["date"].date(), last["close"], last["close"] * np.exp(pred_ret)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "BTC.csv"
    day, close, predicted = predict_next_close(load_prices(path))
    change = (predicted / close - 1) * 100
    print(f"Last close ({day}): ${close:,.2f}")
    print(f"Predicted next close: ${predicted:,.2f} ({change:+.2f}%)")
