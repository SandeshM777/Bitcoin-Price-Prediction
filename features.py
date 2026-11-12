"""
Feature building shared by training (sideproject.py) and prediction (predict_bitcoin.py).

Every feature for day t uses only prices up to and including day t's close.
The target is the NEXT day's log return, log(close[t+1] / close[t]), so nothing
the model sees comes from the day it is predicting.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10", "ret_20",
    "vol_10", "vol_20",
    "ma_gap_20", "ma_gap_50",
    "hl_range", "co_ret",
]
WARMUP_DAYS = 50  # the longest lookback (50-day moving average)


def load_prices(path="BTC.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.drop(columns=["ticker"], errors="ignore")
    return df.sort_values("date").reset_index(drop=True)


def _segment_features(seg):
    seg = seg.copy()
    logc = np.log(seg["close"])
    seg["ret_1"] = logc.diff()
    for k in (2, 3, 5, 10, 20):
        seg[f"ret_{k}"] = logc - logc.shift(k)
    seg["vol_10"] = seg["ret_1"].rolling(10).std()
    seg["vol_20"] = seg["ret_1"].rolling(20).std()
    seg["ma_gap_20"] = seg["close"] / seg["close"].rolling(20).mean() - 1
    seg["ma_gap_50"] = seg["close"] / seg["close"].rolling(50).mean() - 1
    seg["hl_range"] = (seg["high"] - seg["low"]) / seg["close"]
    seg["co_ret"] = np.log(seg["close"] / seg["open"])
    # Target: tomorrow's log return. NaN on the last day of each segment.
    seg["target"] = logc.shift(-1) - logc
    return seg


def build_features(df):
    """Add features and the next-day target.

    The data has a 49-day hole (2024-12-09 to 2025-01-27). Features are computed
    separately inside each run of consecutive days, so no return or rolling window
    ever spans a gap.
    """
    gap = df["date"].diff().dt.days.fillna(1) > 1
    segment = gap.cumsum()
    parts = [_segment_features(seg) for _, seg in df.groupby(segment, sort=True)]
    return pd.concat(parts).sort_values("date").reset_index(drop=True)
