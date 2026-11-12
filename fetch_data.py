"""
Download daily BTC-USD candles (open, high, low, close, volume) from Coinbase's public API.

Coinbase daily candles run 00:00 to 24:00 UTC and start on 2015-07-20. Only complete days
are kept, so today's partial candle is dropped. Volume is BTC traded on Coinbase only, not
the whole market.

Run:  python fetch_data.py
Writes: data/btc_usd_daily_coinbase.csv and data/fetch_info.json
"""

import json
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
FIRST_DAY = datetime(2015, 7, 20, tzinfo=timezone.utc)
OUT = "data/btc_usd_daily_coinbase.csv"
MAX_CANDLES = 300  # Coinbase's limit per request


def fetch_window(start, end):
    params = {"granularity": 86400, "start": start.isoformat(), "end": end.isoformat()}
    for attempt in range(5):
        r = requests.get(URL, params=params, headers={"User-Agent": "btc-research-script"}, timeout=30)
        if r.status_code == 429:  # rate limited, back off and retry
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Coinbase kept rate limiting the window starting {start.date()}")


def main():
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    rows, start = [], FIRST_DAY
    while start < today:
        end = min(start + timedelta(days=MAX_CANDLES - 1), today - timedelta(days=1))
        rows += fetch_window(start, end)
        start = end + timedelta(days=1)
        time.sleep(0.4)

    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
    df = (df[df["date"] < today.replace(tzinfo=None)]   # complete days only
          .drop_duplicates("date").sort_values("date")
          [["date", "open", "high", "low", "close", "volume"]])

    gaps = df["date"].diff().dt.days.fillna(1)
    missing = int((gaps - 1).clip(lower=0).sum())
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), "non-positive price found"
    assert (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all(), "high below another price"
    assert (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all(), "low above another price"

    df.to_csv(OUT, index=False, date_format="%Y-%m-%d")
    info = {
        "source": URL,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "first_day": str(df["date"].min().date()),
        "last_complete_day": str(df["date"].max().date()),
        "rows": len(df),
        "missing_days": missing,
    }
    json.dump(info, open("data/fetch_info.json", "w"), indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
