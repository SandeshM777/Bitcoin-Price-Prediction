"""
Download the extra data used in the feature experiments, through the last complete UTC day.

- ETH-USD daily candles, Coinbase public API (from 2016-05-18)
- Nasdaq Composite, VIX, broad US dollar index and 10-year Treasury yield, FRED (St. Louis Fed)
- Crypto Fear & Greed Index, alternative.me (from 2018-02-01)
- BTCUSDT perpetual futures funding rates, Binance public API (from 2019-09-10, every 8 hours)

Run:  python fetch_external.py
Writes: data/external/*.csv
"""

import io
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

OUT = "data/external"
UA = {"User-Agent": "Mozilla/5.0 (btc-research-script)"}
TODAY = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
FRED = {"nasdaq": "NASDAQCOM", "vix": "VIXCLS", "usd": "DTWEXBGS", "us10y": "DGS10"}


def get(url, **params):
    for attempt in range(5):
        r = requests.get(url, params=params, headers=UA, timeout=60)
        if r.status_code == 429:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError(f"rate limited: {url}")


def coinbase_daily(product, first_day):
    rows, start = [], first_day
    while start < TODAY:
        end = min(start + timedelta(days=299), TODAY - timedelta(days=1))
        rows += get(f"https://api.exchange.coinbase.com/products/{product}/candles",
                    granularity=86400, start=start.isoformat(), end=end.isoformat()).json()
        start = end + timedelta(days=1)
        time.sleep(0.4)
    df = pd.DataFrame(rows, columns=["time", "low", "high", "open", "close", "volume"])
    df["date"] = pd.to_datetime(df["time"], unit="s")
    return df.drop_duplicates("date").sort_values("date")[["date", "open", "high", "low", "close", "volume"]]


def fred(series):
    df = pd.read_csv(io.StringIO(get("https://fred.stlouisfed.org/graph/fredgraph.csv", id=series).text),
                     na_values=".")
    df.columns = ["date", "value"]
    return df.dropna()


def fear_greed():
    data = get("https://api.alternative.me/fng/", limit=0, format="json").json()["data"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["value"] = df["value"].astype(int)
    return df.sort_values("date")[["date", "value"]]


def binance_funding():
    rows, start = [], int(datetime(2019, 9, 1, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(TODAY.timestamp() * 1000)
    while start < end_ms:
        batch = get("https://fapi.binance.com/fapi/v1/fundingRate", symbol="BTCUSDT",
                    startTime=start, endTime=end_ms - 1, limit=1000).json()
        if not batch:
            break
        rows += batch
        start = batch[-1]["fundingTime"] + 1
        time.sleep(0.4)
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["rate"] = df["fundingRate"].astype(float)
    return df.drop_duplicates("time").sort_values("time")[["time", "rate"]]


def main():
    os.makedirs(OUT, exist_ok=True)
    out = {"eth_usd_daily_coinbase": coinbase_daily("ETH-USD", datetime(2016, 5, 18, tzinfo=timezone.utc))}
    for name, series in FRED.items():
        out[f"fred_{name}"] = fred(series)
    out["fear_greed"] = fear_greed()
    out["binance_btcusdt_funding"] = binance_funding()
    for name, df in out.items():
        key = "date" if "date" in df else "time"
        df = df[pd.to_datetime(df[key]) < TODAY.replace(tzinfo=None)]   # complete days only
        df.to_csv(f"{OUT}/{name}.csv", index=False)
        # Report the first day only, never values from the holdout period.
        print(f"{name}: {len(df)} rows, from {pd.to_datetime(df[key]).min().date()}")


if __name__ == "__main__":
    main()
