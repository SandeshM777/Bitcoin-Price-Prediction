"""
Extra feature groups for the experiments: volume, ETH, US markets, sentiment and funding rates.

Timing rule: a feature on row date d (prices up to d's 24:00 UTC close) may only use values
that were public by then. Each outside series gets an availability lag in days, so an
observation dated x counts as known from x + lag. The lags are deliberately conservative:
  - Nasdaq and VIX close at about 21:00 UTC on day x. Lag 1, so day d uses closes up to d-1.
  - The 10-year Treasury yield (FRED DGS10) is published the next business day. Lag 3.
  - The broad dollar index (FRED DTWEXBGS) comes out weekly on Mondays for the week before. Lag 8.
  - The Fear & Greed value dated x is stamped 00:00 UTC on day x. Lag 1, to be safe.
  - Binance funding settles at 00:00, 08:00 and 16:00 UTC, so day d's rates are known at d's close. Lag 0.
  - ETH-USD closes at the same 24:00 UTC as BTC. Lag 0.

`cutoff` drops every outside observation dated on or after it. Development code passes
the holdout start, so no holdout-period value reaches it.
"""

import numpy as np
import pandas as pd

EXT = "data/external"

GROUPS = {
    "volume": ["vol_chg_1", "vol_rel_20", "vol_rel_5_50"],
    "eth": ["eth_ret_1", "eth_ret_5", "ethbtc_ret_5"],
    "macro": ["ndx_ret_1", "ndx_ret_5", "vix_level", "vix_chg_5", "usd_ret_5", "us10y_chg_5"],
    "sentiment": ["fng", "fng_chg_7"],
    "funding": ["fund_1d", "fund_7d"],
}


def _read(name, cutoff, date_col="date"):
    df = pd.read_csv(f"{EXT}/{name}.csv", parse_dates=[date_col])
    if cutoff is not None:
        df = df[df[date_col] < pd.Timestamp(cutoff)]
    return df.sort_values(date_col).reset_index(drop=True)


def _attach(f, series, lag_days):
    """As-of join: each row gets the latest observation available by its date."""
    series = series.copy()
    series["available"] = series["date"] + pd.Timedelta(days=lag_days)
    series = series.drop(columns="date").sort_values("available")
    out = pd.merge_asof(f.sort_values("date"), series, left_on="date", right_on="available",
                        direction="backward")
    return out.drop(columns="available")


def add_volume(f):
    f = f.sort_values("date").copy()
    logv = np.log(f["volume"].clip(lower=1e-9))
    f["vol_chg_1"] = logv.diff()
    f["vol_rel_20"] = logv - np.log(f["volume"].rolling(20).mean())
    f["vol_rel_5_50"] = np.log(f["volume"].rolling(5).mean() / f["volume"].rolling(50).mean())
    return f


def add_eth(f, cutoff):
    eth = _read("eth_usd_daily_coinbase", cutoff)
    logc = np.log(eth["close"])
    eth = pd.DataFrame({"date": eth["date"], "eth_ret_1": logc.diff(), "eth_ret_5": logc - logc.shift(5)})
    f = f.merge(eth, on="date", how="left")
    f["ethbtc_ret_5"] = f["eth_ret_5"] - f["ret_5"]
    return f


def add_macro(f, cutoff):
    nd = _read("fred_nasdaq", cutoff)
    lg = np.log(nd["value"])
    f = _attach(f, pd.DataFrame({"date": nd["date"], "ndx_ret_1": lg.diff(), "ndx_ret_5": lg - lg.shift(5)}), 1)
    vix = _read("fred_vix", cutoff)
    f = _attach(f, pd.DataFrame({"date": vix["date"], "vix_level": vix["value"],
                                 "vix_chg_5": vix["value"] - vix["value"].shift(5)}), 1)
    usd = _read("fred_usd", cutoff)
    f = _attach(f, pd.DataFrame({"date": usd["date"], "usd_ret_5": np.log(usd["value"]).diff(5)}), 8)
    y10 = _read("fred_us10y", cutoff)
    f = _attach(f, pd.DataFrame({"date": y10["date"], "us10y_chg_5": y10["value"].diff(5)}), 3)
    return f


def add_sentiment(f, cutoff):
    fg = _read("fear_greed", cutoff)
    v = fg["value"] / 100
    return _attach(f, pd.DataFrame({"date": fg["date"], "fng": v, "fng_chg_7": v - v.shift(7)}), 1)


def add_funding(f, cutoff):
    fu = _read("binance_btcusdt_funding", cutoff, date_col="time")
    daily = fu.groupby(fu["time"].dt.floor("D"))["rate"].mean().rename("fund_1d").to_frame()
    daily["fund_7d"] = daily["fund_1d"].rolling(7, min_periods=5).mean()
    daily = daily.reset_index().rename(columns={"time": "date"})
    return _attach(f, daily, 0)


def add_all(f, cutoff):
    f = add_volume(f)
    f = add_eth(f, cutoff)
    f = add_macro(f, cutoff)
    f = add_sentiment(f, cutoff)
    f = add_funding(f, cutoff)
    return f.sort_values("date").reset_index(drop=True)
