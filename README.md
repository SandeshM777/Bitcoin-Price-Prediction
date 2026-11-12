# Bitcoin Price Prediction

Predicts Bitcoin's next-day return from daily price history. The evaluation works the way a trading model has to: train on the past, test once on a later period the model never saw, and compare against simple baselines.

## Results (test period 2023-01-01 to 2025-09-25, 901 days)

| Model | Directional accuracy | Days predicted "up" | MAE | RMSE | MAPE |
|---|---|---|---|---|---|
| Lasso | 52.05% | 99.6% | $946 | $1,432 | 1.64% |
| Random Forest | 51.94% | 99.7% | $947 | $1,434 | 1.64% |
| XGBoost (best by cross-validation) | 52.05% | 100.0% | $949 | $1,433 | 1.65% |
| Random walk (tomorrow = today)* | 47.95% | 0.0% | $947 | $1,436 | 1.64% |
| Always "up" | 52.05% | 100% | | | |

MAE, RMSE and MAPE are for the predicted next-day closing price in US dollars. *A no-change forecast never predicts "up", so its directional accuracy is the share of days that were not up.

**What this shows**

- **No model beats the baselines.** The models learned Bitcoin's upward drift in the training years and predict "up" on almost every test day, so their directional accuracy matches the share of up days (52.05%). A one-sided binomial test of the best model against that rate gives p = 0.51.
- **Price errors match the random walk.** The models' MAPE (1.65% for XGBoost) is the same as simply assuming tomorrow's close equals today's (1.64%).
- **R² on price levels is misleading here.** Every model, including the random walk, scores 0.998, because tomorrow's price is almost always close to today's. R² on returns is close to zero (0.0019 for XGBoost).
- Daily open, high, low and close prices alone do not carry a usable next-day signal in this data.

## What changed (November 2026)

The first version reported R² near 0.9998, but its evaluation leaked future information:

1. It predicted the same day's close from that day's high and low, which are only known once the day is over.
2. The 20-day moving average feature included the close being predicted.
3. The train/test split was random, so the model trained on days after the ones it was tested on.
4. Outlier capping and the square-root transform were fitted on the whole dataset, test days included. The capping also clipped every price above the 95th percentile to a single value.

The fixes:

- Predict the **next** day's return using only information available at today's close.
- Split by date and test once on the most recent period.
- Tune hyperparameters with walk-forward cross-validation inside the training years.
- Fit scaling inside the model pipeline, so it only ever sees training data.
- Report results next to a random-walk baseline and an always-up baseline.

The original notebook is kept for reference and carries a note explaining the leak.

## Method

- **Target.** Next-day log return, log(close[t+1] / close[t]). The predicted price is today's close multiplied by exp(predicted return).
- **Features** (all known at day t's close). Log returns over 1, 2, 3, 5, 10 and 20 days, 10- and 20-day volatility, distance from the 20- and 50-day moving averages, the day's high-low range, and the open-to-close return.
- **Data gap.** The data is missing 49 days, from 2024-12-09 to 2025-01-27. Features are computed separately inside each run of consecutive days, so no return or rolling window spans the gap.
- **Split.** Train 2010-09-04 to 2022-12-31 (4,502 days). Test 2023-01-01 to 2025-09-25 (901 days), used once.
- **Tuning.** Grid search with 5 walk-forward folds (`TimeSeriesSplit`), scored on mean absolute error. The final model is chosen by cross-validation score, not by test results.
- **Models.** Lasso on degree-2 polynomial features, Random Forest, and XGBoost.

## Files

- `features.py`: feature and target construction, shared by training and prediction
- `sideproject.py`: trains, tunes and evaluates the models, then saves `btc_model.pkl` and `results.json`
- `predict_bitcoin.py`: predicts the next day's close from recent daily prices
- `results.json`: the numbers in the table above
- `Bitcoin_Price_Prediction_using_Machine_Learning.ipynb`: the original exploratory notebook (leaky evaluation, kept for reference)
- `BTC.csv`: daily BTC open, high, low and close prices from 2010-07-17 to 2025-09-26
- `fetch_data.py`: downloads daily BTC-USD candles with volume from Coinbase into `data/`
- `holdout.py`: the development/holdout split and the rules that protect it
- `fetch_external.py`: downloads ETH, FRED market data, Fear & Greed and Binance funding rates into `data/external/`
- `extra_features.py`: the extra feature groups and their availability lags
- `experiments.py`: walk-forward feature experiments on development data, results in `experiments/`
- `final_test.py`: scores the chosen configuration once on the holdout and logs the run

## Fresh holdout (set up November 2026)

The 2023 to 2025 test period above has now been seen, so it can't give an unbiased score for any further changes. To have one, every day from **2025-09-27** on is held out. None of the earlier work saw those days, because `BTC.csv` ends on 2025-09-26.

- **Data.** `data/btc_usd_daily_coinbase.csv` holds daily Coinbase candles from 2015-07-20 to 2026-11-11 (4,133 days, none missing), including volume. Volume is BTC traded on Coinbase only. On the days both files cover, Coinbase closes differ from `BTC.csv` by a median of 0.37%, and daily returns correlate at 0.94, most likely because the two sources close their days at different hours. New work uses Coinbase only.
- **Development.** 3,672 days, with predictions from 2015-09-08 to 2025-09-26. `holdout.load_development()` cuts the data off before the holdout, so development code never receives a holdout price. Every feature idea, model choice and tuning step is judged by walk-forward validation inside this period.
- **Holdout.** 411 days, predicting 2025-09-27 to 2026-11-11. Only `final_test.py` reads it. It refuses to run without `--confirm`, and appends each run, with its git commit, to `holdout_log.csv`. The plan is to score the final model once and report that result, whatever it is.
- **Known peek.** While checking the download, the last two closes (2026-11-10 and 2026-11-11) were printed. They show the price is lower than in September 2025. No holdout returns or model results have been looked at.

## Feature experiments (development data only, November 2026)

`experiments.py` adds five groups of outside information through `extra_features.py`, each with a conservative availability lag so a row only uses values that were public by that day's close:

- **volume:** Coinbase BTC volume, relative to recent averages
- **eth:** ETH-USD returns, plus ETH's 5-day return relative to BTC
- **macro:** Nasdaq, VIX, the US dollar index and the 10-year Treasury yield, from FRED
- **sentiment:** the Crypto Fear & Greed Index
- **funding:** Binance BTCUSDT perpetual funding rates

Validation is walk-forward. From 2020-01-01 the models are refit every quarter on all earlier data, then predict that quarter, giving 2,096 validation days up to 2025-09-26. The decision rule and model settings were written into `experiments.py` before the first run.

### Direction (always-up rate in validation: 51.29%)

| Features | Model | AUC (95% CI) | Accuracy | Days predicted up |
|---|---|---|---|---|
| price+eth | xgboost | 0.544 (0.519 to 0.569) | 53.9% | 67% |
| price+eth | logistic | 0.542 (0.518 to 0.566) | 52.3% | 89% |
| all | xgboost | 0.540 (0.516 to 0.566) | 54.1% | 60% |
| price+funding | xgboost | 0.537 (0.512 to 0.562) | 52.5% | 55% |
| price+macro | xgboost | 0.536 (0.512 to 0.561) | 52.8% | 64% |
| price | xgboost | 0.535 (0.510 to 0.561) | 53.4% | 68% |
| price | logistic | 0.534 (0.510 to 0.559) | 51.2% | 91% |
| price+volume | logistic | 0.533 (0.511 to 0.558) | 51.8% | 76% |
| price+sentiment | xgboost | 0.533 (0.508 to 0.558) | 52.4% | 61% |
| price+volume | xgboost | 0.529 (0.504 to 0.555) | 51.5% | 70% |
| price+sentiment | logistic | 0.529 (0.503 to 0.553) | 51.1% | 91% |
| price+funding | logistic | 0.528 (0.502 to 0.552) | 51.5% | 90% |
| price+macro | logistic | 0.525 (0.500 to 0.549) | 51.7% | 89% |
| all | logistic | 0.524 (0.500 to 0.549) | 51.9% | 74% |

### Volatility (absolute next-day return)

| Model | MAE | R² | Rank correlation |
|---|---|---|---|
| baseline: 20-day average | 0.01630 | 0.034 | 0.246 |
| HAR linear (1, 5, 20-day averages) | 0.01662 | 0.054 | 0.239 |
| xgboost price+volume | 0.01670 | 0.030 | 0.247 |
| xgboost all | 0.01669 | 0.053 | 0.256 |

### What the experiments show

- **Direction: a small edge that passes the rule.** The top configuration is price+eth with xgboost, at AUC 0.544 (95% CI 0.519 to 0.569) and 53.9% accuracy against the 51.3% always-up rate. Its AUC is above 0.5 in every validation year, but only barely in 2025 (0.507).
- **Most of that edge comes from price history plus quarterly refitting.** Price features alone reach AUC 0.535, and ETH adds about 0.01, which is within the noise. Volume, macro, sentiment and funding don't help beyond price.
- **The edge is small.** Log-loss skill is about 0.4%, so the predicted probabilities are only slightly better than predicting the base rate every day. Picking the best of 14 configurations also flatters the winner, and the confidence intervals don't adjust for that.
- **Volatility: no model beats the simple baseline on the pre-chosen metric (MAE).** The trailing 20-day average already captures that big moves cluster together (rank correlation 0.25). The fitted models explain a little more variance (R² up to 0.054 against 0.034), but they were trained on squared error while MAE was the scoring metric chosen in advance. That metric isn't being changed after seeing the results.

The top direction configuration was then scored once on the fresh holdout (below).

## Holdout result (scored once, 2026-11-12)

The pre-registered winner, price + ETH features with XGBoost, was scored once on the fresh holdout. It was refit each quarter on all earlier data, as in validation. The run is logged in `holdout_log.csv` against commit `7777ffe`, which had no uncommitted changes.

| Holdout (2025-09-27 to 2026-11-11, 411 days) | Result |
|---|---|
| AUC | 0.488 (95% CI 0.434 to 0.546) |
| Directional accuracy | 48.9% |
| Always-up rate | 48.7% |
| Days predicted "up" | 60% |
| Log-loss skill vs. the base rate | -1.2% |

**The validation edge did not survive.** On new data the model ranks up days no better than chance (AUC 0.488), its accuracy matches always-up, and its probabilities are slightly worse than predicting the base rate every day. Validation had already hinted at this: 2025, the most recent validation year, was the weakest at AUC 0.507. The holdout also had more down days than up days, while the model still predicted "up" 60% of the time.

**Conclusion.** Across a leakage-free rebuild, five extra data sources and a sealed holdout, there's no reliable next-day directional edge in these features. The small validation edge is what picking the best of 14 configurations on one period can produce. The holdout has now been used, so testing any further idea needs a new untouched period, starting with data after 2026-11-11.

## Run it

```
pip install -r requirements.txt
python sideproject.py
python predict_bitcoin.py
```

## Next steps

- Add information beyond price, such as trading volume or other assets.
- Retrain on a rolling window through the test period instead of training once.
- Turn predictions into a backtest that includes transaction costs.

This project is for learning only and is not financial advice.
