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
- `final_test.py`: scores a frozen model once on the holdout and logs the run

## Fresh holdout (set up November 2026)

The 2023 to 2025 test period above has now been seen, so it can't give an unbiased score for any further changes. To have one, every day from **2025-09-27** on is held out. None of the earlier work saw those days, because `BTC.csv` ends on 2025-09-26.

- **Data.** `data/btc_usd_daily_coinbase.csv` holds daily Coinbase candles from 2015-07-20 to 2026-11-11 (4,133 days, none missing), including volume. Volume is BTC traded on Coinbase only. On the days both files cover, Coinbase closes differ from `BTC.csv` by a median of 0.37%, and daily returns correlate at 0.94, most likely because the two sources close their days at different hours. New work uses Coinbase only.
- **Development.** 3,672 days, with predictions from 2015-09-08 to 2025-09-26. `holdout.load_development()` cuts the data off before the holdout, so development code never receives a holdout price. Every feature idea, model choice and tuning step is judged by walk-forward validation inside this period.
- **Holdout.** 411 days, predicting 2025-09-27 to 2026-11-11. Only `final_test.py` reads it. It refuses to run without `--confirm`, and appends each run, with its git commit, to `holdout_log.csv`. The plan is to score the final model once and report that result, whatever it is.
- **Known peek.** While checking the download, the last two closes (2026-11-10 and 2026-11-11) were printed. They show the price is lower than in September 2025. No holdout returns or model results have been looked at.

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
