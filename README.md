# Bitcoin Price Prediction Project

A machine learning project that predicts Bitcoin closing prices using historical OHLC (Open, High, Low, Close) data.

## 📁 Project Files

- **`sideproject.py`** - Main training script that processes data and trains ML models
- **`predict_bitcoin.py`** - Simple prediction script to use trained models
- **`Bitcoin_Price_Prediction_using_Machine_Learning.ipynb`** - Jupyter notebook with detailed analysis and visualizations
- **`BTC.csv`** - Bitcoin historical price data

## 🚀 Quick Start

### 1. Train the Models

Run the main script to train all models:

```powershell
python sideproject.py
```

This will:
- Load and clean the Bitcoin dataset
- Apply feature engineering (outlier removal, skewness correction)
- Train three models: Lasso Regression, Random Forest, and XGBoost
- Evaluate all models and save the best one
- Generate these files:
  - `btc_model.pkl` (XGBoost model)
  - `btc_rf_model.pkl` (Random Forest model)
  - `btc_lasso_model.pkl` (Lasso model)
  - `btc_scaler.pkl` (Feature scaler)
  - `btc_imputer.pkl` (Missing value imputer)

### 2. Make Predictions

Use the trained model to predict prices:

```powershell
python predict_bitcoin.py
```

Or in your own Python code:

```python
from predict_bitcoin import predict_bitcoin_price

# Predict closing price
predicted_close = predict_bitcoin_price(
    open_price=67500.0,
    high_price=68200.0,
    low_price=67000.0,
    price_trend=67800.0  # Optional: 20-day moving average
)

print(f"Predicted Close: ${predicted_close:,.2f}")
```

### 3. Explore the Notebook

Open the Jupyter notebook for detailed analysis:

```powershell
jupyter notebook Bitcoin_Price_Prediction_using_Machine_Learning.ipynb
```

The notebook includes:
- Data exploration and visualization
- Feature engineering steps
- Model training with hyperparameter tuning
- Comprehensive accuracy evaluation
- Visualization of predictions vs actual values

## 📊 Model Performance

The project trains three models and evaluates them using:

- **R² Score** - Measures how well the model explains variance (higher is better)
- **MAE** (Mean Absolute Error) - Average prediction error in dollars
- **RMSE** (Root Mean Squared Error) - Penalizes larger errors more
- **MAPE** (Mean Absolute Percentage Error) - Average error as a percentage

### Expected Performance:
- **Excellent**: R² > 0.9, MAPE < 5%
- **Good**: R² 0.7-0.9, MAPE 5-10%
- **Moderate**: R² 0.5-0.7, MAPE 10-20%

## 🔧 Requirements

Install required packages:

```powershell
pip install pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

Or install from requirements file:

```powershell
pip install -r requirements.txt
```

## 📝 Model Features

The model uses these features to predict closing price:

1. **Open Price** - Opening price of the day
2. **High Price** - Highest price of the day
3. **Low Price** - Lowest price of the day
4. **Price Trend** - 20-day moving average

## 🎯 Accuracy Evaluation

To check if predictions are accurate, the script provides:

1. **Numeric Metrics** - R², MAE, RMSE, MAPE scores
2. **Visual Comparison** - Line plots of actual vs predicted prices
3. **Scatter Plots** - Shows how close predictions are to actual values
4. **Residual Analysis** - Distribution of prediction errors
5. **Sample Comparisons** - Individual prediction examples

## 💡 Usage Examples

### Example 1: Quick Prediction
```python
import pickle
import joblib
import pandas as pd

# Load model
model = pickle.load(open('btc_model.pkl', 'rb'))
scaler = joblib.load('btc_scaler.pkl')
imputer = joblib.load('btc_imputer.pkl')

# Prepare data
data = pd.DataFrame([{
    'open': 50000.0,
    'high': 51000.0,
    'low': 49500.0,
    'price_trend': 50200.0
}])

# Predict
data_scaled = imputer.transform(scaler.transform(data))
prediction = model.predict(data_scaled)[0]
print(f"Predicted: ${prediction:,.2f}")
```

### Example 2: Batch Predictions
```python
from predict_bitcoin import predict_bitcoin_price

# Multiple predictions
data_points = [
    (67500, 68200, 67000, 67800),
    (45000, 46500, 44800, 45600),
    (70000, 72000, 69500, 71000),
]

for open_p, high_p, low_p, trend_p in data_points:
    pred = predict_bitcoin_price(open_p, high_p, low_p, trend_p)
    print(f"Open: ${open_p:,} → Predicted Close: ${pred:,.2f}")
```

## 🔍 How It Works

1. **Data Preprocessing**
   - Remove outliers using percentile capping (5th-95th percentile)
   - Apply square root transformation to reduce skewness
   - Create rolling average features for trend analysis

2. **Feature Scaling**
   - StandardScaler normalizes all features to mean=0, std=1
   - SimpleImputer handles any missing values with mean strategy

3. **Model Training**
   - **Lasso Regression**: Linear model with L1 regularization
   - **Random Forest**: Ensemble of decision trees
   - **XGBoost**: Gradient boosted trees (usually best performance)

4. **Prediction**
   - New data goes through same preprocessing pipeline
   - Scaled features fed to trained model
   - Output is the predicted closing price

## 📈 Improvement Ideas

- Add more features (volume, technical indicators, sentiment data)
- Try LSTM/GRU neural networks for time series
- Implement walk-forward validation
- Add real-time data fetching from APIs
- Create a web dashboard for predictions

## ⚠️ Important Notes

- **Not Financial Advice**: This is for educational purposes only
- **Past Performance**: Historical accuracy doesn't guarantee future results
- **Data Quality**: Predictions are only as good as the training data
- **Preprocessing**: Always apply the same preprocessing to new data

## 📧 Support

If you encounter issues:
1. Ensure all dependencies are installed
2. Check that `BTC.csv` is in the same directory
3. Verify Python version is 3.7+
4. Review error messages for missing files

## 🏆 Project Status

✅ Data preprocessing and cleaning  
✅ Multiple model training (Lasso, RF, XGBoost)  
✅ Model evaluation and comparison  
✅ Prediction function for production use  
✅ Comprehensive Jupyter notebook  
✅ Interactive prediction script  

---

**Built with**: Python, Scikit-learn, XGBoost, Pandas, NumPy
