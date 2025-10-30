import sqlite3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import pickle
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Read CSV with daily data
dataset = pd.read_csv("BTC.csv", parse_dates=['date'])

print("Processing Bitcoin dataset...")
print(f"Dataset shape: {dataset.shape}")

# Drop ticker column if exists
if 'ticker' in dataset.columns:
    dataset.drop("ticker", axis=1, inplace=True)

# Set date as index
dataset.set_index("date", inplace=True)

# Add price trend feature
dataset["price_trend"] = dataset["close"].rolling(window=20).mean()

# Reset index for processing
dataset.reset_index("date", inplace=True)

# Check for missing values
print(f"\nMissing values before cleaning:\n{dataset.isna().sum()}")

# Apply skewness transformation
print("\nApplying skewness transformation...")
skewed_columns = dataset.drop("date", axis=1).skew(axis=0, skipna=True)
for col in skewed_columns[skewed_columns > 1].index:
    dataset[col] = dataset[col].apply(lambda x: np.sqrt(x) if x > 0 else 0)

# Outlier removal function
def outlier_removal(column):
    upper_limit = column.quantile(.95)
    lower_limit = column.quantile(.05)
    column.loc[(column > upper_limit)] = upper_limit
    column.loc[(column < lower_limit)] = lower_limit
    return column

# Apply outlier removal
print("Removing outliers...")
dataset[['open', 'high', 'low', 'close']] = dataset[['open', 'high', 'low', 'close']].apply(outlier_removal)

# Prepare data for machine learning
print("\n" + "="*60)
print("PREPARING DATA FOR MACHINE LEARNING")
print("="*60)

# Drop rows with NaN values created by rolling operations
dataset_clean = dataset.dropna()
print(f"Dataset shape after cleaning: {dataset_clean.shape}")

# Select features and target variable
X = dataset_clean.drop(['date', 'close'], axis=1)
y = dataset_clean['close']

print(f"Features: {list(X.columns)}")
print(f"Target: close price")

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining samples: {len(x_train)}")
print(f"Testing samples: {len(x_test)}")

# Feature scaling
print("\nScaling features...")
scaler = StandardScaler()
scaler.fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Handle missing values with imputation
imputer = SimpleImputer(strategy='mean')
x_train_scaled = imputer.fit_transform(x_train_scaled)
x_test_scaled = imputer.transform(x_test_scaled)

# Save preprocessing objects
joblib.dump(scaler, 'btc_scaler.pkl')
joblib.dump(imputer, 'btc_imputer.pkl')
print("Saved scaler and imputer to disk")

# Train models
print("\n" + "="*60)
print("TRAINING MACHINE LEARNING MODELS")
print("="*60)

# 1. Lasso Regression with Polynomial Features
print("\n1. Training Lasso Regression...")
poly = PolynomialFeatures(degree=2)
lasso = Lasso()
param_grid_lasso = {'lasso__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 20, 30, 40]}
pipeline_lasso = make_pipeline(poly, lasso)
lasso_grid_search = GridSearchCV(pipeline_lasso, param_grid_lasso, scoring='r2', cv=3, n_jobs=-1)
lasso_grid_search.fit(x_train_scaled, y_train)
print(f"   Best parameters: {lasso_grid_search.best_params_}")
print(f"   Best CV score: {lasso_grid_search.best_score_:.4f}")

# 2. Random Forest
print("\n2. Training Random Forest...")
param_grid_rf = {'n_estimators': [50, 80, 100], 'max_depth': [3, 5, 7]}
rf = RandomForestRegressor(random_state=42)
rf_grid_search = GridSearchCV(rf, param_grid_rf, scoring='r2', cv=2, n_jobs=-1)
rf_grid_search.fit(x_train_scaled, y_train)
print(f"   Best parameters: {rf_grid_search.best_params_}")
print(f"   Best CV score: {rf_grid_search.best_score_:.4f}")

# 3. XGBoost (Final Model)
print("\n3. Training XGBoost...")
model_xgb = XGBRegressor(random_state=42, n_estimators=100)
model_xgb.fit(x_train_scaled, y_train)
print("   XGBoost training completed")

# Evaluate models
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

models = {
    'Lasso': lasso_grid_search,
    'Random Forest': rf_grid_search,
    'XGBoost': model_xgb
}

best_model_name = None
best_r2 = -float('inf')

for name, model in models.items():
    y_pred_train = model.predict(x_train_scaled)
    y_pred_test = model.predict(x_test_scaled)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = mean_absolute_percentage_error(y_test, y_pred_test) * 100
    
    print(f"\n{name}:")
    print(f"  Train R²: {r2_train:.4f}")
    print(f"  Test R²:  {r2_test:.4f}")
    print(f"  MAE:      ${mae:.4f}")
    print(f"  RMSE:     ${rmse:.4f}")
    print(f"  MAPE:     {mape:.2f}%")
    
    if r2_test > best_r2:
        best_r2 = r2_test
        best_model_name = name

print(f"\n🏆 Best Model: {best_model_name} (R² = {best_r2:.4f})")

# Save the best model (XGBoost)
print("\n" + "="*60)
print("SAVING MODELS")
print("="*60)
pickle.dump(model_xgb, open('btc_model.pkl', 'wb'))
pickle.dump(rf_grid_search, open('btc_rf_model.pkl', 'wb'))
pickle.dump(lasso_grid_search, open('btc_lasso_model.pkl', 'wb'))
print("✓ Saved btc_model.pkl (XGBoost)")
print("✓ Saved btc_rf_model.pkl (Random Forest)")
print("✓ Saved btc_lasso_model.pkl (Lasso)")

# Create prediction function
def predict_bitcoin_price(new_data_dict):
    """
    Predict Bitcoin closing price from OHLC data
    
    Args:
        new_data_dict: Dictionary with keys 'open', 'high', 'low', 'price_trend'
        Example: {'open': 5.5, 'high': 5.8, 'low': 5.3, 'price_trend': 5.6}
    
    Returns:
        Predicted closing price
    """
    # Load model and preprocessing objects
    model = pickle.load(open('btc_model.pkl', 'rb'))
    scaler_obj = joblib.load('btc_scaler.pkl')
    imputer_obj = joblib.load('btc_imputer.pkl')
    
    # Create DataFrame from input
    new_data = pd.DataFrame([new_data_dict])
    
    # Apply same preprocessing steps
    new_data_scaled = scaler_obj.transform(new_data)
    new_data_scaled = imputer_obj.transform(new_data_scaled)
    
    # Make prediction
    prediction = model.predict(new_data_scaled)[0]
    
    return prediction

# Test prediction function
print("\n" + "="*60)
print("TESTING PREDICTION FUNCTION")
print("="*60)
sample_idx = 0
sample_data = {
    'open': x_test.iloc[sample_idx]['open'],
    'high': x_test.iloc[sample_idx]['high'],
    'low': x_test.iloc[sample_idx]['low'],
    'price_trend': x_test.iloc[sample_idx]['price_trend']
}
predicted = predict_bitcoin_price(sample_data)
actual = y_test.iloc[sample_idx]
print(f"Sample prediction:")
print(f"  Input: {sample_data}")
print(f"  Predicted: ${predicted:.4f}")
print(f"  Actual:    ${actual:.4f}")
print(f"  Error:     ${abs(predicted - actual):.4f} ({abs((predicted - actual) / actual * 100):.2f}%)")

print("\n" + "="*60)
print("✅ BITCOIN PRICE PREDICTION MODEL COMPLETE!")
print("="*60)
print(f"Total records processed: {len(dataset_clean)}")
print(f"Model accuracy (R²): {best_r2:.4f}")
print("Models saved and ready for use!")