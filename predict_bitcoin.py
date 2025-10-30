"""
Bitcoin Price Prediction Script
Uses the trained model to predict Bitcoin closing prices
"""

import pickle
import joblib
import pandas as pd
import numpy as np

def predict_bitcoin_price(open_price, high_price, low_price, price_trend=None):
    """
    Predict Bitcoin closing price from OHLC data
    
    Args:
        open_price (float): Opening price
        high_price (float): Highest price
        low_price (float): Lowest price
        price_trend (float, optional): 20-day moving average
    
    Returns:
        float: Predicted closing price
    """
    try:
        # Load model and preprocessing objects
        model = pickle.load(open('btc_model.pkl', 'rb'))
        scaler = joblib.load('btc_scaler.pkl')
        imputer = joblib.load('btc_imputer.pkl')
        
        # If price_trend not provided, estimate it
        if price_trend is None:
            price_trend = (open_price + high_price + low_price) / 3
        
        # Create input data
        new_data = pd.DataFrame([{
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'price_trend': price_trend
        }])
        
        # Apply preprocessing
        new_data_scaled = scaler.transform(new_data)
        new_data_scaled = imputer.transform(new_data_scaled)
        
        # Make prediction
        prediction = model.predict(new_data_scaled)[0]
        
        return prediction
    
    except FileNotFoundError:
        print("Error: Model files not found. Please run sideproject.py first to train the model.")
        return None
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None


def main():
    """Example usage of the prediction function"""
    print("="*60)
    print("BITCOIN PRICE PREDICTION")
    print("="*60)
    
    # Example 1: Recent Bitcoin prices (hypothetical)
    print("\nExample 1: Predicting from recent data")
    predicted = predict_bitcoin_price(
        open_price=67500.0,
        high_price=68200.0,
        low_price=67000.0,
        price_trend=67800.0
    )
    
    if predicted:
        print(f"  Input:")
        print(f"    Open:  $67,500.00")
        print(f"    High:  $68,200.00")
        print(f"    Low:   $67,000.00")
        print(f"    Trend: $67,800.00")
        print(f"  Predicted Close: ${predicted:,.2f}")
    
    # Example 2: Lower prices
    print("\nExample 2: Predicting from lower price range")
    predicted = predict_bitcoin_price(
        open_price=45000.0,
        high_price=46500.0,
        low_price=44800.0
    )
    
    if predicted:
        print(f"  Input:")
        print(f"    Open:  $45,000.00")
        print(f"    High:  $46,500.00")
        print(f"    Low:   $44,800.00")
        print(f"  Predicted Close: ${predicted:,.2f}")
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE PREDICTION")
    print("="*60)
    try:
        print("\nEnter Bitcoin price data (or press Ctrl+C to exit):")
        open_price = float(input("  Opening price: $"))
        high_price = float(input("  Highest price: $"))
        low_price = float(input("  Lowest price: $"))
        
        use_trend = input("  Do you have a 20-day moving average? (y/n): ").lower()
        if use_trend == 'y':
            price_trend = float(input("  Price trend (20-day MA): $"))
        else:
            price_trend = None
        
        predicted = predict_bitcoin_price(open_price, high_price, low_price, price_trend)
        
        if predicted:
            print(f"\n  📊 Predicted Closing Price: ${predicted:,.2f}")
            
            # Calculate implied change from open
            change = predicted - open_price
            change_pct = (change / open_price) * 100
            direction = "↑" if change > 0 else "↓"
            
            print(f"  {direction} Change from Open: ${abs(change):,.2f} ({abs(change_pct):.2f}%)")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
