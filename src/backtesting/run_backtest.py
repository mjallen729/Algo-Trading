from backtesting import Backtest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering.features import FeatureEngineer
from src.models.tft_model import TFTModel
from src.models.predict import Predictor

from src.config import (
  MODEL_PATH, TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH, INITIAL_CAPITAL
)
from src.backtesting.tft_strategy import TFTStrategy


def run_backtest():
  print("--- Starting Backtest ---")
  print("NOTE: Using ETH data for compatibility with trained model")
  print("SOL data testing will require model retraining with SOL features")

  # Load ETH hourly data (what the model was trained on)
  eth_data_path = "/Users/matt/Documents/Algo-Trading/data/eth_hourly.csv"
  print(f"Loading ETH data from {eth_data_path}...")
  
  try:
    eth_df = pd.read_csv(eth_data_path)
    print(f"Loaded {len(eth_df)} ETH data points")
  except Exception as e:
    print(f"Error loading ETH data: {e}")
    return

  # Convert timestamp to datetime and set as index
  eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'])
  eth_df = eth_df.set_index('timestamp')
  
  # Remove timezone info if present to avoid issues
  if eth_df.index.tz is not None:
    eth_df.index = eth_df.index.tz_localize(None)

  print(f"ETH data date range: {eth_df.index.min()} to {eth_df.index.max()}")

  # Initialize feature engineer
  feature_engineer = FeatureEngineer()

  # Apply feature engineering to the entire ETH dataset
  print("Applying feature engineering to ETH data...")
  engineered_df = feature_engineer.engineer_features(eth_df.copy())
  
  # Keep ETH symbol as the model expects
  engineered_df['symbol'] = 'ETH'
  
  # Add time_idx required by TFT model
  engineered_df['time_idx'] = engineered_df.groupby('symbol').cumcount()

  # Drop rows with NaN values at the beginning (due to lagged features, moving averages, etc.)
  print(f"Before dropping NaN: {len(engineered_df)} rows")
  initial_length = len(engineered_df)
  engineered_df = engineered_df.dropna()
  print(f"After dropping NaN: {len(engineered_df)} rows")
  print(f"Dropped {initial_length - len(engineered_df)} rows with NaN values")

  if len(engineered_df) < TFT_MAX_ENCODER_LENGTH + 100:
    print(f"Not enough data after feature engineering. Need at least {TFT_MAX_ENCODER_LENGTH + 100} rows.")
    return

  # Log feature count to verify we have expected number (should be around 72)
  feature_cols = [col for col in engineered_df.columns if col not in ['symbol', 'time_idx']]
  print(f"Generated {len(feature_cols)} features for model input")
  print(f"Sample feature columns: {feature_cols[:10]}")

  # Use out-of-sample data for backtesting (last 20% of data)
  # This simulates testing on data the model hasn't seen
  total_samples = len(engineered_df)
  backtest_start_idx = int(total_samples * 0.8)  # Use last 20% for backtesting
  backtest_df = engineered_df.iloc[backtest_start_idx:].copy()
  
  print(f"Using out-of-sample data for backtesting:")
  print(f"Total samples: {total_samples}")
  print(f"Training samples (0-80%): {backtest_start_idx}")
  print(f"Backtest samples (80-100%): {len(backtest_df)}")

  # Prepare data for backtesting.py format (needs OHLCV columns capitalized)
  backtest_df = backtest_df.rename(columns={
    'open': 'Open', 
    'high': 'High', 
    'low': 'Low', 
    'close': 'Close', 
    'volume': 'Volume'
  })

  print(f"Final backtest data shape: {backtest_df.shape}")
  print(f"Backtest date range: {backtest_df.index.min()} to {backtest_df.index.max()}")

  # Store the engineered data globally so TFTStrategy can access it
  # This is a workaround since backtesting.py doesn't allow passing custom data to Strategy
  import __main__
  __main__.ENGINEERED_DATA = engineered_df
  __main__.FEATURE_ENGINEER = feature_engineer

  # Initialize and run the Backtest
  bt = Backtest(
    backtest_df[['Open', 'High', 'Low', 'Close', 'Volume']],  # Only OHLCV for backtesting.py
    TFTStrategy,
    cash=INITIAL_CAPITAL,
    commission=0.002,  # 0.2% commission per trade
    exclusive_orders=True
  )

  print("Running backtest...")
  try:
    stats = bt.run()
    print("\n--- Backtest Results ---")
    print(stats)

    # Plot the results
    print("\nGenerating backtest plot...")
    bt.plot(filename="backtest_plot.html", open_browser=False)
    print("Backtest plot saved to backtest_plot.html")
    
    return stats
    
  except Exception as e:
    print(f"Error during backtesting: {e}")
    import traceback
    traceback.print_exc()


if __name__ == '__main__':
  run_backtest()
