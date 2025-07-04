from backtesting import Strategy
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from src.data_ingestion.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.models.tft_model import TFTModel
from src.models.predict import Predictor
from src.config import TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH, SYMBOL, MODEL_PATH


class TFTStrategy(Strategy):
  # Define parameters that can be optimized
  # These will be accessible as self.param_name
  # For now, we'll keep them fixed as per the original logic
  # but they can be exposed for optimization later.

  def init(self):
    # Initialize your components here
    # Note: DataLoader and Trader are not directly used in backtesting.Strategy's init/next
    # as backtesting.py provides its own data and order execution.
    # However, we need them for model training/loading outside the strategy.

    # Initialize FeatureEngineer
    self.feature_engineer = FeatureEngineer()

    # Initialize TFTModel and load it
    self.tft_model = TFTModel(
      max_encoder_length=TFT_MAX_ENCODER_LENGTH,
      max_prediction_length=TFT_MAX_PREDICTION_LENGTH
    )

    # Load the pre-trained model
    # In a real backtest, you'd typically train the model once before running the backtest
    # or implement a re-training mechanism within the strategy if needed.
    try:
      self.tft_model.load_model(MODEL_PATH)
      print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
      print(
        f"Error loading model: {e}. Ensure model is trained and saved at {MODEL_PATH}")
      # In a backtest, if model loading fails, the strategy cannot proceed.
      # For simplicity, we'll raise an error, but in a real scenario,
      # you might want to handle this more gracefully (e.g., train a dummy model).
      raise RuntimeError("TFT model could not be loaded for backtesting.")

    # Prepare a dummy dataset for the TFT model to set up its internal structure
    # This is crucial because TimeSeriesDataSet needs to be initialized with data
    # to understand the feature types and normalizers.
    # We'll use a small, representative sample.
    # In a full pipeline, this would ideally be done during model training.
    dummy_data = {
      'open': np.random.rand(TFT_MAX_ENCODER_LENGTH + 50) * 100,
      'high': np.random.rand(TFT_MAX_ENCODER_LENGTH + 50) * 100 + 1,
      'low': np.random.rand(TFT_MAX_ENCODER_LENGTH + 50) * 100 - 1,
      'close': np.random.rand(TFT_MAX_ENCODER_LENGTH + 50) * 100,
      'volume': np.random.rand(TFT_MAX_ENCODER_LENGTH + 50) * 1000
    }
    dummy_index = pd.date_range(
      start='2020-01-01', periods=TFT_MAX_ENCODER_LENGTH + 50, freq='H')
    dummy_df = pd.DataFrame(dummy_data, index=dummy_index)
    dummy_engineered_df = self.feature_engineer.engineer_features(
      dummy_df.copy())
    dummy_engineered_df['symbol'] = SYMBOL.split('/')[0]  # Add symbol column
    dummy_engineered_df['time_idx'] = dummy_engineered_df.groupby(
      'symbol').cumcount()  # Add time_idx

    # Ensure target column is present for prepare_data
    if 'close' not in dummy_engineered_df.columns:
      # Placeholder if 'close' is missing after engineering
      dummy_engineered_df['close'] = dummy_engineered_df['open']

    self.tft_model.prepare_data(
      dummy_engineered_df, target_col='close', group_id_col='symbol')
    print("TFT model dataset structure prepared for backtesting.")

    self.predictor = Predictor(self.tft_model)

    # Keep track of open positions
    self.position_open = False
    self.entry_price = 0.0
    self.stop_loss_price = 0.0

  def next(self):
    # Ensure we have enough data for the encoder length
    if len(self.data.Close) < TFT_MAX_ENCODER_LENGTH:
      return  # Not enough data yet

    # Construct DataFrame for feature engineering and prediction
    # backtesting.py provides data as self.data.Open, self.data.High, etc.
    # We need to convert it to a pandas DataFrame for our feature engineer and TFT model.

    # Get the required lookback window data
    # Use .iloc to get the last TFT_MAX_ENCODER_LENGTH rows
    current_data_raw = pd.DataFrame({
      'open': self.data.Open[-TFT_MAX_ENCODER_LENGTH:],
      'high': self.data.High[-TFT_MAX_ENCODER_LENGTH:],
      'low': self.data.Low[-TFT_MAX_ENCODER_LENGTH:],
      'close': self.data.Close[-TFT_MAX_ENCODER_LENGTH:],
      'volume': self.data.Volume[-TFT_MAX_ENCODER_LENGTH:],
    }, index=self.data.index[-TFT_MAX_ENCODER_LENGTH:])

    # Engineer features
    engineered_df = self.feature_engineer.engineer_features(
      current_data_raw.copy())

    # Add 'symbol' and 'time_idx' columns as expected by TFTModel
    engineered_df['symbol'] = SYMBOL.split('/')[0]
    engineered_df['time_idx'] = engineered_df.groupby('symbol').cumcount()

    # Ensure we still have enough data after feature engineering (dropna might remove some)
    if len(engineered_df) < TFT_MAX_ENCODER_LENGTH:
      # This can happen if feature engineering creates NaNs at the beginning
      # and the remaining data is less than encoder length.
      # In a real scenario, you might want to pad or handle this more robustly.
      return

    # Get the latest data point for prediction input
    prediction_input = engineered_df.iloc[-TFT_MAX_ENCODER_LENGTH:].copy()

    # Generate signal
    signal, predicted_price = self.predictor.generate_signal(prediction_input)
    current_price = self.data.Close[-1]

    # Trading logic
    if self.position_open:
      # Check stop-loss
      if current_price < self.stop_loss_price:
        print(
          f"STOP-LOSS TRIGGERED for {SYMBOL} at price {current_price:.2f} (Stop: {self.stop_loss_price:.2f})")
        self.position.close()
        self.position_open = False
        self.entry_price = 0.0
        self.stop_loss_price = 0.0
        return  # Exit after closing position

      # If holding, check for sell signal
      if signal.endswith("SELL"):
        print(f"Executing {signal} order for {SYMBOL} at {current_price:.2f}")
        self.position.close()
        self.position_open = False
        self.entry_price = 0.0
        self.stop_loss_price = 0.0

    else:  # No open position
      if signal.endswith("BUY"):
        # Calculate quantity based on a fixed percentage of equity for simplicity in backtesting
        # In a real scenario, you'd use the PortfolioManager's logic.
        # For backtesting, we'll just buy a fixed amount or a percentage of available cash.
        # Let's buy 99% of available cash to ensure we can open a position.
        cash_to_use = self.broker.cash * 0.99
        quantity = cash_to_use / current_price

        if quantity > 0:
          print(
            f"Executing {signal} order for {quantity:.6f} of {SYMBOL} at {current_price:.2f}")
          self.buy(size=quantity)
          self.position_open = True
          self.entry_price = current_price
          # Set a simple stop-loss for backtesting
          # This should ideally come from PortfolioManager's logic
          STOP_LOSS_PCT = 0.02  # Example, should be from config
          self.stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
          print(f"Set stop-loss at {self.stop_loss_price:.2f}")
