from backtesting import Strategy
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from src.data_ingestion.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.models.tft_model import TFTModel
from src.models.predict import Predictor
from src.config import (
    TFT_MAX_ENCODER_LENGTH,
    TFT_MAX_PREDICTION_LENGTH,
    SYMBOL,
    MODEL_PATH,
    USE_ATR_STOP_LOSS,
    ATR_MULTIPLIER,
    STOP_LOSS_PCT,
)


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
    try:
      self.tft_model.load_model(MODEL_PATH)
      print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
      print(
        f"Error loading model: {e}. Ensure model is trained and saved at {MODEL_PATH}")
      raise RuntimeError("TFT model could not be loaded for backtesting.")

    # Prepare a dummy dataset for the TFT model to set up its internal structure
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
    dummy_engineered_df['symbol'] = SYMBOL.split('/')[0]
    dummy_engineered_df['time_idx'] = dummy_engineered_df.groupby(
      'symbol').cumcount()

    if 'close' not in dummy_engineered_df.columns:
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
    if len(self.data.Close) < TFT_MAX_ENCODER_LENGTH:
      return

    current_data_raw = pd.DataFrame({
      'open': self.data.Open[-TFT_MAX_ENCODER_LENGTH:],
      'high': self.data.High[-TFT_MAX_ENCODER_LENGTH:],
      'low': self.data.Low[-TFT_MAX_ENCODER_LENGTH:],
      'close': self.data.Close[-TFT_MAX_ENCODER_LENGTH:],
      'volume': self.data.Volume[-TFT_MAX_ENCODER_LENGTH:],
    }, index=self.data.index[-TFT_MAX_ENCODER_LENGTH:])

    engineered_df = self.feature_engineer.engineer_features(
      current_data_raw.copy())

    engineered_df['symbol'] = SYMBOL.split('/')[0]
    engineered_df['time_idx'] = engineered_df.groupby('symbol').cumcount()

    if len(engineered_df) < TFT_MAX_ENCODER_LENGTH:
      return

    prediction_input = engineered_df.iloc[-TFT_MAX_ENCODER_LENGTH:].copy()
    signal, predicted_price = self.predictor.generate_signal(prediction_input)
    current_price = self.data.Close[-1]
    latest_atr = engineered_df['ATR'].iloc[-1] if 'ATR' in engineered_df.columns and not engineered_df['ATR'].empty else 0

    if self.position_open:
      if current_price < self.stop_loss_price:
        print(
          f"STOP-LOSS TRIGGERED for {SYMBOL} at price {current_price:.2f} (Stop: {self.stop_loss_price:.2f})")
        self.position.close()
        self.position_open = False
        return

      if signal.endswith("SELL"):
        print(f"Executing {signal} order for {SYMBOL} at {current_price:.2f}")
        self.position.close()
        self.position_open = False

    else:  # No open position
      if signal.endswith("BUY"):
        cash_to_use = self.broker.cash * 0.99
        quantity = cash_to_use / current_price

        if quantity > 0:
          print(
            f"Executing {signal} order for {quantity:.6f} of {SYMBOL} at {current_price:.2f}")
          self.buy(size=quantity)
          self.position_open = True
          self.entry_price = current_price
          
          if USE_ATR_STOP_LOSS and latest_atr > 0:
            self.stop_loss_price = current_price - (latest_atr * ATR_MULTIPLIER)
            print(f"Set ATR-based stop-loss at {self.stop_loss_price:.2f} (ATR: {latest_atr:.2f})")
          else:
            self.stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
            print(f"Set percentage-based stop-loss at {self.stop_loss_price:.2f}")