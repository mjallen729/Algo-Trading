from backtesting import Strategy
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

from src.models.tft_model import TFTModel
from src.models.predict import Predictor
from src.config import (
    TFT_MAX_ENCODER_LENGTH,
    TFT_MAX_PREDICTION_LENGTH,
    MODEL_PATH,
    USE_ATR_STOP_LOSS,
    ATR_MULTIPLIER,
    STOP_LOSS_PCT,
)


class TFTStrategy(Strategy):

  def init(self):
    print("Initializing TFT Strategy...")
    
    # Access the global engineered data passed from run_backtest
    import __main__
    if not hasattr(__main__, 'ENGINEERED_DATA'):
      raise RuntimeError("ENGINEERED_DATA not found. Ensure run_backtest.py sets this global variable.")
    
    self.engineered_data = __main__.ENGINEERED_DATA
    self.feature_engineer = __main__.FEATURE_ENGINEER
    
    print(f"Loaded engineered data with {len(self.engineered_data)} rows and {len(self.engineered_data.columns)} columns")

    # Initialize TFTModel
    self.tft_model = TFTModel(
      max_encoder_length=TFT_MAX_ENCODER_LENGTH,
      max_prediction_length=TFT_MAX_PREDICTION_LENGTH
    )

    # Prepare the TFT model with the engineered data FIRST
    try:
      # Use the FULL engineered dataset to prepare the model structure
      # This ensures all categorical features have correct dimensions
      prep_data = self.engineered_data.copy()
      
      # Ensure required columns exist
      if 'close' not in prep_data.columns:
        prep_data['close'] = prep_data['open']  # Fallback
      
      print(f"Preparing model with full dataset ({len(prep_data)} samples)")
      self.tft_model.prepare_data(prep_data, target_col='close', group_id_col='symbol')
      print("TFT model data preparation successful")
      
    except Exception as e:
      print(f"Error preparing TFT model: {e}")
      import traceback
      traceback.print_exc()
      raise

    # Now load the pre-trained model
    try:
      self.tft_model.load_model(MODEL_PATH)
      print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
      print(f"Error loading model: {e}")
      raise RuntimeError("TFT model could not be loaded for backtesting.")

    self.predictor = Predictor(self.tft_model)

    # Track positions
    self.position_open = False
    self.entry_price = 0.0
    self.stop_loss_price = 0.0
    
    print("TFT Strategy initialization complete")

  def next(self):
    # Need enough data for encoder window
    if len(self.data.Close) < TFT_MAX_ENCODER_LENGTH:
      return

    try:
      # Get current timestamp from backtest data
      current_timestamp = self.data.index[-1]
      
      # Find corresponding row in engineered data
      # Since we're using SOL data with ETH symbol, match by timestamp
      matching_rows = self.engineered_data.loc[
        self.engineered_data.index <= current_timestamp
      ]
      
      if len(matching_rows) < TFT_MAX_ENCODER_LENGTH:
        return
        
      # Get the last TFT_MAX_ENCODER_LENGTH rows for prediction
      prediction_input = matching_rows.tail(TFT_MAX_ENCODER_LENGTH).copy()
      
      # Ensure we have all required columns
      if len(prediction_input) < TFT_MAX_ENCODER_LENGTH:
        return
        
      # Generate trading signal
      signal, predicted_price = self.predictor.generate_signal(prediction_input)
      current_price = self.data.Close[-1]
      
      # Get ATR for stop loss (if available)
      latest_atr = 0
      if 'ATR' in prediction_input.columns and not prediction_input['ATR'].empty:
        latest_atr = prediction_input['ATR'].iloc[-1]
        if pd.isna(latest_atr):
          latest_atr = 0

      # Handle existing position
      if self.position_open:
        # Check stop loss
        if current_price < self.stop_loss_price:
          print(f"STOP-LOSS TRIGGERED at price {current_price:.2f} (Stop: {self.stop_loss_price:.2f})")
          self.position.close()
          self.position_open = False
          return

        # Check for sell signal
        if signal.endswith("SELL"):
          print(f"Executing {signal} order at {current_price:.2f}")
          self.position.close()
          self.position_open = False

      else:  # No open position
        # Check for buy signal
        if signal.endswith("BUY"):
          cash_to_use = self.broker.cash * 0.99  # Leave small buffer
          quantity = cash_to_use / current_price

          if quantity > 0:
            print(f"Executing {signal} order for {quantity:.6f} at {current_price:.2f}")
            self.buy(size=quantity)
            self.position_open = True
            self.entry_price = current_price
            
            # Set stop loss
            if USE_ATR_STOP_LOSS and latest_atr > 0:
              self.stop_loss_price = current_price - (latest_atr * ATR_MULTIPLIER)
              print(f"Set ATR-based stop-loss at {self.stop_loss_price:.2f} (ATR: {latest_atr:.2f})")
            else:
              self.stop_loss_price = current_price * (1 - STOP_LOSS_PCT)
              print(f"Set percentage-based stop-loss at {self.stop_loss_price:.2f}")
              
    except Exception as e:
      print(f"Error in TFT Strategy next(): {e}")
      # Don't crash the backtest, just skip this iteration
      pass
