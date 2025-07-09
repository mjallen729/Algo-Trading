import time
import os
from datetime import datetime, timedelta
import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
import argparse

from src.config import (
  ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
  SYMBOL, INITIAL_CAPITAL, MODEL_PATH, TIME_FRAME,
  TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH, TRAINING_EPOCHS,
  DATA_DIR, TFT_EARLY_STOP_MONITOR, TFT_EARLY_STOP_MIN_DELTA, 
  TFT_EARLY_STOP_PATIENCE, TFT_EARLY_STOP_MODE, TFT_GRADIENT_CLIP_VAL,
  TFT_ACCELERATOR, TFT_DEVICES
)
from src.data_ingestion.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.models.tft_model import TFTModel
from src.models.predict import Predictor
from src.risk_management.portfolio import PortfolioManager
from src.trading.trader import Trader
from src.backtesting.run_backtest import run_backtest  # Import the backtest runner


def train_model(data_loader: DataLoader, feature_engineer: FeatureEngineer, tft_model: TFTModel):
  """
  Trains the TFT model on historical data.
  """
  print("--- Initial Model Training ---")

  # 1. Load historical data - Use Alpaca API for hourly data with caching
  print("Loading hourly historical data...")
  # Fetch 2 years of hourly data (enough for training + avoiding API limits)
  end_date = datetime.now()
  start_date = end_date - timedelta(days=730)  # 2 years
  
  # Check if we have cached hourly data
  hourly_cache_path = os.path.join(DATA_DIR, f"{SYMBOL.split('/')[0].lower()}_hourly.csv")
  
  if os.path.exists(hourly_cache_path):
    print(f"Loading cached hourly data from {hourly_cache_path}")
    try:
      historical_df = pd.read_csv(hourly_cache_path, index_col=0, parse_dates=True)
      # Check if cached data is recent enough (within last 24 hours)
      latest_cached = historical_df.index.max()
      if (datetime.now() - latest_cached).days < 1:
        print(f"Using cached data (latest: {latest_cached})")
      else:
        print("Cached data is outdated, fetching fresh data...")
        raise ValueError("Outdated cache")
    except:
      print("Cache invalid, fetching fresh data...")
      historical_df = None
  else:
    print("No cached data found, fetching from Alpaca API...")
    historical_df = None
  
  # Fetch fresh data if needed
  if historical_df is None:
    print(f"Fetching hourly data from {start_date.date()} to {end_date.date()}")
    historical_df = data_loader.get_data(SYMBOL, start_date, end_date, use_local=False)
    
    if not historical_df.empty:
      # Save to cache
      print(f"Saving hourly data to cache: {hourly_cache_path}")
      historical_df.to_csv(hourly_cache_path)
    else:
      print("Could not load historical data from Alpaca. Falling back to local daily data...")
      historical_df = data_loader.get_data(SYMBOL, None, None, use_local=True)
  
  if historical_df.empty:
    print("No data available. Exiting.")
    return False
  
  print(f"Loaded {len(historical_df)} rows of historical data")
  print(f"Date range: {historical_df.index.min()} to {historical_df.index.max()}")
  print(f"Data spans {(historical_df.index.max() - historical_df.index.min()).days} days")

  # 2. Engineer features
  print("Engineering features for training data...")
  engineered_df = feature_engineer.engineer_features(historical_df.copy())
  engineered_df['symbol'] = SYMBOL.split('/')[0]

  # 3. Prepare data and train model
  print("Preparing data for TFT model...")
  # With 5 years of daily data, use last 6 months for validation
  total_samples = len(engineered_df)
  validation_days = 180  # 6 months
  validation_cutoff = total_samples - validation_days
  
  # Ensure we have a reasonable training set (at least 80% of data)
  min_training_samples = int(0.8 * total_samples)
  if validation_cutoff < min_training_samples:
    validation_cutoff = min_training_samples
    print(f"Adjusted validation cutoff to ensure sufficient training data")
  
  tft_model.training_cutoff = validation_cutoff
  
  train_size = len(engineered_df[engineered_df['time_idx'] <= validation_cutoff])
  val_size = len(engineered_df[engineered_df['time_idx'] > validation_cutoff])
  print(f"Training set: {train_size} samples")
  print(f"Validation set: {val_size} samples")
  print(f"Train/Val split: {train_size/(train_size+val_size)*100:.1f}% / {val_size/(train_size+val_size)*100:.1f}%")
  tft_model.prepare_data(
    engineered_df, target_col='close', group_id_col='symbol')

  print("Building TFT model...")
  tft_model.build_model()

  print("Training TFT model...")
  print(f"Training configuration: {TRAINING_EPOCHS} epochs, {TFT_ACCELERATOR} accelerator")
  early_stop_callback = EarlyStopping(
    monitor=TFT_EARLY_STOP_MONITOR, 
    min_delta=TFT_EARLY_STOP_MIN_DELTA, 
    patience=TFT_EARLY_STOP_PATIENCE, 
    verbose=False, 
    mode=TFT_EARLY_STOP_MODE
  )
  trainer = Trainer(
    max_epochs=TRAINING_EPOCHS,
    accelerator=TFT_ACCELERATOR,
    devices=TFT_DEVICES,
    gradient_clip_val=TFT_GRADIENT_CLIP_VAL,
    callbacks=[early_stop_callback],
    enable_model_summary=False,
    logger=False,
  )
  tft_model.train_model(trainer)

  # 4. Save the trained model
  tft_model.save_model(MODEL_PATH)
  print(f"Model trained and saved to {MODEL_PATH}")
  return True


def run_trading_bot():
  """
  Main function to run the algorithmic trading bot.
  """
  if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    print("Error: Alpaca API keys not found in .env. Please set them up.")
    return

  # --- Initialization ---
  data_loader = DataLoader(ALPACA_API_KEY, ALPACA_SECRET_KEY)
  feature_engineer = FeatureEngineer()
  portfolio_manager = PortfolioManager(initial_capital=INITIAL_CAPITAL)
  trader = Trader(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

  tft_model = TFTModel(
    max_encoder_length=TFT_MAX_ENCODER_LENGTH,
    max_prediction_length=TFT_MAX_PREDICTION_LENGTH
  )

  # --- Model Loading/Training ---
  # Try to load a pre-trained model, otherwise train a new one.
  try:
    tft_model.load_model(MODEL_PATH)
    # We still need to prepare the dataset structure for prediction
    print("Preparing dataset structure from a small data sample...")
    dummy_df = data_loader.get_data(
      SYMBOL, datetime.now() - timedelta(days=10), datetime.now())
    dummy_engineered_df = feature_engineer.engineer_features(dummy_df)
    dummy_engineered_df['symbol'] = SYMBOL.split('/')[0]
    tft_model.prepare_data(dummy_engineered_df)
    print("Dataset structure prepared.")
  except (FileNotFoundError, Exception) as e:
    print(f"Could not load model ({e}), starting training process...")
    if not train_model(data_loader, feature_engineer, tft_model):
      return  # Exit if training fails

  predictor = Predictor(tft_model)
  print("--- Starting Trading Loop ---")

  # --- Trading Loop ---
  while True:
    print(f"{'=' * 20} New Trading Cycle at {datetime.now()} {'=' * 20}")

    try:
      # 1. Get latest market data
      print("Fetching latest market data...")
      fetch_start_date = datetime.now() - timedelta(hours=TFT_MAX_ENCODER_LENGTH +
                                                    50)  # Buffer for NaNs
      current_data_df = data_loader.get_data(
        SYMBOL, fetch_start_date, datetime.now())

      if current_data_df.empty or len(current_data_df) < TFT_MAX_ENCODER_LENGTH:
        print("Not enough data for prediction. Waiting...")
        time.sleep(60 * 5)
        continue

      current_price = current_data_df['close'].iloc[-1]
      print(f"Current {SYMBOL} price: {current_price:.2f}")

      # 2. Check for stop-loss triggers
      quantity_to_sell = portfolio_manager.check_stop_loss(
        current_price, SYMBOL)
      if quantity_to_sell > 0:
        print(
          f"Executing stop-loss sell order for {quantity_to_sell:.6f} {SYMBOL}")
        order = trader.place_market_order(
          SYMBOL.replace('/', ''), quantity_to_sell, "SELL")
        if order:
          portfolio_manager.record_trade(
            SYMBOL, "SELL", quantity_to_sell, current_price)
        time.sleep(60 * 60)  # Wait for the next hour after a stop-loss
        continue

      # 3. Engineer features for prediction
      engineered_df = feature_engineer.engineer_features(
        current_data_df.copy())
      engineered_df['symbol'] = SYMBOL.split('/')[0]

      if len(engineered_df) < TFT_MAX_ENCODER_LENGTH:
        print("Not enough data for prediction after feature engineering. Waiting...")
        time.sleep(60 * 5)
        continue

      # 4. Generate prediction and signal
      prediction_input = engineered_df.iloc[-TFT_MAX_ENCODER_LENGTH:].copy()
      signal, predicted_price = predictor.generate_signal(prediction_input)
      print(f"Prediction Horizon: {TFT_MAX_PREDICTION_LENGTH} hours")
      print(f"Predicted Price: {predicted_price:.2f}, Signal: {signal}")

      # 5. Execute trade based on signal
      if signal.endswith("BUY"):
        if portfolio_manager.check_risk(signal, current_price):
          quantity = portfolio_manager.calculate_quantity(current_price)
          print(f"Executing {signal} order for {quantity:.6f} {SYMBOL}")
          order = trader.place_market_order(
            SYMBOL.replace('/', ''), quantity, "BUY")
          if order:
            portfolio_manager.record_trade(
              SYMBOL, signal, quantity, current_price)

      elif signal.endswith("SELL"):
        open_positions = portfolio_manager.get_open_positions()
        if SYMBOL in open_positions and portfolio_manager.check_risk(signal, current_price):
          quantity_to_sell = open_positions[SYMBOL]["quantity"]
          print(
            f"Executing {signal} order for {quantity_to_sell:.6f} {SYMBOL}")
          order = trader.place_market_order(
            SYMBOL.replace('/', ''), quantity_to_sell, "SELL")
          if order:
            portfolio_manager.record_trade(
              SYMBOL, signal, quantity_to_sell, current_price)

      # 6. Log portfolio status
      print(f"Current Capital: {portfolio_manager.get_current_capital():.2f}")
      print(f"Open Positions: {portfolio_manager.get_open_positions()}")

    except Exception as e:
      print(f"An error occurred in the trading loop: {e}")

    # Wait for the next trading interval
    print(f"--- Cycle complete. Waiting for 1 hour. ---")
    time.sleep(60 * 60)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="Algorithmic Trading Bot")
  parser.add_argument('--mode', type=str, default='trade', choices=['train', 'backtest', 'trade'],
                      help='Operation mode: train, backtest, or trade (live bot).')
  args = parser.parse_args()

  if args.mode == 'train':
    data_loader = DataLoader(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    feature_engineer = FeatureEngineer()
    tft_model = TFTModel(
      max_encoder_length=TFT_MAX_ENCODER_LENGTH,
      max_prediction_length=TFT_MAX_PREDICTION_LENGTH
    )
    train_model(data_loader, feature_engineer, tft_model)
  elif args.mode == 'backtest':
    run_backtest()
  elif args.mode == 'trade':
    run_trading_bot()
