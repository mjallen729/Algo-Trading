import time
from datetime import datetime, timedelta
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
import argparse

from src.config import (
  ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL,
  SYMBOL, INITIAL_CAPITAL, MODEL_PATH, TIME_FRAME,
  TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH, TRAINING_EPOCHS
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

  # 1. Load historical data
  print("Loading historical data for training...")
  # Fetch a year of data ending 1 day ago to ensure we have fresh data for validation
  end_train_date = datetime.now() - timedelta(days=1)
  start_train_date = end_train_date - timedelta(days=365)

  historical_df = data_loader.get_data(
    SYMBOL, start_train_date, end_train_date, use_local=True)
  if historical_df.empty:
    print("Could not load historical data for training. Exiting.")
    return False

  # 2. Engineer features
  print("Engineering features for training data...")
  engineered_df = feature_engineer.engineer_features(historical_df.copy())
  engineered_df['symbol'] = SYMBOL.split('/')[0]

  # 3. Prepare data and train model
  print("Preparing data for TFT model...")
  # Use last 20% of data for validation
  training_cutoff = engineered_df['time_idx'].max(
  ) - int(0.2 * len(engineered_df))
  tft_model.training_cutoff = training_cutoff
  tft_model.prepare_data(
    engineered_df, target_col='close', group_id_col='symbol')

  print("Building TFT model...")
  tft_model.build_model()

  print("Training TFT model...")
  early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min")
  trainer = Trainer(
    max_epochs=TRAINING_EPOCHS,
    accelerator="cpu",  # Use "gpu" if available
    devices=1,
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
