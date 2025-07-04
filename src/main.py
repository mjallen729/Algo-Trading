import time
from datetime import datetime, timedelta
import pandas as pd
import torch

from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH, TIME_FRAME
from src.data_ingestion.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.models.tft_model import TFTModel
from src.models.predict import Predictor
from src.risk_management.portfolio import PortfolioManager
from src.trading.trader import Trader

# --- Configuration --- #
SYMBOL = "ETH/USD" # The cryptocurrency pair to trade
INITIAL_CAPITAL = 10000.0 # Starting capital for paper trading
MODEL_PATH = "tft_model.pth" # Path to save/load the trained model

# --- Main Trading Loop --- #
def run_trading_bot():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("Error: Alpaca API keys not found in .env. Please set them up.")
        return

    data_loader = DataLoader(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    feature_engineer = FeatureEngineer()
    portfolio_manager = PortfolioManager(initial_capital=INITIAL_CAPITAL)
    trader = Trader(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

    # Initialize TFT Model (will be trained or loaded)
    tft_model = TFTModel(TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH)
    predictor = None # Will be initialized after model is trained/loaded

    print("Starting algorithmic trading bot...")

    # --- Model Training (Initial or Retrain) ---
    # For MVP, we'll train on a fixed historical period. In production, this would be continuous.
    print("Loading historical data for initial model training...")
    end_train_date = datetime.now() - timedelta(days=30) # Train on data up to 30 days ago
    start_train_date = end_train_date - timedelta(days=365) # One year of historical data

    historical_df = data_loader.get_data(SYMBOL, start_train_date, end_train_date)
    if historical_df.empty:
        print("Could not load historical data for training. Exiting.")
        return
    
    print("Engineering features for training data...")
    engineered_train_df = feature_engineer.engineer_features(historical_df.copy())
    engineered_train_df['symbol'] = SYMBOL.split('/')[0] # Add symbol column for TFT
    engineered_train_df['time_idx'] = engineered_train_df.groupby('symbol').cumcount()

    print("Preparing and training TFT model...")
    # Define training cutoff (e.g., 80% for training, 20% for validation)
    training_cutoff = engineered_train_df['time_idx'].max() - int(0.2 * len(engineered_train_df))
    tft_model.training_cutoff = training_cutoff # Set cutoff for train/val split

    tft_model.prepare_data(engineered_train_df, target_col='close', group_id_col='symbol')
    tft_model.build_model()

    # Using a dummy trainer for now. In a real scenario, use pytorch_lightning.Trainer
    class DummyTrainer:
        def fit(self, model, train_dataloader, val_dataloader):
            print("Training model (dummy fit)...")
            # In a real scenario, this would run the training loop
            # For actual training, you'd need to set up a PyTorch Lightning Trainer
            # from pytorch_lightning import Trainer
            # trainer = Trainer(max_epochs=10, accelerator="cpu") # Configure as needed
            # trainer.fit(model, train_dataloader, val_dataloader)
            pass

    dummy_trainer = DummyTrainer()
    tft_model.train_model(dummy_trainer)
    tft_model.save_model(MODEL_PATH)
    print(f"Model trained and saved to {MODEL_PATH}")

    # Initialize predictor after model is trained/loaded
    predictor = Predictor(MODEL_PATH, tft_model.tft_dataset)

    # --- Live Trading Loop ---
    while True:
        print(f"\n--- Running trading cycle at {datetime.now()} ---")
        
        # 1. Data Ingestion (get latest data)
        # Fetch enough data for feature engineering and model input (max_encoder_length)
        start_data_fetch = datetime.now() - timedelta(hours=TFT_MAX_ENCODER_LENGTH + 5) # Fetch a bit more than needed
        current_data_df = data_loader.get_data(SYMBOL, start_data_fetch, datetime.now())

        if current_data_df.empty or len(current_data_df) < TFT_MAX_ENCODER_LENGTH:
            print("Not enough data for prediction. Waiting for more data...")
            time.sleep(60 * 5) # Wait 5 minutes
            continue

        # 2. Feature Engineering
        engineered_current_df = feature_engineer.engineer_features(current_data_df.copy())
        engineered_current_df['symbol'] = SYMBOL.split('/')[0]
        engineered_current_df['time_idx'] = engineered_current_df.groupby('symbol').cumcount()

        # Ensure we have enough data points after feature engineering for prediction
        if len(engineered_current_df) < TFT_MAX_ENCODER_LENGTH:
            print("Not enough engineered data for prediction after dropping NaNs. Waiting...")
            time.sleep(60 * 5)
            continue

        # Prepare data for prediction (last `max_encoder_length` rows)
        prediction_input_data = engineered_current_df.iloc[-TFT_MAX_ENCODER_LENGTH:].copy()

        # 3. Prediction
        predicted_direction = predictor.predict_direction(prediction_input_data)
        print(f"Predicted direction for {SYMBOL}: {predicted_direction}")

        # Get current price for risk management and trading
        current_price = current_data_df['close'].iloc[-1]
        print(f"Current {SYMBOL} price: {current_price}")

        # 4. Risk Management & Trade Execution
        if predicted_direction == "UP":
            if portfolio_manager.check_risk("BUY", current_price, SYMBOL):
                # Calculate quantity based on position sizing
                # For simplicity, let's use a fixed quantity for now or a calculated one
                # This needs to be refined based on portfolio_manager.get_position_size()
                # Example: quantity = portfolio_manager.get_position_size(current_price) / current_price
                quantity_to_buy = 0.001 # Example fixed quantity
                print(f"Attempting to BUY {quantity_to_buy} of {SYMBOL}")
                order = trader.place_market_order(SYMBOL.replace('/', ''), quantity_to_buy, "BUY")
                if order and order.status == 'accepted':
                    portfolio_manager.record_trade(SYMBOL, "BUY", quantity_to_buy, current_price)

        elif predicted_direction == "DOWN":
            if portfolio_manager.check_risk("SELL", current_price, SYMBOL):
                # For simplicity, sell a fixed quantity or all current position
                quantity_to_sell = portfolio_manager.positions.get(SYMBOL, 0) # Sell all if exists
                if quantity_to_sell > 0:
                    print(f"Attempting to SELL {quantity_to_sell} of {SYMBOL}")
                    order = trader.place_market_order(SYMBOL.replace('/', ''), quantity_to_sell, "SELL")
                    if order and order.status == 'accepted':
                        portfolio_manager.record_trade(SYMBOL, "SELL", quantity_to_sell, current_price)

        print(f"Current Capital: {portfolio_manager.get_current_capital():.2f}")
        print(f"Current Positions: {portfolio_manager.positions}")

        # Wait for the next trading interval (e.g., every hour)
        time.sleep(60 * 60) # Sleep for 1 hour

if __name__ == '__main__':
    run_trading_bot()
