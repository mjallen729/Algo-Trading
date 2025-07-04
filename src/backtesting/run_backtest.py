from backtesting import Backtest
import pandas as pd
from datetime import datetime, timedelta

from src.data_ingestion.data_loader import DataLoader
from src.feature_engineering.features import FeatureEngineer
from src.models.tft_model import TFTModel
from src.models.predict import Predictor
from src.risk_management.portfolio import PortfolioManager
from src.trading.trader import Trader

from src.config import (
    ALPACA_API_KEY, ALPACA_SECRET_KEY, SYMBOL, MODEL_PATH,
    TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH, INITIAL_CAPITAL
)
from src.backtesting.tft_strategy import TFTStrategy

def run_backtest():
    print("--- Starting Backtest ---")

    # Initialize components needed for data loading and feature engineering
    data_loader = DataLoader(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    feature_engineer = FeatureEngineer()

    # Load historical data for backtesting
    # It's crucial to use data that the model was NOT trained on for a realistic backtest.
    # For this example, let's use data from the last 6 months.
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180) # Last 6 months
    
    print(f"Loading historical data for backtest from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
    historical_df = data_loader.get_data(SYMBOL, start_date, end_date, use_local=True)

    if historical_df.empty:
        print("Could not load historical data for backtesting. Exiting.")
        return

    # Rename columns to match backtesting.py's expected format
    historical_df = historical_df.rename(columns={
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'
    }))
    
    # Ensure index is datetime
    historical_df.index = pd.to_datetime(historical_df.index)
    
    # Drop any NaNs that might result from initial data loading or incomplete data
    historical_df.dropna(inplace=True)

    print(f"Loaded {len(historical_df)} data points for backtesting.")

    # Initialize and run the Backtest
    bt = Backtest(
        historical_df,
        TFTStrategy,
        cash=INITIAL_CAPITAL,
        commission=0.002, # 0.2% commission per trade
        exclusive_orders=True
    )

    print("Running backtest...")
    stats = bt.run()

    print("\n--- Backtest Results ---")
    print(stats)

    # Plot the results
    print("\nGenerating backtest plot...")
    bt.plot(filename="backtest_plot.html", open_browser=False)
    print("Backtest plot saved to backtest_plot.html")

if __name__ == '__main__':
    run_backtest()
