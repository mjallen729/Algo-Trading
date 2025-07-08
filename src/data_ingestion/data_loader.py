import pandas as pd
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import CryptoHistoricalDataClient
from datetime import datetime
import os
from src.config import DATA_DIR, TIME_FRAME


class DataLoader:
  def __init__(self, api_key: str, secret_key: str):
    self.client = CryptoHistoricalDataClient(api_key, secret_key)

  def load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    request_params = CryptoBarsRequest(
      symbol_or_symbols=[symbol],
      timeframe=TimeFrame.Hour,  # Using 1Hour as per spec, can be configured
      start=start_date,
      end=end_date
    )
    bars = self.client.get_crypto_bars(request_params).df
    if not bars.empty:
      # Alpaca returns a MultiIndex DataFrame, flatten it
      bars = bars.loc[symbol]
      bars.index = pd.to_datetime(bars.index)
      bars = bars[['open', 'high', 'low', 'close', 'volume']]
    return bars

  def load_local_csv(self, symbol: str) -> pd.DataFrame:
    # Clean the symbol to match the cleaned CSV file names (e.g., "ETH/USD" -> "eth")
    cleaned_symbol = symbol.split('/')[0].lower()
    file_path = os.path.join(DATA_DIR, f"{cleaned_symbol}.csv")
    if not os.path.exists(file_path):
      print(f"Warning: Local CSV for {symbol} not found at {file_path}")
      return pd.DataFrame()

    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df = df.set_index('Date')
    df = df.drop(columns=['Symbol', 'Year', 'Month', 'Day'], errors='ignore')
    
    # Rename OHLCV columns to match Alpaca's output for consistency
    df = df.rename(columns={
      'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    })
    
    # Keep additional features available in the cleaned data
    # Features like Market Cap, Volatility25, SMA25, EMA25, ATR25, RSI25, NextClose
    print(f"Loaded local data with columns: {list(df.columns)}")
    print(f"Available additional features: {[col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']]}")
    
    return df

  def get_data(self, symbol: str, start_date: datetime, end_date: datetime, use_local: bool = False) -> pd.DataFrame:
    if use_local:
      print(f"Loading local data for {symbol}...")
      return self.load_local_csv(symbol)
    else:
      print(
        f"Loading historical data from Alpaca for {symbol} from {start_date} to {end_date}...")
      return self.load_historical_data(symbol, start_date, end_date)


if __name__ == '__main__':
  # Example Usage (requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env)
  from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY
  if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    data_loader = DataLoader(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # Load from Alpaca
    eth_data_alpaca = data_loader.get_data(
      "ETH/USD",
      datetime(2024, 1, 1),
      datetime(2024, 1, 5)
    )
    print("ETH/USD data from Alpaca (first 5 rows):")
    print(eth_data_alpaca.head())

    # Load from local CSV
    btc_data_local = data_loader.get_data("BTC", use_local=True)
    print("\nBTC data from local CSV (first 5 rows):")
    print(btc_data_local.head())
  else:
    print("Alpaca API keys not found in .env. Cannot run example.")
