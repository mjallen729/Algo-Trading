"""
Data loading utilities for historical and real-time cryptocurrency data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ccxt
from datetime import datetime, timedelta
import asyncio
import websocket
import json
import threading
import time

from utils import get_logger, config

logger = get_logger(__name__)


class DataLoader:
  """Main data loading class for cryptocurrency market data."""

  def __init__(self, data_dir: str = None):
    """
    Initialize data loader.

    Args:
      data_dir: Path to directory containing CSV data files
    """
    if data_dir is None:
      self.data_dir = Path(__file__).parent.parent.parent / "data"
    else:
      self.data_dir = Path(data_dir)

    self.supported_assets = config.get(
      'trading.target_assets', ['BTC', 'ETH', 'SOL'])
    self.feature_columns = config.get('data.feature_columns')

    logger.info(f"DataLoader initialized with data directory: {self.data_dir}")

  def load_historical_data(self,
                           symbol: str,
                           start_date: str = None,
                           end_date: str = None) -> pd.DataFrame:
    """
    Load historical data for a cryptocurrency.

    Args:
      symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
      start_date: Start date in YYYY-MM-DD format
      end_date: End date in YYYY-MM-DD format

    Returns:
      DataFrame with historical market data
    """
    symbol_lower = symbol.lower()
    file_path = self.data_dir / f"{symbol_lower}.csv"

    if not file_path.exists():
      raise FileNotFoundError(f"Data file not found: {file_path}")

    logger.info(f"Loading historical data for {symbol} from {file_path}")

    # Load data
    df = pd.read_csv(file_path)

    # Create datetime index
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df.set_index('datetime', inplace=True)

    # Filter by date range if specified
    if start_date:
      df = df[df.index >= start_date]
    if end_date:
      df = df[df.index <= end_date]

    # Ensure we have required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
      raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"Loaded {len(df)} records for {symbol}")
    return df

  def load_multiple_assets(self,
                           symbols: List[str] = None,
                           start_date: str = None,
                           end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load historical data for multiple cryptocurrencies.

    Args:
        symbols: List of cryptocurrency symbols
        start_date: Start date in YYYY-MM-DD format  
        end_date: End date in YYYY-MM-DD format

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    if symbols is None:
      symbols = self.supported_assets

    data = {}
    for symbol in symbols:
      try:
        data[symbol] = self.load_historical_data(symbol, start_date, end_date)
        logger.info(f"Successfully loaded data for {symbol}")
      except Exception as e:
        logger.error(f"Failed to load data for {symbol}: {e}")

    return data

  def prepare_training_data(self,
                            df: pd.DataFrame,
                            sequence_length: int = 60,
                            target_column: str = 'NextClose') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for machine learning model training.

    Args:
      df: DataFrame with market data
      sequence_length: Length of input sequences
      target_column: Name of target column

    Returns:
      Tuple of (features, targets) arrays
    """
    # Select feature columns
    feature_cols = [col for col in self.feature_columns if col in df.columns]

    if not feature_cols:
      raise ValueError("No valid feature columns found in data")

    # Prepare features
    features_df = df[feature_cols].copy()

    # Handle missing values
    features_df = features_df.fillna(method='ffill').fillna(method='bfill')

    # Normalize features (z-score normalization)
    features_normalized = (
      features_df - features_df.mean()) / features_df.std()

    # Create sequences
    X, y = [], []

    for i in range(sequence_length, len(features_normalized)):
      if target_column in df.columns and not pd.isna(df[target_column].iloc[i]):
        X.append(features_normalized.iloc[i - sequence_length:i].values)
        y.append(df[target_column].iloc[i])

    X = np.array(X)
    y = np.array(y)

    logger.info(
      f"Prepared training data: X shape {X.shape}, y shape {y.shape}")
    return X, y


class RealTimeDataStream:
  """Real-time data streaming for live trading."""

  def __init__(self, exchange_id: str = 'binance'):
    """
    Initialize real-time data stream.

    Args:
      exchange_id: Exchange identifier for CCXT
    """
    self.exchange_id = exchange_id
    self.exchange = getattr(ccxt, exchange_id)()
    self.is_streaming = False
    self.data_callbacks = []

    logger.info(f"RealTimeDataStream initialized for {exchange_id}")

  def add_callback(self, callback):
    """Add callback function for real-time data."""
    self.data_callbacks.append(callback)

  async def fetch_current_price(self, symbol: str) -> Dict:
    """
    Fetch current price for a symbol.

    Args:
      symbol: Trading symbol (e.g., 'BTC/USDT')

    Returns:
      Dictionary with current market data
    """
    try:
      ticker = await self.exchange.fetch_ticker(symbol)
      return {
        'symbol': symbol,
        'price': ticker['last'],
        'bid': ticker['bid'],
        'ask': ticker['ask'],
        'volume': ticker['baseVolume'],
        'timestamp': ticker['timestamp']
      }
    except Exception as e:
      logger.error(f"Error fetching price for {symbol}: {e}")
      return None

  def start_streaming(self, symbols: List[str]):
    """Start real-time data streaming for multiple symbols."""
    self.is_streaming = True
    logger.info(f"Starting real-time data stream for {symbols}")

    # This would typically connect to WebSocket feeds
    # For MVP, we'll simulate with periodic API calls
    threading.Thread(target=self._simulate_stream,
                     args=(symbols,), daemon=True).start()

  def _simulate_stream(self, symbols: List[str]):
    """Simulate real-time data stream with periodic API calls."""
    while self.is_streaming:
      for symbol in symbols:
        try:
          # Convert symbol format for CCXT
          ccxt_symbol = f"{symbol}/USDT"
          data = asyncio.run(self.fetch_current_price(ccxt_symbol))

          if data:
            # Notify all callbacks
            for callback in self.data_callbacks:
              callback(data)

        except Exception as e:
          logger.error(f"Error in data stream for {symbol}: {e}")

      time.sleep(30)  # Update every 30 seconds for MVP

  def stop_streaming(self):
    """Stop real-time data streaming."""
    self.is_streaming = False
    logger.info("Real-time data streaming stopped")
