import pandas as pd
import numpy as np

try:
  import talib
  TALIB_AVAILABLE = True
except ImportError:
  print("TA-Lib not installed. Some features will be unavailable.")
  TALIB_AVAILABLE = False


class FeatureEngineer:
  def __init__(self):
    pass

  def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    if not TALIB_AVAILABLE:
      print("Skipping technical indicator generation: TA-Lib not available.")
      return df

    # Ensure the DataFrame has the required columns
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
      print("Warning: Missing OHLCV data for technical indicators.")
      return df

    # Moving Averages
    df['SMA_10'] = talib.SMA(df['close'], timeperiod=10)
    df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)

    # Relative Strength Index (RSI)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)

    # Moving Average Convergence Divergence (MACD)
    macd, macdsignal, macdhist = talib.MACD(
      df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    # Bollinger Bands
    upper, middle, lower = talib.BBANDS(
      df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    # Average True Range (ATR)
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    # On-Balance Volume (OBV)
    df['OBV'] = talib.OBV(df['close'], df['volume'])

    return df

  def add_lagged_features(self, df: pd.DataFrame, lags: list = [1, 2, 3]) -> pd.DataFrame:
    for col in ['open', 'high', 'low', 'close', 'volume']:
      for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

  def add_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['Volatility'] = df['close'].rolling(window=window).std()
    return df

  def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self.add_technical_indicators(df)
    df = self.add_lagged_features(df)
    df = self.add_volatility(df)

    # Drop rows with NaN values created by feature engineering
    df = df.dropna()

    return df


if __name__ == '__main__':
  # Example Usage
  data = {
    'open': np.random.rand(100) * 100,
    'high': np.random.rand(100) * 100 + 1,
    'low': np.random.rand(100) * 100 - 1,
    'close': np.random.rand(100) * 100,
    'volume': np.random.rand(100) * 1000
  }
  index = pd.date_range(start='2023-01-01', periods=100, freq='H')
  sample_df = pd.DataFrame(data, index=index)

  feature_engineer = FeatureEngineer()
  engineered_df = feature_engineer.engineer_features(sample_df.copy())

  print("Original DataFrame head:")
  print(sample_df.head())
  print("\nEngineered DataFrame head (with new features):")
  print(engineered_df.head())
  print("\nEngineered DataFrame columns:")
  print(engineered_df.columns)
