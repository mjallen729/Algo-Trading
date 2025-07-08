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

    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
      print("Warning: Missing OHLCV data for technical indicators.")
      return df

    df['SMA_10'] = talib.SMA(df['close'], timeperiod=10)
    df['EMA_10'] = talib.EMA(df['close'], timeperiod=10)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['EMA_50'] = talib.EMA(df['close'], timeperiod=50)

    df['RSI'] = talib.RSI(df['close'], timeperiod=14)

    macd, macdsignal, macdhist = talib.MACD(
      df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist

    upper, middle, lower = talib.BBANDS(
      df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    df['BB_Upper'] = upper
    df['BB_Middle'] = middle
    df['BB_Lower'] = lower

    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)

    df['OBV'] = talib.OBV(df['close'], df['volume'])

    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    df['CCI'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)

    df['MFI'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)

    return df

  def add_lagged_features(self, df: pd.DataFrame, lags: list = [1, 2, 3, 5, 8]) -> pd.DataFrame:
    for col in ['open', 'high', 'low', 'close', 'volume', 'RSI', 'ATR', 'MFI']:
        if col in df.columns:
            for lag in lags:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

  def add_volatility(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    df['Volatility'] = df['close'].rolling(window=window).std()
    df['Volatility_pct'] = (df['Volatility'] / df['close']) * 100
    return df
    
  def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    return df
    
  def add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
    if 'SMA_10' in df.columns and 'SMA_50' in df.columns:
        df['SMA_ratio'] = df['SMA_10'] / df['SMA_50']
    if 'close' in df.columns and 'SMA_50' in df.columns:
        df['price_div_sma50'] = df['close'] / df['SMA_50']
    if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
        df['bb_width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    return df

  def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
    df = self.add_technical_indicators(df)
    df = self.add_volatility(df)
    df = self.add_time_features(df)
    df = self.add_interaction_features(df)
    df = self.add_lagged_features(df)

    df = df.dropna()

    return df


if __name__ == '__main__':
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