import pandas as pd
from pytorch_forecasting import GroupNormalizer, TimeSeriesDataSet

"""
INBOUND DATA (crypto_processed.csv)
  0   time_idx     1120061 non-null  int64
  1   symbol       1120061 non-null  object
  2   open         1120061 non-null  float64
  3   high         1120061 non-null  float64
  4   low          1120061 non-null  float64
  5   close        1120061 non-null  float64
  6   volume usdt  1120061 non-null  int64
  7   tradecount   1120061 non-null  int64
  8   token        1120061 non-null  object
  9   hour         1120061 non-null  int64
  10  weekday      1120061 non-null  object 
  11  year         1120061 non-null  int32  
  12  month        1120061 non-null  int32  
  13  day          1120061 non-null  int32
"""


def _preprocessing(df: pd.DataFrame) -> pd.DataFrame:
  # Mark all time series categories
  df['hour'] = df['hour'].astype(str).astype('category')
  df['weekday'] = df['weekday'].astype('category')
  df['month'] = df['month'].astype(str).astype('category')
  df['day'] = df['day'].astype(str).astype('category')

  # Convert to floats for normalizers
  df['volume usdt'] = df['volume usdt'].astype("float64")
  df['tradecount'] = df['tradecount'].astype("float64")
  df['year'] = df['year'].astype("float64")

  return df


def generateTemporal(df: pd.DataFrame) -> TimeSeriesDataSet:
  df = _preprocessing(df)

  dataset = TimeSeriesDataSet(
    data=df,
    time_idx="time_idx",
    target="close",
    group_ids=["symbol"],
    min_encoder_length=72,
    max_encoder_length=72,
    min_prediction_length=12,
    max_prediction_length=12,
    static_categoricals=["token"],
    time_varying_known_categoricals=["hour", "weekday", "month", "day"],
    time_varying_known_reals=["time_idx", "year"],
    time_varying_unknown_reals=["open", "high",
                                "low", "close", "volume usdt", "tradecount"],
    target_normalizer=GroupNormalizer(
      groups=['symbol'], transformation='softplus').fit(y=df["close"], X=df[["symbol"]]),
    scalers={
      "open": GroupNormalizer(groups=['symbol'], transformation='softplus').fit(y=df["close"], X=df[["symbol"]]),
      "high": GroupNormalizer(groups=['symbol'], transformation='softplus').fit(y=df["close"], X=df[["symbol"]]),
      "low": GroupNormalizer(groups=['symbol'], transformation='softplus').fit(y=df["close"], X=df[["symbol"]]),
      "volume usdt": GroupNormalizer(groups=['symbol'], transformation='softplus').fit(y=df["close"], X=df[["symbol"]]),
      "tradecount": GroupNormalizer(groups=['symbol'], transformation='softplus').fit(y=df["close"], X=df[["symbol"]]),
      "year": GroupNormalizer(groups=['symbol'], transformation='softplus').fit(y=df["close"], X=df[["symbol"]])
    }
  )

  return dataset


if __name__ == '__main__':
  p = _preprocessing(pd.read_csv(
    '~/Documents/Algo-Trading/data/crypto_processed.csv'))
  tp = generateTemporal(p)
