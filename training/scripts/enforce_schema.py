"""
Adjust the schema to match standard OHLCV data pipelines. This will overwrite the
old data csv (an original still exists in the zip).

This should only be run once.
"""

import pandas as pd

df = pd.read_csv("../data/raw/aggregated/crypto_data.csv")

# Adjust the schema to match standard OHLCV data pipelines
df.drop(columns=["symbol", "tradecount", "day", "hour"], inplace=True)
df.rename(columns={"volume usdt": "volume"}, inplace=True)

# Overwrite the old data (a copy still exists in the zip)
df.to_csv("../data/raw/aggregated/crypto_data.csv", index=False)
