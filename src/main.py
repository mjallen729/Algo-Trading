from pipeline.generateTemporal import generateTemporal
import pandas as pd

# Main event loop
# TODO schedule recurring job
def main():
  df = pd.read_csv('~/Documents/Algo-Trading/data/crypto_processed.csv')
  if df:
    print('Found data')
  else:
    print('No training data found!')
    raise

  print('Converting to TimeSeries...')
  data = generateTemporal(df)

  # train the model