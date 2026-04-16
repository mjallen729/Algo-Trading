# Algorithmic Cryptocurrency Trading System

A deep learning-based algorithmic trading system for cryptocurrency markets using Temporal Fusion Transformers (TFT) to predict hourly price movements across 19 altcoins.

## Overview

This project implements a machine learning pipeline for cryptocurrency price forecasting and algorithmic trading. The system uses PyTorch Forecasting's Temporal Fusion Transformer architecture to predict next-hour closing prices by analyzing historical price patterns, broader market context (BTC/ETH), and temporal features.

### Key Objectives

- **Prediction Task**: Forecast 12-step ahead log returns for cryptocurrency closing prices
- **Trading Universe**: 19 altcoins (ADA, ALGO, ATOM, BAL, BCH, BNB, COMP, DENT, DOGE, ETC, HBAR, LINK, LTC, MATIC, MKR, RVN, SOL, TRX, XMR)
- **Market Context**: BTC and ETH excluded from trading universe but used as features (market too efficient)
- **Data Frequency**: Hourly OHLCV data
- **Time Horizon**: August 2017 - October 2023

## Obstacles

### Data Integrity
The initial dataset, merged from multiple Alpaca API fetches, contained 10,735 missing hourly timesteps across tokens. Some gaps spanned entire days — the largest being 784 hours (~33 days) for a single token. This rendered the merged dataset unusable for sequential time series modeling, requiring a switch to a bulk historical data source. After processing, 108,270 rows were dropped during feature engineering (log return NaNs at series boundaries), leaving 736,065 clean rows with zero missing values.

### Training
Because it is so large, the model cannot be trained locally on Apple Silicon hardware within a reasonable timeframe. Even with a reduced architecture (0.1M parameters, hidden size of 32) and aggressive batching (batch size 512, num_workers=0 for stability), training across ~1,000 batches per epoch on 512K samples is impractical without dedicated GPU resources. Migrated to Amazon SageMaker for training and hyperparameter tuning (via bayesian optimization).

### Extreme Outliers
Cryptocurrency data exhibits ~2.5% extreme outliers (flash crashes, pumps) that distort standard normalization. Z-score scaling compresses the usable range when mean and standard deviation are contaminated by these tail events. This required adopting robust quantile-based scaling (10th/90th percentiles) and per-token normalization to prevent high-volatility tokens from dominating gradients over lower-volatility tokens.

## Technical Architecture

### Model: Temporal Fusion Transformer (TFT)

The TFT architecture is specifically designed for multi-horizon time series forecasting with:
- **Attention mechanisms** for interpretable feature importance
- **Multi-horizon prediction** with quantile forecasting for uncertainty estimation
- **Mixed data types** supporting both continuous and categorical features
- **Static and temporal covariates** for richer context

## Data Pipeline

### Raw Data Schema
- **Source**: Alpaca Markets API (see `scripts/alpaca_data_fetch.ipynb`)
- **Format**: `[symbol, date, open, high, low, close, volume_usd]`
- **Total observations**: 844,335 hourly bars across 21 symbols
- **Date range**: 2017-08-17 to 2023-10-19

### Preprocessing Pipeline

The preprocessing transforms raw OHLCV data into stationary, normalized features suitable for deep learning:

#### 1. Log Returns Transformation
All price features are converted to log returns to achieve stationarity:
```
log_return = ln(price_t / price_{t-1})
```

This transformation:
- Makes the data stationary (constant mean/variance over time)
- Normalizes different price scales (e.g., $0.001 vs $50,000)
- Captures percentage changes rather than absolute movements

**Features created**:
- `open_log`, `high_log`, `low_log`, `close_log`: OHLC log returns
- `volume_log`: ln(1 + volume) to compress range
- `btc_close_log`, `eth_close_log`: BTC/ETH log returns
- `next_close_log`: Target variable (next hour's close log return)

#### 2. Market Context Features
- `btc_close_log`: Bitcoin log returns (proxy for crypto market sentiment)
- `eth_close_log`: Ethereum log returns (alternative L1 indicator)
- `eth_btc_ratio`: ETH/BTC price ratio (regime indicator - captures altcoin vs BTC dominance shifts)

**Rationale**: Strong correlation exists between BTC/ETH movements and altcoin prices. These features provide market-wide context for individual token predictions.

#### 3. Cyclical Temporal Features
Time-based features encoded as sine/cosine pairs to preserve cyclicity:

```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
```

**Features**:
- `hour_sin`, `hour_cos`: Intraday patterns (24-hour cycle)
- `norm_day_sin`, `norm_day_cos`: Day-of-month patterns (normalized by month length)
- `weekday_sin`, `weekday_cos`: Day-of-week patterns (7-day cycle)
- `month_sin`, `month_cos`: Seasonal patterns (12-month cycle)
- `year`: Linear year feature normalized to [0, 1] per decade (allows extrapolation)

**Rationale**: Cyclical encoding ensures that hour 23 and hour 0 are close in feature space, preventing discontinuities at cycle boundaries.

#### 4. Robust Normalization
Uses `GroupNormalizer` with robust scaling to handle crypto's extreme outliers:

- **Method**: Robust scaling based on quantiles (10th and 90th percentiles)
- **Grouping**: Per-token normalization for token-specific features, global for BTC/ETH
- **Why robust**: Standard z-score normalization fails with outliers (crypto has ~2.5% extreme outliers)

### Preprocessed Data Schema

**Final feature set** (20 columns):
```
time_idx, symbol,
open_log, high_log, low_log, close_log, volume_log,
btc_close_log, eth_close_log, eth_btc_ratio,
hour_sin, hour_cos, norm_day_sin, norm_day_cos,
weekday_sin, weekday_cos, month_sin, month_cos, year,
next_close_log (target)
```

**Statistics**:
- Total rows: 736,065
- Tokens: 19
- Time indices: 0 to 52,029 (varies by token based on listing date)
- Missing data: 0 (all NaN rows dropped after feature engineering)

### Data Splits

**Sequential split to prevent lookahead bias**:
- Train: 70% (earliest data) - 515,244 samples
- Validation: 15% (middle) - 110,407 samples
- Test: 15% (latest data) - 110,414 samples

**Critical**: Splits are done per-token on `time_idx` to maintain temporal ordering and simulate production (train on past, predict future).

## TimeSeriesDataSet Configuration

### Sequence Parameters
- **Encoder length**: 168 timesteps (1 week of hourly data)
- **Prediction horizon**: 12 timesteps (12 hours ahead)
- **Minimum encoder length**: 168 (fixed-length sequences)

**Rationale**: 1-week lookback captures weekly patterns while maintaining stationarity. Longer sequences risk non-stationarity in volatile crypto markets.

### Feature Categories

**Time-varying known reals** (available at prediction time):
- All cyclical temporal features (hour, day, weekday, month, year)

**Time-varying unknown reals** (only known historically):
- All financial features (OHLCV log returns, BTC/ETH context)

**Static categoricals**:
- `symbol`: Token identifier for embeddings

**Target**:
- `next_close_log`: 1-hour ahead close log return

### Normalization Strategy

All financial features use `GroupNormalizer` with:
- **Method**: `robust` (quantile-based, resistant to outliers)
- **Center**: `True` (zero-mean)
- **Quantiles**: 10th and 90th percentiles (wider than IQR to preserve tail information)

**Group-specific normalization**:
- Token features (OHLCV): Normalized per `symbol` to handle different volatility regimes
- Market features (BTC/ETH): Global normalization to preserve cross-token relationships

## Key Design Decisions

### 1. Log Returns vs. Raw Prices
**Decision**: Use log returns exclusively
**Rationale**:
- Stationarity: Prices are non-stationary (trending), log returns are stationary
- Scale-invariance: $0.001 to $0.002 = same % change as $1000 to $2000
- Interpretability: Log returns ≈ percentage changes for small values
- Model-friendly: Bounded range, approximately normal distribution

### 2. BTC/ETH as Features, Not Trading Targets
**Decision**: Include BTC/ETH data as features but don't trade them
**Rationale**:
- Market efficiency: BTC/ETH markets are highly efficient, harder to predict
- Context provider: Strong correlation between BTC dominance and altcoin performance
- Regime detection: ETH/BTC ratio captures market regime shifts (altcoin vs. BTC preference)

### 3. Robust Scaling over Standard Normalization
**Decision**: Use quantile-based robust scaling
**Rationale**:
- Outlier prevalence: Crypto has ~2.5% extreme outliers (flash crashes, pumps)
- Standard scaling issues: Mean/std contaminated by outliers, leads to compressed ranges
- Quantile robustness: 10th/90th percentiles ignore extreme values while preserving distribution shape

### 4. Per-Token Normalization
**Decision**: Normalize OHLCV features separately for each token
**Rationale**:
- Heterogeneous volatility: MATIC variance = 0.000276 vs XMR variance = 0.000104 (2.7x difference)
- Token-specific patterns: Each token has unique volatility regime and price dynamics
- Prevents dominance: High-volatility tokens won't dominate gradients over low-volatility tokens

### 5. 1-Week Encoder Length
**Decision**: 168-hour (1 week) lookback window
**Rationale**:
- Captures weekly patterns (weekday effects, weekend dynamics)
- Short enough to maintain stationarity assumption
- Balances model capacity with overfitting risk
- Practical for inference (reasonable context window)

## Trading Strategy Considerations

See `docs/profitability.md` for detailed profitability optimization strategies.

### Critical Friction Factors
When implementing trading strategies, account for:

1. **Transaction costs**: Trading fees (0.1-0.5% depending on exchange/tier)
2. **Slippage**: Market impact and bid-ask spread (especially on smaller caps)
3. **Funding rates**: For perpetual futures positions
4. **Latency**: Execution delay between signal and fill
5. **Liquidity constraints**: Position size limits based on order book depth

### Signal Interpretation

The model predicts `next_close_log` (log return for the next 12 hours). To convert back to price:

```python
predicted_price = current_close * exp(predicted_next_close_log)
```

For trading signals:
- **Long signal**: predicted_next_close_log > threshold (e.g., 0.002 for 0.2% expected gain)
- **Short signal**: predicted_next_close_log < -threshold
- **No trade**: |predicted_next_close_log| < threshold

**Uncertainty-aware trading**: Use TFT's quantile predictions for confidence filtering:
- Only trade when prediction confidence is high (narrow quantile spread)
- Larger position sizes when uncertainty is low

## Directory Structure

```
.
├── data/
│   ├── coins.csv              # Raw OHLCV data from Alpaca
│   ├── preproc_coins.csv      # Preprocessed features
│   └── bin/                   # Serialized TimeSeriesDataSet objects
├── notebooks/
│   ├── preprocess.ipynb       # Data preprocessing pipeline
│   ├── explore_preproc.ipynb  # Data exploration & validation
│   ├── visuals.ipynb          # Distribution analysis & correlations
│   └── timeSeriesDataset.ipynb # PyTorch Forecasting dataset creation
├── scripts/
│   ├── alpaca_data_fetch.ipynb # Alpaca API data ingestion
│   └── merge_datasets.ipynb   # Dataset merging utilities
├── docs/
│   ├── profitability.md       # Trading strategy & profitability guidelines
│   └── archive/               # Historical documentation
├── models/                    # Saved model checkpoints
├── checkpoints/               # Training checkpoints
└── requirements.txt           # Python dependencies
```

## Setup

### Prerequisites
- Python 3.13+
- GPU recommended (CUDA-compatible) for training

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Algo-Trading
```

2. Create virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Loading Preprocessed Data
```python
import pandas as pd

# Load preprocessed features
df = pd.read_csv('data/preproc_coins.csv')

# Or load PyTorch Forecasting datasets
from pytorch_forecasting import TimeSeriesDataSet

training = TimeSeriesDataSet.load('data/bin/training_temporaldataset.pkl')
validation = TimeSeriesDataSet.load('data/bin/validation_temporaldataset.pkl')
test = TimeSeriesDataSet.load('data/bin/test_temporaldataset.pkl')
```

### Example: Inverse Transform Predictions
```python
# Model outputs normalized log returns
# Convert back to price predictions:
import numpy as np

current_price = 1.50  # Current close price
predicted_log_return = 0.003  # Model prediction

predicted_price = current_price * np.exp(predicted_log_return)
print(f"Predicted next close: ${predicted_price:.4f}")
# Expected gain: {(predicted_log_return * 100):.2f}%
```

## References

- [Temporal Fusion Transformers Paper](https://arxiv.org/abs/1912.09363)
- [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/)
- [Alpaca Markets API](https://alpaca.markets/docs/)
