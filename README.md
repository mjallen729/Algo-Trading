# Algorithmic Cryptocurrency Trading Bot

This project is an algorithmic trading bot that uses a sophisticated deep learning model to trade cryptocurrencies on the Alpaca paper trading platform. The core of the bot is a **Temporal Fusion Transformer (TFT)** model, which is a state-of-the-art architecture for time-series forecasting.

## Features

- **Advanced Forecasting Model**: Utilizes a Temporal Fusion Transformer (TFT) for multi-horizon time-series forecasting, implemented with `pytorch-forecasting`.
- **Automated Trading**: Connects to the Alpaca API to execute trades in a paper trading environment.
- **Modular Architecture**: The code is organized into distinct modules for data ingestion, feature engineering, model training, prediction, risk management, and trading.
- **Research-Optimized Hyperparameters**: TFT hyperparameters optimized for cryptocurrency trading based on academic research.
- **Smart Data Caching**: Automatically caches hourly data from Alpaca API for efficient training.
- **Checkpoint Resume**: Option to resume training from interrupted sessions.
- **Rich Feature Engineering**: 100+ features including technical indicators, lagged features, and pre-calculated market data.
- **Robust Risk Management**: Implements ATR-based dynamic stop-loss and position sizing strategies.
- **Comprehensive Backtesting**: Includes an event-driven backtesting framework to rigorously test strategies.

## How It Works

The bot operates in different modes: **training**, **backtesting**, and **live trading**.

### Training
The model is trained on **2 years of hourly cryptocurrency data** (~17,500 samples) fetched from Alpaca API and cached locally. The TFT model learns complex temporal patterns with:
- **72-hour lookback window** (3 days of context)
- **12-hour prediction horizon** 
- **108 engineered features** (technical indicators, lagged features, time-based features)
- **Apple Silicon GPU acceleration** (MPS) for faster training

### Backtesting
The backtesting framework simulates the trading strategy on historical data, accounting for transaction costs and realistic market conditions.

### Live Trading
In live trading mode, the bot operates in a continuous loop:

1. **Data Ingestion**: Fetches latest market data from Alpaca API
2. **Feature Engineering**: Generates 100+ technical indicators and features
3. **Prediction**: Uses TFT model to forecast price movements 12 hours ahead
4. **Signal Generation**: Translates predictions into trading signals with configurable thresholds
5. **Risk Management**: ATR-based stop-loss and dynamic position sizing
6. **Trade Execution**: Places market orders on Alpaca paper trading platform

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Alpaca paper trading account**
- **Apple Silicon Mac** (recommended for MPS GPU acceleration) or CUDA-compatible GPU

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Create a `.env` file** in the root directory.

2. **Add your Alpaca API keys** and configuration variables:

```env
# === REQUIRED API CREDENTIALS ===
ALPACA_API_KEY="YOUR_API_KEY"
ALPACA_SECRET_KEY="YOUR_SECRET_KEY"
ALPACA_BASE_URL="https://paper-api.alpaca.markets/v2"

# === TRADING CONFIGURATION ===
SYMBOL="ETH/USD"              # Cryptocurrency pair to trade
INITIAL_CAPITAL="100000.0"    # Starting capital
```

**Note**: Most hyperparameters are optimized in `src/config.py` based on cryptocurrency trading research. The `.env` file is primarily for API keys and deployment-specific settings.

## Usage

### 1. Train the Model

Train the TFT model on 2 years of hourly data:

```bash
./env/bin/python -m src.main --mode train
```

**Features:**
- ✅ **Smart caching**: Downloads data once, reuses cached hourly data
- ✅ **17,500+ training samples** from 2 years of ETH hourly data  
- ✅ **Apple Silicon acceleration** with MPS GPU support
- ✅ **Research-optimized hyperparameters** for crypto trading

**Training Time**: ~15-30 minutes on Apple M1 Pro with MPS acceleration

### 2. Resume Training (Optional)

If training was interrupted, resume from the latest checkpoint:

```bash
./env/bin/python -m src.main --mode train --resume
```

### 3. Run Backtesting

Evaluate strategy performance on historical data:

```bash
./env/bin/python -m src.main --mode backtest
```

### 4. Live Trading

Run the algorithmic trading bot:

```bash
./env/bin/python -m src.main --mode trade
```

## Architecture & Technical Details

### Temporal Fusion Transformer Configuration

**Research-Optimized for Cryptocurrency Trading:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `hidden_size` | 128 | 4x increase for crypto complexity |
| `attention_heads` | 4 | Standard for TFT models |
| `dropout` | 0.3 | Higher dropout for crypto volatility |
| `learning_rate` | 3e-4 | Conservative for stable training |
| `encoder_length` | 72 hours | 3 days of market context |
| `prediction_length` | 12 hours | Realistic trading horizon |
| `batch_size` | 64 | Optimized for Apple Silicon |

### Data Pipeline

1. **Hourly Data**: 2 years of ETH/USD data from Alpaca API (~17,500 samples)
2. **Feature Engineering**: 108 features including:
   - Technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
   - Lagged features (1, 2, 3, 5, 8 period lags)
   - Time-based features (cyclical hour/day encoding)
   - Pre-calculated features from cleaned data
3. **Train/Validation Split**: 80/20 split with temporal ordering
4. **Caching**: Smart caching system for efficient re-training

### Risk Management

- **ATR-based Stop Loss**: Dynamic stop-loss based on market volatility
- **Position Sizing**: Risk-adjusted position sizing
- **Signal Thresholds**: Configurable buy/sell signal thresholds
- **Portfolio Management**: Tracks positions, capital, and trade history

## Technology Stack

- **Python 3.10+**: Core language
- **PyTorch & Lightning**: Deep learning framework with Apple Silicon support
- **PyTorch Forecasting**: TFT implementation and time-series utilities
- **Pandas & NumPy**: Data manipulation
- **TA-Lib**: Technical analysis indicators (optional)
- **Alpaca-py**: Trading API integration
- **Backtesting.py**: Strategy evaluation framework

## Project Structure

```
src/
├── main.py                    # Main entry point with mode selection
├── config.py                  # Centralized configuration (hyperparameters)
├── data_ingestion/
│   └── data_loader.py         # Alpaca API + caching system
├── feature_engineering/
│   └── features.py            # 100+ feature engineering pipeline
├── models/
│   ├── tft_model.py           # TFT model implementation
│   └── predict.py             # Signal generation
├── risk_management/
│   └── portfolio.py           # Position sizing & risk management
├── trading/
│   └── trader.py              # Alpaca trading interface
└── backtesting/
    ├── run_backtest.py        # Backtesting orchestration
    └── tft_strategy.py        # TFT trading strategy
```

## Performance Expectations

**Based on research-optimized hyperparameters:**

- **Training Data**: 17,500+ hourly samples (vs 1,800 daily)
- **Model Capacity**: 4x larger hidden size for crypto complexity  
- **Validation**: Proper 80/20 split for reliable performance metrics
- **Hardware**: 2-5x faster training with Apple Silicon MPS acceleration

## Development & Contribution

### Running Tests
```bash
pytest tests/
```

### Configuration Philosophy

- **`src/config.py`**: Research-optimized defaults for all hyperparameters
- **`.env` file**: Only secrets and deployment-specific settings
- **Override capability**: Can override any parameter in `.env` for experiments

### Contribution Guidelines

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Follow existing code style and add tests
4. Update documentation as needed
5. Submit pull request with detailed description

## License

[Add your license information here]

## Disclaimer

This software is for educational and research purposes only. Cryptocurrency trading involves substantial risk of loss. Past performance does not guarantee future results. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.
