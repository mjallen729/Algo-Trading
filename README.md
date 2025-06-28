# Cryptocurrency Trading Superalgorithm 🚀

A sophisticated algorithmic trading system for cryptocurrencies that combines cutting-edge machine learning, advanced market regime detection, and intelligent risk management to maximize trading profitability.

## 🎯 Project Goals

1. **Maximum Profitability**: Trade as profitably as possible regardless of strategy/risk
2. **Prediction Accuracy**: Make extremely accurate predictions that can be actionably traded upon  
3. **Cutting-Edge Technology**: Use state-of-the-art machine learning and algorithmic trading techniques

## 🏗️ Architecture Overview

### Core Components

```
algo-trading/
├── src/
│   ├── data/           # Data loading & preprocessing
│   ├── models/         # ML models & regime detection
│   ├── strategies/     # Trading strategies
│   ├── execution/      # Order execution & portfolio management
│   ├── risk/           # Risk management & position sizing
│   └── utils/          # Utilities & configuration
├── data/               # Historical market data
├── models/             # Trained ML models
└── configs/            # Configuration files
```

### Data Flow

```
1. Data Ingestion (1-minute intervals)
   ├── Historical data loading
   ├── Real-time price streaming
   └── Technical indicator calculation

2. Feature Engineering
   ├── Price-based features (returns, volatility)
   ├── Volume profiles
   ├── Technical indicators (50+ indicators)
   └── Time-based features

3. Prediction Pipeline
   ├── Regime detection → Market state
   ├── Price prediction → Direction & magnitude
   └── Confidence scoring → Position sizing

4. Strategy Execution
   ├── Strategy selection based on regime
   ├── Signal generation
   ├── Position sizing calculation
   └── Order creation

5. Risk Management
   ├── Pre-trade checks (drawdown, exposure)
   ├── Order validation
   ├── Stop-loss/take-profit setting
   └── Portfolio rebalancing

6. Order Execution
   ├── Submit to paper trading API
   ├── Monitor fills
   └── Update portfolio

7. Performance Monitoring
   ├── Real-time P&L tracking
   ├── Risk metrics calculation
   └── Strategy performance logging
```

## 🧠 Machine Learning Models

### 1. Hybrid Prediction Engine
- **LSTM Networks**: Capture temporal dependencies and short-term patterns
- **Transformer Models**: Analyze long-range dependencies and complex relationships
- **Ensemble Approach**: Combine predictions with confidence weighting

### 2. Market Regime Detection
- **Hidden Markov Models**: Sequential regime modeling
- **K-means Clustering**: Pattern-based regime identification  
- **Gaussian Mixture Models**: Probabilistic regime classification
- **Ensemble Voting**: Weighted combination of all approaches

### 3. Advanced Features
- **Technical Indicators**: 50+ indicators including RSI, MACD, Bollinger Bands
- **Volume Analysis**: Order flow, volume profiles, whale tracking
- **Time-based Features**: Market sessions, cyclical encoding
- **Momentum Features**: Multi-timeframe momentum analysis

## ⚡ Current Status

### ✅ Completed Components

- **Data Loading System**: Loads historical cryptocurrency data (BTC, ETH, SOL, DOGE, SHIB)
- **Advanced Preprocessor**: Comprehensive feature engineering with 50+ technical indicators
- **Regime Detection**: Multi-model approach with HMM, K-means, and GMM
- **Position Sizing**: Kelly Criterion, ATR-based, and confidence-weighted sizing
- **Risk Management**: Portfolio heat monitoring and drawdown management
- **Model Management**: Centralized model storage and loading system

### 🔄 In Progress

- **LSTM/Transformer Models**: Deep learning prediction models
- **Strategy Implementation**: Momentum, mean reversion, and arbitrage strategies
- **Paper Trading Integration**: Binance testnet and simulation environment
- **Real-time Data Streaming**: Live market data feeds

### 📋 Next Steps

1. Complete ML model implementation (LSTM + Transformer ensemble)
2. Integrate paper trading APIs for safe testing
3. Add real-time data streaming
4. Implement strategy backtesting framework
5. Deploy monitoring and alerting systems

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+ required
python3 --version

# Virtual environment (recommended)
python3 -m venv env
source env/bin/activate  # Linux/Mac
# or
env\Scripts\activate     # Windows
```

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd algo-trading

# Install dependencies
pip install -r requirements.txt

# Test components
python3 test_components.py
```

### Basic Usage

```python
from src.data.loaders import DataLoader
from src.data.preprocessors import AdvancedPreprocessor
from src.models.regime_detector import RegimeDetector
from src.risk.position_sizing import AdvancedPositionSizer, TradingSignal

# Load and preprocess data
loader = DataLoader()
preprocessor = AdvancedPreprocessor()

data = loader.load_historical_data("btc")
processed_data = preprocessor.preprocess(data)

# Detect market regime
regime_detector = RegimeDetector()
regime_detector.fit(processed_data)
current_regime = regime_detector.predict(processed_data.tail(100))

# Calculate position size
position_sizer = AdvancedPositionSizer()
signal = TradingSignal(
    action="buy",
    symbol="BTC",
    price=50000.0,
    confidence=0.75,
    strategy="momentum",
    timestamp=datetime.now()
)

position_size = position_sizer.calculate_position_size(
    signal=signal,
    symbol="BTC",
    portfolio_value=10000.0
)

print(f"Current regime: {current_regime['regime_label']}")
print(f"Recommended position: ${position_size:.2f}")
```

## 📊 Data Sources

### Historical Data
- **Assets**: BTC, ETH, SOL, DOGE, SHIB
- **Timeframe**: Daily data from 2020-2025
- **Features**: OHLCV + 50+ technical indicators

### Real-time Data (Planned)
- **Exchanges**: Binance, Coinbase Pro, Kraken
- **Frequencies**: 1m, 5m, 15m, 1h intervals
- **Order Book**: Level 2 market depth data

## 🛡️ Risk Management

### Position Sizing Methods
- **Kelly Criterion**: Optimal growth with fractional safety
- **ATR-based**: Volatility-adjusted position sizing
- **Confidence Weighted**: Signal strength-based allocation
- **Portfolio Heat**: Overall exposure monitoring

### Risk Controls
- **Maximum Drawdown**: 15% portfolio limit
- **Position Limits**: 25% maximum per trade
- **Regime Filtering**: Reduce exposure in high volatility
- **Dynamic Stop-Loss**: ATR and volatility-based exits

## 🔧 Technology Stack

### Core Technologies
- **Python 3.10+**: Main programming language
- **PyTorch**: Deep learning framework
- **NumPy/Pandas**: Data manipulation
- **Scikit-learn**: Traditional ML algorithms
- **TA-Lib**: Technical analysis indicators

### Trading Infrastructure
- **CCXT**: Unified exchange connectivity
- **WebSockets**: Real-time data streaming
- **Docker**: Containerized deployment
- **Redis**: High-frequency data caching

### Machine Learning
- **HMMLearn**: Hidden Markov Models
- **Transformers**: Attention-based models
- **Optuna**: Hyperparameter optimization
- **MLflow**: Experiment tracking

## 📈 Performance Targets

### MVP Goals
- **Prediction Accuracy**: 65%+ directional accuracy
- **Sharpe Ratio**: > 2.0
- **Maximum Drawdown**: < 15%
- **Win Rate**: > 55%
- **Monthly Returns**: 20%+ in paper trading

### Advanced Goals
- **Multi-asset Support**: 10+ cryptocurrencies
- **High-frequency Trading**: Sub-second execution
- **Alternative Data**: Sentiment, on-chain metrics
- **Reinforcement Learning**: Adaptive strategy selection

## 🚨 Important Notes

### Current Limitations
- **Paper Trading Only**: No real money at risk during development
- **Limited Assets**: Currently supports 5 major cryptocurrencies
- **Historical Data**: Backtesting on 2020-2025 daily data only
- **No Live Trading**: Real trading requires additional safety measures

### Risk Disclaimers
- **Development Stage**: This is experimental software under active development
- **No Financial Advice**: This system is for educational and research purposes
- **Market Risk**: Cryptocurrency trading involves significant financial risk
- **Use at Own Risk**: Users are responsible for their own trading decisions

## 🤝 Contributing

This is currently a private development project. Future open-sourcing is under consideration.

## 📄 License

Proprietary - All rights reserved.

---

**Last Updated**: June 28, 2025
**Version**: 0.1.0 (MVP Development)
**Status**: Active Development 🚧
