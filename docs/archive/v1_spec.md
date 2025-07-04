# MVP Implementation Plan: Algorithmic Trading System

This document outlines the implementation plan for the Minimum Viable Product (MVP) of our cryptocurrency algorithmic trading system. The plan prioritizes ease of startup and low-cost deployment while laying the foundation for a sophisticated, scalable "superalgorithm."

## 1. Project Philosophy

The MVP will be a fully functional trading bot capable of making profitable trades in a paper trading environment. It will be built on a modular architecture that allows for easy expansion and integration of more advanced features in the future. The core of the MVP will be a machine learning model that generates trading signals based on market data.

## 2. Chosen API: Alpaca

For the MVP, we will use the **Alpaca API**. It has been selected for the following reasons:

*   **Free Paper Trading:** Alpaca provides a robust paper trading environment that mirrors live market conditions, which is ideal for testing and development without risking real capital.
*   **Crypto Support:** It offers real-time and historical data for major cryptocurrencies.
*   **Python SDK:** Alpaca has a well-documented and easy-to-use Python library (`alpaca-py`), which will accelerate development.
*   **Modern & Developer-Friendly:** The API is modern, well-documented, and designed for algorithmic trading.

## 3. Proposed File Structure

The project will be organized into the following structure to ensure modularity and scalability:

```
/
├── data/                 # Existing historical data for backtesting
├── docs/                 # Project documentation
│   └── MVP_Implement.md  # This file
├── src/
│   ├── __init__.py
│   ├── main.py             # Main application entry point
│   ├── config.py           # Configuration (API keys, trading params)
│   ├── data_ingestion/
│   │   ├── __init__.py
│   │   └── data_loader.py  # Module for fetching data (Alpaca & local CSVs)
│   ├── feature_engineering/
│   │   ├── __init__.py
│   │   └── features.py     # Feature creation and selection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tft_model.py    # The ML model implementation
│   │   └── predict.py      # Prediction generation
│   ├── trading/
│   │   ├── __init__.py
│   │   └── trader.py       # Logic for placing trades via Alpaca
│   └── risk_management/
│       ├── __init__.py
│       └── portfolio.py    # Portfolio and risk management logic
├── notebooks/              # Jupyter notebooks for exploration and research
│   └── 1_model_development.ipynb
├── tests/                  # Unit and integration tests
│   ├── __init__.py
│   ├── test_features.py
│   └── test_trader.py
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration for deployment
└── .env                    # For storing secrets like API keys
```

## 4. Data Flow

The data will flow through the system in a clear, linear fashion:

1.  **Data Ingestion:** The `data_loader.py` module will fetch historical and real-time OHLCV (Open, High, Low, Close, Volume) data from Alpaca's API. For backtesting and initial model training, it will use the pre-existing CSV files in the `/data` directory.
2.  **Feature Engineering:** The raw data will be passed to `features.py`, which will generate a set of predictive features. For the MVP, this will primarily consist of technical indicators like RSI, MACD, Bollinger Bands, and rolling averages.
3.  **Prediction:** The engineered features will be fed into the pre-trained machine learning model in `tft_model.py`. The `predict.py` script will use the model to generate a directional price prediction for the next time period (e.g., "UP", "DOWN", "SIDEWAYS").
4.  **Signal Generation:** The prediction will be translated into a concrete trading signal (BUY, SELL, or HOLD).
5.  **Risk Management:** Before execution, the signal will be evaluated by `portfolio.py`. This module will enforce risk management rules, such as fixed fractional position sizing (e.g., risking only 1-2% of the portfolio per trade) and checking for maximum drawdown limits.
6.  **Trade Execution:** If the signal is approved by the risk management module, `trader.py` will connect to the Alpaca API and execute the trade in the paper trading account.
7.  **Logging & Monitoring:** The entire process, from data ingestion to trade execution, will be logged for performance analysis and debugging.

## 5. Technology Stack

The MVP will be built using Python and a curated set of libraries to ensure a balance of performance and development speed.

*   **Programming Language:** Python 3.10+
*   **Trading API:** Alpaca
*   **Machine Learning:** PyTorch, PyTorch Forecasting
*   **Data Manipulation:** Pandas, NumPy
*   **Technical Analysis:** TA-Lib
*   **Deployment:** Docker
*   **Environment Variables:** `python-dotenv`

## 6. Machine Learning Model: Temporal Fusion Transformer (TFT)

For the MVP, we will implement a **Temporal Fusion Transformer (TFT)**. This state-of-the-art model is specifically designed for multi-horizon time-series forecasting and offers significant advantages over simpler models like LSTMs.

*   **Architecture:** The TFT uses a self-attention mechanism to learn temporal patterns over long and complex time series. Its architecture is inherently multi-modal, allowing it to incorporate different types of inputs seamlessly. The initial inputs will be:
    1.  **Static Features:** Metadata that does not change over time (e.g., asset ID).
    2.  **Known Future Inputs:** Data that is known in advance (e.g., day of the week, month).
    3.  **Observed Inputs:** The core time-series data (OHLCV and technical indicators).
*   **Training:** The model will be trained on the historical data in the `/data` directory using the `PyTorch Forecasting` library, which provides a high-level API for training TFT models.
*   **Interpretability:** A key advantage of the TFT is its built-in interpretability. The model can expose the importance of different features and attention patterns, providing valuable insights into what drives its predictions.

## 7. Deployment and Operations

The goal is a cheap and easy-to-manage deployment.

*   **Containerization:** The entire application will be containerized using **Docker**. This ensures that the trading environment is consistent and portable.
*   **Hosting:** The Docker container can be deployed on any low-cost Virtual Private Server (VPS), such as a **DigitalOcean Droplet** or an **AWS EC2 t2.micro instance**. A basic server should be sufficient for the MVP's needs and will keep costs low.
*   **Automation:** The `main.py` script will be the entry point for the application, running a continuous loop to fetch data, generate predictions, and place trades at a defined interval (e.g., every hour).
*   **Secrets Management:** API keys and other sensitive information will be stored in a `.env` file. This file will not be committed to version control and will be securely passed to the Docker container at runtime.

## 8. Next Steps (Post-MVP)

This MVP provides a solid foundation. Future iterations will focus on:

*   **Advanced Models:** Implementing more complex models like hybrid TFTs or Reinforcement Learning agents.
*   **More Data Sources:** Integrating on-chain data (e.g., from Glassnode) and sentiment data (e.g., from Twitter).
*   **Live Trading:** After proving consistent profitability in the paper trading environment, we will transition to live trading.
*   **Robust Backtesting:** Implementing a more sophisticated event-driven backtesting engine to more accurately simulate historical performance.
