# Algorithmic Cryptocurrency Trading Bot

This project is an algorithmic trading bot that uses a sophisticated deep learning model to trade cryptocurrencies on the Alpaca paper trading platform. The core of the bot is a Temporal Fusion Transformer (TFT) model, which is a state-of-the-art architecture for time-series forecasting.

## Features

- **Advanced Forecasting Model**: Utilizes a Temporal Fusion Transformer (TFT) for multi-horizon time-series forecasting, implemented with `pytorch-forecasting`.
- **Automated Trading**: Connects to the Alpaca API to execute trades in a paper trading environment.
- **Modular Architecture**: The code is organized into distinct modules for data ingestion, feature engineering, model training, prediction, risk management, and trading.
- **Configurable Parameters**: Key parameters, such as the trading symbol, risk management rules, and model hyperparameters, are easily configurable through a `.env` file.
- **Robust Risk Management**: Implements a fixed-fractional position sizing strategy with a stop-loss mechanism to protect capital.
- **Comprehensive Backtesting**: Includes an event-driven backtesting framework to rigorously test and evaluate trading strategies on historical data.

## How It Works

The bot operates in different modes: training, backtesting, and live trading.

### Training
The model is trained on historical data to learn patterns and relationships for forecasting.

### Backtesting
The backtesting framework simulates the trading strategy on historical data, allowing for performance evaluation and optimization without risking real capital. It accounts for commissions and slippage to provide a realistic assessment.

### Live Trading
In live trading mode, the bot operates in a continuous loop, performing the following steps at a regular interval (e.g., every hour):

1.  **Data Ingestion**: Fetches the latest market data for the specified cryptocurrency pair from the Alpaca API.
2.  **Feature Engineering**: Generates a rich set of technical indicators and time-based features from the raw market data.
3.  **Prediction**: Uses the pre-trained TFT model to forecast the price movement over the next `TFT_MAX_PREDICTION_LENGTH` hours.
4.  **Signal Generation**: Translates the model's prediction into a trading signal (`STRONG_BUY`, `BUY`, `SELL`, `STRONG_SELL`, or `HOLD`).
5.  **Risk Management**: Checks for stop-loss triggers and ensures that any new trades adhere to the defined risk parameters.
6.  **Trade Execution**: Places market orders on the Alpaca paper trading platform based on the generated signals.

## Getting Started

### Prerequisites

- Python 3.10+
- An Alpaca paper trading account

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment and activate it:**

    ```bash
    python -m venv env
    source env/bin/activate # On Windows, use `env\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the root directory of the project.

2.  **Add your Alpaca API keys** and other configuration variables to the `.env` file. Below are the key parameters you can configure:

    ```
    ALPACA_API_KEY="YOUR_API_KEY"
    ALPACA_SECRET_KEY="YOUR_SECRET_KEY"
    ALPACA_BASE_URL="https://paper-api.alpaca.markets" # Optional: Defaults to paper trading URL

    # --- Trading Parameters ---
    SYMBOL="ETH/USD" # Cryptocurrency pair to trade (e.g., "ETH/USD", "BTC/USD")
    RISK_PER_TRADE="0.01" # Risk 1% of capital per trade (e.01 for 1%)
    STOP_LOSS_PCT="0.02"  # 2% stop-loss (e.g., 0.02 for 2%)
    TIME_FRAME="1Hour" # Data aggregation time frame (e.g., "1Hour", "1Day")
    INITIAL_CAPITAL="10000.0" # Starting capital for the portfolio manager

    # --- Model & Training Parameters ---
    MODEL_PATH="tft_model.pth" # Path to save/load the trained model
    TFT_MAX_ENCODER_LENGTH="72" # Number of past time steps to consider for prediction (e.g., 72 hours = 3 days)
    TFT_MAX_PREDICTION_LENGTH="12" # Number of future time steps to predict (e.g., 12 hours)
    TRAINING_EPOCHS="10" # Number of training epochs for the TFT model

    # --- Optional TFT Hyperparameters (Advanced) ---
    TFT_HIDDEN_SIZE="32"
    TFT_LSTM_LAYERS="2"
    TFT_ATTENTION_HEADS="4"
    TFT_DROPOUT="0.2"
    TFT_LEARNING_RATE="0.001"
    ```

## Usage

The `main.py` script now supports different operation modes: `train`, `backtest`, and `trade`.

### 1. Train the Model

Before running backtests or live trading, you need to train the TFT model. This will download historical data, engineer features, train the model, and save it to the path specified in `MODEL_PATH` (defaulting to `tft_model.pth`).

```bash
python src/main.py --mode train
```

This process may take some time, especially on the first run as it fetches data and trains the deep learning model.

### 2. Run a Backtest

Once the model is trained, you can run a backtest to simulate the trading strategy on historical data. This will output performance statistics and generate an interactive HTML plot of the backtest results.

```bash
python src/main.py --mode backtest
```

The backtest plot will be saved as `backtest_plot.html` in the root directory of your project.

### 3. Run the Live Trading Bot

After you are satisfied with the backtest results, you can run the live trading bot. This will load the pre-trained model and begin the continuous trading loop, connecting to the Alpaca paper trading platform.

```bash
python src/main.py --mode trade
```

**Important Note:** Ensure your Alpaca API keys (`ALPACA_API_KEY` and `ALPACA_SECRET_KEY`) are set in your `.env` file, as the `DataLoader` requires them even when loading local data for training or backtesting.

## Deployment with Docker

This application is designed to be deployed using Docker, which ensures a consistent and isolated environment.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed on your machine.

### Building the Docker Image

From the project's root directory, run the following command to build the Docker image:

```bash
docker build -t algo-trading-bot .
```

### Running the Docker Container

To run the trading bot inside a Docker container, you need to pass your environment variables (from the `.env` file) to it.

1.  **Ensure your `.env` file is created** in the root of the project as described in the "Configuration" section.

2.  **Run the Docker container** using the `--env-file` flag. By default, the Docker container will run in `trade` mode. You can specify a different mode using the `--mode` argument after the image name.

    ```bash
    docker run --env-file ./.env -d --name trading-bot algo-trading-bot
    # To run in backtest mode within Docker:
    # docker run --env-file ./.env -d --name trading-bot-backtest algo-trading-bot --mode backtest
    ```

    - `--env-file ./.env`: Passes the environment variables from your `.env` file to the container.
    - `-d`: Runs the container in detached mode (in the background).
    - `--name trading-bot`: Assigns a name to the container for easier management.

### Managing the Container

-   **View logs:** To see the bot's output and monitor its activity, run:
    ```bash
    docker logs -f trading-bot
    ```

-   **Stop the container:**
    ```bash
    docker stop trading-bot
    ```

-   **Restart the container:**
    ```bash
    docker start trading-bot
    ```

## Technology Stack

- **Python**: The core programming language.
- **PyTorch & PyTorch Lightning**: For building and training the TFT model.
- **PyTorch Forecasting**: A high-level library for time-series forecasting with PyTorch.
- **Pandas & NumPy**: For data manipulation and numerical operations.
- **TA-Lib**: For generating technical analysis indicators (optional, some features will be unavailable if not installed).
- **Alpaca-py**: The official Python SDK for the Alpaca API.
- **python-dotenv**: For managing environment variables.
- **Backtesting.py**: For event-driven backtesting and strategy evaluation.
- **Bokeh**: For interactive plotting of backtest results.

## Disclaimer

This trading bot is for educational and research purposes only. Algorithmic trading involves significant risk, and you should not use this bot for live trading with real money without fully understanding the risks involved. The authors are not responsible for any financial losses incurred.