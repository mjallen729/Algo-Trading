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

## Developer Documentation

This section provides more in-depth information for developers looking to understand, modify, or extend the project.

### Module Overview

The `src/` directory contains the core logic of the trading bot, organized into several modules:

-   `src/main.py`: The main entry point of the application. It parses command-line arguments to determine the operation mode (`train`, `backtest`, or `trade`) and orchestrates the execution flow by calling functions from other modules.
-   `src/config.py`: Centralized configuration file that loads environment variables from `.env` and defines various parameters for Alpaca API, trading, model training, and TFT hyperparameters.
-   `src/data_ingestion/data_loader.py`: Responsible for fetching historical cryptocurrency data. It can load data from Alpaca's API or from local CSV files for training and backtesting purposes.
-   `src/feature_engineering/features.py`: Handles the creation of various technical indicators (e.g., SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV), lagged features, and volatility measures from raw OHLCV data. It optionally uses the `TA-Lib` library.
-   `src/models/tft_model.py`: Implements the Temporal Fusion Transformer (TFT) model using `pytorch-forecasting`. It manages data preparation for the TFT (including time-based features and normalization), model building, training, prediction, and saving/loading model weights.
-   `src/models/predict.py`: Utilizes a trained `TFTModel` to generate trading signals (`STRONG_BUY`, `BUY`, `SELL`, `STRONG_SELL`, `HOLD`) based on the model's predicted future price movements. It translates quantile predictions into actionable signals.
-   `src/risk_management/portfolio.py`: Manages the trading portfolio's state, including current capital, open positions, and trade history. It implements fixed-fractional position sizing, calculates trade quantities, and performs stop-loss checks.
-   `src/trading/trader.py`: Interfaces with the Alpaca Trading API to place market orders (buy/sell), retrieve account information, and manage open positions in the paper trading environment.
-   `src/backtesting/run_backtest.py`: Orchestrates the backtesting process using the `backtesting.py` library. It loads historical data, initializes the `TFTStrategy`, and runs the backtest, outputting performance statistics and an interactive plot.
-   `src/backtesting/tft_strategy.py`: Defines the trading strategy for the `backtesting.py` framework. It integrates the `TFTModel` and `Predictor` to generate signals within the backtest environment and executes simulated trades, including a basic stop-loss mechanism.

### Design Decisions and Rationale

-   **Modular Architecture**: The project is structured into distinct modules (e.g., `data_ingestion`, `feature_engineering`, `models`, `risk_management`, `trading`) to promote code organization, reusability, and maintainability. This allows for easier independent development and testing of components.
-   **Temporal Fusion Transformer (TFT)**: Chosen for its state-of-the-art performance in multi-horizon time-series forecasting and its built-in interpretability. TFT's ability to handle various input types (static, known future, observed) makes it suitable for complex financial data.
-   **`backtesting.py` for Backtesting**: Selected for its simplicity, event-driven nature, and ability to generate interactive plots, providing a quick and effective way to evaluate strategy performance on historical data.
-   **Fixed-Fractional Position Sizing**: Implemented in `portfolio.py` to ensure consistent risk management by risking a fixed percentage of capital per trade, which helps in capital preservation.
-   **Environment Variables for Configuration**: Utilizing `.env` files and `python-dotenv` ensures that sensitive API keys and configurable parameters are kept separate from the codebase, enhancing security and ease of deployment across different environments.
-   **Docker for Deployment**: Containerization with Docker provides a consistent and isolated environment, simplifying deployment and ensuring that the application runs reliably across different machines.

### Technical Deep Dive

#### Temporal Fusion Transformer (TFT)

The TFT model is implemented using `pytorch-forecasting`. Key aspects include:
-   **Input Features**: The model is configured to use `time_idx` for temporal ordering, `symbol` as a group ID, and various time-based categorical features (`month`, `day`, `weekday`, `hour`). Real-valued features include OHLCV data and all engineered technical indicators.
-   **Normalization**: `GroupNormalizer` with `softplus` transformation is applied to the target variable (`close`) to handle potential non-stationarity and improve model stability.
-   **Training**: The model is trained with `QuantileLoss` and the `Ranger` optimizer, which are common choices for robust time-series forecasting. Early stopping is used to prevent overfitting.

#### Feature Engineering

The `FeatureEngineer` class dynamically adds features. If `TA-Lib` is not installed, it gracefully skips the generation of technical indicators, ensuring the application can still run, albeit with a reduced feature set. The `dropna()` call after feature engineering is crucial to remove rows with `NaN` values introduced by rolling calculations or lagged features, ensuring clean data for the model.

#### Backtesting Framework Integration

The `TFTStrategy` class in `src/backtesting/tft_strategy.py` acts as a bridge between our core logic and the `backtesting.py` framework. It's important to note that `backtesting.py` handles its own data feeding and order execution, so the `DataLoader` and `Trader` classes are not directly used within the `Strategy`'s `init` or `next` methods. The model is loaded once in `init`, and predictions are made in the `next` method for each time step.

### Testing

Unit and integration tests are located in the `tests/` directory.
-   `tests/test_features.py`: Contains tests for the `FeatureEngineer` class to ensure that features are correctly generated.
-   `tests/test_trader.py`: Contains tests for the `Trader` class, primarily focusing on its interaction with the Alpaca API (though these might be mocked for true unit testing).

To run the tests, navigate to the project root and execute:
```bash
pytest
```
*(Note: Some tests might require mocking external API calls for reliable and fast execution.)*

### Contribution Guidelines

We welcome contributions to this project! If you'd like to contribute, please follow these steps:

1.  **Fork the repository.**
2.  **Create a new branch** for your feature or bug fix: `git checkout -b feature/your-feature-name` or `git checkout -b bugfix/issue-description`.
3.  **Make your changes**, adhering to the existing code style and conventions.
4.  **Write unit and integration tests** for your changes, ensuring adequate test coverage.
5.  **Run all tests** (`pytest`) to ensure no regressions are introduced.
6.  **Update documentation** as necessary (e.g., `README.md`, module docstrings).
7.  **Commit your changes** with a clear and concise commit message.
8.  **Push your branch** to your forked repository.
9.  **Open a Pull Request** to the `main` branch of the original repository, describing your changes in detail.

Please ensure your code is well-commented where necessary and follows Python best practices.
