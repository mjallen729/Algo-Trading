import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca API ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# --- Trading Parameters ---
SYMBOL = os.getenv("SYMBOL", "ETH/USD") # Default to ETH/USD
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01")) # 1% risk per trade
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02")) # 2% stop-loss
TIME_FRAME = os.getenv("TIME_FRAME", "1Hour") # Data aggregation time frame
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000.0"))

# --- Model & Training Parameters ---
MODEL_PATH = os.getenv("MODEL_PATH", "tft_model.pth")
TFT_MAX_ENCODER_LENGTH = int(os.getenv("TFT_MAX_ENCODER_LENGTH", "72")) # Use 3 days of data
TFT_MAX_PREDICTION_LENGTH = int(os.getenv("TFT_MAX_PREDICTION_LENGTH", "12")) # Predict 12 hours ahead
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "10")) # Number of training epochs

# --- TFT Hyperparameters ---
TFT_HIDDEN_SIZE = int(os.getenv("TFT_HIDDEN_SIZE", "32"))
TFT_LSTM_LAYERS = int(os.getenv("TFT_LSTM_LAYERS", "2"))
TFT_ATTENTION_HEADS = int(os.getenv("TFT_ATTENTION_HEADS", "4"))
TFT_DROPOUT = float(os.getenv("TFT_DROPOUT", "0.2"))
TFT_LEARNING_RATE = float(os.getenv("TFT_LEARNING_RATE", "1e-3"))

# --- Data Paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")
