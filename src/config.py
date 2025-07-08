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
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.02")) # 2% stop-loss for non-ATR mode
TIME_FRAME = os.getenv("TIME_FRAME", "1Hour") # Data aggregation time frame
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "10000.0"))

# --- Risk Management ---
USE_ATR_STOP_LOSS = os.getenv("USE_ATR_STOP_LOSS", "True").lower() in ('true', '1', 't')
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "2.0")) # Multiplier for ATR-based stop-loss


# --- Model & Training Parameters ---
MODEL_PATH = os.getenv("MODEL_PATH", "tft_model.pth")
TFT_MAX_ENCODER_LENGTH = int(os.getenv("TFT_MAX_ENCODER_LENGTH", "72")) # Use 3 days of data
TFT_MAX_PREDICTION_LENGTH = int(os.getenv("TFT_MAX_PREDICTION_LENGTH", "12")) # Predict 12 hours ahead
TRAINING_EPOCHS = int(os.getenv("TRAINING_EPOCHS", "50")) # Increased from 10 to 50

# --- TFT Hyperparameters ---
TFT_HIDDEN_SIZE = int(os.getenv("TFT_HIDDEN_SIZE", "128")) # Increased from 32 to 128 for crypto complexity
TFT_HIDDEN_CONTINUOUS_SIZE = int(os.getenv("TFT_HIDDEN_CONTINUOUS_SIZE", "32")) # Increased from 8 to 32
TFT_LSTM_LAYERS = int(os.getenv("TFT_LSTM_LAYERS", "2")) # Keep at 2 (optimal for most cases)
TFT_ATTENTION_HEADS = int(os.getenv("TFT_ATTENTION_HEADS", "2")) # Reduced from 4 to 2 for crypto short-term patterns
TFT_DROPOUT = float(os.getenv("TFT_DROPOUT", "0.3")) # Increased from 0.2 to 0.3 for crypto volatility
TFT_LEARNING_RATE = float(os.getenv("TFT_LEARNING_RATE", "3e-4")) # Reduced from 1e-3 to 3e-4 for stability

# --- Training Hyperparameters ---
TFT_BATCH_SIZE = int(os.getenv("TFT_BATCH_SIZE", "64")) # Reduced from 128 to 64 for M1 Pro memory efficiency
TFT_REDUCE_ON_PLATEAU_PATIENCE = int(os.getenv("TFT_REDUCE_ON_PLATEAU_PATIENCE", "6")) # Increased from 4 to 6
TFT_GRADIENT_CLIP_VAL = float(os.getenv("TFT_GRADIENT_CLIP_VAL", "0.5")) # Increased from 0.1 to 0.5

# --- Early Stopping Hyperparameters ---
TFT_EARLY_STOP_MONITOR = os.getenv("TFT_EARLY_STOP_MONITOR", "val_loss")
TFT_EARLY_STOP_MIN_DELTA = float(os.getenv("TFT_EARLY_STOP_MIN_DELTA", "1e-5")) # More sensitive
TFT_EARLY_STOP_PATIENCE = int(os.getenv("TFT_EARLY_STOP_PATIENCE", "8")) # Increased from 5 to 8
TFT_EARLY_STOP_MODE = os.getenv("TFT_EARLY_STOP_MODE", "min")

# --- Training Infrastructure ---
TFT_ACCELERATOR = os.getenv("TFT_ACCELERATOR", "mps")  # Changed from "cpu" to "mps" for Apple Silicon
TFT_DEVICES = int(os.getenv("TFT_DEVICES", "1"))
TFT_NUM_WORKERS = int(os.getenv("TFT_NUM_WORKERS", "4")) # Increased from 0 to 4 for M1 Pro's 10 cores

# --- Data Paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")

# --- Signal Thresholds ---
STRONG_BUY_THRESHOLD = float(os.getenv("STRONG_BUY_THRESHOLD", "0.02"))
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.005"))
STRONG_SELL_THRESHOLD = float(os.getenv("STRONG_SELL_THRESHOLD", "-0.02"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "-0.005"))
