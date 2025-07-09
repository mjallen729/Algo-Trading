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
MODEL_PATH = "tft_model.pth"
TFT_MAX_ENCODER_LENGTH = 72  # 3 days of hourly data
TFT_MAX_PREDICTION_LENGTH = 12  # 12 hours ahead
TRAINING_EPOCHS = 50  # Research-optimized for proper convergence

# --- TFT Hyperparameters (Research-Optimized for Crypto + M1 Pro) ---
TFT_HIDDEN_SIZE = 128  # 4x increase for crypto complexity
TFT_HIDDEN_CONTINUOUS_SIZE = 32  # 4x increase for rich features
TFT_LSTM_LAYERS = 2  # Optimal for most cases
TFT_ATTENTION_HEADS = 4  # Standard for TFT
TFT_DROPOUT = 0.3  # Higher dropout for crypto noise
TFT_LEARNING_RATE = 3e-4  # Conservative LR for stability

# --- Training Hyperparameters (M1 Pro Optimized) ---
TFT_BATCH_SIZE = 64  # Memory-efficient for MPS
TFT_REDUCE_ON_PLATEAU_PATIENCE = 6  # Stable value
TFT_GRADIENT_CLIP_VAL = 0.5  # Gradient stability

# --- Early Stopping Hyperparameters (Research-Optimized for Crypto) ---
TFT_EARLY_STOP_MONITOR = "val_loss"  # Always monitor val_loss
TFT_EARLY_STOP_MIN_DELTA = 0.01  # Research-optimized for crypto
TFT_EARLY_STOP_PATIENCE = 15  # Crypto-appropriate patience
TFT_EARLY_STOP_MODE = "min"  # Always minimize loss

# --- Training Infrastructure ---
TFT_ACCELERATOR = os.getenv("TFT_ACCELERATOR", "mps")  # Hardware dependent - keep .getenv()
TFT_DEVICES = 1  # Single device
TFT_NUM_WORKERS = 4  # M1 Pro optimized

# --- Data Paths ---
DATA_DIR = os.getenv("DATA_DIR", "data")

# --- Signal Thresholds ---
STRONG_BUY_THRESHOLD = float(os.getenv("STRONG_BUY_THRESHOLD", "0.02"))
BUY_THRESHOLD = float(os.getenv("BUY_THRESHOLD", "0.005"))
STRONG_SELL_THRESHOLD = float(os.getenv("STRONG_SELL_THRESHOLD", "-0.02"))
SELL_THRESHOLD = float(os.getenv("SELL_THRESHOLD", "-0.005"))
