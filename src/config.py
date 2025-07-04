import os
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

# Trading parameters
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01")) # 1% risk per trade
TIME_FRAME = os.getenv("TIME_FRAME", "1Hour") # Data aggregation time frame

# Model parameters
TFT_MAX_ENCODER_LENGTH = int(os.getenv("TFT_MAX_ENCODER_LENGTH", "24"))
TFT_MAX_PREDICTION_LENGTH = int(os.getenv("TFT_MAX_PREDICTION_LENGTH", "1"))

# Data paths
DATA_DIR = os.getenv("DATA_DIR", "data")
