import pandas as pd
import torch
from src.models.tft_model import TFTModel


class Predictor:
  def __init__(self, model: TFTModel):
    self.model = model

  def generate_signal(self, data: pd.DataFrame) -> tuple[str, float]:
    """
    Generates a trading signal based on the model's prediction.

    Args:
      data: The input DataFrame for prediction.

    Returns:
      A tuple containing the signal (str) and the predicted price (float).
    """
    # The model returns quantile predictions. Shape: (batch_size, prediction_length, num_quantiles)
    raw_predictions = self.model.predict(data)

    # We are predicting for a horizon, let's take the median prediction at the end of the horizon
    # Default quantiles: [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
    # Median (0.5 quantile) is the 4th value (index 3)
    predicted_price = raw_predictions[0, -1, 3].item()

    last_close_price = data['close'].iloc[-1]
    predicted_change_pct = (
      predicted_price - last_close_price) / last_close_price

    # Define thresholds for signals (can be moved to config)
    STRONG_BUY_THRESHOLD = 0.02
    BUY_THRESHOLD = 0.005
    STRONG_SELL_THRESHOLD = -0.02
    SELL_THRESHOLD = -0.005

    signal = "HOLD"
    if predicted_change_pct > STRONG_BUY_THRESHOLD:
      signal = "STRONG_BUY"
    elif predicted_change_pct > BUY_THRESHOLD:
      signal = "BUY"
    elif predicted_change_pct < STRONG_SELL_THRESHOLD:
      signal = "STRONG_SELL"
    elif predicted_change_pct < SELL_THRESHOLD:
      signal = "SELL"

    return signal, predicted_price
