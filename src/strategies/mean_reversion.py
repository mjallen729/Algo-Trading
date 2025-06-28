"""
Mean reversion trading strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base import BaseStrategy, TradingSignal, SignalType
from utils import get_logger, config

logger = get_logger(__name__)


class MeanReversionStrategy(BaseStrategy):
  """
  Mean reversion trading strategy for range-bound markets.

  This strategy identifies oversold/overbought conditions and trades
  on the expectation that prices will revert to their mean:
  - Bollinger Bands for mean reversion signals
  - RSI for overbought/oversold conditions
  - Price deviation from moving averages
  - Statistical measures of price extremes
  """

  def __init__(self, config=None):
    """
    Initialize mean reversion strategy.

    Args:
        config: Configuration object or dict with strategy parameters
    """
    # Extract strategy config if config object is passed
    if hasattr(config, 'get'):
      # Config object passed
      strategy_config = {
          'bollinger_std': config.get('strategies.mean_reversion.bollinger_std', 2.0),
          'rsi_oversold': config.get('strategies.mean_reversion.rsi_oversold', 30),
          'rsi_overbought': config.get('strategies.mean_reversion.rsi_overbought', 70),
      }
    elif isinstance(config, dict):
      # Dict passed directly
      strategy_config = config
    else:
      # No config or None passed
      strategy_config = {}

    default_config = {
        'bollinger_std': strategy_config.get('bollinger_std', 2.0),
        'rsi_oversold': strategy_config.get('rsi_oversold', 30),
        'rsi_overbought': strategy_config.get('rsi_overbought', 70),
        'mean_reversion_period': 20,
        'z_score_threshold': 2.0,
        'min_confidence': 0.5,
        'max_hold_periods': 10
    }

    super().__init__("MeanReversion", default_config)

    self.bollinger_std = self.config['bollinger_std']
    self.rsi_oversold = self.config['rsi_oversold']
    self.rsi_overbought = self.config['rsi_overbought']
    self.z_score_threshold = self.config['z_score_threshold']

    logger.info(
      f"MeanReversionStrategy initialized with Bollinger std: {self.bollinger_std}")

  def generate_signal(self,
                      data: pd.DataFrame,
                      current_price: float,
                      regime: str = None,
                      ml_prediction: float = None,
                      ml_confidence: float = None,
                      **kwargs) -> TradingSignal:
    """
    Generate mean reversion trading signal.

    Args:
        data: Historical market data (raw OHLCV)
        current_price: Current asset price
        regime: Current market regime
        ml_prediction: ML model price prediction
        ml_confidence: ML model confidence score

    Returns:
        TradingSignal object
    """
    if len(data) < 50:
      return TradingSignal(SignalType.HOLD, 0.0, current_price)

    # Calculate indicators
    indicators = self.calculate_indicators(data)
    
    # Check if indicators were calculated successfully
    if indicators.empty or 'RSI' not in indicators.columns:
      logger.warning("Failed to calculate technical indicators")
      return TradingSignal(SignalType.HOLD, 0.0, current_price)

    # Calculate mean reversion specific indicators
    indicators = self._add_mean_reversion_indicators(indicators)

    latest = indicators.iloc[-1]
    
    # Check for NaN values in critical indicators
    if pd.isna(latest['RSI']) or pd.isna(latest['BB_Upper']) or pd.isna(latest['BB_Lower']):
      logger.warning("NaN values in mean reversion indicators")
      return TradingSignal(SignalType.HOLD, 0.0, current_price)

    # Mean reversion signals
    bb_signal = self._bollinger_band_signal(latest, current_price)
    rsi_signal = self._rsi_mean_reversion_signal(latest)
    z_score_signal = self._z_score_signal(indicators, current_price)
    price_deviation_signal = self._price_deviation_signal(indicators, current_price)

    # Initialize signal
    signal_type = SignalType.HOLD
    base_confidence = 0.0

    # Use ML confidence as base if available
    if ml_confidence is not None and ml_confidence > 0:
      base_confidence = ml_confidence
    else:
      base_confidence = 0.5  # Default moderate confidence

    # Determine signal based on mean reversion logic and ML prediction
    signal_strength = (bb_signal + rsi_signal + z_score_signal + price_deviation_signal) / 4.0

    # If ML prediction available, combine with technical signals
    if ml_prediction is not None:
      # For mean reversion, we often trade against the trend
      # If ML predicts up but we're overbought, sell signal
      # If ML predicts down but we're oversold, buy signal
      
      if signal_strength > 0.5:  # Strong buy signal from technical
        if ml_prediction < 0.6:  # ML doesn't strongly contradict
          signal_type = SignalType.BUY
          base_confidence = (base_confidence * 0.6 + signal_strength * 0.4)
          
      elif signal_strength < -0.5:  # Strong sell signal from technical
        if ml_prediction > 0.4:  # ML doesn't strongly contradict
          signal_type = SignalType.SELL
          base_confidence = (base_confidence * 0.6 + abs(signal_strength) * 0.4)

    else:
      # Traditional mean reversion logic without ML
      if signal_strength > 0.7:
        signal_type = SignalType.BUY
        base_confidence = min(0.9, signal_strength)
      elif signal_strength < -0.7:
        signal_type = SignalType.SELL
        base_confidence = min(0.9, abs(signal_strength))

    # Regime adjustments
    if regime:
      base_confidence = self._adjust_confidence_for_regime(
        base_confidence, signal_type, regime)

    # Ensure minimum confidence threshold
    if base_confidence < self.config.get('min_confidence', 0.1):
      signal_type = SignalType.HOLD
      base_confidence = 0.0

    # Create signal with metadata
    metadata = {
        'bb_signal': bb_signal,
        'rsi_signal': rsi_signal,
        'z_score_signal': z_score_signal,
        'price_deviation_signal': price_deviation_signal,
        'signal_strength': signal_strength,
        'rsi': latest['RSI'],
        'bb_position': (current_price - latest['BB_Lower']) / (latest['BB_Upper'] - latest['BB_Lower']),
        'regime': regime,
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence,
        'strategy': 'mean_reversion'
    }

    signal = TradingSignal(signal_type, base_confidence, current_price, metadata=metadata)

    if self.validate_signal(signal):
      self.update_performance_metrics(signal)

    return signal

    # Determine signal type and confidence
    signal_type = SignalType.HOLD
    confidence = 0.0

    # Oversold conditions (BUY signal)
    oversold_score = self._calculate_oversold_score(
        bb_signal, rsi_signal, z_score_signal, price_deviation_signal
    )

    # Overbought conditions (SELL signal)
    overbought_score = self._calculate_overbought_score(
        bb_signal, rsi_signal, z_score_signal, price_deviation_signal
    )

    if oversold_score > 0.6:
      signal_type = SignalType.BUY
      confidence = min(0.95, oversold_score)
    elif overbought_score > 0.6:
      signal_type = SignalType.SELL
      confidence = min(0.95, overbought_score)

    # Regime-based adjustments
    if regime:
      confidence = self._adjust_confidence_for_regime(
        confidence, signal_type, regime)

    # Create signal with metadata
    metadata = {
        'bb_signal': bb_signal,
        'rsi_signal': rsi_signal,
        'z_score_signal': z_score_signal,
        'price_deviation_signal': price_deviation_signal,
        'oversold_score': oversold_score,
        'overbought_score': overbought_score,
        'rsi': latest['RSI'],
        'bb_position': latest['BB_Position'],
        'regime': regime
    }

    signal = TradingSignal(signal_type, confidence,
                           current_price, metadata=metadata)

    if self.validate_signal(signal):
      self.update_performance_metrics(signal)

    return signal

  def _add_mean_reversion_indicators(self, indicators: pd.DataFrame) -> pd.DataFrame:
    """Add mean reversion specific indicators."""
    try:
      # Check for required base columns
      required_cols = ['Close', 'BB_Upper', 'BB_Lower', 'EMA_12', 'EMA_26']
      for col in required_cols:
        if col not in indicators.columns:
          logger.warning(f"Missing column for mean reversion indicators: {col}")
          return indicators
      
      # Z-score of price relative to moving average
      period = self.config.get('mean_reversion_period', 20)
      indicators['Price_MA'] = indicators['Close'].rolling(period).mean()
      indicators['Price_Std'] = indicators['Close'].rolling(period).std()
      
      # Avoid division by zero
      std_mask = indicators['Price_Std'] != 0
      indicators['Z_Score'] = 0.0
      indicators.loc[std_mask, 'Z_Score'] = (
        (indicators.loc[std_mask, 'Close'] - indicators.loc[std_mask, 'Price_MA']) / 
        indicators.loc[std_mask, 'Price_Std']
      )

      # Bollinger Band position (0 = lower band, 1 = upper band)
      bb_range = indicators['BB_Upper'] - indicators['BB_Lower']
      range_mask = bb_range != 0
      indicators['BB_Position'] = 0.5  # Default to middle
      indicators.loc[range_mask, 'BB_Position'] = (
        (indicators.loc[range_mask, 'Close'] - indicators.loc[range_mask, 'BB_Lower']) / 
        bb_range.loc[range_mask]
      )

      # Percentage price oscillator
      ema26_mask = indicators['EMA_26'] != 0
      indicators['PPO'] = 0.0
      indicators.loc[ema26_mask, 'PPO'] = (
        (indicators.loc[ema26_mask, 'EMA_12'] - indicators.loc[ema26_mask, 'EMA_26']) / 
        indicators.loc[ema26_mask, 'EMA_26']
      ) * 100

      # Williams %R
      indicators['Williams_R'] = self._calculate_williams_r(indicators)

      # Stochastic oscillator
      indicators['Stoch_K'], indicators['Stoch_D'] = self._calculate_stochastic(indicators)

      return indicators
      
    except Exception as e:
      logger.warning(f"Error adding mean reversion indicators: {e}")
      return indicators

  def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Williams %R indicator."""
    highest_high = data['High'].rolling(period).max()
    lowest_low = data['Low'].rolling(period).min()
    williams_r = -100 * \
        (highest_high - data['Close']) / (highest_high - lowest_low)
    return williams_r

  def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple:
    """Calculate Stochastic oscillator."""
    lowest_low = data['Low'].rolling(k_period).min()
    highest_high = data['High'].rolling(k_period).max()

    k_percent = 100 * (data['Close'] - lowest_low) / \
        (highest_high - lowest_low)
    d_percent = k_percent.rolling(d_period).mean()

    return k_percent, d_percent

  def _bollinger_band_signal(self, latest: pd.Series, current_price: float) -> float:
    """Calculate Bollinger Band mean reversion signal."""
    bb_position = latest['BB_Position']

    # Strong oversold signal
    if bb_position < 0.1:  # Below lower band
      return min(1.0, (0.1 - bb_position) * 10)  # Scale signal strength

    # Strong overbought signal
    elif bb_position > 0.9:  # Above upper band
      return max(-1.0, (0.9 - bb_position) * 10)  # Scale signal strength

    # Weak signals near bands
    elif bb_position < 0.2:
      return (0.2 - bb_position) * 2.5
    elif bb_position > 0.8:
      return (0.8 - bb_position) * 2.5

    return 0.0

  def _rsi_mean_reversion_signal(self, latest: pd.Series) -> float:
    """Calculate RSI mean reversion signal."""
    rsi = latest['RSI']

    if rsi < self.rsi_oversold:
      # Stronger signal for more extreme RSI values
      return min(1.0, (self.rsi_oversold - rsi) / 20)
    elif rsi > self.rsi_overbought:
      # Stronger signal for more extreme RSI values
      return max(-1.0, (self.rsi_overbought - rsi) / 20)

    return 0.0

  def _z_score_signal(self, indicators: pd.DataFrame, current_price: float) -> float:
    """Calculate Z-score mean reversion signal."""
    latest_z = indicators['Z_Score'].iloc[-1]

    if latest_z < -self.z_score_threshold:
      # Price significantly below mean
      return min(1.0, abs(latest_z) / 3)
    elif latest_z > self.z_score_threshold:
      # Price significantly above mean
      return max(-1.0, -latest_z / 3)

    return 0.0

  def _price_deviation_signal(self, indicators: pd.DataFrame, current_price: float) -> float:
    """Calculate price deviation from multiple moving averages."""
    latest = indicators.iloc[-1]

    # Calculate deviations from different MAs
    deviations = []
    mas = ['SMA_10', 'SMA_20', 'SMA_50']

    for ma in mas:
      if pd.notna(latest[ma]):
        deviation = (current_price - latest[ma]) / latest[ma]
        deviations.append(deviation)

    if not deviations:
      return 0.0

    avg_deviation = np.mean(deviations)

    # Strong negative deviation suggests oversold
    if avg_deviation < -0.05:  # 5% below MAs
      return min(1.0, abs(avg_deviation) * 10)
    # Strong positive deviation suggests overbought
    elif avg_deviation > 0.05:  # 5% above MAs
      return max(-1.0, -avg_deviation * 10)

    return 0.0

  def _calculate_oversold_score(self, bb_signal: float, rsi_signal: float,
                                z_score_signal: float, price_dev_signal: float) -> float:
    """Calculate overall oversold score."""
    # Only consider positive signals for oversold
    signals = [max(0, s) for s in [bb_signal, rsi_signal,
                                   z_score_signal, price_dev_signal]]

    if not any(signals):
      return 0.0

    # Weighted combination
    weights = [0.3, 0.3, 0.25, 0.15]  # BB, RSI, Z-score, Price deviation
    oversold_score = sum(w * s for w, s in zip(weights, signals))

    return min(1.0, oversold_score)

  def _calculate_overbought_score(self, bb_signal: float, rsi_signal: float,
                                  z_score_signal: float, price_dev_signal: float) -> float:
    """Calculate overall overbought score."""
    # Only consider negative signals for overbought
    signals = [abs(min(0, s)) for s in [bb_signal, rsi_signal,
                                        z_score_signal, price_dev_signal]]

    if not any(signals):
      return 0.0

    # Weighted combination
    weights = [0.3, 0.3, 0.25, 0.15]  # BB, RSI, Z-score, Price deviation
    overbought_score = sum(w * s for w, s in zip(weights, signals))

    return min(1.0, overbought_score)

  def _adjust_confidence_for_regime(self, confidence: float,
                                    signal_type: SignalType, regime: str) -> float:
    """Adjust confidence based on market regime."""
    # Mean reversion works best in ranging markets
    if "Ranging" in regime or "Sideways" in regime:
      # Boost confidence in ranging markets
      confidence = min(0.95, confidence * 1.3)

    elif "Trending" in regime:
      # Reduce confidence in trending markets
      confidence = confidence * 0.6

      # But allow counter-trend signals in extreme conditions
      if confidence > 0.8:  # Very strong mean reversion signal
        confidence = min(0.85, confidence * 1.1)

    return max(0.0, confidence)

  def check_exit_conditions(self, indicators: pd.DataFrame,
                            entry_price: float, signal_type: SignalType) -> bool:
    """
    Check if mean reversion position should be closed.

    Args:
        indicators: Current market indicators
        entry_price: Price at entry
        signal_type: Type of position (BUY/SELL)

    Returns:
        True if position should be closed
    """
    latest = indicators.iloc[-1]
    current_price = latest['Close']

    # Target hit (price reverted to mean)
    if signal_type == SignalType.BUY:
      # Exit if price reached middle/upper Bollinger Band
      if latest['BB_Position'] > 0.6:
        return True
      # Exit if RSI becomes overbought
      if latest['RSI'] > 65:
        return True

    elif signal_type == SignalType.SELL:
      # Exit if price reached middle/lower Bollinger Band
      if latest['BB_Position'] < 0.4:
        return True
      # Exit if RSI becomes oversold
      if latest['RSI'] < 35:
        return True

    # Stop loss (trend continuation against mean reversion)
    price_change = (current_price - entry_price) / entry_price

    if signal_type == SignalType.BUY and price_change < -0.03:  # 3% stop loss
      return True
    elif signal_type == SignalType.SELL and price_change > 0.03:  # 3% stop loss
      return True

    return False
