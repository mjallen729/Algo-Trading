"""
Momentum-based trading strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base import BaseStrategy, TradingSignal, SignalType
from utils import get_logger, config

logger = get_logger(__name__)


class MomentumStrategy(BaseStrategy):
  """
  Momentum trading strategy that capitalizes on strong directional movements.

  This strategy identifies trends and momentum breakouts using:
  - RSI for momentum confirmation
  - MACD for trend direction
  - Volume analysis for breakout validation
  - Moving average alignment for trend strength
  """

  def __init__(self, config=None):
    """
    Initialize momentum strategy.

    Args:
        config: Configuration object or dict with strategy parameters
    """
    # Extract strategy config if config object is passed
    if hasattr(config, 'get'):
      # Config object passed
      strategy_config = {
          'rsi_threshold': config.get('strategies.momentum.rsi_threshold', 70),
          'volume_multiplier': config.get('strategies.momentum.volume_multiplier', 1.5),
          'trend_strength_min': config.get('strategies.momentum.trend_strength_min', 0.6),
      }
    elif isinstance(config, dict):
      # Dict passed directly
      strategy_config = config
    else:
      # No config or None passed
      strategy_config = {}

    default_config = {
        'rsi_threshold': strategy_config.get('rsi_threshold', 70),
        'volume_multiplier': strategy_config.get('volume_multiplier', 1.5),
        'trend_strength_min': strategy_config.get('trend_strength_min', 0.6),
        'min_confidence': 0.6,
        'lookback_period': 20,
        'breakout_threshold': 0.02  # 2% price movement for breakout
    }

    super().__init__("Momentum", default_config)

    self.rsi_threshold = self.config['rsi_threshold']
    self.volume_multiplier = self.config['volume_multiplier']
    self.trend_strength_min = self.config['trend_strength_min']

    logger.info(
      f"MomentumStrategy initialized with RSI threshold: {self.rsi_threshold}")

  def generate_signal(self,
                      data: pd.DataFrame,
                      current_price: float,
                      regime: str = None,
                      ml_prediction: float = None,
                      ml_confidence: float = None,
                      **kwargs) -> TradingSignal:
    """
    Generate momentum-based trading signal.

    Args:
        data: Historical market data (raw OHLCV)
        current_price: Current asset price
        regime: Current market regime
        ml_prediction: ML model price prediction
        ml_confidence: ML model confidence score

    Returns:
        TradingSignal object
    """
    if len(data) < 50:  # Need sufficient data for indicators
      return TradingSignal(SignalType.HOLD, 0.0, current_price)

    # Calculate technical indicators
    indicators = self.calculate_indicators(data)

    # Check if indicators were calculated successfully
    if indicators.empty or 'RSI' not in indicators.columns:
      logger.warning("Failed to calculate technical indicators")
      return TradingSignal(SignalType.HOLD, 0.0, current_price)

    # Get latest values
    latest = indicators.iloc[-1]

    # Check for NaN values in critical indicators
    if pd.isna(latest['RSI']) or pd.isna(latest['MACD']):
      logger.warning("NaN values in technical indicators")
      return TradingSignal(SignalType.HOLD, 0.0, current_price)

    # Calculate momentum signals
    momentum_score = self._calculate_momentum_score(indicators)
    trend_strength = self._calculate_trend_strength(indicators)
    volume_confirmation = self._check_volume_confirmation(indicators)
    breakout_signal = self._detect_breakout(indicators, current_price)

    # Initialize signal
    signal_type = SignalType.HOLD
    base_confidence = 0.0

    # Use ML confidence as base if available
    if ml_confidence is not None and ml_confidence > 0:
      base_confidence = ml_confidence
    else:
      base_confidence = 0.5  # Default moderate confidence

    # Determine signal direction based on ML prediction and technical analysis
    if ml_prediction is not None:
      # ML suggests price increase
      if ml_prediction > 0.5:
        # Bullish momentum conditions
        if (momentum_score > 0.3 and  # Lowered threshold
            trend_strength > 0.4 and  # Lowered threshold
            latest['RSI'] > 45 and latest['RSI'] < 85):

          signal_type = SignalType.BUY
          # Combine ML confidence with technical confidence
          tech_confidence = min(0.95, momentum_score * 0.6 + trend_strength * 0.4)
          base_confidence = (base_confidence * 0.7 + tech_confidence * 0.3)

          # Boost for strong volume and breakouts
          if volume_confirmation:
            base_confidence = min(0.95, base_confidence + 0.05)
          if breakout_signal > 0:
            base_confidence = min(0.95, base_confidence + 0.1)

      # ML suggests price decrease
      elif ml_prediction < 0.5:
        # Bearish momentum conditions
        if (momentum_score < -0.3 and  # Lowered threshold
            trend_strength > 0.4 and   # Lowered threshold
            latest['RSI'] < 55 and latest['RSI'] > 15):

          signal_type = SignalType.SELL
          # Combine ML confidence with technical confidence
          tech_confidence = min(0.95, abs(momentum_score) * 0.6 + trend_strength * 0.4)
          base_confidence = (base_confidence * 0.7 + tech_confidence * 0.3)

          # Boost for strong volume and breakouts
          if volume_confirmation:
            base_confidence = min(0.95, base_confidence + 0.05)
          if breakout_signal < 0:
            base_confidence = min(0.95, base_confidence + 0.1)

    # If no ML prediction, rely purely on technical analysis
    else:
      # Traditional momentum logic
      if (momentum_score > 0.6 and
          trend_strength > self.trend_strength_min and
          volume_confirmation and
          latest['RSI'] > 50 and latest['RSI'] < 80):

        signal_type = SignalType.BUY
        base_confidence = min(0.95, momentum_score * 0.8 + trend_strength * 0.2)

      elif (momentum_score < -0.6 and
            trend_strength > self.trend_strength_min and
            volume_confirmation and
            latest['RSI'] < 50 and latest['RSI'] > 20):

        signal_type = SignalType.SELL
        base_confidence = min(0.95, abs(momentum_score) * 0.8 + trend_strength * 0.2)

    # Regime-based adjustments
    if regime:
      base_confidence = self._adjust_confidence_for_regime(
        base_confidence, signal_type, regime)

    # Ensure minimum confidence threshold
    if base_confidence < self.config.get('min_confidence', 0.1):
      signal_type = SignalType.HOLD
      base_confidence = 0.0

    # Create signal with metadata
    metadata = {
        'momentum_score': momentum_score,
        'trend_strength': trend_strength,
        'volume_confirmation': volume_confirmation,
        'breakout_signal': breakout_signal,
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'regime': regime,
        'ml_prediction': ml_prediction,
        'ml_confidence': ml_confidence,
        'strategy': 'momentum'
    }

    signal = TradingSignal(signal_type, base_confidence,
                           current_price, metadata=metadata)

    # Update performance metrics
    if self.validate_signal(signal):
      self.update_performance_metrics(signal)

    return signal

  def _calculate_momentum_score(self, indicators: pd.DataFrame) -> float:
    """
    Calculate overall momentum score.

    Args:
        indicators: DataFrame with technical indicators

    Returns:
        Momentum score between -1 and 1
    """
    try:
      latest = indicators.iloc[-1]

      # Check for required columns and NaN values
      required_cols = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram', 'Close', 'SMA_10', 'SMA_20', 'SMA_50']
      for col in required_cols:
        if col not in indicators.columns:
          logger.warning(f"Missing column for momentum calculation: {col}")
          return 0.0
        if pd.isna(latest[col]):
          logger.warning(f"NaN value in column: {col}")
          return 0.0

      # RSI momentum component
      rsi_norm = (latest['RSI'] - 50) / 50  # Normalize RSI to -1 to 1

      # MACD momentum component
      macd_signal = 1 if latest['MACD'] > latest['MACD_Signal'] else -1
      macd_std = indicators['MACD_Histogram'].rolling(20).std().iloc[-1]
      if pd.isna(macd_std) or macd_std == 0:
        macd_momentum = macd_signal * 0.5  # Default strength
      else:
        macd_strength = abs(latest['MACD_Histogram']) / macd_std
        macd_momentum = macd_signal * min(1.0, macd_strength)

      # Price momentum (rate of change)
      if len(indicators) >= 10:
        price_change = (latest['Close'] - indicators['Close'].iloc[-10]) / indicators['Close'].iloc[-10]
        price_momentum = np.tanh(price_change * 10)  # Normalize to -1 to 1
      else:
        price_momentum = 0.0

      # Moving average momentum
      ma_momentum = 0
      if latest['SMA_10'] > latest['SMA_20'] > latest['SMA_50']:
        ma_momentum = 1
      elif latest['SMA_10'] < latest['SMA_20'] < latest['SMA_50']:
        ma_momentum = -1

      # Weighted combination
      momentum_score = (
          0.3 * rsi_norm +
          0.3 * macd_momentum +
          0.25 * price_momentum +
          0.15 * ma_momentum
      )

      return np.clip(momentum_score, -1.0, 1.0)
      
    except Exception as e:
      logger.warning(f"Error calculating momentum score: {e}")
      return 0.0

  def _calculate_trend_strength(self, indicators: pd.DataFrame) -> float:
    """
    Calculate trend strength.

    Args:
        indicators: DataFrame with technical indicators

    Returns:
        Trend strength between 0 and 1
    """
    try:
      latest = indicators.iloc[-1]
      
      # Check for required columns
      required_cols = ['Close', 'SMA_10', 'SMA_20', 'SMA_50', 'ATR']
      for col in required_cols:
        if col not in indicators.columns or pd.isna(latest[col]):
          logger.warning(f"Missing or NaN value in trend calculation: {col}")
          return 0.0

      # Moving average alignment
      mas = [latest['SMA_10'], latest['SMA_20'], latest['SMA_50']]
      if mas == sorted(mas) or mas == sorted(mas, reverse=True):
        ma_alignment = 1.0
      else:
        ma_alignment = 0.3

      # Price position relative to moving averages
      price_above_mas = sum([
          latest['Close'] > latest['SMA_10'],
          latest['Close'] > latest['SMA_20'],
          latest['Close'] > latest['SMA_50']
      ]) / 3

      price_position_strength = abs(price_above_mas - 0.5) * 2  # 0 to 1

      # ATR-based volatility (higher volatility can indicate stronger trends)
      atr_mean = indicators['ATR'].rolling(20).mean().iloc[-1]
      if pd.isna(atr_mean) or atr_mean == 0:
        volatility_strength = 0.5
      else:
        atr_ratio = latest['ATR'] / atr_mean
        volatility_strength = min(1.0, atr_ratio)

      # Combine components
      trend_strength = (
          0.4 * ma_alignment +
          0.4 * price_position_strength +
          0.2 * volatility_strength
      )

      return min(1.0, trend_strength)
      
    except Exception as e:
      logger.warning(f"Error calculating trend strength: {e}")
      return 0.0

    # ATR-based volatility (higher volatility can indicate stronger trends)
    atr_ratio = latest['ATR'] / indicators['ATR'].rolling(20).mean().iloc[-1]
    volatility_strength = min(1.0, atr_ratio)

    # Combine components
    trend_strength = (
        0.4 * ma_alignment +
        0.4 * price_position_strength +
        0.2 * volatility_strength
    )

    return min(1.0, trend_strength)

  def _check_volume_confirmation(self, indicators: pd.DataFrame) -> bool:
    """
    Check if volume confirms momentum.

    Args:
        indicators: DataFrame with technical indicators

    Returns:
        True if volume confirms momentum
    """
    try:
      if 'Volume_Ratio' not in indicators.columns:
        logger.warning("Volume_Ratio column missing")
        return False
        
      latest = indicators.iloc[-1]
      
      if pd.isna(latest['Volume_Ratio']):
        logger.warning("NaN value in Volume_Ratio")
        return False

      # Volume should be above average for momentum confirmation
      volume_condition = latest['Volume_Ratio'] > self.volume_multiplier

      # Recent volume trend (last 3 periods)
      if len(indicators) >= 3:
        recent_volume_trend = indicators['Volume_Ratio'].tail(3).mean() > 1.0
      else:
        recent_volume_trend = volume_condition

      return volume_condition and recent_volume_trend
      
    except Exception as e:
      logger.warning(f"Error checking volume confirmation: {e}")
      return False

  def _detect_breakout(self, indicators: pd.DataFrame, current_price: float) -> float:
    """
    Detect price breakouts from consolidation.

    Args:
        indicators: DataFrame with technical indicators
        current_price: Current price

    Returns:
        Breakout signal (-1 to 1)
    """
    try:
      # Check for required columns
      required_cols = ['BB_Upper', 'BB_Lower', 'High', 'Low']
      for col in required_cols:
        if col not in indicators.columns:
          logger.warning(f"Missing column for breakout detection: {col}")
          return 0.0
          
      latest = indicators.iloc[-1]
      
      # Check for NaN values
      if any(pd.isna(latest[col]) for col in required_cols):
        logger.warning("NaN values in breakout indicators")
        return 0.0

      # Bollinger Band breakout
      bb_breakout = 0
      if current_price > latest['BB_Upper']:
        bb_breakout = (current_price - latest['BB_Upper']) / latest['BB_Upper']
      elif current_price < latest['BB_Lower']:
        bb_breakout = (current_price - latest['BB_Lower']) / latest['BB_Lower']

      # High/Low breakout (20-period)
      if len(indicators) >= 20:
        period_high = indicators['High'].tail(20).max()
        period_low = indicators['Low'].tail(20).min()
      else:
        period_high = indicators['High'].max()
        period_low = indicators['Low'].min()

      high_breakout = 0
      if current_price > period_high:
        high_breakout = (current_price - period_high) / period_high
      elif current_price < period_low:
        high_breakout = (current_price - period_low) / period_low

      # Combine breakout signals
      breakout_signal = max(abs(bb_breakout), abs(high_breakout))
      if bb_breakout < 0 or high_breakout < 0:
        breakout_signal = -breakout_signal

      return np.clip(breakout_signal * 10, -1.0, 1.0)  # Scale and clip
      
    except Exception as e:
      logger.warning(f"Error detecting breakout: {e}")
      return 0.0

  def _adjust_confidence_for_regime(self,
                                    confidence: float,
                                    signal_type: SignalType,
                                    regime: str) -> float:
    """
    Adjust confidence based on market regime.

    Args:
        confidence: Base confidence
        signal_type: Signal type
        regime: Market regime

    Returns:
        Adjusted confidence
    """
    # Momentum strategies work best in trending markets
    if "Trending" in regime:
      # Boost confidence in trending markets
      if ((signal_type == SignalType.BUY and "Up" in regime) or
              (signal_type == SignalType.SELL and "Down" in regime)):
        confidence = min(0.95, confidence * 1.2)
      else:
        # Reduce confidence for counter-trend signals
        confidence = confidence * 0.8

    elif "Ranging" in regime:
      # Reduce confidence in ranging markets (momentum less reliable)
      confidence = confidence * 0.7

    return max(0.0, confidence)
