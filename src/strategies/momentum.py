"""
Momentum-based trading strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

from .base import BaseStrategy, TradingSignal, SignalType
from ..utils import get_logger, config

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

  def __init__(self, config: Dict = None):
    """
    Initialize momentum strategy.

    Args:
        config: Strategy configuration parameters
    """
    default_config = {
        'rsi_threshold': config.get('strategies.momentum.rsi_threshold', 70),
        'volume_multiplier': config.get('strategies.momentum.volume_multiplier', 1.5),
        'trend_strength_min': config.get('strategies.momentum.trend_strength_min', 0.6),
        'min_confidence': 0.6,
        'lookback_period': 20,
        'breakout_threshold': 0.02  # 2% price movement for breakout
    }

    if config:
      default_config.update(config)

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
                      **kwargs) -> TradingSignal:
    """
    Generate momentum-based trading signal.

    Args:
        data: Historical market data
        current_price: Current asset price
        regime: Current market regime

    Returns:
        TradingSignal object
    """
    if len(data) < 50:  # Need sufficient data for indicators
      return TradingSignal(SignalType.HOLD, 0.0, current_price)

    # Calculate technical indicators
    indicators = self.calculate_indicators(data)

    # Get latest values
    latest = indicators.iloc[-1]
    prev = indicators.iloc[-2]

    # Momentum signals
    momentum_score = self._calculate_momentum_score(indicators)
    trend_strength = self._calculate_trend_strength(indicators)
    volume_confirmation = self._check_volume_confirmation(indicators)
    breakout_signal = self._detect_breakout(indicators, current_price)

    # Determine signal type and confidence
    signal_type = SignalType.HOLD
    confidence = 0.0

    # Bullish momentum conditions
    if (momentum_score > 0.6 and
        trend_strength > self.trend_strength_min and
        volume_confirmation and
            latest['RSI'] > 50 and latest['RSI'] < 80):

      signal_type = SignalType.BUY
      confidence = min(0.95, momentum_score * 0.8 + trend_strength * 0.2)

      # Boost confidence for breakouts
      if breakout_signal > 0:
        confidence = min(0.95, confidence + 0.1)

    # Bearish momentum conditions
    elif (momentum_score < -0.6 and
          trend_strength > self.trend_strength_min and
          volume_confirmation and
          latest['RSI'] < 50 and latest['RSI'] > 20):

      signal_type = SignalType.SELL
      confidence = min(0.95, abs(momentum_score) * 0.8 + trend_strength * 0.2)

      # Boost confidence for breakouts
      if breakout_signal < 0:
        confidence = min(0.95, confidence + 0.1)

    # Regime-based adjustments
    if regime:
      confidence = self._adjust_confidence_for_regime(
        confidence, signal_type, regime)

    # Create signal with metadata
    metadata = {
        'momentum_score': momentum_score,
        'trend_strength': trend_strength,
        'volume_confirmation': volume_confirmation,
        'breakout_signal': breakout_signal,
        'rsi': latest['RSI'],
        'macd': latest['MACD'],
        'regime': regime
    }

    signal = TradingSignal(signal_type, confidence,
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
    latest = indicators.iloc[-1]

    # RSI momentum component
    rsi_norm = (latest['RSI'] - 50) / 50  # Normalize RSI to -1 to 1

    # MACD momentum component
    macd_signal = 1 if latest['MACD'] > latest['MACD_Signal'] else -1
    macd_strength = abs(latest['MACD_Histogram']) / \
        indicators['MACD_Histogram'].rolling(20).std().iloc[-1]
    macd_momentum = macd_signal * min(1.0, macd_strength)

    # Price momentum (rate of change)
    price_change = (
      latest['Close'] - indicators['Close'].iloc[-10]) / indicators['Close'].iloc[-10]
    price_momentum = np.tanh(price_change * 10)  # Normalize to -1 to 1

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

  def _calculate_trend_strength(self, indicators: pd.DataFrame) -> float:
    """
    Calculate trend strength.

    Args:
        indicators: DataFrame with technical indicators

    Returns:
        Trend strength between 0 and 1
    """
    latest = indicators.iloc[-1]

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
    latest = indicators.iloc[-1]

    # Volume should be above average for momentum confirmation
    volume_condition = latest['Volume_Ratio'] > self.volume_multiplier

    # Recent volume trend (last 3 periods)
    recent_volume_trend = indicators['Volume_Ratio'].tail(3).mean() > 1.0

    return volume_condition and recent_volume_trend

  def _detect_breakout(self, indicators: pd.DataFrame, current_price: float) -> float:
    """
    Detect price breakouts from consolidation.

    Args:
        indicators: DataFrame with technical indicators
        current_price: Current price

    Returns:
        Breakout signal (-1 to 1)
    """
    latest = indicators.iloc[-1]

    # Bollinger Band breakout
    bb_breakout = 0
    if current_price > latest['BB_Upper']:
      bb_breakout = (current_price - latest['BB_Upper']) / latest['BB_Upper']
    elif current_price < latest['BB_Lower']:
      bb_breakout = (current_price - latest['BB_Lower']) / latest['BB_Lower']

    # High/Low breakout (20-period)
    period_high = indicators['High'].tail(20).max()
    period_low = indicators['Low'].tail(20).min()

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
