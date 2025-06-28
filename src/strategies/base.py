"""
Base strategy class for all trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from enum import Enum

from utils import get_logger

logger = get_logger(__name__)


class SignalType(Enum):
  """Trading signal types."""
  BUY = "BUY"
  SELL = "SELL"
  HOLD = "HOLD"


class TradingSignal:
  """Trading signal with metadata."""

  def __init__(self,
               signal_type: SignalType,
               confidence: float,
               price: float,
               quantity: float = 0.0,
               metadata: Dict = None):
    """
    Initialize trading signal.

    Args:
        signal_type: Type of signal (BUY, SELL, HOLD)
        confidence: Signal confidence (0.0 to 1.0)
        price: Signal price
        quantity: Suggested position size
        metadata: Additional signal information
    """
    self.signal_type = signal_type
    self.confidence = confidence
    self.price = price
    self.quantity = quantity
    self.metadata = metadata or {}
    self.timestamp = pd.Timestamp.now()

  def __repr__(self):
    return (f"TradingSignal({self.signal_type.value}, "
            f"confidence={self.confidence:.3f}, "
            f"price={self.price:.2f})")


class BaseStrategy(ABC):
  """
  Abstract base class for all trading strategies.

  All trading strategies must implement the generate_signal method
  and can optionally override other methods for customization.
  """

  def __init__(self, name: str, config: Dict = None):
    """
    Initialize base strategy.

    Args:
        name: Strategy name
        config: Strategy configuration parameters
    """
    self.name = name
    self.config = config or {}
    self.is_active = True
    self.performance_metrics = {
        'total_signals': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'avg_confidence': 0.0,
        'last_signal_time': None
    }

    logger.info(f"Strategy '{name}' initialized")

  @abstractmethod
  def generate_signal(self,
                      data: pd.DataFrame,
                      current_price: float,
                      regime: str = None,
                      **kwargs) -> TradingSignal:
    """
    Generate trading signal based on market data.

    Args:
        data: Historical market data
        current_price: Current asset price
        regime: Current market regime (if available)
        **kwargs: Additional parameters

    Returns:
        TradingSignal object
    """
    pass

  def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators needed for the strategy.

    Args:
        data: Market data with OHLCV columns

    Returns:
        DataFrame with calculated indicators
    """
    indicators = data.copy()

    # Basic moving averages
    indicators['SMA_10'] = data['Close'].rolling(10).mean()
    indicators['SMA_20'] = data['Close'].rolling(20).mean()
    indicators['SMA_50'] = data['Close'].rolling(50).mean()
    indicators['EMA_12'] = data['Close'].ewm(span=12).mean()
    indicators['EMA_26'] = data['Close'].ewm(span=26).mean()

    # RSI
    indicators['RSI'] = self._calculate_rsi(data['Close'])

    # MACD
    indicators['MACD'] = indicators['EMA_12'] - indicators['EMA_26']
    indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
    indicators['MACD_Histogram'] = indicators['MACD'] - \
        indicators['MACD_Signal']

    # Bollinger Bands
    bb_period = 20
    bb_std = 2
    indicators['BB_Middle'] = data['Close'].rolling(bb_period).mean()
    bb_std_dev = data['Close'].rolling(bb_period).std()
    indicators['BB_Upper'] = indicators['BB_Middle'] + (bb_std_dev * bb_std)
    indicators['BB_Lower'] = indicators['BB_Middle'] - (bb_std_dev * bb_std)

    # Average True Range (ATR)
    indicators['ATR'] = self._calculate_atr(data)

    # Volume indicators
    indicators['Volume_SMA'] = data['Volume'].rolling(20).mean()
    indicators['Volume_Ratio'] = data['Volume'] / indicators['Volume_SMA']

    return indicators

  def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

  def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())

    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    return true_range.rolling(period).mean()

  def validate_signal(self, signal: TradingSignal) -> bool:
    """
    Validate trading signal before execution.

    Args:
        signal: Trading signal to validate

    Returns:
        True if signal is valid, False otherwise
    """
    # Basic validation checks
    if signal.confidence < 0.0 or signal.confidence > 1.0:
      logger.warning(f"Invalid confidence value: {signal.confidence}")
      return False

    if signal.price <= 0:
      logger.warning(f"Invalid price value: {signal.price}")
      return False

    if signal.signal_type not in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]:
      logger.warning(f"Invalid signal type: {signal.signal_type}")
      return False

    return True

  def update_performance_metrics(self, signal: TradingSignal):
    """
    Update strategy performance metrics.

    Args:
        signal: Generated trading signal
    """
    self.performance_metrics['total_signals'] += 1

    if signal.signal_type == SignalType.BUY:
      self.performance_metrics['buy_signals'] += 1
    elif signal.signal_type == SignalType.SELL:
      self.performance_metrics['sell_signals'] += 1

    # Update average confidence
    total = self.performance_metrics['total_signals']
    current_avg = self.performance_metrics['avg_confidence']
    self.performance_metrics['avg_confidence'] = (
        (current_avg * (total - 1) + signal.confidence) / total
    )

    self.performance_metrics['last_signal_time'] = signal.timestamp

  def get_performance_summary(self) -> Dict:
    """
    Get strategy performance summary.

    Returns:
        Dictionary with performance metrics
    """
    return {
        'name': self.name,
        'is_active': self.is_active,
        **self.performance_metrics
    }

  def reset_performance_metrics(self):
    """Reset strategy performance metrics."""
    self.performance_metrics = {
        'total_signals': 0,
        'buy_signals': 0,
        'sell_signals': 0,
        'avg_confidence': 0.0,
        'last_signal_time': None
    }

    logger.info(f"Performance metrics reset for strategy '{self.name}'")

  def set_active(self, active: bool):
    """
    Set strategy active status.

    Args:
        active: Whether strategy should be active
    """
    self.is_active = active
    status = "activated" if active else "deactivated"
    logger.info(f"Strategy '{self.name}' {status}")

  def __str__(self):
    return f"{self.name}Strategy"

  def __repr__(self):
    return f"{self.__class__.__name__}(name='{self.name}', active={self.is_active})"

  def _adjust_confidence_for_regime(self, confidence: float, signal_type: SignalType, regime: str) -> float:
    """
    Adjust signal confidence based on market regime.
    
    Args:
        confidence: Base signal confidence
        signal_type: Type of trading signal
        regime: Current market regime
        
    Returns:
        Adjusted confidence score
    """
    if not regime or confidence <= 0:
      return confidence
      
    # Boost confidence when signal aligns with regime
    if "Trending Up" in regime and signal_type == SignalType.BUY:
      return min(0.95, confidence * 1.1)
    elif "Trending Down" in regime and signal_type == SignalType.SELL:
      return min(0.95, confidence * 1.1)
    elif ("Ranging" in regime or "Sideways" in regime):
      # Mean reversion strategies work better in ranging markets
      if hasattr(self, 'name') and 'Reversion' in self.name:
        return min(0.95, confidence * 1.05)
      # Reduce momentum strategy confidence in ranging markets
      elif hasattr(self, 'name') and 'Momentum' in self.name:
        return confidence * 0.9
        
    # Reduce confidence when signal contradicts regime
    if "Trending Up" in regime and signal_type == SignalType.SELL:
      return confidence * 0.8
    elif "Trending Down" in regime and signal_type == SignalType.BUY:
      return confidence * 0.8
        
    return confidence
