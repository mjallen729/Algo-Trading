"""
Advanced Position Sizing Module

This module provides sophisticated position sizing algorithms for cryptocurrency trading,
including Kelly Criterion, ATR-based sizing, and confidence-weighted approaches.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from dataclasses import dataclass
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradingSignal:
  """Trading signal with confidence and metadata."""
  action: str  # 'buy', 'sell', 'hold'
  symbol: str
  price: float
  confidence: float  # 0.0 to 1.0
  strategy: str
  timestamp: Any
  stop_loss: Optional[float] = None
  take_profit: Optional[float] = None
  metadata: Optional[Dict] = None


class AdvancedPositionSizer:
  """
  Advanced position sizing with multiple methodologies:
  - Fixed percentage
  - Kelly Criterion (fractional)
  - ATR-based sizing
  - Confidence-weighted sizing
  - Risk parity approach
  """

  def __init__(self,
               method: str = "kelly_fractional",
               max_position_size: float = 0.25,  # 25% max
               base_risk_per_trade: float = 0.02,  # 2% base risk
               kelly_fraction: float = 0.25,  # Use 25% of Kelly
               confidence_scaling: bool = True):
    """
    Initialize position sizer.

    Args:
        method: Position sizing method ('fixed', 'kelly_fractional', 'atr', 'confidence')
        max_position_size: Maximum position size as fraction of portfolio
        base_risk_per_trade: Base risk per trade as fraction of portfolio
        kelly_fraction: Fraction of full Kelly to use (for safety)
        confidence_scaling: Whether to scale positions by signal confidence
    """
    self.method = method
    self.max_position_size = max_position_size
    self.base_risk_per_trade = base_risk_per_trade
    self.kelly_fraction = kelly_fraction
    self.confidence_scaling = confidence_scaling

    # Performance tracking for Kelly calculation
    self.trade_history = []
    self.win_rate = 0.5  # Initial assumption
    self.avg_win = 0.02  # Initial assumption
    self.avg_loss = 0.015  # Initial assumption

    logger.info(f"PositionSizer initialized with method: {method}")

  def calculate_position_size(self,
                              signal: TradingSignal,
                              symbol: str,
                              portfolio_value: float,
                              current_volatility: float = None,
                              market_regime: str = None) -> float:
    """
    Calculate optimal position size based on selected method.

    Args:
        signal: Trading signal with confidence
        symbol: Symbol being traded
        portfolio_value: Current portfolio value
        current_volatility: Current market volatility (for ATR method)
        market_regime: Current market regime for adjustments

    Returns:
        Position size in USD
    """
    if signal.action == 'hold':
      return 0.0

    # Calculate base position size using selected method
    if self.method == "fixed":
      base_size = self._fixed_percentage_sizing(portfolio_value)
    elif self.method == "kelly_fractional":
      base_size = self._kelly_fractional_sizing(portfolio_value)
    elif self.method == "atr":
      base_size = self._atr_based_sizing(portfolio_value, current_volatility)
    elif self.method == "confidence":
      base_size = self._confidence_based_sizing(signal, portfolio_value)
    else:
      # Default to fixed percentage
      base_size = self._fixed_percentage_sizing(portfolio_value)

    # Apply confidence scaling if enabled
    if self.confidence_scaling and signal.confidence is not None:
      base_size = self._apply_confidence_adjustment(
        base_size, signal.confidence)

    # Apply market regime adjustments
    if market_regime:
      base_size = self._apply_regime_adjustment(base_size, market_regime)

    # Ensure position size doesn't exceed maximum
    max_size = portfolio_value * self.max_position_size
    final_size = min(base_size, max_size)

    # Ensure minimum viable position size
    min_size = 10.0  # Minimum $10 position
    if final_size < min_size:
      final_size = 0.0

    logger.debug(f"Position size calculated: ${final_size:.2f} for {symbol}")

    return final_size

  def _fixed_percentage_sizing(self, portfolio_value: float) -> float:
    """Fixed percentage of portfolio."""
    return portfolio_value * self.base_risk_per_trade

  def _kelly_fractional_sizing(self, portfolio_value: float) -> float:
    """
    Kelly Criterion sizing with fractional approach for safety.

    Kelly % = (Win Rate * Avg Win - Loss Rate * Avg Loss) / Avg Win
    """
    if self.win_rate <= 0 or self.avg_win <= 0:
      # Fall back to fixed percentage if no history
      return self._fixed_percentage_sizing(portfolio_value)

    loss_rate = 1 - self.win_rate

    # Calculate Kelly percentage
    kelly_pct = (self.win_rate * self.avg_win -
                 loss_rate * self.avg_loss) / self.avg_win

    # Apply safety fraction and ensure positive
    kelly_pct = max(0, kelly_pct * self.kelly_fraction)

    # Cap at maximum allowed
    kelly_pct = min(kelly_pct, self.max_position_size)

    return portfolio_value * kelly_pct

  def _atr_based_sizing(self, portfolio_value: float, volatility: float) -> float:
    """
    ATR-based position sizing - larger positions in low volatility.

    Args:
        portfolio_value: Portfolio value
        volatility: Current ATR or volatility measure
    """
    if volatility is None or volatility <= 0:
      return self._fixed_percentage_sizing(portfolio_value)

    # Inverse relationship with volatility
    # Higher volatility = smaller position
    base_size = portfolio_value * self.base_risk_per_trade

    # Normalize volatility (assume 2% as baseline)
    volatility_adjustment = 0.02 / \
        max(volatility, 0.005)  # Prevent division by zero

    # Cap adjustment between 0.5x and 2.0x
    volatility_adjustment = max(0.5, min(2.0, volatility_adjustment))

    return base_size * volatility_adjustment

  def _confidence_based_sizing(self, signal: TradingSignal, portfolio_value: float) -> float:
    """
    Confidence-based sizing - higher confidence = larger position.
    """
    base_size = portfolio_value * self.base_risk_per_trade

    if signal.confidence is None:
      return base_size

    # Scale based on confidence (0.5x to 2.0x)
    confidence_multiplier = 0.5 + (signal.confidence * 1.5)

    return base_size * confidence_multiplier

  def _apply_confidence_adjustment(self, base_size: float, confidence: float) -> float:
    """
    Adjust position size based on signal confidence.

    Args:
        base_size: Base position size
        confidence: Signal confidence (0.0 to 1.0)

    Returns:
        Adjusted position size
    """
    # Confidence scaling (0.5x to 1.5x based on confidence)
    confidence_multiplier = 0.5 + confidence

    return base_size * confidence_multiplier

  def _apply_regime_adjustment(self, base_size: float, regime: str) -> float:
    """
    Adjust position size based on market regime.

    Args:
        base_size: Base position size
        regime: Market regime ('trending_up', 'trending_down', 'ranging', etc.)

    Returns:
        Adjusted position size
    """
    regime_multipliers = {
        'trending_up': 1.2,      # Slightly larger positions in uptrends
        'trending_down': 0.8,    # Smaller positions in downtrends
        'ranging': 1.0,          # Normal positions in ranging markets
        'high_volatility': 0.7,  # Much smaller positions in high volatility
        'low_volatility': 1.1    # Slightly larger in stable markets
    }

    multiplier = regime_multipliers.get(regime.lower(), 1.0)
    return base_size * multiplier

  def calculate_stop_loss_size(self,
                               entry_price: float,
                               stop_loss_price: float,
                               risk_amount: float) -> float:
    """
    Calculate position size based on stop loss distance.

    Args:
        entry_price: Entry price
        stop_loss_price: Stop loss price
        risk_amount: Maximum risk amount in USD

    Returns:
        Position size (quantity)
    """
    if entry_price <= 0 or stop_loss_price <= 0:
      return 0.0

    # Calculate risk per unit
    risk_per_unit = abs(entry_price - stop_loss_price)

    if risk_per_unit == 0:
      return 0.0

    # Calculate quantity based on risk
    quantity = risk_amount / risk_per_unit

    return quantity

  def update_performance(self, trade_result: Dict):
    """
    Update performance statistics for Kelly calculation.

    Args:
        trade_result: Dictionary with trade outcome
                     {'pnl': float, 'return_pct': float, 'win': bool}
    """
    self.trade_history.append(trade_result)

    # Keep only recent trades (last 100)
    if len(self.trade_history) > 100:
      self.trade_history = self.trade_history[-100:]

    # Recalculate statistics
    if len(self.trade_history) >= 10:  # Need minimum sample size
      wins = [t for t in self.trade_history if t.get('win', False)]
      losses = [t for t in self.trade_history if not t.get('win', False)]

      self.win_rate = len(wins) / len(self.trade_history)

      if wins:
        self.avg_win = np.mean([t['return_pct'] for t in wins])
      if losses:
        self.avg_loss = abs(np.mean([t['return_pct'] for t in losses]))

    logger.debug(f"Performance updated: WR={self.win_rate:.2f}, "
                 f"AvgWin={self.avg_win:.3f}, AvgLoss={self.avg_loss:.3f}")

  def get_sizing_metrics(self, signal: TradingSignal, portfolio_value: float) -> Dict:
    """
    Get detailed sizing metrics for analysis.

    Args:
        signal: Trading signal
        portfolio_value: Portfolio value

    Returns:
        Dictionary with sizing metrics
    """
    base_size = self.calculate_position_size(
      signal, signal.symbol, portfolio_value)

    metrics = {
        'method': self.method,
        'base_position_size': base_size,
        'position_percentage': (base_size / portfolio_value) * 100,
        'confidence': signal.confidence,
        'kelly_fraction_used': self.kelly_fraction,
        'max_position_limit': self.max_position_size * 100,
        'risk_amount': base_size,  # Simplified - actual risk depends on stop loss
        'win_rate': self.win_rate,
        'avg_win': self.avg_win,
        'avg_loss': self.avg_loss,
        'trades_in_history': len(self.trade_history)
    }

    return metrics

  def calculate_portfolio_heat(self, open_positions: Dict, portfolio_value: float) -> Dict:
    """
    Calculate portfolio heat (total risk exposure).

    Args:
        open_positions: Dictionary of open positions {symbol: position_size}
        portfolio_value: Current portfolio value

    Returns:
        Dictionary with heat metrics
    """
    total_exposure = sum(abs(size) for size in open_positions.values())

    heat_metrics = {
        'total_exposure_usd': total_exposure,
        'total_exposure_pct': (total_exposure / portfolio_value) * 100,
        'number_of_positions': len(open_positions),
        'average_position_size': total_exposure / len(open_positions) if open_positions else 0,
        'largest_position': max(abs(size) for size in open_positions.values()) if open_positions else 0,
        'portfolio_utilization': (total_exposure / portfolio_value) * 100
    }

    return heat_metrics

  def suggest_position_adjustments(self,
                                   current_positions: Dict,
                                   portfolio_value: float,
                                   target_heat: float = 0.5) -> Dict:
    """
    Suggest position adjustments to maintain target portfolio heat.

    Args:
        current_positions: Current positions
        portfolio_value: Portfolio value
        target_heat: Target portfolio heat (0.5 = 50%)

    Returns:
        Dictionary with adjustment suggestions
    """
    heat_metrics = self.calculate_portfolio_heat(
      current_positions, portfolio_value)
    current_heat = heat_metrics['portfolio_utilization'] / 100

    suggestions = {
        'current_heat': current_heat,
        'target_heat': target_heat,
        'action_needed': False,
        'adjustments': {}
    }

    if current_heat > target_heat:
      # Reduce positions
      reduction_factor = target_heat / current_heat
      suggestions['action_needed'] = True
      suggestions['action'] = 'reduce'

      for symbol, size in current_positions.items():
        new_size = size * reduction_factor
        suggestions['adjustments'][symbol] = {
            'current': size,
            'suggested': new_size,
            'change': new_size - size
        }

    elif current_heat < target_heat * 0.8:  # Allow for some buffer
      # Can increase positions
      suggestions['action'] = 'can_increase'
      available_capacity = (target_heat - current_heat) * portfolio_value
      suggestions['available_capacity'] = available_capacity

    return suggestions
