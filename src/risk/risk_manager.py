"""
Main risk management system.
"""
from typing import Dict, Optional, List
from datetime import datetime
import numpy as np

from .position_sizing import AdvancedPositionSizer
from .drawdown_manager import DrawdownManager
from ..strategies.base import TradingSignal, SignalType
from ..utils import get_logger, config

logger = get_logger(__name__)


class RiskManager:
  """
  Comprehensive risk management system.

  Features:
  - Position sizing
  - Drawdown management
  - Exposure limits
  - Correlation risk
  - Volatility-based risk adjustment
  """

  def __init__(self):
    """Initialize risk manager."""
    self.position_sizer = PositionSizer()
    self.drawdown_manager = DrawdownManager()

    # Risk parameters from config
    self.max_risk_per_trade = config.get('trading.max_risk_per_trade', 0.02)
    self.max_portfolio_drawdown = config.get(
      'trading.max_portfolio_drawdown', 0.15)
    self.max_position_concentration = config.get(
      'risk.max_position_concentration', 0.25)
    self.max_correlation_exposure = config.get(
      'risk.max_correlation_exposure', 0.60)

    # Risk state
    self.risk_alerts = []
    self.emergency_stop = False
    self.last_risk_check = datetime.now()

    logger.info("RiskManager initialized")

  def evaluate_trade_risk(self,
                          signal: TradingSignal,
                          asset: str,
                          current_price: float,
                          regime: str = None,
                          portfolio_state: Dict = None) -> bool:
    """
    Evaluate if a trade meets risk criteria.

    Args:
        signal: Trading signal to evaluate
        asset: Asset symbol
        current_price: Current market price
        regime: Market regime
        portfolio_state: Current portfolio state

    Returns:
        True if trade is approved, False if rejected
    """
    try:
      # Emergency stop check
      if self.emergency_stop:
        logger.warning("Emergency stop active - rejecting all trades")
        return False

      # Signal confidence check
      if signal.confidence < config.get('trading.min_confidence_threshold', 0.6):
        self._add_risk_alert(
          f"Signal confidence too low: {signal.confidence:.3f}")
        return False

      # Portfolio drawdown check
      if not self.drawdown_manager.check_drawdown_limits(portfolio_state):
        self._add_risk_alert("Portfolio drawdown limit exceeded")
        return False

      # Position concentration check
      if not self._check_position_concentration(asset, signal, portfolio_state):
        self._add_risk_alert(
          f"Position concentration limit exceeded for {asset}")
        return False

      # Volatility-based risk adjustment
      if not self._check_volatility_risk(asset, signal, regime):
        self._add_risk_alert(f"Volatility risk too high for {asset}")
        return False

      # Correlation risk check
      if not self._check_correlation_risk(asset, portfolio_state):
        self._add_risk_alert(f"Correlation risk too high for {asset}")
        return False

      # Regime-based risk adjustment
      if not self._check_regime_risk(signal, regime):
        self._add_risk_alert(f"Regime risk check failed for {regime}")
        return False

      logger.info(
        f"Risk check passed for {asset} {signal.signal_type.value} signal")
      return True

    except Exception as e:
      logger.error(f"Error in risk evaluation: {e}")
      return False  # Fail safe

  def _check_position_concentration(self,
                                    asset: str,
                                    signal: TradingSignal,
                                    portfolio_state: Dict = None) -> bool:
    """Check if position concentration is within limits."""
    if not portfolio_state:
      return True

    # Calculate position size
    position_value = self.position_sizer.calculate_position_size(
        signal=signal,
        asset=asset,
        portfolio_value=portfolio_state.get('total_value', 10000)
    )

    # Check concentration limit
    portfolio_value = portfolio_state.get('total_value', 10000)
    concentration = position_value / portfolio_value

    if concentration > self.max_position_concentration:
      logger.warning(
        f"Position concentration {concentration:.2%} exceeds limit {self.max_position_concentration:.2%}")
      return False

    return True

  def _check_volatility_risk(self,
                             asset: str,
                             signal: TradingSignal,
                             regime: str = None) -> bool:
    """Check volatility-based risk limits."""
    # Get volatility from signal metadata
    volatility = signal.metadata.get('volatility', 0.02)  # Default 2%

    # Higher volatility requires higher confidence
    min_confidence_for_volatility = 0.5 + \
        (volatility * 10)  # Scale with volatility

    if signal.confidence < min_confidence_for_volatility:
      logger.warning(
        f"Signal confidence {signal.confidence:.3f} too low for volatility {volatility:.3f}")
      return False

    # Regime-based volatility limits
    if regime and "High" in regime and volatility > 0.05:  # 5% daily volatility
      logger.warning(
        f"Volatility {volatility:.3f} too high for regime {regime}")
      return False

    return True

  def _check_correlation_risk(self, asset: str, portfolio_state: Dict = None) -> bool:
    """Check correlation exposure limits."""
    if not portfolio_state or 'positions' not in portfolio_state:
      return True

    # Simplified correlation risk check
    # In a full implementation, this would use historical correlation matrices

    positions = portfolio_state['positions']
    crypto_assets = ['BTC', 'ETH', 'SOL', 'ADA', 'DOT']  # Major crypto assets

    # Count exposure to correlated crypto assets
    crypto_exposure = 0
    for pos_asset, position in positions.items():
      if pos_asset in crypto_assets:
        crypto_exposure += position.get('market_value', 0)

    total_portfolio_value = portfolio_state.get('total_value', 10000)
    correlation_exposure = crypto_exposure / total_portfolio_value

    if correlation_exposure > self.max_correlation_exposure:
      logger.warning(
        f"Correlation exposure {correlation_exposure:.2%} exceeds limit {self.max_correlation_exposure:.2%}")
      return False

    return True

  def _check_regime_risk(self, signal: TradingSignal, regime: str = None) -> bool:
    """Check regime-specific risk factors."""
    if not regime:
      return True

    # Reduce risk in highly volatile regimes
    if "High" in regime and signal.confidence < 0.8:
      return False

    # Be more conservative during regime transitions
    regime_transition_prob = signal.metadata.get('regime_transition_prob', 0)
    if regime_transition_prob > 0.3 and signal.confidence < 0.75:
      return False

    return True

  def update_risk_state(self, portfolio_state: Dict):
    """Update internal risk state based on portfolio."""
    try:
      # Update drawdown manager
      self.drawdown_manager.update_portfolio_state(portfolio_state)

      # Check for emergency stop conditions
      current_drawdown = portfolio_state.get('max_drawdown', 0)
      if current_drawdown > self.max_portfolio_drawdown * 1.2:  # 20% buffer
        self.emergency_stop = True
        logger.critical(
          f"Emergency stop activated: drawdown {current_drawdown:.2%}")

      # Reset emergency stop if conditions improve
      elif self.emergency_stop and current_drawdown < self.max_portfolio_drawdown * 0.8:
        self.emergency_stop = False
        logger.info("Emergency stop deactivated")

      self.last_risk_check = datetime.now()

    except Exception as e:
      logger.error(f"Error updating risk state: {e}")

  def _add_risk_alert(self, message: str):
    """Add a risk alert."""
    alert = {
        'timestamp': datetime.now(),
        'message': message,
        'severity': 'warning'
    }
    self.risk_alerts.append(alert)

    # Keep only recent alerts
    if len(self.risk_alerts) > 100:
      self.risk_alerts = self.risk_alerts[-50:]

    logger.warning(f"Risk Alert: {message}")

  def get_risk_summary(self) -> Dict:
    """Get comprehensive risk summary."""
    return {
        'emergency_stop': self.emergency_stop,
        'max_risk_per_trade': self.max_risk_per_trade,
        'max_portfolio_drawdown': self.max_portfolio_drawdown,
        'max_position_concentration': self.max_position_concentration,
        'recent_alerts': len([a for a in self.risk_alerts
                              # Last hour
                              if (datetime.now() - a['timestamp']).seconds < 3600]),
        'last_risk_check': self.last_risk_check,
        'drawdown_status': self.drawdown_manager.get_status()
    }

  def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
    """Get recent risk alerts."""
    cutoff_time = datetime.now().timestamp() - (hours * 3600)

    return [
        alert for alert in self.risk_alerts
        if alert['timestamp'].timestamp() > cutoff_time
    ]

  def reset_risk_state(self):
    """Reset risk manager state."""
    self.risk_alerts.clear()
    self.emergency_stop = False
    self.drawdown_manager.reset()

    logger.info("Risk manager state reset")

  def set_emergency_stop(self, active: bool, reason: str = "Manual override"):
    """Manually set emergency stop state."""
    self.emergency_stop = active

    if active:
      self._add_risk_alert(f"Emergency stop activated: {reason}")
    else:
      self._add_risk_alert(f"Emergency stop deactivated: {reason}")
