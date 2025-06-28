"""
Drawdown management and portfolio protection.
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np

from ..utils import get_logger, config

logger = get_logger(__name__)


class DrawdownManager:
    """
    Advanced drawdown management system.
    
    Features:
    - Real-time drawdown monitoring
    - Dynamic position scaling
    - Circuit breakers
    - Recovery tracking
    """
    
    def __init__(self):
        """Initialize drawdown manager."""
        self.max_portfolio_drawdown = config.get('trading.max_portfolio_drawdown', 0.15)
        self.circuit_breaker_threshold = self.max_portfolio_drawdown * 0.8  # 80% of max
        self.recovery_threshold = 0.05  # 5% recovery to resume normal trading
        
        # State tracking
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        self.drawdown_start_time = None
        self.in_drawdown = False
        self.circuit_breaker_active = False
        
        # History tracking
        self.drawdown_history = []
        self.value_history = []
        
        logger.info(f"DrawdownManager initialized with max drawdown: {self.max_portfolio_drawdown:.1%}")
    
    def update_portfolio_state(self, portfolio_state: Dict):
        """
        Update drawdown calculations with current portfolio state.
        
        Args:
            portfolio_state: Current portfolio state dictionary
        """
        try:
            current_value = portfolio_state.get('current_value', 0)
            
            if current_value <= 0:
                logger.warning("Invalid portfolio value for drawdown calculation")
                return
            
            # Update peak value
            if current_value > self.peak_value:
                self.peak_value = current_value
                
                # If we were in drawdown and recovered, mark recovery
                if self.in_drawdown:
                    recovery_amount = (current_value - self._get_drawdown_low()) / self._get_drawdown_low()
                    if recovery_amount >= self.recovery_threshold:
                        self._mark_drawdown_recovery()
            
            # Calculate current drawdown
            if self.peak_value > 0:
                self.current_drawdown = (self.peak_value - current_value) / self.peak_value
            else:
                self.current_drawdown = 0.0
            
            # Track maximum drawdown reached
            if self.current_drawdown > self.max_drawdown_reached:
                self.max_drawdown_reached = self.current_drawdown
            
            # Check if entering drawdown
            if not self.in_drawdown and self.current_drawdown > 0.01:  # 1% threshold
                self._enter_drawdown()
            
            # Check circuit breaker conditions
            self._check_circuit_breaker()
            
            # Record history
            self._record_state(current_value)
            
        except Exception as e:
            logger.error(f"Error updating drawdown state: {e}")
    
    def check_drawdown_limits(self, portfolio_state: Dict = None) -> bool:
        """
        Check if current drawdown is within acceptable limits.
        
        Args:
            portfolio_state: Portfolio state (optional)
            
        Returns:
            True if within limits, False if exceeded
        """
        if portfolio_state:
            self.update_portfolio_state(portfolio_state)
        
        # Check maximum drawdown limit
        if self.current_drawdown > self.max_portfolio_drawdown:
            logger.warning(f"Portfolio drawdown {self.current_drawdown:.2%} exceeds limit {self.max_portfolio_drawdown:.2%}")
            return False
        
        # Check circuit breaker
        if self.circuit_breaker_active:
            logger.warning("Circuit breaker active - blocking new trades")
            return False
        
        return True
    
    def get_position_scaling_factor(self) -> float:
        """
        Get position scaling factor based on current drawdown.
        
        Returns:
            Scaling factor (0.0 to 1.0)
        """
        if self.current_drawdown <= 0.02:  # Less than 2% drawdown
            return 1.0
        elif self.current_drawdown <= 0.05:  # 2-5% drawdown
            return 0.8
        elif self.current_drawdown <= 0.10:  # 5-10% drawdown
            return 0.5
        elif self.current_drawdown <= self.max_portfolio_drawdown:  # 10-15% drawdown
            return 0.3
        else:  # Above maximum
            return 0.0
    
    def _enter_drawdown(self):
        """Mark the start of a drawdown period."""
        self.in_drawdown = True
        self.drawdown_start_time = datetime.now()
        
        logger.info(f"Entering drawdown period: {self.current_drawdown:.2%}")
    
    def _mark_drawdown_recovery(self):
        """Mark recovery from drawdown."""
        if self.in_drawdown:
            drawdown_duration = datetime.now() - self.drawdown_start_time
            
            # Record drawdown event
            drawdown_event = {
                'start_time': self.drawdown_start_time,
                'end_time': datetime.now(),
                'duration': drawdown_duration,
                'max_drawdown': self.current_drawdown,
                'peak_value': self.peak_value
            }
            self.drawdown_history.append(drawdown_event)
            
            self.in_drawdown = False
            self.drawdown_start_time = None
            
            # Deactivate circuit breaker if recovery is sufficient
            if self.circuit_breaker_active:
                self.circuit_breaker_active = False
                logger.info("Circuit breaker deactivated due to recovery")
            
            logger.info(f"Recovered from drawdown after {drawdown_duration}")
    
    def _check_circuit_breaker(self):
        """Check and activate circuit breaker if necessary."""
        if not self.circuit_breaker_active and self.current_drawdown > self.circuit_breaker_threshold:
            self.circuit_breaker_active = True
            logger.critical(f"Circuit breaker activated at {self.current_drawdown:.2%} drawdown")
    
    def _get_drawdown_low(self) -> float:
        """Get the lowest value during current drawdown."""
        if not self.value_history:
            return self.peak_value
        
        drawdown_start_idx = max(0, len(self.value_history) - 100)  # Look back 100 periods
        recent_values = [v['value'] for v in self.value_history[drawdown_start_idx:]]
        
        return min(recent_values) if recent_values else self.peak_value
    
    def _record_state(self, current_value: float):
        """Record current state for history tracking."""
        state_record = {
            'timestamp': datetime.now(),
            'value': current_value,
            'peak_value': self.peak_value,
            'drawdown': self.current_drawdown,
            'in_drawdown': self.in_drawdown,
            'circuit_breaker': self.circuit_breaker_active
        }
        
        self.value_history.append(state_record)
        
        # Keep only recent history (last 1000 records)
        if len(self.value_history) > 1000:
            self.value_history = self.value_history[-500:]
    
    def get_drawdown_statistics(self) -> Dict:
        """
        Get comprehensive drawdown statistics.
        
        Returns:
            Dictionary with drawdown metrics
        """
        stats = {
            'current_drawdown': self.current_drawdown,
            'max_drawdown_reached': self.max_drawdown_reached,
            'peak_value': self.peak_value,
            'in_drawdown': self.in_drawdown,
            'circuit_breaker_active': self.circuit_breaker_active,
            'position_scaling_factor': self.get_position_scaling_factor(),
            'drawdown_limit': self.max_portfolio_drawdown
        }
        
        # Add historical statistics if available
        if self.drawdown_history:
            drawdown_durations = [(d['end_time'] - d['start_time']).total_seconds() / 3600 
                                for d in self.drawdown_history]  # In hours
            max_historical_drawdowns = [d['max_drawdown'] for d in self.drawdown_history]
            
            stats.update({
                'total_drawdown_events': len(self.drawdown_history),
                'avg_drawdown_duration_hours': np.mean(drawdown_durations) if drawdown_durations else 0,
                'max_historical_drawdown': max(max_historical_drawdowns) if max_historical_drawdowns else 0,
                'avg_historical_drawdown': np.mean(max_historical_drawdowns) if max_historical_drawdowns else 0
            })
        
        # Current drawdown duration
        if self.in_drawdown and self.drawdown_start_time:
            current_duration = (datetime.now() - self.drawdown_start_time).total_seconds() / 3600
            stats['current_drawdown_duration_hours'] = current_duration
        
        return stats
    
    def get_status(self) -> Dict:
        """Get current drawdown manager status."""
        return {
            'current_drawdown_pct': self.current_drawdown * 100,
            'max_allowed_pct': self.max_portfolio_drawdown * 100,
            'circuit_breaker_active': self.circuit_breaker_active,
            'in_drawdown_period': self.in_drawdown,
            'position_scaling': self.get_position_scaling_factor(),
            'peak_portfolio_value': self.peak_value
        }
    
    def reset(self):
        """Reset drawdown manager state."""
        self.peak_value = 0.0
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        self.drawdown_start_time = None
        self.in_drawdown = False
        self.circuit_breaker_active = False
        self.drawdown_history.clear()
        self.value_history.clear()
        
        logger.info("Drawdown manager state reset")
    
    def force_circuit_breaker(self, active: bool, reason: str = "Manual override"):
        """Manually activate/deactivate circuit breaker."""
        self.circuit_breaker_active = active
        
        if active:
            logger.critical(f"Circuit breaker manually activated: {reason}")
        else:
            logger.info(f"Circuit breaker manually deactivated: {reason}")
    
    def get_recovery_progress(self) -> Optional[float]:
        """
        Get recovery progress if in drawdown.
        
        Returns:
            Recovery percentage (0.0 to 1.0) or None if not in drawdown
        """
        if not self.in_drawdown or not self.value_history:
            return None
        
        drawdown_low = self._get_drawdown_low()
        current_value = self.value_history[-1]['value']
        
        if drawdown_low >= self.peak_value:
            return 0.0
        
        recovery_progress = (current_value - drawdown_low) / (self.peak_value - drawdown_low)
        return max(0.0, min(1.0, recovery_progress))
