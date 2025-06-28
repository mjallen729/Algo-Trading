"""
Portfolio management for cryptocurrency trading.
Tracks positions, P&L, and performance metrics.
"""
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class Position:
  """Represents a trading position."""
  asset: str
  quantity: float
  entry_price: float
  current_price: float
  entry_time: datetime
  side: str  # 'long' or 'short'
  
  @property
  def market_value(self) -> float:
    """Current market value of position."""
    return self.quantity * self.current_price
  
  @property
  def unrealized_pnl(self) -> float:
    """Unrealized P&L of position."""
    if self.side == 'long':
      return self.quantity * (self.current_price - self.entry_price)
    else:
      return self.quantity * (self.entry_price - self.current_price)
  
  @property
  def unrealized_pnl_pct(self) -> float:
    """Unrealized P&L percentage."""
    if self.side == 'long':
      return (self.current_price - self.entry_price) / self.entry_price
    else:
      return (self.entry_price - self.current_price) / self.entry_price
  
  def to_dict(self) -> Dict:
    """Convert position to dictionary."""
    return {
        'asset': self.asset,
        'quantity': self.quantity,
        'entry_price': self.entry_price,
        'current_price': self.current_price,
        'market_value': self.market_value,
        'unrealized_pnl': self.unrealized_pnl,
        'unrealized_pnl_pct': self.unrealized_pnl_pct,
        'entry_time': self.entry_time,
        'side': self.side
    }


class Portfolio:
  """
  Portfolio management system for cryptocurrency trading.
  
  Features:
  - Position tracking and P&L calculation
  - Performance metrics and analytics
  - Risk monitoring and reporting
  - Trade history management
  """
  
  def __init__(self, initial_capital: float = 10000.0):
    """Initialize portfolio."""
    self.initial_capital = initial_capital
    self.available_capital = initial_capital
    self.positions: Dict[str, Position] = {}
    self.closed_positions: List[Dict] = []
    self.trade_history: List[Dict] = []
    
    # Performance tracking
    self.total_trades = 0
    self.winning_trades = 0
    self.total_fees = 0.0
    self.max_drawdown = 0.0
    self.peak_value = initial_capital
    
    logger.info(f"Portfolio initialized with ${initial_capital:,.2f}")
  
  def add_position(self, 
                   asset: str, 
                   quantity: float, 
                   entry_price: float, 
                   side: str = 'long') -> bool:
    """
    Add a new position to portfolio.
    
    Args:
        asset: Asset symbol
        quantity: Position quantity
        entry_price: Entry price
        side: Position side ('long' or 'short')
    
    Returns:
        True if position added successfully
    """
    try:
      cost = quantity * entry_price
      
      if cost > self.available_capital:
        logger.warning(f"Insufficient capital for position: {asset}")
        return False
      
      # Create position
      position = Position(
          asset=asset,
          quantity=quantity,
          entry_price=entry_price,
          current_price=entry_price,
          entry_time=datetime.now(),
          side=side
      )
      
      # Update portfolio
      self.positions[asset] = position
      self.available_capital -= cost
      self.total_trades += 1
      
      logger.info(f"Added position: {side.upper()} {quantity:.6f} {asset} @ ${entry_price:.2f}")
      
      return True
      
    except Exception as e:
      logger.error(f"Error adding position: {e}")
      return False
  
  def update_position_price(self, asset: str, current_price: float):
    """Update current price for a position."""
    if asset in self.positions:
      self.positions[asset].current_price = current_price
  
  def close_position(self, asset: str, exit_price: Optional[float] = None) -> Optional[Dict]:
    """
    Close a position and realize P&L.
    
    Args:
        asset: Asset symbol
        exit_price: Exit price (uses current price if None)
    
    Returns:
        Closed position details
    """
    if asset not in self.positions:
      logger.warning(f"No position found for {asset}")
      return None
    
    try:
      position = self.positions[asset]
      final_price = exit_price or position.current_price
      
      # Calculate realized P&L
      if position.side == 'long':
        realized_pnl = position.quantity * (final_price - position.entry_price)
      else:
        realized_pnl = position.quantity * (position.entry_price - final_price)
      
      # Update capital
      proceeds = position.quantity * final_price
      self.available_capital += proceeds
      
      # Track winning trades
      if realized_pnl > 0:
        self.winning_trades += 1
      
      # Create closed position record
      closed_position = {
          'asset': asset,
          'quantity': position.quantity,
          'entry_price': position.entry_price,
          'exit_price': final_price,
          'entry_time': position.entry_time,
          'exit_time': datetime.now(),
          'holding_period': datetime.now() - position.entry_time,
          'realized_pnl': realized_pnl,
          'realized_pnl_pct': realized_pnl / (position.quantity * position.entry_price),
          'side': position.side
      }
      
      self.closed_positions.append(closed_position)
      
      # Remove from active positions
      del self.positions[asset]
      
      logger.info(f"Closed position: {asset} - Realized P&L: ${realized_pnl:.2f}")
      
      return closed_position
      
    except Exception as e:
      logger.error(f"Error closing position {asset}: {e}")
      return None
  
  def get_position(self, asset: str) -> Optional[Dict]:
    """Get position details for an asset."""
    if asset in self.positions:
      return self.positions[asset].to_dict()
    return None
  
  def get_all_positions(self) -> Dict[str, Dict]:
    """Get all active positions."""
    return {asset: pos.to_dict() for asset, pos in self.positions.items()}
  
  def get_total_value(self) -> float:
    """Calculate total portfolio value."""
    positions_value = sum(pos.market_value for pos in self.positions.values())
    return self.available_capital + positions_value
  
  def get_available_capital(self) -> float:
    """Get available capital for trading."""
    return self.available_capital
  
  def get_unrealized_pnl(self) -> float:
    """Get total unrealized P&L."""
    return sum(pos.unrealized_pnl for pos in self.positions.values())
  
  def get_realized_pnl(self) -> float:
    """Get total realized P&L."""
    return sum(pos.get('realized_pnl', 0) for pos in self.closed_positions)
  
  def get_total_pnl(self) -> float:
    """Get total P&L (realized + unrealized)."""
    return self.get_realized_pnl() + self.get_unrealized_pnl()
  
  def get_total_return(self) -> float:
    """Get total return percentage."""
    current_value = self.get_total_value()
    return (current_value - self.initial_capital) / self.initial_capital
  
  def update_drawdown(self):
    """Update maximum drawdown calculation."""
    current_value = self.get_total_value()
    
    if current_value > self.peak_value:
      self.peak_value = current_value
    
    current_drawdown = (self.peak_value - current_value) / self.peak_value
    self.max_drawdown = max(self.max_drawdown, current_drawdown)
  
  def get_performance_metrics(self) -> Dict:
    """
    Calculate comprehensive performance metrics.
    
    Returns:
        Dictionary containing performance metrics
    """
    current_value = self.get_total_value()
    total_return = self.get_total_return()
    
    # Win rate
    win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
    
    # Average trade metrics
    if self.closed_positions:
      avg_win = sum(p['realized_pnl'] for p in self.closed_positions if p['realized_pnl'] > 0) / max(1, self.winning_trades)
      avg_loss = sum(p['realized_pnl'] for p in self.closed_positions if p['realized_pnl'] < 0) / max(1, (self.total_trades - self.winning_trades))
      profit_factor = abs(avg_win * self.winning_trades / (avg_loss * (self.total_trades - self.winning_trades))) if avg_loss != 0 else float('inf')
      
      # Holding period statistics
      holding_periods = [p['holding_period'].total_seconds() / 3600 for p in self.closed_positions]  # in hours
      avg_holding_period = sum(holding_periods) / len(holding_periods) if holding_periods else 0
    else:
      avg_win = avg_loss = profit_factor = avg_holding_period = 0
    
    # Sharpe ratio (simplified - assumes daily returns)
    if len(self.closed_positions) > 1:
      returns = [p['realized_pnl_pct'] for p in self.closed_positions]
      avg_return = sum(returns) / len(returns)
      return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
      sharpe_ratio = avg_return / return_std if return_std > 0 else 0
    else:
      sharpe_ratio = 0
    
    return {
        'total_value': current_value,
        'available_capital': self.available_capital,
        'total_return': total_return,
        'total_return_pct': total_return * 100,
        'realized_pnl': self.get_realized_pnl(),
        'unrealized_pnl': self.get_unrealized_pnl(),
        'total_pnl': self.get_total_pnl(),
        'max_drawdown': self.max_drawdown,
        'max_drawdown_pct': self.max_drawdown * 100,
        'total_trades': self.total_trades,
        'winning_trades': self.winning_trades,
        'win_rate': win_rate,
        'win_rate_pct': win_rate * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe_ratio,
        'avg_holding_period_hours': avg_holding_period,
        'total_fees': self.total_fees,
        'active_positions': len(self.positions),
        'last_updated': datetime.now().isoformat()
    }
  
  def get_position_summary(self) -> pd.DataFrame:
    """Get current positions as DataFrame."""
    if not self.positions:
      return pd.DataFrame()
    
    positions_data = []
    for asset, position in self.positions.items():
      positions_data.append(position.to_dict())
    
    return pd.DataFrame(positions_data)
  
  def get_trade_history_df(self) -> pd.DataFrame:
    """Get trade history as DataFrame."""
    if not self.trade_history:
      return pd.DataFrame()
    
    return pd.DataFrame(self.trade_history)
  
  def get_closed_positions_df(self) -> pd.DataFrame:
    """Get closed positions as DataFrame."""
    if not self.closed_positions:
      return pd.DataFrame()
    
    return pd.DataFrame(self.closed_positions)
  
  def reset_portfolio(self):
    """Reset portfolio to initial state."""
    self.available_capital = self.initial_capital
    self.positions.clear()
    self.closed_positions.clear()
    self.trade_history.clear()
    self.total_trades = 0
    self.winning_trades = 0
    self.total_fees = 0.0
    self.max_drawdown = 0.0
    self.peak_value = self.initial_capital
    
    logger.info("Portfolio reset to initial state")
  
  def add_trade_record(self, trade_record: Dict):
    """Add a trade record to history."""
    self.trade_history.append(trade_record)
  
  def add_fees(self, fees: float):
    """Add trading fees to total."""
    self.total_fees += fees
    self.available_capital -= fees
  
  def export_performance_report(self, filepath: str):
    """Export detailed performance report."""
    try:
      metrics = self.get_performance_metrics()
      positions_df = self.get_position_summary()
      trades_df = self.get_trade_history_df()
      closed_df = self.get_closed_positions_df()
      
      with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Performance metrics
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_excel(writer, sheet_name='Performance', index=False)
        
        # Current positions
        if not positions_df.empty:
          positions_df.to_excel(writer, sheet_name='Positions', index=False)
        
        # Trade history
        if not trades_df.empty:
          trades_df.to_excel(writer, sheet_name='Trades', index=False)
        
        # Closed positions
        if not closed_df.empty:
          closed_df.to_excel(writer, sheet_name='Closed_Positions', index=False)
      
      logger.info(f"Performance report exported to {filepath}")
      
    except Exception as e:
      logger.error(f"Error exporting performance report: {e}")
