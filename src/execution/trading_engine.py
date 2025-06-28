"""
Trading engine for executing trades and managing orders.
Core component that orchestrates strategy signals and portfolio management.
"""
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .broker import PaperTradingBroker
from .portfolio import Portfolio
from ..utils import get_logger, config
from ..risk import RiskManager, AdvancedPositionSizer
from ..strategies.base import TradingSignal

logger = get_logger(__name__)


@dataclass
class ExecutionResult:
  """Result of trade execution."""
  success: bool
  order_id: Optional[str]
  executed_price: Optional[float]
  executed_quantity: Optional[float]
  fees: float
  message: str


class TradingEngine:
  """
  Core trading engine that orchestrates signal execution.
  
  Features:
  - Signal-based trade execution
  - Risk management integration
  - Portfolio tracking and management
  - Performance monitoring
  - Paper trading support
  """
  
  def __init__(self, 
               initial_capital: float = 10000.0,
               execution_mode: str = 'paper'):
    """
    Initialize trading engine.
    
    Args:
        initial_capital: Starting capital amount
        execution_mode: 'paper' or 'live' trading mode
    """
    self.initial_capital = initial_capital
    self.execution_mode = execution_mode
    
    # Initialize components
    self.broker = PaperTradingBroker()
    self.portfolio = Portfolio(initial_capital)
    self.risk_manager = RiskManager()
    self.position_sizer = AdvancedPositionSizer()
    
    # State tracking
    self.is_initialized = False
    self.trade_history: List[Dict] = []
    self.active_orders: Dict[str, Dict] = {}
    
    logger.info(f"TradingEngine initialized in {execution_mode} mode with ${initial_capital:,.2f}")
  
  async def initialize(self):
    """Initialize all engine components."""
    try:
      logger.info("Initializing trading engine...")
      
      # Initialize broker
      await self.broker.initialize()
      
      self.is_initialized = True
      logger.info("Trading engine initialization completed")
      
    except Exception as e:
      logger.error(f"Failed to initialize trading engine: {e}")
      raise
  
  async def execute_signal(self, signal: TradingSignal) -> ExecutionResult:
    """
    Execute a trading signal.
    
    Args:
        signal: Trading signal to execute
    
    Returns:
        ExecutionResult with trade details
    """
    if not self.is_initialized:
      return ExecutionResult(
          success=False,
          order_id=None,
          executed_price=None,
          executed_quantity=None,
          fees=0.0,
          message="Trading engine not initialized"
      )
    
    try:
      logger.info(f"Executing signal: {signal.action} {signal.asset} @ {signal.price:.2f}")
      
      # Risk management check
      risk_check = self.risk_manager.validate_signal(signal)
      if not risk_check['approved']:
        return ExecutionResult(
            success=False,
            order_id=None,
            executed_price=None,
            executed_quantity=None,
            fees=0.0,
            message=f"Risk check failed: {risk_check['reason']}"
        )
      
      # Determine position size
      position_size = self.position_sizer.calculate_position_size(
          signal=signal,
          symbol=signal.asset,
          portfolio_value=self.portfolio.get_total_value(),
          current_volatility=signal.metadata.get('volatility', 0.02)
      )
      
      if position_size <= 0:
        return ExecutionResult(
            success=False,
            order_id=None,
            executed_price=None,
            executed_quantity=None,
            fees=0.0,
            message="Position size calculation resulted in zero or negative size"
        )
      
      # Execute based on signal action
      if signal.action.lower() == 'buy':
        result = await self._execute_buy_order(signal.asset, position_size, signal)
      elif signal.action.lower() == 'sell':
        result = await self._execute_sell_order(signal.asset, position_size, signal)
      else:
        return ExecutionResult(
            success=False,
            order_id=None,
            executed_price=None,
            executed_quantity=None,
            fees=0.0,
            message=f"Unknown signal action: {signal.action}"
        )
      
      # Update portfolio drawdown tracking
      self.portfolio.update_drawdown()
      
      return result
      
    except Exception as e:
      logger.error(f"Error executing signal: {e}")
      return ExecutionResult(
          success=False,
          order_id=None,
          executed_price=None,
          executed_quantity=None,
          fees=0.0,
          message=f"Execution error: {str(e)}"
      )
  
  async def _execute_buy_order(self, asset: str, position_size: float, signal: TradingSignal) -> ExecutionResult:
    """Execute a buy order."""
    try:
      # Calculate quantity based on position size
      quantity = position_size / signal.price
      
      # Place buy order
      order = await self.broker.place_order(
          symbol=asset,
          side='buy',
          quantity=quantity,
          price=signal.price,
          order_type='limit'
      )
      
      if order and order.get('status') == 'filled':
        # Update portfolio
        self.portfolio.add_position(
            asset=asset,
            quantity=quantity,
            entry_price=order['executed_price'],
            side='long'
        )
        
        # Add fees to portfolio
        self.portfolio.add_fees(order.get('fees', 0))
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now(),
            'asset': asset,
            'side': 'buy',
            'quantity': quantity,
            'price': order['executed_price'],
            'value': position_size,
            'fees': order.get('fees', 0),
            'signal_confidence': signal.confidence,
            'signal_metadata': signal.metadata,
            'order_id': order['id']
        }
        self.trade_history.append(trade_record)
        self.portfolio.add_trade_record(trade_record)
        
        logger.info(f"BUY order executed: {asset} @ {order['executed_price']:.2f} (qty: {quantity:.6f})")
        
        return ExecutionResult(
            success=True,
            order_id=order['id'],
            executed_price=order['executed_price'],
            executed_quantity=quantity,
            fees=order.get('fees', 0),
            message="Buy order executed successfully"
        )
      else:
        return ExecutionResult(
            success=False,
            order_id=order.get('id') if order else None,
            executed_price=None,
            executed_quantity=None,
            fees=0.0,
            message="Buy order failed to execute"
        )
        
    except Exception as e:
      logger.error(f"Error executing buy order for {asset}: {e}")
      return ExecutionResult(
          success=False,
          order_id=None,
          executed_price=None,
          executed_quantity=None,
          fees=0.0,
          message=f"Buy order error: {str(e)}"
      )
  
  async def _execute_sell_order(self, asset: str, position_size: float, signal: TradingSignal) -> ExecutionResult:
    """Execute a sell order."""
    try:
      # Check if we have a position to sell
      current_position = self.portfolio.get_position(asset)
      
      if current_position and current_position['quantity'] > 0:
        # Sell existing position
        quantity = current_position['quantity']
        
        # Place sell order
        order = await self.broker.place_order(
            symbol=asset,
            side='sell',
            quantity=quantity,
            price=signal.price,
            order_type='limit'
        )
        
        if order and order.get('status') == 'filled':
          # Close position in portfolio
          closed_position = self.portfolio.close_position(asset, order['executed_price'])
          
          # Add fees to portfolio
          self.portfolio.add_fees(order.get('fees', 0))
          
          # Record trade
          trade_record = {
              'timestamp': datetime.now(),
              'asset': asset,
              'side': 'sell',
              'quantity': quantity,
              'price': order['executed_price'],
              'value': quantity * order['executed_price'],
              'fees': order.get('fees', 0),
              'signal_confidence': signal.confidence,
              'signal_metadata': signal.metadata,
              'order_id': order['id'],
              'realized_pnl': closed_position['realized_pnl'] if closed_position else 0
          }
          self.trade_history.append(trade_record)
          self.portfolio.add_trade_record(trade_record)
          
          logger.info(f"SELL order executed: {asset} @ {order['executed_price']:.2f} (qty: {quantity:.6f})")
          
          return ExecutionResult(
              success=True,
              order_id=order['id'],
              executed_price=order['executed_price'],
              executed_quantity=quantity,
              fees=order.get('fees', 0),
              message="Sell order executed successfully"
          )
        else:
          return ExecutionResult(
              success=False,
              order_id=order.get('id') if order else None,
              executed_price=None,
              executed_quantity=None,
              fees=0.0,
              message="Sell order failed to execute"
          )
      else:
        return ExecutionResult(
            success=False,
            order_id=None,
            executed_price=None,
            executed_quantity=None,
            fees=0.0,
            message=f"No position to sell for {asset}"
        )
        
    except Exception as e:
      logger.error(f"Error executing sell order for {asset}: {e}")
      return ExecutionResult(
          success=False,
          order_id=None,
          executed_price=None,
          executed_quantity=None,
          fees=0.0,
          message=f"Sell order error: {str(e)}"
      )
  
  async def update_position_prices(self, price_data: Dict[str, float]):
    """Update current prices for all positions."""
    for asset, current_price in price_data.items():
      self.portfolio.update_position_price(asset, current_price)
  
  async def close_all_positions(self, current_prices: Optional[Dict[str, float]] = None):
    """Close all open positions."""
    logger.info("Closing all positions...")
    
    try:
      positions = self.portfolio.get_all_positions()
      
      for asset, position in positions.items():
        if position['quantity'] > 0:
          # Get current market price
          current_price = current_prices.get(asset) if current_prices else position['current_price']
          
          # Place sell order
          order = await self.broker.place_order(
              symbol=asset,
              side='sell',
              quantity=position['quantity'],
              price=current_price,
              order_type='market'
          )
          
          if order and order.get('status') == 'filled':
            self.portfolio.close_position(asset, order['executed_price'])
            self.portfolio.add_fees(order.get('fees', 0))
            logger.info(f"Closed position: {asset}")
      
      logger.info("All positions closed")
      
    except Exception as e:
      logger.error(f"Error closing positions: {e}")
  
  async def cancel_order(self, order_id: str) -> bool:
    """Cancel a pending order."""
    try:
      success = await self.broker.cancel_order(order_id)
      if success and order_id in self.active_orders:
        del self.active_orders[order_id]
      return success
    except Exception as e:
      logger.error(f"Error cancelling order {order_id}: {e}")
      return False
  
  async def shutdown(self):
    """Shutdown the trading engine."""
    logger.info("Shutting down trading engine...")
    
    try:
      # Close all positions if in paper trading mode
      if self.execution_mode == 'paper':
        await self.close_all_positions()
      
      # Shutdown broker connection
      await self.broker.shutdown()
      
      logger.info("Trading engine shutdown completed")
      
    except Exception as e:
      logger.error(f"Error during trading engine shutdown: {e}")
  
  def get_performance_report(self) -> Dict:
    """
    Generate comprehensive performance report.
    
    Returns:
        Dictionary with performance metrics
    """
    try:
      portfolio_metrics = self.portfolio.get_performance_metrics()
      
      # Trade statistics
      total_trades = len(self.trade_history)
      buy_trades = len([t for t in self.trade_history if t['side'] == 'buy'])
      sell_trades = len([t for t in self.trade_history if t['side'] == 'sell'])
      
      # Calculate win rate
      winning_trades = 0
      total_pnl = 0
      if sell_trades > 0:
        for trade in self.trade_history:
          if trade['side'] == 'sell' and 'realized_pnl' in trade:
            if trade['realized_pnl'] > 0:
              winning_trades += 1
            total_pnl += trade['realized_pnl']
      
      win_rate = winning_trades / sell_trades if sell_trades > 0 else 0
      
      # Average signal confidence
      avg_confidence = sum(t.get('signal_confidence', 0) for t in self.trade_history) / total_trades if total_trades > 0 else 0
      
      # Execution statistics
      execution_stats = self.broker.get_execution_stats()
      
      report = {
          'portfolio_metrics': portfolio_metrics,
          'trading_stats': {
              'total_trades': total_trades,
              'buy_trades': buy_trades,
              'sell_trades': sell_trades,
              'winning_trades': winning_trades,
              'win_rate': win_rate,
              'win_rate_pct': win_rate * 100,
              'avg_signal_confidence': avg_confidence,
              'total_realized_pnl': total_pnl
          },
          'execution_stats': execution_stats,
          'execution_mode': self.execution_mode,
          'report_timestamp': datetime.now().isoformat()
      }
      
      return report
      
    except Exception as e:
      logger.error(f"Error generating performance report: {e}")
      return {'error': str(e)}
  
  def get_status(self) -> Dict:
    """Get current trading engine status."""
    return {
        'is_initialized': self.is_initialized,
        'execution_mode': self.execution_mode,
        'total_trades': len(self.trade_history),
        'portfolio_value': self.portfolio.get_total_value(),
        'available_capital': self.portfolio.get_available_capital(),
        'active_positions': len(self.portfolio.get_all_positions()),
        'unrealized_pnl': self.portfolio.get_unrealized_pnl(),
        'total_return_pct': self.portfolio.get_total_return() * 100,
        'max_drawdown_pct': self.portfolio.max_drawdown * 100
    }
  
  def get_trade_history(self) -> List[Dict]:
    """Get complete trade history."""
    return self.trade_history.copy()
  
  def get_recent_trades(self, limit: int = 10) -> List[Dict]:
    """Get recent trades."""
    return self.trade_history[-limit:] if self.trade_history else []
  
  def get_active_positions(self) -> Dict[str, Dict]:
    """Get all active positions."""
    return self.portfolio.get_all_positions()
  
  def get_broker_stats(self) -> Dict:
    """Get broker execution statistics."""
    return self.broker.get_execution_stats()
  
  async def process_market_data(self, market_data: Dict[str, float]):
    """
    Process incoming market data and update positions.
    
    Args:
        market_data: Dictionary of {asset: current_price}
    """
    try:
      # Update position prices
      await self.update_position_prices(market_data)
      
      # Update portfolio drawdown
      self.portfolio.update_drawdown()
      
    except Exception as e:
      logger.error(f"Error processing market data: {e}")
  
  async def emergency_shutdown(self):
    """Emergency shutdown - close all positions immediately."""
    logger.warning("Emergency shutdown initiated!")
    
    try:
      positions = self.portfolio.get_all_positions()
      
      for asset, position in positions.items():
        if position['quantity'] > 0:
          # Place market order for immediate execution
          order = await self.broker.place_order(
              symbol=asset,
              side='sell',
              quantity=position['quantity'],
              price=position['current_price'],
              order_type='market'
          )
          
          if order and order.get('status') == 'filled':
            self.portfolio.close_position(asset, order['executed_price'])
            logger.warning(f"Emergency close: {asset}")
      
      await self.broker.shutdown()
      logger.warning("Emergency shutdown completed")
      
    except Exception as e:
      logger.error(f"Error during emergency shutdown: {e}")
