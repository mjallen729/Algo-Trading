"""
Paper trading broker for simulated trade execution.
"""
import asyncio
import uuid
from typing import Dict, Optional, List
from datetime import datetime
import random

from utils import get_logger, config

logger = get_logger(__name__)


class PaperTradingBroker:
  """
  Paper trading broker that simulates real trading without actual money.

  Features:
  - Simulated order execution
  - Realistic slippage and fees
  - Order book simulation
  - Market data integration
  """

  def __init__(self):
    """Initialize paper trading broker."""
    self.orders = {}
    self.executed_orders = []
    self.is_connected = False

    # Trading parameters
    self.slippage_range = config.get(
      'execution.slippage_tolerance', 0.001)  # 0.1%
    self.trading_fee = 0.001  # 0.1% trading fee
    self.execution_delay = 0.1  # 100ms execution delay

    logger.info("PaperTradingBroker initialized")

  async def initialize(self):
    """Initialize broker connection."""
    try:
      # Simulate connection delay
      await asyncio.sleep(0.5)

      self.is_connected = True
      logger.info("Paper trading broker connected")

    except Exception as e:
      logger.error(f"Failed to initialize paper broker: {e}")
      raise

  async def place_order(self,
                        symbol: str,
                        side: str,
                        quantity: float,
                        price: float,
                        order_type: str = 'limit') -> Optional[Dict]:
    """
    Place a trading order.

    Args:
        symbol: Trading symbol (e.g., 'BTC')
        side: 'buy' or 'sell'
        quantity: Order quantity
        price: Order price
        order_type: 'market' or 'limit'

    Returns:
        Order result dictionary
    """
    if not self.is_connected:
      logger.error("Broker not connected")
      return None

    try:
      # Generate order ID
      order_id = str(uuid.uuid4())

      # Create order
      order = {
          'id': order_id,
          'symbol': symbol,
          'side': side,
          'quantity': quantity,
          'price': price,
          'order_type': order_type,
          'status': 'pending',
          'timestamp': datetime.now(),
          'executed_price': None,
          'executed_quantity': None,
          'fees': 0.0
      }

      self.orders[order_id] = order

      # Simulate order execution
      executed_order = await self._execute_order(order)

      return executed_order

    except Exception as e:
      logger.error(f"Error placing order: {e}")
      return None

  async def _execute_order(self, order: Dict) -> Dict:
    """
    Simulate order execution with realistic delays and slippage.

    Args:
        order: Order to execute

    Returns:
        Executed order dictionary
    """
    # Simulate execution delay
    await asyncio.sleep(self.execution_delay)

    # Calculate slippage
    slippage = random.uniform(-self.slippage_range, self.slippage_range)
    executed_price = order['price'] * (1 + slippage)

    # For market orders, add more slippage
    if order['order_type'] == 'market':
      additional_slippage = random.uniform(0, self.slippage_range)
      if order['side'] == 'buy':
        executed_price *= (1 + additional_slippage)
      else:
        executed_price *= (1 - additional_slippage)

    # Calculate fees
    trade_value = order['quantity'] * executed_price
    fees = trade_value * self.trading_fee

    # Update order
    order.update({
        'status': 'filled',
        'executed_price': executed_price,
        'executed_quantity': order['quantity'],
        'fees': fees,
        'execution_time': datetime.now()
    })

    # Store executed order
    self.executed_orders.append(order.copy())

    logger.info(f"Order executed: {order['side'].upper()} {order['quantity']:.6f} {order['symbol']} "
                f"@ {executed_price:.2f} (slippage: {slippage * 100:.3f}%, fees: ${fees:.2f})")

    return order

  async def cancel_order(self, order_id: str) -> bool:
    """
    Cancel a pending order.

    Args:
        order_id: Order ID to cancel

    Returns:
        True if cancelled successfully
    """
    if order_id in self.orders:
      order = self.orders[order_id]
      if order['status'] == 'pending':
        order['status'] = 'cancelled'
        logger.info(f"Order {order_id} cancelled")
        return True

    return False

  def get_order_status(self, order_id: str) -> Optional[Dict]:
    """
    Get order status.

    Args:
        order_id: Order ID

    Returns:
        Order status dictionary
    """
    return self.orders.get(order_id)

  def get_all_orders(self) -> List[Dict]:
    """Get all orders."""
    return list(self.orders.values())

  def get_executed_orders(self) -> List[Dict]:
    """Get all executed orders."""
    return self.executed_orders.copy()

  def get_trading_fees(self) -> float:
    """Get total trading fees paid."""
    return sum(order.get('fees', 0) for order in self.executed_orders)

  async def get_account_balance(self) -> Dict:
    """
    Get simulated account balance.

    Returns:
        Account balance dictionary
    """
    # Calculate total fees paid
    total_fees = self.get_trading_fees()

    # Calculate P&L from executed trades
    total_trades_value = 0
    for order in self.executed_orders:
      trade_value = order['executed_quantity'] * order['executed_price']
      if order['side'] == 'buy':
        total_trades_value -= trade_value
      else:
        total_trades_value += trade_value

    # Starting capital minus fees and trade costs
    initial_capital = config.get('trading.initial_capital', 10000)
    available_balance = initial_capital + total_trades_value - total_fees

    return {
        'currency': 'USD',
        'available_balance': available_balance,
        'total_fees_paid': total_fees,
        'initial_capital': initial_capital,
        'unrealized_pnl': 0.0,  # Simplified for paper trading
        'realized_pnl': total_trades_value
    }

  async def get_positions(self) -> Dict:
    """
    Get current positions.

    Returns:
        Dictionary of current positions
    """
    positions = {}

    # Calculate positions from executed orders
    for order in self.executed_orders:
      symbol = order['symbol']

      if symbol not in positions:
        positions[symbol] = {
            'quantity': 0.0,
            'avg_price': 0.0,
            'unrealized_pnl': 0.0
        }

      if order['side'] == 'buy':
        # Add to position
        old_quantity = positions[symbol]['quantity']
        old_avg_price = positions[symbol]['avg_price']
        new_quantity = order['executed_quantity']
        new_price = order['executed_price']

        total_quantity = old_quantity + new_quantity
        if total_quantity > 0:
          positions[symbol]['avg_price'] = (
              (old_quantity * old_avg_price +
               new_quantity * new_price) / total_quantity
          )
          positions[symbol]['quantity'] = total_quantity

      else:  # sell
        positions[symbol]['quantity'] -= order['executed_quantity']

        # Remove position if quantity is effectively zero
        if abs(positions[symbol]['quantity']) < 1e-8:
          positions[symbol]['quantity'] = 0.0

    # Filter out zero positions
    active_positions = {k: v for k,
                        v in positions.items() if v['quantity'] > 0}

    return active_positions

  async def shutdown(self):
    """Shutdown broker connection."""
    self.is_connected = False
    logger.info("Paper trading broker disconnected")

  def get_execution_stats(self) -> Dict:
    """Get execution statistics."""
    if not self.executed_orders:
      return {
          'total_orders': 0,
          'avg_slippage': 0.0,
          'total_fees': 0.0,
          'avg_execution_time': 0.0
      }

    total_orders = len(self.executed_orders)
    total_fees = sum(order.get('fees', 0) for order in self.executed_orders)

    # Calculate average slippage
    slippages = []
    for order in self.executed_orders:
      if order.get('executed_price') and order.get('price'):
        slippage = (order['executed_price'] - order['price']) / order['price']
        slippages.append(abs(slippage))

    avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0

    return {
        'total_orders': total_orders,
        'avg_slippage': avg_slippage,
        'total_fees': total_fees,
        'avg_execution_time': self.execution_delay,
        'buy_orders': len([o for o in self.executed_orders if o['side'] == 'buy']),
        'sell_orders': len([o for o in self.executed_orders if o['side'] == 'sell'])
    }
