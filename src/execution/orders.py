"""
Order management and execution utilities.
Handles different order types and execution strategies.
"""
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..utils import get_logger

logger = get_logger(__name__)


class OrderType(Enum):
  """Order type enumeration."""
  MARKET = "market"
  LIMIT = "limit"
  STOP = "stop"
  STOP_LIMIT = "stop_limit"
  TRAILING_STOP = "trailing_stop"
  ICEBERG = "iceberg"


class OrderStatus(Enum):
  """Order status enumeration."""
  PENDING = "pending"
  OPEN = "open"
  FILLED = "filled"
  PARTIALLY_FILLED = "partially_filled"
  CANCELLED = "cancelled"
  REJECTED = "rejected"
  EXPIRED = "expired"


class TimeInForce(Enum):
  """Time in force enumeration."""
  GTC = "gtc"  # Good Till Cancelled
  IOC = "ioc"  # Immediate or Cancel
  FOK = "fok"  # Fill or Kill
  DAY = "day"  # Day order


@dataclass
class Order:
  """Represents a trading order."""
  id: str
  symbol: str
  side: str  # 'buy' or 'sell'
  quantity: float
  order_type: OrderType
  price: Optional[float] = None
  stop_price: Optional[float] = None
  time_in_force: TimeInForce = TimeInForce.GTC
  status: OrderStatus = OrderStatus.PENDING
  created_at: datetime = None
  updated_at: datetime = None
  filled_quantity: float = 0.0
  avg_fill_price: Optional[float] = None
  fees: float = 0.0
  metadata: Optional[Dict] = None
  
  def __post_init__(self):
    if self.created_at is None:
      self.created_at = datetime.now()
    self.updated_at = self.created_at
  
  @property
  def remaining_quantity(self) -> float:
    """Get remaining quantity to be filled."""
    return self.quantity - self.filled_quantity
  
  @property
  def is_active(self) -> bool:
    """Check if order is still active."""
    return self.status in [OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED]
  
  @property
  def is_filled(self) -> bool:
    """Check if order is completely filled."""
    return self.status == OrderStatus.FILLED
  
  @property
  def fill_percentage(self) -> float:
    """Get fill percentage."""
    return (self.filled_quantity / self.quantity) * 100 if self.quantity > 0 else 0
  
  def update_fill(self, fill_quantity: float, fill_price: float, fees: float = 0.0):
    """Update order with partial or complete fill."""
    self.filled_quantity += fill_quantity
    self.fees += fees
    self.updated_at = datetime.now()
    
    # Update average fill price
    if self.avg_fill_price is None:
      self.avg_fill_price = fill_price
    else:
      total_filled_value = (self.filled_quantity - fill_quantity) * self.avg_fill_price + fill_quantity * fill_price
      self.avg_fill_price = total_filled_value / self.filled_quantity
    
    # Update status
    if self.filled_quantity >= self.quantity:
      self.status = OrderStatus.FILLED
    elif self.filled_quantity > 0:
      self.status = OrderStatus.PARTIALLY_FILLED
  
  def cancel(self):
    """Cancel the order."""
    if self.is_active:
      self.status = OrderStatus.CANCELLED
      self.updated_at = datetime.now()
  
  def to_dict(self) -> Dict:
    """Convert order to dictionary."""
    return {
        'id': self.id,
        'symbol': self.symbol,
        'side': self.side,
        'quantity': self.quantity,
        'order_type': self.order_type.value,
        'price': self.price,
        'stop_price': self.stop_price,
        'time_in_force': self.time_in_force.value,
        'status': self.status.value,
        'created_at': self.created_at.isoformat(),
        'updated_at': self.updated_at.isoformat(),
        'filled_quantity': self.filled_quantity,
        'remaining_quantity': self.remaining_quantity,
        'avg_fill_price': self.avg_fill_price,
        'fees': self.fees,
        'fill_percentage': self.fill_percentage,
        'is_active': self.is_active,
        'is_filled': self.is_filled,
        'metadata': self.metadata
    }


class OrderManager:
  """
  Manages orders and execution strategies.
  
  Features:
  - Order validation and lifecycle management
  - Advanced order types (TWAP, VWAP, Iceberg)
  - Execution algorithms
  - Slippage and fee estimation
  """
  
  def __init__(self):
    """Initialize order manager."""
    self.orders: Dict[str, Order] = {}
    self.execution_history: List[Dict] = []
    
    logger.info("OrderManager initialized")
  
  def create_order(self,
                   symbol: str,
                   side: str,
                   quantity: float,
                   order_type: OrderType = OrderType.MARKET,
                   price: Optional[float] = None,
                   stop_price: Optional[float] = None,
                   time_in_force: TimeInForce = TimeInForce.GTC,
                   metadata: Optional[Dict] = None) -> Order:
    """Create a new order."""
    # Generate unique order ID
    order_id = f"ord_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.orders)}"
    
    # Validate order parameters
    if not self._validate_order_params(symbol, side, quantity, order_type, price, stop_price):
      raise ValueError("Invalid order parameters")
    
    # Create order
    order = Order(
        id=order_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=order_type,
        price=price,
        stop_price=stop_price,
        time_in_force=time_in_force,
        metadata=metadata
    )
    
    self.orders[order_id] = order
    logger.info(f"Created order: {order_id} - {side.upper()} {quantity} {symbol}")
    
    return order
  
  def _validate_order_params(self,
                             symbol: str,
                             side: str,
                             quantity: float,
                             order_type: OrderType,
                             price: Optional[float],
                             stop_price: Optional[float]) -> bool:
    """Validate order parameters."""
    # Basic validation
    if not symbol or side not in ['buy', 'sell'] or quantity <= 0:
      return False
    
    # Price validation for limit orders
    if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and (price is None or price <= 0):
      return False
    
    # Stop price validation
    if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TRAILING_STOP] and (stop_price is None or stop_price <= 0):
      return False
    
    return True
  
  def get_order(self, order_id: str) -> Optional[Order]:
    """Get order by ID."""
    return self.orders.get(order_id)
  
  def get_active_orders(self, symbol: Optional[str] = None) -> List[Order]:
    """Get all active orders, optionally filtered by symbol."""
    active_orders = [order for order in self.orders.values() if order.is_active]
    
    if symbol:
      active_orders = [order for order in active_orders if order.symbol == symbol]
    
    return active_orders
  
  def cancel_order(self, order_id: str) -> bool:
    """Cancel an order."""
    order = self.get_order(order_id)
    if order and order.is_active:
      order.cancel()
      logger.info(f"Cancelled order: {order_id}")
      return True
    return False
  
  def get_execution_statistics(self) -> Dict:
    """Get execution statistics."""
    if not self.orders:
      return {
          'total_orders': 0,
          'filled_orders': 0,
          'cancelled_orders': 0,
          'fill_rate': 0.0,
          'total_fees': 0.0
      }
    
    filled_orders = [o for o in self.orders.values() if o.is_filled]
    cancelled_orders = [o for o in self.orders.values() if o.status == OrderStatus.CANCELLED]
    
    total_fees = sum(o.fees for o in self.orders.values())
    fill_rate = len(filled_orders) / len(self.orders) if self.orders else 0
    
    return {
        'total_orders': len(self.orders),
        'filled_orders': len(filled_orders),
        'cancelled_orders': len(cancelled_orders),
        'fill_rate': fill_rate,
        'total_fees': total_fees
    }
