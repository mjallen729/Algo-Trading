"""
Quick test script to verify the execution module functionality.
"""
import asyncio
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.execution import TradingEngine, Portfolio, PaperTradingBroker
from src.strategies.base import TradingSignal, SignalType
from datetime import datetime


async def test_execution_module():
  """Test the execution module components."""
  print("🚀 Testing Execution Module Components...")
  
  # Test Portfolio
  print("\n1. Testing Portfolio...")
  portfolio = Portfolio(initial_capital=10000.0)
  
  # Add a position
  success = portfolio.add_position(
      asset="BTC",
      quantity=0.1,
      entry_price=50000.0,
      side="long"
  )
  print(f"   Added BTC position: {success}")
  
  # Get performance metrics
  metrics = portfolio.get_performance_metrics()
  print(f"   Portfolio value: ${metrics['total_value']:,.2f}")
  print(f"   Available capital: ${metrics['available_capital']:,.2f}")
  
  # Test Broker
  print("\n2. Testing Paper Trading Broker...")
  broker = PaperTradingBroker()
  await broker.initialize()
  
  # Place a test order
  order = await broker.place_order(
      symbol="ETH",
      side="buy",
      quantity=1.0,
      price=3000.0,
      order_type="limit"
  )
  print(f"   Order placed: {order['status']}")
  print(f"   Executed at: ${order['executed_price']:,.2f}")
  
  # Test Trading Engine
  print("\n3. Testing Trading Engine...")
  engine = TradingEngine(initial_capital=10000.0)
  await engine.initialize()
  
  # Create a test signal
  signal = TradingSignal(
      signal_type=SignalType.BUY,
      confidence=0.8,
      price=50000.0,
      quantity=0.0,
      metadata={"strategy": "test", "asset": "BTC"}
  )
  
  # Add compatibility properties for trading engine
  signal.action = "buy"
  signal.asset = "BTC"
  
  # Execute the signal
  result = await engine.execute_signal(signal)
  print(f"   Signal execution: {result.success}")
  print(f"   Message: {result.message}")
  
  # Get status
  status = engine.get_status()
  print(f"   Engine initialized: {status['is_initialized']}")
  print(f"   Total trades: {status['total_trades']}")
  print(f"   Portfolio value: ${status['portfolio_value']:,.2f}")
  
  # Cleanup
  await engine.shutdown()
  print("\n✅ All execution module tests completed successfully!")


if __name__ == "__main__":
  asyncio.run(test_execution_module())
