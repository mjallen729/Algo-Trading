from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from src.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL


class Trader:
  def __init__(self, api_key: str, secret_key: str, base_url: str):
    self.trading_client = TradingClient(
      api_key, secret_key, paper=True, url_override=base_url)

  def place_market_order(self, symbol: str, quantity: float, side: str):
    # Ensure quantity is positive
    if quantity <= 0:
      print(f"Cannot place order with non-positive quantity: {quantity}")
      return None

    # Determine order side
    order_side = OrderSide.BUY if side.upper() == "BUY" else OrderSide.SELL

    market_order_data = MarketOrderRequest(
      symbol=symbol,
      qty=quantity,
      side=order_side,
      time_in_force=TimeInForce.GTC  # Good Til Canceled
    )

    try:
      # Place the order
      market_order = self.trading_client.submit_order(
        order_data=market_order_data)
      print(
        f"Successfully placed {side} order for {quantity} of {symbol}. Order ID: {market_order.id}")
      return market_order
    except Exception as e:
      print(f"Error placing order for {symbol}: {e}")
      return None

  def get_account_info(self):
    try:
      account = self.trading_client.get_account()
      return account
    except Exception as e:
      print(f"Error getting account info: {e}")
      return None

  def get_open_positions(self):
    try:
      positions = self.trading_client.get_all_positions()
      return positions
    except Exception as e:
      print(f"Error getting open positions: {e}")
      return None


if __name__ == '__main__':
  # Example Usage (requires ALPACA_API_KEY and ALPACA_SECRET_KEY in .env)
  if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    trader = Trader(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)

    # Get account info
    account_info = trader.get_account_info()
    if account_info:
      print(f"Account Equity: {account_info.equity}")
      print(f"Buying Power: {account_info.buying_power}")

    # Place a dummy buy order (e.g., 0.001 BTC)
    # Note: Alpaca requires crypto symbols in the format 'BTCUSD' not 'BTC/USD'
    # You might need to adjust symbol formatting based on Alpaca's requirements
    # For simplicity, using a common crypto symbol here.
    # Make sure you have enough buying power in your paper trading account.
    # buy_order = trader.place_market_order("BTCUSD", 0.001, "BUY")
    # if buy_order:
    #     print(f"Buy order status: {buy_order.status}")

    # Get open positions
    # open_positions = trader.get_open_positions()
    # if open_positions:
    #     print("\nOpen Positions:")
    #     for position in open_positions:
    #         print(f"Symbol: {position.symbol}, Quantity: {position.qty}, Market Value: {position.market_value}")

  else:
    print("Alpaca API keys not found in .env. Cannot run example.")
