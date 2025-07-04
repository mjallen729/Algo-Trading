import pandas as pd
from src.config import RISK_PER_TRADE

class PortfolioManager:
    def __init__(self, initial_capital: float):
        self.capital = initial_capital
        self.positions = {}
        self.trade_history = []

    def get_current_capital(self) -> float:
        return self.capital

    def get_position_size(self, current_price: float) -> float:
        # Fixed fractional position sizing
        # Risk a percentage of total capital per trade
        # For simplicity, assuming 1 unit of crypto is traded
        # This needs to be refined for actual crypto trading (e.g., fractional shares)
        
        # Calculate the amount of capital to risk
        risk_amount = self.capital * RISK_PER_TRADE
        
        # Determine the number of units to trade based on risk and current price
        # This is a simplified example. In a real scenario, you'd consider stop-loss
        # and target prices to calculate position size more accurately.
        if current_price > 0:
            # For now, let's just return a fixed percentage of capital as the value of the position
            # This will be the dollar value of the position to take
            return risk_amount * 10 # Example: risk 1% to gain 10% of that risk
        return 0.0

    def record_trade(self, symbol: str, trade_type: str, quantity: float, price: float):
        trade_value = quantity * price
        if trade_type == "BUY":
            self.capital -= trade_value
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
        elif trade_type == "SELL":
            self.capital += trade_value
            self.positions[symbol] = self.positions.get(symbol, 0) - quantity
        
        self.trade_history.append({
            'symbol': symbol,
            'type': trade_type,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'timestamp': pd.Timestamp.now()
        })
        print(f"Trade recorded: {trade_type} {quantity} of {symbol} at {price}. Current Capital: {self.capital:.2f}")

    def check_risk(self, signal: str, current_price: float, symbol: str) -> bool:
        # Implement risk checks here
        # For MVP, a simple check if we have enough capital to buy or position to sell
        
        if signal == "BUY":
            # Check if we have enough capital to open a position
            # This is a very basic check, needs to be more sophisticated
            position_value = self.get_position_size(current_price)
            if self.capital >= position_value:
                print(f"Risk check passed for BUY signal on {symbol}. Position value: {position_value:.2f}")
                return True
            else:
                print(f"Risk check failed for BUY signal on {symbol}: Insufficient capital. Needed {position_value:.2f}, Have {self.capital:.2f}")
                return False
        elif signal == "SELL":
            # Check if we have a position to sell
            if self.positions.get(symbol, 0) > 0:
                print(f"Risk check passed for SELL signal on {symbol}. Current position: {self.positions.get(symbol, 0)}")
                return True
            else:
                print(f"Risk check failed for SELL signal on {symbol}: No position to sell.")
                return False
        elif signal == "HOLD":
            print(f"Risk check passed for HOLD signal on {symbol}.")
            return True
        return False

if __name__ == '__main__':
    # Example Usage
    portfolio_manager = PortfolioManager(initial_capital=10000.0)
    print(f"Initial Capital: {portfolio_manager.get_current_capital()}")

    # Simulate a BUY signal
    symbol = "BTC/USD"
    current_price = 30000.0
    if portfolio_manager.check_risk("BUY", current_price, symbol):
        position_value = portfolio_manager.get_position_size(current_price)
        # For simplicity, let's assume we buy 0.001 BTC
        quantity_to_buy = 0.001 # This should be calculated based on position_value and current_price
        portfolio_manager.record_trade(symbol, "BUY", quantity_to_buy, current_price)

    # Simulate a SELL signal
    if portfolio_manager.check_risk("SELL", current_price, symbol):
        quantity_to_sell = 0.0005 # Sell half of the position
        portfolio_manager.record_trade(symbol, "SELL", quantity_to_sell, current_price)

    print(f"Final Capital: {portfolio_manager.get_current_capital()}")
    print("Trade History:")
    for trade in portfolio_manager.trade_history:
        print(trade)
