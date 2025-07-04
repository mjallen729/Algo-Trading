import pandas as pd
from src.config import RISK_PER_TRADE, STOP_LOSS_PCT, INITIAL_CAPITAL

class PortfolioManager:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.capital = initial_capital
        self.positions = {}  # e.g., {"ETH/USD": {"quantity": 0.1, "entry_price": 3000, "stop_loss": 2940}}
        self.trade_history = []

    def get_current_capital(self) -> float:
        return self.capital

    def calculate_position_size(self) -> float:
        """
        Calculates the position size in dollars based on fixed fractional risk.
        """
        if STOP_LOSS_PCT <= 0:
            return 0.0
        return (self.capital * RISK_PER_TRADE) / STOP_LOSS_PCT

    def calculate_quantity(self, current_price: float) -> float:
        """
        Calculates the quantity of the asset to trade.
        """
        if current_price <= 0:
            return 0.0
        
        position_size_dollars = self.calculate_position_size()
        return position_size_dollars / current_price

    def record_trade(self, symbol: str, trade_type: str, quantity: float, price: float):
        trade_value = quantity * price
        
        if trade_type.upper() in ["BUY", "STRONG_BUY"]:
            self.capital -= trade_value
            stop_loss_price = price * (1 - STOP_LOSS_PCT)
            
            if symbol in self.positions:
                current_quantity = self.positions[symbol]["quantity"]
                current_value = self.positions[symbol]["entry_price"] * current_quantity
                
                new_quantity = current_quantity + quantity
                new_total_value = current_value + trade_value
                new_entry_price = new_total_value / new_quantity
                
                self.positions[symbol]["quantity"] = new_quantity
                self.positions[symbol]["entry_price"] = new_entry_price
                self.positions[symbol]["stop_loss"] = new_entry_price * (1 - STOP_LOSS_PCT)
            else:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "entry_price": price,
                    "stop_loss": stop_loss_price
                }

        elif trade_type.upper() in ["SELL", "STRONG_SELL"]:
            self.capital += trade_value
            if symbol in self.positions:
                self.positions[symbol]["quantity"] -= quantity
                if self.positions[symbol]["quantity"] <= 1e-9: # Use a small threshold for float comparison
                    del self.positions[symbol]
            else:
                print(f"Warning: Attempted to sell {symbol} without an open position.")

        self.trade_history.append({
            'symbol': symbol,
            'type': trade_type,
            'quantity': quantity,
            'price': price,
            'value': trade_value,
            'timestamp': pd.Timestamp.now()
        })
        print(f"Trade recorded: {trade_type} {quantity:.6f} of {symbol} at {price:.2f}. Current Capital: {self.capital:.2f}")

    def check_risk(self, signal: str, current_price: float) -> bool:
        if signal.upper() in ["BUY", "STRONG_BUY"]:
            # Prevent buying if already holding a position
            if self.get_open_positions():
                print("Risk check failed: A position is already open.")
                return False
            position_size = self.calculate_position_size()
            if self.capital < position_size:
                print(f"Risk check failed: Insufficient capital. Need {position_size:.2f}, have {self.capital:.2f}")
                return False
            return True
        
        elif signal.upper() in ["SELL", "STRONG_SELL"]:
            if not self.get_open_positions():
                 print("Risk check failed: No open positions to sell.")
                 return False
            return True
            
        return True # For HOLD signal

    def check_stop_loss(self, current_price: float, symbol: str) -> float:
        if symbol in self.positions:
            position = self.positions[symbol]
            if current_price < position["stop_loss"]:
                print(f"STOP-LOSS TRIGGERED for {symbol} at price {current_price:.2f} (Stop: {position['stop_loss']:.2f})")
                return position["quantity"]
        return 0.0

    def get_open_positions(self) -> dict:
        return self.positions
