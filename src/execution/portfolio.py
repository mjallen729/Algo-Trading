d.DataFrame()
        
        positions_data = []
        for asset, position in self.positions.items():
            positions_data.append(position.to_dict())
        
        return pd.DataFrame(positions_data)
    
    def get_trade_history_df(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
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
    
    def export_performance_report(self, filepath: str):
        """Export detailed performance report."""
        try:
            metrics = self.get_performance_metrics()
            positions_df = self.get_position_summary()
            trades_df = self.get_trade_history_df()
            
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
                if self.closed_positions:
                    closed_df = pd.DataFrame(self.closed_positions)
                    closed_df.to_excel(writer, sheet_name='Closed_Positions', index=False)
            
            logger.info(f"Performance report exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
