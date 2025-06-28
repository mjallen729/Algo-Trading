    
    def _apply_confidence_adjustment(self, base_size: float, confidence: float) -> float:
        """
        Adjust position size based on signal confidence.
        
        Args:
            base_size: Base position size
            confidence: Signal confidence (0.0 to 1.0)
            
        Returns:
            Adjusted position size
        """
        # Confidence scaling (0.5x to 1.5x based on confidence)
        confidence_multiplier = 0.5 + confidence
        
        return base_size * confidence_multiplier
    
    def calculate_stop_loss_size(self, 
                                entry_price: float,
                                stop_loss_price: float,
                                risk_amount: float) -> float:
        """
        Calculate position size based on stop loss distance.
        
        Args:
            entry_price: Entry price
            stop_loss_price: Stop loss price
            risk_amount: Maximum risk amount in USD
            
        Returns:
            Position size (quantity)
        """
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0.0
        
        # Calculate risk per unit
        risk_per_unit = abs(entry_price - stop_loss_price)
        
        if risk_per_unit == 0:
            return 0.0
        
        # Calculate quantity based on risk
        quantity = risk_amount / risk_per_unit
        
        return quantity
    
    def get_sizing_metrics(self, signal: TradingSignal, portfolio_value: float) -> Dict:
        """
        Get detailed sizing metrics for analysis.
        
        Args:
            signal: Trading signal
            portfolio_value: Portfolio value
            
        Returns:
            Dictionary with sizing metrics
        """
        base_size = self.calculate_position_size(signal, "TEST", portfolio_value)
        
        metrics = {
            'method': self.method,
            'base_position_size': base_size,
            'position_percentage': (base_size / portfolio_value) * 100,
            'confidence': signal.confidence,
            'kelly_fraction_used': self.kelly_fraction,
            'max_position_limit': self.max_position_size * 100,
            'risk_amount': base_size  # Simplified - actual risk depends on stop loss
        }
        
        return metrics
