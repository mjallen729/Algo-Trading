"""
Performance metrics and evaluation utilities.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PerformanceMetrics:
    """Calculate trading performance metrics."""
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (focuses on downside risk).
        
        Args:
            returns: Series of returns
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sortino ratio
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf') if excess_returns.mean() > 0 else 0.0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    @staticmethod
    def max_drawdown(equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of portfolio values
            
        Returns:
            Maximum drawdown as percentage
        """
        if len(equity_curve) == 0:
            return 0.0
        
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min())
    
    @staticmethod
    def calmar_ratio(returns: pd.Series) -> float:
        """
        Calculate Calmar ratio (annual return / max drawdown).
        
        Args:
            returns: Series of returns
            
        Returns:
            Calmar ratio
        """
        if len(returns) == 0:
            return 0.0
        
        annual_return = returns.mean() * 252
        equity_curve = (1 + returns).cumprod()
        max_dd = PerformanceMetrics.max_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / max_dd
    
    @staticmethod
    def calculate_all_metrics(returns: pd.Series, 
                            equity_curve: pd.Series = None) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            returns: Series of returns
            equity_curve: Series of portfolio values
            
        Returns:
            Dictionary of performance metrics
        """
        if equity_curve is None:
            equity_curve = (1 + returns).cumprod()
        
        metrics = {
            'total_return': (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0] if len(equity_curve) > 0 else 0.0,
            'annual_return': returns.mean() * 252 if len(returns) > 0 else 0.0,
            'volatility': returns.std() * np.sqrt(252) if len(returns) > 0 else 0.0,
            'sharpe_ratio': PerformanceMetrics.sharpe_ratio(returns),
            'sortino_ratio': PerformanceMetrics.sortino_ratio(returns),
            'max_drawdown': PerformanceMetrics.max_drawdown(equity_curve),
            'calmar_ratio': PerformanceMetrics.calmar_ratio(returns),
            'win_rate': (returns > 0).mean() if len(returns) > 0 else 0.0,
            'profit_factor': abs(returns[returns > 0].sum() / returns[returns < 0].sum()) if len(returns[returns < 0]) > 0 else float('inf'),
            'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0.0,
            'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0.0,
        }
        
        return metrics
