"""
Arbitrage trading strategy.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

from .base import BaseStrategy, TradingSignal, SignalType
from ..utils import get_logger, config

logger = get_logger(__name__)


class ArbitrageStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy focusing on price inefficiencies.
    
    This strategy identifies:
    - Statistical arbitrage opportunities between correlated assets
    - Price discrepancies that are likely to converge
    - Short-term inefficiencies in price discovery
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize arbitrage strategy.
        
        Args:
            config: Strategy configuration parameters
        """
        default_config = {
            'min_spread': config.get('strategies.arbitrage.min_spread', 0.001),
            'max_execution_time': config.get('strategies.arbitrage.max_execution_time', 5),
            'correlation_threshold': 0.7,
            'spread_z_threshold': 2.0,
            'lookback_period': 100,
            'min_confidence': 0.6
        }
        
        if config:
            default_config.update(config)
        
        super().__init__("Arbitrage", default_config)
        
        self.min_spread = self.config['min_spread']
        self.correlation_threshold = self.config['correlation_threshold']
        self.spread_z_threshold = self.config['spread_z_threshold']
        
        # Store price history for correlation analysis
        self.price_history = {}
        
        logger.info(f"ArbitrageStrategy initialized with min spread: {self.min_spread}")
    
    def generate_signal(self, 
                       data: pd.DataFrame, 
                       current_price: float,
                       regime: str = None,
                       reference_data: Dict[str, pd.DataFrame] = None,
                       **kwargs) -> TradingSignal:
        """
        Generate arbitrage trading signal.
        
        Args:
            data: Historical market data for primary asset
            current_price: Current asset price
            regime: Current market regime
            reference_data: Data for reference/correlated assets
            
        Returns:
            TradingSignal object
        """
        if len(data) < self.config['lookback_period']:
            return TradingSignal(SignalType.HOLD, 0.0, current_price)
        
        # Store current price data
        asset_symbol = kwargs.get('symbol', 'UNKNOWN')
        self._update_price_history(asset_symbol, data)
        
        # Statistical arbitrage signals
        stat_arb_signal = self._statistical_arbitrage_signal(
            data, current_price, reference_data
        )
        
        # Mean reversion of spreads
        spread_reversion_signal = self._spread_mean_reversion_signal(
            data, reference_data
        )
        
        # Price inefficiency detection
        inefficiency_signal = self._detect_price_inefficiencies(data, current_price)
        
        # Combine signals
        signal_type = SignalType.HOLD
        confidence = 0.0
        
        # Determine overall arbitrage signal
        total_signal_strength = (
            0.4 * stat_arb_signal +
            0.4 * spread_reversion_signal +
            0.2 * inefficiency_signal
        )
        
        if total_signal_strength > 0.6:
            signal_type = SignalType.BUY
            confidence = min(0.95, total_signal_strength)
        elif total_signal_strength < -0.6:
            signal_type = SignalType.SELL
            confidence = min(0.95, abs(total_signal_strength))
        
        # Regime adjustments
        if regime:
            confidence = self._adjust_confidence_for_regime(confidence, signal_type, regime)
        
        metadata = {
            'stat_arb_signal': stat_arb_signal,
            'spread_reversion_signal': spread_reversion_signal,
            'inefficiency_signal': inefficiency_signal,
            'total_signal_strength': total_signal_strength,
            'regime': regime
        }
        
        signal = TradingSignal(signal_type, confidence, current_price, metadata=metadata)
        
        if self.validate_signal(signal):
            self.update_performance_metrics(signal)
        
        return signal
    
    def _update_price_history(self, symbol: str, data: pd.DataFrame):
        """Update price history for correlation analysis."""
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        # Keep only recent history
        max_history = self.config['lookback_period']
        current_prices = data['Close'].tail(max_history).tolist()
        self.price_history[symbol] = current_prices
    
    def _statistical_arbitrage_signal(self, 
                                    data: pd.DataFrame, 
                                    current_price: float,
                                    reference_data: Dict[str, pd.DataFrame] = None) -> float:
        """
        Calculate statistical arbitrage signal based on asset correlations.
        
        Args:
            data: Primary asset data
            current_price: Current price
            reference_data: Reference asset data
            
        Returns:
            Signal strength (-1 to 1)
        """
        if not reference_data:
            return 0.0
        
        signals = []
        
        for ref_symbol, ref_data in reference_data.items():
            if len(ref_data) < 50:
                continue
            
            # Calculate correlation
            common_length = min(len(data), len(ref_data))
            if common_length < 50:
                continue
            
            asset_returns = data['Close'].tail(common_length).pct_change().dropna()
            ref_returns = ref_data['Close'].tail(common_length).pct_change().dropna()
            
            correlation = asset_returns.corr(ref_returns)
            
            # Only consider highly correlated assets
            if abs(correlation) < self.correlation_threshold:
                continue
            
            # Calculate spread
            spread = self._calculate_normalized_spread(
                data['Close'].tail(common_length),
                ref_data['Close'].tail(common_length)
            )
            
            # Mean reversion signal on spread
            spread_mean = spread.mean()
            spread_std = spread.std()
            current_spread = spread.iloc[-1]
            
            if spread_std > 0:
                z_score = (current_spread - spread_mean) / spread_std
                
                # Generate mean reversion signal
                if abs(z_score) > self.spread_z_threshold:
                    if correlation > 0:
                        # Positive correlation: expect convergence
                        signal_strength = -np.sign(z_score) * min(1.0, abs(z_score) / 3)
                    else:
                        # Negative correlation: expect divergence
                        signal_strength = np.sign(z_score) * min(1.0, abs(z_score) / 3)
                    
                    signals.append(signal_strength)
        
        return np.mean(signals) if signals else 0.0
    
    def _calculate_normalized_spread(self, prices1: pd.Series, prices2: pd.Series) -> pd.Series:
        """
        Calculate normalized spread between two price series.
        
        Args:
            prices1: First price series
            prices2: Second price series
            
        Returns:
            Normalized spread series
        """
        # Log price ratio
        spread = np.log(prices1 / prices2)
        
        # Normalize by removing trend
        spread_detrended = spread - spread.rolling(20).mean()
        
        return spread_detrended.dropna()
    
    def _spread_mean_reversion_signal(self, 
                                    data: pd.DataFrame,
                                    reference_data: Dict[str, pd.DataFrame] = None) -> float:
        """
        Calculate mean reversion signal for price spreads.
        
        Args:
            data: Primary asset data
            reference_data: Reference asset data
            
        Returns:
            Mean reversion signal (-1 to 1)
        """
        if not reference_data:
            return 0.0
        
        reversion_signals = []
        
        for ref_symbol, ref_data in reference_data.items():
            common_length = min(len(data), len(ref_data))
            if common_length < 50:
                continue
            
            # Calculate price ratio
            ratio = (data['Close'].tail(common_length) / 
                    ref_data['Close'].tail(common_length))
            
            # Calculate mean and standard deviation
            ratio_mean = ratio.rolling(30).mean()
            ratio_std = ratio.rolling(30).std()
            
            # Current ratio relative to mean
            current_ratio = ratio.iloc[-1]
            current_mean = ratio_mean.iloc[-1]
            current_std = ratio_std.iloc[-1]
            
            if pd.notna(current_mean) and pd.notna(current_std) and current_std > 0:
                z_score = (current_ratio - current_mean) / current_std
                
                # Mean reversion signal
                if abs(z_score) > 1.5:
                    signal = -np.sign(z_score) * min(1.0, abs(z_score) / 3)
                    reversion_signals.append(signal)
        
        return np.mean(reversion_signals) if reversion_signals else 0.0
    
    def _detect_price_inefficiencies(self, data: pd.DataFrame, current_price: float) -> float:
        """
        Detect short-term price inefficiencies.
        
        Args:
            data: Market data
            current_price: Current price
            
        Returns:
            Inefficiency signal (-1 to 1)
        """
        if len(data) < 20:
            return 0.0
        
        # Short-term price momentum vs long-term trend
        short_ma = data['Close'].rolling(5).mean().iloc[-1]
        long_ma = data['Close'].rolling(20).mean().iloc[-1]
        
        # Price acceleration
        recent_returns = data['Close'].pct_change().tail(3)
        acceleration = recent_returns.diff().mean()
        
        # Volume-price divergence
        price_change = data['Close'].pct_change().tail(5)
        volume_change = data['Volume'].pct_change().tail(5)
        
        # Correlation between price and volume changes
        if len(price_change.dropna()) > 3 and len(volume_change.dropna()) > 3:
            pv_correlation = price_change.corr(volume_change)
        else:
            pv_correlation = 0
        
        # Combine inefficiency signals
        trend_inefficiency = 0
        if pd.notna(short_ma) and pd.notna(long_ma):
            if current_price > short_ma > long_ma:
                # Potential overextension to upside
                trend_inefficiency = -0.3
            elif current_price < short_ma < long_ma:
                # Potential overextension to downside
                trend_inefficiency = 0.3
        
        # Acceleration inefficiency
        accel_inefficiency = -np.sign(acceleration) * min(0.5, abs(acceleration) * 1000)
        
        # Volume divergence inefficiency
        volume_inefficiency = 0
        if pd.notna(pv_correlation) and abs(pv_correlation) < 0.3:
            # Low correlation suggests potential inefficiency
            volume_inefficiency = -np.sign(price_change.mean()) * 0.2
        
        total_inefficiency = (
            0.4 * trend_inefficiency +
            0.4 * accel_inefficiency +
            0.2 * volume_inefficiency
        )
        
        return np.clip(total_inefficiency, -1.0, 1.0)
    
    def _adjust_confidence_for_regime(self, confidence: float, 
                                     signal_type: SignalType, regime: str) -> float:
        """Adjust confidence based on market regime."""
        # Arbitrage works better in stable/ranging markets
        if "Ranging" in regime:
            confidence = min(0.95, confidence * 1.2)
        elif "Trending" in regime:
            # Reduce confidence in strongly trending markets
            confidence = confidence * 0.8
        
        return max(0.0, confidence)
    
    def calculate_arbitrage_metrics(self, 
                                  data: pd.DataFrame,
                                  reference_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate arbitrage opportunity metrics.
        
        Args:
            data: Primary asset data
            reference_data: Reference asset data
            
        Returns:
            Dictionary with arbitrage metrics
        """
        metrics = {
            'correlations': {},
            'spread_z_scores': {},
            'arbitrage_opportunities': 0
        }
        
        for ref_symbol, ref_data in reference_data.items():
            common_length = min(len(data), len(ref_data))
            if common_length < 50:
                continue
            
            # Calculate correlation
            asset_returns = data['Close'].tail(common_length).pct_change().dropna()
            ref_returns = ref_data['Close'].tail(common_length).pct_change().dropna()
            correlation = asset_returns.corr(ref_returns)
            
            metrics['correlations'][ref_symbol] = correlation
            
            # Calculate current spread z-score
            spread = self._calculate_normalized_spread(
                data['Close'].tail(common_length),
                ref_data['Close'].tail(common_length)
            )
            
            if len(spread) > 0:
                current_z = (spread.iloc[-1] - spread.mean()) / spread.std()
                metrics['spread_z_scores'][ref_symbol] = current_z
                
                # Count arbitrage opportunities
                if abs(current_z) > self.spread_z_threshold:
                    metrics['arbitrage_opportunities'] += 1
        
        return metrics
