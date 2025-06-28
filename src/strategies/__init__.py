"""
Trading strategies for the cryptocurrency superalgorithm.
"""

from .base import BaseStrategy
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy
from .arbitrage import ArbitrageStrategy

__all__ = [
    'BaseStrategy',
    'MomentumStrategy', 
    'MeanReversionStrategy',
    'ArbitrageStrategy'
]
