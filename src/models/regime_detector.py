"""
Market regime detection using Hidden Markov Models and clustering.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .model_utils import model_paths
from ..utils import get_logger, config

logger = get_logger(__name__)


class RegimeDetector:
    """
    Market regime detection using multiple approaches:
    - Hidden Markov Models for sequential regime modeling
    - K-means clustering for pattern-based regimes
    - Gaussian Mixture Models for probabilistic regimes
    """
    
    def __init__(self, n_regimes: int = 3):
        """
        Initialize regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
                      (typical: 3 = trending up, ranging, trending down)
        """
        self.n_regimes = n_regimes
        self.scaler = StandardScaler()
        
        # Initialize models
        self.hmm_model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=1000,
            random_state=42
        )
        
        self.kmeans_model = KMeans(
            n_clusters=n_regimes,
            random_state=42,
            n_init=10
        )
        
        self.gmm_model = GaussianMixture(
            n_components=n_regimes,
            random_state=42,
            max_iter=200
        )
        
        self.is_fitted = False
        self.feature_columns = None
        self.regime_labels = {
            0: "Trending Down",
            1: "Ranging/Sideways", 
            2: "Trending Up"
        }
        
        logger.info(f"RegimeDetector initialized with {n_regimes} regimes")
    
    def extract_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for regime detection.
        
        Args:
            data: Market data with OHLCV columns
            
        Returns:
            DataFrame with regime detection features
        """
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['volatility'] = features['returns'].rolling(20).std()
        
        # Trend strength features
        features['sma_20'] = data['Close'].rolling(20).mean()
        features['sma_50'] = data['Close'].rolling(50).mean()
        features['trend_strength'] = (data['Close'] - features['sma_20']) / features['sma_20']
        features['trend_direction'] = np.where(
            features['sma_20'] > features['sma_50'], 1, 
            np.where(features['sma_20'] < features['sma_50'], -1, 0)
        )
        
        # Momentum features
        features['rsi'] = self._calculate_rsi(data['Close'])
        features['macd'] = self._calculate_macd(data['Close'])
        features['momentum'] = data['Close'] / data['Close'].shift(10) - 1
        
        # Volume-based features
        features['volume_sma'] = data['Volume'].rolling(20).mean()
        features['volume_ratio'] = data['Volume'] / features['volume_sma']
        features['price_volume_trend'] = (features['returns'] * features['volume_ratio']).rolling(5).mean()
        
        # Volatility regime features
        features['volatility_regime'] = pd.cut(
            features['volatility'], 
            bins=3, 
            labels=['Low', 'Medium', 'High']
        ).cat.codes
        
        # Range-bound vs trending features
        features['high_low_ratio'] = (data['High'] - data['Low']) / data['Close']
        features['price_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Rolling correlation with trend
        features['trend_correlation'] = features['returns'].rolling(20).corr(
            pd.Series(range(20), index=features.index[-20:])
        ) if len(features) >= 20 else 0
        
        # Clean up features
        features = features.fillna(method='ffill').fillna(0)
        
        # Select final feature columns
        self.feature_columns = [
            'returns', 'volatility', 'trend_strength', 'trend_direction',
            'rsi', 'macd', 'momentum', 'volume_ratio', 'price_volume_trend',
            'volatility_regime', 'high_low_ratio', 'price_position'
        ]
        
        return features[self.feature_columns]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator."""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow
    
    def fit(self, data: pd.DataFrame) -> Dict:
