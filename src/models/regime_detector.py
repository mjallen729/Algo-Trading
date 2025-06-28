"""
Market Regime Detection Module

This module provides advanced market regime detection using multiple approaches:
- Hidden Markov Models for sequential regime modeling
- K-means clustering for pattern-based regimes
- Gaussian Mixture Models for probabilistic regimes
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from typing import Dict, List, Tuple, Optional
import warnings
import os
import pickle

from src.utils.logger import get_logger
from src.models.model_utils import ModelPathManager

warnings.filterwarnings("ignore")
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
        2: "Trending Up",
    }

    # Model path manager
    self.path_manager = ModelPathManager()

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
    features["returns"] = data["Close"].pct_change()
    features["log_returns"] = np.log(data["Close"] / data["Close"].shift(1))
    features["volatility"] = features["returns"].rolling(20).std()

    # Trend strength features
    features["sma_20"] = data["Close"].rolling(20).mean()
    features["sma_50"] = data["Close"].rolling(50).mean()
    features["trend_strength"] = (
      data["Close"] - features["sma_20"]) / features["sma_20"]
    features["trend_direction"] = np.where(
        features["sma_20"] > features["sma_50"],
        1,
        np.where(features["sma_20"] < features["sma_50"], -1, 0),
    )

    # Momentum features
    features["rsi"] = self._calculate_rsi(data["Close"])
    features["macd"] = self._calculate_macd(data["Close"])
    features["momentum"] = data["Close"] / data["Close"].shift(10) - 1

    # Volume-based features
    features["volume_sma"] = data["Volume"].rolling(20).mean()
    features["volume_ratio"] = data["Volume"] / features["volume_sma"]
    features["price_volume_trend"] = (
        (features["returns"] * features["volume_ratio"]).rolling(5).mean()
    )

    # Volatility regime features
    features["volatility_regime"] = pd.cut(
        features["volatility"], bins=3, labels=["Low", "Medium", "High"]
    ).cat.codes

    # Range-bound vs trending features
    features["high_low_ratio"] = (data["High"] - data["Low"]) / data["Close"]
    features["price_position"] = (data["Close"] - data["Low"]) / (
        data["High"] - data["Low"]
    )

    # Rolling correlation with trend
    if len(features) >= 20:
      trend_index = pd.Series(range(20))
      features["trend_correlation"] = features["returns"].rolling(20).apply(
          lambda x: x.corr(trend_index) if len(x) == 20 else 0,
          raw=False
      )
    else:
      features["trend_correlation"] = 0

    # Clean up features
    features = features.fillna(method='ffill').fillna(0)

    # Select final feature columns
    self.feature_columns = [
        "returns",
        "volatility",
        "trend_strength",
        "trend_direction",
        "rsi",
        "macd",
        "momentum",
        "volume_ratio",
        "price_volume_trend",
        "volatility_regime",
        "high_low_ratio",
        "price_position",
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
    """
    Fit regime detection models to historical data.

    Args:
        data: Historical market data with OHLCV columns

    Returns:
        Dictionary with training results and performance metrics
    """
    logger.info("Training regime detection models...")

    # Extract features
    features = self.extract_regime_features(data)

    # Remove any remaining NaN values
    features = features.dropna()

    if len(features) < 50:
      raise ValueError(
        "Insufficient data for regime detection (minimum 50 observations required)")

    # Scale features
    scaled_features = self.scaler.fit_transform(features)

    results = {}

    try:
      # Train Hidden Markov Model
      logger.info("Training Hidden Markov Model...")
      self.hmm_model.fit(scaled_features)
      hmm_states = self.hmm_model.predict(scaled_features)
      results["hmm_log_likelihood"] = self.hmm_model.score(scaled_features)
      results["hmm_states"] = hmm_states
      logger.info(
        f"HMM trained successfully. Log-likelihood: {results['hmm_log_likelihood']:.2f}")

    except Exception as e:
      logger.warning(f"HMM training failed: {e}")
      results["hmm_error"] = str(e)

    try:
      # Train K-means clustering
      logger.info("Training K-means clustering...")
      kmeans_labels = self.kmeans_model.fit_predict(scaled_features)
      results["kmeans_inertia"] = self.kmeans_model.inertia_
      results["kmeans_labels"] = kmeans_labels
      logger.info(
        f"K-means trained successfully. Inertia: {results['kmeans_inertia']:.2f}")

    except Exception as e:
      logger.warning(f"K-means training failed: {e}")
      results["kmeans_error"] = str(e)

    try:
      # Train Gaussian Mixture Model
      logger.info("Training Gaussian Mixture Model...")
      self.gmm_model.fit(scaled_features)
      gmm_labels = self.gmm_model.predict(scaled_features)
      results["gmm_bic"] = self.gmm_model.bic(scaled_features)
      results["gmm_aic"] = self.gmm_model.aic(scaled_features)
      results["gmm_labels"] = gmm_labels
      logger.info(f"GMM trained successfully. BIC: {results['gmm_bic']:.2f}")

    except Exception as e:
      logger.warning(f"GMM training failed: {e}")
      results["gmm_error"] = str(e)

    self.is_fitted = True

    # Calculate regime statistics
    results["regime_statistics"] = self._calculate_regime_statistics(
      features, results)

    logger.info("Regime detection models training completed")

    return results

  def _calculate_regime_statistics(self, features: pd.DataFrame, results: Dict) -> Dict:
    """Calculate statistics for each detected regime."""
    regime_stats = {}

    # Analyze each model's regimes
    for model_name in ["hmm_states", "kmeans_labels", "gmm_labels"]:
      if model_name in results:
        labels = results[model_name]
        model_stats = {}

        for regime in range(self.n_regimes):
          regime_mask = labels == regime
          if np.any(regime_mask):
            regime_features = features[regime_mask]

            model_stats[f"regime_{regime}"] = {
                "count": int(np.sum(regime_mask)),
                "percentage": float(np.mean(regime_mask) * 100),
                "avg_returns": float(regime_features["returns"].mean()),
                "avg_volatility": float(regime_features["volatility"].mean()),
                "avg_trend_strength": float(regime_features["trend_strength"].mean()),
            }

        regime_stats[model_name] = model_stats

    return regime_stats

  def predict(self, data: pd.DataFrame) -> Dict:
    """
    Predict current market regime.

    Args:
        data: Recent market data for regime prediction

    Returns:
        Dictionary with regime predictions from all models
    """
    if not self.is_fitted:
      raise ValueError("Models must be fitted before prediction")

    # Extract features
    features = self.extract_regime_features(data)
    features = features.dropna()

    if len(features) == 0:
      raise ValueError("No valid features extracted from data")

    # Scale features
    scaled_features = self.scaler.transform(features)

    results = {}

    # HMM prediction
    try:
      hmm_states = self.hmm_model.predict(scaled_features)
      results["hmm_regime"] = int(hmm_states[-1])
      results["hmm_sequence"] = hmm_states.tolist()
    except Exception as e:
      logger.warning(f"HMM prediction failed: {e}")
      results["hmm_regime"] = 1  # Default to ranging

    # K-means prediction
    try:
      kmeans_regimes = self.kmeans_model.predict(scaled_features)
      results["kmeans_regime"] = int(kmeans_regimes[-1])
    except Exception as e:
      logger.warning(f"K-means prediction failed: {e}")
      results["kmeans_regime"] = 1  # Default to ranging

    # GMM prediction
    try:
      gmm_regimes = self.gmm_model.predict(scaled_features)
      gmm_probs = self.gmm_model.predict_proba(scaled_features)
      results["gmm_regime"] = int(gmm_regimes[-1])
      results["gmm_probabilities"] = gmm_probs[-1].tolist()
    except Exception as e:
      logger.warning(f"GMM prediction failed: {e}")
      results["gmm_regime"] = 1  # Default to ranging
      results["gmm_probabilities"] = [0.0, 1.0, 0.0]  # Default probabilities

    # Ensemble prediction (weighted voting)
    ensemble_regime = self._ensemble_regime_prediction(results)
    results["ensemble_regime"] = ensemble_regime

    # Add regime labels
    results["regime_label"] = self.regime_labels.get(
      ensemble_regime, "Unknown")

    return results

  def _ensemble_regime_prediction(self, predictions: Dict) -> int:
    """
    Combine predictions from multiple models using weighted voting.

    Args:
        predictions: Dictionary with model predictions

    Returns:
        Ensemble regime prediction
    """
    # Weights for different models (can be tuned based on performance)
    weights = {"hmm": 0.4, "kmeans": 0.3, "gmm": 0.3}

    # Create voting array
    regime_votes = np.zeros(self.n_regimes)

    # Add weighted votes
    if "hmm_regime" in predictions:
      regime_votes[predictions["hmm_regime"]] += weights["hmm"]

    if "kmeans_regime" in predictions:
      regime_votes[predictions["kmeans_regime"]] += weights["kmeans"]

    if "gmm_regime" in predictions:
      regime_votes[predictions["gmm_regime"]] += weights["gmm"]

    # Return regime with highest vote
    return int(np.argmax(regime_votes))

  def save_models(self, filepath: str = None):
    """Save regime detection models."""
    if not self.is_fitted:
      raise ValueError("Models must be fitted before saving")

    if filepath is None:
      filepath = self.path_manager.get_regime_detector_path()

    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    save_dict = {
        "hmm_model": self.hmm_model,
        "kmeans_model": self.kmeans_model,
        "gmm_model": self.gmm_model,
        "scaler": self.scaler,
        "feature_columns": self.feature_columns,
        "n_regimes": self.n_regimes,
        "regime_labels": self.regime_labels,
        "is_fitted": self.is_fitted,
    }

    with open(filepath, "wb") as f:
      pickle.dump(save_dict, f)

    logger.info(f"Regime detection models saved to {filepath}")

  def load_models(self, filepath: str = None):
    """Load regime detection models."""
    if filepath is None:
      filepath = self.path_manager.get_regime_detector_path()

    if not os.path.exists(filepath):
      logger.warning(f"Model file not found: {filepath}")
      return False

    try:
      with open(filepath, "rb") as f:
        save_dict = pickle.load(f)

      self.hmm_model = save_dict["hmm_model"]
      self.kmeans_model = save_dict["kmeans_model"]
      self.gmm_model = save_dict["gmm_model"]
      self.scaler = save_dict["scaler"]
      self.feature_columns = save_dict["feature_columns"]
      self.n_regimes = save_dict["n_regimes"]
      self.regime_labels = save_dict["regime_labels"]
      self.is_fitted = save_dict["is_fitted"]

      logger.info(f"Regime detection models loaded from {filepath}")
      return True

    except Exception as e:
      logger.error(f"Failed to load regime detection models: {e}")
      return False
