"""
Ensemble model combining LSTM and Transformer predictions.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import pickle
from pathlib import Path

from .lstm_predictor import LSTMPredictor
from .transformer import TransformerPredictor
from .model_utils import model_paths
from utils import get_logger, config

logger = get_logger(__name__)


class HybridPredictor:
  """
  Ensemble predictor combining LSTM and Transformer models.

  This model leverages the strengths of both architectures:
  - LSTM: Strong temporal dependencies and short-term patterns
  - Transformer: Long-range dependencies and attention mechanisms
  """

  def __init__(self):
    """Initialize the hybrid predictor."""
    self.lstm_model = LSTMPredictor()
    self.transformer_model = TransformerPredictor()

    # Ensemble weights from config
    self.lstm_weight = config.get('models.ensemble.lstm_weight', 0.5)
    self.transformer_weight = config.get(
      'models.ensemble.transformer_weight', 0.5)
    self.confidence_threshold = config.get(
      'models.ensemble.confidence_threshold', 0.65)

    # Normalize weights to sum to 1
    total_weight = self.lstm_weight + self.transformer_weight
    self.lstm_weight /= total_weight
    self.transformer_weight /= total_weight

    self.is_trained = False
    self.performance_history = {'lstm': [], 'transformer': [], 'ensemble': []}

    logger.info(f"HybridPredictor initialized with weights - LSTM: {self.lstm_weight:.2f}, "
                f"Transformer: {self.transformer_weight:.2f}")

  def train(self,
            data: pd.DataFrame,
            target_column: str = 'Close',
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32) -> Dict:
    """
    Train both LSTM and Transformer models.

    Args:
        data: Training data
        target_column: Target column to predict
        validation_split: Fraction of data for validation
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Combined training history
    """
    logger.info("Starting hybrid model training...")

    # Train LSTM model
    logger.info("Training LSTM component...")
    lstm_history = self.lstm_model.train(
        data=data,
        target_column=target_column,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size
    )

    # Train Transformer model
    logger.info("Training Transformer component...")
    transformer_history = self.transformer_model.train(
        data=data,
        target_column=target_column,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size
    )

    self.is_trained = True

    # Combine training histories
    combined_history = {
        'lstm': lstm_history,
        'transformer': transformer_history
    }

    logger.info("Hybrid model training completed")
    return combined_history

  def predict(self,
              data: pd.DataFrame,
              return_confidence: bool = True,
              adaptive_weighting: bool = True) -> Tuple[float, float]:
    """
    Make ensemble prediction combining LSTM and Transformer outputs.

    Args:
        data: Recent market data for prediction
        return_confidence: Whether to return confidence score
        adaptive_weighting: Whether to use adaptive weights based on model agreement

    Returns:
        Tuple of (prediction, confidence)
    """
    if not self.is_trained:
      raise ValueError("Models must be trained before making predictions")

    # Get predictions from both models
    lstm_pred, lstm_conf = self.lstm_model.predict(
      data, return_confidence=True)
    transformer_pred, transformer_conf = self.transformer_model.predict(
      data, return_confidence=True)

    if adaptive_weighting:
      # Adaptive weighting based on recent performance and agreement
      lstm_weight, transformer_weight = self._calculate_adaptive_weights(
          lstm_pred, transformer_pred, lstm_conf, transformer_conf
      )
    else:
      lstm_weight = self.lstm_weight
      transformer_weight = self.transformer_weight

    # Weighted ensemble prediction
    ensemble_pred = (lstm_weight * lstm_pred +
                     transformer_weight * transformer_pred)

    if return_confidence:
      # Ensemble confidence based on model agreement and individual confidences
      model_agreement = 1 - abs(lstm_pred - transformer_pred) / \
          max(abs(lstm_pred), abs(transformer_pred), 1e-8)
      avg_confidence = (lstm_conf + transformer_conf) / 2

      # Final confidence combines agreement and individual model confidence
      ensemble_confidence = (0.6 * model_agreement + 0.4 * avg_confidence)
      ensemble_confidence = max(0.3, min(0.95, ensemble_confidence))

      return ensemble_pred, ensemble_confidence

    return ensemble_pred

  def _calculate_adaptive_weights(self,
                                  lstm_pred: float,
                                  transformer_pred: float,
                                  lstm_conf: float,
                                  transformer_conf: float) -> Tuple[float, float]:
    """
    Calculate adaptive weights based on model confidence and agreement.

    Args:
        lstm_pred: LSTM prediction
        transformer_pred: Transformer prediction
        lstm_conf: LSTM confidence
        transformer_conf: Transformer confidence

    Returns:
        Tuple of (lstm_weight, transformer_weight)
    """
    # Base weights on individual model confidence
    total_conf = lstm_conf + transformer_conf
    if total_conf > 0:
      adaptive_lstm_weight = lstm_conf / total_conf
      adaptive_transformer_weight = transformer_conf / total_conf
    else:
      # Fallback to default weights
      adaptive_lstm_weight = self.lstm_weight
      adaptive_transformer_weight = self.transformer_weight

    # Smooth the weights to avoid dramatic changes
    smoothing_factor = 0.7
    final_lstm_weight = (smoothing_factor * self.lstm_weight +
                         (1 - smoothing_factor) * adaptive_lstm_weight)
    final_transformer_weight = (smoothing_factor * self.transformer_weight +
                                (1 - smoothing_factor) * adaptive_transformer_weight)

    # Normalize weights
    total_weight = final_lstm_weight + final_transformer_weight
    final_lstm_weight /= total_weight
    final_transformer_weight /= total_weight

    return final_lstm_weight, final_transformer_weight

  def evaluate_prediction_accuracy(self,
                                   prediction: float,
                                   actual: float,
                                   model_type: str = 'ensemble'):
    """
    Track prediction accuracy for model performance monitoring.

    Args:
        prediction: Model prediction
        actual: Actual observed value
        model_type: Type of model ('lstm', 'transformer', 'ensemble')
    """
    error = abs(prediction - actual) / max(abs(actual), 1e-8)

    if model_type in self.performance_history:
      self.performance_history[model_type].append(error)

      # Keep only recent history (last 100 predictions)
      if len(self.performance_history[model_type]) > 100:
        self.performance_history[model_type] = self.performance_history[model_type][-100:]

  def get_model_performance(self) -> Dict[str, float]:
    """
    Get recent performance metrics for all models.

    Returns:
        Dictionary with average errors for each model
    """
    performance = {}

    for model_type, errors in self.performance_history.items():
      if errors:
        performance[f'{model_type}_avg_error'] = np.mean(errors)
        performance[f'{model_type}_std_error'] = np.std(errors)
      else:
        performance[f'{model_type}_avg_error'] = None
        performance[f'{model_type}_std_error'] = None

    return performance

  def save_ensemble(self, filepath: str = None):
    """Save the entire ensemble model."""
    if not self.is_trained:
      raise ValueError("No trained ensemble to save")

    if filepath is None:
      filepath = model_paths.get_ensemble_path()

    # Save individual models to models directory
    lstm_path = model_paths.get_lstm_path()
    transformer_path = model_paths.get_transformer_path()

    self.lstm_model.save_model(lstm_path)
    self.transformer_model.save_model(transformer_path)

    # Save ensemble configuration
    ensemble_config = {
        'lstm_weight': self.lstm_weight,
        'transformer_weight': self.transformer_weight,
        'confidence_threshold': self.confidence_threshold,
        'performance_history': self.performance_history,
        'is_trained': self.is_trained,
        'lstm_path': lstm_path,
        'transformer_path': transformer_path
    }

    with open(filepath, 'wb') as f:
      pickle.dump(ensemble_config, f)

    logger.info(f"Ensemble model saved to {filepath}")

  def load_ensemble(self, filepath: str = None):
    """Load the entire ensemble model."""
    if filepath is None:
      filepath = model_paths.get_ensemble_path()

    # Load ensemble configuration
    with open(filepath, 'rb') as f:
      ensemble_config = pickle.load(f)

    self.lstm_weight = ensemble_config['lstm_weight']
    self.transformer_weight = ensemble_config['transformer_weight']
    self.confidence_threshold = ensemble_config['confidence_threshold']
    self.performance_history = ensemble_config['performance_history']
    self.is_trained = ensemble_config['is_trained']

    # Load individual models using stored paths or defaults
    lstm_path = ensemble_config.get('lstm_path', model_paths.get_lstm_path())
    transformer_path = ensemble_config.get(
      'transformer_path', model_paths.get_transformer_path())

    self.lstm_model.load_model(lstm_path)
    self.transformer_model.load_model(transformer_path)

    logger.info(f"Ensemble model loaded from {filepath}")
