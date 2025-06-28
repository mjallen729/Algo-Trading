"""
Model utilities for centralized save/load paths.
"""
import os
from pathlib import Path
from typing import Optional

from ..utils import config, get_logger

logger = get_logger(__name__)


class ModelPathManager:
  """
  Centralized model path management.

  Handles all model save/load paths from a single configuration point.
  """

  def __init__(self):
    """Initialize model path manager."""
    # Get model directory from config
    self.models_dir = config.get('models.save_directory', 'models')
    self.auto_create = config.get('models.auto_create_directory', True)

    # Create absolute path relative to project root
    project_root = Path(__file__).parent.parent.parent
    self.models_path = project_root / self.models_dir

    # Create directory if needed
    if self.auto_create:
      self._ensure_directory_exists()

    logger.info(f"ModelPathManager initialized: {self.models_path}")

  def _ensure_directory_exists(self):
    """Create models directory if it doesn't exist."""
    try:
      self.models_path.mkdir(parents=True, exist_ok=True)
      logger.info(f"Models directory ensured: {self.models_path}")
    except Exception as e:
      logger.error(f"Failed to create models directory: {e}")
      raise

  def get_model_path(self, filename: str) -> str:
    """
    Get full path for model file.

    Args:
        filename: Model filename

    Returns:
        Full path to model file
    """
    return str(self.models_path / filename)

  def get_lstm_path(self, suffix: str = "") -> str:
    """Get LSTM model path."""
    filename = f"lstm_model{suffix}.pth"
    return self.get_model_path(filename)

  def get_transformer_path(self, suffix: str = "") -> str:
    """Get Transformer model path."""
    filename = f"transformer_model{suffix}.pth"
    return self.get_model_path(filename)

  def get_ensemble_path(self, suffix: str = "") -> str:
    """Get ensemble model path."""
    filename = f"ensemble_model{suffix}.pkl"
    return self.get_model_path(filename)

  def get_regime_detector_path(self, suffix: str = "") -> str:
    """Get regime detector path."""
    filename = f"regime_detector{suffix}.pkl"
    return self.get_model_path(filename)

  def get_temp_path(self, filename: str) -> str:
    """Get path for temporary files (training checkpoints)."""
    temp_filename = f"temp_{filename}"
    return self.get_model_path(temp_filename)

  def list_saved_models(self) -> dict:
    """
    List all saved models in the directory.

    Returns:
        Dictionary of model types and their files
    """
    if not self.models_path.exists():
      return {}

    models = {
        'lstm': [],
        'transformer': [],
        'ensemble': [],
        'regime_detector': [],
        'other': []
    }

    for file_path in self.models_path.iterdir():
      if file_path.is_file():
        filename = file_path.name

        if 'lstm' in filename.lower():
          models['lstm'].append(filename)
        elif 'transformer' in filename.lower():
          models['transformer'].append(filename)
        elif 'ensemble' in filename.lower():
          models['ensemble'].append(filename)
        elif 'regime' in filename.lower():
          models['regime_detector'].append(filename)
        else:
          models['other'].append(filename)

    return models

  def cleanup_temp_files(self):
    """Remove temporary training files."""
    try:
      temp_files = list(self.models_path.glob("temp_*"))
      for temp_file in temp_files:
        temp_file.unlink()
        logger.info(f"Removed temp file: {temp_file.name}")
    except Exception as e:
      logger.warning(f"Error cleaning temp files: {e}")

  def get_models_directory(self) -> str:
    """Get the models directory path."""
    return str(self.models_path)


# Global instance
model_paths = ModelPathManager()
