"""
Configuration management utilities for the trading system.
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging


class Config:
  """Configuration manager for the trading system."""

  def __init__(self, config_path: str = None):
    """
    Initialize configuration manager.

    Args:
        config_path: Path to configuration file
    """
    if config_path is None:
      project_root = Path(__file__).parent.parent.parent
      config_path = project_root / "configs" / "config.yaml"

    self.config_path = Path(config_path)
    self.config = self._load_config()

  def _load_config(self) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
      with open(self.config_path, 'r') as file:
        config = yaml.safe_load(file)
        logging.info(f"Configuration loaded from {self.config_path}")
        return config
    except FileNotFoundError:
      logging.error(f"Configuration file not found: {self.config_path}")
      raise
    except yaml.YAMLError as e:
      logging.error(f"Error parsing configuration file: {e}")
      raise

  def get(self, key_path: str, default=None):
    """
    Get configuration value using dot notation.

    Args:
        key_path: Dot-separated path to configuration value
        default: Default value if key not found

    Returns:
        Configuration value
    """
    keys = key_path.split('.')
    value = self.config

    try:
      for key in keys:
        value = value[key]
      return value
    except (KeyError, TypeError):
      if default is not None:
        return default
      raise KeyError(f"Configuration key not found: {key_path}")

  def set(self, key_path: str, value: Any):
    """Set configuration value using dot notation."""
    keys = key_path.split('.')
    config_ref = self.config

    for key in keys[:-1]:
      if key not in config_ref:
        config_ref[key] = {}
      config_ref = config_ref[key]

    config_ref[keys[-1]] = value

  def save(self):
    """Save current configuration to file."""
    with open(self.config_path, 'w') as file:
      yaml.safe_dump(self.config, file, default_flow_style=False)
      logging.info(f"Configuration saved to {self.config_path}")


# Global configuration instance
config = Config()
