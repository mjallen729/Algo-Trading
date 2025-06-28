"""
LSTM-based price prediction model for cryptocurrency trading.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Tuple, Optional, Dict, List
import joblib
from pathlib import Path

from .model_utils import model_paths
from utils import get_logger, config

logger = get_logger(__name__)


class LSTMModel(nn.Module):
  """LSTM neural network model for time series prediction."""

  def __init__(self,
               input_size: int,
               hidden_size: int = 128,
               num_layers: int = 2,
               dropout: float = 0.2,
               output_size: int = 1):
    """
    Initialize LSTM model.

    Args:
        input_size: Number of input features
        hidden_size: Hidden layer size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        output_size: Number of output predictions
    """
    super(LSTMModel, self).__init__()

    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.input_size = input_size

    # LSTM layers
    self.lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout if num_layers > 1 else 0,
        batch_first=True
    )

    # Attention mechanism
    self.attention = nn.MultiheadAttention(
        embed_dim=hidden_size,
        num_heads=8,
        dropout=dropout
    )

    # Output layers
    self.dropout = nn.Dropout(dropout)
    self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
    self.fc2 = nn.Linear(hidden_size // 2, output_size)
    self.relu = nn.ReLU()

  def forward(self, x):
    """Forward pass through the network."""
    batch_size = x.size(0)

    # Initialize hidden state
    h0 = torch.zeros(self.num_layers, batch_size,
                     self.hidden_size).to(x.device)
    c0 = torch.zeros(self.num_layers, batch_size,
                     self.hidden_size).to(x.device)

    # LSTM forward pass
    lstm_out, _ = self.lstm(x, (h0, c0))

    # Apply attention to LSTM output
    lstm_out_transposed = lstm_out.transpose(0, 1)  # (seq_len, batch, hidden)
    attended_out, _ = self.attention(
        lstm_out_transposed,
        lstm_out_transposed,
        lstm_out_transposed
    )
    attended_out = attended_out.transpose(0, 1)  # (batch, seq_len, hidden)

    # Use the last time step
    out = attended_out[:, -1, :]

    # Fully connected layers
    out = self.dropout(out)
    out = self.relu(self.fc1(out))
    out = self.dropout(out)
    out = self.fc2(out)

    return out


class LSTMPredictor:
  """LSTM-based cryptocurrency price predictor."""

  def __init__(self):
    """Initialize LSTM predictor."""
    self.model = None
    self.scaler = MinMaxScaler()
    self.sequence_length = config.get('models.lstm.sequence_length', 60)
    self.hidden_size = config.get('models.lstm.hidden_size', 128)
    self.num_layers = config.get('models.lstm.num_layers', 2)
    self.dropout = config.get('models.lstm.dropout', 0.2)
    self.learning_rate = config.get('models.lstm.learning_rate', 0.001)

    self.is_trained = False
    self.feature_columns = None
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"LSTMPredictor initialized with device: {self.device}")

  def prepare_sequences(self,
                        data: pd.DataFrame,
                        target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM training.

    Args:
        data: DataFrame with market data
        target_column: Column to predict

    Returns:
        Tuple of (X, y) sequences
    """
    # Select feature columns
    if self.feature_columns is None:
      self.feature_columns = [col for col in data.columns
                              if col not in ['Date', 'Timestamp']]

    # Scale the data
    scaled_data = self.scaler.fit_transform(data[self.feature_columns])

    # Create sequences
    X, y = [], []
    target_idx = self.feature_columns.index(target_column)

    for i in range(self.sequence_length, len(scaled_data)):
      X.append(scaled_data[i - self.sequence_length:i])
      y.append(scaled_data[i, target_idx])

    return np.array(X), np.array(y)

  def train(self,
            data: pd.DataFrame,
            target_column: str = 'Close',
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32) -> Dict:
    """
    Train the LSTM model.

    Args:
        data: Training data
        target_column: Target column to predict
        validation_split: Fraction of data for validation
        epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Training history dictionary
    """
    logger.info("Starting LSTM model training...")

    # Prepare sequences
    X, y = self.prepare_sequences(data, target_column)

    # Train/validation split
    split_idx = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(self.device)
    y_train = torch.FloatTensor(y_train).to(self.device)
    X_val = torch.FloatTensor(X_val).to(self.device)
    y_val = torch.FloatTensor(y_val).to(self.device)

    # Initialize model
    input_size = X_train.shape[2]
    self.model = LSTMModel(
        input_size=input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        dropout=self.dropout
    ).to(self.device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20

    for epoch in range(epochs):
      # Training phase
      self.model.train()
      total_train_loss = 0

      for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = self.model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        optimizer.step()
        total_train_loss += loss.item()

      # Validation phase
      self.model.eval()
      with torch.no_grad():
        val_outputs = self.model(X_val)
        val_loss = criterion(val_outputs.squeeze(), y_val).item()

      avg_train_loss = total_train_loss / (len(X_train) // batch_size)
      history['train_loss'].append(avg_train_loss)
      history['val_loss'].append(val_loss)

      scheduler.step(val_loss)

      if epoch % 10 == 0:
        logger.info(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                    f"Val Loss: {val_loss:.6f}")

      # Early stopping
      if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(self.model.state_dict(),
                   model_paths.get_temp_path('best_lstm_model.pth'))
      else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
          logger.info(f"Early stopping at epoch {epoch}")
          break

    # Load best model
    self.model.load_state_dict(torch.load(
      model_paths.get_temp_path('best_lstm_model.pth'), weights_only=True))
    self.is_trained = True

    logger.info("LSTM model training completed")
    return history

  def predict(self,
              data: pd.DataFrame,
              return_confidence: bool = True) -> Tuple[float, float]:
    """
    Make price prediction.

    Args:
        data: Recent market data (last sequence_length periods)
        return_confidence: Whether to return confidence score

    Returns:
        Tuple of (prediction, confidence)
    """
    if not self.is_trained:
      raise ValueError("Model must be trained before making predictions")

    self.model.eval()

    # Prepare input sequence
    if len(data) < self.sequence_length:
      raise ValueError(f"Need at least {self.sequence_length} data points")

    # Use the most recent sequence_length periods
    recent_data = data.tail(self.sequence_length)
    scaled_data = self.scaler.transform(recent_data[self.feature_columns])

    # Convert to tensor
    X = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)

    with torch.no_grad():
      prediction = self.model(X).cpu().numpy()[0, 0]

    # Inverse transform prediction
    # Create dummy array for inverse transform
    dummy = np.zeros((1, len(self.feature_columns)))
    target_idx = self.feature_columns.index('Close')
    dummy[0, target_idx] = prediction

    prediction_unscaled = self.scaler.inverse_transform(dummy)[0, target_idx]

    # Calculate confidence based on recent prediction accuracy
    if return_confidence:
      # Simple confidence metric based on prediction uncertainty
      # In practice, this could be enhanced with ensemble variance
      confidence = min(0.95, max(0.5, 1.0 - abs(prediction) * 0.1))
      return prediction_unscaled, confidence

    return prediction_unscaled

  def save_model(self, filepath: str = None):
    """Save the trained model."""
    if not self.is_trained:
      raise ValueError("No trained model to save")

    if filepath is None:
      filepath = model_paths.get_lstm_path()

    save_dict = {
        'model_state_dict': self.model.state_dict(),
        'scaler': self.scaler,
        'feature_columns': self.feature_columns,
        'config': {
            'sequence_length': self.sequence_length,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout
        }
    }

    torch.save(save_dict, filepath)
    logger.info(f"Model saved to {filepath}")

  def load_model(self, filepath: str = None):
    """Load a trained model."""
    if filepath is None:
      filepath = model_paths.get_lstm_path()

    checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

    # Restore configuration
    config_dict = checkpoint['config']
    self.sequence_length = config_dict['sequence_length']
    self.hidden_size = config_dict['hidden_size']
    self.num_layers = config_dict['num_layers']
    self.dropout = config_dict['dropout']

    # Restore scaler and features
    self.scaler = checkpoint['scaler']
    self.feature_columns = checkpoint['feature_columns']

    # Initialize and load model
    input_size = len(self.feature_columns)
    self.model = LSTMModel(
        input_size=input_size,
        hidden_size=self.hidden_size,
        num_layers=self.num_layers,
        dropout=self.dropout
    ).to(self.device)

    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.is_trained = True

    logger.info(f"Model loaded from {filepath}")
