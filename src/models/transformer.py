"""
Transformer-based price prediction model for cryptocurrency trading.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import math
from typing import Tuple, Optional, Dict
from pathlib import Path

from .model_utils import model_paths
from ..utils import get_logger, config

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer model."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    """Transformer model for time series prediction."""
    
    def __init__(self,
                 input_size: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 output_size: int = 1):
        super(TransformerModel, self).__init__()
        
        self.d_model = d_model
        self.input_size = input_size
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.fc2 = nn.Linear(d_model // 2, output_size)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        """Forward pass through the network."""
        # Input projection
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Global average pooling
        pooled = torch.mean(transformer_out, dim=1)
        
        # Output layers
        out = self.layer_norm(pooled)
        out = self.dropout(out)
        out = self.gelu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TransformerPredictor:
    """Transformer-based cryptocurrency price predictor."""
    
    def __init__(self):
        """Initialize Transformer predictor."""
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = config.get('models.lstm.sequence_length', 60)
        self.d_model = config.get('models.transformer.d_model', 128)
        self.nhead = config.get('models.transformer.nhead', 8)
        self.num_layers = config.get('models.transformer.num_layers', 4)
        self.dropout = config.get('models.transformer.dropout', 0.1)
        self.learning_rate = config.get('models.transformer.learning_rate', 0.001)
        
        self.is_trained = False
        self.feature_columns = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"TransformerPredictor initialized with device: {self.device}")
    
    def prepare_sequences(self, 
                         data: pd.DataFrame, 
                         target_column: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for Transformer training."""
        if self.feature_columns is None:
            self.feature_columns = [col for col in data.columns 
                                  if col not in ['Date', 'Timestamp']]
        
        scaled_data = self.scaler.fit_transform(data[self.feature_columns])
        
        X, y = [], []
        target_idx = self.feature_columns.index(target_column)
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def train(self, 
              data: pd.DataFrame, 
              target_column: str = 'Close',
              validation_split: float = 0.2,
              epochs: int = 100,
              batch_size: int = 32) -> Dict:
        """Train the Transformer model."""
        logger.info("Starting Transformer model training...")
        
        X, y = self.prepare_sequences(data, target_column)
        
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).to(self.device)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 15
        
        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                
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
            
            scheduler.step()
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                          f"Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), model_paths.get_temp_path('best_transformer_model.pth'))
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        self.model.load_state_dict(torch.load(model_paths.get_temp_path('best_transformer_model.pth')))
        self.is_trained = True
        logger.info("Transformer model training completed")
        return history
    
    def predict(self, data: pd.DataFrame, return_confidence: bool = True) -> Tuple[float, float]:
        """Make price prediction."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        self.model.eval()
        
        if len(data) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} data points")
        
        recent_data = data.tail(self.sequence_length)
        scaled_data = self.scaler.transform(recent_data[self.feature_columns])
        
        X = torch.FloatTensor(scaled_data).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(X).cpu().numpy()[0, 0]
        
        # Inverse transform prediction
        dummy = np.zeros((1, len(self.feature_columns)))
        target_idx = self.feature_columns.index('Close')
        dummy[0, target_idx] = prediction
        
        prediction_unscaled = self.scaler.inverse_transform(dummy)[0, target_idx]
        
        if return_confidence:
            confidence = min(0.95, max(0.5, 1.0 - abs(prediction) * 0.08))
            return prediction_unscaled, confidence
        
        return prediction_unscaled
    
    def save_model(self, filepath: str = None):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        if filepath is None:
            filepath = model_paths.get_transformer_path()
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'config': {
                'sequence_length': self.sequence_length,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_layers': self.num_layers,
                'dropout': self.dropout
            }
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Transformer model saved to {filepath}")
    
    def load_model(self, filepath: str = None):
        """Load a trained model."""
        if filepath is None:
            filepath = model_paths.get_transformer_path()
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        config_dict = checkpoint['config']
        self.sequence_length = config_dict['sequence_length']
        self.d_model = config_dict['d_model']
        self.nhead = config_dict['nhead']
        self.num_layers = config_dict['num_layers']
        self.dropout = config_dict['dropout']
        
        self.scaler = checkpoint['scaler']
        self.feature_columns = checkpoint['feature_columns']
        
        input_size = len(self.feature_columns)
        self.model = TransformerModel(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True
        
        logger.info(f"Transformer model loaded from {filepath}")
