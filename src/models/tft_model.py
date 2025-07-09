import torch
import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import LightningModule

from src.config import (
  TFT_HIDDEN_SIZE,
  TFT_HIDDEN_CONTINUOUS_SIZE,
  TFT_LSTM_LAYERS,
  TFT_ATTENTION_HEADS,
  TFT_DROPOUT,
  TFT_LEARNING_RATE,
  TFT_BATCH_SIZE,
  TFT_REDUCE_ON_PLATEAU_PATIENCE,
  TFT_NUM_WORKERS,
)


class TFTModel:
  def __init__(self, max_encoder_length: int, max_prediction_length: int, training_cutoff: int = None):
    self.max_encoder_length = max_encoder_length
    self.max_prediction_length = max_prediction_length
    self.training_cutoff = training_cutoff
    self.model = None
    self.training_data = None
    self.validation_data = None
    self.tft_dataset = None

  def prepare_data(self, df: pd.DataFrame, time_idx_col: str = 'time_idx', target_col: str = 'close', group_id_col: str = 'symbol'):
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set up categorical and time index columns
    df[group_id_col] = df[group_id_col].astype("category")
    if time_idx_col not in df.columns:
      df[time_idx_col] = df.groupby(group_id_col, observed=True).cumcount()

    # Convert time features to categorical (should already exist from feature engineering)
    if 'month' in df.columns:
      df["month"] = df["month"].astype(str).astype("category")
    if 'day' in df.columns:
      df["day"] = df["day"].astype(str).astype("category") 
    if 'weekday' in df.columns:
      df["weekday"] = df["weekday"].astype(str).astype("category")
    if 'hour' in df.columns:
      df["hour"] = df["hour"].astype(str).astype("category")
    
    static_features = []
    
    # Time varying categorical features (only include if they exist)
    available_time_categoricals = ["month", "day", "weekday", "hour"]
    time_varying_known_categoricals = [col for col in available_time_categoricals if col in df.columns]
    
    # Time varying known real features (cyclical time features)
    available_time_reals = ['hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos']
    time_varying_known_reals = [col for col in available_time_reals if col in df.columns]

    # Base OHLCV features
    base_features = [target_col, 'open', 'high', 'low', 'volume']
    
    # Technical indicators that might be available
    technical_indicators = [
      'SMA_10', 'EMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
      'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR', 'OBV', 'ADX', 'CCI', 'MFI',
      'Volatility', 'Volatility_pct', 'SMA_ratio', 'price_div_sma50', 'bb_width'
    ]
    
    # Pre-calculated features from cleaned data
    precalc_features = ['SMA25', 'EMA25', 'ATR25', 'RSI25', 'Volatility25', 'Market Cap']
    
    # Combine all potential features
    potential_features = base_features + technical_indicators + precalc_features
    
    # Only include features that actually exist in the DataFrame
    time_varying_unknown_reals = [col for col in potential_features if col in df.columns]

    # Add lagged features to the list of unknown reals
    lagged_cols = [col for col in df.columns if '_lag_' in col]
    time_varying_unknown_reals.extend(lagged_cols)
    
    # Ensure no duplicates
    time_varying_unknown_reals = list(dict.fromkeys(time_varying_unknown_reals))
    
    print(f"TFT Dataset setup:")
    print(f"  - Categorical features: {time_varying_known_categoricals}")
    print(f"  - Known real features: {time_varying_known_reals}")
    print(f"  - Unknown real features: {len(time_varying_unknown_reals)} features")
    print(f"  - Lagged features: {len(lagged_cols)} features")
    print(f"  - DataFrame shape: {df.shape}")
    print(f"  - DataFrame index type: {type(df.index)}")
    
    # Ensure DataFrame has a proper integer index
    if not isinstance(df.index, pd.RangeIndex):
      df = df.reset_index(drop=True)
      print("  - Reset DataFrame index to RangeIndex")
    
    # Ensure time_idx is properly set as integer sequence (already done in feature engineering)
    # df[time_idx_col] = range(len(df)) # Removed: time_idx should be consistent from feature engineering
    # print(f"  - Set time_idx from 0 to {df[time_idx_col].max()}") # Removed: time_idx should be consistent from feature engineering
    
    # Check for any NaN values that might cause issues
    nan_cols = df.columns[df.isnull().any()].tolist()
    if nan_cols:
      print(f"  - Warning: NaN values found in columns: {nan_cols}")
      df = df.ffill().bfill()
      print("  - Filled NaN values using forward/backward fill")

    if self.training_cutoff is not None:
      # Split the data based on time_idx before creating datasets
      train_df = df[df[time_idx_col] <= self.training_cutoff].copy()
      val_df = df[df[time_idx_col] > self.training_cutoff].copy()
      
      print(f"  - Training data: {len(train_df)} samples (time_idx 0 to {self.training_cutoff})")
      print(f"  - Validation data: {len(val_df)} samples (time_idx {self.training_cutoff+1} to {df[time_idx_col].max()})")
      
      if len(train_df) == 0:
        raise ValueError(f"Training set is empty! training_cutoff={self.training_cutoff}, max_time_idx={df[time_idx_col].max()}")
      
      # Create training dataset first
      self.training_data = TimeSeriesDataSet(
        train_df,
        time_idx='time_idx',
        target=target_col,
        group_ids=[group_id_col],
        min_encoder_length=self.max_encoder_length,
        max_encoder_length=self.max_encoder_length,
        min_prediction_length=self.max_prediction_length,
        max_prediction_length=self.max_prediction_length,
        static_categoricals=static_features,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
          groups=[group_id_col], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
      )
      
      # Create validation dataset from the training dataset template
      self.validation_data = TimeSeriesDataSet.from_dataset(
        self.training_data, val_df, predict=True, stop_randomization=True
      )
      
      # Set the main dataset as training data
      self.tft_dataset = self.training_data
      
    else:
      # Use entire dataset for training if no cutoff specified
      self.tft_dataset = TimeSeriesDataSet(
        df,
        time_idx='time_idx',
        target=target_col,
        group_ids=[group_id_col],
        min_encoder_length=self.max_encoder_length,
        max_encoder_length=self.max_encoder_length,
        min_prediction_length=self.max_prediction_length,
        max_prediction_length=self.max_prediction_length,
        static_categoricals=static_features,
        time_varying_known_categoricals=time_varying_known_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(
          groups=[group_id_col], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True
      )
      
      self.training_data = self.tft_dataset
      self.validation_data = self.tft_dataset

  def build_model(self):
    # Use training dataset to build model if available, otherwise use full dataset
    dataset_for_model = self.training_data if self.training_data is not None else self.tft_dataset
    
    if dataset_for_model is None:
      raise ValueError("Data not prepared. Call prepare_data() first.")

    print(f"Building model from dataset with {len(dataset_for_model)} samples...")
    self.model = TemporalFusionTransformer.from_dataset(
      dataset_for_model,
      hidden_size=TFT_HIDDEN_SIZE,
      hidden_continuous_size=TFT_HIDDEN_CONTINUOUS_SIZE,
      lstm_layers=TFT_LSTM_LAYERS,
      attention_head_size=TFT_ATTENTION_HEADS,
      dropout=TFT_DROPOUT,
      learning_rate=TFT_LEARNING_RATE,
      loss=QuantileLoss(),
      optimizer="Adam",
      reduce_on_plateau_patience=TFT_REDUCE_ON_PLATEAU_PATIENCE,
    )

  def train_model(self, trainer, ckpt_path=None):
    if self.training_data is None or self.validation_data is None:
      raise ValueError("Data not prepared. Call prepare_data() first.")
    
    train_dataloader = self.training_data.to_dataloader(
      train=True, batch_size=TFT_BATCH_SIZE, num_workers=TFT_NUM_WORKERS, persistent_workers=True)
    val_dataloader = self.validation_data.to_dataloader(
      train=False, batch_size=TFT_BATCH_SIZE, num_workers=TFT_NUM_WORKERS, persistent_workers=True)
    
    trainer.fit(self.model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

  def predict(self, data: pd.DataFrame) -> torch.Tensor:
    if self.model is None or self.tft_dataset is None:
      raise ValueError("Model not built or dataset not prepared.")

    # Create a prediction dataset from the input data
    prediction_dataset = TimeSeriesDataSet.from_dataset(
      self.tft_dataset, data, predict=True)
    prediction_dataloader = prediction_dataset.to_dataloader(
      train=False, batch_size=1)

    # Make predictions
    predictions = self.model.predict(prediction_dataloader, mode="prediction")
    return predictions

  def save_model(self, path: str):
    if self.model is None:
      raise ValueError("No model to save.")
    torch.save(self.model.state_dict(), path)

  def load_model(self, path: str):
    if self.tft_dataset is None:
      raise ValueError(
        "Dataset not prepared. Call prepare_data() before loading model.")
    # First build the model architecture
    self.build_model()
    # Then load the saved weights
    self.model.load_state_dict(torch.load(path, map_location='cpu'))
