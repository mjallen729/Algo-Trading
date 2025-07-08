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
    df[group_id_col] = df[group_id_col].astype("category")
    df[time_idx_col] = df.groupby(group_id_col, observed=True).cumcount()

    # The FeatureEngineer now adds cyclical time features (sin/cos)
    # We still need the categorical time features for TFT
    df["month"] = df.index.month.astype(str).astype("category")
    df["day"] = df.index.day.astype(str).astype("category")
    df["weekday"] = df.index.dayofweek.astype(str).astype("category")
    df["hour"] = df.index.hour.astype(str).astype("category")

    static_features = []
    time_varying_known_categoricals = ["month", "day", "weekday", "hour"]
    
    time_varying_known_reals = [
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos'
    ]

    time_varying_unknown_reals = [
      target_col, 'open', 'high', 'low', 'volume',
      'SMA_10', 'EMA_10', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
      'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR', 'OBV', 'ADX', 'CCI', 'MFI',
      'Volatility', 'Volatility_pct', 'SMA_ratio', 'price_div_sma50', 'bb_width'
    ]
    
    # Add lagged features to the list of unknown reals
    lagged_cols = [col for col in df.columns if '_lag_' in col]
    time_varying_unknown_reals.extend(lagged_cols)

    # Filter out columns that might not be present
    time_varying_known_reals = [col for col in time_varying_known_reals if col in df.columns]
    time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in df.columns]
    
    # Ensure no duplicates
    time_varying_unknown_reals = list(dict.fromkeys(time_varying_unknown_reals))


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

    if self.training_cutoff is not None:
      # Split the data based on time_idx before creating datasets
      train_df = df[df[time_idx_col] <= self.training_cutoff].copy()
      val_df = df[df[time_idx_col] > self.training_cutoff].copy()
      
      # Create training dataset
      training_dataset = TimeSeriesDataSet(
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
      
      # Create validation dataset from training dataset
      if len(val_df) > 0:
        self.validation_data = TimeSeriesDataSet.from_dataset(
          training_dataset, val_df, predict=False)
      else:
        self.validation_data = training_dataset
        
      self.training_data = training_dataset
    else:
      self.training_data = self.tft_dataset
      # Validate on the whole dataset if no cutoff
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

  def train_model(self, trainer):
    if self.training_data is None or self.validation_data is None:
      raise ValueError("Data not prepared. Call prepare_data() first.")
    
    train_dataloader = self.training_data.to_dataloader(
      train=True, batch_size=128, num_workers=0)
    val_dataloader = self.validation_data.to_dataloader(
      train=False, batch_size=128, num_workers=0)
    
    trainer.fit(self.model, train_dataloader, val_dataloader)

  def predict(self, data: pd.DataFrame) -> torch.Tensor:
    if self.model is None or self.tft_dataset is None:
      raise ValueError("Model not built or dataset not prepared.")

    # Create a dataloader for prediction
    predict_dataset = TimeSeriesDataSet.from_dataset(
      self.tft_dataset, data, predict=True, stop_after_last_index=True)
    predict_dataloader = predict_dataset.to_dataloader(
      train=False, batch_size=1)

    # Raw predictions are quantile predictions
    raw_predictions, _ = self.model.predict(predict_dataloader, return_x=True)
    return raw_predictions

  def save_model(self, path: str):
    if self.model:
      torch.save(self.model.state_dict(), path)

  def load_model(self, path: str, map_location=None):
    self.build_model()  # Build model structure first
    try:
      self.model.load_state_dict(torch.load(path, map_location=map_location))
      self.model.eval()  # Set to evaluation mode
      print(f"Successfully loaded model from {path}")
    except FileNotFoundError:
      print(
        f"Warning: Model file not found at {path}. A new model will be trained.")
    except Exception as e:
      print(f"Error loading model: {e}. A new model will be trained.")
