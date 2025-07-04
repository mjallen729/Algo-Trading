import torch
import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from src.config import (
    TFT_HIDDEN_SIZE,
    TFT_LSTM_LAYERS,
    TFT_ATTENTION_HEADS,
    TFT_DROPOUT,
    TFT_LEARNING_RATE,
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
        df[time_idx_col] = df.groupby(group_id_col, observed=True).cumcount()

        # Add time-based features that are known in the future
        df["month"] = df.index.month.astype(str).astype("category")
        df["day"] = df.index.day.astype(str).astype("category")
        df["weekday"] = df.index.dayofweek.astype(str).astype("category")
        df["hour"] = df.index.hour.astype(str).astype("category")

        static_features = []
        time_varying_known_categoricals = ["month", "day", "weekday", "hour"]
        time_varying_unknown_reals = [
            target_col, 'open', 'high', 'low', 'volume',
            'SMA_10', 'EMA_10', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR', 'OBV', 'Volatility'
        ]
        # Filter out columns that might not be present (e.g., if TA-lib fails)
        time_varying_unknown_reals = [col for col in time_varying_unknown_reals if col in df.columns]

        self.tft_dataset = TimeSeriesDataSet(
            df,
            time_idx=time_idx_col,
            target=target_col,
            group_ids=[group_id_col],
            min_encoder_length=self.max_encoder_length,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=self.max_prediction_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_features,
            time_varying_known_categoricals=time_varying_known_categoricals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=[group_id_col], transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True
        )

        if self.training_cutoff is not None:
            self.training_data = self.tft_dataset.filter(lambda x: x[time_idx_col] <= self.training_cutoff, deepcopy=True)
            self.validation_data = self.tft_dataset.filter(lambda x: x[time_idx_col] > self.training_cutoff, deepcopy=True)
        else:
            self.training_data = self.tft_dataset
            self.validation_data = self.tft_dataset # Validate on the whole dataset if no cutoff

    def build_model(self):
        if self.tft_dataset is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
            
        self.model = TemporalFusionTransformer.from_dataset(
            self.tft_dataset,
            hidden_size=TFT_HIDDEN_SIZE,
            lstm_layers=TFT_LSTM_LAYERS,
            attention_head_size=TFT_ATTENTION_HEADS,
            dropout=TFT_DROPOUT,
            learning_rate=TFT_LEARNING_RATE,
            loss=QuantileLoss(),
            optimizer="Ranger",
            reduce_on_plateau_patience=4,
        )

    def train_model(self, trainer):
        if self.training_data is None or self.validation_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        train_dataloader = self.training_data.to_dataloader(train=True, batch_size=128, num_workers=0)
        val_dataloader = self.validation_data.to_dataloader(train=False, batch_size=128, num_workers=0)

        trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, data: pd.DataFrame) -> torch.Tensor:
        if self.model is None or self.tft_dataset is None:
            raise ValueError("Model not built or dataset not prepared.")

        # Create a dataloader for prediction
        predict_dataset = TimeSeriesDataSet.from_dataset(self.tft_dataset, data, predict=True, stop_after_last_index=True)
        predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=1)

        # Raw predictions are quantile predictions
        raw_predictions, _ = self.model.predict(predict_dataloader, return_x=True)
        return raw_predictions

    def save_model(self, path: str):
        if self.model:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, map_location=None):
        self.build_model() # Build model structure first
        try:
            self.model.load_state_dict(torch.load(path, map_location=map_location))
            self.model.eval() # Set to evaluation mode
            print(f"Successfully loaded model from {path}")
        except FileNotFoundError:
            print(f"Warning: Model file not found at {path}. A new model will be trained.")
        except Exception as e:
            print(f"Error loading model: {e}. A new model will be trained.")
