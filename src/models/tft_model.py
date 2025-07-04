import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
import pandas as pd

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
        # Ensure time_idx is sequential for PyTorch Forecasting
        df[time_idx_col] = df.groupby(group_id_col).cumcount()

        # Define static and time-varying features
        # For MVP, we'll use a simplified set. Expand as needed.
        static_features = [] # No static features for now, assuming single asset
        time_varying_known_reals = [] # No known future inputs for now
        time_varying_unknown_reals = [target_col] + [col for col in df.columns if col not in [time_idx_col, target_col, group_id_col] and df[col].dtype in [np.float32, np.float64, np.int32, np.int64]]

        self.tft_dataset = TimeSeriesDataSet(
            df,
            time_idx=time_idx_col,
            target=target_col,
            group_ids=[group_id_col],
            min_encoder_length=self.max_encoder_length // 2,
            max_encoder_length=self.max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=static_features,
            time_varying_known_reals=time_varying_known_reals,
            time_varying_unknown_reals=time_varying_unknown_reals,
            target_normalizer=GroupNormalizer(groups=[group_id_col], transformation_type="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
            allow_missing_timesteps=True # Important for real-world data
        )

        if self.training_cutoff is not None:
            self.training_data = self.tft_dataset.filter(lambda x: x[time_idx_col] <= self.training_cutoff, deepcopy=True)
            self.validation_data = self.tft_dataset.filter(lambda x: x[time_idx_col] > self.training_cutoff, deepcopy=True)
        else:
            self.training_data = self.tft_dataset
            self.validation_data = None # Or split manually if needed

    def build_model(self):
        self.model = TemporalFusionTransformer.from_dataset(
            self.tft_dataset,
            # Set parameters for the model. These are examples, tune as needed.
            hidden_size=16,
            lstm_layers=1,
            dropout=0.1,
            attention_head_size=4,
            learning_rate=1e-3,
            loss=QuantileLoss(),
            optimizer="Ranger",
            # reduce learning rate if no improvement in validation loss over 10 epochs
            reduce_on_plateau_patience=10,
        )

    def train_model(self, trainer):
        if self.training_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        train_dataloader = self.training_data.to_dataloader(train=True, batch_size=64)
        val_dataloader = None
        if self.validation_data:
            val_dataloader = self.validation_data.to_dataloader(train=False, batch_size=64)

        trainer.fit(self.model, train_dataloader, val_dataloader)

    def predict(self, data: pd.DataFrame) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Model not built or trained. Call build_model() and train_model() first.")
        
        # Ensure the prediction data has the same structure as training data
        # This might require careful handling of time_idx and other features
        # For simplicity, assuming 'data' is already preprocessed and ready
        
        # Create a TimeSeriesDataSet for prediction
        predict_dataset = TimeSeriesDataSet.from_dataset(self.tft_dataset, data, predict=True, stop_after_last_index=True)
        predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=64)

        predictions = self.model.predict(predict_dataloader)
        return predictions

    def save_model(self, path: str):
        if self.model:
            torch.save(self.model.state_dict(), path)

    def load_model(self, path: str, map_location=None):
        self.build_model() # Build model structure first
        self.model.load_state_dict(torch.load(path, map_location=map_location))
        self.model.eval() # Set to evaluation mode

if __name__ == '__main__':
    # Example Usage (requires dummy data with 'symbol' and 'time_idx')
    from src.feature_engineering.features import FeatureEngineer
    
    # Create dummy data
    n_samples = 1000
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    dummy_data = {
        'open': np.random.rand(n_samples) * 100,
        'high': np.random.rand(n_samples) * 100 + 1,
        'low': np.random.rand(n_samples) * 100 - 1,
        'close': np.random.rand(n_samples) * 100,
        'volume': np.random.rand(n_samples) * 1000,
        'symbol': ['BTC'] * n_samples # Assuming a single symbol for simplicity
    }
    dummy_df = pd.DataFrame(dummy_data, index=dates)
    dummy_df.index.name = 'Date'

    # Engineer features
    feature_engineer = FeatureEngineer()
    engineered_df = feature_engineer.engineer_features(dummy_df.copy())
    
    # Add a time_idx column for PyTorch Forecasting
    engineered_df['time_idx'] = engineered_df.groupby('symbol').cumcount()

    # Define training cutoff (e.g., 80% for training, 20% for validation)
    training_cutoff = engineered_df['time_idx'].max() - int(0.2 * len(engineered_df))

    tft_model = TFTModel(
        max_encoder_length=24, 
        max_prediction_length=1, 
        training_cutoff=training_cutoff
    )
    tft_model.prepare_data(engineered_df, target_col='close')
    tft_model.build_model()

    # Dummy trainer for example (in real scenario, use pytorch_lightning.Trainer)
    class DummyTrainer:
        def fit(self, model, train_dataloader, val_dataloader):
            print("Training model (dummy fit)...")
            # In a real scenario, this would run the training loop
            pass

    dummy_trainer = DummyTrainer()
    tft_model.train_model(dummy_trainer)

    # Example prediction
    # For prediction, you'd typically pass the last `max_encoder_length` data points
    # For this example, let's just take the last few rows of the engineered_df
    prediction_data = engineered_df.iloc[-tft_model.max_encoder_length:]
    predictions = tft_model.predict(prediction_data)
    print("\nExample Prediction (first 5 values):", predictions.flatten()[:5].detach().numpy())
