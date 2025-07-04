import pandas as pd
import torch
from src.models.tft_model import TFTModel
from src.config import TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH

class Predictor:
    def __init__(self, model_path: str, tft_dataset):
        self.tft_model = TFTModel(TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH)
        self.tft_model.tft_dataset = tft_dataset # Pass the dataset structure
        self.tft_model.load_model(model_path)

    def predict_direction(self, data: pd.DataFrame) -> str:
        # Ensure data has 'symbol' and 'time_idx' for TFTModel's predict method
        # This assumes 'data' is already preprocessed and contains necessary features
        
        # Generate raw predictions
        raw_predictions = self.tft_model.predict(data)
        
        # For simplicity, let's assume we are predicting the next 'close' price
        # and we want to determine if it's 'UP', 'DOWN', or 'SIDEWAYS'
        # This is a very basic interpretation and should be refined.
        
        # Get the median prediction from the quantiles
        predicted_price = raw_predictions.mean().item() # Or use a specific quantile like .median()
        
        # Get the last known close price from the input data
        last_close_price = data['close'].iloc[-1]
        
        if predicted_price > last_close_price * 1.001: # 0.1% threshold for UP
            return "UP"
        elif predicted_price < last_close_price * 0.999: # 0.1% threshold for DOWN
            return "DOWN"
        else:
            return "SIDEWAYS"

if __name__ == '__main__':
    # Example Usage (requires a trained model and dummy data)
    # This is a simplified example. In a real scenario, you'd have a saved model.
    
    # 1. Create dummy data (similar to tft_model.py example)
    import numpy as np
    from src.feature_engineering.features import FeatureEngineer
    from pytorch_forecasting import TimeSeriesDataSet

    n_samples = 100
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    dummy_data = {
        'open': np.random.rand(n_samples) * 100,
        'high': np.random.rand(n_samples) * 100 + 1,
        'low': np.random.rand(n_samples) * 100 - 1,
        'close': np.random.rand(n_samples) * 100,
        'volume': np.random.rand(n_samples) * 1000,
        'symbol': ['BTC'] * n_samples
    }
    dummy_df = pd.DataFrame(dummy_data, index=dates)
    dummy_df.index.name = 'Date'

    feature_engineer = FeatureEngineer()
    engineered_df = feature_engineer.engineer_features(dummy_df.copy())
    engineered_df['time_idx'] = engineered_df.groupby('symbol').cumcount()

    # 2. Create a dummy TFTModel and save it (for demonstration)
    # In a real scenario, this model would be trained and saved separately.
    tft_model_dummy = TFTModel(TFT_MAX_ENCODER_LENGTH, TFT_MAX_PREDICTION_LENGTH)
    tft_model_dummy.prepare_data(engineered_df, target_col='close')
    tft_model_dummy.build_model()
    
    # Create a dummy model file
    dummy_model_path = "dummy_tft_model.pth"
    torch.save(tft_model_dummy.model.state_dict(), dummy_model_path)

    # 3. Initialize Predictor with the dummy model and dataset structure
    predictor = Predictor(dummy_model_path, tft_model_dummy.tft_dataset)

    # 4. Prepare data for prediction (last `max_encoder_length` rows)
    prediction_input_data = engineered_df.iloc[-TFT_MAX_ENCODER_LENGTH:].copy()
    
    # 5. Get prediction
    predicted_direction = predictor.predict_direction(prediction_input_data)
    print(f"\nPredicted direction: {predicted_direction}")

    # Clean up dummy model file
    import os
    os.remove(dummy_model_path)
