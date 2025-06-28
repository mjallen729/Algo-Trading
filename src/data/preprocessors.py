        
        # Remove duplicate timestamps
        if 'Timestamp' in cleaned_data.columns:
            cleaned_data = cleaned_data.drop_duplicates(subset=['Timestamp'])
        
        # Sort by timestamp if available
        if 'Date' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values('Date')
        elif 'Timestamp' in cleaned_data.columns:
            cleaned_data = cleaned_data.sort_values('Timestamp')
        
        # Validate OHLCV data
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in cleaned_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove rows with invalid OHLCV data
        cleaned_data = cleaned_data[
            (cleaned_data['High'] >= cleaned_data['Low']) &
            (cleaned_data['High'] >= cleaned_data['Open']) &
            (cleaned_data['High'] >= cleaned_data['Close']) &
            (cleaned_data['Low'] <= cleaned_data['Open']) &
            (cleaned_data['Low'] <= cleaned_data['Close']) &
            (cleaned_data['Volume'] >= 0)
        ]
        
        # Remove extreme outliers (more than 50% price change)
        returns = cleaned_data['Close'].pct_change()
        outlier_mask = (returns.abs() < 0.5)
        cleaned_data = cleaned_data[outlier_mask]
        
        return cleaned_data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators."""
        df = data.copy()
        
        # Price-based indicators
        df['SMA25'] = ta.trend.sma_indicator(df['Close'], window=25)
        df['EMA25'] = ta.trend.ema_indicator(df['Close'], window=25)
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
        
        # Momentum indicators
        df['RSI25'] = ta.momentum.rsi(df['Close'], window=25)
        df['RSI14'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'])
        
        # Volatility indicators
        df['ATR25'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=25)
        df['Volatility25'] = df['Close'].pct_change().rolling(25).std()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        df['BB_Middle'] = bollinger.bollinger_mavg()
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = ta.volume.volume_sma(df['Close'], df['Volume'])
        df['Volume_EMA'] = df['Volume'].ewm(span=20).mean()
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['CMF'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'])
        
        # Trend indicators
        df['ADX'] = ta.trend.adx(df['High'], df['Low'], df['Close'])
        df['CCI'] = ta.trend.cci(df['High'], df['Low'], df['Close'])
        df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'])
        
        # Stochastic oscillator
        df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'])
        df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'])
        
        return df
    
    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = data.copy()
        
        # Ensure we have a datetime index
        if 'Date' in df.columns:
            df['datetime'] = pd.to_datetime(df['Date'])
        elif 'Timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['Timestamp'])
        else:
            # Use index if it's datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df['datetime'] = df.index
            else:
                logger.warning("No datetime column found, skipping time features")
                return df
        
        # Extract time components
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['quarter'] = df['datetime'].dt.quarter
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Market session indicators (assuming UTC time)
        df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
        df['is_european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['is_american_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def _add_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add advanced engineered features."""
        df = data.copy()
        
        # Price momentum features
        df['returns_1h'] = df['Close'].pct_change(1)
        df['returns_4h'] = df['Close'].pct_change(4)
        df['returns_24h'] = df['Close'].pct_change(24)
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'returns_{window}h_mean'] = df['returns_1h'].rolling(window).mean()
            df[f'returns_{window}h_std'] = df['returns_1h'].rolling(window).std()
            df[f'volume_{window}h_mean'] = df['Volume'].rolling(window).mean()
            df[f'high_low_ratio_{window}h'] = (df['High'] / df['Low']).rolling(window).mean()
        
        # Price levels and ranges
        df['hl_pct'] = (df['High'] - df['Low']) / df['Close']
        df['oc_pct'] = (df['Close'] - df['Open']) / df['Open']
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Volume features
        df['volume_price_trend'] = df['returns_1h'] * (df['Volume'] / df['Volume'].rolling(20).mean())
        df['volume_volatility'] = df['Volume'].rolling(20).std() / df['Volume'].rolling(20).mean()
        
        # Momentum and trend features
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Moving average relationships
        df['price_above_sma25'] = (df['Close'] > df['SMA25']).astype(int)
        df['sma25_above_sma50'] = (df['SMA25'] > df['SMA50']).astype(int)
        df['price_sma25_ratio'] = df['Close'] / df['SMA25']
        df['sma25_sma50_ratio'] = df['SMA25'] / df['SMA50']
        
        # Volatility features
        df['volatility_ratio'] = df['Volatility25'] / df['Volatility25'].rolling(50).mean()
        df['atr_ratio'] = df['ATR25'] / df['Close']
        
        # Support and resistance levels
        df['resistance_20'] = df['High'].rolling(20).max()
        df['support_20'] = df['Low'].rolling(20).min()
        df['distance_to_resistance'] = (df['resistance_20'] - df['Close']) / df['Close']
        df['distance_to_support'] = (df['Close'] - df['support_20']) / df['Close']
        
        # Gap features
        df['gap_up'] = ((df['Open'] > df['Close'].shift(1)) & 
                       ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1) > 0.01)).astype(int)
        df['gap_down'] = ((df['Open'] < df['Close'].shift(1)) & 
                         ((df['Close'].shift(1) - df['Open']) / df['Close'].shift(1) > 0.01)).astype(int)
        
        return df
    
    def _normalize_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize numerical features."""
        df = data.copy()
        
        # Identify numerical columns to normalize (exclude time features and binary indicators)
        exclude_columns = ['Date', 'Timestamp', 'datetime', 'hour', 'day_of_week', 'day_of_month', 
                          'month', 'quarter', 'is_asian_session', 'is_european_session', 
                          'is_american_session', 'is_weekend', 'price_above_sma25', 
                          'sma25_above_sma50', 'gap_up', 'gap_down']
        
        numerical_columns = [col for col in df.columns 
                           if df[col].dtype in ['float64', 'int64'] 
                           and col not in exclude_columns]
        
        # Use RobustScaler for better handling of outliers
        scaler = RobustScaler()
        
        # Fit and transform numerical columns
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        
        # Store scaler for future use
        self.scalers['features'] = scaler
        
        return df
    
    def create_sequences(self, 
                        data: pd.DataFrame, 
                        sequence_length: int,
                        target_column: str = 'Close',
                        feature_columns: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series modeling.
        
        Args:
            data: Preprocessed data
            sequence_length: Length of input sequences
            target_column: Column to predict
            feature_columns: Columns to use as features
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        if feature_columns is None:
            feature_columns = [col for col in data.columns 
                             if col not in ['Date', 'Timestamp', 'datetime']]
        
        # Ensure target column is in features
        if target_column not in feature_columns:
            feature_columns.append(target_column)
        
        # Select features
        feature_data = data[feature_columns].values
        target_idx = feature_columns.index(target_column)
        
        X, y = [], []
        
        for i in range(sequence_length, len(feature_data)):
            # Input sequence
            X.append(feature_data[i-sequence_length:i])
            # Target (next value)
            y.append(feature_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def calculate_feature_importance(self, 
                                   data: pd.DataFrame,
                                   target_column: str = 'Close') -> pd.Series:
        """
        Calculate feature importance using correlation analysis.
        
        Args:
            data: Preprocessed data
            target_column: Target variable
            
        Returns:
            Series with feature importance scores
        """
        # Calculate returns for target
        target_returns = data[target_column].pct_change().dropna()
        
        # Calculate correlations with all features
        correlations = {}
        
        for column in data.columns:
            if column != target_column and pd.api.types.is_numeric_dtype(data[column]):
                try:
                    # Align series for correlation calculation
                    aligned_target, aligned_feature = target_returns.align(
                        data[column].dropna(), join='inner'
                    )
                    
                    if len(aligned_target) > 10:  # Need sufficient data points
                        correlation = abs(aligned_target.corr(aligned_feature))
                        correlations[column] = correlation if not pd.isna(correlation) else 0
                    else:
                        correlations[column] = 0
                        
                except Exception:
                    correlations[column] = 0
        
        importance = pd.Series(correlations).sort_values(ascending=False)
        return importance
    
    def get_feature_summary(self, data: pd.DataFrame) -> Dict:
        """
        Get summary statistics for all features.
        
        Args:
            data: Preprocessed data
            
        Returns:
            Dictionary with feature summary statistics
        """
        summary = {
            'total_features': len(data.columns),
            'total_samples': len(data),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'numerical_features': len(data.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(data.select_dtypes(include=['object', 'category']).columns)
        }
        
        # Add statistics for numerical columns
        numerical_data = data.select_dtypes(include=[np.number])
        summary['numerical_stats'] = {
            'mean': numerical_data.mean().to_dict(),
            'std': numerical_data.std().to_dict(),
            'min': numerical_data.min().to_dict(),
            'max': numerical_data.max().to_dict()
        }
        
        return summary
