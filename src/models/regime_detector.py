        """
        logger.info("Fitting regime detection models...")
        
        # Extract features for regime detection
        features = self.extract_regime_features(data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit HMM model
        try:
            self.hmm_model.fit(scaled_features)
            hmm_fitted = True
        except Exception as e:
            logger.warning(f"HMM fitting failed: {e}")
            hmm_fitted = False
        
        # Fit K-means model
        self.kmeans_model.fit(scaled_features)
        
        # Fit GMM model
        self.gmm_model.fit(scaled_features)
        
        self.is_fitted = True
        
        # Get regime predictions for evaluation
        regimes = self.detect_regime(data)
        
        results = {
            'hmm_fitted': hmm_fitted,
            'n_samples': len(features),
            'features_used': self.feature_columns,
            'regime_distribution': {
                regime: count for regime, count in 
                pd.Series(regimes['ensemble_regime']).value_counts().items()
            }
        }
        
        logger.info("Regime detection models fitted successfully")
        return results
    
    def detect_regime(self, data: pd.DataFrame) -> Dict:
        """
        Detect current market regime using ensemble approach.
        
        Args:
            data: Recent market data
            
        Returns:
            Dictionary with regime predictions and probabilities
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before regime detection")
        
        # Extract features
        features = self.extract_regime_features(data)
        scaled_features = self.scaler.transform(features)
        
        # Get predictions from all models
        results = {}
        
        # HMM prediction
        try:
            hmm_regimes = self.hmm_model.predict(scaled_features)
            hmm_probs = self.hmm_model.predict_proba(scaled_features)
            results['hmm_regime'] = hmm_regimes[-1]  # Most recent
            results['hmm_probabilities'] = hmm_probs[-1]
        except:
            results['hmm_regime'] = 1  # Default to ranging
            results['hmm_probabilities'] = np.array([0.33, 0.34, 0.33])
        
        # K-means prediction
        kmeans_regimes = self.kmeans_model.predict(scaled_features)
        results['kmeans_regime'] = kmeans_regimes[-1]
        
        # GMM prediction
        gmm_regimes = self.gmm_model.predict(scaled_features)
        gmm_probs = self.gmm_model.predict_proba(scaled_features)
        results['gmm_regime'] = gmm_regimes[-1]
        results['gmm_probabilities'] = gmm_probs[-1]
        
        # Ensemble prediction (weighted voting)
        ensemble_regime = self._ensemble_regime_prediction(results)
        results['ensemble_regime'] = ensemble_regime
        
        # Add regime labels
        results['regime_label'] = self.regime_labels[ensemble_regime]
        
        return results
    
    def _ensemble_regime_prediction(self, predictions: Dict) -> int:
        """
        Combine predictions from multiple models using weighted voting.
        
        Args:
            predictions: Dictionary with model predictions
            
        Returns:
            Ensemble regime prediction
        """
        # Weights for different models (can be tuned based on performance)
        weights = {'hmm': 0.4, 'kmeans': 0.3, 'gmm': 0.3}
        
        # Create voting array
        regime_votes = np.zeros(self.n_regimes)
        
        # Add weighted votes
        if 'hmm_regime' in predictions:
            regime_votes[predictions['hmm_regime']] += weights['hmm']
        
        regime_votes[predictions['kmeans_regime']] += weights['kmeans']
        regime_votes[predictions['gmm_regime']] += weights['gmm']
        
        # Return regime with highest vote
        return np.argmax(regime_votes)
    
    def get_regime_transition_probability(self, data: pd.DataFrame) -> Dict:
        """
        Calculate probability of regime transition in next period.
        
        Args:
            data: Recent market data
            
        Returns:
            Dictionary with transition probabilities
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before calculating transitions")
        
        current_regime = self.detect_regime(data)['ensemble_regime']
        
        # Get HMM transition matrix if available
        try:
            transition_matrix = self.hmm_model.transmat_
            transition_probs = transition_matrix[current_regime]
        except:
            # Default uniform transition probabilities
            transition_probs = np.ones(self.n_regimes) / self.n_regimes
        
        return {
            'current_regime': current_regime,
            'transition_probabilities': {
                self.regime_labels[i]: prob 
                for i, prob in enumerate(transition_probs)
            }
        }
    
    def analyze_regime_characteristics(self, data: pd.DataFrame) -> Dict:
        """
        Analyze characteristics of detected regimes.
        
        Args:
            data: Historical market data
            
        Returns:
            Dictionary with regime analysis
        """
        if not self.is_fitted:
            raise ValueError("Models must be fitted before analysis")
        
        # Get regime sequence
        regimes_history = []
        features = self.extract_regime_features(data)
        
        # Get regime for each period
        for i in range(len(features)):
            if i < 50:  # Need minimum data for regime detection
                continue
            
            period_data = data.iloc[:i+1]
            regime_result = self.detect_regime(period_data)
            regimes_history.append(regime_result['ensemble_regime'])
        
        regimes_series = pd.Series(regimes_history, index=data.index[-len(regimes_history):])
        
        # Analyze each regime
        analysis = {}
        
        for regime in range(self.n_regimes):
            regime_mask = regimes_series == regime
            regime_data = data.loc[regime_mask]
            
            if len(regime_data) > 0:
                returns = regime_data['Close'].pct_change()
                
                analysis[self.regime_labels[regime]] = {
                    'frequency': regime_mask.sum() / len(regimes_series),
                    'avg_return': returns.mean(),
                    'volatility': returns.std(),
                    'avg_volume': regime_data['Volume'].mean(),
                    'max_drawdown': self._calculate_max_drawdown(regime_data['Close']),
                    'avg_duration': self._calculate_avg_regime_duration(regimes_series, regime)
                }
        
        return analysis
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown for a price series."""
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_avg_regime_duration(self, regimes: pd.Series, target_regime: int) -> float:
        """Calculate average duration of a specific regime."""
        durations = []
        current_duration = 0
        
        for regime in regimes:
            if regime == target_regime:
                current_duration += 1
            else:
                if current_duration > 0:
                    durations.append(current_duration)
                current_duration = 0
        
        # Add final duration if ended in target regime
        if current_duration > 0:
            durations.append(current_duration)
        
        return np.mean(durations) if durations else 0
    
    def save_models(self, filepath: str):
        """Save regime detection models."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before saving")
        
        import pickle
        
        save_dict = {
            'hmm_model': self.hmm_model,
            'kmeans_model': self.kmeans_model,
            'gmm_model': self.gmm_model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'n_regimes': self.n_regimes,
            'regime_labels': self.regime_labels,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Regime detection models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load regime detection models."""
        import pickle
        
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.hmm_model = save_dict['hmm_model']
        self.kmeans_model = save_dict['kmeans_model']
        self.gmm_model = save_dict['gmm_model']
        self.scaler = save_dict['scaler']
        self.feature_columns = save_dict['feature_columns']
        self.n_regimes = save_dict['n_regimes']
        self.regime_labels = save_dict['regime_labels']
        self.is_fitted = save_dict['is_fitted']
        
        logger.info(f"Regime detection models loaded from {filepath}")
