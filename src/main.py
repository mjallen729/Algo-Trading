"""
Main entry point for the cryptocurrency trading superalgorithm.
"""
import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, List

# Add src to path
sys.path.append(str(Path(__file__).parent))

from utils import get_logger, config
from data import DataLoader, DataPreprocessor
from models import HybridPredictor, RegimeDetector, model_paths
from strategies import MomentumStrategy, MeanReversionStrategy, ArbitrageStrategy
from execution import TradingEngine
from risk import RiskManager

logger = get_logger(__name__)


class SuperAlgorithm:
    """
    Main cryptocurrency trading superalgorithm orchestrator.
    
    Integrates all components:
    - Data loading and preprocessing
    - ML model predictions
    - Market regime detection
    - Strategy selection and execution
    - Risk management
    """
    
    def __init__(self):
        """Initialize the superalgorithm."""
        logger.info("Initializing Cryptocurrency Superalgorithm...")
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_preprocessor = DataPreprocessor()
        self.hybrid_predictor = HybridPredictor()
        self.regime_detector = RegimeDetector()
        self.risk_manager = RiskManager()
        
        # Initialize strategies
        self.strategies = {
            'momentum': MomentumStrategy(),
            'mean_reversion': MeanReversionStrategy(),
            'arbitrage': ArbitrageStrategy()
        }
        
        # Initialize trading engine
        self.trading_engine = TradingEngine()
        
        # Algorithm state
        self.is_running = False
        self.target_assets = config.get('trading.target_assets', ['BTC', 'ETH', 'SOL'])
        self.models_trained = False
        
        logger.info("Superalgorithm initialization completed")
    
    async def initialize_models(self):
        """Initialize and train ML models."""
        logger.info("Initializing ML models...")
        
        try:
            # Check for existing trained models
            if self._check_existing_models():
                logger.info("Loading existing trained models...")
                self._load_existing_models()
                return
            
            # Train new models if none exist
            logger.info("No existing models found, training new models...")
            await self._train_new_models()
            
        except Exception as e:
            logger.error(f"Error during model initialization: {e}")
            raise
    
    def _check_existing_models(self) -> bool:
        """Check if trained models already exist."""
        try:
            ensemble_path = model_paths.get_ensemble_path()
            regime_path = model_paths.get_regime_detector_path()
            
            import os
            return (os.path.exists(ensemble_path) and 
                   os.path.exists(regime_path))
        except Exception:
            return False
    
    def _load_existing_models(self):
        """Load existing trained models."""
        try:
            # Load models using centralized paths
            self.hybrid_predictor.load_ensemble()
            self.regime_detector.load_models()
            
            self.models_trained = True
            logger.info("Existing models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading existing models: {e}")
            raise
    
    async def _train_new_models(self):
        """Train new models from scratch."""
        try:
            # Load historical data for training
            training_data = {}
            for asset in self.target_assets:
                data = self.data_loader.load_historical_data(asset)
                if len(data) > 100:  # Ensure sufficient data
                    # Preprocess data
                    processed_data = self.data_preprocessor.preprocess_market_data(
                        data, 
                        add_technical_indicators=True,
                        add_time_features=True,
                        normalize=True
                    )
                    training_data[asset] = processed_data
                    logger.info(f"Loaded {len(data)} samples for {asset}")
            
            if not training_data:
                raise ValueError("No training data available")
            
            # Train models on BTC data (primary asset)
            primary_asset = 'BTC'
            if primary_asset in training_data:
                primary_data = training_data[primary_asset]
                
                # Train prediction models
                logger.info("Training hybrid prediction model...")
                self.hybrid_predictor.train(primary_data)
                
                # Train regime detector
                logger.info("Training regime detection model...")
                self.regime_detector.fit(primary_data)
                
                # Save trained models using centralized paths
                logger.info("Saving trained models...")
                self.hybrid_predictor.save_ensemble()
                self.regime_detector.save_models()
                
                self.models_trained = True
                logger.info("All models trained and saved successfully")
            else:
                logger.error(f"Primary asset {primary_asset} not available for training")
                
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise
    
    async def run_trading_cycle(self):
        """Execute one complete trading cycle."""
        try:
            for asset in self.target_assets:
                await self._process_asset(asset)
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
    
    async def _process_asset(self, asset: str):
        """Process a single asset for trading opportunities."""
        try:
            # Load recent data
            recent_data = self.data_loader.load_historical_data(asset)
            if len(recent_data) < 100:
                logger.warning(f"Insufficient data for {asset}")
                return
            
            # Preprocess data
            processed_data = self.data_preprocessor.preprocess_market_data(
                recent_data,
                add_technical_indicators=True,
                add_time_features=True,
                normalize=True
            )
            
            current_price = recent_data['Close'].iloc[-1]
            
            # Market regime detection
            regime_result = self.regime_detector.detect_regime(processed_data)
            current_regime = regime_result['regime_label']
            
            logger.info(f"{asset} - Current regime: {current_regime}")
            
            # ML prediction
            prediction, confidence = self.hybrid_predictor.predict(processed_data)
            
            logger.info(f"{asset} - Prediction: {prediction:.2f} (confidence: {confidence:.3f})")
            
            # Strategy selection based on regime
            active_strategy = self._select_strategy(current_regime)
            
            # Generate trading signal
            signal = active_strategy.generate_signal(
                data=processed_data,
                current_price=current_price,
                regime=current_regime,
                prediction=prediction,
                confidence=confidence
            )
            
            logger.info(f"{asset} - Signal: {signal}")
            
            # Risk management check
            risk_approved = self.risk_manager.evaluate_trade_risk(
                signal=signal,
                asset=asset,
                current_price=current_price,
                regime=current_regime
            )
            
            if risk_approved:
                # Execute trade
                await self.trading_engine.execute_signal(signal, asset)
            else:
                logger.info(f"{asset} - Trade rejected by risk management")
                
        except Exception as e:
            logger.error(f"Error processing {asset}: {e}")
    
    def _select_strategy(self, regime: str) -> object:
        """
        Select optimal strategy based on market regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Selected strategy object
        """
        if "Trending" in regime:
            return self.strategies['momentum']
        elif "Ranging" in regime or "Sideways" in regime:
            return self.strategies['mean_reversion']
        else:
            # Default to momentum for unknown regimes
            return self.strategies['momentum']
    
    async def start(self):
        """Start the trading algorithm."""
        logger.info("Starting Cryptocurrency Superalgorithm...")
        
        if not self.models_trained:
            await self.initialize_models()
        
        self.is_running = True
        
        # Initialize trading engine
        await self.trading_engine.initialize()
        
        # Main trading loop
        cycle_interval = config.get('trading.cycle_interval', 300)  # 5 minutes default
        
        while self.is_running:
            try:
                logger.info("Starting new trading cycle...")
                await self.run_trading_cycle()
                
                # Wait for next cycle
                await asyncio.sleep(cycle_interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying
        
        await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the algorithm."""
        logger.info("Shutting down Cryptocurrency Superalgorithm...")
        
        self.is_running = False
        
        # Close all positions
        await self.trading_engine.close_all_positions()
        
        # Shutdown trading engine
        await self.trading_engine.shutdown()
        
        # Generate final performance report
        performance_report = self.trading_engine.get_performance_report()
        logger.info(f"Final Performance Report: {performance_report}")
        
        logger.info("Superalgorithm shutdown completed")
    
    def get_status(self) -> Dict:
        """Get current algorithm status."""
        return {
            'is_running': self.is_running,
            'models_trained': self.models_trained,
            'target_assets': self.target_assets,
            'active_strategies': list(self.strategies.keys()),
            'trading_engine_status': self.trading_engine.get_status() if hasattr(self.trading_engine, 'get_status') else 'Unknown'
        }


async def main():
    """Main entry point."""
    # Setup signal handlers for graceful shutdown
    algorithm = SuperAlgorithm()
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}")
        asyncio.create_task(algorithm.shutdown())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await algorithm.start()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        await algorithm.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    # Run the algorithm
    asyncio.run(main())
