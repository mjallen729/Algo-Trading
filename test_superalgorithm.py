#!/usr/bin/env python3
"""
Test script for the cryptocurrency trading superalgorithm.
"""
import sys
import asyncio
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils import get_logger, config
from data import DataLoader, DataPreprocessor
from models import HybridPredictor, RegimeDetector
from strategies import MomentumStrategy, MeanReversionStrategy
from execution import TradingEngine
from risk import RiskManager

logger = get_logger(__name__)


async def test_data_loading():
    """Test data loading and preprocessing."""
    logger.info("Testing data loading and preprocessing...")
    
    try:
        # Test data loader
        data_loader = DataLoader()
        btc_data = data_loader.load_historical_data('BTC')
        
        if len(btc_data) > 0:
            logger.info(f"✅ Data loading successful: {len(btc_data)} BTC samples")
        else:
            logger.error("❌ No data loaded")
            return False
        
        # Test data preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_market_data(
            btc_data, 
            add_technical_indicators=True,
            add_time_features=True
        )
        
        logger.info(f"✅ Data preprocessing successful: {len(processed_data.columns)} features")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data loading test failed: {e}")
        return False


async def test_ml_models():
    """Test ML model training and prediction."""
    logger.info("Testing ML models...")
    
    try:
        # Load and preprocess data
        data_loader = DataLoader()
        preprocessor = DataPreprocessor()
        
        btc_data = data_loader.load_historical_data('BTC')
        processed_data = preprocessor.preprocess_market_data(btc_data)
        
        if len(processed_data) < 100:
            logger.error("❌ Insufficient data for model training")
            return False
        
        # Test regime detector
        regime_detector = RegimeDetector()
        regime_result = regime_detector.fit(processed_data)
        
        if regime_result['n_samples'] > 0:
            logger.info("✅ Regime detector training successful")
            
            # Test regime detection
            current_regime = regime_detector.detect_regime(processed_data)
            logger.info(f"✅ Current regime detected: {current_regime['regime_label']}")
        else:
            logger.error("❌ Regime detector training failed")
            return False
        
        # Test hybrid predictor (simplified for speed)
        logger.info("⏳ Testing hybrid predictor (this may take a moment)...")
        hybrid_predictor = HybridPredictor()
        
        # Train with small subset for testing
        test_data = processed_data.tail(200)  # Use last 200 samples
        
        # Train models with reduced epochs for testing
        history = hybrid_predictor.train(test_data, epochs=5, batch_size=16)
        
        if history:
            logger.info("✅ Hybrid predictor training successful")
            
            # Test prediction
            prediction, confidence = hybrid_predictor.predict(test_data)
            logger.info(f"✅ Prediction generated: {prediction:.2f} (confidence: {confidence:.3f})")
        else:
            logger.error("❌ Hybrid predictor training failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ ML models test failed: {e}")
        return False


async def test_strategies():
    """Test trading strategies."""
    logger.info("Testing trading strategies...")
    
    try:
        # Load test data
        data_loader = DataLoader()
        btc_data = data_loader.load_historical_data('BTC')
        
        if len(btc_data) < 50:
            logger.error("❌ Insufficient data for strategy testing")
            return False
        
        current_price = btc_data['Close'].iloc[-1]
        
        # Test momentum strategy
        momentum_strategy = MomentumStrategy()
        momentum_signal = momentum_strategy.generate_signal(
            data=btc_data,
            current_price=current_price,
            regime="Trending Up"
        )
        
        if momentum_signal:
            logger.info(f"✅ Momentum strategy signal: {momentum_signal.signal_type.value} "
                       f"(confidence: {momentum_signal.confidence:.3f})")
        else:
            logger.error("❌ Momentum strategy failed")
            return False
        
        # Test mean reversion strategy
        mean_reversion_strategy = MeanReversionStrategy()
        mr_signal = mean_reversion_strategy.generate_signal(
            data=btc_data,
            current_price=current_price,
            regime="Ranging"
        )
        
        if mr_signal:
            logger.info(f"✅ Mean reversion strategy signal: {mr_signal.signal_type.value} "
                       f"(confidence: {mr_signal.confidence:.3f})")
        else:
            logger.error("❌ Mean reversion strategy failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Strategy test failed: {e}")
        return False


async def test_execution_system():
    """Test execution and risk management."""
    logger.info("Testing execution system...")
    
    try:
        # Initialize trading engine
        trading_engine = TradingEngine()
        await trading_engine.initialize()
        
        if trading_engine.is_initialized:
            logger.info("✅ Trading engine initialized")
        else:
            logger.error("❌ Trading engine initialization failed")
            return False
        
        # Test risk manager
        risk_manager = RiskManager()
        
        # Create test signal
        from strategies.base import TradingSignal, SignalType
        test_signal = TradingSignal(
            signal_type=SignalType.BUY,
            confidence=0.75,
            price=50000.0,
            metadata={'test': True}
        )
        
        # Test risk evaluation
        risk_approved = risk_manager.evaluate_trade_risk(
            signal=test_signal,
            asset='BTC',
            current_price=50000.0,
            regime='Trending Up'
        )
        
        if risk_approved:
            logger.info("✅ Risk management approval successful")
        else:
            logger.info("ℹ️  Risk management rejection (expected for test)")
        
        # Test portfolio tracking
        portfolio_metrics = trading_engine.portfolio.get_performance_metrics()
        logger.info(f"✅ Portfolio metrics: ${portfolio_metrics['current_value']:.2f}")
        
        # Cleanup
        await trading_engine.shutdown()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Execution system test failed: {e}")
        return False


async def run_integration_test():
    """Run a simple integration test."""
    logger.info("Running integration test...")
    
    try:
        # Load minimal data
        data_loader = DataLoader()
        btc_data = data_loader.load_historical_data('BTC')
        
        if len(btc_data) < 50:
            logger.error("❌ Insufficient data for integration test")
            return False
        
        # Quick preprocessing
        preprocessor = DataPreprocessor()
        processed_data = preprocessor.preprocess_market_data(btc_data)
        
        # Quick regime detection
        regime_detector = RegimeDetector(n_regimes=3)
        regime_detector.fit(processed_data)
        regime_result = regime_detector.detect_regime(processed_data)
        
        # Generate strategy signal
        strategy = MomentumStrategy()
        signal = strategy.generate_signal(
            data=btc_data,
            current_price=btc_data['Close'].iloc[-1],
            regime=regime_result['regime_label']
        )
        
        # Risk check
        risk_manager = RiskManager()
        risk_approved = risk_manager.evaluate_trade_risk(
            signal=signal,
            asset='BTC',
            current_price=btc_data['Close'].iloc[-1]
        )
        
        logger.info(f"✅ Integration test successful:")
        logger.info(f"   Regime: {regime_result['regime_label']}")
        logger.info(f"   Signal: {signal.signal_type.value} (confidence: {signal.confidence:.3f})")
        logger.info(f"   Risk Approved: {risk_approved}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration test failed: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("🚀 Starting Cryptocurrency Superalgorithm Tests")
    logger.info("=" * 60)
    
    test_results = {}
    
    # Run tests
    test_results['data_loading'] = await test_data_loading()
    test_results['strategies'] = await test_strategies()
    test_results['execution_system'] = await test_execution_system()
    test_results['integration'] = await run_integration_test()
    
    # Optional: Run ML model test (takes longer)
    run_ml_test = input("\nRun ML model test? (takes 2-3 minutes) [y/N]: ").lower().strip() == 'y'
    if run_ml_test:
        test_results['ml_models'] = await test_ml_models()
    else:
        logger.info("⏭️  Skipping ML model test")
        test_results['ml_models'] = None
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("🎯 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = 0
    
    for test_name, result in test_results.items():
        if result is not None:
            total += 1
            if result:
                passed += 1
                status = "✅ PASSED"
            else:
                status = "❌ FAILED"
        else:
            status = "⏭️  SKIPPED"
        
        logger.info(f"{test_name.replace('_', ' ').title():<20} {status}")
    
    logger.info("=" * 60)
    if total > 0:
        success_rate = (passed / total) * 100
        logger.info(f"Overall Success Rate: {passed}/{total} ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            logger.info("🎉 Superalgorithm core systems are functional!")
        elif success_rate >= 60:
            logger.info("⚠️  Most systems working, some issues to resolve")
        else:
            logger.info("🔧 Significant issues detected, debugging needed")
    
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
