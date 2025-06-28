#!/usr/bin/env python3
"""
Test script to verify all components import correctly and basic functionality works.
"""

import sys
import os
import traceback

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_imports():
    """Test all component imports."""
    print("🧪 Testing component imports...")
    
    try:
        from src.data.preprocessors import AdvancedPreprocessor
        print("✅ AdvancedPreprocessor imported successfully")
    except Exception as e:
        print(f"❌ AdvancedPreprocessor import failed: {e}")
        traceback.print_exc()
    
    try:
        from src.models.regime_detector import RegimeDetector
        print("✅ RegimeDetector imported successfully")
    except Exception as e:
        print(f"❌ RegimeDetector import failed: {e}")
        traceback.print_exc()
    
    try:
        from src.risk.position_sizing import AdvancedPositionSizer, TradingSignal
        print("✅ AdvancedPositionSizer and TradingSignal imported successfully")
    except Exception as e:
        print(f"❌ AdvancedPositionSizer import failed: {e}")
        traceback.print_exc()
    
    try:
        from src.data.loaders import DataLoader
        print("✅ DataLoader imported successfully")
    except Exception as e:
        print(f"❌ DataLoader import failed: {e}")
        traceback.print_exc()
    
    try:
        from src.utils.logger import get_logger
        print("✅ Logger utility imported successfully")
    except Exception as e:
        print(f"❌ Logger utility import failed: {e}")
        traceback.print_exc()

def test_basic_functionality():
    """Test basic functionality of core components."""
    print("\n🔧 Testing basic functionality...")
    
    try:
        # Test data loading
        from src.data.loaders import DataLoader
        loader = DataLoader()
        print("✅ DataLoader instantiated successfully")
        
        # Load sample data
        data = loader.load_historical_data("btc")
        print(f"✅ Sample data loaded: {data.shape[0]} rows, {data.shape[1]} columns")
        
    except Exception as e:
        print(f"❌ DataLoader test failed: {e}")
        traceback.print_exc()
    
    try:
        # Test preprocessor
        from src.data.preprocessors import AdvancedPreprocessor
        preprocessor = AdvancedPreprocessor()
        print("✅ AdvancedPreprocessor instantiated successfully")
        
    except Exception as e:
        print(f"❌ AdvancedPreprocessor test failed: {e}")
        traceback.print_exc()
    
    try:
        # Test regime detector
        from src.models.regime_detector import RegimeDetector
        regime_detector = RegimeDetector()
        print("✅ RegimeDetector instantiated successfully")
        
    except Exception as e:
        print(f"❌ RegimeDetector test failed: {e}")
        traceback.print_exc()
    
    try:
        # Test position sizer
        from src.risk.position_sizing import AdvancedPositionSizer, TradingSignal
        import datetime
        
        position_sizer = AdvancedPositionSizer()
        print("✅ AdvancedPositionSizer instantiated successfully")
        
        # Create a sample trading signal
        signal = TradingSignal(
            action="buy",
            symbol="BTC",
            price=50000.0,
            confidence=0.75,
            strategy="test",
            timestamp=datetime.datetime.now()
        )
        
        # Test position sizing
        position_size = position_sizer.calculate_position_size(
            signal=signal,
            symbol="BTC", 
            portfolio_value=10000.0
        )
        print(f"✅ Position sizing test: ${position_size:.2f} position calculated")
        
    except Exception as e:
        print(f"❌ Position sizing test failed: {e}")
        traceback.print_exc()

def test_data_flow():
    """Test the complete data processing flow."""
    print("\n🚀 Testing complete data flow...")
    
    try:
        # Load data
        from src.data.loaders import DataLoader
        from src.data.preprocessors import AdvancedPreprocessor
        
        loader = DataLoader()
        preprocessor = AdvancedPreprocessor()
        
        # Load sample data
        raw_data = loader.load_historical_data("btc")
        print(f"✅ Raw data loaded: {raw_data.shape}")
        
        # Preprocess data (minimal config to avoid errors)
        config = {
            'add_technical_indicators': False,  # Skip TA-Lib for now
            'add_time_features': True,
            'add_advanced_features': False,
            'normalize_features': False
        }
        preprocessor = AdvancedPreprocessor(config)
        
        processed_data = preprocessor.preprocess(raw_data.head(100))  # Use first 100 rows
        print(f"✅ Data preprocessed: {processed_data.shape}")
        
        # Get feature summary
        summary = preprocessor.get_feature_summary(processed_data)
        print(f"✅ Feature summary generated: {summary['total_features']} features")
        
    except Exception as e:
        print(f"❌ Data flow test failed: {e}")
        traceback.print_exc()

def main():
    """Main test runner."""
    print("🎯 Cryptocurrency Superalgorithm - Component Testing")
    print("=" * 60)
    
    test_imports()
    test_basic_functionality()
    test_data_flow()
    
    print("\n✨ Testing completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
