"""
Test import resolution for the superalgorithm.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    print("Testing imports...")
    
    print("✓ Testing utils...")
    from utils import get_logger, config
    
    print("✓ Testing risk...")
    from risk import RiskManager
    
    print("✓ Testing execution...")
    from execution import TradingEngine
    
    print("✓ Testing data...")
    from data import DataLoader, AdvancedPreprocessor
    
    print("✓ Testing models...")
    from models import HybridPredictor, RegimeDetector
    
    print("✓ Testing strategies...")
    from strategies import MomentumStrategy, MeanReversionStrategy, ArbitrageStrategy
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("The superalgorithm is ready to start.")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("There are still import issues that need to be resolved.")
except Exception as e:
    print(f"❌ Unexpected Error: {e}")
