"""
Logging utilities for the trading system.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import os

class TradingLogger:
    """Centralized logging for the trading system."""
    
    def __init__(self, name: str = "trading_system", log_level: str = "INFO"):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        self.formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler
        self._add_console_handler()
        
        # Add file handler
        self._add_file_handler()
    
    def _add_console_handler(self):
        """Add console handler to logger."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self.logger.addHandler(console_handler)
    
    def _add_file_handler(self):
        """Add file handler to logger."""
        # Create logs directory if it doesn't exist
        log_dir = Path(__file__).parent.parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = log_dir / f"trading_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Get the configured logger instance."""
        return self.logger

# Create default logger instance
default_logger = TradingLogger().get_logger()

def get_logger(name: str = None):
    """Get a logger instance."""
    if name:
        return TradingLogger(name).get_logger()
    return default_logger
