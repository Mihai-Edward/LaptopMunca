# File: src/__init__.py

"""Source package for lottery prediction system."""

import sys
import os

# Add project root to path FIRST
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
# Debug: Print sys.path to verify
print("Python path:", sys.path)

# Re-export all major components for convenience
try:
    from .data_analysis import DataAnalysis
    from .lottery_predictor import LotteryPredictor
    from .draw_handler import DrawHandler
    from .data_collector_selenium import KinoDataCollector  
    from .prediction_evaluator import PredictionEvaluator
except ImportError as e:
    print(f"Error in src/__init__.py imports: {e}")