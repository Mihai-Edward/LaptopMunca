import json
import os
import sys
import platform
import getpass
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets src directory
project_root = os.path.dirname(current_dir)               # Gets project root
sys.path.append(project_root)

# Standard library imports
from config.paths import PATHS, ensure_directories, validate_paths, print_system_info
from data_collector_selenium import KinoDataCollector
from data_analysis import DataAnalysis
from draw_handler import DrawHandler
from prediction_evaluator import PredictionEvaluator
from lottery_predictor import LotteryPredictor

# Configuration parameters
CONFIG = {
    'analysis_draws': 24,            # Number of draws to use for analysis
    'prediction_timeout': 60,        # Timeout for prediction operations in seconds
    'backup_prediction_enabled': True,  # Whether to use backup prediction when primary fails
    'debug_level': "INFO"            # Default debug level
}

class DrawDataFormatter:
    """Helper class to ensure consistent draw data formatting"""
    
    @staticmethod
    def format_historical_data(historical_data):
        """Format historical data into consistent (date, numbers) tuples"""
        formatted_draws = []
        try:
            for idx, row in historical_data.iterrows():
                try:
                    draw_date = row['date']
                    numbers = []
                    for i in range(1, 21):
                        num = row[f'number{i}']
                        if isinstance(num, (int, float)) and 1 <= num <= 80:
                            numbers.append(int(num))
                    if len(numbers) == 20:
                        formatted_draws.append((draw_date, sorted(numbers)))
                except Exception as e:
                    debug_print(f"Error formatting row {idx}: {e}", "ERROR")
            return formatted_draws
        except Exception as e:
            debug_print(f"Error in data formatting: {e}", "ERROR")
            return []

    @staticmethod
    def validate_draw_format(draw):
        """Validate the format of a single draw tuple"""
        try:
            if not isinstance(draw, tuple) or len(draw) != 2:
                return False, "Invalid draw format"
            
            date, numbers = draw
            if not isinstance(numbers, list) or len(numbers) != 20:
                return False, "Invalid numbers format"
                
            if not all(isinstance(n, int) and 1 <= n <= 80 for n in numbers):
                return False, "Invalid number values"
                
            return True, "Valid format"
        except Exception as e:
            return False, f"Validation error: {e}"

def debug_print(message, level="INFO"):
    """Enhanced debug printing with timestamp and level"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}")

def validate_environment():
    """Validate the execution environment"""
    try:
        debug_print("Starting environment validation...")
        
        # Print system information
        print_system_info()
        
        # Ensure all directories exist
        debug_print("Ensuring required directories exist...")
        if not ensure_directories():
            raise Exception("Failed to create required directories")
            
        # Validate all paths
        debug_print("Validating paths...")
        if not validate_paths():
            raise Exception("Path validation failed")
            
        # Check historical data file - FIXED PATH
        historical_file = PATHS['HISTORICAL_DATA']
        if not os.path.exists(historical_file):
            debug_print(f"Historical data file not found: {historical_file}", "WARNING")
        else:
            debug_print(f"Historical data file found: {historical_file}")
            
        debug_print("Environment validation completed successfully")
        return True
        
    except Exception as e:
        debug_print(f"Environment validation failed: {str(e)}", "ERROR")
        traceback.print_exc()
        return False

def collect_data():
    """Execute data collection with sorting"""
    try:
        debug_print("Starting data collection...")
        collector = KinoDataCollector()
        
        # First sort existing data
        debug_print("Sorting historical draws...")
        collector.sort_historical_draws()
        
        # Fetch new draws
        draws = collector.fetch_latest_draws()
        if draws:
            debug_print(f"Successfully collected {len(draws)} draws")
            
            # Print collected draws
            debug_print("\nCollected draws:")
            for draw_date, numbers in draws:
                debug_print(f"Date: {draw_date}, Numbers: {', '.join(map(str, numbers))}")
            
            # Sort again after collecting new draws
            debug_print("\nSorting updated historical draws...")
            if collector.sort_historical_draws():
                debug_print("Historical draws successfully sorted from newest to oldest")
            
            return True
        return False
        
    except Exception as e:
        debug_print(f"Data collection failed: {str(e)}", "ERROR")
        traceback.print_exc()
        return False

def run_analysis():
    """Execute data analysis with full initialization and debug output"""
    try:
        debug_print("Starting data analysis...")
        
        # Load real data from the CSV file
        csv_file = PATHS['HISTORICAL_DATA']
        
        if not os.path.exists(csv_file):
            debug_print(f"Historical data file not found: {csv_file}", "ERROR")
            return False
            
        # Load and format data using DrawHandler
        handler = DrawHandler()
        historical_data = handler._load_historical_data()
        if historical_data is None:
            debug_print("Failed to load historical data", "ERROR")
            return False
            
        # Use the configured number of draws for analysis
        historical_data = historical_data.tail(CONFIG['analysis_draws'])
        debug_print(f"Using last {len(historical_data)} draws for analysis")
            
        formatted_draws = DrawDataFormatter.format_historical_data(historical_data)
        if not formatted_draws:
            debug_print("No valid draws to analyze", "ERROR")
            return False
            
        # Initialize analysis with loaded data
        analysis = DataAnalysis(formatted_draws)
        
        # Run analyses
        frequency = analysis.count_frequency()
        debug_print(f"Number of unique numbers: {len(frequency)}")
        
        top_numbers = analysis.get_top_numbers(20)
        debug_print(f"Top 20 numbers: {', '.join(map(str, top_numbers))}")
        
        # Save results
        try:
            # Define the fixed analysis file path - no timestamps
            analysis_file = PATHS['ANALYSIS']
            debug_print(f"Saving analysis to file: {analysis_file}")
            
            # Save directly to the main analysis file
            if analysis.save_to_excel(analysis_file):
                debug_print("Analysis file updated successfully")
                debug_print("Analysis complete!")
                return True
            else:
                debug_print("Failed to save analysis results", "ERROR")
                return False
                
        except Exception as save_error:
            debug_print(f"Error saving analysis: {str(save_error)}", "ERROR")
            debug_print(f"Attempted to save to: {analysis_file}", "DEBUG")
            traceback.print_exc()
            return False
            
    except Exception as e:
        debug_print(f"Error in data analysis: {str(e)}", "ERROR")
        traceback.print_exc()
        return False

def get_analysis_data():
    """Get analysis data from the Excel file"""
    try:
        analysis_file = PATHS['ANALYSIS']
        if not os.path.exists(analysis_file):
            debug_print(f"Analysis file not found: {analysis_file}", "ERROR")
            return None
            
        # Read analysis file
        analysis_df = pd.read_excel(analysis_file, sheet_name='Frequency')
        
        # Make sure we have a number column and frequency column
        if 'Number' in analysis_df.columns and 'Frequency' in analysis_df.columns:
            return analysis_df
        # Try different column name combinations
        elif len(analysis_df.columns) >= 2:
            # Rename columns to ensure consistency
            analysis_df.columns = ['Number', 'Frequency'] + list(analysis_df.columns[2:])
            return analysis_df
        else:
            debug_print("Invalid analysis file format", "ERROR")
            return None
    except Exception as e:
        debug_print(f"Error getting analysis data: {e}", "ERROR")
        return None

def create_backup_prediction():
    """Create a backup prediction based on analysis data"""
    try:
        debug_print("Creating backup prediction from analysis data...")
        
        # Get analysis data
        analysis_df = get_analysis_data()
        if analysis_df is None:
            return None, None
            
        # Find the number and frequency columns
        number_column = 'Number' if 'Number' in analysis_df.columns else analysis_df.columns[0]
        freq_column = 'Frequency' if 'Frequency' in analysis_df.columns else analysis_df.columns[1]
        
        # Sort by frequency
        sorted_df = analysis_df.sort_values(by=freq_column, ascending=False)
        top_numbers = sorted_df[number_column].head(20).tolist()
        
        # Calculate probabilities
        total = sorted_df[freq_column].sum()
        if total > 0:
            probabilities = (sorted_df[freq_column].head(20) / total).tolist()
        else:
            probabilities = [1.0/20] * 20
            
        debug_print(f"Created backup prediction with {len(top_numbers)} numbers")
        return top_numbers, probabilities
    except Exception as e:
        debug_print(f"Error creating backup prediction: {e}", "ERROR")
        return None, None

def save_prediction(numbers, probabilities, is_backup=False):
    """Save prediction to files"""
    try:
        if numbers is None or len(numbers) == 0:
            debug_print("No valid numbers to save", "ERROR")
            return False
            
        # Calculate next draw time
        now = datetime.now()
        next_draw_time = get_next_draw_time(now)
        next_draw_time_formatted = next_draw_time.strftime('%H:%M  %d-%m-%Y')
        
        # Create timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save predictions to CSV
        pred_file = os.path.join(PATHS['PREDICTIONS_DIR'], f"prediction_{timestamp}.csv")
        debug_print(f"Saving predictions to: {pred_file}")
        
        # Create DataFrame and save to CSV
        pred_data = {
            'number': numbers,
            'probability': probabilities if probabilities else [1.0/len(numbers)] * len(numbers)
        }
        pred_df = pd.DataFrame(pred_data)
        pred_df.to_csv(pred_file, index=False)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'next_draw_time': next_draw_time_formatted,
            'source': 'backup_prediction' if is_backup else 'ml_prediction',
            'prediction_count': len(numbers),
            'creation_time': now.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = os.path.join(PATHS['PREDICTIONS_METADATA_DIR'], f"prediction_{timestamp}_metadata.json")
        debug_print(f"Saving metadata to: {metadata_file}")
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        debug_print(f"Successfully saved {'backup' if is_backup else 'primary'} prediction")
        return True
    except Exception as e:
        debug_print(f"Error saving prediction: {e}", "ERROR")
        traceback.print_exc()
        return False

def generate_prediction():
    """Generate prediction using DrawHandler without modifying its methods"""
    try:
        debug_print("\n=== Starting Prediction Generation ===")
        debug_print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Initialize DrawHandler
        debug_print("\nInitializing prediction pipeline...")
        handler = DrawHandler()
        
        # Make sure we have the latest analysis
        debug_print("Ensuring analysis is up to date...")
        run_analysis()
        
        # Train models if needed
        debug_print("\nChecking/Training models...")
        if not handler.train_ml_models():
            debug_print("Model training failed - trying to continue with existing models", "WARNING")
        
        # Call the prediction pipeline directly (no modification)
        debug_print("\nExecuting prediction pipeline...")
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        # Check if we have valid predictions
        if predictions is not None and len(predictions) > 0:
            debug_print(f"Prediction successful: {len(predictions)} numbers generated")
            return save_prediction(predictions, probabilities)
        else:
            debug_print("Primary prediction failed, using backup", "WARNING")
            if CONFIG['backup_prediction_enabled']:
                backup_numbers, backup_probs = create_backup_prediction()
                if backup_numbers is not None:
                    return save_prediction(backup_numbers, backup_probs, is_backup=True)
            
            debug_print("Prediction generation failed", "ERROR")
            return False

    except Exception as e:
        debug_print(f"Error generating prediction: {str(e)}", "ERROR")
        traceback.print_exc()
        return False

def get_next_draw_time(current_time):
    """Calculate the next draw time (5 minute intervals)"""
    minute = (current_time.minute // 5 * 5 + 5) % 60
    hour = current_time.hour + (current_time.minute // 5 * 5 + 5) // 60
    
    next_time = current_time.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
    if hour >= 24:
        next_time += timedelta(days=1)
    
    return next_time

def evaluate_predictions():
    """Execute prediction evaluation"""
    try:
        debug_print("Starting prediction evaluation...")
        evaluator = PredictionEvaluator()
        evaluator.evaluate_past_predictions()
        
        # Get and display performance stats
        stats = evaluator.get_performance_stats()
        if stats:
            debug_print("\n=== Evaluation Results ===")
            debug_print(f"Total evaluated: {stats.get('total_predictions', 0)}")
            debug_print(f"Average accuracy: {stats.get('avg_accuracy', 0):.2f}%")
            debug_print(f"Best prediction: {stats.get('best_prediction', 0)} correct")
        
        return True
    except Exception as e:
        debug_print(f"Prediction evaluation failed: {str(e)}", "ERROR")
        traceback.print_exc()
        return False

def main():
    """Main program loop"""
    try:
        # Initial environment validation
        if not validate_environment():
            debug_print("Environment validation failed. Exiting...", "ERROR")
            return
            
        debug_print("=== Lottery Prediction System ===")
        debug_print("System initialized successfully")
        
        while True:
            print("\n=== Main Menu ===")
            print("1. Collect Data")
            print("2. Run Data Analysis")
            print("3. Generate Prediction")
            print("4. Evaluate Predictions")
            print("5. Exit")
            
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == "1":
                debug_print("\nExecuting Data Collection...")
                if collect_data():
                    debug_print("Data collection completed successfully")
                else:
                    debug_print("Data collection failed", "ERROR")
                    
            elif choice == "2":
                debug_print("\nExecuting Data Analysis...")
                if run_analysis():
                    debug_print("Data analysis completed successfully")
                else:
                    debug_print("Data analysis failed", "ERROR")
                    
            elif choice == "3":
                debug_print("\nExecuting Prediction Generation...")
                if generate_prediction():
                    debug_print("Prediction generation completed successfully")
                else:
                    debug_print("Prediction generation failed", "ERROR")
                    
            elif choice == "4":
                debug_print("\nExecuting Prediction Evaluation...")
                if evaluate_predictions():
                    debug_print("Prediction evaluation completed successfully")
                else:
                    debug_print("Prediction evaluation failed", "ERROR")
                    
            elif choice == "5":
                debug_print("\nExiting program...")
                break
                
            else:
                debug_print("Invalid choice. Please try again.", "WARNING")
                
    except Exception as e:
        debug_print(f"Critical error in main program: {str(e)}", "ERROR")
        traceback.print_exc()

if __name__ == "__main__":
    main()