import json
import os
import sys
import platform
import getpass
import traceback
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

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
        historical_file = os.path.join(PATHS['SRC_DIR'], 'historical_draws.csv')
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
        csv_file = os.path.join(PATHS['SRC_DIR'], 'historical_draws.csv')
        
        if not os.path.exists(csv_file):
            debug_print(f"Historical data file not found: {csv_file}", "ERROR")
            return False
            
        # Load and format data using DrawHandler
        handler = DrawHandler()
        historical_data = handler._load_historical_data()
        if historical_data is None:
            debug_print("Failed to load historical data", "ERROR")
            return False
            
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
            # First save the timestamped version
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            timestamped_file = os.path.join(PATHS['PROCESSED_DIR'], f'analysis_results_{timestamp}.xlsx')
            
            # Save the analysis results
            if analysis.save_to_excel(timestamped_file):
                debug_print(f"Analysis saved to: {timestamped_file}")
                
                # Also update the main analysis file
                if analysis.save_to_excel(PATHS['ANALYSIS']):
                    debug_print("Main analysis file updated")
                    debug_print("Analysis complete!")
                    return True
                else:
                    debug_print("Failed to update main analysis file", "WARNING")
                    return True  # Still return True as we saved the timestamped version
            else:
                debug_print("Failed to save analysis results", "ERROR")
                return False
                
        except Exception as save_error:
            debug_print(f"Error saving analysis: {str(save_error)}", "ERROR")
            debug_print(f"Attempted to save to: {timestamped_file}", "DEBUG")
            traceback.print_exc()
            return False
            
    except Exception as e:
        debug_print(f"Error in data analysis: {str(e)}", "ERROR")
        traceback.print_exc()
        return False
def generate_prediction():
    """Execute prediction generation with proper error handling"""
    try:
        debug_print("\n=== Starting Prediction Generation ===")
        debug_print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Initialize DrawHandler
        handler = DrawHandler()
        ensure_directories()

        # Load historical data
        debug_print("\nLoading historical data...")
        historical_data = handler._load_historical_data()
        if historical_data is None:
            debug_print("Failed to load historical data", "ERROR")
            return False

        # Format data using the formatter
        formatted_draws = DrawDataFormatter.format_historical_data(historical_data)
        if not formatted_draws:
            debug_print("No valid draws after formatting", "ERROR")
            return False

        # Validate first draw format
        is_valid, message = DrawDataFormatter.validate_draw_format(formatted_draws[0])
        if not is_valid:
            debug_print(f"Invalid draw format: {message}", "ERROR")
            return False

        debug_print(f"Successfully formatted {len(formatted_draws)} draws")

        # Train models
        debug_print("\nChecking/Training models...")
        if handler.train_ml_models():
            debug_print("Models ready")

            try:
                # Initialize DataAnalysis with validated data
                analyzer = DataAnalysis(formatted_draws)
                
                # Get analysis results
                debug_print("\nGetting analysis results...")
                analysis_results = analyzer.get_analysis_results()

                # Execute prediction pipeline
                debug_print("\nExecuting prediction pipeline...")
                pipeline_result = handler.handle_prediction_pipeline()

                if pipeline_result is not None and isinstance(pipeline_result, tuple):
                    predictions, probabilities, analysis = pipeline_result
                    
                    if predictions is not None:
                        # Convert numpy values if needed
                        if hasattr(predictions[0], 'item'):
                            predictions = [p.item() for p in predictions]
                        
                        # Save results
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        next_draw = get_next_draw_time(datetime.now())
                        
                        # Save prediction CSV
                        csv_path = os.path.join(PATHS['PREDICTIONS_DIR'], f'prediction_{timestamp}.csv')
                        pd.DataFrame({
                            'number': sorted(predictions),
                            'probability': [0.0500] * len(predictions)
                        }).to_csv(csv_path, index=False)
                        
                        # Save analysis Excel
                        analysis_path = os.path.join(PATHS['ANALYSIS'], f'analysis_{timestamp}.xlsx')
                        analyzer.save_to_excel(analysis_path)
                        
                        # Save metadata JSON
                        metadata_path = os.path.join(PATHS['PREDICTIONS_METADATA_DIR'], f'prediction_{timestamp}_metadata.json')
                        metadata = {
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'next_draw_time': next_draw.strftime('%H:%M  %d-%m-%Y'),
                            'prediction_info': {
                                'numbers': sorted(predictions),
                                'confidence': 0.0500
                            },
                            'analysis_context': analysis_results
                        }
                        
                        with open(metadata_path, 'w') as f:
                            json.dump(metadata, f, indent=4)
                        
                        debug_print("\nResults saved successfully")
                        return True
                    else:
                        debug_print("No valid predictions generated", "ERROR")
                        return False
                else:
                    debug_print(f"Invalid pipeline result: {pipeline_result}", "ERROR")
                    return False
                    
            except Exception as e:
                debug_print(f"Error in prediction generation: {str(e)}", "ERROR")
                debug_print(f"Full error context: {traceback.format_exc()}", "DEBUG")
                return False
        else:
            debug_print("Model training failed", "ERROR")
            return False
            
    except Exception as e:
        debug_print(f"Critical error in prediction generation: {str(e)}", "ERROR")
        debug_print(f"Critical error details: {traceback.format_exc()}", "DEBUG")
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