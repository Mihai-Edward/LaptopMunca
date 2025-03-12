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

from lottery_predictor import LotteryPredictor

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
            
        # CHANGE HERE: Only use the last 24 draws for analysis
        historical_data = historical_data.tail(24)
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
            # CHANGE HERE: Define the fixed analysis file path - no timestamps
            # Make sure this is always the specific file we want to update
            analysis_file = os.path.join(PATHS['PROCESSED_DIR'], "analysis_results.xlsx")
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

def generate_prediction():
    try:
        debug_print("\n=== Starting Prediction Generation ===")
        debug_print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Initialize DrawHandler and ensure directories
        debug_print("\nInitializing prediction pipeline...")
        handler = DrawHandler()
        ensure_directories()

        # Load all historical data for training
        debug_print("\nLoading historical data...")
        historical_data = handler._load_historical_data()
        if historical_data is None:
            debug_print("Failed to load historical data", "ERROR")
            return False

        # STEP 1: Save the original methods for both train_and_predict and handle_prediction_pipeline
        debug_print("\nPreparing data and model integration...")
        original_train_predict = handler.predictor.train_and_predict
        original_handle_pipeline = handler.handle_prediction_pipeline
        saved_analysis = {}
        
        # STEP 2: Create wrapper for train_and_predict that saves analysis data
        def wrapper_train_predict(*args, **kwargs):
            nonlocal saved_analysis
            try:
                debug_print("Wrapper train_and_predict called")
                result = original_train_predict(*args, **kwargs)
                debug_print(f"Original train_and_predict returned: {type(result)}")
                
                if isinstance(result, tuple):
                    if len(result) == 3:
                        predictions, probabilities, analysis = result
                        saved_analysis = analysis  # Save the analysis for later
                        debug_print(f"Extracted analysis from 3-tuple result with {len(saved_analysis) if isinstance(saved_analysis, dict) else 'non-dict'} items")
                        return predictions, probabilities  # Return only what draw_handler expects
                    elif len(result) == 2:
                        predictions, probabilities = result
                        debug_print("Original returned 2-tuple, passing through")
                        return predictions, probabilities
                
                # If it's not a tuple with the expected format, return a safe default
                debug_print(f"Unexpected result type: {type(result)}, returning safe default")
                return result if result is not None else ([], [])
            
            except Exception as e:
                debug_print(f"Error in train_and_predict wrapper: {e}", "ERROR")
                traceback.print_exc()
                return [], []  # Return empty lists as safe defaults

        # STEP 3: Also create a wrapper for handle_prediction_pipeline to capture the results
        def wrapper_handle_pipeline(*args, **kwargs):
            try:
                # Add debug before calling original pipeline
                debug_print("Calling original handle_prediction_pipeline...")
                
                # Use the original with our modified train_and_predict in place
                result = original_handle_pipeline(*args, **kwargs)
                
                # Add debug after calling pipeline
                debug_print(f"Original handle_prediction_pipeline returned: {result}")
                
                if result is not None:
                    # Handle various result formats gracefully
                    if isinstance(result, tuple):
                        # If result is already a tuple, extract the first two elements
                        # and add our saved analysis as the third
                        first_element = result[0] if len(result) > 0 else None
                        second_element = result[1] if len(result) > 1 else None
                        
                        debug_print(f"Extracted from original pipeline - predictions: {type(first_element)}, probabilities: {type(second_element)}")
                        debug_print(f"Saved analysis has {len(saved_analysis)} items")
                        
                        # Check if we have actual prediction data
                        if first_element is None or (isinstance(first_element, list) and len(first_element) == 0):
                            debug_print("Warning: No prediction data was generated!")
                            # Try to generate some predictions directly using the model
                            try:
                                debug_print("Attempting direct model prediction...")
                                # Try using the most recent draw as input
                                direct_preds, direct_probs = handler.predictor.generate_predictions(1)
                                if direct_preds is not None and len(direct_preds) > 0:
                                    debug_print(f"Direct prediction successful, got {len(direct_preds)} predictions")
                                    first_element = direct_preds
                                    second_element = direct_probs
                            except Exception as e:
                                debug_print(f"Direct prediction attempt failed: {e}", "ERROR")
                        
                        return (first_element, second_element, saved_analysis)
                    else:
                        # If result is a single value, use it as predictions
                        debug_print(f"Result is not a tuple, but a {type(result)}")
                        return (result, None, saved_analysis)
                return (None, None, saved_analysis)  # Return empty result with analysis
            except Exception as e:
                debug_print(f"Error in handle_pipeline wrapper: {e}", "ERROR")
                traceback.print_exc()  # Add this to see the full stack trace
                return (None, None, saved_analysis)  # Return empty with analysis
        
        # STEP 4: Apply both wrappers
        handler.predictor.train_and_predict = wrapper_train_predict
        handler.handle_prediction_pipeline = wrapper_handle_pipeline

        debug_print(f"Original train_predict: {id(original_train_predict)}")
        debug_print(f"Wrapper train_predict: {id(wrapper_train_predict)}")
        debug_print(f"Handler predictor train_predict: {id(handler.predictor.train_and_predict)}")

        # Train models (with wrapper in place)
        debug_print("\nChecking/Training models...")
        if not handler.train_ml_models():
            debug_print("Model training failed", "ERROR")
            # Restore original methods before returning
            handler.predictor.train_and_predict = original_train_predict
            handler.handle_prediction_pipeline = original_handle_pipeline
            return False
        debug_print("Models ready")

        # Get analysis results and save them
        debug_print("\nGetting analysis results...")
        if not run_analysis():
            debug_print("Failed to run analysis", "ERROR")
            # Restore original methods before returning
            handler.predictor.train_and_predict = original_train_predict
            handler.handle_prediction_pipeline = original_handle_pipeline
            return False

        # Now call the pipeline with our wrappers in place
        debug_print("\nExecuting prediction pipeline...")
        try:
            result = handler.handle_prediction_pipeline()
            
            # Restore original methods
            handler.predictor.train_and_predict = original_train_predict
            handler.handle_prediction_pipeline = original_handle_pipeline
            
            # Process result (this should now have 3 elements)
            if result is not None:
                if isinstance(result, tuple) and len(result) >= 3:
                    predictions = result[0]
                    probabilities = result[1]
                    analysis = result[2]
                    
                    # Actually save the predictions to a file
                    if predictions is not None:
                        # Calculate next draw time for metadata
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
                            'number': predictions,
                            'probability': probabilities
                        }
                        pred_df = pd.DataFrame(pred_data)  # <-- Add this line
                        pred_df.to_csv(pred_file, index=False)
                        
                        # Save metadata
                        metadata = {
                            'timestamp': timestamp,
                            'next_draw_time': next_draw_time_formatted,
                            'model_info': handler.predictor.get_model_info() if hasattr(handler.predictor, 'get_model_info') else {},
                            'analysis_summary': analysis
                        }
                        
                        metadata_file = os.path.join(PATHS['PREDICTIONS_METADATA_DIR'], f"prediction_{timestamp}_metadata.json")
                        debug_print(f"Saving metadata to: {metadata_file}")
                        
                        with open(metadata_file, 'w') as f:
                            json.dump(metadata, f, indent=4)
                        
                        debug_print("Successfully generated predictions")
                        return True
                    else:
                        # NEW IMPROVED: Use backup prediction method when original returns None
                        debug_print("Prediction data is None, using top numbers from analysis")
                        try:
                            # Rather than relying on saved_analysis, let's directly read the analysis file
                            # that we just created in run_analysis()
                            analysis_file = os.path.join(PATHS['PROCESSED_DIR'], "analysis_results.xlsx")
                            debug_print(f"Loading analysis from: {analysis_file}")
                            
                            if os.path.exists(analysis_file):
                                # Read the first sheet (frequency data)
                                analysis_df = pd.read_excel(analysis_file, sheet_name=0)
                                
                                # Debug print column names to see what's available
                                debug_print(f"Available columns in analysis file: {list(analysis_df.columns)}")
                                
                                # Check which column to sort by - try different possible names
                                sort_column = None
                                for possible_name in ['frequency', 'count', 'freq', 'occurrence', 'occurrences', 'Frequency']:
                                    if possible_name in analysis_df.columns:
                                        sort_column = possible_name
                                        break
                                
                                # If we didn't find a match, use the second column (typical format is number, frequency)
                                if not sort_column and len(analysis_df.columns) > 1:
                                    sort_column = analysis_df.columns[1]
                                    debug_print(f"No explicit frequency column found, using column: {sort_column}")
                                
                                # Make sure we have a number column too
                                number_column = 'number'
                                if 'number' not in analysis_df.columns and len(analysis_df.columns) > 0:
                                    number_column = analysis_df.columns[0]
                                    debug_print(f"Using '{number_column}' as the number column")
                                
                                debug_print(f"Using columns - number: {number_column}, frequency: {sort_column}")
                                
                                # Sort by the identified frequency column
                                sorted_df = analysis_df.sort_values(by=sort_column, ascending=False)
                                top_numbers = sorted_df[number_column].head(20).tolist()
                                
                                # For probabilities, we'll normalize the frequency to get probabilities
                                total = sorted_df[sort_column].sum()
                                if total > 0:
                                    probabilities = (sorted_df[sort_column].head(20) / total).tolist()
                                else:
                                    probabilities = [1.0/20] * 20  # Equal probabilities if we can't calculate
                                
                                debug_print(f"Found {len(top_numbers)} top numbers from analysis")
                                
                                # Calculate next draw time for metadata
                                now = datetime.now()
                                next_draw_time = get_next_draw_time(now)
                                next_draw_time_formatted = next_draw_time.strftime('%H:%M  %d-%m-%Y')
                                
                                # Create timestamp for filenames
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                
                                # Save predictions to CSV
                                pred_file = os.path.join(PATHS['PREDICTIONS_DIR'], f"prediction_{timestamp}.csv")
                                debug_print(f"Saving backup predictions to: {pred_file}")
                                
                                # Create DataFrame and save to CSV
                                pred_data = {
                                    'number': top_numbers,
                                    'probability': probabilities
                                }
                                pred_df = pd.DataFrame(pred_data)  # <-- Add this line
                                pred_df.to_csv(pred_file, index=False)
                                
                                # Save metadata
                                metadata = {
                                    'timestamp': timestamp,
                                    'next_draw_time': next_draw_time_formatted,
                                    'model_info': {"source": "analysis_backup", "method": "frequency_analysis"},
                                    'analysis_summary': {"top_numbers": top_numbers}
                                }
                                
                                metadata_file = os.path.join(PATHS['PREDICTIONS_METADATA_DIR'], f"prediction_{timestamp}_metadata.json")
                                debug_print(f"Saving metadata to: {metadata_file}")
                                
                                with open(metadata_file, 'w') as f:
                                    json.dump(metadata, f, indent=4)
                                
                                debug_print("Successfully generated backup predictions from analysis data")
                                return True
                            else:
                                debug_print(f"Analysis file not found: {analysis_file}", "ERROR")
                                return False
                        except Exception as e:
                            debug_print(f"Error creating backup prediction: {e}", "ERROR")
                            traceback.print_exc()  # This will help debug any other issues
                            return False
                else:
                    debug_print(f"Invalid result format from pipeline: {result}", "ERROR")
                    return False
            else:
                debug_print("No result returned from prediction pipeline", "ERROR")
                return False

        except Exception as e:
            debug_print(f"Error in prediction pipeline: {str(e)}", "ERROR")
            traceback.print_exc()
            # Restore original methods before returning from error
            handler.predictor.train_and_predict = original_train_predict
            handler.handle_prediction_pipeline = original_handle_pipeline
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