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

def save_prediction(predictions, probabilities, is_backup=False):
    """Save prediction using consolidated format"""
    try:
        # Create a new DrawHandler instance
        draw_handler = DrawHandler()
        success = draw_handler.predictor.save_prediction(
            prediction=predictions,
            probabilities=probabilities
        )
        if success:
            print("Prediction saved successfully")
            return True
        print("Failed to save prediction")
        return False
    except Exception as e:
        print(f"Error saving prediction: {e}")
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
def save_prediction_to_excel(predictions, probabilities, next_draw_time=None):
    """
    Save prediction to consolidated Excel file for easier evaluation.
    
    Args:
        predictions: List of predicted numbers
        probabilities: List of corresponding probabilities
        next_draw_time: The formatted time of the next draw (optional)
    
    Returns:
        bool: Success status
    """
    try:
        debug_print("\nSaving prediction to consolidated Excel file...")
        import pandas as pd
        from datetime import datetime
        
        # Set up paths and ensure directories exist
        excel_file = os.path.join(PATHS['PROCESSED_DIR'], 'all_predictions.xlsx')
        
        # Create timestamps
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if next_draw_time is None:
            next_draw_time = get_next_draw_time(datetime.now()).strftime('%H:%M  %d-%m-%Y')
        
        # Format data for new record
        new_data = {
            'timestamp': timestamp,
            'next_draw_time': next_draw_time,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'prediction_time': datetime.now().strftime('%H:%M:%S')
        }
        
        # Add predicted numbers to record
        for i, num in enumerate(predictions, 1):
            new_data[f'number{i}'] = int(num)
            
        # Add probabilities to record (if available)
        if probabilities and len(probabilities) == len(predictions):
            for i, prob in enumerate(probabilities, 1):
                new_data[f'probability{i}'] = float(prob)

        # Create a DataFrame for the new record
        new_row = pd.DataFrame([new_data])
        
        # Check if file exists
        if os.path.exists(excel_file):
            try:
                # Load existing data
                existing_df = pd.read_excel(excel_file)
                
                # Check if we already have this prediction
                if 'next_draw_time' in existing_df.columns:
                    date_matches = existing_df['next_draw_time'] == next_draw_time
                    if any(date_matches):
                        # Update existing prediction
                        for col, value in new_data.items():
                            if col in existing_df.columns:
                                existing_df.loc[date_matches, col] = value
                        result_df = existing_df
                    else:
                        # Append new prediction
                        result_df = pd.concat([existing_df, new_row], ignore_index=True)
                else:
                    # No matching column, just append
                    result_df = pd.concat([existing_df, new_row], ignore_index=True)
            except Exception as e:
                debug_print(f"Error reading existing Excel file: {e}", "WARNING")
                debug_print("Creating new Excel file", "INFO")
                result_df = new_row
        else:
            # Create new file
            result_df = new_row
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(excel_file), exist_ok=True)
        
        # Save to Excel
        result_df.to_excel(excel_file, index=False)
        debug_print(f"Prediction saved to consolidated Excel file: {excel_file}")
        return True
        
    except Exception as e:
        debug_print(f"Error saving prediction to Excel: {e}", "ERROR")
        traceback.print_exc()
        return False
def evaluate_predictions():
    """Execute prediction evaluation"""
    try:
        debug_print("Starting prediction evaluation...")
        evaluator = PredictionEvaluator()
        
        # First check if consolidated Excel file exists
        excel_file = os.path.join(PATHS['PROCESSED_DIR'], 'all_predictions.xlsx')
        if os.path.exists(excel_file):
            debug_print(f"Found consolidated predictions Excel file: {excel_file}")
            debug_print("Using consolidated file for evaluation...")
            
            try:
                # Load historical data
                historical_df = pd.read_csv(PATHS['HISTORICAL_DATA'])
                if historical_df is None or len(historical_df) == 0:
                    debug_print("No historical data found, falling back to standard evaluation", "WARNING")
                else:
                    # Load the Excel file with predictions
                    try:
                        predictions_df = pd.read_excel(excel_file)
                        debug_print(f"Loaded {len(predictions_df)} predictions from Excel file")
                        
                        # Track evaluation results
                        total_evaluated = 0
                        total_correct = 0
                        best_prediction = 0
                        
                        # Process each prediction
                        for idx, row in predictions_df.iterrows():
                            try:
                                if 'next_draw_time' not in row:
                                    debug_print(f"Missing next_draw_time in row {idx}, skipping", "WARNING")
                                    continue
                                    
                                next_draw_time = row['next_draw_time']
                                
                                # Find corresponding draw in historical data
                                matching_draws = historical_df[historical_df['date'] == next_draw_time]
                                if len(matching_draws) == 0:
                                    debug_print(f"No matching draw found for {next_draw_time}, skipping", "INFO")
                                    continue
                                    
                                # Extract predicted numbers
                                predicted_numbers = []
                                for i in range(1, 21):
                                    column_name = f'number{i}'
                                    if column_name in row and not pd.isna(row[column_name]):
                                        predicted_numbers.append(int(row[column_name]))
                                
                                if len(predicted_numbers) != 20:
                                    debug_print(f"Invalid prediction count: {len(predicted_numbers)}, skipping", "WARNING")
                                    continue
                                    
                                # Extract actual drawn numbers
                                actual_numbers = []
                                for i in range(1, 21):
                                    column_name = f'number{i}'
                                    if column_name in matching_draws.columns:
                                        val = matching_draws.iloc[0][column_name]
                                        if not pd.isna(val):
                                            actual_numbers.append(int(val))
                                
                                if len(actual_numbers) != 20:
                                    debug_print(f"Invalid actual draw count: {len(actual_numbers)}, skipping", "WARNING")
                                    continue
                                    
                                # Compare and count matches
                                correct_count = len(set(predicted_numbers).intersection(set(actual_numbers)))
                                
                                # Update statistics
                                total_evaluated += 1
                                total_correct += correct_count
                                best_prediction = max(best_prediction, correct_count)
                                
                                debug_print(f"Prediction for {next_draw_time}: {correct_count} correct", "INFO")
                                
                            except Exception as row_error:
                                debug_print(f"Error processing prediction row {idx}: {row_error}", "ERROR")
                                continue
                                
                        # Calculate and display stats
                        if total_evaluated > 0:
                            avg_accuracy = (total_correct / (total_evaluated * 20)) * 100
                            debug_print("\n=== Evaluation Results (Excel) ===")
                            debug_print(f"Total evaluated: {total_evaluated}")
                            debug_print(f"Average accuracy: {avg_accuracy:.2f}%")
                            debug_print(f"Best prediction: {best_prediction} correct")
                            
                            # Store results for later use
                            evaluator.set_evaluation_stats({
                                'total_predictions': total_evaluated,
                                'total_correct': total_correct,
                                'best_prediction': best_prediction,
                                'avg_accuracy': avg_accuracy
                            })
                            
                            return True
                    except Exception as excel_error:
                        debug_print(f"Error processing Excel file: {excel_error}", "ERROR")
                        debug_print("Falling back to standard evaluation", "WARNING")
            except Exception as e:
                debug_print(f"Error in Excel evaluation: {e}", "ERROR")
                debug_print("Falling back to standard evaluation", "WARNING")
        
        # Fall back to the original evaluation method
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