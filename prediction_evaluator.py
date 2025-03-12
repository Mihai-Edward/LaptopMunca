import os
import sys
import json
import glob
import pandas as pd
import numpy as np
import traceback
from datetime import datetime
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from config.paths import PATHS
# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import configuration
from config.paths import PATHS, ensure_directories

class PredictionEvaluator:
    def __init__(self):
        """Initialize PredictionEvaluator with enhanced configuration"""
        # Use PATHS configuration
        ensure_directories()
        
        # Fix path assignments
        self.predictions_dir = PATHS['PREDICTIONS_DIR']
        self.metadata_dir = PATHS['PREDICTIONS_METADATA_DIR']
        self.historical_file = PATHS['HISTORICAL_DATA']
        self.results_dir = PATHS['PROCESSED_DIR']
        self.results_file = os.path.join(self.results_dir, 'evaluation_results.xlsx')
        
        # Create required directories
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Validate the historical data file
        self.historical_data_valid = self.validate_historical_data()
        
        # Initialize evaluation metrics tracking
        self.evaluation_metrics = {
            'accuracy_history': [],
            'precision_history': [],
            'recall_history': [],
            'f1_history': []
        }

    def compare_prediction_with_actual(self, predicted_numbers, actual_numbers):
        """Compare predicted numbers with actual draw numbers"""
        try:
            predicted_set = set(predicted_numbers)
            actual_set = set(actual_numbers)
            correct_numbers = predicted_set.intersection(actual_set)
            accuracy = len(correct_numbers) / 20  # There are 20 numbers drawn

            precision = len(correct_numbers) / len(predicted_set) if predicted_set else 0
            recall = len(correct_numbers) / len(actual_set) if actual_set else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            return {
                'correct_numbers': sorted(list(correct_numbers)),
                'num_correct': len(correct_numbers),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'predicted_numbers': sorted(list(predicted_numbers)),
                'actual_numbers': sorted(list(actual_numbers))
            }
        except Exception as e:
            print(f"Error in comparison: {e}")
            traceback.print_exc()
            return None

    def save_comparison(self, predicted_numbers, actual_numbers, draw_date=None, metadata=None):
        """Save comparison between predicted and actual numbers"""
        try:
            # Calculate matches
            matches = set(predicted_numbers).intersection(set(actual_numbers))
            num_correct = len(matches)
            accuracy = (num_correct / len(actual_numbers)) * 100 if actual_numbers else 0
            precision = (num_correct / len(predicted_numbers)) * 100 if predicted_numbers else 0
            
            # Create result dictionary
            result = {
                'date': draw_date if draw_date else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'prediction_file': metadata.get('prediction_file', 'Unknown'),
                'num_correct': num_correct,
                'accuracy': accuracy,
                'precision': precision,
                'matches': sorted(list(matches)),
                'predicted': sorted(predicted_numbers),
                'actual': sorted(actual_numbers),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save to Excel file
            results_file = os.path.join(PATHS.get('PROCESSED_DIR', ''), 'evaluation_results.xlsx')
            
            # Initialize evaluation metrics if needed
            if not hasattr(self, 'evaluation_metrics'):
                self.evaluation_metrics = {
                    'accuracy_history': [],
                    'precision_history': []
                }
                
            # Update evaluation metrics
            self.evaluation_metrics['accuracy_history'].append(result['accuracy'])
            self.evaluation_metrics['precision_history'].append(result['precision'])
            
            try:
                # Check if file exists
                if os.path.exists(results_file):
                    # Read existing results
                    existing_df = pd.read_excel(results_file)
                    
                    # Check if we already have an entry for this draw date
                    date_match = existing_df['date'] == result['date']
                    if any(date_match):
                        # Update the existing entry instead of adding a new one
                        print(f"Updating existing evaluation for draw date: {draw_date}")
                        for key, value in result.items():
                            existing_df.loc[date_match, key] = value
                        results_df = existing_df
                    else:
                        # This is a new draw date, append it
                        results_df = pd.concat([existing_df, pd.DataFrame([result])], ignore_index=True)
                else:
                    # No existing file, create new
                    results_df = pd.DataFrame([result])
                    
                # Save to Excel
                results_df.to_excel(results_file, index=False)
                print(f"Results saved to {results_file}")
                
                print(f"Evaluated prediction for {draw_date}: {num_correct} correct")
            except Exception as e:
                print(f"Error saving to Excel: {e}")
                # Try creating a new Excel file with clean data
                try:
                    results_df = pd.DataFrame([result])
                    results_df.to_excel(results_file, index=False)
                    print(f"Created new results file: {results_file}")
                except Exception as new_e:
                    print(f"Error creating Excel file: {new_e}")
            
            return result
                
        except Exception as e:
            print(f"Error saving comparison: {e}")
            traceback.print_exc()
            return None

    def analyze_prediction_patterns(self, results_df):
        """Analyze prediction patterns with enhanced pattern detection"""
        patterns = {
            'frequent_correct': Counter(),
            'frequent_missed': Counter(),
            'time_based_accuracy': {},
            'streak_analysis': {'current_streak': 0, 'best_streak': 0},
            'pair_patterns': Counter()
        }

        try:
            for index, row in results_df.iterrows():
                try:
                    correct_nums = eval(str(row['Correct_Numbers']))
                    predicted = eval(str(row['Predicted_Numbers']))
                    actual = eval(str(row['Actual_Numbers']))
                    
                    # Update frequency counters
                    patterns['frequent_correct'].update(correct_nums)
                    missed = set(actual) - set(predicted)
                    patterns['frequent_missed'].update(missed)
                    
                    # Analyze pairs in correct predictions
                    for i in range(len(correct_nums)):
                        for j in range(i + 1, len(correct_nums)):
                            patterns['pair_patterns'].update([(correct_nums[i], correct_nums[j])])
                    
                    # Time-based analysis
                    try:
                        date = pd.to_datetime(row['Date'])
                        time_key = f"{date.hour:02d}:00"
                        if time_key not in patterns['time_based_accuracy']:
                            patterns['time_based_accuracy'][time_key] = []
                        patterns['time_based_accuracy'][time_key].append(row['Number_Correct'])
                    except Exception as e:
                        print(f"Error processing date for row {index}: {e}")
                    
                    # Update streak analysis
                    if row['Number_Correct'] >= 5:
                        patterns['streak_analysis']['current_streak'] += 1
                        patterns['streak_analysis']['best_streak'] = max(
                            patterns['streak_analysis']['best_streak'],
                            patterns['streak_analysis']['current_streak']
                        )
                    else:
                        patterns['streak_analysis']['current_streak'] = 0

                except Exception as e:
                    print(f"Error processing row {index}: {e}")
                    continue

            return patterns

        except Exception as e:
            print(f"Error in pattern analysis: {e}")
            traceback.print_exc()
            return None

    def get_performance_stats(self):
        """Get statistical performance metrics from evaluation results"""
        try:
            # Load results from Excel
            results_file = os.path.join(PATHS.get('PROCESSED_DIR', ''), 'evaluation_results.xlsx')
            if not os.path.exists(results_file):
                print("No evaluation results file found.")
                return None
                
            try:
                results_df = pd.read_excel(results_file)
            except Exception as xlsx_e:
                print(f"Error reading Excel file: {xlsx_e}")
                # Try fallback CSV
                fallback_file = os.path.join(PATHS.get('PROCESSED_DIR', ''), 'evaluation_results_fallback.csv')
                if os.path.exists(fallback_file):
                    results_df = pd.read_csv(fallback_file)
                else:
                    print("No evaluation results available.")
                    return None
                
            if len(results_df) < 1:
                print("No evaluation data found in results file.")
                return None
                
            # Calculate statistics
            avg_correct = results_df['num_correct'].mean()
            best_prediction = results_df['num_correct'].max()
            worst_prediction = results_df['num_correct'].min()
            avg_accuracy = results_df['accuracy'].mean()
            consistency = results_df['num_correct'].std() if len(results_df) > 1 else 0
            
            # Calculate streak of good predictions (above average)
            results_df['above_avg'] = results_df['num_correct'] > avg_correct
            streaks = []
            current_streak = 0
            
            for above_avg in results_df['above_avg']:
                if above_avg:
                    current_streak += 1
                else:
                    streaks.append(current_streak)
                    current_streak = 0
            
            streaks.append(current_streak)  # Add the last streak
            best_streak = max(streaks) if streaks else 0
            
            # Calculate trend
            if len(results_df) > 2:
                recent_results = results_df.tail(min(10, len(results_df)))
                trend_slope = np.polyfit(range(len(recent_results)), recent_results['num_correct'], 1)[0]
                trend = "Improving" if trend_slope > 0.1 else "Declining" if trend_slope < -0.1 else "Stable"
                improvement_rate = trend_slope * 100 / avg_correct if avg_correct else 0
            else:
                trend = "Insufficient data"
                improvement_rate = 0
                
            # Process matches column
            all_matches = []
            for matches in results_df['matches']:
                if isinstance(matches, list):
                    all_matches.extend(matches)
            
            # Handle missing or empty data
            if not all_matches and 'matches' in results_df.columns:
                # The matches might be stored as strings
                for match_str in results_df['matches'].astype(str):
                    try:
                        if match_str and match_str != 'nan':
                            # Try to evaluate the string representation of a list
                            matches = eval(match_str)
                            if isinstance(matches, list):
                                all_matches.extend(matches)
                    except:
                        continue
                        
            # Count frequencies
            correct_counter = Counter(all_matches)
            
            # Find missed numbers
            missed_numbers = []
            for i, row in results_df.iterrows():
                predicted = row['predicted'] if isinstance(row['predicted'], list) else eval(str(row['predicted'])) if str(row['predicted']) != 'nan' else []
                matches = row['matches'] if isinstance(row['matches'], list) else eval(str(row['matches'])) if str(row['matches']) != 'nan' else []
                missed = [p for p in predicted if p not in matches]
                missed_numbers.extend(missed)
            
            missed_counter = Counter(missed_numbers)
            
            # Find successful pairs
            pairs = []
            for i, row in results_df.iterrows():
                matches = row['matches'] if isinstance(row['matches'], list) else eval(str(row['matches'])) if str(row['matches']) != 'nan' else []
                if len(matches) >= 2:
                    for i in range(len(matches)):
                        for j in range(i+1, len(matches)):
                            pairs.append((min(matches[i], matches[j]), max(matches[i], matches[j])))
            
            pair_counter = Counter(pairs)
            
            # Time analysis
            if 'date' in results_df.columns:
                try:
                    if isinstance(results_df['date'].iloc[0], str) and ':' in results_df['date'].iloc[0]:
                        # Format like "08:20  12-03-2025"
                        results_df['hour'] = results_df['date'].str.split(':').str[0].astype(int) 
                    else:
                        # Standard datetime
                        results_df['hour'] = pd.to_datetime(results_df['date']).dt.hour
                        
                    hour_performance = results_df.groupby('hour')['num_correct'].mean().to_dict()
                    best_hour = max(hour_performance.items(), key=lambda x: x[1]) if hour_performance else (0, 0)
                except Exception as e:
                    print(f"Error in time analysis: {e}")
                    best_hour = (0, 0)
            else:
                best_hour = (0, 0)
            
            # Compile stats
            stats = {
                'total_predictions': len(results_df),
                'avg_correct': avg_correct,
                'best_prediction': best_prediction,
                'worst_prediction': worst_prediction,
                'avg_accuracy': avg_accuracy,
                'consistency_score': consistency,
                'best_streak': best_streak,
                'trend': trend,
                'improvement_rate': improvement_rate,
                'most_correct': dict(correct_counter.most_common(5)) if correct_counter else {},
                'most_missed': dict(missed_counter.most_common(5)) if missed_counter else {},
                'successful_pairs': dict(pair_counter.most_common(3)) if pair_counter else {},
                'best_performing_time': f"{best_hour[0]:02d}:00" if best_hour[0] else "N/A",
                'best_time_avg': best_hour[1]
            }
            
            print("Performance statistics calculated successfully.")
            return stats
            
        except Exception as e:
            print(f"Error calculating performance stats: {e}")
            traceback.print_exc()
            return None

    def _calculate_trend(self, recent_values):
        """Calculate trend using linear regression"""
        if len(recent_values) < 2:
            return 0
        return np.polyfit(range(len(recent_values)), recent_values, 1)[0]

    def _calculate_improvement_rate(self, df):
        """Calculate improvement rate between halves"""
        if len(df) < 2:
            return 0
        first_half_avg = df['Number_Correct'].iloc[:len(df) // 2].mean()
        second_half_avg = df['Number_Correct'].iloc[len(df) // 2:].mean()
        return ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0

    def plot_performance_trends(self):
        """Plot trends in prediction accuracy"""
        try:
            # Load results from Excel file
            results_file = os.path.join(PATHS.get('PROCESSED_DIR', ''), 'evaluation_results.xlsx')
            if not os.path.exists(results_file):
                print("No evaluation results file found.")
                return
                
            # Try to load the data
            try:
                df = pd.read_excel(results_file)
            except Exception as e:
                print(f"Error reading Excel file: {e}")
                return
                
            if len(df) < 1:
                print("Not enough data for trend analysis.")
                return
            
            # Check that required column exists
            if 'num_correct' not in df.columns:
                print(f"Column 'num_correct' not found in results file. Available columns: {df.columns.tolist()}")
                return
                
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Plot number of correct predictions over time
            plt.plot(df.index, df['num_correct'], marker='o', linestyle='-', color='blue')
            
            # Add average line
            avg_correct = df['num_correct'].mean()
            plt.axhline(y=avg_correct, color='r', linestyle='--', 
                        label=f'Average ({avg_correct:.1f})')
            
            # Add labels and title
            plt.title('Prediction Accuracy Over Time')
            plt.xlabel('Prediction Number')
            plt.ylabel('Correct Numbers')
            plt.grid(True)
            plt.legend()
            
            # Add date labels if available
            if 'date' in df.columns:
                # Just show a few date labels to avoid overcrowding
                if len(df) > 10:
                    step = len(df) // 5
                    tick_positions = df.index[::step].tolist()
                    if df.index[-1] not in tick_positions:  # Ensure the last index is included
                        tick_positions.append(df.index[-1])
                    tick_labels = [str(df['date'].iloc[i]) for i in tick_positions]
                    plt.xticks(tick_positions, tick_labels, rotation=45)
                else:
                    plt.xticks(df.index, df['date'], rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(PATHS.get('PROCESSED_DIR', ''), 'performance_trends.png')
            plt.savefig(plot_file)
            print(f"Performance trends plot saved to {plot_file}")
            
        except Exception as e:
            print(f"Error plotting performance trends: {e}")
            traceback.print_exc()

    def evaluate_past_predictions(self):
        """Evaluate past predictions with enhanced analysis"""
        try:
            # Get all prediction files
            prediction_files = sorted(glob.glob(os.path.join(self.predictions_dir, 'prediction_*.csv')))
            if not prediction_files:
                print("\nNo prediction files found.")
                return

            # Load historical data
            if not os.path.exists(self.historical_file):
                print(f"\nHistorical draw data file not found: {self.historical_file}")
                print("Please make sure the file exists at this location.")
                return

            print(f"\nLoading historical draws from: {self.historical_file}")
            try:
                historical_df = pd.read_csv(self.historical_file)
                print(f"Successfully loaded historical data with {len(historical_df)} draws")
            except Exception as e:
                print(f"Error loading historical data: {e}")
                return

            evaluation_results = []
            current_time = datetime.now()

            print(f"\nProcessing {len(prediction_files)} prediction files...")
            for pred_file in prediction_files:
                try:
                    # Load prediction file
                    pred_df = pd.read_csv(pred_file)
                    
                    # Load corresponding metadata
                    metadata_file = os.path.join(
                        self.metadata_dir,
                        os.path.basename(pred_file).replace('.csv', '_metadata.json')
                    )
                    metadata = {}
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)

                    # Check metadata for next_draw_time
                    next_draw_time = None
                    if 'next_draw_time' in metadata:
                        next_draw_time = metadata['next_draw_time']
                        print(f"Looking for draw at time: {next_draw_time}")
                    else:
                        print(f"Metadata does not contain next_draw_time for {pred_file}")
                        continue
                    
                    # Find matching historical draw directly using the string
                    matching_rows = historical_df[historical_df['date'] == next_draw_time]
                    if matching_rows.empty:
                        print(f"No matching historical draw for time '{next_draw_time}'")
                        continue

                    print(f"Found matching draw for time: {next_draw_time}")
                    
                    # Get predicted and actual numbers
                    predicted_numbers = pred_df['number'].tolist()
                    number_cols = [f'number{i}' for i in range(1, 21)]
                    actual_numbers = matching_rows.iloc[0][number_cols].tolist()
                    
                    # Convert to integers and handle missing values
                    actual_numbers = [int(x) for x in actual_numbers if pd.notna(x) and str(x).strip()]
                    
                    if len(actual_numbers) < 20:
                        print(f"Warning: Draw at {next_draw_time} has only {len(actual_numbers)} numbers")

                    # Compare and save results
                    result = self.save_comparison(
                        predicted_numbers,
                        actual_numbers,
                        draw_date=next_draw_time,
                        metadata=metadata
                    )
                    
                    if result:
                        evaluation_results.append(result)
                except Exception as e:
                    print(f"Error processing prediction file {pred_file}: {e}")
                    traceback.print_exc()
                    continue

            # Get and display performance stats even if there's only one result
            if evaluation_results:
                stats = self.get_performance_stats()
                self.display_summary_results(stats)
                self.plot_performance_trends()
            else:
                print("\nNo valid predictions found to evaluate.")

        except Exception as e:
            print(f"\nError in evaluation: {e}")
            traceback.print_exc()

    def display_summary_results(self, stats):
        """Display summary of evaluation results"""
        if not stats:
            print("\nNo performance statistics available.")
            return
            
        print("\n=== Overall Performance ===")
        print(f"Total predictions evaluated: {stats['total_predictions']}")
        print(f"Average correct numbers: {stats['avg_correct']:.1f}")
        print(f"Best prediction: {stats['best_prediction']} correct numbers")
        print(f"Worst prediction: {stats['worst_prediction']} correct numbers")
        print(f"Average accuracy: {stats['avg_accuracy']:.1f}%")
        print(f"Consistency score: {stats['consistency_score']:.2f}")
        print(f"Best streak: {stats['best_streak']} predictions")
        
        print("\n=== Trend Analysis ===")
        print(f"Recent trend: {stats['trend']}")
        print(f"Improvement rate: {stats['improvement_rate']:.1f}%")
        
        print("\n=== Pattern Analysis ===")
        print("Most frequently correct numbers:")
        for num, count in list(stats['most_correct'].items())[:5]:
            print(f"  Number {num}: {count} times")
        
        print("\nMost frequently missed numbers:")
        for num, count in list(stats['most_missed'].items())[:5]:
            print(f"  Number {num}: {count} times")
        
        print("\nMost successful number pairs:")
        for pair, count in list(stats['successful_pairs'].items())[:3]:
            print(f"  {pair}: {count} times")
        
        print(f"\nBest performing time: {stats['best_performing_time']} (Average correct: {stats['best_time_avg']:.1f})")

    def evaluate_single_prediction(self, prediction_file):
        """Evaluate a single prediction file"""
        try:
            if not os.path.exists(prediction_file):
                print(f"Prediction file not found: {prediction_file}")
                return None

            # Load prediction
            pred_df = pd.read_csv(prediction_file)
            
            # Load metadata
            metadata_file = prediction_file.replace('.csv', '_metadata.json')
            metadata = {}
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

            # Try to get prediction timestamp from metadata
            pred_date = None
            if 'timestamp' in metadata:
                timestamp_str = metadata.get('timestamp')
                try:
                    # First try ISO format (2025-03-12 07:36:44)
                    pred_date = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                        # Then try compact format (20250312_073644)
                        pred_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    except ValueError:
                        print(f"Could not parse timestamp: {timestamp_str}")
            
            # If no valid timestamp from metadata, try from filename
            if pred_date is None:
                timestamp_str = os.path.basename(prediction_file).replace('prediction_', '').replace('.csv', '')
                try:
                    pred_date = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                except ValueError:
                    print(f"Could not parse timestamp from filename: {timestamp_str}")
                    return None

            # Load historical data
            historical_df = pd.read_csv(self.historical_file)
            historical_df['date'] = pd.to_datetime(historical_df['date'].str.strip())
            
            # Find matching historical draw
            actual_rows = historical_df[historical_df['date'] == pred_date]
            if actual_rows.empty:
                print(f"No matching historical draw found for date {pred_date}")
                return None

            # Get numbers
            predicted_numbers = pred_df['number'].tolist()
            number_cols = [f'number{i}' for i in range(1, 21)]
            actual_numbers = actual_rows.iloc[0][number_cols].astype(int).tolist()

            # Compare and save results
            result = self.save_comparison(
                predicted_numbers,
                actual_numbers,
                draw_date=pred_date.strftime('%Y-%m-%d %H:%M:%S'),
                metadata=metadata
            )

            if result:
                print("\n=== Single Prediction Evaluation ===")
                print(f"Date: {pred_date}")
                print(f"Predicted numbers: {sorted(predicted_numbers)}")
                print(f"Actual numbers: {sorted(actual_numbers)}")
                print(f"Correct numbers: {sorted(result['correct_numbers'])}")
                print(f"Accuracy: {result['accuracy']*100:.1f}%")
                print(f"Number of correct predictions: {result['num_correct']}")
                
                if metadata:
                    print("\nModel Information:")
                    print(f"Confidence: {metadata.get('average_probability', 0):.3f}")
                    print(f"Analysis weight: {metadata.get('analysis_weight', 0):.2f}")
                    print(f"Pattern weight: {metadata.get('pattern_weight', 0):.2f}")
                    print(f"Probability weight: {metadata.get('prob_weight', 0):.2f}")

            return result

        except Exception as e:
            print(f"Error evaluating prediction: {e}")
            traceback.print_exc()
            return None

    def validate_historical_data(self):
        """Validate the historical data file format"""
        if not os.path.exists(self.historical_file):
            print(f"Historical data file not found: {self.historical_file}")
            return False
            
        try:
            print(f"Validating historical data file: {self.historical_file}")
            # Try to read the file
            df = pd.read_csv(self.historical_file)
            
            # Check required columns
            required_cols = ['date'] + [f'number{i}' for i in range(1, 21)]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"Error: Missing columns in historical data: {missing_cols}")
                print("Required format: 'date, number1, number2, ..., number20'")
                return False
                
            # Validate data types
            try:
                df['date'] = pd.to_datetime(df['date'].str.strip())
            except:
                print("Warning: Could not parse date column. Make sure it's in a standard format.")
                print("Example: '2025-03-12 12:00:00' or '12-03-2025 12:00'")
                
            # Validate number columns
            for col in [f'number{i}' for i in range(1, 21)]:
                try:
                    # Make sure numbers are valid integers
                    df[col] = df[col].astype(int)
                    # Check range (1-80 for lottery numbers)
                    if df[col].min() < 1 or df[col].max() > 80:
                        print(f"Warning: Column {col} has values outside the expected range (1-80)")
                except:
                    print(f"Warning: Column {col} has non-numeric values")
                    
            print(f"Validation complete. File contains {len(df)} draws.")
            return True
            
        except Exception as e:
            print(f"Error validating historical data: {e}")
            traceback.print_exc()
            return False

def main():
    try:
        print("\n=== Lottery Prediction Evaluator ===")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        evaluator = PredictionEvaluator()
        
        # Create sample data file if it doesn't exist
        if not os.path.exists(evaluator.historical_file):
            evaluator.create_sample_historical_data()
        
        evaluator.evaluate_past_predictions()
        
        print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nCritical error in evaluation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()