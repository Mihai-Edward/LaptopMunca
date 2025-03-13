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
# Import configuration
# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config.paths import PATHS,ensure_directories
class PredictionEvaluator:
    def __init__(self):
        """Initialize evaluator with consolidated file support"""
        self.excel_file = PATHS['ALL_PREDICTIONS_FILE']  # Changed from PREDICTIONS_DIR to ALL_PREDICTIONS_FILE
        self.historical_file = PATHS['HISTORICAL_DATA']
        self.predictions_dir = PATHS['PREDICTIONS_DIR']
        self.metadata_dir = PATHS['PREDICTIONS_METADATA_DIR']
        self.processed_dir = PATHS['PROCESSED_DIR']
        
        # Initialize all metric histories
        self.evaluation_metrics = {
            'accuracy_history': [],
            'precision_history': [],  # Add this
            'recall_history': [],     # Add this
            'total_evaluated': 0,
            'total_correct': 0,
            'best_prediction': 0,
            'avg_accuracy': 0.0
        }
    def evaluate_predictions(self):
        """Evaluate predictions from consolidated Excel file"""
        try:
            if not os.path.exists(self.excel_file):
                print(f"No consolidated predictions file found at: {self.excel_file}")
                return False

            # Load predictions from Excel
            predictions_df = pd.read_excel(self.excel_file)
            
            # Get historical data for comparison
            historical_df = pd.read_csv(PATHS['HISTORICAL_DATA'])
            
            results = []
            for _, pred_row in predictions_df.iterrows():
                # Extract prediction time and numbers
                pred_time = pred_row['next_draw_time']
                predicted_numbers = [
                    pred_row[f'number{i}'] 
                    for i in range(1, 21) 
                    if f'number{i}' in pred_row.columns
                ]
                
                # Find matching draw in historical data
                actual_draw = historical_df[historical_df['date'].str.contains(pred_time)]
                
                if not actual_draw.empty:
                    actual_numbers = [
                        actual_draw.iloc[0][f'number{i}'] 
                        for i in range(1, 21)
                    ]
                    
                    # Count matches
                    matches = len(set(predicted_numbers) & set(actual_numbers))
                    
                    results.append({
                        'prediction_time': pred_time,
                        'predicted': predicted_numbers,
                        'actual': actual_numbers,
                        'matches': matches
                    })
            
            # Store evaluation results
            self.evaluation_metrics.update({
                'total_evaluated': len(results),
                'average_matches': np.mean([r['matches'] for r in results]) if results else 0,
                'best_prediction': max([r['matches'] for r in results]) if results else 0,
                'last_evaluation': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            
            return True
            
        except Exception as e:
            print(f"Error in evaluation: {e}")
            traceback.print_exc()
            return False

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
        """Save comparison between predicted and actual numbers with enhanced metrics"""
        try:
            # Initialize metadata if None
            metadata = metadata or {}
            
            # Set up timestamp handling
            current_time = datetime.now()
            if draw_date is None:
                # Format timestamp in required format: "HH:MM  %d-%m-%Y"
                draw_date = current_time.strftime('%H:%M  %d-%m-%Y')
            
            # Calculate matches
            matches = set(predicted_numbers).intersection(set(actual_numbers))
            num_correct = len(matches)
            
            # Calculate metrics
            accuracy = (num_correct / len(actual_numbers)) * 100 if actual_numbers else 0
            precision = (num_correct / len(predicted_numbers)) * 100 if predicted_numbers else 0
            recall = (num_correct / len(actual_numbers)) * 100 if actual_numbers else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Initialize evaluation metrics if not exists
            if not hasattr(self, 'evaluation_metrics'):
                self.evaluation_metrics = {
                    'accuracy_history': [],
                    'precision_history': [],
                    'recall_history': [],
                    'f1_history': [],
                    'total_evaluated': 0,
                    'total_correct': 0,
                    'best_prediction': 0,
                    'avg_accuracy': 0.0,
                    'matched_numbers': Counter()
                }
            
            # Create result dictionary
            result = {
                'date': draw_date,
                'prediction_file': metadata.get('prediction_file', 'Unknown'),
                'num_correct': num_correct,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'matches': sorted(list(matches)),
                'predicted': sorted(predicted_numbers),
                'actual': sorted(actual_numbers),
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Update evaluation metrics
            self.evaluation_metrics['accuracy_history'].append(accuracy)
            self.evaluation_metrics['precision_history'].append(precision)
            self.evaluation_metrics['recall_history'].append(recall)
            self.evaluation_metrics['f1_history'].append(f1_score)
            self.evaluation_metrics['total_evaluated'] += 1
            self.evaluation_metrics['total_correct'] += num_correct
            self.evaluation_metrics['best_prediction'] = max(
                self.evaluation_metrics['best_prediction'], 
                num_correct
            )
            
            # Update matched numbers counter
            self.evaluation_metrics['matched_numbers'].update(matches)
            
            # Calculate average accuracy
            if self.evaluation_metrics['total_evaluated'] > 0:
                self.evaluation_metrics['avg_accuracy'] = (
                    self.evaluation_metrics['total_correct'] / 
                    (self.evaluation_metrics['total_evaluated'] * 20)
                ) * 100
            
            # Save to Excel file
            results_file = os.path.join(self.processed_dir, 'evaluation_results.xlsx')
            
            try:
                # Check if file exists
                if os.path.exists(results_file):
                    # Read existing results
                    existing_df = pd.read_excel(results_file)
                    
                    # Check if we already have this draw date
                    date_match = existing_df['date'] == draw_date
                    if any(date_match):
                        # Update existing entry
                        for key, value in result.items():
                            if key in existing_df.columns:
                                existing_df.loc[date_match, key] = str(value) if isinstance(value, (list, set)) else value
                        results_df = existing_df
                    else:
                        # Convert lists to strings for Excel storage
                        result_copy = result.copy()
                        for key in ['matches', 'predicted', 'actual']:
                            result_copy[key] = str(result_copy[key])
                        
                        # Append new result
                        new_df = pd.DataFrame([result_copy])
                        results_df = pd.concat([existing_df, new_df], ignore_index=True)
                else:
                    # Create new file
                    # Convert lists to strings for Excel storage
                    result_copy = result.copy()
                    for key in ['matches', 'predicted', 'actual']:
                        result_copy[key] = str(result_copy[key])
                    results_df = pd.DataFrame([result_copy])
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(results_file), exist_ok=True)
                
                # Save to Excel
                results_df.to_excel(results_file, index=False)
                print(f"Results saved to {results_file}")
                
                # Print evaluation summary
                print(f"\nEvaluation Summary for {draw_date}:")
                print(f"Numbers matched: {num_correct} of 20")
                print(f"Accuracy: {accuracy:.2f}%")
                print(f"Precision: {precision:.2f}%")
                print(f"Recall: {recall:.2f}%")
                print(f"F1 Score: {f1_score:.2f}")
                if matches:
                    print(f"Matched numbers: {sorted(matches)}")
                
                return result
                
            except Exception as excel_error:
                print(f"Error saving to Excel: {excel_error}")
                traceback.print_exc()
                # Still return the result even if Excel save fails
                return result
                
        except Exception as e:
            print(f"Error in save_comparison: {e}")
            traceback.print_exc()
            return None
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
                if (above_avg):
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

    def set_evaluation_stats(self, stats):
        """
        Store evaluation statistics for later use by other components
        
        Args:
            stats: Dictionary containing evaluation statistics
        """
        try:
            print("Storing evaluation statistics for external use")
            
            # Store the stats in the instance for later use
            if not hasattr(self, 'stored_stats'):
                self.stored_stats = {}
                
            # Update the stored stats with new values
            self.stored_stats.update(stats)
            
            # Save a copy to the evaluation_metrics attribute too
            self.evaluation_metrics.update({
                'last_evaluation': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_predictions': stats.get('total_predictions', 0),
                'avg_accuracy': stats.get('avg_accuracy', 0),
                'best_prediction': stats.get('best_prediction', 0)
            })
            
            return True
        except Exception as e:
            print(f"Error storing evaluation stats: {e}")
            traceback.print_exc()
            return False

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
            # Instead of looking for CSV files in predictions_dir, 
            # we should look directly at the Excel file
            print("\n=== Evaluating Predictions ===")
            
            if not os.path.exists(self.excel_file):
                print(f"\nNo predictions file found at: {self.excel_file}")
                return
                
            try:
                # Load predictions from Excel
                predictions_df = pd.read_excel(self.excel_file)
                print(f"Loaded {len(predictions_df)} predictions from Excel")
                
                if len(predictions_df) == 0:
                    print("No predictions found in Excel file.")
                    return
                    
                # Load historical data
                if not os.path.exists(self.historical_file):
                    print(f"\nHistorical draw data file not found: {self.historical_file}")
                    return

                historical_df = pd.read_csv(self.historical_file)
                print(f"Loaded {len(historical_df)} historical draws")
                
                evaluation_results = []
                
                for idx, pred_row in predictions_df.iterrows():
                    try:
                        # Get the draw time
                        next_draw_time = pred_row['next_draw_time']
                        
                        # Extract predicted numbers
                        predicted_numbers = []
                        for i in range(1, 21):
                            col_name = f'number{i}'
                            if col_name in pred_row and not pd.isna(pred_row[col_name]):
                                predicted_numbers.append(int(pred_row[col_name]))
                        
                        if len(predicted_numbers) != 20:
                            print(f"Warning: Invalid prediction for {next_draw_time} - wrong number count")
                            continue
                        
                        # Find matching actual draw
                        matching_draws = historical_df[historical_df['date'].str.contains(str(next_draw_time))]
                        if len(matching_draws) == 0:
                            print(f"No matching draw found for {next_draw_time}")
                            continue
                            
                        # Get actual numbers
                        actual_numbers = []
                        for i in range(1, 21):
                            col_name = f'number{i}'
                            val = matching_draws.iloc[0][col_name]
                            if not pd.isna(val):
                                actual_numbers.append(int(val))
                        
                        # Compare and save results
                        result = self.save_comparison(
                            predicted_numbers,
                            actual_numbers,
                            draw_date=next_draw_time
                        )
                        
                        if result:
                            evaluation_results.append(result)
                            print(f"Evaluated prediction for {next_draw_time}: {result['num_correct']} correct")
                        
                    except Exception as row_error:
                        print(f"Error processing prediction row {idx}: {row_error}")
                        continue
                
                # Get and display performance stats
                if evaluation_results:
                    stats = self.get_performance_stats()
                    self.display_summary_results(stats)
                    self.plot_performance_trends()
                    print(f"\nSuccessfully evaluated {len(evaluation_results)} predictions")
                else:
                    print("\nNo valid predictions found to evaluate.")
                    
            except Exception as e:
                print(f"Error processing predictions: {e}")
                traceback.print_exc()
                
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
        
        # Just call evaluate_past_predictions directly
        evaluator.evaluate_past_predictions()
        
        print(f"\nEvaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nCritical error in evaluation: {e}")
        traceback.print_exc()