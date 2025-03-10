import os
import sys
import pandas as pd
from datetime import datetime
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import configuration – ensure PATHS and ensure_directories are correctly configured in config/paths.py
from config.paths import PATHS, ensure_directories

class PredictionEvaluator:
    def __init__(self):
        # Use PATHS configuration for relative paths.
        # Predictions file is expected to be in the processed folder (without header):
        #   Column 0: draw date (format "HH:MM %d-%m-%Y")
        #   Column 1: predicted numbers as comma-separated string
        #   Column 2: probabilities (ignored)
        self.predictions_file = PATHS['PREDICTIONS']       # e.g., .../data/processed/predictions.csv
        # Historical draws file is expected to be in the src folder (with header):
        #   Header row: "date", "number1", "number2", ..., "number20"
        self.historical_file = PATHS['HISTORICAL_DATA']      # e.g., .../src/historical_draws.csv
        self.results_dir = os.path.dirname(self.predictions_file)
        self.results_file = os.path.join(self.results_dir, 'evaluation_results.xlsx')
        ensure_directories()

        # Initialize evaluation metrics tracking (optional)
        self.evaluation_metrics = {
            'accuracy_history': [],
            'precision_history': [],
            'recall_history': [],
            'f1_history': []
        }

    def compare_prediction_with_actual(self, predicted_numbers, actual_numbers):
        """
        Compare predicted numbers with actual draw numbers.
        Returns a dictionary with accuracy, precision, recall, F1 score, and lists of matching numbers.
        """
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

    def save_comparison(self, predicted_numbers, actual_numbers, draw_date=None):
        """
        Compare one prediction with actual draw numbers and append the results to an Excel file.
        """
        if draw_date is None:
            draw_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        result = self.compare_prediction_with_actual(predicted_numbers, actual_numbers)

        data = {
            'Date': [draw_date],
            'Predicted_Numbers': [str(result['predicted_numbers'])],
            'Actual_Numbers': [str(result['actual_numbers'])],
            'Correct_Numbers': [str(result['correct_numbers'])],
            'Number_Correct': [result['num_correct']],
            'Accuracy (%)': [f"{result['accuracy'] * 100:.2f}"],
            'Precision (%)': [f"{result['precision'] * 100:.2f}"],
            'Recall (%)': [f"{result['recall'] * 100:.2f}"],
            'F1 Score (%)': [f"{result['f1_score'] * 100:.2f}"]
        }

        df = pd.DataFrame(data)
        # Append to existing evaluation results if available
        if os.path.exists(self.results_file):
            try:
                existing_df = pd.read_excel(self.results_file)
                df = pd.concat([existing_df, df], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not load existing results file: {e}")

        try:
            df.to_excel(self.results_file, index=False)
            print(f"\nResults saved to {self.results_file}")
            # Update evaluation metrics
            self.evaluation_metrics['accuracy_history'].append(result['accuracy'])
            self.evaluation_metrics['precision_history'].append(result['precision'])
            self.evaluation_metrics['recall_history'].append(result['recall'])
            self.evaluation_metrics['f1_history'].append(result['f1_score'])
        except Exception as e:
            print(f"Warning: Could not save evaluation results: {e}")

        return result

    def analyze_prediction_patterns(self, results_df):
        """
        Analyze prediction patterns including frequency of correct and missed numbers, and streaks.
        """
        patterns = {
            'frequent_correct': Counter(),
            'frequent_missed': Counter(),
            'time_based_accuracy': {},
            'streak_analysis': {'current_streak': 0, 'best_streak': 0}
        }

        for index, row in results_df.iterrows():
            try:
                correct_nums = eval(str(row['Correct_Numbers']))
                predicted = eval(str(row['Predicted_Numbers']))
                actual = eval(str(row['Actual_Numbers']))
            except Exception as e:
                print(f"Error evaluating row {index}: {e}")
                continue

            patterns['frequent_correct'].update(correct_nums)
            missed = set(actual) - set(predicted)
            patterns['frequent_missed'].update(missed)

            try:
                date = pd.to_datetime(row['Date'])
                time_key = f"{date.hour:02d}:00"
                if time_key not in patterns['time_based_accuracy']:
                    patterns['time_based_accuracy'][time_key] = []
                patterns['time_based_accuracy'][time_key].append(row['Number_Correct'])
            except Exception as e:
                print(f"Error processing date for row {index}: {e}")

            if row['Number_Correct'] >= 5:
                patterns['streak_analysis']['current_streak'] += 1
                patterns['streak_analysis']['best_streak'] = max(
                    patterns['streak_analysis']['best_streak'],
                    patterns['streak_analysis']['current_streak']
                )
            else:
                patterns['streak_analysis']['current_streak'] = 0

        return patterns

    def get_performance_stats(self):
        """
        Calculate overall performance statistics based on evaluation results.
        """
        try:
            if not os.path.exists(self.results_file):
                return None

            df = pd.read_excel(self.results_file)
            if df.empty:
                return None

            stats = {
                'total_predictions': len(df),
                'average_correct': df['Number_Correct'].mean(),
                'best_prediction': df['Number_Correct'].max(),
                'worst_prediction': df['Number_Correct'].min(),
                'average_accuracy': df['Number_Correct'].mean() / 20 * 100,
                'recent_trend': self._calculate_trend(df['Number_Correct'].tail(5).values),
                'consistency_score': np.std(df['Number_Correct']),
                'improvement_rate': self._calculate_improvement_rate(df)
            }

            patterns = self.analyze_prediction_patterns(df)
            stats.update({
                'best_streak': patterns['streak_analysis']['best_streak'],
                'most_frequent_correct': dict(patterns['frequent_correct'].most_common(5)),
                'most_frequently_missed': dict(patterns['frequent_missed'].most_common(5))
            })

            return stats
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return None

    def _calculate_trend(self, recent_values):
        """Calculate trend using a linear fit on recent evaluation values."""
        if len(recent_values) < 2:
            return 0
        return np.polyfit(range(len(recent_values)), recent_values, 1)[0]

    def _calculate_improvement_rate(self, df):
        """Calculate improvement rate by comparing averages of first and second halves."""
        if len(df) < 2:
            return 0
        first_half_avg = df['Number_Correct'].iloc[:len(df) // 2].mean()
        second_half_avg = df['Number_Correct'].iloc[len(df) // 2:].mean()
        return ((second_half_avg - first_half_avg) / first_half_avg) * 100 if first_half_avg > 0 else 0

    def plot_performance_trends(self):
        """Generate and save plots of performance trends."""
        if not os.path.exists(self.results_file):
            return

        try:
            df = pd.read_excel(self.results_file)
            if df.empty:
                return

            plt.figure(figsize=(15, 10))
            plt.subplot(2, 2, 1)
            plt.plot(df['Number_Correct'])
            plt.title('Correct Numbers Over Evaluations')
            plt.xlabel('Evaluation Index')
            plt.ylabel('Correct Numbers')

            plt.subplot(2, 2, 2)
            sns.histplot(df['Number_Correct'])
            plt.title('Distribution of Correct Predictions')

            plot_path = os.path.join(self.results_dir, 'performance_trends.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Performance trends plot saved to {plot_path}")
        except Exception as e:
            print(f"Error plotting performance trends: {e}")

    def evaluate_past_predictions(self):
        """
        Evaluate past predictions by comparing the predictions CSV (without header) 
        with the historical draws CSV (with header).
        Expected format:
          Predictions file columns:
            0: date (format "HH:MM %d-%m-%Y")
            1: predicted numbers as a comma-separated string
            2: probabilities (ignored)
          Historical draws file:
            Header: date,number1,number2,...,number20
        """
        try:
            if not os.path.exists(self.predictions_file):
                print("\nNo predictions file found.")
                return
            if not os.path.exists(self.historical_file):
                print("\nNo historical draw data found.")
                return

            predictions_df = pd.read_csv(self.predictions_file, header=None)
            historical_df = pd.read_csv(self.historical_file, header=0)

            print(f"Predictions file shape: {predictions_df.shape}")
            print(f"Historical draws file shape: {historical_df.shape}")
            print("First row of predictions:", predictions_df.iloc[0].tolist())
            print("First row of historical draws:", historical_df.iloc[0].tolist())

            if predictions_df.empty or historical_df.empty:
                print("\nNo data to compare.")
                return

            current_time = datetime.now()

            # Convert predictions date in column 0
            predictions_df[0] = pd.to_datetime(predictions_df[0].str.strip(), format='%H:%M %d-%m-%Y', errors='coerce')
            # Convert historical date (column "date")
            historical_df['date'] = pd.to_datetime(historical_df['date'].str.strip(), format='%H:%M %d-%m-%Y', errors='coerce')

            print("Converted date in first prediction row:", predictions_df.loc[0, 0])
            print("Converted date in first historical row:", historical_df.loc[0, "date"])

            # Filter predictions that are in the past
            past_predictions = predictions_df[predictions_df[0] < current_time]
            print(f"Number of past predictions: {len(past_predictions)}")

            evaluation_results = []
            for idx, pred_row in past_predictions.iterrows():
                pred_date = pred_row[0]
                print(f"Processing prediction row {idx} with date {pred_date}")

                # Find matching historical draw by date
                actual_rows = historical_df[historical_df["date"] == pred_date]
                if actual_rows.empty:
                    print(f"No matching historical draw for date {pred_date}")
                    continue

                # Get predicted numbers from column 1
                try:
                    predicted_numbers = [int(x.strip()) for x in pred_row[1].split(",") if x.strip().isdigit()]
                except Exception as e:
                    print(f"Error parsing predicted numbers for row {idx}: {e}")
                    continue

                # Get actual numbers from historical draw
                number_cols = [f"number{i}" for i in range(1, 21)]
                try:
                    actual_numbers = actual_rows.iloc[0][number_cols].astype(int).tolist()
                except Exception as e:
                    print(f"Error parsing actual numbers for row {idx}: {e}")
                    continue

                print(f"Row {idx} predicted numbers: {predicted_numbers}")
                print(f"Row {idx} actual numbers: {actual_numbers}")

                result = self.save_comparison(
                    predicted_numbers,
                    actual_numbers,
                    draw_date=pred_date.strftime('%Y-%m-%d %H:%M:%S')
                )
                evaluation_results.append(result)

            if evaluation_results:
                stats = self.get_performance_stats()
                self.display_summary_results(stats)
                self.plot_performance_trends()
            else:
                print("\nNo past predictions found to evaluate.")
        except Exception as e:
            print(f"\nError in evaluation: {e}")
            print("Please try again.")

    def display_summary_results(self, stats):
        """
        Display overall performance statistics.
        """
        if stats:
            print("\n=== Overall Performance ===")
            print(f"Total predictions evaluated: {stats['total_predictions']}")
            print(f"Average correct numbers: {stats['average_correct']:.1f}")
            print(f"Best prediction: {stats['best_prediction']} correct numbers")
            print(f"Worst prediction: {stats['worst_prediction']} correct numbers")
            print(f"Average accuracy: {stats['average_accuracy']:.1f}%")
            print(f"Best streak: {stats['best_streak']} predictions")
            print(f"\nRecent trend: {'Improving' if stats['recent_trend'] > 0 else 'Declining'}")
            print(f"Improvement rate: {stats['improvement_rate']:.1f}%")
            print("\nMost frequently correct numbers:")
            for num, count in stats['most_frequent_correct'].items():
                print(f"Number {num}: {count} times")
            print("\nMost frequently missed numbers:")
            for num, count in stats['most_frequently_missed'].items():
                print(f"Number {num}: {count} times")
        else:
            print("\nNo performance statistics available.")

if __name__ == "__main__":
    evaluator = PredictionEvaluator()
    evaluator.evaluate_past_predictions()