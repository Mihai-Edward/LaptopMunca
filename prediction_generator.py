import os
import sys
import argparse
from datetime import datetime
from collections import Counter
from itertools import combinations

# Add parent directory to path to import from data_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_analysis import DataAnalysis
from config.paths import PATHS, ensure_directories

class PredictionGenerator:
    """Generate predictions based on historical data analysis"""
    
    def __init__(self, data_analysis=None):
        """Initialize with optional DataAnalysis object"""
        self.analysis = data_analysis if data_analysis else DataAnalysis()
    
    def generate_prediction(self, predict_count=15, recent_window=150, 
                           freq_weight=0.25, hot_cold_weight=0.25,
                           gap_weight=0.25, triplets_weight=0.25,
                           historical_influence=0.3):
        """
        Generate enhanced prediction using multiple analysis methods
        with tiered confidence levels and triplet analysis
        
        Args:
            predict_count (int): Number of numbers to predict (default 15)
            recent_window (int): Number of recent draws to focus on (default 150)
            freq_weight (float): Weight for frequency analysis (default 0.25)
            hot_cold_weight (float): Weight for hot/cold analysis (default 0.25)
            gap_weight (float): Weight for gap analysis (default 0.25)
            triplets_weight (float): Weight for common triplets analysis (default 0.25)
            historical_influence (float): Weight given to historical patterns vs. recent (default 0.3)
            
        Returns:
            dict: Prediction results with confidence tiers
        """
        # Initialize base scores
        base_scores = {num: 0 for num in range(1, 81)}
        
        # Get recent draws for windowing
        recent_draws = self.analysis.draws[-recent_window:] if len(self.analysis.draws) >= recent_window else self.analysis.draws
        
        # Print debug info about window size
        print(f"DEBUG: Using {len(recent_draws)} recent draws out of {len(self.analysis.draws)} total draws")
        
        # Create a windowed analysis object that only uses recent draws
        windowed_analysis = DataAnalysis(recent_draws, debug=False)
        
        # 1. RECENT PATTERNS ANALYSIS (70% weight by default)
        # --------------------------------------------------
        
        # Component 1: Frequency Analysis
        recent_frequency = windowed_analysis.count_frequency()
        max_recent_freq = max(recent_frequency.values()) if recent_frequency else 1
        
        for num, freq in recent_frequency.items():
            base_scores[num] += ((freq / max_recent_freq) * freq_weight) * (1 - historical_influence)
        
        # Component 2: Hot/Cold Analysis - NO ADDITIONAL WINDOW
        recent_hot_cold = windowed_analysis.hot_and_cold_numbers()  # Use the entire windowed dataset
        # Get trending numbers with their scores
        recent_trending_up = recent_hot_cold['trending_up']
        recent_trending_numbers = {num: data['trend'] for num, data in recent_trending_up}
        max_recent_trend = max(recent_trending_numbers.values()) if recent_trending_numbers else 1
        
        for num, trend in recent_trending_numbers.items():
            # Normalize and add to score
            base_scores[num] += ((trend / max_recent_trend) * hot_cold_weight) * (1 - historical_influence)
        
        # Component 3: Gap Analysis
        recent_gaps = windowed_analysis.analyze_gaps()
        for num, gap_data in recent_gaps.items():
            current_gap = gap_data['current_gap']
            avg_gap = gap_data['avg_gap']
            
            if avg_gap > 0:
                # Treat overdue numbers neutrally and score based on readiness
                if current_gap >= avg_gap:
                    # Number is overdue or exactly on time - neutral score
                    gap_score = 0.5 * gap_weight
                else:
                    # Number has appeared recently - score based on "readiness" to appear again
                    readiness = 1 - (current_gap / avg_gap)  # Readiness is inversely proportional to gap
                    gap_score = readiness * gap_weight
                
                base_scores[num] += gap_score * (1 - historical_influence)
        
        # 2. HISTORICAL PATTERNS ANALYSIS (30% weight by default)
        # ------------------------------------------------------
        
        # Component 1: Overall Frequency Analysis
        historical_frequency = self.analysis.count_frequency()
        max_hist_freq = max(historical_frequency.values()) if historical_frequency else 1
        
        for num, freq in historical_frequency.items():
            base_scores[num] += ((freq / max_hist_freq) * freq_weight) * historical_influence
        
        # Component 2: Overall Hot/Cold Analysis - NO ADDITIONAL WINDOW
        historical_hot_cold = self.analysis.hot_and_cold_numbers()  # Use the entire historical dataset
        # Get trending numbers with their scores
        historical_trending_up = historical_hot_cold['trending_up']
        historical_trending_numbers = {num: data['trend'] for num, data in historical_trending_up}
        max_hist_trend = max(historical_trending_numbers.values()) if historical_trending_numbers else 1
        
        for num, trend in historical_trending_numbers.items():
            # Normalize and add to score
            base_scores[num] += ((trend / max_hist_trend) * hot_cold_weight) * historical_influence
        
        # Component 3: Overall Gap Analysis
        historical_gaps = self.analysis.analyze_gaps()
        for num, gap_data in historical_gaps.items():
            current_gap = gap_data['current_gap']
            avg_gap = gap_data['avg_gap']
            # Give higher scores to numbers more overdue than their average
            if avg_gap > 0:
                # Improved gap ratio calculation with cap to prevent extreme values
                gap_ratio = min(2.0, current_gap / avg_gap)
                base_scores[num] += ((gap_ratio / 2.0) * gap_weight) * historical_influence
        
        # Get initial candidate numbers based on base scores
        candidates = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_numbers = [num for num, _ in candidates[:predict_count*2]]  # Get more candidates than needed
        
        # Component 4: Common Triplets Analysis (using both recent and historical)
        # Find triplets instead of pairs
        recent_common_triplets = self.find_common_triplets(windowed_analysis.draws, top_n=30)
        historical_common_triplets = self.find_common_triplets(self.analysis.draws, top_n=30)
        
        # Combine triplets with appropriate weights
        combined_triplets = {}
        for triplet, freq in recent_common_triplets:
            combined_triplets[triplet] = freq * (1 - historical_influence)
        
        for triplet, freq in historical_common_triplets:
            if triplet in combined_triplets:
                combined_triplets[triplet] += freq * historical_influence
            else:
                combined_triplets[triplet] = freq * historical_influence
        
        # Sort combined triplets
        sorted_triplets = sorted(combined_triplets.items(), key=lambda x: x[1], reverse=True)
        
        # Store the top common triplets for display later
        top_triplets = sorted_triplets[:10]
        
        # Final scores with triplet boosts
        final_scores = base_scores.copy()
        
        # Boost scores for numbers that appear in common triplets with high-scoring candidates
        # First, check all possible triplets among candidates
        for i, num1 in enumerate(candidate_numbers):
            for j, num2 in enumerate(candidate_numbers[i+1:], i+1):
                for num3 in candidate_numbers[j+1:]:
                    triplet = tuple(sorted((num1, num2, num3)))
                    if triplet in combined_triplets:
                        triplet_strength = combined_triplets[triplet] / max(combined_triplets.values()) if combined_triplets else 0
                        # Boost all three numbers in the triplet
                        final_scores[num1] += (triplets_weight / 15) * triplet_strength
                        final_scores[num2] += (triplets_weight / 15) * triplet_strength
                        final_scores[num3] += (triplets_weight / 15) * triplet_strength
        
        # Select final numbers
        sorted_final = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        final_numbers = [num for num, _ in sorted_final[:predict_count]]
        final_scores = [score for _, score in sorted_final[:predict_count]]
        
        # Create confidence tiers
        top_2 = final_numbers[:2]
        top_5 = final_numbers[:5]
        top_10 = final_numbers[:10]
        top_15 = final_numbers[:15]
        
        # Normalize confidence scores for readability (0-100%)
        max_score = max(final_scores) if final_scores else 1
        normalized_scores = [round((score / max_score) * 100, 1) for score in final_scores]
        
        # Create prediction result
        prediction = {
            'numbers': final_numbers,
            'confidence_scores': normalized_scores,
            'confidence_tiers': {
                'very_high': top_2,
                'high': top_5[2:5],  # Excluding the first 2
                'medium': top_10[5:10],  # Excluding the first 5
                'moderate': top_15[10:15]  # Excluding the first 10
            },
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Add two types of common triplets information:
        # 1. Triplets that exist within the final prediction
        common_triplets_in_prediction = []
        for i, num1 in enumerate(final_numbers):
            for j, num2 in enumerate(final_numbers[i+1:], i+1):
                for num3 in final_numbers[j+1:]:
                    triplet = tuple(sorted((num1, num2, num3)))
                    if triplet in combined_triplets:
                        common_triplets_in_prediction.append({
                            'triplet': triplet,
                            'frequency': combined_triplets[triplet]
                        })
        
        # 2. Overall most common triplets from the analysis
        top_common_triplets = [
            {
                'triplet': triplet,
                'frequency': freq,
                'in_prediction': all(num in final_numbers for num in triplet)
            }
            for triplet, freq in top_triplets
        ]
        
        prediction['common_triplets'] = sorted(
            common_triplets_in_prediction, 
            key=lambda x: x['frequency'], 
            reverse=True
        )[:10]  # Top 10 common triplets in our prediction
        
        prediction['top_common_triplets'] = top_common_triplets
        
        return prediction

    def find_common_triplets(self, draws, top_n=30):
        """Find most common triplets of numbers"""
        triplets = Counter()
        for _, numbers in draws:
            # Get all possible combinations of 3 numbers from this draw
            triplets.update(combinations(sorted(numbers), 3))
        
        # Return the most common triplets
        common_triplets = triplets.most_common(top_n)
        print(f"DEBUG: Found {len(common_triplets)} common triplets")
        return common_triplets

    def display_prediction(self, prediction):
        """Display prediction results in a user-friendly format"""
        print("\n" + "="*50)
        print(" "*15 + "KENO NUMBER PREDICTION")
        print("="*50)
        print(f"Generated at: {prediction['timestamp']}")
        
        print("\n" + "="*20 + " CONFIDENCE TIERS " + "="*20)
        
        print("\n🔴 VERY HIGH CONFIDENCE (Top 2)")
        for i, num in enumerate(prediction['confidence_tiers']['very_high']):
            idx = prediction['numbers'].index(num)
            print(f"    {num} ({prediction['confidence_scores'][idx]}%)")
        
        print("\n🟠 HIGH CONFIDENCE (Next 3)")
        for i, num in enumerate(prediction['confidence_tiers']['high']):
            idx = prediction['numbers'].index(num)
            print(f"    {num} ({prediction['confidence_scores'][idx]}%)")
        
        print("\n🟡 MEDIUM CONFIDENCE (Next 5)")
        for i, num in enumerate(prediction['confidence_tiers']['medium']):
            idx = prediction['numbers'].index(num)
            print(f"    {num} ({prediction['confidence_scores'][idx]}%)")
        
        print("\n🟢 MODERATE CONFIDENCE (Next 5)")
        for i, num in enumerate(prediction['confidence_tiers']['moderate']):
            idx = prediction['numbers'].index(num)
            print(f"    {num} ({prediction['confidence_scores'][idx]}%)")
        
        # Display common triplets instead of pairs
        print("\n" + "="*20 + " COMMON TRIPLETS IN PREDICTION " + "="*20)
        if 'common_triplets' in prediction and prediction['common_triplets']:
            for i, triplet_info in enumerate(prediction['common_triplets'][:5]):
                print(f"    {triplet_info['triplet'][0]}, {triplet_info['triplet'][1]}, and {triplet_info['triplet'][2]} (Appears together {int(triplet_info['frequency'])} times)")
        else:
            print("    No common triplets found among predicted numbers")
        
        if 'top_common_triplets' in prediction:
            print("\n" + "="*20 + " TOP COMMON TRIPLETS OVERALL " + "="*20)
            for i, triplet_info in enumerate(prediction['top_common_triplets'][:5]):
                status = "✓" if triplet_info['in_prediction'] else " "
                print(f"    {status} {triplet_info['triplet'][0]}, {triplet_info['triplet'][1]}, and {triplet_info['triplet'][2]} (Appears together {int(triplet_info['frequency'])} times)")
        
        print("\n" + "="*20 + " ALL PREDICTED NUMBERS " + "="*20)
        formatted_numbers = ", ".join([str(num).rjust(2) for num in prediction['numbers']])
        print(f"    {formatted_numbers}")
        print("\n" + "="*50)

    # The other methods remain the same
    def evaluate_prediction(self, prediction, actual_draw):
        """Evaluate the accuracy of a prediction against actual draw numbers"""
        prediction_set = set(prediction['numbers'])
        actual_set = set(actual_draw)
        correct_numbers = prediction_set.intersection(actual_set)
        
        # Debug the actual draw
        print(f"DEBUG: Evaluating against draw: {sorted(actual_draw)}")
        
        # Overall accuracy
        accuracy = len(correct_numbers) / len(prediction['numbers'])
        
        # Accuracy by tier
        very_high_correct = len(set(prediction['confidence_tiers']['very_high']).intersection(actual_set))
        high_correct = len(set(prediction['confidence_tiers']['high']).intersection(actual_set))
        medium_correct = len(set(prediction['confidence_tiers']['medium']).intersection(actual_set))
        moderate_correct = len(set(prediction['confidence_tiers']['moderate']).intersection(actual_set))
        
        # Expected random accuracy (20/80 = 0.25 hit rate)
        expected_correct = len(prediction['numbers']) * (20/80)
        
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_numbers': prediction['numbers'],
            'actual_numbers': list(actual_set),
            'correct_numbers': list(correct_numbers),
            'total_correct': len(correct_numbers),
            'accuracy': accuracy,
            'tier_accuracy': {
                'very_high': very_high_correct / len(prediction['confidence_tiers']['very_high']) if prediction['confidence_tiers']['very_high'] else 0,
                'high': high_correct / len(prediction['confidence_tiers']['high']) if prediction['confidence_tiers']['high'] else 0,
                'medium': medium_correct / len(prediction['confidence_tiers']['medium']) if prediction['confidence_tiers']['medium'] else 0,
                'moderate': moderate_correct / len(prediction['confidence_tiers']['moderate']) if prediction['confidence_tiers']['moderate'] else 0
            },
            'better_than_random': len(correct_numbers) > expected_correct
        }
        
        return result
    
    def display_evaluation(self, evaluation):
        """Display evaluation results in a user-friendly format"""
        print("\n" + "="*50)
        print(" "*15 + "PREDICTION EVALUATION")
        print("="*50)
        print(f"Evaluated at: {evaluation['timestamp']}")
        
        # Correct numbers
        print("\n" + "="*20 + " CORRECT PREDICTIONS " + "="*20)
        correct_str = ", ".join([str(num).rjust(2) for num in sorted(evaluation['correct_numbers'])])
        print(f"    {correct_str}" if correct_str else "    None")
        
        # Overall metrics
        print("\n" + "="*20 + " PERFORMANCE METRICS " + "="*20)
        print(f"    Total Correct: {evaluation['total_correct']} out of {len(evaluation['predicted_numbers'])}")
        print(f"    Overall Accuracy: {evaluation['accuracy']*100:.1f}%")
        print(f"    Better Than Random: {'✅ Yes' if evaluation['better_than_random'] else '❌ No'}")
        
        # Tier accuracy
        print("\n" + "="*20 + " TIER ACCURACY " + "="*20)
        print(f"    Very High Confidence: {evaluation['tier_accuracy']['very_high']*100:.1f}%")
        print(f"    High Confidence: {evaluation['tier_accuracy']['high']*100:.1f}%")
        print(f"    Medium Confidence: {evaluation['tier_accuracy']['medium']*100:.1f}%")
        print(f"    Moderate Confidence: {evaluation['tier_accuracy']['moderate']*100:.1f}%")
        
        print("\n" + "="*50)
    
    def save_prediction(self, prediction, filename=None):
        """Save prediction to file"""
        import json
        from pathlib import Path
        
        # Ensure directories exist
        ensure_directories()
        
        # Use default filename if none provided
        if filename is None:
            predictions_dir = PATHS['PREDICTIONS_DIR']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = os.path.join(predictions_dir, f"prediction_{timestamp}.json")
        
        # Save prediction as JSON
        try:
            with open(filename, 'w') as f:
                json.dump(prediction, f, indent=4)
            print(f"\nPrediction saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def get_latest_draw(self):
        """Get the most recent draw from historical data"""
        if not self.analysis.draws:
            return None
        return self.analysis.draws[-1][1]  # Return just the numbers from the last draw


def main():
    """Main function to run the prediction generator"""
    try:
        print("Starting prediction generation...")
        parser = argparse.ArgumentParser(description='Generate Keno number predictions')
        parser.add_argument('--save', action='store_true', help='Save prediction to file')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate against most recent draw')
        args = parser.parse_args()
        
        # Initialize prediction generator
        print("Initializing prediction generator...")
        predictor = PredictionGenerator()
        
        # Generate prediction with configurable parameters
        print("Generating prediction with triplet analysis...")
        prediction = predictor.generate_prediction(
            predict_count=15,          # Number of numbers to predict
            recent_window=60,         # Focus on most recent 60 draws
            freq_weight=0.25,          # Weight for frequency analysis
            hot_cold_weight=0.25,      # Weight for hot/cold analysis
            gap_weight=0.25,           # Weight for gap analysis
            triplets_weight=0.25,      # Weight for common triplets analysis
            historical_influence=0.10   # 10% weight to overall historical patterns
        )
        
        # Display prediction
        print("Displaying prediction...")
        predictor.display_prediction(prediction)
        
        # Save prediction with timestamp info
        print("Preparing to save prediction...")
        predictions_dir = PATHS['PREDICTIONS_DIR']
        
        # Ensure the predictions directory exists
        print(f"Checking if directory exists: {predictions_dir}")
        if not os.path.exists(predictions_dir):
            print(f"Creating directory: {predictions_dir}")
            os.makedirs(predictions_dir, exist_ok=True)
        
        timestamp = datetime.now()
        
        # Get current time and format it for filename
        time_str = timestamp.strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(predictions_dir, f"prediction_{time_str}.json")
        print(f"Will save prediction to: {filename}")
        
        # Add target draw time info to the prediction
        # Assuming draws are every 5 minutes (00, 05, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
        current_minute = timestamp.minute
        target_minute = ((current_minute // 5) + 1) * 5  # Next 5-minute interval
        
        if target_minute >= 60:
            target_hour = (timestamp.hour + 1) % 24
            target_minute = 0
        else:
            target_hour = timestamp.hour
        
        # Create target timestamp
        target_time = timestamp.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)
        target_time_str = target_time.strftime('%H:%M %d-%m-%Y')
        
        # Add target draw time to prediction
        prediction['target_draw_time'] = target_time_str
        prediction['prediction_made_at'] = timestamp.strftime('%H:%M %d-%m-%Y')
        
        # Save the prediction
        print("Saving prediction...")
        save_result = predictor.save_prediction(prediction, filename)
        print(f"Save result: {'Success' if save_result else 'Failed'}")
        
        # Evaluate the previous prediction against the latest draw
        print("Getting latest draw for evaluation...")
        latest_draw = predictor.get_latest_draw()
        
        if latest_draw:
            print(f"Latest draw found: {latest_draw}")
            
            # Check if predictions directory exists
            if os.path.exists(predictions_dir):
                print("Looking for previous prediction files...")
                
                # Find prediction files (excluding the one we just created)
                prediction_files = [f for f in os.listdir(predictions_dir) 
                                  if f.startswith('prediction_') and f.endswith('.json')
                                  and f != os.path.basename(filename)]
                
                print(f"Found {len(prediction_files)} previous prediction files")
                
                if prediction_files:
                    # Sort by timestamp, get the most recent
                    prediction_files.sort(reverse=True)
                    prev_prediction_file = os.path.join(predictions_dir, prediction_files[0])
                    print(f"Using most recent prediction file: {prev_prediction_file}")
                    
                    try:
                        # Load the previous prediction
                        print("Loading previous prediction...")
                        with open(prev_prediction_file, 'r') as f:
                            import json
                            prev_prediction = json.load(f)
                        
                        # Get the latest draw date string 
                        draw_time_str = predictor.analysis.draws[-1][0]
                        print(f"\nEvaluating prediction made for: {prev_prediction.get('target_draw_time', 'unknown time')}")
                        print(f"Against actual draw at: {draw_time_str}")
                        
                        # Evaluate and display
                        print("Evaluating prediction...")
                        evaluation = predictor.evaluate_prediction(prev_prediction, latest_draw)
                        predictor.display_evaluation(evaluation)
                        
                    except Exception as e:
                        print(f"Error evaluating previous prediction: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("\nNo previous predictions found to evaluate")
            else:
                print(f"Predictions directory does not exist: {predictions_dir}")
        else:
            print("No historical draws available for evaluation")
        
        print("Prediction generator completed successfully!")
        
    except Exception as e:
        print(f"ERROR in main function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()