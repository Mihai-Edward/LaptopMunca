
import os
import sys
import argparse
from datetime import datetime
from collections import Counter

# Add parent directory to path to import from data_analysis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_analysis import DataAnalysis
from config.paths import PATHS, ensure_directories

class PredictionGenerator:
    """Generate predictions based on historical data analysis"""
    
    def __init__(self, data_analysis=None):
        """Initialize with optional DataAnalysis object"""
        self.analysis = data_analysis if data_analysis else DataAnalysis()
    
    def generate_prediction(self, predict_count=15):
        """
        Generate enhanced prediction using multiple analysis methods
        with tiered confidence levels and pair analysis
        
        Args:
            predict_count (int): Number of numbers to predict (default 15)
            
        Returns:
            dict: Prediction results with confidence tiers
        """
        # Initialize base scores
        base_scores = {num: 0 for num in range(1, 81)}
        
        # Component 1: Frequency Analysis (25%)
        frequency = self.analysis.count_frequency()
        max_freq = max(frequency.values()) if frequency else 1
        for num, freq in frequency.items():
            base_scores[num] += (freq / max_freq) * 0.25
        
        # Component 2: Hot/Cold Analysis (25%)
        hot_cold = self.analysis.hot_and_cold_numbers(window_size=30)
        # Get trending numbers with their scores
        trending_up = hot_cold['trending_up']
        trending_numbers = {num: data['trend'] for num, data in trending_up}
        max_trend = max(trending_numbers.values()) if trending_numbers else 1
        for num, trend in trending_numbers.items():
            # Normalize and add to score
            base_scores[num] += (trend / max_trend) * 0.25
        
        # Component 3: Gap Analysis (25%)
        gaps = self.analysis.analyze_gaps()
        for num, gap_data in gaps.items():
            current_gap = gap_data['current_gap']
            avg_gap = gap_data['avg_gap']
            # Give higher scores to numbers more overdue than their average
            if avg_gap > 0:
                gap_ratio = current_gap / avg_gap
                # Cap the gap score to avoid extreme values
                gap_score = min(1.0, gap_ratio) * 0.25
                base_scores[num] += gap_score
        
        # Get initial candidate numbers based on base scores
        candidates = sorted(base_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_numbers = [num for num, _ in candidates[:predict_count*2]]  # Get more candidates than needed
        
        # Component 4: Common Pairs Analysis (25%)
        common_pairs = self.analysis.find_common_pairs(top_n=50)
        pairs = {pair: freq for pair, freq in common_pairs}
        
        # Create final scores with pair boosts
        final_scores = base_scores.copy()
        
        # Boost scores for numbers that appear in common pairs with high-scoring candidates
        for i, num1 in enumerate(candidate_numbers):
            for num2 in candidate_numbers[i+1:]:
                pair = tuple(sorted((num1, num2)))
                if pair in pairs:
                    pair_strength = pairs[pair] / max(pairs.values()) if pairs else 0
                    # Boost both numbers in the pair
                    final_scores[num1] += 0.025 * pair_strength
                    final_scores[num2] += 0.025 * pair_strength
        
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
        
        # Add common pair information
        common_pairs_in_prediction = []
        for i, num1 in enumerate(final_numbers):
            for num2 in final_numbers[i+1:]:
                pair = tuple(sorted((num1, num2)))
                if pair in pairs:
                    common_pairs_in_prediction.append({
                        'pair': pair,
                        'frequency': pairs[pair]
                    })
        
        prediction['common_pairs'] = sorted(
            common_pairs_in_prediction, 
            key=lambda x: x['frequency'], 
            reverse=True
        )[:10]  # Top 10 common pairs in our prediction
        
        return prediction

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
        
        print("\n" + "="*20 + " COMMON PAIRS DETECTED " + "="*20)
        for i, pair_info in enumerate(prediction['common_pairs'][:5]):
            print(f"    {pair_info['pair'][0]} and {pair_info['pair'][1]} (Appears together {pair_info['frequency']} times)")
        
        print("\n" + "="*20 + " ALL PREDICTED NUMBERS " + "="*20)
        formatted_numbers = ", ".join([str(num).rjust(2) for num in prediction['numbers']])
        print(f"    {formatted_numbers}")
        print("\n" + "="*50)

    def evaluate_prediction(self, prediction, actual_draw):
        """Evaluate the accuracy of a prediction against actual draw numbers"""
        prediction_set = set(prediction['numbers'])
        actual_set = set(actual_draw)
        correct_numbers = prediction_set.intersection(actual_set)
        
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
        print(f"    {correct_str}")
        
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
    parser = argparse.ArgumentParser(description='Generate Keno number predictions')
    parser.add_argument('--save', action='store_true', help='Save prediction to file')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate against most recent draw')
    args = parser.parse_args()
    
    # Initialize prediction generator
    predictor = PredictionGenerator()
    
    # Generate prediction
    prediction = predictor.generate_prediction()
    
    # Display prediction
    predictor.display_prediction(prediction)
    
    # Save prediction if requested
    if args.save:
        predictor.save_prediction(prediction)
    
    # Evaluate prediction if requested
    if args.evaluate:
        latest_draw = predictor.get_latest_draw()
        if latest_draw:
            evaluation = predictor.evaluate_prediction(prediction, latest_draw)
            predictor.display_evaluation(evaluation)
        else:
            print("No historical draws available for evaluation")


if __name__ == "__main__":
    main()