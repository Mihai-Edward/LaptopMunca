import csv
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from joblib import dump, load

# Import your existing DataAnalysis class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_analysis import DataAnalysis
from config.paths import PATHS, ensure_directories

class PatternPredictionModel:
    """
    Pattern recognition model for predicting draw numbers based on historical patterns
    Works with the existing DataAnalysis class to leverage historical draw data
    """
    
    def __init__(self, data_analysis_instance, sequence_length=10):
        """
        Initialize the pattern prediction model
        
        Args:
            data_analysis_instance: Instance of DataAnalysis class with historical data
            sequence_length: Number of previous draws to consider for pattern recognition
        """
        # Initialize core attributes
        self.data_analysis = data_analysis_instance
        self.sequence_length = sequence_length
        self.models = {}
        self.feature_importances = {}
        self.prediction_history = []
        self.model_performance = {}
        
        # Use paths directly from PATHS with error checking
        if 'PREDICTIONS' not in PATHS or 'MODELS' not in PATHS:
            raise ValueError("PREDICTIONS and MODELS paths must be configured in PATHS")
            
        self.predictions_path = PATHS['PREDICTIONS']
        self.models_path = PATHS['MODELS']
        
        # Ensure directories exist
        ensure_directories()
        
        # Verify directories are accessible
        if not os.path.exists(self.predictions_path) or not os.path.exists(self.models_path):
            raise RuntimeError(f"Required directories could not be created or accessed")
        
    def _extract_pattern_features(self, target_position=None):
        """
        Extract features from historical draws for pattern recognition
        
        Args:
            target_position: Position in self.data_analysis.draws to use as target
                            (if None, features are extracted for prediction)
        
        Returns:
            Dictionary of features for model input
        """
        draws = self.data_analysis.draws
        
        if target_position is None:
            # Use the latest draws for prediction
            sequence = draws[-self.sequence_length:]
        else:
            # Use sequence before target_position for training
            start_pos = max(0, target_position - self.sequence_length)
            sequence = draws[start_pos:target_position]
        
        # Basic sanity check
        if len(sequence) < 2:
            raise ValueError(f"Not enough draws in sequence (got {len(sequence)}, need at least 2)")
            
        # Get results from DataAnalysis for the sequence
        # We create a temporary DataAnalysis instance just for the sequence
        sequence_analysis = DataAnalysis(sequence, debug=False)
        
        # -- FEATURE EXTRACTION --
        features = {}
        
        # 1. Basic frequency features
        frequency = sequence_analysis.count_frequency()
        for num in range(1, 81):
            features[f'freq_{num}'] = frequency.get(num, 0) / len(sequence)
        
        # 2. Hot and cold numbers features
        hot_cold = sequence_analysis.hot_and_cold_numbers(top_n=20)
        hot_numbers = [num for num, _ in hot_cold['hot_numbers']]
        cold_numbers = [num for num, _ in hot_cold['cold_numbers']]
        trending_up = [num for num, _ in hot_cold['trending_up']]
        trending_down = [num for num, _ in hot_cold['trending_down']]
        
        for num in range(1, 81):
            features[f'is_hot_{num}'] = 1 if num in hot_numbers else 0
            features[f'is_cold_{num}'] = 1 if num in cold_numbers else 0
            features[f'trending_up_{num}'] = 1 if num in trending_up else 0
            features[f'trending_down_{num}'] = 1 if num in trending_down else 0
        
        # 3. Gap features
        gap_analysis = sequence_analysis.analyze_gaps()
        for num in range(1, 81):
            if num in gap_analysis:
                features[f'current_gap_{num}'] = gap_analysis[num]['current_gap']
                features[f'avg_gap_{num}'] = gap_analysis[num]['avg_gap']
                features[f'gap_ratio_{num}'] = (gap_analysis[num]['current_gap'] / 
                                              max(1, gap_analysis[num]['avg_gap']))
            else:
                features[f'current_gap_{num}'] = len(sequence)
                features[f'avg_gap_{num}'] = len(sequence)
                features[f'gap_ratio_{num}'] = 1.0
        
        # 4. Recent appearance patterns
        last_draw = sequence[-1][1] if sequence else []
        second_last_draw = sequence[-2][1] if len(sequence) >= 2 else []
        
        for num in range(1, 81):
            features[f'in_last_draw_{num}'] = 1 if num in last_draw else 0
            features[f'in_second_last_{num}'] = 1 if num in second_last_draw else 0
            
            # Check for pattern of "skipping" - appeared, disappeared, might appear again
            if len(sequence) >= 3:
                third_last_draw = sequence[-3][1]
                features[f'skip_pattern_{num}'] = 1 if (num in third_last_draw and 
                                                     num not in second_last_draw) else 0
        
        # 5. Consecutive appearance ratios
        consecutive_pairs = sequence_analysis.find_consecutive_numbers(top_n=40)
        consecutive_pairs_dict = {pair: freq for pair, freq in consecutive_pairs}
        
        for num in range(1, 80):  # 1 to 79, as 80 has no consecutive number
            pair = (num, num+1)
            features[f'consecutive_{num}'] = consecutive_pairs_dict.get(pair, 0) / len(sequence)
        
        # 6. Common pairs features
        common_pairs = sequence_analysis.find_common_pairs(top_n=100)
        common_pairs_dict = {pair: freq for pair, freq in common_pairs}
        
        # For each number, calculate its "connection strength" with other numbers
        for num in range(1, 81):
            connection_strength = 0
            for pair, freq in common_pairs:
                if num in pair:
                    connection_strength += freq
            features[f'connection_strength_{num}'] = connection_strength / len(sequence)
        
        # 7. Range balance features
        range_analysis = sequence_analysis.number_range_analysis()
        for range_key, count in range_analysis.items():
            features[f'range_{range_key}'] = count / (len(sequence) * 20)  # Normalize by total numbers
        
        # 8. Number of times a number appeared in the sequence
        for num in range(1, 81):
            appearances = sum(1 for _, numbers in sequence if num in numbers)
            features[f'appearances_{num}'] = appearances / len(sequence)
        
        return features
    
    def prepare_training_data(self):
        """
        Prepare features and targets for model training
        
        Returns:
            X_train, X_val, y_train, y_val: Training and validation datasets
        """
        draws = self.data_analysis.draws
        
        # Minimum required draws for meaningful training
        min_required = self.sequence_length + 50
        if len(draws) < min_required:
            raise ValueError(f"Need at least {min_required} draws for training (got {len(draws)})")
        
        features_list = []
        targets = []
        
        # Create training examples from historical data
        # Start from sequence_length to have enough history
        for i in range(self.sequence_length, len(draws)):
            try:
                # Extract features from draws before position i
                features = self._extract_pattern_features(target_position=i)
                
                # Target is which numbers appeared in draw at position i
                target_draw = draws[i][1]
                target = [1 if num in target_draw else 0 for num in range(1, 81)]
                
                features_list.append(features)
                targets.append(target)
            except Exception as e:
                print(f"Error processing draw {i}: {e}")
                continue
        
        if not features_list:
            raise ValueError("No valid training examples could be created")
            
        # Convert to DataFrame for easier handling
        X = pd.DataFrame(features_list)
        y = np.array(targets)
        
        # Split into training and validation sets (chronologically)
        train_size = int(len(X) * 0.8)
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:]
        y_val = y[train_size:]
        
        print(f"Prepared training data: {X_train.shape[0]} training examples, {X_val.shape[0]} validation examples")
        return X_train, X_val, y_train, y_val
    
    def train_ensemble_model(self, n_estimators=100, learning_rate=0.05, max_depth=4):
        """
        Train an ensemble of XGBoost models for predicting number probabilities

        Args:
            n_estimators: Number of estimators for XGBoost (default: 100)
            learning_rate: Learning rate for XGBoost (default: 0.05) 
            max_depth: Maximum tree depth for XGBoost (default: 4)

        Returns:
            Dictionary of trained models, one for each number
        """
        try:
            print("Preparing training data...")
            X_train, X_val, y_train, y_val = self.prepare_training_data()
            
            # Initialize containers
            models = {}
            performances = {}
            importances = {}
            
            # Progress tracking
            total_models = 80
            successful_models = 0
            failed_models = 0
            
            print(f"Training {total_models} models (one per number)...")
            
            # Configure XGBoost parameters
            params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'use_label_encoder': False,
                'eval_metric': 'logloss',
                'tree_method': 'hist',  # Faster training method
                'callbacks': [
                    xgb.callback.EarlyStopping(
                        rounds=10,
                        save_best=True,
                        metric='logloss'
                    )
                ]
            }
            
            # Train models for each number
            for num in range(80):
                actual_num = num + 1
                try:
                    print(f"\nTraining model for number {actual_num} ({num + 1}/{total_models})...")
                    
                    # Initialize and train model
                    model = xgb.XGBClassifier(**params)
                    
                    # Get class weights to handle imbalanced data
                    pos_weight = np.sum(y_train[:, num] == 0) / np.sum(y_train[:, num] == 1)
                    model.set_params(scale_pos_weight=pos_weight)
                    
                    # Train with validation set
                    model.fit(
                        X_train, y_train[:, num],
                        eval_set=[(X_val, y_val[:, num])],
                        verbose=False
                    )
                    
                    # Evaluate performance
                    train_preds = model.predict(X_train)
                    val_preds = model.predict(X_val)
                    
                    train_acc = np.mean(train_preds == y_train[:, num])
                    val_acc = np.mean(val_preds == y_val[:, num])
                    
                    # Record performance metrics
                    performances[actual_num] = {
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'val_positive_rate': np.mean(val_preds),
                        'actual_positive_rate': np.mean(y_val[:, num])
                    }
                    
                    print(f"  Model {actual_num}: Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
                    
                    # Save model and feature importance
                    models[actual_num] = model
                    importances[actual_num] = {
                        'importance': model.feature_importances_,
                        'features': X_train.columns.tolist()
                    }
                    
                    successful_models += 1
                except Exception as e:
                    print(f"Error training model for number {actual_num}: {e}")
                    failed_models += 1
            
            self.models['xgboost'] = models
            self.model_performance['xgboost'] = performances
            self.feature_importances['xgboost'] = importances
            
            # Calculate overall model performance
            avg_val_acc = np.mean([perf['val_accuracy'] for perf in performances.values()])
            print(f"\nTraining complete. Average validation accuracy: {avg_val_acc:.4f}")
            print(f"Successfully trained {successful_models} models, failed to train {failed_models} models.")
            
            # Save the models
            self.save_models()
            
            return models
        except Exception as e:
            print(f"Error during training: {e}")
            return None
    
    def predict_next_draw(self, num_predictions=15):
        """
        Predict the next draw using the trained models
        
        Args:
            num_predictions: Number of numbers to predict (default 15)
            
        Returns:
            List of predicted numbers and confidence scores
        """
        if 'xgboost' not in self.models or not self.models['xgboost']:
            raise ValueError("Models not trained yet. Call train_ensemble_model first.")
        
        # Extract features for prediction
        features = self._extract_pattern_features()
        features_df = pd.DataFrame([features])
        
        # Get predictions from each model
        probabilities = {}
        
        for num, model in self.models['xgboost'].items():
            # Get probability of the number appearing
            prob = model.predict_proba(features_df)[0][1]
            probabilities[num] = prob
        
        # Apply additional weights based on analysis insights
        self._apply_prediction_weights(probabilities)
        
        # Sort by probability and get top predictions
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        predicted_numbers = [num for num, _ in sorted_probs[:num_predictions]]
        confidence_scores = [prob for _, prob in sorted_probs[:num_predictions]]
        
        # Create prediction record
        prediction = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'predicted_numbers': predicted_numbers,
            'confidence_scores': [round(score, 4) for score in confidence_scores],
            'full_probabilities': {k: round(v, 4) for k, v in sorted_probs}
        }
        
        # Save prediction to history
        self.prediction_history.append(prediction)
        self._save_prediction(prediction)
        
        return predicted_numbers, confidence_scores
    
    def _apply_prediction_weights(self, probabilities):
        """
        Apply additional weighting to probabilities based on analysis insights
        
        Args:
            probabilities: Dictionary of number->probability mappings to adjust
        """
        # Get insights from data analysis
        gap_analysis = self.data_analysis.analyze_gaps()
        hot_cold = self.data_analysis.hot_and_cold_numbers(top_n=20)
        
        # Numbers that haven't appeared for longer than their average gap
        # have increased probability of appearing (due for appearance)
        for num in range(1, 81):
            if num in gap_analysis:
                gap_stats = gap_analysis[num]
                current_gap = gap_stats['current_gap']
                avg_gap = gap_stats['avg_gap']
                
                # If number is "due" (current gap > average gap)
                if current_gap > avg_gap * 1.2:  # 20% above average
                    # Boost probability by up to 20% based on how overdue it is
                    boost = min(0.2, (current_gap - avg_gap) / (avg_gap * 5))
                    probabilities[num] = probabilities[num] * (1 + boost)
        
               # Apply weights based on hot/cold analysis
        hot_numbers = [num for num, _ in hot_cold['hot_numbers']]
        trending_up = [num for num, _ in hot_cold['trending_up']]
        
        # Boost hot numbers slightly
        for num in hot_numbers:
            probabilities[num] = probabilities[num] * 1.05  
                    # Boost trending up numbers
        for num in trending_up:
            probabilities[num] = probabilities[num] * 1.08
            
        # Analyze skip patterns from recent draws
        try:
            skip_analysis = self.data_analysis.analyze_skip_patterns(window_size=5)
            recent_patterns = skip_analysis.get('recent_patterns', [])
            
            # Look for numbers that frequently return after being skipped
            if recent_patterns:
                skipped_last_draw = set()
                for pattern in recent_patterns[-2:-1]:  # Get second-to-last pattern
                    skipped_last_draw.update(pattern.get('skipped_numbers', []))
                    
                # Boost numbers that were recently skipped
                for num in skipped_last_draw:
                    probabilities[num] = probabilities[num] * 1.03
        except Exception as e:
            print(f"Error applying skip pattern weights: {e}")
            
        # Normalize probabilities to ensure they sum to a reasonable value
        # This prevents extreme values from dominating
        total_prob = sum(probabilities.values())
        if total_prob > 0:
            avg_prob = total_prob / 80
            for num in probabilities:
                # Cap extreme probabilities at 3x average
                if probabilities[num] > 3 * avg_prob:
                    probabilities[num] = 3 * avg_prob
                    
        return probabilities
    
    def evaluate_prediction(self, prediction, actual_draw):
        """
        Evaluate a prediction against actual draw results
        
        Args:
            prediction: List of predicted numbers
            actual_draw: List of actual drawn numbers
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not prediction or not actual_draw:
            return {"error": "Empty prediction or actual draw"}
            
        # Calculate hit rate
        hits = set(prediction).intersection(set(actual_draw))
        hit_count = len(hits)
        hit_rate = hit_count / len(prediction) if prediction else 0
        
        # Calculate expected hit rate (random)
        expected_hit_rate = len(prediction) * (len(actual_draw) / 80)
        
        # Calculate lift over random
        lift = hit_rate / expected_hit_rate if expected_hit_rate > 0 else 0
        
        evaluation = {
            'hit_count': hit_count,
            'hit_rate': hit_rate,
            'expected_hit_rate': expected_hit_rate,
            'lift': lift,
            'hit_numbers': list(hits),
            'missed_numbers': list(set(actual_draw) - hits),
            'false_positives': list(set(prediction) - set(actual_draw)),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save evaluation
        self._save_evaluation(evaluation)
        
        return evaluation
    
    def analyze_prediction_history(self):
        """
        Analyze historical prediction performance
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.prediction_history:
            return {"error": "No prediction history available"}
            
        # Calculate average hit rate if we have actual results
        evaluations = []
        eval_file = os.path.join(self.predictions_path, "prediction_evaluations.pkl")
        
        if os.path.exists(eval_file):
            try:
                with open(eval_file, 'rb') as f:
                    evaluations = pickle.load(f)
            except Exception as e:
                print(f"Error loading evaluations: {e}")
                
        if not evaluations:
            return {"warning": "No evaluations available yet"}
            
        # Calculate metrics
        hit_rates = [e['hit_rate'] for e in evaluations]
        lifts = [e['lift'] for e in evaluations]
        
        # Calculate average performance
        avg_hit_rate = sum(hit_rates) / len(hit_rates)
        avg_lift = sum(lifts) / len(lifts)
        
        # Find most successful numbers (appeared most often in hits)
        all_hits = [num for e in evaluations for num in e.get('hit_numbers', [])]
        hit_counter = Counter(all_hits)
        most_successful = hit_counter.most_common(10)
        
        # Calculate success trend
        if len(hit_rates) >= 5:
            recent_hit_rate = sum(hit_rates[-5:]) / 5
            recent_lift = sum(lifts[-5:]) / 5
            trend = "Improving" if recent_hit_rate > avg_hit_rate else "Declining" 
        else:
            recent_hit_rate = avg_hit_rate
            recent_lift = avg_lift
            trend = "Insufficient data"
            
        performance = {
            'average_hit_rate': avg_hit_rate,
            'average_lift': avg_lift,
            'recent_hit_rate': recent_hit_rate,
            'recent_lift': recent_lift,
            'trend': trend,
            'most_successful_numbers': most_successful,
            'total_evaluations': len(evaluations),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return performance
    
    def _save_prediction(self, prediction):
        """Save a prediction to disk"""
        try:
            # Create a unique filename for this prediction
            timestamp = prediction['timestamp'].replace(':', '-').replace(' ', '_')
            filename = os.path.join(self.predictions_path, f"prediction_{timestamp}.pkl")
            
            with open(filename, 'wb') as f:
                pickle.dump(prediction, f)
                
            # Also update the history file
            history_file = os.path.join(self.predictions_path, "prediction_history.pkl")
            with open(history_file, 'wb') as f:
                pickle.dump(self.prediction_history[-100:], f)  # Keep last 100 predictions
                
            print(f"Prediction saved to {filename}")
            return True
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False
    
    def _save_evaluation(self, evaluation):
        """Save an evaluation to disk"""
        try:
            evaluations = []
            eval_file = os.path.join(self.predictions_path, "prediction_evaluations.pkl")
            
            # Load existing evaluations if available
            if os.path.exists(eval_file):
                try:
                    with open(eval_file, 'rb') as f:
                        evaluations = pickle.load(f)
                except Exception:
                    evaluations = []
            
            # Add new evaluation
            evaluations.append(evaluation)
            
            # Save updated evaluations
            with open(eval_file, 'wb') as f:
                pickle.dump(evaluations, f)
                
            print(f"Evaluation saved to {eval_file}")
            return True
        except Exception as e:
            print(f"Error saving evaluation: {e}")
            return False
    
    def save_models(self):
        """Save all models to disk"""
        try:
            if 'xgboost' not in self.models:
                print("No XGBoost models to save")
                return False
                
            # Ensure models directory exists
            os.makedirs(self.models_path, exist_ok=True)
                
            # Save each number's model separately
            for num, model in self.models['xgboost'].items():
                model_file = os.path.join(self.models_path, f"xgb_model_{num}.joblib")
                try:
                    dump(model, model_file)
                except Exception as e:
                    print(f"Error saving model {num}: {e}")
                    continue
            
            # Prepare metadata
            metadata = {
                'feature_importances': self.feature_importances,
                'model_performance': self.model_performance,
                'sequence_length': self.sequence_length,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save metadata
            metadata_file = os.path.join(self.models_path, "model_metadata.pkl")
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
            print(f"Successfully saved models and metadata to {self.models_path}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load models from disk"""
        try:
            models = {}
            metadata_file = os.path.join(self.models_path, "model_metadata.pkl")
            
            # Check if models exist
            if not os.path.exists(metadata_file):
                print("No saved models found.")
                return False
                
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                
            self.feature_importances = metadata.get('feature_importances', {})
            self.model_performance = metadata.get('model_performance', {})
            self.sequence_length = metadata.get('sequence_length', 10)
            
            # Load each model
            for num in range(1, 81):
                model_file = os.path.join(self.models_path, f"xgb_model_{num}.joblib")
                if os.path.exists(model_file):
                    models[num] = load(model_file)
            
            self.models['xgboost'] = models
            print(f"Loaded {len(models)} models from {self.models_path}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def visualize_predictions(self, prediction, confidence_scores):
        """
        Generate a visualization of the prediction and confidence scores
        
        Args:
            prediction: List of predicted numbers
            confidence_scores: List of confidence scores
            
        Returns:
            Figure object
        """
        try:
            # Create a figure with number grid and confidence visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Create a grid of all numbers
            grid_data = np.zeros((8, 10))
            
            # Mark predicted numbers
            for num in prediction:
                row = (num - 1) // 10
                col = (num - 1) % 10
                grid_data[row, col] = 1
            
            # Create heatmap for all numbers
            ax1.imshow(grid_data, cmap='coolwarm', vmin=0, vmax=1)
            
            # Add number labels
            for i in range(8):
                for j in range(10):
                    num = i * 10 + j + 1
                    color = 'white' if num in prediction else 'black'
                    ax1.text(j, i, str(num), ha='center', va='center', color=color)
                    
            ax1.set_title('Predicted Numbers')
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            # Create bar chart of confidence scores
            ax2.bar(prediction, confidence_scores)
            ax2.set_xlabel('Number')
            ax2.set_ylabel('Confidence Score')
            ax2.set_title('Prediction Confidence')
            
            plt.tight_layout()
            
            # Save the figure
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fig_path = os.path.join(self.predictions_path, f"prediction_viz_{timestamp}.png")
            plt.savefig(fig_path)
            
            return fig
        except Exception as e:
            print(f"Error creating visualization: {e}")
            return None
    
    def get_top_pattern_features(self, top_n=10):
        """
        Get the most important features for pattern prediction
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of top features for each number
        """
        if not self.feature_importances:
            return {"error": "No feature importance data available"}
            
        top_features = {}
        
        for num, importance_data in self.feature_importances.get('xgboost', {}).items():
            importance = importance_data.get('importance')
            features = importance_data.get('features')
            
            if importance is not None and features is not None:
                # Sort features by importance
                sorted_idx = np.argsort(importance)[::-1]
                
                # Get top features
                top = [
                    {
                        'feature': features[idx],
                        'importance': float(importance[idx])
                    }
                    for idx in sorted_idx[:top_n]
                ]
                
                top_features[num] = top
                
        return top_features
    
    def export_training_data(self, file_path=None):
        """
        Export the training data to a CSV file
        
        Args:
            file_path: Path to save the CSV file
            
        Returns:
            Path to the saved file
        """
        try:
            X_train, X_val, y_train, y_val = self.prepare_training_data()
            
            # Combine features and targets
            train_df = pd.DataFrame(X_train)
            
            # Add target columns
            for i in range(80):
                num = i + 1
                train_df[f'target_{num}'] = y_train[:, i]
                
            # Save to CSV
            if file_path is None:
                file_path = os.path.join(self.predictions_path, f"training_data_{datetime.now().strftime('%Y%m%d')}.csv")
                
            train_df.to_csv(file_path, index=False)
            print(f"Training data exported to {file_path}")
            
            return file_path
        except Exception as e:
            print(f"Error exporting training data: {e}")
            return None


# Example usage of the pattern prediction model
if __name__ == "__main__":
    try:
        # Ensure all directories exist first
        ensure_directories()
        
        # Get historical data path with proper error handling
        historical_data_path = PATHS.get('HISTORICAL_DATA')
        if not historical_data_path:
            print("Error: HISTORICAL_DATA path not configured in config/paths.py")
            sys.exit(1)
            
        # Convert to absolute path if relative
        if not os.path.isabs(historical_data_path):
            base_dir = PATHS.get('BASE_DIR', os.path.dirname(os.path.dirname(__file__)))
            historical_data_path = os.path.join(base_dir, historical_data_path)
            
        # Check if file exists
        if not os.path.exists(historical_data_path):
            print(f"Error: Historical data file not found at: {historical_data_path}")
            sys.exit(1)
            
        # Load and parse CSV data
        draws = []
        try:
            with open(historical_data_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader, None)  # Skip header
                
                for row_num, row in enumerate(csv_reader, start=1):
                    if len(row) >= 21:  # Date/time + 20 numbers
                        draw_date = row[0]
                        try:
                            numbers = [int(num.strip()) for num in row[1:21] if num.strip()]
                            if len(numbers) == 20:  # Ensure exactly 20 numbers
                                draws.append((draw_date, numbers))
                            else:
                                print(f"Warning: Row {row_num} has {len(numbers)} numbers instead of 20")
                        except ValueError as e:
                            print(f"Warning: Skipping row {row_num}, invalid number format: {e}")
                            continue
                    else:
                        print(f"Warning: Skipping row {row_num}, insufficient columns")

        except Exception as e:
            print(f"Error reading historical data file: {e}")
            sys.exit(1)
            
        if not draws:
            print("Error: No valid draws found in historical data")
            sys.exit(1)
            
        print(f"Successfully loaded {len(draws)} historical draws")
        
        # Create DataAnalysis instance
        data_analysis = DataAnalysis(draws)
        
        # Create and train the pattern prediction model
        model = PatternPredictionModel(data_analysis, sequence_length=15)
        
        # Try to load existing models first
        if not model.load_models():
            print("No existing models found. Training new models...")
            model.train_ensemble_model(n_estimators=100, learning_rate=0.05, max_depth=3)
        
        # Generate prediction for next draw
        prediction, confidence = model.predict_next_draw(num_predictions=15)
        
        print("\n==== PATTERN-BASED PREDICTION ====")
        print(f"Top 15 predicted numbers:")
        for num, conf in zip(prediction, confidence):
            print(f"Number {num}: {conf:.4f} confidence")
            
        # Visualize the prediction
        model.visualize_predictions(prediction, confidence)
        plt.show()
        
  # Analyze prediction history if available
        performance = model.analyze_prediction_history()
        if 'error' not in performance and 'warning' not in performance:
            print("\n==== PREDICTION PERFORMANCE ====")
            print(f"Average hit rate: {performance['average_hit_rate']:.4f}")
            print(f"Average lift over random: {performance['average_lift']:.4f}x")
            print(f"Recent performance trend: {performance['trend']}")
            print("Most successful numbers:")
            for num, count in performance['most_successful_numbers']:
                print(f"  Number {num}: hit {count} times")
        
        # Compare with traditional frequency-based approach
        print("\n==== COMPARISON WITH FREQUENCY ANALYSIS ====")
        frequency_pred = data_analysis.get_top_numbers(15)
        print(f"Traditional frequency-based prediction: {frequency_pred}")
        print(f"Pattern-based prediction: {prediction}")
        
        # Calculate overlap between methods
        overlap = set(prediction).intersection(set(frequency_pred))
        print(f"Overlap between methods: {len(overlap)} numbers")
        print(f"Overlap percentage: {len(overlap)/15*100:.1f}%")
        print(f"Overlapping numbers: {sorted(list(overlap))}")
        
        # Show most important features for pattern recognition
        top_features = model.get_top_pattern_features(top_n=5)
        print("\n==== TOP PATTERN FEATURES ====")
        print("Sample of important features for number prediction:")
        
        # Show features for a few example numbers
        example_numbers = prediction[:3]  # First 3 predicted numbers
        for num in example_numbers:
            if num in top_features:
                print(f"\nFeatures for number {num}:")
                for feature in top_features[num]:
                    print(f"  {feature['feature']}: {feature['importance']:.4f}")
        
        print("\n==== NEXT STEPS ====")
        print("1. To evaluate prediction: After the draw occurs, run:")
        print("   actual_draw = [list of actual numbers]")
        print("   evaluation = model.evaluate_prediction(prediction, actual_draw)")
        print("   print(evaluation)")
        print("2. To update model with new draws: Run this script regularly")
        print("3. To export training data for further analysis:")
        print("   model.export_training_data()")
        
        # Current prediction timestamp
        print(f"\nPrediction generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"Error in pattern prediction: {e}")
        import traceback
        traceback.print_exc()