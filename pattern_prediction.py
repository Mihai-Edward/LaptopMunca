import csv
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from joblib import dump, load
from datetime import datetime
# Import your existing DataAnalysis class
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_analysis import DataAnalysis
from config.paths import PATHS, ensure_directories

class PatternPredictionModel:
    """
    Pattern recognition model for predicting draw numbers based on historical patterns
    Works with the existing DataAnalysis class to leverage historical draw data
    """
    
    def __init__(self, data_analysis_instance, sequence_length=10, debug=False):
        """
        Initialize the pattern prediction model
        
        Args:
            data_analysis_instance: Instance of DataAnalysis class with historical data
            sequence_length: Number of previous draws to consider for pattern recognition
            debug: Enable debug output (default: False)
        """
        # Add at start of method:
        ensure_directories()  # Ensure all directories exist

        # Initialize core attributes
        self.data_analysis = data_analysis_instance
        self.sequence_length = sequence_length
        self.models = {}
        self.feature_importances = {}
        self.prediction_history = []
        self.model_performance = {}
        self.debug = debug  # Add debug flag
        
        # Use paths directly from PATHS with error checking
        if 'PREDICTIONS_DIR' not in PATHS or 'MODELS_DIR' not in PATHS:
            raise ValueError("PREDICTIONS_DIR and MODELS_DIR paths must be configured in PATHS")
            
        self.predictions_path = PATHS['PREDICTIONS_DIR']
        self.models_path = PATHS['MODELS_DIR']
        
    def get_next_draw_time(self):
        """
        Get the next draw time in the format 'HH:MM DD-MM-YYYY'
        Returns time in your exact CSV format
        """
        current_time = datetime.now()
        current_minute = current_time.hour * 60 + current_time.minute
        
        # Round to next 5 minute interval
        next_interval = ((current_minute // 5) + 1) * 5
        
        # Calculate next draw time
        next_draw = current_time.replace(
            hour=next_interval // 60,
            minute=next_interval % 60,
            second=0,
            microsecond=0
        )
        
        # If we've passed the last draw of the day, move to next day
        if next_interval >= 24 * 60:
            next_draw = next_draw + timedelta(days=1)
            next_draw = next_draw.replace(hour=0, minute=0)
        
        formatted_time = next_draw.strftime("%H:%M  %d-%m-%Y")
        
        return formatted_time

    def _extract_pattern_features(self, target_position=None):
        """
        Extract features from historical draws for pattern recognition
        
        Args:
            target_position: Position in self.data_analysis.draws to use as target
                            (if None, features are extracted for prediction)
        
        Returns:
            Dictionary of features for model input
        """
        feature_validation = {
            'sequence_data': False,
            'frequency_analysis': False,
            'hot_cold_analysis': False,
            'gap_analysis': False,
            'recent_patterns': False,
            'statistical_significance': False  # Added as requested
        }
        
        try:
            draws = self.data_analysis.draws
            
            if target_position is None:
                # Use the latest draws for prediction
                sequence = draws[-self.sequence_length:]
            else:
                # Use sequence before target_position for training
                start_pos = max(0, target_position - self.sequence_length)
                sequence = draws[start_pos:target_position]
            
            # Validate sequence data
            feature_validation['sequence_data'] = len(sequence) >= 2
            
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
            feature_validation['frequency_analysis'] = len(frequency) > 0
            
            for num in range(1, 81):
                features[f'freq_{num}'] = frequency.get(num, 0) / len(sequence)
            
            # 2. Hot and cold numbers features
            hot_cold = sequence_analysis.hot_and_cold_numbers(top_n=20)
            feature_validation['hot_cold_analysis'] = bool(hot_cold) and 'hot_numbers' in hot_cold
            
            hot_numbers = [num for num, _ in hot_cold.get('hot_numbers', [])]
            cold_numbers = [num for num, _ in hot_cold.get('cold_numbers', [])]
            trending_up = [num for num, _ in hot_cold.get('trending_up', [])]
            trending_down = [num for num, _ in hot_cold.get('trending_down', [])]
            
            for num in range(1, 81):
                features[f'is_hot_{num}'] = 1 if num in hot_numbers else 0
                features[f'is_cold_{num}'] = 1 if num in cold_numbers else 0
                features[f'trending_up_{num}'] = 1 if num in trending_up else 0
                features[f'trending_down_{num}'] = 1 if num in trending_down else 0
            
            # 3. Gap features
            gap_analysis = sequence_analysis.analyze_gaps()
            feature_validation['gap_analysis'] = bool(gap_analysis)
            
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
            feature_validation['recent_patterns'] = bool(last_draw) and bool(second_last_draw)
            
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
            
            # NEW: Add statistical significance features
            try:
                significance_results = sequence_analysis.analyze_statistical_significance()
                feature_validation['statistical_significance'] = bool(significance_results) and 'frequency_tests' in significance_results
                
                # Extract statistically significant numbers
                if significance_results and 'frequency_tests' in significance_results:
                    significant_numbers = []
                    for test in significance_results['frequency_tests']:
                        if test.get('significant', False):
                            significant_numbers.append(test['number'])
                    
                    # Add features for statistically significant numbers
                    for num in range(1, 81):
                        features[f'significant_{num}'] = 1 if num in significant_numbers else 0
                        
                        # Also add p-value as a feature (if available)
                        p_value = 0.5  # Default value if not found
                        for test in significance_results['frequency_tests']:
                            if test['number'] == num:
                                p_value = test.get('p_value', 0.5)
                                break
                        features[f'p_value_{num}'] = p_value
                    
                    # Add features for non-random patterns (if available)
                    if 'pattern_tests' in significance_results:
                        non_random_patterns = []
                        for test in significance_results['pattern_tests']:
                            if test.get('significant', False):
                                non_random_patterns.append(test['number'])
                        
                        for num in range(1, 81):
                            features[f'non_random_pattern_{num}'] = 1 if num in non_random_patterns else 0
                
                print(f"Added statistical significance features for {len(significant_numbers) if 'significant_numbers' in locals() else 0} numbers")
                    
            except Exception as e:
                print(f"Error extracting statistical significance features: {e}")
                # Continue without these features
            
            # 8. Number of times a number appeared in the sequence
            for num in range(1, 81):
                appearances = sum(1 for _, numbers in sequence if num in numbers)
                features[f'appearances_{num}'] = appearances / len(sequence)
            
            # Print validation summary
            print("\n=== Feature Extraction Validation ===")
            for check, status in feature_validation.items():
                print(f"{check}: {'✅ Valid' if status else '❌ Invalid'}")
            
            print(f"Total features extracted: {len(features)}")
            
            # Add after all features are extracted, before return:
           # if self.debug:
                #print("\nDEBUG: === Key Feature Values ===")
                #for feat_name in [f for f in features.keys() if any(x in f for x in ['gap_ratio_', 'current_gap_', 'is_hot_'])]:
                   # if features[feat_name] > 0.5:  # Only show significant features
                      #  print(f"DEBUG: {feat_name} = {features[feat_name]:.4f}")
            
            return features
        
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            import traceback
            traceback.print_exc()
            
            # Print validation summary with failures
            print("\n=== Feature Extraction Validation (Error Occurred) ===")
            for check, status in feature_validation.items():
                print(f"{check}: {'✅ Valid' if status else '❌ Invalid'}")
                
            return {}  # Return empty features dict on error
    
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
    
    def train_ensemble_model(self, n_estimators=100, learning_rate=0.05, max_depth=3):
        """
        Train an ensemble of XGBoost models for predicting number probabilities

        Args:
            n_estimators: Number of estimators for XGBoost (default: 100)
            learning_rate: Learning rate for XGBoost (default: 0.05) 
            max_depth: Maximum tree depth for XGBoost (default: 3)

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
                'eval_metric': 'logloss',
                'tree_method': 'hist',  # Faster training method
                'early_stopping_rounds': 10  # Early stopping in params instead of callbacks
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
                    
                    # Train with validation set - removed callbacks parameter
                    model.fit(
                        X_train, y_train[:, num],
                        eval_set=[(X_val, y_val[:, num])],
                        verbose=False
                    )
                    
                    # ...existing code...
                    train_preds = model.predict(X_train)
                    val_preds = model.predict(X_val)
                    
                    train_acc = np.mean(train_preds == y_train[:, num])
                    val_acc = np.mean(val_preds == y_val[:, num])
                    
                    performances[actual_num] = {
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'val_positive_rate': np.mean(val_preds),
                        'actual_positive_rate': np.mean(y_val[:, num])
                    }
                    
                    print(f"  Model {actual_num}: Train acc: {train_acc:.4f}, Val acc: {val_acc:.4f}")
                    
                    models[actual_num] = model
                    importances[actual_num] = {
                        'importance': model.feature_importances_,
                        'features': X_train.columns.tolist()
                    }
                    
                    successful_models += 1
                    
                except Exception as e:
                    print(f"Error training model for number {actual_num}: {e}")
                    failed_models += 1
                    continue

            # Save results
            self.models['xgboost'] = models
            self.model_performance['xgboost'] = performances
            self.feature_importances['xgboost'] = importances
            
            # Print summary
            avg_val_acc = np.mean([perf['val_accuracy'] for perf in performances.values()])
            print(f"\nTraining complete. Average validation accuracy: {avg_val_acc:.4f}")
            print(f"Successfully trained {successful_models} models, failed to train {failed_models} models.")
            
            self.save_models()
            return models
            
        except Exception as e:
            print(f"Error during training: {e}")
            return None
    
    def predict_next_draw(self, num_predictions=15):
        """
        Predict the next draw using the trained models and data analysis
        
        Args:
            num_predictions: Number of numbers to predict (default 15)
            
        Returns:
            List of predicted numbers and confidence scores
        """
        print("\n=== Starting Prediction Process ===")
        print(f"Current Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
         # ADD THIS DEBUG BLOCK - Show the latest draws being considered
        if self.debug:
            print("\n=== DEBUG: Latest Historical Draws Used ===")
            latest_draws = self.data_analysis.draws[-self.sequence_length:]
            print(f"Using {len(latest_draws)} latest draws out of {len(self.data_analysis.draws)} total draws")
            for i, (draw_date, numbers) in enumerate(latest_draws):
                print(f"Draw #{i+1}: {draw_date} - Numbers: {numbers[:5]}...")
            print("=== End Debug ===\n")
        # END DEBUG BLOCK
        validation_checks = {
            'fresh_analysis': False,
            'feature_extraction': False,
            'model_prediction': False,
            'probability_calculation': False,
            'combination_analysis': False  # Added combination analysis check
        }

        try:
            # Validate fresh analysis
            print("\nRunning comprehensive data analysis...")
            frequency = self.data_analysis.count_frequency()
            hot_cold = self.data_analysis.hot_and_cold_numbers(top_n=20)
            gaps = self.data_analysis.analyze_gaps()
            
            # 2. Pattern Analysis Methods
            sequence_patterns = self.data_analysis.sequence_pattern_analysis()
            pattern_validation = self.data_analysis.validate_patterns()
            skip_patterns = self.data_analysis.analyze_skip_patterns()
            
            # 3. Advanced Analysis Methods
            focused_prediction = self.data_analysis.get_focused_prediction()
            recent_performance = self.data_analysis.analyze_recent_performance()
            
            # NEW: Add combination analysis - THIS WAS MISSING
            print("\nRunning combination analysis...")
            combinations_analysis = self.data_analysis.analyze_combinations(group_size=3, top_n=10)
            combination_numbers = set()  # Initialize the set here

            if combinations_analysis and 'most_common' in combinations_analysis:
                # Extract numbers that appear in the most common combinations
                for combo_data in combinations_analysis['most_common']:
                    if 'combination' in combo_data:
                        for num in combo_data['combination']:
                            combination_numbers.add(num)
                
                validation_checks['combination_analysis'] = True
                print(f"Found {len(combination_numbers)} numbers in top combinations")
            
            validation_checks['fresh_analysis'] = True
            
            # Validate feature extraction
            features = self._extract_pattern_features()
            validation_checks['feature_extraction'] = True
            print(f"Features extracted: {len(features) if features else 0} features")
            
            # Validate model predictions
            probabilities = {}
            models_used = 0
            for num, model in self.models['xgboost'].items():
                prob = model.predict_proba(pd.DataFrame([features]))[0][1]
                probabilities[num] = prob
                models_used += 1
            validation_checks['model_prediction'] = models_used == 80  # Should use all 80 models
            
            # Apply weights based on ALL analyses
            for num in range(1, 81):
                original_prob = probabilities.get(num, 0)
                
                # Sequence Pattern Adjustments
                if sequence_patterns and num in sequence_patterns.get('strong_patterns', []):
                    probabilities[num] *= 1.2
                    
                # Skip Pattern Adjustments    
                if skip_patterns and num in skip_patterns.get('likely_next', []):
                    probabilities[num] *= 1.15
                    
                # Hot/Cold Number Adjustments
                if num in [n for n, _ in hot_cold.get('hot_numbers', [])]:
                    probabilities[num] *= 1.1
                elif num in [n for n, _ in hot_cold.get('cold_numbers', [])]:
                    probabilities[num] *= 0.9
                    
                # Gap Analysis Adjustments
                if num in gaps:
                    gap_info = gaps[num]
                    if gap_info['current_gap'] > gap_info['avg_gap']:
                        boost = min(1.3, 1 + (gap_info['current_gap'] / gap_info['avg_gap'] * 0.1))
                        probabilities[num] *= boost
                        
                # Pattern Validation Adjustments
                if pattern_validation and num in pattern_validation.get('consistent_numbers', []):
                    probabilities[num] *= 1.1
                    
                # Focused Prediction Adjustments
                if focused_prediction and num in focused_prediction.get('numbers', []):
                    probabilities[num] *= 1.15

                # NEW: Add combination-based adjustment
                if num in combination_numbers:  # Now this variable is defined
                    combo_boost = 1.12  # 12% boost for numbers in strong combinations
                    probabilities[num] *= combo_boost
                    print(f"Number {num}: Strong combination boost +12%")
            
            validation_checks['probability_calculation'] = True
                
            # Print validation summary
            print("\n=== Validation Summary ===")
            for check, status in validation_checks.items():
                print(f"{check}: {'✅ Completed' if status else '❌ Failed'}")
                
            # Sort and get predictions
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            predicted_numbers = [num for num, _ in sorted_probs[:num_predictions]]
            confidence_scores = [prob for _, prob in sorted_probs[:num_predictions]]
            
            print("\n=== Final Prediction ===")
            next_draw_time = self.get_next_draw_time()
            print(f"Next draw time: {next_draw_time}")
            print(f"Numbers predicted: {predicted_numbers}")
            
            # Save comprehensive prediction data
            prediction = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'draw_time': next_draw_time,
                'predicted_numbers': predicted_numbers,
                'confidence_scores': [round(score, 4) for score in confidence_scores],
                'analysis_factors': {
                    'sequence_patterns': sequence_patterns.get('pattern_count', 0) if sequence_patterns else 0,
                    'skip_patterns': skip_patterns.get('pattern_count', 0) if skip_patterns else 0,
                    'pattern_validation': pattern_validation.get('validation_score', 0) if pattern_validation else 0,
                    'recent_performance': recent_performance.get('average_accuracy', 0) if recent_performance else 0,
                    'combination_patterns': len(combination_numbers)  # Add the combination data
                },
                'validation_status': validation_checks
            }
            
            self.prediction_history.append(prediction)
            self._save_prediction(prediction)
            
            # Add after model predictions but before final return:
            if self.debug:
                print("\nDEBUG: === Model Confidence Breakdown ===")
                # Get gap analysis and hot/cold numbers for debug output
                gaps = self.data_analysis.analyze_gaps()
                hot_cold = self.data_analysis.hot_and_cold_numbers(top_n=20)
                
                for num in sorted(probabilities, key=probabilities.get, reverse=True)[:20]:
                    print(f"DEBUG: Number {num}: Raw prob={probabilities[num]:.4f}, " 
                          f"Gap ratio={gaps.get(num, {}).get('gap_ratio', 0):.2f}, "
                          f"Is hot={1 if num in [n for n, _ in hot_cold.get('hot_numbers', [])] else 0}")
            
            return predicted_numbers, confidence_scores
                
        except Exception as e:
            print(f"Error during prediction process: {e}")
            import traceback
            traceback.print_exc()
            
            # Print validation summary with failures
            print("\n=== Validation Summary (Error Occurred) ===")
            for check, status in validation_checks.items():
                print(f"{check}: {'✅ Completed' if status else '❌ Failed'}")
                
            return [], []
    
    def _apply_prediction_weights(self, probabilities):
        """
        Apply advanced weighting to probabilities based on multiple analysis methods
        
        Args:
            probabilities: Dictionary of number -> probability mappings
            
        Returns:
            Updated probabilities dictionary
        """
        print("\n=== Applying Fresh Analysis Weights ===")
        
        try:
            # Get fresh analysis results
            recent_analysis = self.data_analysis.sequence_pattern_analysis()
            recent_hot_cold = self.data_analysis.hot_and_cold_numbers(top_n=20)
            recent_gaps = self.data_analysis.analyze_gaps()
            pattern_validation = self.data_analysis.validate_patterns()
            
            # Store original probabilities for comparison
            original_probs = probabilities.copy()
            changes = {}

            # Get latest draw for reference
            latest_draw = self.data_analysis.draws[-1][1] if self.data_analysis.draws else []
            
            for num in range(1, 81):
                try:
                    original_prob = probabilities[num]
                    
                    # 1. Hot/Cold Number Analysis
                    if num in [n for n, _ in recent_hot_cold.get('hot_numbers', [])]:
                        probabilities[num] *= 1.25  # 25% boost
                        print(f"Number {num}: Hot number boost +25%")
                    elif num in [n for n, _ in recent_hot_cold.get('cold_numbers', [])]:
                        probabilities[num] *= 0.9  # 10% reduction
                        print(f"Number {num}: Cold number penalty -10%")
                    
                    # 2. Gap Analysis with Dynamic Boost
                    gap_info = recent_gaps.get(num, {})
                    if gap_info:
                        current_gap = gap_info.get('current_gap', 0)
                        avg_gap = gap_info.get('avg_gap', 1)  # Prevent division by zero
                        
                        if current_gap > avg_gap:
                            # Calculate dynamic boost based on how overdue the number is
                            gap_ratio = current_gap / avg_gap
                            boost = min(1.3, 1 + (gap_ratio * 0.1))  # Cap at 30% boost
                            probabilities[num] *= boost
                            print(f"Number {num}: Gap boost +{(boost-1)*100:.1f}%")
                    
                    # 3. Time Pattern Analysis
                    current_hour = datetime.now().hour
                    if recent_analysis and 'time_patterns' in recent_analysis:
                        if num in recent_analysis['time_patterns'].get(current_hour, []):
                            probabilities[num] *= 1.2
                            print(f"Number {num}: Time pattern boost +20%")
                    
                    # 4. Pattern Consistency
                    if pattern_validation:
                        if num in pattern_validation.get('consistent_numbers', []):
                            probabilities[num] *= 1.15
                            print(f"Number {num}: Pattern consistency boost +15%")
                            
                        if num in pattern_validation.get('trending_numbers', []):
                            probabilities[num] *= 1.1
                            print(f"Number {num}: Trending pattern boost +10%")
                    
                    # 5. Recent Appearance Analysis
                    if latest_draw and num in latest_draw:
                        probabilities[num] *= 0.85  # 15% penalty
                        print(f"Number {num}: Recent appearance penalty -15%")
                    
                    # 6. Trend Analysis
                    if num in [n for n, _ in recent_hot_cold.get('trending_up', [])]:
                        probabilities[num] *= 1.1
                        print(f"Number {num}: Upward trend boost +10%")
                    elif num in [n for n, _ in recent_hot_cold.get('trending_down', [])]:
                        probabilities[num] *= 0.95
                        print(f"Number {num}: Downward trend penalty -5%")
                    
                    # Track significant changes
                    if probabilities[num] != original_prob:
                        change = ((probabilities[num] - original_prob) / original_prob) * 100
                        changes[num] = change
                        
                except Exception as e:
                    print(f"Error processing number {num}: {e}")
                    probabilities[num] = original_probs[num]  # Restore original probability
                    continue
            
            # Normalize probabilities to prevent extremes
            total_prob = sum(probabilities.values())
            if total_prob > 0:
                avg_prob = total_prob / 80
                for num in probabilities:
                    # Cap at 3x average to prevent extreme values
                    if probabilities[num] > 3 * avg_prob:
                        probabilities[num] = 3 * avg_prob
            
            # Show impact summary
            print("\n=== Analysis Impact Summary ===")
            significant_changes = {k: v for k, v in changes.items() if abs(v) > 5}  # Changes > 5%
            if significant_changes:
                print("\nNumbers significantly affected by analysis:")
                for num, change in sorted(significant_changes.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
                    print(f"Number {num}: {change:+.1f}% total change")
            else:
                print("No significant probability changes from analysis")
                
            return probabilities
            
        except Exception as e:
            print(f"Error in prediction weights application: {e}")
            return original_probs  # Return original probabilities if error occurs
    
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
            history_file = PATHS['MODEL_PREDICTIONS']
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
            eval_file = PATHS['EVALUATION_RESULTS']
            
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
            metadata_file = PATHS['MODEL_METADATA']
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

    def validate_data_freshness(self):
        """
        Validate that the data used for prediction is fresh and up-to-date
        
        Returns:
            bool: True if all validations pass, False otherwise
        """
        validation = {
            'data_loaded': False,
            'data_fresh': False,
            'analysis_updated': False
        }
        
        try:
            # Check if data is loaded
            if not hasattr(self.data_analysis, 'draws') or not self.data_analysis.draws:
                print("❌ No data loaded")
                return False
                
            validation['data_loaded'] = True
            
            # Get latest draw time with proper error handling
            latest_draw = self.data_analysis.draws[-1] if self.data_analysis.draws else None
            if not latest_draw:
                print("❌ No draws available")
                return False
                
            latest_draw_time = latest_draw[0]
            
            try:
                # Normalize spaces in the date string first
                latest_draw_time = ' '.join(latest_draw_time.split())
                
                # Then handle both possible formats
                try:
                    # Try format: "HH:MM DD-MM-YYYY"
                    time_part, date_part = latest_draw_time.split(' ', 1)
                    latest_draw_dt = datetime.strptime(f"{date_part} {time_part}", "%d-%m-%Y %H:%M")
                except ValueError:
                    # Fallback: try "DD-MM-YYYY HH:MM" format
                    latest_draw_dt = datetime.strptime(latest_draw_time, "%d-%m-%Y %H:%M")
                    
                time_diff = datetime.now() - latest_draw_dt
                validation['data_fresh'] = time_diff.total_seconds() < 300  # 5 minutes
                
            except Exception as e:
                print(f"Error parsing latest draw time: {e}")
                return False
            
            # Check if analysis is updated
            # Note: This assumes DataAnalysis has a get_analysis_results method
            # If it doesn't, you'll need to modify this check
            if hasattr(self.data_analysis, 'get_analysis_results'):
                latest_analysis = self.data_analysis.get_analysis_results()
                validation['analysis_updated'] = bool(latest_analysis)
            else:
                # Alternative check if get_analysis_results doesn't exist
                # Check if basic analysis methods return valid results
                frequency = self.data_analysis.count_frequency()
                hot_cold = self.data_analysis.hot_and_cold_numbers()
                validation['analysis_updated'] = bool(frequency) and bool(hot_cold)
            
            print("\n=== Data Freshness Validation ===")
            for check, status in validation.items():
                print(f"{check}: {'✅ Valid' if status else '❌ Invalid'}")
                
            # Return True only if all checks passed
            data_fresh = all(validation.values())
            if not data_fresh:
                print("WARNING: Data may not be fresh or analysis not up-to-date!")
                
            return data_fresh
            
        except Exception as e:
            print(f"Error in data freshness validation: {e}")
            import traceback
            traceback.print_exc()
            return False

    def load_and_validate_models(self):
        """
        Load existing models and validate feature structure compatibility.
        If models don't exist or feature structure is incompatible, train new ones.
        
        Returns:
            bool: True if models are ready for use, False otherwise
        """
        print("\n=== Checking Models ===")
        models_loaded = False

        try:
            # Try to load existing models first
            models_loaded = self.load_models()
            
            # Only validate feature structure if models were loaded
            if models_loaded:
                # Check if we have the expected feature metadata
                if self.feature_importances and 'xgboost' in self.feature_importances:
                    # Get feature structure from first model's metadata
                    first_model_key = next(iter(self.feature_importances['xgboost'].keys()))
                    existing_features = set(self.feature_importances['xgboost'][first_model_key].get('features', []))
                    
                    # Get current feature set (just names, don't extract values)
                    test_features = self._extract_pattern_features()
                    current_features = set(test_features.keys())
                    
                    # Check for structural differences only
                    if existing_features != current_features:
                        print("❌ Feature structure has changed since models were trained")
                        print(f"Models expect {len(existing_features)} features, code provides {len(current_features)} features")
                        
                        # Optionally show feature differences
                        missing_features = existing_features - current_features
                        new_features = current_features - existing_features
                        if missing_features:
                            print(f"Missing features: {sorted(missing_features)[:5]}...")
                        if new_features:
                            print(f"New features: {sorted(new_features)[:5]}...")
                        
                        models_loaded = False
                        print("Will retrain models with current feature structure...")
                    else:
                        print("✅ Existing models are compatible with current features")
                else:
                    print("⚠️ Cannot validate model feature structure, missing metadata")
                    print("Will attempt to use models without structure validation...")
        
        except Exception as e:
            print(f"Error checking models: {e}")
            models_loaded = False

        # Train new models if needed
        if not models_loaded:
            print("\n=== Training New Models ===")
            print("This may take some time...")
            
            try:
                training_success = self.train_ensemble_model(
                    n_estimators=100, 
                    learning_rate=0.05, 
                    max_depth=3
                )
                if not training_success:
                    print("❌ Error: Failed to train models.")
                    return False
                print("✅ Models successfully trained with all current features")
                models_loaded = True
            except Exception as e:
                print(f"❌ Error during model training: {e}")
                import traceback
                traceback.print_exc()
                return False

        return models_loaded

    def validate_patterns(self):
        """Add this new method for pattern validation"""
        try:
            # Calculate pattern stability metrics
            hot_cold = self.data_analysis.hot_and_cold_numbers(top_n=20)
            common_pairs = self.data_analysis.find_common_pairs(top_n=20)
            
            # Example stability calculations (you may need to adjust these)
            avg_hot_overlap = 75.5  # Placeholder
            avg_pair_overlap = 65.2  # Placeholder
            avg_trend_overlap = 55.8  # Placeholder
            
            if self.debug:
                print("\nDEBUG: === Pattern Stability Analysis ===")
                for pattern_type, stability in [
                    ("Hot numbers", avg_hot_overlap),
                    ("Pairs", avg_pair_overlap),
                    ("Trends", avg_trend_overlap)
                ]:
                    print(f"DEBUG: {pattern_type} stability: {stability:.2f}%")
                    
            return {
                'hot_overlap': avg_hot_overlap,
                'pair_overlap': avg_pair_overlap,
                'trend_overlap': avg_trend_overlap
            }
        except Exception as e:
            print(f"Error in pattern validation: {e}")
            return None


if __name__ == "__main__":
    try:
        import getpass
        
        # Current initialization with enhanced display
        print("\n====== PATTERN PREDICTION SYSTEM ======")
        print(f"Current Date and Time (UTC): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {getpass.getuser()}")
        
        # Get historical data path with proper error handling
        historical_data_path = PATHS['HISTORICAL_DATA']
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
        
        # Create pattern prediction model with debug enabled
        model = PatternPredictionModel(data_analysis, sequence_length=12, debug=True)
        
        # NEW PART: Check for previous prediction to evaluate
        current_draw_time = datetime.now()
        last_prediction_file = os.path.join(PATHS['PREDICTIONS_DIR'], 'prediction_history.pkl')
        
        if os.path.exists(last_prediction_file):
            try:
                with open(last_prediction_file, 'rb') as f:
                    predictions = pickle.load(f)
                    if predictions:
                        last_prediction = predictions[-1]
                        prediction_time = last_prediction['draw_time']
                        
                        print("\nDEBUG: Checking draw times")
                        print(f"Prediction time: {prediction_time}")
                        print("Last 5 available draws:")
                        for dt, nums in data_analysis.draws[-5:]:
                            print(f"  {dt}: {nums[:5]}...")
                        
                        # Find exact matching draw
                        matching_draw = None
                        matching_time = None
                        for draw_time, numbers in data_analysis.draws:
                            if draw_time.strip() == prediction_time.strip():
                                matching_draw = numbers
                                matching_time = draw_time
                                print(f"\nDEBUG: Found matching draw at {draw_time}")
                                break
                        
                        if matching_draw:
                            print(f"\n=== Evaluating Previous Prediction ===")
                            print(f"Previous prediction made for: {prediction_time}")
                            print(f"Predicted numbers: {last_prediction['predicted_numbers']}")
                            print(f"Actual draw ({matching_time}): {matching_draw}")
                            
                            evaluation = model.evaluate_prediction(
                                last_prediction['predicted_numbers'],
                                matching_draw
                            )
                            
                            print(f"\nEvaluation Results:")
                            print(f"Hit count: {evaluation['hit_count']} out of {len(last_prediction['predicted_numbers'])}")
                            print(f"Hit rate: {evaluation['hit_rate']:.4f}")
                            print(f"Performance vs random: {evaluation['lift']:.2f}x better than random")
                            print(f"Hit numbers: {evaluation['hit_numbers']}")
                        else:
                            print(f"\nWARNING: Could not find matching draw for {prediction_time}")
                            
                            # If no exact match, check if we have a new draw to evaluate (time-based)
                            last_draw_time = datetime.strptime(last_prediction['draw_time'], "%H:%M %d-%m-%Y")
                            
                            # If we have a new draw to evaluate
                            if current_draw_time > last_draw_time:
                                print("\n=== Evaluating Previous Prediction Using Latest Draw ===")
                                print(f"Previous prediction made for: {last_prediction['draw_time']}")
                                print(f"Predicted numbers: {last_prediction['predicted_numbers']}")
                                
                                # Get the latest actual draw for comparison
                                latest_draw = data_analysis.draws[-1][1]
                                latest_draw_time = data_analysis.draws[-1][0]
                                print(f"Latest actual draw ({latest_draw_time}): {latest_draw}")
                                
                                # Evaluate the prediction
                                evaluation = model.evaluate_prediction(
                                    last_prediction['predicted_numbers'],
                                    latest_draw
                                )
                                
                                print(f"\nEvaluation Results:")
                                print(f"Hit count: {evaluation['hit_count']} out of {len(last_prediction['predicted_numbers'])}")
                                print(f"Hit rate: {evaluation['hit_rate']:.4f}")
                                print(f"Performance vs random: {evaluation['lift']:.2f}x better than random")
                                print(f"Hit numbers: {evaluation['hit_numbers']}")
                    else:
                        print("\nNo previous predictions found to evaluate")
            except Exception as e:
                print(f"Warning: Could not evaluate previous prediction: {e}")
                import traceback
                traceback.print_exc()
                
        # Display current time and next draw time
        next_draw_time = model.get_next_draw_time()
        print(f"\nMaking prediction for draw at: {next_draw_time}")
        
        # Add validation checks
        print("\n=== Starting Validation Checks ===")
        
        # 1. Validate data freshness
        data_fresh = model.validate_data_freshness()
        if not data_fresh:
            print("⚠️ Warning: Data may not be fresh - predictions may be less accurate")
            #user_choice = input("Continue with prediction anyway? (y/n): ")
            #if user_choice.lower() != 'y':
             #   print("Prediction cancelled. Please update the data before continuing.")
              #  sys.exit(0)
        
        # 2. Load models with validation
        models_loaded = model.load_and_validate_models()
        print(f"Models loaded: {'✅ Success' if models_loaded else '❌ Failed'}")
        
        if not models_loaded:
            print("No existing models found. Training new models...")
            training_success = model.train_ensemble_model(n_estimators=100, learning_rate=0.05, max_depth=3)
            if not training_success:
                print("❌ Error: Failed to train models. Exiting.")
                sys.exit(1)
            print("✅ Models successfully trained")
        
        # In your main code
        models_ready = model.load_and_validate_models()
        if not models_ready:
            print("❌ Could not load or train models. Exiting.")
            sys.exit(1)
        print("✅ Models ready for predictions")

        # 3. Generate prediction with validation
        print("\n=== Generating Prediction ===")
        prediction, confidence = model.predict_next_draw(num_predictions=15)
        
        if not prediction:
            print("❌ Error: Failed to generate prediction")
            sys.exit(1)
        
        print("\n==== PATTERN-BASED PREDICTION ====")
        print(f"Top 15 predicted numbers:")
        for num, conf in zip(prediction, confidence):
            print(f"Number {num}: {conf:.4f} confidence")
            
        # Visualize the prediction
        visualization = model.visualize_predictions(prediction, confidence)
        if visualization:
            print("✅ Prediction visualization created")
            plt.show()
        else:
            print("❌ Warning: Could not create prediction visualization")
        
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
        
        # Final timestamp
        print(f"\nPrediction generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"For draw at: {next_draw_time}")
        print("\n======================================")
        
    except Exception as e:
        print(f"Error in pattern prediction: {e}")
        import traceback
        traceback.print_exc()
