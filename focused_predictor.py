import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
from pathlib import Path
from config.paths import PATHS
import joblib  # for model saving/loading

class ModelHealthMonitor:
    """Monitors model health and determines if retraining is needed"""
    def __init__(self):
        self.thresholds = {
            'pattern_change': 0.30,    # 30% pattern change trigger
            'accuracy_drop': 0.15,     # 15% minimum accuracy
            'consecutive_failures': 5   # 5 consecutive failures trigger
        }
        self.performance_history = []
        self.load_thresholds()

    def load_thresholds(self):
        """Load thresholds from config file if exists"""
        if os.path.exists(PATHS['MODEL_THRESHOLDS']):
            with open(PATHS['MODEL_THRESHOLDS'], 'r') as f:
                self.thresholds.update(json.load(f))

class FocusedPredictor:
    """Main class for focused 15-number prediction"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.health_monitor = ModelHealthMonitor()
        self.last_training_date = None
        self.current_user = os.getenv('USER', 'Mihai-Edward')
        self.setup_directories()

    def setup_directories(self):
        """Ensure all required directories exist"""
        for path in [PATHS['MODELS_DIR'], PATHS['ANALYSIS_DIR'], 
                    PATHS['PREDICTIONS_DIR'], PATHS['MONITORING_DIR']]:
            os.makedirs(path, exist_ok=True)

    def prepare_features(self, numbers_data):
        """Prepare features for the model"""
        features = []
        for num in range(1, 81):
            feature_vector = [
                numbers_data['frequency'].get(num, 0),
                numbers_data['recent_performance'].get(num, 0),
                numbers_data['pattern_strength'].get(num, 0)
            ]
            features.append(feature_vector)
        return np.array(features)

    def predict_numbers(self, features):
        """Get probability predictions for all numbers"""
        if self.model is None:
            self.load_model()
        
        probabilities = self.model.predict_proba(features)[:, 1]
        number_probs = {i+1: prob for i, prob in enumerate(probabilities)}
        return number_probs

    def get_top_15_numbers(self, probabilities):
        """Get the top 15 numbers based on probabilities"""
        sorted_numbers = sorted(probabilities.items(), 
                              key=lambda x: x[1], 
                              reverse=True)
        
        return {
            'numbers': [num for num, _ in sorted_numbers[:15]],
            'confidence': [score for _, score in sorted_numbers[:15]]
        }

    def save_model(self):
        """Save the model using joblib"""
        model_path = os.path.join(PATHS['MODELS_DIR'], 'focused_predictor.joblib')
        joblib.dump(self.model, model_path)
        
        # Save metadata
        metadata = {
            'last_training': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'user': self.current_user,
            'version': '1.0'
        }
        
        metadata_path = os.path.join(PATHS['MODELS_DIR'], 'model_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def load_model(self):
        """Load the model using joblib"""
        model_path = os.path.join(PATHS['MODELS_DIR'], 'focused_predictor.joblib')
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            return True
        return False

    def save_prediction(self, prediction):
        """Save prediction with metadata"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prediction_data = {
            'timestamp': timestamp,
            'user': self.current_user,
            'numbers': prediction['numbers'],
            'confidence_scores': prediction['confidence']
        }
        
        # Save to CSV
        self.update_predictions_csv(prediction_data)
        
        # Save detailed JSON
        log_file = os.path.join(
            PATHS['PREDICTIONS_DIR'], 
            f'prediction_{timestamp.replace(":", "-")}.json'
        )
        with open(log_file, 'w') as f:
            json.dump(prediction_data, f, indent=4)

    def update_predictions_csv(self, prediction_data):
        """Update the consolidated predictions CSV file"""
        csv_file = PATHS['TOP_15_PREDICTIONS']
        
        # Prepare row data
        row_data = {
            'timestamp': prediction_data['timestamp'],
            'user': prediction_data['user']
        }
        
        # Add numbers and confidence scores
        for i, (num, conf) in enumerate(zip(
            prediction_data['numbers'], 
            prediction_data['confidence_scores']), 1):
            row_data[f'number_{i}'] = num
            row_data[f'confidence_{i}'] = conf
        
        # Update CSV
        df = pd.DataFrame([row_data])
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)

    def train_model(self, historical_data):
        """
        Train the model on historical data
        
        Args:
            historical_data (dict): Dictionary containing historical data with features
            
        Returns:
            bool: True if training successful, False otherwise
        """
        try:
            # Prepare features
            X = self.prepare_features(historical_data)
            # Prepare labels (1 for numbers that appeared, 0 for those that didn't)
            y = self.prepare_labels(historical_data)
            
            # Train the model
            self.model.fit(X, y)
            
            # Update training metadata
            self.last_training_date = datetime.now()
            
            # Save the trained model
            self.save_model()
            
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def evaluate_model(self, test_data):
        """
        Evaluate model performance
        
        Args:
            test_data (dict): Dictionary containing test data with features
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        try:
            # Prepare test features
            X_test = self.prepare_features(test_data)
            # Prepare test labels
            y_test = self.prepare_labels(test_data)
            
            # Get predictions
            predictions = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = self.calculate_accuracy(predictions, y_test)
            
            return {
                'accuracy': accuracy,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        except Exception as e:
            print(f"Error evaluating model: {e}")
            return None

    def analyze_patterns(self, historical_data):
        """
        Analyze number patterns
        
        Args:
            historical_data (list): List of historical draw data
        
        Returns:
            dict: Dictionary containing pattern analysis features
        """
        try:
            pattern_features = {
                'frequency': self.analyze_frequency(historical_data),
                'gaps': self.analyze_gaps(historical_data),
                'trends': self.analyze_trends(historical_data),
                'combinations': self.analyze_combinations(historical_data)
            }
            return pattern_features
        except Exception as e:
            print(f"Error analyzing patterns: {e}")
            return None
