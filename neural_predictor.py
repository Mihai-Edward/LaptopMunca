import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
import joblib
import sys

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Now import from config and src
from config.paths import PATHS, ensure_directories
from src.data_analysis import DataAnalysis

class NeuralPredictor:
    def __init__(self, debug=False):
        self.debug = debug
        self.scaler = StandardScaler()
        
        self.model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            max_iter=200,
            random_state=42,
            verbose=True if debug else False
        )
        
        # Cache for analysis results
        self._frequency_cache = None
        self._hot_cold_cache = None
        self._gaps_cache = None

    def _calculate_analysis(self, data_analysis):
        """Cache analysis results to avoid recalculation"""
        if self.debug:
            print("Calculating analysis results...")
            
        self._frequency_cache = data_analysis.count_frequency()
        self._hot_cold_cache = data_analysis.hot_and_cold_numbers(top_n=80)
        self._gaps_cache = data_analysis.analyze_gaps()

    def prepare_features(self, data_analysis):
        """Convert data analysis into features"""
        if self._frequency_cache is None:
            self._calculate_analysis(data_analysis)
            
        features = []
        
        if self.debug:
            print("Preparing features...")
            
        for i in range(len(data_analysis.draws) - 1):
            feature_vector = []
            
            # 1. Frequency features (80 features)
            freq_features = [self._frequency_cache.get(num, 0) for num in range(1, 81)]
            feature_vector.extend(freq_features)
            
            # 2. Hot/Cold features (160 features)
            hot_numbers = [num for num, _ in self._hot_cold_cache['hot_numbers']]
            cold_numbers = [num for num, _ in self._hot_cold_cache['cold_numbers']]
            hot_features = [1 if num in hot_numbers else 0 for num in range(1, 81)]
            cold_features = [1 if num in cold_numbers else 0 for num in range(1, 81)]
            feature_vector.extend(hot_features)
            feature_vector.extend(cold_features)
            
            # 3. Gap features (80 features)
            gap_features = [self._gaps_cache[num]['current_gap'] for num in range(1, 81)]
            feature_vector.extend(gap_features)
            
            # 4. Recent history (80 features)
            current_draw = data_analysis.draws[i][1]
            history_features = [1 if num in current_draw else 0 for num in range(1, 81)]
            feature_vector.extend(history_features)
            
            features.append(feature_vector)
        
        # Scale features
        features = np.array(features)
        if len(features) > 0:
            return self.scaler.fit_transform(features)
        return features

    def prepare_labels(self, data_analysis):
        """Prepare labels (next draw numbers) for training"""
        if self.debug:
            print("Preparing labels...")
            
        labels = []
        
        for i in range(len(data_analysis.draws) - 1):
            next_draw = data_analysis.draws[i + 1][1]
            label = [1 if num in next_draw else 0 for num in range(1, 81)]
            labels.append(label)
        
        return np.array(labels)

    def train(self, data_analysis):
        """Train the neural network"""
        try:
            if self.debug:
                print("\nStarting training process...")
            
            X = self.prepare_features(data_analysis)
            y = self.prepare_labels(data_analysis)
            
            if self.debug:
                print(f"Feature shape: {X.shape}")
                print(f"Labels shape: {y.shape}")
                print("Training model...")
            
            # Train model
            self.model.fit(X, y)
            
            # Save model
            self.save_model()
            
            if self.debug:
                print(f"Training score: {self.model.score(X, y):.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def predict(self, data_analysis, top_n=20):
        """Predict next draw numbers"""
        try:
            # Calculate analysis if not cached
            if self._frequency_cache is None:
                self._calculate_analysis(data_analysis)
                
            # Prepare features for the latest draw
            latest_features = self.prepare_features(data_analysis)[-1:]
            
            # Get probabilities for each number
            probabilities = self.model.predict_proba(latest_features)[0]
            
            # Get top N numbers with highest probabilities
            top_indices = np.argsort(probabilities)[-top_n:][::-1]
            top_numbers = [idx + 1 for idx in top_indices]
            top_probabilities = probabilities[top_indices]
            
            return {
                'numbers': top_numbers,
                'probabilities': top_probabilities.tolist(),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None

    # ... rest of the code remains the same ...

    def save_model(self):
        """Save the trained model and scaler"""
        try:
            model_dir = os.path.join(PATHS['BASE_DIR'], 'models')
            os.makedirs(model_dir, exist_ok=True)
            
            # Save model
            model_path = os.path.join(model_dir, 'neural_predictor.joblib')
            joblib.dump(self.model, model_path)
            
            # Save scaler
            scaler_path = os.path.join(model_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            
            if self.debug:
                print(f"Model saved to: {model_path}")
                print(f"Scaler saved to: {scaler_path}")
                
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            model_path = os.path.join(PATHS['BASE_DIR'], 'models', 'neural_predictor.joblib')
            scaler_path = os.path.join(PATHS['BASE_DIR'], 'models', 'scaler.joblib')
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                return True
            return False
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

def main():
    # Example usage
    try:
        # Load your draw data
        real_draws = []
        csv_file = PATHS['HISTORICAL_DATA']
        
        if os.path.exists(csv_file):
            print(f"\nLoading historical draws from: {csv_file}")
            
            import csv
            with open(csv_file, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                header = next(csv_reader, None)  # Skip header if exists
                
                for row in csv_reader:
                    if len(row) >= 21:  # Date/time + 20 numbers
                        draw_date = row[0]
                        try:
                            numbers = [int(num.strip()) for num in row[1:21] if num.strip()]
                            real_draws.append((draw_date, numbers))
                        except ValueError as e:
                            print(f"Skipping row with invalid numbers: {e}")
            
            if real_draws:
                print(f"Successfully loaded {len(real_draws)} historical draws")
                
                # Create DataAnalysis instance
                data_analysis = DataAnalysis(real_draws, debug=True)
                
                # Create and train predictor
                predictor = NeuralPredictor(debug=True)
                success = predictor.train(data_analysis)
                
                if success:
                    print("\nTraining completed!")
                    
                    # Make prediction
                    prediction = predictor.predict(data_analysis)
                    if prediction:
                        print("\nPredicted numbers for next draw:")
                        for num, prob in zip(prediction['numbers'], prediction['probabilities']):
                            print(f"Number {num}: {prob:.3f} probability")
                
            else:
                print(f"No valid draws found in {csv_file}")
                
        else:
            print(f"Historical data file not found: {csv_file}")
            
    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()