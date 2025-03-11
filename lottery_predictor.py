import traceback
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
import joblib
from collections import OrderedDict, Counter
from collections import defaultdict

from xgboost import XGBClassifier
from data_analysis import DataAnalysis
from datetime import datetime
import os
import glob
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import PATHS, ensure_directories

class LotteryPredictor:
    def __init__(self, numbers_range=(1, 80), numbers_to_draw=20, use_combined_features=True):
        """Initialize LotteryPredictor with enhanced model configuration"""
        # Validate input parameters
        if not isinstance(numbers_range, tuple) or len(numbers_range) != 2:
            raise ValueError("numbers_range must be a tuple of (min, max)")
        if not isinstance(numbers_to_draw, int) or numbers_to_draw <= 0:
            raise ValueError("numbers_to_draw must be a positive integer")
            
        # Core settings with validation
        self.numbers_range = numbers_range
        self.numbers_to_draw = numbers_to_draw
        self.num_classes = numbers_range[1] - numbers_range[0] + 1
        
        # Initialize paths using config
        ensure_directories()
        self.models_dir = PATHS['MODELS_DIR']
        self.data_file = PATHS['HISTORICAL_DATA']
        self.predictions_file = PATHS['PREDICTIONS']
        
        # Models initialization with enhanced configuration
        self.scaler = StandardScaler()
        
        # Modified probabilistic model initialization
        self.probabilistic_model = None
    
        # Neural network with optimized architecture
        self.pattern_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 80),  # Optimized for lottery prediction
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization parameter
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            shuffle=True,
            random_state=42,
            early_stopping=True,  # Enable early stopping
            validation_fraction=0.1,  # Use 10% of training data for validation
            n_iter_no_change=10  # Number of iterations with no improvement
        )
        
        # Initialize analysis components
        self.analyzer = None  # Will be initialized when needed
        
        # Enhanced state tracking
        self.training_status = {
            'success': False,
            'model_loaded': False,
            'timestamp': None,
            'error': None,
            'prob_score': None,
            'pattern_score': None,
            'features': None,
            'model_config': {
                'num_classes': self.num_classes,
                'numbers_to_draw': self.numbers_to_draw,
                'feature_dimension': None  # Will be set during training
            }
        }
        
        # Initialize pipeline data storage
        self.pipeline_data = {
            # NEW: Flag to control whether to use combined features or just base features
            'use_combined_features': use_combined_features
        }
        
        # Initialize pipeline stages
        self._initialize_pipeline()
        
        print(f"\nInitialized LotteryPredictor:")
        print(f"- Number range: {numbers_range}")
        print(f"- Numbers to draw: {numbers_to_draw}")
        print(f"- Number of classes: {self.num_classes}")
        print(f"- Using combined features: {use_combined_features}")
        
    def validate_model_state(self):
        """Validate model state before prediction"""
        try:
            # Check if models exist
            if self.probabilistic_model is None or self.pattern_model is None:
                return False, "Models not initialized"

            # Check if models are trained
            if not hasattr(self.probabilistic_model, 'class_prior_') or \
               not hasattr(self.pattern_model, 'coefs_'):
                return False, "Models not properly trained"

            # Check scaler
            if self.scaler is None or not hasattr(self.scaler, 'mean_'):
                return False, "Scaler not initialized"

            # Validate feature dimensions
            if hasattr(self.probabilistic_model, 'n_features_in_'):
                expected_features = self.probabilistic_model.n_features_in_
                # Check if using combined features (base + analysis)
                use_combined = self.pipeline_data.get('use_combined_features', False)
                if use_combined:
                    # When using combined features, we expect 84 base + 20 analysis = 104 features
                    if expected_features != 104:
                        return False, f"Invalid feature dimension for combined model: {expected_features}"
                else:
                    # When using only base features, we expect 84 features
                    if expected_features != 84:
                        return False, f"Invalid feature dimension: {expected_features}"

            return True, "Model state valid"
        except Exception as e:
            return False, f"Model validation error: {str(e)}"
    
    def _initialize_pipeline(self):
        """Initialize the prediction pipeline with ordered stages"""
        self.pipeline_stages = OrderedDict({
            'data_preparation': self._prepare_pipeline_data,
            'feature_engineering': self._create_enhanced_features,
            'model_prediction': self._generate_model_predictions,
            'post_processing': self._post_process_predictions
        })
        self.pipeline_data = {}
    
    def _prepare_pipeline_data(self, data):
        """First pipeline stage: Prepare and validate input data"""
        print("\nPreparing data for prediction pipeline...")
        try:
            # Ensure we have a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            # Ensure we have the required number of rows (5 recent draws)
            if len(data) < 5:
                raise ValueError("Need at least 5 recent draws for prediction")
                
            # Clean the data
            prepared_data = self.clean_data(data)
            if prepared_data is None or len(prepared_data) < 5:
                raise ValueError("Data cleaning resulted in insufficient data")
                
            # Ensure all required columns exist
            required_cols = ['date'] + [f'number{i+1}' for i in range(20)]
            missing_cols = [col for col in required_cols if col not in prepared_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Sort by date and get most recent 5 draws
            prepared_data = prepared_data.sort_values('date').tail(5)
            
            # Validate number ranges
            number_cols = [f'number{i+1}' for i in range(20)]
            for col in number_cols:
                invalid_numbers = prepared_data[~prepared_data[col].between(1, 80)]
                if not invalid_numbers.empty:
                    raise ValueError(f"Invalid numbers found in column {col}")
                    
            # Add date-based features
            if 'date' in prepared_data.columns:
                prepared_data['day_of_week'] = prepared_data['date'].dt.dayofweek
                prepared_data['month'] = prepared_data['date'].dt.month
                prepared_data['day_of_year'] = prepared_data['date'].dt.dayofyear
                prepared_data['days_since_first_draw'] = (
                    prepared_data['date'] - prepared_data['date'].min()
                ).dt.days
                
                # Store additional features in pipeline data
                self.pipeline_data['date_features'] = {
                    'day_of_week': prepared_data['day_of_week'].tolist(),
                    'month': prepared_data['month'].tolist(),
                    'day_of_year': prepared_data['day_of_year'].tolist(),
                    'days_since_first_draw': prepared_data['days_since_first_draw'].tolist()
                }
            
            # Store in pipeline data for potential later use
            self.pipeline_data['prepared_data'] = prepared_data
            print("Data preparation completed successfully")
            return prepared_data
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            return None

    def load_data(self, file_path=None):
        """Enhanced data loading with validation"""
        if file_path is None:
            file_path = self.data_file
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
        
        try:
            # Load CSV with header
            df = pd.read_csv(file_path, header=0)
            
            # Clean and convert dates
            try:
                df['date'] = df['date'].str.strip()
                df['date'] = pd.to_datetime(df['date'], format='%H:%M  %d-%m-%Y')
            except Exception as e:
                try:
                    # Fallback to single space format
                    df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y')
                except Exception as e2:
                    print(f"Warning: Date conversion failed: {e2}")
                    return None
            
            # Convert number columns to float
            number_cols = [f'number{i}' for i in range(1, 21)]
            for col in number_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            # Validate number ranges
            for col in number_cols:
                invalid_mask = ~df[col].between(1, 80)
                if invalid_mask.any():
                    print(f"Warning: Found {invalid_mask.sum()} invalid numbers in {col}")
                    # Fill invalid numbers with the mode of valid numbers
                    valid_mode = df.loc[~invalid_mask, col].mode()
                    df.loc[invalid_mask, col] = valid_mode[0] if not valid_mode.empty else 0
            
            return df
                
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None

    def clean_data(self, data):
        """Enhanced data cleaning with validation"""
        try:
            # Remove duplicates
            data = data.drop_duplicates()
            
            # Ensure numbers are within valid range (1-80)
            number_cols = [f'number{i+1}' for i in range(20)]
            for col in number_cols:
                data = data[(data[col] >= 1) & (data[col] <= 80)]
            
            # Sort numbers in each draw
            for _, row in data.iterrows():
                numbers = sorted(row[number_cols])
                for i, num in enumerate(numbers):
                    data.at[row.name, f'number{i+1}'] = num
            
            # Remove rows with missing values
            data = data.dropna()
            
            print("Data cleaning completed successfully")
            return data
            
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            return data

    def prepare_data(self, historical_data):
        """Prepare data for training with enhanced validation and preprocessing"""
        try:
            print("\nPreparing training data...")
            if historical_data is None or len(historical_data) < 6:
                raise ValueError("Insufficient historical data for training (minimum 6 draws required)")

            # Sort data chronologically
            historical_data = historical_data.sort_values('date')
            features = []
            labels = []

            # Define column names for the 20 numbers
            number_cols = [f'number{i+1}' for i in range(20)]

            # Validate number columns exist
            missing_cols = [col for col in number_cols if col not in historical_data.columns]
            if missing_cols:
                raise ValueError(f"Missing number columns: {missing_cols}")

            print(f"\nProcessing {len(historical_data) - 5} potential training samples...")
            valid_samples = 0
            skipped_samples = 0
            errors = defaultdict(int)

            # Create sliding window for feature extraction
            for i in range(len(historical_data) - 5):
                try:
                    # Get current window and next draw
                    window = historical_data.iloc[i:i+5]
                    next_draw = historical_data.iloc[i+5]

                    # Validate window dates are consecutive
                    dates = pd.to_datetime(window['date'])
                    if (dates.diff()[1:] > pd.Timedelta(days=2)).any():
                        errors['non_consecutive_dates'] += 1
                        print(f"Skipped sample at index {i} due to non-consecutive dates")
                        continue

                    # Create feature vector from window with validation
                    feature_vector = self._create_feature_vector(window)
                    if feature_vector is None:
                        errors['invalid_feature_vector'] += 1
                        print(f"Skipped sample at index {i} due to invalid feature vector")
                        continue

                    if len(feature_vector) != 84:  # Expected feature dimension
                        errors['wrong_feature_dimension'] += 1
                        print(f"Skipped sample at index {i} due to wrong feature dimension")
                        continue

                    # Get all 20 numbers as labels with validation
                    try:
                        draw_numbers = next_draw[number_cols].values.astype(int)

                        # Basic number validation
                        if len(draw_numbers) != self.numbers_to_draw:
                            errors['wrong_number_count'] += 1
                            print(f"Skipped sample at index {i} due to wrong number count")
                            continue

                        if not all((1 <= n <= 80) for n in draw_numbers):
                            errors['numbers_out_of_range'] += 1
                            print(f"Skipped sample at index {i} due to numbers out of range")
                            continue

                        # Check for duplicates
                        if len(set(draw_numbers)) != self.numbers_to_draw:
                            errors['duplicate_numbers'] += 1
                            print(f"Skipped sample at index {i} due to duplicate numbers")
                            continue

                        # Sort numbers for consistency
                        draw_numbers = np.sort(draw_numbers)

                        features.append(feature_vector)
                        labels.append(draw_numbers)
                        valid_samples += 1

                        if valid_samples % 100 == 0:  # Progress update every 100 valid samples
                            print(f"Processed {valid_samples} valid samples...")

                    except Exception as e:
                        print(f"Error processing draw at index {i+5}: {e}")
                        errors['draw_processing'] += 1
                        continue

                except Exception as e:
                    print(f"Error processing window at index {i}: {e}")
                    errors['window_processing'] += 1
                    continue

            # Convert to numpy arrays with validation
            if len(features) == 0 or len(labels) == 0:
                raise ValueError("No valid training samples generated")

            features = np.array(features)
            labels = np.array(labels)

            # Final validation
            if len(features) != len(labels):
                raise ValueError(f"Feature/label mismatch: {len(features)} features vs {len(labels)} labels")

            # Print detailed summary
            print("\nData Preparation Summary:")
            print(f"- Total potential samples: {len(historical_data) - 5}")
            print(f"- Valid samples generated: {valid_samples}")
            print(f"- Feature shape: {features.shape}")
            print(f"- Labels shape: {labels.shape}")
            print(f"- Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

            if errors:
                print("\nErrors encountered:")
                for error_type, count in errors.items():
                    print(f"- {error_type}: {count}")

            # Additional statistics
            unique_first_numbers = len(np.unique([label[0] for label in labels]))
            print(f"\nLabel Statistics:")
            print(f"- Unique first numbers: {unique_first_numbers}/80")
            print(f"- Numbers distribution range: {labels.min()}-{labels.max()}")

            return features, labels

        except Exception as e:
            print(f"Error preparing training data: {e}")
            return None, None

    def _create_feature_vector(self, window):
        """Create standardized 80-dimension feature vector"""
        try:
            # Check if window is a DataFrame
            if isinstance(window, pd.DataFrame):
                # Get number columns
                number_cols = [col for col in window.columns if col.startswith('number')]
                if not number_cols:
                    raise ValueError("No number columns found in DataFrame")
                
                # Create frequency vector
                frequency_vector = np.zeros(80)
                
                # Process each row
                for _, row in window.iterrows():
                    numbers = row[number_cols].values.astype(float)
                    valid_numbers = numbers[(numbers >= 1) & (numbers <= 80)]
                    for num in valid_numbers:
                        frequency_vector[int(num)-1] += 1
                        
                # Normalize if we have any valid numbers
                total_numbers = np.sum(frequency_vector)
                if total_numbers > 0:
                    frequency_vector = frequency_vector / total_numbers
                else:
                    raise ValueError("No valid numbers found in window")
                    
                return frequency_vector
                
            else:
                raise ValueError("Input must be a pandas DataFrame")
                
        except Exception as e:
            print(f"Error creating feature vector: {e}")
            print(f"Window type: {type(window)}")
            if isinstance(window, pd.DataFrame):
                print(f"Columns: {window.columns.tolist()}")
            return None

    def _create_analysis_features(self, data):
        """Create enhanced features from data analysis"""
        print("\nGenerating analysis features...")
        
        try:
            # Update analyzer with current data
            formatted_draws = []
            for _, row in data.iterrows():
                numbers = []
                for i in range(1, 21):
                    num = row.get(f'number{i}')
                    if isinstance(num, (int, float)) and 1 <= num <= 80:
                        numbers.append(int(num))
                if len(numbers) == 20:
                    formatted_draws.append((row['date'], numbers))
            
            self.analyzer = DataAnalysis(formatted_draws)
            
            # Get analysis results
            frequency = self.analyzer.count_frequency()
            hot_numbers, cold_numbers = self.analyzer.hot_and_cold_numbers()
            common_pairs = self.analyzer.find_common_pairs()
            range_analysis = self.analyzer.number_range_analysis()
            
            # Convert analysis results to features - fixed size array
            analysis_features = np.zeros(160)  # 80 for frequency + 80 for hot/cold
            
            # Frequency features (first 80)
            total_freq = sum(frequency.values()) or 1  # Avoid division by zero
            for num, freq in frequency.items():
                if 1 <= num <= 80:
                    analysis_features[num-1] = freq / total_freq
            
            # Hot/Cold features (second 80)
            hot_nums = dict(hot_numbers)
            max_hot_score = max(hot_nums.values()) if hot_nums else 1
            for num, score in hot_nums.items():
                if 1 <= num <= 80:
                    analysis_features[80 + num-1] = score / max_hot_score
            
            # Store analysis context with additional metadata
            self.pipeline_data['analysis_context'] = {
                'frequency': frequency,
                'hot_cold': (hot_numbers, cold_numbers),
                'common_pairs': common_pairs,
                'range_analysis': range_analysis,
                'feature_stats': {
                    'total_frequency': total_freq,
                    'max_hot_score': max_hot_score,
                    'feature_range': (np.min(analysis_features), np.max(analysis_features))
                }
            }
            
            print(f"Generated {len(analysis_features)} analysis features")
            print(f"Feature range: {np.min(analysis_features):.4f} to {np.max(analysis_features):.4f}")
            
            return analysis_features
            
        except Exception as e:
            print(f"Error in analysis features generation: {e}")
            return np.zeros(160)  # Return zero vector of fixed size

    def extract_sequence_patterns(self, data, sequence_length=3):
        """Extract sequence patterns with validation"""
        try:
            sequences = Counter()
            for _, row in data.iterrows():
                numbers = [row[f'number{i+1}'] for i in range(20)]
                if all(isinstance(n, (int, float)) and 1 <= n <= 80 for n in numbers):
                    for i in range(len(numbers) - sequence_length + 1):
                        sequence = tuple(sorted(numbers[i:i+sequence_length]))
                        sequences.update([sequence])
            return sequences.most_common()
            
        except Exception as e:
            print(f"Error in sequence pattern extraction: {e}")
            return []

    def extract_clusters(self, data, n_clusters=3):
        """Extract clusters with enhanced validation"""
        try:
            frequency = Counter()
            for _, row in data.iterrows():
                numbers = [row[f'number{i+1}'] for i in range(20)]
                valid_numbers = [n for n in numbers if isinstance(n, (int, float)) and 1 <= n <= 80]
                frequency.update(valid_numbers)
            
            numbers = list(frequency.keys())
            frequencies = list(frequency.values())
            X = np.array(frequencies).reshape(-1, 1)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
            
            clusters = {i: [] for i in range(n_clusters)}
            for number, label in zip(numbers, kmeans.labels_):
                clusters[label].append(number)
            
            return clusters
            
        except Exception as e:
            print(f"Error in clustering: {e}")
            return None

    def get_analysis_features(self, data):
        """Get enhanced analysis features from TrainPredictor"""
        try:
            draws = [(row['date'].strftime('%H:%M %d-%m-%Y'), 
                     [row[f'number{i}'] for i in range(1, 21)]) 
                    for _, row in data.iterrows()]
            
            analyzer = DataAnalysis(draws)
            analysis_features = {
                'frequency': analyzer.count_frequency(),
                'hot_cold': analyzer.hot_and_cold_numbers(),
                'common_pairs': analyzer.find_common_pairs(),
                'range_analysis': analyzer.number_range_analysis(),
                'sequences': self.extract_sequence_patterns(data),
                'clusters': self.extract_clusters(data)
            }
            
            # Store in pipeline data
            self.pipeline_data['analysis_features'] = analysis_features
            return analysis_features
            
        except Exception as e:
            print(f"Error getting analysis features: {e}")
            return {}

    def _create_enhanced_features(self, data):
        """Create enhanced feature vector combining all features"""
        print("\nGenerating enhanced features...")
        try:
            # Get base features (80 frequency features + 4 statistical features)
            base_features = self._create_feature_vector(data)
            
            # Get analysis features (160 features)
            analysis_features = self._create_analysis_features(data)
            
            # Validate feature dimensions
            if base_features is None or analysis_features is None:
                raise ValueError("Failed to generate either base or analysis features")
            
            print(f"Base features shape: {base_features.shape}")
            print(f"Analysis features shape: {analysis_features.shape}")
            
            # Ensure consistent feature dimensions
            if len(base_features) == 84 and len(analysis_features) == 160:
                # Store both feature sets in pipeline data
                self.pipeline_data['base_features'] = base_features
                self.pipeline_data['analysis_features'] = analysis_features
                
                # Check if we should use combined features or just base features
                use_combined = self.pipeline_data.get('use_combined_features', False)
                
                if use_combined:
                    # CHANGE: Actually use the analysis features by combining them with base features
                    # This is the key change - we're combining both feature sets
                    try:
                        # Take the first 20 analysis features (most important ones) to avoid dimensionality issues
                        analysis_subset = analysis_features[:20]
                        enhanced_features = np.concatenate([base_features, analysis_subset])
                        print(f"Using COMBINED feature vector of shape: {enhanced_features.shape}")
                        print("Using 60% analysis features weight")
                    except Exception as e:
                        print(f"Error combining features: {e}. Falling back to base features.")
                        enhanced_features = base_features
                else:
                    # For compatibility with existing models
                    print("Using only base features for compatibility with existing models")
                    enhanced_features = base_features
                
                # Add feature metadata
                self.pipeline_data['feature_metadata'] = {
                    'base_features_size': len(base_features),
                    'analysis_features_size': len(analysis_features),
                    'used_features_size': len(enhanced_features),
                    'using_combined': use_combined,
                    'feature_stats': {
                        'mean': float(np.mean(enhanced_features)),
                        'std': float(np.std(enhanced_features)),
                        'min': float(np.min(enhanced_features)),
                        'max': float(np.max(enhanced_features))
                    }
                }
                
                print(f"Using feature vector of shape: {enhanced_features.shape}")
                print("Feature statistics:")
                for key, value in self.pipeline_data['feature_metadata']['feature_stats'].items():
                    print(f"  {key}: {value:.4f}")
                
                self.pipeline_data['features'] = enhanced_features
                return enhanced_features
            else:
                print(f"Feature dimension mismatch: base={len(base_features)}, analysis={len(analysis_features)}")
                print("Falling back to base features only")
                return base_features
                
        except Exception as e:
            print(f"Error in enhanced feature creation: {e}")
            print("Falling back to base feature generation")
            return self._create_feature_vector(data)  # Fallback to base features
    
    def _generate_model_predictions(self, features):
        """Generate predictions for all 80 possible numbers"""
        print("\nGenerating model predictions...")
        try:
            # Validate features
            if features is None:
                raise ValueError("No features provided for prediction")
            
            # Ensure features are 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
                
            # Scale features
            scaled_features = self.scaler.transform(features)
            
            # Generate predictions for all 80 numbers
            prob_pred = np.zeros(80)  # Initialize array for all possible numbers
            pattern_pred = np.zeros(80)  # Initialize array for all possible numbers
            
            # Get model predictions
            prob_raw = self.probabilistic_model.predict_proba(scaled_features)[0]
            pattern_raw = self.pattern_model.predict_proba(scaled_features)[0]
            
            # Ensure we have probabilities for all 80 numbers
            prob_pred[:len(prob_raw)] = prob_raw
            pattern_pred[:len(pattern_raw)] = pattern_raw
            
            # Normalize probabilities
            prob_pred = prob_pred / np.sum(prob_pred)
            pattern_pred = pattern_pred / np.sum(pattern_pred)
            
            # Validate predictions
            if np.any(np.isnan(prob_pred)) or np.any(np.isnan(pattern_pred)):
                raise ValueError("NaN values detected in predictions")
                
            print(f"\nProbabilistic Model - Number of predictions: {len(prob_pred)}")
            print(f"Pattern Model - Number of predictions: {len(pattern_pred)}")
            
            return prob_pred, pattern_pred
            
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return None, None

    def _post_process_predictions(self, predictions):
        """Process and combine predictions to get exactly 20 numbers"""
        print("\nPost-processing predictions...")
        try:
            # Better validation of predictions input
            if (predictions is None or 
                not isinstance(predictions, (list, tuple)) or 
                len(predictions) != 2 or 
                any(p is None for p in predictions)):
                print("ERROR: Invalid predictions input")
                return None, None, None
                
            prob_pred, pattern_pred = predictions
            
            # Convert to numpy arrays if needed
            if isinstance(prob_pred, list):
                prob_pred = np.array(prob_pred)
            if isinstance(pattern_pred, list):
                pattern_pred = np.array(pattern_pred)
            
            # Validate array shapes
            if prob_pred.shape != (self.num_classes,) or pattern_pred.shape != (self.num_classes,):
                print(f"ERROR: Invalid prediction shapes - prob: {prob_pred.shape}, pattern: {pattern_pred.shape}")
                return None, None, None
            
            # Normalize and combine predictions
            prob_pred = prob_pred / np.sum(prob_pred)
            pattern_pred = pattern_pred / np.sum(pattern_pred)
            
            # NEW: Check for analysis data and create analysis prediction
            analysis_pred = np.zeros(self.num_classes)
            analysis_weight = 0.0
            
            if 'analysis_context' in self.pipeline_data:
                # Try to get hot numbers from analysis context
                if 'hot_cold' in self.pipeline_data['analysis_context']:
                    hot_cold = self.pipeline_data['analysis_context']['hot_cold']
                    if isinstance(hot_cold, tuple) and len(hot_cold) > 0:
                        hot_numbers = hot_cold[0]  # First element is hot numbers
                        
                        # Create weights based on hot number frequency
                        for num, count in hot_numbers:
                            if 1 <= num <= self.num_classes:
                                analysis_pred[num-1] = count
                                
                        # Normalize
                        sum_analysis = np.sum(analysis_pred)
                        if sum_analysis > 0:
                            analysis_pred = analysis_pred / sum_analysis
                            analysis_weight = 0.6  # Use 60% weight for analysis
                            print("✓ Using 60% weight from data analysis (hot numbers)")
                
                # If no hot numbers, try frequency data
                elif 'frequency' in self.pipeline_data['analysis_context'] and analysis_weight == 0:
                    frequency = self.pipeline_data['analysis_context']['frequency']
                    if frequency:
                        total_freq = sum(frequency.values())
                        if total_freq > 0:
                            for num, freq in frequency.items():
                                if 1 <= num <= self.num_classes:
                                    analysis_pred[num-1] = freq / total_freq
                            
                            analysis_weight = 0.6  # Use 60% weight for analysis
                            print("✓ Using 60% weight from data analysis (frequency)")
            
            # Combine predictions with appropriate weights
            if analysis_weight > 0:
                # Redistribute remaining 40% between prob and pattern
                prob_weight = 0.25
                pattern_weight = 0.15
                combined_pred = (prob_weight * prob_pred + 
                               pattern_weight * pattern_pred + 
                               analysis_weight * analysis_pred)
                print(f"Combined weights: {prob_weight:.2f} prob, {pattern_weight:.2f} pattern, {analysis_weight:.2f} analysis")
            else:
                # Fall back to original weights
                combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
                print("Using original weights: 0.4 prob, 0.6 pattern (no analysis data)")
                
            # Get top numbers and ensure uniqueness
            top_indices = np.argsort(combined_pred)[::-1]  # Sort in descending order
            final_numbers = []
            used_indices = set()
            
            # Select unique valid numbers
            for idx in top_indices:
                number = int(idx) + self.numbers_range[0]
                if len(final_numbers) >= self.numbers_to_draw:
                    break
                if number not in final_numbers and 1 <= number <= 80:
                    final_numbers.append(number)
                    used_indices.add(idx)
            
            # Validate we have enough numbers
            if len(final_numbers) != self.numbers_to_draw:
                print(f"ERROR: Could not generate {self.numbers_to_draw} unique valid numbers")
                return None, None, None
            
            # Sort final numbers
            final_numbers.sort()
            
            # Create probability mapping
            prob_map = {}
            for num in final_numbers:
                idx = num - self.numbers_range[0]
                if 0 <= idx < len(combined_pred):
                    prob_map[num] = float(combined_pred[idx])
                else:
                    prob_map[num] = 1.0 / self.numbers_to_draw
            
            # Create final probability array
            final_probs = [prob_map[num] for num in final_numbers]
            
            # Normalize final probabilities
            sum_probs = sum(final_probs)
            if sum_probs > 0:
                final_probs = [p/sum_probs for p in final_probs]
            else:
                final_probs = [1.0/len(final_numbers) for _ in final_numbers]
            
            # Store results in pipeline data
            self.pipeline_data.update({
                'final_prediction': final_numbers,
                'probabilities': final_probs,
                'prediction_metadata': {
                    'prob_weight': 0.4,
                    'pattern_weight': 0.6,
                    'combined_confidence': float(np.mean(final_probs)),
                    'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            })
            
            print("\nPrediction Summary:")
            print(f"Final Numbers: {final_numbers}")
            print(f"Number of predictions: {len(final_numbers)}")
            print(f"Number of probabilities: {len(final_probs)}")
            print(f"Average Confidence: {self.pipeline_data['prediction_metadata']['combined_confidence']:.4f}")
            
            return final_numbers, final_probs, self.pipeline_data.get('analysis_context', {})
                
        except Exception as e:
            print(f"Error in post-processing: {e}")
            traceback.print_exc()
            return None, None, None

    def save_models(self, path_prefix=None):
        """Save models with timestamp"""
        try:
            if path_prefix is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path_prefix = os.path.join(self.models_dir, f'lottery_predictor_{timestamp}')
            
            # Ensure models directory exists
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Save models
            model_files = {
                '_prob_model.pkl': self.probabilistic_model,
                '_pattern_model.pkl': self.pattern_model,
                '_scaler.pkl': self.scaler
            }
            
            # Save each model file
            for suffix, model in model_files.items():
                model_path = f'{path_prefix}{suffix}'
                joblib.dump(model, model_path)
                print(f"Saved model: {os.path.basename(model_path)}")
            
            # Update timestamp file - now in the models directory
            timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
            with open(timestamp_file, 'w') as f:
                f.write(timestamp)
            
            print(f"Models saved successfully in {self.models_dir}")
            return True
            
        except Exception as e:
            print(f"Error saving models: {e}")
            return False

    def load_models(self, path_prefix=None):
        """Enhanced model loading with validation"""
        try:
            if path_prefix is None:
                # First try to get path from timestamp file
                timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
                if os.path.exists(timestamp_file):
                    with open(timestamp_file, 'r') as f:
                        timestamp = f.read().strip()
                        path_prefix = os.path.join(self.models_dir, f'lottery_predictor_{timestamp}')
                else:
                    # Fallback to finding latest model file
                    model_files = glob.glob(os.path.join(self.models_dir, "*_prob_model.pkl"))
                    if not model_files:
                        raise FileNotFoundError("No models found in directory")
                    path_prefix = max(model_files, key=os.path.getctime).replace('_prob_model.pkl', '')
            
            # Validate all required files exist
            required_files = ['_prob_model.pkl', '_pattern_model.pkl', '_scaler.pkl']
            missing_files = []
            for file in required_files:
                if not os.path.exists(f"{path_prefix}{file}"):
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"Missing model files: {', '.join(missing_files)}")
            
            # Load models
            print(f"Loading models from: {path_prefix}")
            self.probabilistic_model = joblib.load(f'{path_prefix}_prob_model.pkl')
            self.pattern_model = joblib.load(f'{path_prefix}_pattern_model.pkl')
            self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')
            
            # Update status
            self.training_status.update({
                'model_loaded': True,
                'timestamp': datetime.fromtimestamp(
                    os.path.getctime(f'{path_prefix}_prob_model.pkl')
                ),
                'features': getattr(self.probabilistic_model, 'feature_names_in_', None)
            })
            
            print("Models loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.training_status['error'] = str(e)
            return False

    def train_and_predict(self, historical_data=None, recent_draws=None):
        """Enhanced prediction generation with analysis integration"""
        try:
            if historical_data is None:
                historical_data = self.load_data()
            
            if recent_draws is None:
                recent_draws = historical_data.tail(5)
            
            # Check if model needs training
            if not self.training_status['model_loaded']:
                print("\nTraining new model...")
                X, y = self.prepare_data(historical_data)
                
                # Add analysis features
                analysis_features = self.get_analysis_features(historical_data)
                
                # Train models
                self.train_models(X, y)
                self.save_models()
            
            # Generate prediction with analysis context
            predicted_numbers, probabilities, analysis_context = self.predict(recent_draws)
            
            # Get next draw time
            next_draw_time = datetime.now().replace(
                minute=(datetime.now().minute // 5 + 1) * 5,
                second=0, 
                microsecond=0
            )
            
            return predicted_numbers, probabilities, analysis_context
            
        except Exception as e:
            print(f"Error in train_and_predict: {e}")
            return None, None, None

    def train_models(self, data):
        """Train models with proper data preprocessing"""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
                
            print("\nProcessing training data...")
            
            # Create features and labels
            features = []
            labels = []
            
            window_size = 10  # Adjust as needed
            for i in range(len(data) - window_size):
                window = data.iloc[i:i + window_size]
                target_row = data.iloc[i + window_size]
                
                # Get feature vector
                feature_vector = self._create_feature_vector(window)
                if feature_vector is not None:
                    features.append(feature_vector)
                    
                    # Get target numbers from the next draw
                    target_cols = [col for col in data.columns if col.startswith('number')]
                    target_numbers = target_row[target_cols].values.astype(float)
                    valid_targets = target_numbers[(target_numbers >= 1) & (target_numbers <= 80)]
                    labels.extend(valid_targets)
            
            if not features:
                raise ValueError("No valid features generated")
                
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(labels)
            
            print(f"\nTraining data prepared:")
            print(f"- Features: {X.shape[1]}")
            print(f"- Classes represented: {len(np.unique(y))}")
            
            # Continue with model training...
            
        except Exception as e:
            print(f"Error in model training: {e}")
            return False

    def predict(self, recent_draws):
        """Enhanced prediction with pipeline execution"""
        pipeline_tracking = {
            'start_time': datetime.now(),
            'stages_completed': [],
            'current_stage': None,
            'error': None
        }
        try:
            # Check if models are trained
            if self.probabilistic_model is None or not hasattr(self.probabilistic_model, 'class_prior_'):
                raise ValueError("Models not properly trained. Please train models first.")
            is_valid, message = self.validate_model_state()
            if not is_valid:

                print(f"Warning: {message}")  # Log the warning but don't fail
                print("Attempting to continue with existing validation...")
            try:
                # Ensure pipeline is initialized
                if not hasattr(self, 'pipeline_stages'):
                    self._initialize_pipeline()
            except Exception as e:
                print(f"Error initializing pipeline: {e}")
                raise
            
            # Validate input data
            if recent_draws is None:
                raise ValueError("No input data provided for prediction")
                
      
                
            # Run prediction pipeline
            result = recent_draws
            for stage_name, stage_func in self.pipeline_stages.items():
                print(f"\nExecuting pipeline stage: {stage_name}")
                pipeline_tracking['current_stage'] = stage_name
                
                # Execute stage with timing
                stage_start = datetime.now()
                result = stage_func(result)
                stage_duration = (datetime.now() - stage_start).total_seconds()
                
                # Validate stage result
                if result is None:
                    raise ValueError(f"Pipeline stage {stage_name} failed")
                    
                # Update tracking
                pipeline_tracking['stages_completed'].append({
                    'stage': stage_name,
                    'duration': stage_duration,
                    'success': True
                })
                print(f"Stage {stage_name} completed in {stage_duration:.2f} seconds")
            
            # Get final predictions and analysis
            final_numbers = self.pipeline_data.get('final_prediction')
            probabilities = self.pipeline_data.get('probabilities')
            analysis_context = self.pipeline_data.get('analysis_context', {})
            
            # Additional validation for probabilities
            if isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()
            
            # Ensure we have the correct number of predictions
            if final_numbers is None or len(final_numbers) != self.numbers_to_draw:
                raise ValueError("Invalid prediction results")
            
            # Store pipeline execution metadata
            self.pipeline_data['pipeline_execution'] = {
                'execution_time': (datetime.now() - pipeline_tracking['start_time']).total_seconds(),
                'stages': pipeline_tracking['stages_completed'],
                'timestamp': datetime.now()
            }
            
            print("\nPipeline execution completed successfully")
            print(f"Total execution time: {self.pipeline_data['pipeline_execution']['execution_time']:.2f} seconds")
            
            return final_numbers, probabilities, analysis_context
            
        except Exception as e:
            error_msg = f"Error in prediction pipeline: {str(e)}"
            print(error_msg)
            
            # Update pipeline data with error information
            self.pipeline_data['error'] = {
                'message': error_msg,
                'stage': pipeline_tracking.get('current_stage'),
                'timestamp': datetime.now()
            }
            
            return None, None, None


    def prepare_feature_columns(self, data):
        """Ensure all required feature columns exist"""
        try:
            if self.training_status['model_loaded'] and hasattr(self.probabilistic_model, 'feature_names_in_'):
                required_features = self.probabilistic_model.feature_names_in_
                for col in required_features:
                    if col not in data.columns:
                        data[col] = 0
                return data[required_features]
            return data
        except Exception as e:
            print(f"Error preparing feature columns: {e}")
            return data

    def _process_raw_data(self, data):
        """Process raw DataFrame to extract numbers"""
        try:
            # Get number columns (expecting 'number1' through 'number20')
            number_cols = [col for col in data.columns if col.startswith('number')]
            
            # Convert DataFrame to list of draws
            processed_draws = []
            for _, row in data.iterrows():
                # Extract numbers and convert to integers
                draw_numbers = []
                for col in number_cols:
                    try:
                        # Remove 'number' prefix and convert to int
                        value = row[col]
                        if isinstance(value, str) and value.startswith('number'):
                            # Skip column names that got into data
                            continue
                        num = int(value)
                        if 1 <= num <= 80:
                            draw_numbers.append(num)
                    except (ValueError, TypeError):
                        continue
                
                if draw_numbers:  # Only add valid draws
                    processed_draws.append(draw_numbers)
            
            if not processed_draws:
                raise ValueError("No valid draws found in data")
                
            print(f"Processed {len(processed_draws)} valid draws")
            return processed_draws

        except Exception as e:
            print(f"Error processing raw data: {e}")
            return None

if __name__ == "__main__":
    predictor = LotteryPredictor()
    
    try:
        # Load historical data
        historical_data = predictor.load_data()
        if historical_data is None:
            raise ValueError("Failed to load historical data")
        
        # Generate prediction
        prediction, probabilities, analysis = predictor.train_and_predict(
            historical_data=historical_data
        )
        
        if prediction is not None:
            print("\n=== Prediction Results ===")
            print(f"Predicted numbers: {sorted(prediction)}")
            print("\n=== Analysis Context ===")
            for key, value in analysis.items():
                print(f"{key}: {value}")
            
            # Save prediction
            predictor.save_prediction_to_csv(prediction, probabilities)
        
    except Exception as e:
        print(f"\nError in main execution: {e}")