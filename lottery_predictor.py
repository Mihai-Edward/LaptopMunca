import json
import pickle
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
from data_analysis import DataAnalysis
from datetime import datetime, timedelta
import os
import glob
from sklearn.model_selection import train_test_split
import sys
from config.paths import PATHS, ensure_directories


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        # Fix: Use PREDICTIONS_DIR instead of non-existent PREDICTIONS key
        self.predictions_dir = PATHS['PREDICTIONS_DIR']  # Use the correct key
        
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
           'use_combined_features': use_combined_features,
           'model_config': {
            'use_combined_features': use_combined_features
            }
        }
        self.use_combined_features = use_combined_features
        # Initialize pipeline stages
        self._initialize_pipeline()
        
        print(f"\nInitialized LotteryPredictor:")
        print(f"- Number range: {numbers_range}")
        print(f"- Numbers to draw: {numbers_to_draw}")
        print(f"- Number of classes: {self.num_classes}")
        print(f"- Using combined features: {use_combined_features}")
        
    def _validate_data(self, data):
        """Validate input data format"""
        try:
            required_cols = ['date'] + [f'number{i}' for i in range(1, 21)]
            
            # Check columns exist
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Validate date format - Now handles double spaces correctly
            try:
                # Clean date strings
                data['date'] = data['date'].astype(str).str.strip()
                # Keep the double space format from historical data
                data['date'] = pd.to_datetime(data['date'], format='%H:%M  %d-%m-%Y')
            except Exception as e:
                print(f"Warning: Date parsing error with double space format: {e}")
                try:
                    # Fallback to single space format
                    data['date'] = pd.to_datetime(data['date'], format='%H:%M %d-%m-%Y')
                except:
                    raise ValueError("Invalid date format in data")
                
            # Validate numbers
            number_cols = [f'number{i}' for i in range(1, 21)]
            for col in number_cols:
                invalid_nums = data[~data[col].between(1, 80)]
                if not invalid_nums.empty:
                    raise ValueError(f"Invalid numbers found in column {col}")
                    
            return True
            
        except Exception as e:
            print(f"Data validation error: {e}")
            return False
            
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
                use_combined = self.pipeline_data.get('use_combined_features', True)
                
                # When using combined features, we expect 84 base + 80 analysis = 164 features
                if use_combined and expected_features != 164:
                    return False, f"Invalid feature dimension for combined model: expected 164, got {expected_features}"
                elif not use_combined and expected_features != 84:
                    return False, f"Invalid feature dimension for base features: expected 84, got {expected_features}"

            # Check feature dimension consistency between models
            if hasattr(self.probabilistic_model, 'n_features_in_') and \
               hasattr(self.pattern_model, 'n_features_in_'):
                if self.probabilistic_model.n_features_in_ != self.pattern_model.n_features_in_:
                    return False, "Feature dimension mismatch between models"

            # Verify training status
            if not self.training_status.get('model_loaded', False):
                return False, "Models not marked as loaded"

            # Verify feature dimension in training status
            model_dim = self.training_status.get('model_config', {}).get('feature_dimension')
            if model_dim is not None and model_dim != expected_features:
                return False, f"Feature dimension mismatch with training status: {model_dim} vs {expected_features}"

            return True, "Model state valid"
        except Exception as e:
            return False, f"Model validation error: {str(e)}"
    def _initialize_pipeline(self):
        """Initialize prediction pipeline with enhanced stages and validation"""
        try:
            print("\nInitializing prediction pipeline...")
            
            # Preserve the existing use_combined_features value
            use_combined = getattr(self, 'use_combined_features', True)
            
            # Define ordered pipeline stages with better structure
            self.pipeline_stages = OrderedDict({
                'data_preparation': {
                    'function': self._prepare_pipeline_data,
                    'description': 'Prepare and validate input data',
                    'required_inputs': ['date', 'number1', 'number2', '...', 'number20'],
                    'outputs': ['prepared_data']
                },
                'feature_engineering': {
                    'function': self._create_enhanced_features,
                    'description': 'Create feature vectors from prepared data',
                    'required_inputs': ['prepared_data'],
                    'outputs': ['features', 'analysis_features']
                },
                'model_prediction': {
                    'function': self._generate_model_predictions,
                    'description': 'Generate model predictions',
                    'required_inputs': ['features'],
                    'outputs': ['prob_predictions', 'pattern_predictions']
                },
                'post_processing': {
                    'function': self._post_process_predictions,
                    'description': 'Process and combine predictions',
                    'required_inputs': ['prob_predictions', 'pattern_predictions'],
                    'outputs': ['final_numbers', 'probabilities', 'analysis_context']
                }
            })
            
            # Create new pipeline data while preserving existing settings
            new_pipeline_data = {
                'pipeline_config': {
                    'initialized_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'use_combined_features': use_combined,  # Use preserved value
                    'stages': list(self.pipeline_stages.keys())
                },
                'use_combined_features': use_combined,  # Keep original setting
                'execution_context': {
                    'current_stage': None,
                    'completed_stages': [],
                    'errors': [],
                    'warnings': []
                },
                'stage_results': {},
                'metadata': {
                    'model_config': {
                        'num_classes': self.num_classes,
                        'numbers_to_draw': self.numbers_to_draw,
                        'feature_dimension': None,  # Will be set during execution
                        'use_combined_features': use_combined  # Add here too
                    },
                    'runtime_config': {
                        'use_analysis': True,
                        'analysis_weight': 0.6,
                        'prob_weight': 0.25,
                        'pattern_weight': 0.15
                    }
                }
            }
            
            # Update existing pipeline data instead of overwriting
            if hasattr(self, 'pipeline_data'):
                # Preserve any existing data
                existing_data = self.pipeline_data.copy()
                # Update with new configuration while preserving use_combined_features
                existing_data.update(new_pipeline_data)
                self.pipeline_data = existing_data
            else:
                # Initialize if not exists
                self.pipeline_data = new_pipeline_data
            
            # Convert stage definitions to actual functions
            self.pipeline_stages = OrderedDict({
                name: stage['function'] 
                for name, stage in self.pipeline_stages.items()
            })
            
            print("Pipeline initialization complete with stages:")
            for stage_name in self.pipeline_stages.keys():
                print(f"- {stage_name}")
            
            # Debug info
            print(f"\nPipeline configuration:")
            print(f"- Using combined features: {use_combined}")
            print(f"- Feature dimension will be: {164 if use_combined else 84}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            if hasattr(self, 'pipeline_data'):
                self.pipeline_data['errors'] = [{
                    'stage': 'initialization',
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }]
            traceback.print_exc()
            return False
    
    def _prepare_pipeline_data(self, data):
        """First pipeline stage: Prepare and validate input data with enhanced checks"""
        print("\nPreparing data for prediction pipeline...")
        try:
            # Input validation
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            if len(data) < 5:
                raise ValueError("Need at least 5 recent draws for prediction")
                
            # Deep copy to avoid modifying original data
            prepared_data = data.copy()
            
            # Ensure all required columns exist
            required_cols = ['date'] + [f'number{i}' for i in range(1, 21)]
            missing_cols = [col for col in required_cols if col not in prepared_data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            # Clean and validate date column
            try:
                prepared_data['date'] = pd.to_datetime(prepared_data['date'])
                prepared_data = prepared_data.sort_values('date').reset_index(drop=True)
            except Exception as e:
                print(f"Warning: Date conversion error: {e}")
                print("Attempting to proceed without date sorting...")
                
            # Validate and clean number columns
            number_cols = [f'number{i}' for i in range(1, 21)]
            for col in number_cols:
                # Convert to numeric, coerce errors to NaN
                prepared_data[col] = pd.to_numeric(prepared_data[col], errors='coerce')
                
                # Check for invalid numbers
                invalid_mask = ~prepared_data[col].between(1, 80)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    print(f"Warning: Found {invalid_count} invalid numbers in {col}")
                    # Fill invalid numbers with the mode of valid numbers
                    valid_mode = prepared_data.loc[~invalid_mask, col].mode()
                    if not valid_mode.empty:
                        prepared_data.loc[invalid_mask, col] = valid_mode[0]
                    else:
                        raise ValueError(f"No valid numbers found in column {col}")
            
            # Sort numbers within each draw
            for idx in prepared_data.index:
                numbers = sorted(prepared_data.loc[idx, number_cols])
                for i, num in enumerate(numbers):
                    prepared_data.loc[idx, f'number{i+1}'] = num
            
            # Get most recent 5 draws
            prepared_data = prepared_data.tail(12).reset_index(drop=True)
            
            # Add time-based features
            if 'date' in prepared_data.columns:
                prepared_data['day_of_week'] = prepared_data['date'].dt.dayofweek
                prepared_data['month'] = prepared_data['date'].dt.month
                prepared_data['day'] = prepared_data['date'].dt.day
                prepared_data['hour'] = prepared_data['date'].dt.hour
                prepared_data['minute'] = prepared_data['date'].dt.minute
            
            # Store metadata in pipeline data
            self.pipeline_data['prepared_data'] = prepared_data
            self.pipeline_data['data_preparation_metadata'] = {
                'original_rows': len(data),
                'prepared_rows': len(prepared_data),
                'date_range': {
                    'start': prepared_data['date'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': prepared_data['date'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'number_stats': {
                    col: {
                        'min': float(prepared_data[col].min()),
                        'max': float(prepared_data[col].max()),
                        'mean': float(prepared_data[col].mean())
                    } for col in number_cols
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print("\nData preparation completed:")
            print(f"- Processed {len(prepared_data)} draws")
            print(f"- Time range: {self.pipeline_data['data_preparation_metadata']['date_range']['start']} to "
                  f"{self.pipeline_data['data_preparation_metadata']['date_range']['end']}")
            
            return prepared_data
            
        except Exception as e:
            print(f"Error in data preparation: {e}")
            if 'pipeline_data' in self.__dict__:
                self.pipeline_data['errors'] = self.pipeline_data.get('errors', []) + [{
                    'stage': 'data_preparation',
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }]
            traceback.print_exc()
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
            
            # Clean and convert dates - Now handles double spaces correctly
            try:
                # Clean date strings but preserve double spaces
                df['date'] = df['date'].astype(str).str.strip()
                # Try double space format first (matches historical data)
                df['date'] = pd.to_datetime(df['date'], format='%H:%M  %d-%m-%Y')
                print("DEBUG: Successfully parsed dates with double space format")
            except Exception as e:
                print(f"Warning: Double space date parsing failed: {e}")
                try:
                    # Fallback to single space format
                    df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y')
                    print("DEBUG: Successfully parsed dates with single space format")
                except Exception as e2:
                    print(f"Error: All date parsing attempts failed: {e2}")
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
            if historical_data is None or len(historical_data) < 13:
                raise ValueError("Insufficient historical data (minimum 6 draws required)")

            # Sort chronologically and reset index
            historical_data = historical_data.sort_values('date').reset_index(drop=True)
            
            features = []
            labels = []
            window_size = 12  # Use last 5 draws to predict next
            number_cols = [f'number{i}' for i in range(1, 21)]

            print(f"Processing {len(historical_data) - window_size} potential training samples...")

            # Create sliding windows
            for i in range(len(historical_data) - window_size):
                try:
                    # Get window of 5 consecutive draws
                    window = historical_data.iloc[i:i+window_size]
                    
                    # Get base features (80-dimensional frequency vector)
                    frequency_vector = np.zeros(80)
                    for _, row in window.iterrows():
                        for col in number_cols:
                            num = int(row[col])
                            if 1 <= num <= 80:
                                frequency_vector[num-1] += 1

                    # Normalize frequency vector
                    total = np.sum(frequency_vector)
                    if total > 0:
                        frequency_vector = frequency_vector / total

                    # Get target draw (for labels)
                    target_draw = historical_data.iloc[i+window_size]
                    
                    # Add time-based features
                    target_date = pd.to_datetime(target_draw['date'])
                    time_features = np.array([
                        target_date.hour,
                        target_date.minute,
                        target_date.dayofweek,
                        target_date.day
                    ])

                    # Create base feature vector
                    base_features = np.concatenate([frequency_vector, time_features])
                    
                    # Get analysis features for the window
                    try:
                        # Convert window data to format expected by DataAnalysis
                        window_draws = [(row['date'], 
                                       [int(row[f'number{j}']) for j in range(1, 21)]) 
                                      for _, row in window.iterrows()]
                        
                        analyzer = DataAnalysis(window_draws)
                        
                        # Get frequency and hot-cold analysis
                        freq = analyzer.count_frequency()
                        hot_nums, _ = analyzer.hot_and_cold_numbers()
                        
                        # Create analysis feature vector (80 dimensions)
                        analysis_features = np.zeros(80)
                        
                        # Fill with frequency data
                        for num, count in freq.items():
                            if 1 <= num <= 80:
                                analysis_features[num-1] = count / len(window)
                                
                        # Normalize analysis features
                        if np.sum(analysis_features) > 0:
                            analysis_features = analysis_features / np.sum(analysis_features)
                    except Exception as e:
                        print(f"Warning: Error generating analysis features for window {i}: {e}")
                        analysis_features = np.zeros(80)  # Fallback to zeros if analysis fails

                    # Combine base and analysis features if enabled
                    if self.pipeline_data.get('use_combined_features', True):
                        feature_vector = np.concatenate([base_features, analysis_features])
                    else:
                        feature_vector = base_features
                    
                    # Create label - Use the numbers as individual labels
                    for num in [int(target_draw[col]) for col in number_cols]:
                        if 1 <= num <= 80:
                            features.append(feature_vector)
                            labels.append(num - 1)  # Convert to 0-based index for classification
                            self.label_mapping = {i: i+1 for i in range(80)}
                except Exception as e:
                    print(f"Warning: Error processing window {i}: {e}")
                    continue

            if not features:
                raise ValueError("No valid training samples could be generated")

            # Convert to numpy arrays
            features = np.array(features)
            labels = np.array(labels)

            print(f"\nTraining data preparation complete:")
            print(f"- Number of samples: {len(features)}")
            print(f"- Feature vector shape: {features.shape}")
            print(f"- Label shape: {labels.shape}")
            
            # Store feature dimension in training status
            self.training_status['model_config']['feature_dimension'] = features.shape[1]

            return features, labels

        except Exception as e:
            print(f"Error in data preparation: {e}")
            traceback.print_exc()
            return None, None

    def _create_feature_vector(self, window, target_date=None):
        """Create standardized feature vector with time features"""
        try:
            # Validate input
            if not isinstance(window, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            # Get number columns
            number_cols = [f'number{i}' for i in range(1, 21)]
            if not all(col in window.columns for col in number_cols):
                raise ValueError("Missing required number columns")
                
            # Create frequency vector (80 dimensions)
            frequency_vector = np.zeros(80)
            
            # Count frequencies from all draws in window
            for _, row in window.iterrows():
                numbers = row[number_cols].values.astype(float)
                valid_numbers = numbers[(numbers >= 1) & (numbers <= 80)]
                for num in valid_numbers:
                    frequency_vector[int(num)-1] += 1  # Convert to 0-based index
                    
            # Normalize frequency vector
            total_numbers = np.sum(frequency_vector)
            if total_numbers > 0:
                frequency_vector = frequency_vector / total_numbers
                
            # Always include time features to maintain consistent 84-dimensional vector
            if target_date is None:
                # Use the last date from the window if target_date not provided
                target_date = pd.to_datetime(window['date'].iloc[-1])
            elif isinstance(target_date, str):
                target_date = pd.to_datetime(target_date)
                    
            time_features = np.array([
                target_date.hour,
                target_date.minute,
                target_date.dayofweek,
                target_date.day
            ])
            
            # Combine frequency and time features
            feature_vector = np.concatenate([frequency_vector, time_features])
            
            # Validate final dimension
            if len(feature_vector) != 84:  # 80 numbers + 4 time features
                raise ValueError(f"Invalid feature vector dimension: expected 84, got {len(feature_vector)}")
                
            print(f"Created feature vector with shape: {feature_vector.shape}")
            
            # Store feature metadata
            self.pipeline_data['latest_feature_vector'] = {
                'frequency_dims': 80,
                'time_dims': 4,
                'total_dims': len(feature_vector),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            return feature_vector
            
        except Exception as e:
            print(f"Error creating feature vector: {e}")
            traceback.print_exc()
            return None

    def _create_analysis_features(self, data):
        """Create enhanced features from data analysis with improved integration"""
        print("\nGenerating analysis features...")
        
        try:
            # Format data for analysis
            formatted_draws = []
            for _, row in data.iterrows():
                try:
                    # Get numbers and validate
                    numbers = []
                    for i in range(1, 21):
                        num = row.get(f'number{i}')
                        if isinstance(num, (int, float)) and 1 <= num <= 80:
                            numbers.append(int(num))
                    
                    if len(numbers) == 20:  # Only use complete draws
                        formatted_draws.append((row['date'], sorted(numbers)))
                except Exception as e:
                    print(f"Warning: Skipping draw due to error: {e}")
                    continue

            if not formatted_draws:
                raise ValueError("No valid draws for analysis")

            # Initialize or update analyzer
            if self.analyzer is None:
                self.analyzer = DataAnalysis(formatted_draws)
            
            # Get analysis results
            frequency = self.analyzer.count_frequency()
            hot_numbers, cold_numbers = self.analyzer.hot_and_cold_numbers()
            common_pairs = self.analyzer.find_common_pairs()
            range_analysis = self.analyzer.number_range_analysis()
            
            # Create feature arrays
            analysis_features = np.zeros(160)  # 80 for frequency + 80 for hot/cold
            
            # Fill frequency features (first 80 dimensions)
            total_freq = sum(frequency.values()) or 1  # Avoid division by zero
            for num, freq in frequency.items():
                if 1 <= num <= 80:
                    analysis_features[num-1] = freq / total_freq
            
            # Fill hot/cold features (second 80 dimensions)
            hot_nums = dict(hot_numbers)
            max_hot_score = max(hot_nums.values()) if hot_nums else 1
            for num, score in hot_nums.items():
                if 1 <= num <= 80:
                    analysis_features[80 + num-1] = score / max_hot_score
            
            # Store analysis context with metadata
            self.pipeline_data['analysis_context'] = {
                'frequency': frequency,
                'hot_cold': (hot_numbers, cold_numbers),
                'common_pairs': common_pairs,
                'range_analysis': range_analysis,
                'feature_stats': {
                    'total_frequency': total_freq,
                    'max_hot_score': max_hot_score,
                    'feature_range': (float(np.min(analysis_features)), float(np.max(analysis_features))),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
            # Print analysis summary
            print(f"\nAnalysis features generated:")
            print(f"- Total features: {len(analysis_features)}")
            print(f"- Frequency features: 80")
            print(f"- Hot/Cold features: 80")
            print(f"- Feature range: {np.min(analysis_features):.4f} to {np.max(analysis_features):.4f}")
            
            return analysis_features
            
        except Exception as e:
            print(f"Error in analysis features generation: {e}")
            traceback.print_exc()
            return np.zeros(160)  # Return zero vector as fallback

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
                use_combined = getattr(self, 'use_combined_features', True)
                
                if use_combined:
                 
                    try:
                        # Take the first 20 analysis features (most important ones) to avoid dimensionality issues
                        enhanced_features = np.concatenate([base_features, analysis_features[:80]])  # Use first 80 analysis features
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
        """Generate model predictions with enhanced validation and combination"""
        print("\nGenerating model predictions...")
        try:
            # Validate features input
            if features is None:
                raise ValueError("No features provided for prediction")
                
            if not isinstance(features, np.ndarray):
                try:
                    features = np.array(features)
                except Exception as e:
                    raise ValueError(f"Could not convert features to numpy array: {e}")
            
            # Ensure features are 2D
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
                
            # Validate feature dimensions
            expected_dims = self.training_status.get('feature_dimension')
            if expected_dims and features.shape[1] != expected_dims:
                raise ValueError(f"Feature dimension mismatch: expected {expected_dims}, got {features.shape[1]}")
            
            print(f"Processing features shape: {features.shape}")
            
            # Scale features
            try:
                scaled_features = self.scaler.transform(features)
            except Exception as e:
                raise ValueError(f"Feature scaling failed: {e}")
            
            # Initialize prediction arrays
            prob_pred = np.zeros(80)  # For all possible numbers (1-80)
            pattern_pred = np.zeros(80)
            
            # Generate probabilistic model predictions
            try:
                print("Generating probabilistic model predictions...")
                prob_raw = self.probabilistic_model.predict_proba(scaled_features)[0]
                prob_pred[:len(prob_raw)] = prob_raw
                print(f"- Probabilistic predictions shape: {prob_pred.shape}")
            except Exception as e:
                print(f"Warning: Probabilistic model prediction failed: {e}")
                # Use uniform distribution as fallback
                prob_pred = np.ones(80) / 80
            
            # Generate pattern model predictions
            try:
                print("Generating pattern model predictions...")
                pattern_raw = self.pattern_model.predict_proba(scaled_features)[0]
                pattern_pred[:len(pattern_raw)] = pattern_raw
                print(f"- Pattern predictions shape: {pattern_pred.shape}")
            except Exception as e:
                print(f"Warning: Pattern model prediction failed: {e}")
                # Use uniform distribution as fallback
                pattern_pred = np.ones(80) / 80
            
            # Validate predictions
            if np.any(np.isnan(prob_pred)) or np.any(np.isnan(pattern_pred)):
                raise ValueError("NaN values detected in predictions")
                
            # Normalize predictions
            prob_pred = prob_pred / np.sum(prob_pred)
            pattern_pred = pattern_pred / np.sum(pattern_pred)
            
            # Store prediction metadata
            self.pipeline_data['model_predictions'] = {
                'probabilistic': {
                    'raw_shape': prob_raw.shape if 'prob_raw' in locals() else None,
                    'normalized_shape': prob_pred.shape,
                    'min': float(np.min(prob_pred)),
                    'max': float(np.max(prob_pred)),
                    'mean': float(np.mean(prob_pred))
                },
                'pattern': {
                    'raw_shape': pattern_raw.shape if 'pattern_raw' in locals() else None,
                    'normalized_shape': pattern_pred.shape,
                    'min': float(np.min(pattern_pred)),
                    'max': float(np.max(pattern_pred)),
                    'mean': float(np.mean(pattern_pred))
                },
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print("\nPrediction Statistics:")
            print("Probabilistic Model:")
            print(f"- Min: {self.pipeline_data['model_predictions']['probabilistic']['min']:.4f}")
            print(f"- Max: {self.pipeline_data['model_predictions']['probabilistic']['max']:.4f}")
            print(f"- Mean: {self.pipeline_data['model_predictions']['probabilistic']['mean']:.4f}")
            print("Pattern Model:")
            print(f"- Min: {self.pipeline_data['model_predictions']['pattern']['min']:.4f}")
            print(f"- Max: {self.pipeline_data['model_predictions']['pattern']['max']:.4f}")
            print(f"- Mean: {self.pipeline_data['model_predictions']['pattern']['mean']:.4f}")
            
            return prob_pred, pattern_pred
            
        except Exception as e:
            print(f"Error in model prediction generation: {e}")
            if 'pipeline_data' in self.__dict__:
                self.pipeline_data['errors'] = self.pipeline_data.get('errors', []) + [{
                    'stage': 'model_prediction',
                    'error': str(e),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }]
            traceback.print_exc()
            return None, None

    def _post_process_predictions(self, predictions):
        """Process and combine predictions with enhanced validation and analysis"""
        print("\nPost-processing predictions...")
        try:
            # Validate predictions input
            if (predictions is None or 
                not isinstance(predictions, (list, tuple)) or 
                len(predictions) != 2 or 
                any(p is None for p in predictions)):
                raise ValueError("Invalid predictions input format")
                
            prob_pred, pattern_pred = predictions
            
            # Convert to numpy arrays if needed
            prob_pred = np.array(prob_pred) if isinstance(prob_pred, list) else prob_pred
            pattern_pred = np.array(pattern_pred) if isinstance(pattern_pred, list) else pattern_pred
            
            # Validate array shapes
            if prob_pred.shape != (self.num_classes,) or pattern_pred.shape != (self.num_classes,):
                raise ValueError(f"Invalid prediction shapes - prob: {prob_pred.shape}, pattern: {pattern_pred.shape}")
            
            # Normalize predictions
            prob_pred = prob_pred / np.sum(prob_pred)
            pattern_pred = pattern_pred / np.sum(pattern_pred)
            
            # Initialize analysis prediction
            analysis_pred = np.zeros(self.num_classes)
            analysis_weight = 0.0
            
            # Try to incorporate analysis data
            if 'analysis_context' in self.pipeline_data:
                if 'hot_cold' in self.pipeline_data['analysis_context']:
                    hot_cold = self.pipeline_data['analysis_context']['hot_cold']
                    if isinstance(hot_cold, tuple) and len(hot_cold) > 0:
                        hot_numbers = hot_cold[0]
                        
                        # Create weights from hot numbers - using label mapping
                        for num, count in hot_numbers:
                            if 1 <= num <= self.num_classes:
                                # Convert to 0-based index for internal processing
                                analysis_pred[num-1] = count
                        
                        # Normalize analysis predictions
                        sum_analysis = np.sum(analysis_pred)
                        if sum_analysis > 0:
                            analysis_pred = analysis_pred / sum_analysis
                            analysis_weight = 0.6
                            print("Using 60% weight from hot numbers analysis")
                
                # Fallback to frequency data if no hot numbers
                elif 'frequency' in self.pipeline_data['analysis_context'] and analysis_weight == 0:
                    frequency = self.pipeline_data['analysis_context']['frequency']
                    if frequency:
                        total_freq = sum(frequency.values())
                        if total_freq > 0:
                            for num, freq in frequency.items():
                                if 1 <= num <= self.num_classes:
                                    # Convert to 0-based index for internal processing
                                    analysis_pred[num-1] = freq / total_freq
                            analysis_weight = 0.6
                            print("Using 60% weight from frequency analysis")
            
            # Combine predictions with weights
            if analysis_weight > 0:
                prob_weight = 0.25
                pattern_weight = 0.15
                combined_pred = (prob_weight * prob_pred + 
                               pattern_weight * pattern_pred + 
                               analysis_weight * analysis_pred)
                print(f"Combined weights: {prob_weight:.2f} prob, {pattern_weight:.2f} pattern, {analysis_weight:.2f} analysis")
            else:
                combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
                print("Using default weights: 0.4 prob, 0.6 pattern")
            
            # Get top numbers ensuring uniqueness
            top_indices = np.argsort(combined_pred)[::-1]
            final_numbers = []
            used_indices = set()
            
            # Select unique valid numbers and map back to 1-80 range
            for idx in top_indices:
                if len(final_numbers) >= self.numbers_to_draw:
                    break
                # Convert to 1-based number using label mapping
                number = idx + 1  # Map back to 1-80 range
                if number not in final_numbers and 1 <= number <= 80:
                    final_numbers.append(number)
                    used_indices.add(idx)
            
            # Validate final numbers
            if len(final_numbers) != self.numbers_to_draw:
                raise ValueError(f"Could not generate {self.numbers_to_draw} unique valid numbers")
            
            # Sort final numbers
            final_numbers.sort()
            
            # Create probability mapping using 1-based numbers
            prob_map = {num: float(combined_pred[num-1]) for num in final_numbers}
            
            # Create final probability array
            final_probs = [prob_map[num] for num in final_numbers]
            
            # Normalize final probabilities with safety check
            sum_probs = sum(final_probs)
            if sum_probs > 0:
                final_probs = [p / sum_probs for p in final_probs]
                print(f"DEBUG: Final probabilities normalized, sum = {sum(final_probs):.4f}")
            else:
                # If for some reason the sum is 0, assign equal probability to each number
                final_probs = [1.0 / len(final_numbers) for _ in final_numbers]
                print("DEBUG: Final probabilities were 0; assigned equal probabilities to each predicted number")
            
            # Store results in pipeline data
            self.pipeline_data.update({
                'final_prediction': final_numbers,
                'probabilities': final_probs,
                'prediction_metadata': {
                    'prob_weight': 0.4 if analysis_weight == 0 else 0.25,
                    'pattern_weight': 0.6 if analysis_weight == 0 else 0.15,
                    'analysis_weight': analysis_weight,
                    'combined_confidence': float(np.mean(final_probs)),
                    'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            })
            
            # Get next_draw_time with safety check
            next_draw_time_value = self.pipeline_data.get('next_draw_time')
            if isinstance(next_draw_time_value, datetime):
                formatted_time = next_draw_time_value.strftime('%Y-%m-%d %H:%M:%S')
            else:
                formatted_time = next_draw_time_value  # Assume it's already a string

            # Then update pipeline_data safely with the formatted next draw time
            self.pipeline_data.update({
                'latest_prediction': {
                    'numbers': final_numbers,
                    'probabilities': final_probs,  # Use final_probs instead of prob_map
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),     # Ensure timestamp is defined as needed
                    'next_draw_time': formatted_time
                }
            })
            
            print("\nPrediction Summary:")
            print(f"Final Numbers: {final_numbers}")
            print(f"Average Confidence: {self.pipeline_data['prediction_metadata']['combined_confidence']:.4f}")
            
            return final_numbers, final_probs, self.pipeline_data.get('analysis_context', {})
            
        except Exception as e:
            print(f"Error in post-processing: {e}")
            traceback.print_exc()
            return None, None, None

    def save_models(self, path_prefix=None, base_path=None, custom_suffix=None):
        """Save models with enhanced timestamp and path handling"""
        try:
            # Get current timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Handle path parameters to maintain compatibility with existing calls
            if path_prefix is not None:
                # If path_prefix is provided, use it directly
                final_path = path_prefix
            else:
                # Otherwise use base_path and custom_suffix logic
                if base_path is None:
                    base_path = self.models_dir
                
                if custom_suffix is None:
                    # Use timestamp as default suffix
                    final_path = os.path.join(base_path, f'lottery_predictor_{timestamp}')
                else:
                    # Support custom suffix parameter
                    final_path = os.path.join(base_path, f'lottery_predictor_{custom_suffix}')
            
            # Ensure models directory exists
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            
            # Save models with consistent naming
            model_files = {
                '_prob_model.pkl': self.probabilistic_model,
                '_pattern_model.pkl': self.pattern_model,
                '_scaler.pkl': self.scaler
            }
            
            saved_files = []
            for suffix, model in model_files.items():
                model_path = f'{final_path}{suffix}'
                try:
                    joblib.dump(model, model_path)
                    saved_files.append(os.path.basename(model_path))
                except Exception as e:
                    print(f"Error saving {suffix}: {e}")
                    # Clean up partially saved files
                    for saved_file in saved_files:
                        try:
                            os.remove(os.path.join(os.path.dirname(final_path), saved_file))
                        except:
                            pass
                    return None  # Match original return type on error
            
            # Save timestamp and metadata
            try:
                # Extract timestamp from the final path
                path_timestamp = os.path.basename(final_path).replace('lottery_predictor_', '')
                timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
                with open(timestamp_file, 'w') as f:
                    f.write(path_timestamp)
                
                # Save additional metadata
                metadata_file = os.path.join(self.models_dir, f'model_metadata_{path_timestamp}.json')
                metadata = {
                    'timestamp': path_timestamp,
                    'model_version': getattr(self, 'version', '1.0'),
                    'feature_dimension': self.training_status.get('model_config', {}).get('feature_dimension'),
                    'use_combined_features': getattr(self, 'use_combined_features', True),
                    'saved_files': saved_files
                }
                
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
            
            except Exception as e:
                print(f"Warning: Error saving metadata: {e}")
                # Continue if metadata saving fails
            
            # Store the timestamp for possible later use
            self.last_save_timestamp = path_timestamp
            
            print(f"Models saved successfully in {os.path.dirname(final_path)}")
            print(f"Saved files: {', '.join(saved_files)}")
            
            # Return timestamp to maintain backward compatibility
            return path_timestamp
            
        except Exception as e:
            print(f"Error saving models: {e}")
            traceback.print_exc()
            return None

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
            
        print("\n=== Training State Debug Info ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Historical data rows: {len(historical_data) if historical_data is not None else 'None'}")
        print(f"Recent draws rows: {len(recent_draws) if recent_draws is not None else 'None'}")
        print(f"Model loaded: {self.training_status.get('model_loaded')}")
        print("==============================\n")
        """Enhanced prediction generation with analysis integration"""
        try:
            if historical_data is None:
                historical_data = self.load_data()
            
            if recent_draws is None:
                recent_draws = historical_data.tail(12)
            
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
            
            # Format the next_draw_time to match historical_draws.csv format (07:50  12-03-2025)
            next_draw_time_formatted = next_draw_time.strftime('%H:%M  %d-%m-%Y')

            # Add next draw time to analysis context
            if analysis_context is None:
                analysis_context = {}
            analysis_context['next_draw_time'] = next_draw_time_formatted

            # Also add it to pipeline data for saving in metadata
            if hasattr(self, 'pipeline_data'):
                self.pipeline_data['next_draw_time'] = next_draw_time_formatted
            
            print(f"\nPrediction for next draw at: {analysis_context['next_draw_time']}")
            
            return predicted_numbers, probabilities, analysis_context
            
        except Exception as e:
            print(f"Error in train_and_predict: {e}")
            return None, None, None

    def train_models(self, features, labels):
        """Train models with enhanced preprocessing and validation"""
        try:
            print("\nStarting model training...")
            
            # Validate inputs
            if not isinstance(features, np.ndarray) or not isinstance(labels, np.ndarray):
                raise ValueError("Features and labels must be numpy arrays")
            
            if len(features) != len(labels):
                raise ValueError(f"Mismatched lengths: features ({len(features)}) vs labels ({len(labels)})")
            
            # Print dataset information
            print(f"\nTraining dataset:")
            print(f"- Total samples: {len(features)}")
            print(f"- Feature dimensions: {features.shape[1]}")
            print(f"- Label shape: {labels.shape}")
            
            # Reshape labels if needed
            if len(labels.shape) > 1:
                labels = labels.ravel()
                print("- Reshaped labels to 1D array")
            
            # Split data with stratification
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, 
                    labels, 
                    test_size=0.2, 
                    random_state=42,
                    shuffle=True
                )
                print("\nData split complete:")
                print(f"- Training samples: {len(X_train)}")
                print(f"- Test samples: {len(X_test)}")
            except Exception as e:
                print(f"Error in data splitting: {e}")
                raise
            
            # Scale features
            try:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                print("\nFeature scaling complete")
            except Exception as e:
                print(f"Error in feature scaling: {e}")
                raise
            
            # Train probabilistic model (Naive Bayes)
            try:
                print("\nTraining probabilistic model...")
                self.probabilistic_model = GaussianNB()
                self.probabilistic_model.fit(X_train_scaled, y_train)
                
                prob_train_score = self.probabilistic_model.score(X_train_scaled, y_train)
                prob_test_score = self.probabilistic_model.score(X_test_scaled, y_test)
                
                print(f"Probabilistic Model Performance:")
                print(f"- Training accuracy: {prob_train_score:.4f}")
                print(f"- Test accuracy: {prob_test_score:.4f}")
            except Exception as e:
                print(f"Error training probabilistic model: {e}")
                raise
            
            # Train pattern model (Neural Network)
            try:
                print("\nTraining pattern model...")
                self.pattern_model.fit(X_train_scaled, y_train)
                
                pattern_train_score = self.pattern_model.score(X_train_scaled, y_train)
                pattern_test_score = self.pattern_model.score(X_test_scaled, y_test)
                
                print(f"Pattern Model Performance:")
                print(f"- Training accuracy: {pattern_train_score:.4f}")
                print(f"- Test accuracy: {pattern_test_score:.4f}")
            except Exception as e:
                print(f"Error training pattern model: {e}")
                raise
            
            # Update training status
            self.training_status.update({
                'success': True,
                'model_loaded': True,
                'timestamp': datetime.now(),
                'prob_score': prob_test_score,
                'pattern_score': pattern_test_score,
                'features': features.shape[1],
                'model_config': {
                    'num_classes': self.num_classes,
                    'numbers_to_draw': self.numbers_to_draw,
                    'feature_dimension': features.shape[1],
                    'training_samples': len(X_train),
                    'test_samples': len(X_test)
                },
                'training_metrics': {
                    'prob_train_acc': prob_train_score,
                    'prob_test_acc': prob_test_score,
                    'pattern_train_acc': pattern_train_score,
                    'pattern_test_acc': pattern_test_score,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            })
            
            print("\nModel training completed successfully")
            return True
            
        except Exception as e:
            print(f"\nError in model training: {e}")
            self.training_status.update({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now()
            })
            traceback.print_exc()
            return False

    def predict(self, recent_draws):
        # Add these logging lines right after the first print statement
        print("\nStarting prediction pipeline...")
        print("=== Model State Debug Info ===")
        print(f"Model timestamp: {self.training_status.get('timestamp')}")
        print(f"Model loaded status: {self.training_status.get('model_loaded')}")
        print(f"Using combined features: {self.pipeline_data.get('use_combined_features', False)}")
        print(f"Latest model file: {self.models_dir}")
        print("===========================\n")
        """Enhanced prediction with improved pipeline execution and validation"""
        pipeline_tracking = {
            'start_time': datetime.now(),
            'stages_completed': [],
            'current_stage': None,
            'error': None
        }
        
        try:
            print("\nStarting prediction pipeline...")
            
            # Validate model state
            is_valid, message = self.validate_model_state()
            if not is_valid:
                print(f"Warning: {message}")
                print("Attempting to proceed with available model state...")
            
            # Validate input data
            if recent_draws is None or len(recent_draws) < 12:
                raise ValueError("Need at least 5 recent draws for prediction")
                
            if not isinstance(recent_draws, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
                
            # Initialize pipeline if needed
            if not hasattr(self, 'pipeline_stages'):
                self._initialize_pipeline()
                
            # Execute pipeline stages
            result = recent_draws
            for stage_name, stage_func in self.pipeline_stages.items():
                try:
                    print(f"\nExecuting stage: {stage_name}")
                    pipeline_tracking['current_stage'] = stage_name
                    stage_start = datetime.now()
                    
                    # Execute stage
                    result = stage_func(result)
                    
                    # Validate stage result
                    if result is None:
                        raise ValueError(f"Stage {stage_name} failed to produce valid results")
                        
                    # Record stage completion
                    stage_duration = (datetime.now() - stage_start).total_seconds()
                    pipeline_tracking['stages_completed'].append({
                        'stage': stage_name,
                        'duration': stage_duration,
                        'success': True,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    
                    print(f"Stage completed in {stage_duration:.2f} seconds")
                    
                except Exception as e:
                    print(f"Error in stage {stage_name}: {e}")
                    pipeline_tracking['stages_completed'].append({
                        'stage': stage_name,
                        'error': str(e),
                        'success': False,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    raise
            
            # Get final predictions
            final_numbers = self.pipeline_data.get('final_prediction')
            probabilities = self.pipeline_data.get('probabilities')
            analysis_context = self.pipeline_data.get('analysis_context', {})
            
            # Validate predictions
            if final_numbers is None or len(final_numbers) != self.numbers_to_draw:
                raise ValueError(f"Invalid prediction results: expected {self.numbers_to_draw} numbers")
                
            # Convert probabilities to list if needed
            if isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()
                
            # Store execution metadata
            self.pipeline_data['prediction_metadata'] = {
                'execution_time': (datetime.now() - pipeline_tracking['start_time']).total_seconds(),
                'stages_completed': pipeline_tracking['stages_completed'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'numbers_generated': len(final_numbers),
                'prediction_confidence': float(np.mean(probabilities)) if probabilities else None
            }
            
            print("\nPrediction pipeline completed successfully:")
            print(f"- Numbers generated: {sorted(final_numbers)}")
            print(f"- Total execution time: {self.pipeline_data['prediction_metadata']['execution_time']:.2f} seconds")
            
            return final_numbers, probabilities, analysis_context
            
        except Exception as e:
            error_msg = f"Error in prediction pipeline: {str(e)}"
            print(f"\n{error_msg}")
            
            # Update pipeline data with error information
            self.pipeline_data['error'] = {
                'message': error_msg,
                'stage': pipeline_tracking.get('current_stage'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            traceback.print_exc()
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

    def save_predictions_to_csv(self, prediction, probabilities):
        """Save predictions and metadata to files"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Use correct paths from PATHS config
            excel_file = PATHS['ALL_PREDICTIONS_FILE']
            predictions_file = os.path.join(PATHS['PREDICTIONS_DIR'], f'prediction_{timestamp}.csv')
            metadata_file = os.path.join(PATHS['PREDICTIONS_METADATA_DIR'], f'prediction_{timestamp}_metadata.json')

            # Ensure directories exist
            os.makedirs(os.path.dirname(excel_file), exist_ok=True)
            os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)

            # Convert numpy int64 to regular integers if needed
            if hasattr(prediction[0], 'item'):
                prediction = [num.item() for num in prediction]
            
            # Save predictions to CSV
            pred_df = pd.DataFrame({
                'number': prediction,
                'probability': probabilities if probabilities else [0] * len(prediction)
            })
            pred_df.to_csv(predictions_file, index=False)
            
            # Get next draw time if it exists in pipeline data
            next_draw_time = self.pipeline_data.get('next_draw_time', 
                                                   datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            
            # Prepare single consolidated metadata with all information
            metadata = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'next_draw_time': next_draw_time,  # Include next draw time
                'prediction_info': {
                    'total_numbers': len(prediction),
                    'number_range': self.numbers_range,
                    'numbers_to_draw': self.numbers_to_draw,
                    'average_probability': float(np.mean(probabilities)) if probabilities else 0
                },
                'model_info': {
                    'prob_model': self.pipeline_data.get('prediction_metadata', {}).get('prob_weight', 0.4),
                    'pattern_model': self.pipeline_data.get('prediction_metadata', {}).get('pattern_weight', 0.6),
                    'analysis_weight': self.pipeline_data.get('prediction_metadata', {}).get('analysis_weight', 0),
                    'training_status': {
                        'success': self.training_status['success'],
                        'timestamp': self.training_status['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(self.training_status['timestamp'], datetime) else self.training_status['timestamp'],
                        'prob_score': self.training_status['prob_score'],
                        'pattern_score': self.training_status['pattern_score']
                    }
                },
                'analysis_context': self.pipeline_data.get('analysis_context', {}),
                'feature_config': {
                    'use_combined_features': getattr(self, 'use_combined_features', True),
                    'feature_dimension': self.training_status.get('model_config', {}).get('feature_dimension', None)
                }
            }
            
            # Convert any numpy types to native Python types for JSON serialization
            def convert_to_native(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, tuple):
                    return list(obj)
                elif isinstance(obj, datetime):  # Add datetime handling
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                elif isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_native(v) for v in obj]
                return obj
            
            # Convert numpy types in metadata
            metadata = convert_to_native(metadata)
            
            # Save metadata to JSON
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            # NEW CODE: Also save to Excel file
            try:
                # Format data for new record
                new_data = {
                    'timestamp': metadata['timestamp'],
                    'next_draw_time': next_draw_time,
                    'prediction_date': datetime.now().strftime('%Y-%m-%d'),
                    'prediction_time': datetime.now().strftime('%H:%M:%S')
                }
                
                # Add predicted numbers to record
                for i, num in enumerate(prediction, 1):
                    new_data[f'number{i}'] = int(num)
                    
                # Add probabilities to record
                if probabilities and len(probabilities) == len(prediction):
                    for i, prob in enumerate(probabilities, 1):
                        new_data[f'probability{i}'] = float(prob)
                        
                new_row = pd.DataFrame([new_data])
                
                # Check if Excel file exists
                if os.path.exists(excel_file):
                    try:
                        # Load existing data
                        existing_df = pd.read_excel(excel_file)
                        
                        # Check if we already have this prediction
                        if 'next_draw_time' in existing_df.columns:
                            date_matches = existing_df['next_draw_time'] == next_draw_time
                            if any(date_matches):
                                # Update existing prediction
                                for col, value in new_data.items():
                                    if col in existing_df.columns:
                                        existing_df.loc[date_matches, col] = value
                                result_df = existing_df
                            else:
                                # Append new prediction
                                result_df = pd.concat([existing_df, new_row], ignore_index=True)
                        else:
                            # No matching column, just append
                            result_df = pd.concat([existing_df, new_row], ignore_index=True)
                    except Exception as e:
                        print(f"Warning: Error reading Excel file: {e}")
                        print("Creating new Excel file")
                        result_df = new_row
                else:
                    # Create new file
                    result_df = new_row
                    
                # Save to Excel
                result_df.to_excel(excel_file, index=False)
                print(f"- Excel file: {excel_file}")
            except Exception as e:
                print(f"Warning: Could not save to Excel: {e}")
                # Non-fatal error, we still saved the CSV
                
            print(f"\nPredictions saved successfully:")
            print(f"- CSV file: {predictions_file}")
            print(f"- Metadata: {metadata_file}")
            
            return True
            
        except Exception as e:
            print(f"Error saving predictions to CSV: {e}")
            traceback.print_exc()
            return False

    def save_prediction(self, prediction, probabilities, next_draw_time=None):
        """Save predictions in consolidated format with proper timestamp handling"""
        try:
            # Get current timestamp and calculate next draw time if not provided
            current_time = datetime.now()
            if next_draw_time is None:
                # Calculate minutes since midnight
                minutes_since_midnight = current_time.hour * 60 + current_time.minute
                # Find next 5-minute interval
                next_interval = ((minutes_since_midnight // 5) + 1) * 5
                # Calculate new hour and minute
                next_hour = (next_interval // 60) % 24
                next_minute = next_interval % 60
                # Create next draw time
                next_time = current_time.replace(hour=next_hour, minute=next_minute, second=0, microsecond=0)
                # If we've crossed to next day
                if next_hour < current_time.hour:
                    next_time += timedelta(days=1)
                next_draw_time = next_time.strftime('%H:%M  %d-%m-%Y')

            # Use ALL_PREDICTIONS_FILE directly from PATHS
            excel_file = PATHS['ALL_PREDICTIONS_FILE']
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(excel_file), exist_ok=True)

            # Prepare prediction data
            prediction_data = {
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'next_draw_time': next_draw_time,  # Use the calculated or provided next_draw_time
                'prediction_date': current_time.strftime('%Y-%m-%d'),
                'prediction_time': current_time.strftime('%H:%M:%S')
            }

            # Add predicted numbers and probabilities
            for i, (num, prob) in enumerate(zip(prediction, probabilities), 1):
                prediction_data[f'number{i}'] = int(num)
                prediction_data[f'probability{i}'] = float(prob)

            # Create DataFrame for new prediction
            new_row = pd.DataFrame([prediction_data])
            
            try:
                if os.path.exists(excel_file):
                    # Load existing predictions
                    existing_df = pd.read_excel(excel_file)
                    
                    # Check if we already have a prediction for this draw time
                    if 'next_draw_time' in existing_df.columns:
                        # IMPORTANT: Exact match for next_draw_time
                        date_matches = existing_df['next_draw_time'].str.strip() == next_draw_time.strip()
                        if any(date_matches):
                            print(f"Info: Prediction already exists for {next_draw_time} - skipping duplicate save")
                            # Return True instead of False to indicate this isn't an error
                            return True
                        else:
                            # Append new prediction only if it doesn't exist
                            result_df = pd.concat([existing_df, new_row], ignore_index=True)
                    else:
                        # No next_draw_time column, create new file
                        result_df = new_row
                else:
                    # Create new file
                    result_df = new_row

                # Save to Excel
                result_df.to_excel(excel_file, index=False)
                print(f"Prediction saved for draw at {next_draw_time}")

                # Save metadata
                metadata_file = os.path.join(PATHS['PREDICTIONS_METADATA_DIR'], 
                                             f'prediction_{current_time.strftime("%Y%m%d_%H%M%S")}_metadata.json')
                metadata = {
                    'timestamp': prediction_data['timestamp'],
                    'next_draw_time': next_draw_time,
                    'prediction_info': {
                        'total_numbers': len(prediction),
                        'number_range': self.numbers_range,
                        'numbers_to_draw': self.numbers_to_draw
                    },
                    'model_info': self.training_status
                }
                
                # Utility function to convert objects into JSON-serializable native types
                def convert_to_json_serializable(obj):
                    if isinstance(obj, datetime):
                        return obj.strftime('%Y-%m-%d %H:%M:%S')
                    elif isinstance(obj, dict):
                        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_to_json_serializable(v) for v in obj]
                    return obj
                
                # Convert metadata to a JSON-serializable format
                metadata = convert_to_json_serializable(metadata)
                
                # Ensure metadata directory exists
                os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
                
                # Save metadata to JSON
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)

                return True

            except Exception as excel_error:
                print(f"Error saving to Excel: {excel_error}")
                traceback.print_exc()
                return False

        except Exception as e:
            print(f"Error in save_prediction: {e}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    try:
        # Initialize predictor with configuration
        predictor = LotteryPredictor(
            numbers_range=(1, 80),
            numbers_to_draw=20,
            use_combined_features=True
        )
        
        print("\n=== Lottery Predictor Initialization ===")
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load historical data
        print("\nLoading historical data...")
        historical_data = predictor.load_data()
        if historical_data is None:
            raise ValueError("Failed to load historical data")
            
        print(f"Loaded {len(historical_data)} historical draws")
        
        # Generate prediction
        print("\nGenerating prediction...")
        prediction, probabilities, analysis = predictor.train_and_predict(
            historical_data=historical_data
        )
        
        if prediction is not None:
            print("\n=== Prediction Results ===")
            print(f"Predicted numbers: {sorted(prediction)}")
            
            if probabilities is not None:
                confidence = np.mean(probabilities) if isinstance(probabilities, (list, np.ndarray)) else 0
                print(f"Average confidence: {confidence:.4f}")
            
            print("\n=== Analysis Context ===")
            if analysis:
                for key, value in analysis.items():
                    print(f"{key}: {value}")
            
            # Save prediction
            predictor.save_predictions_to_csv(prediction, probabilities)
            
        else:
            print("\nError: Prediction generation failed")
            
        print(f"\nExecution completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nCritical error in main execution: {e}")
        traceback.print_exc()