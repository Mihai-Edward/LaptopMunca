import glob
import json
import traceback
import pandas as pd
from datetime import datetime, timedelta
import os
import numpy as np
from openpyxl import Workbook, load_workbook
from src.lottery_predictor import LotteryPredictor
from collections import Counter
from sklearn.cluster import KMeans
from src.data_analysis import DataAnalysis
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import PATHS, ensure_directories
from src.data_collector_selenium import KinoDataCollector
from src.prediction_evaluator import PredictionEvaluator
import joblib

class DrawHandler:
    def __init__(self):
        """Initialize DrawHandler with enhanced configuration and tracking"""
        try:
            # Initialize paths using config
            ensure_directories()
            self.csv_file = PATHS['HISTORICAL_DATA']
            self.models_dir = PATHS['MODELS_DIR']
            self.predictions_dir = PATHS['PREDICTIONS_DIR']
            
        
            self.number_cols = [f'number{i}' for i in range(1, 21)]
            
            # Enhanced pipeline status tracking
            self.pipeline_status = {
                'success': False,
                'stage': None,
                'error': None,
                'timestamp': None,
                'stages_completed': [],
                'warnings': [],
                'last_successful_run': None,
                'performance_metrics': {
                    'accuracy': None,
                    'reliability': None,
                    'last_update': None
                }
            }
            
            # Initialize predictor with enhanced configuration
            self.predictor = LotteryPredictor()
            
            # Initialize pipeline tracking
            self.pipeline_tracking = {
                'start_time': None,
                'stages_completed': [],
                'current_stage': None,
                'error': None,
                'metrics': {
                    'processing_time': {},
                    'success_rate': {},
                    'error_counts': {}
                }
            }

            # Initialize continuous learning tracking with enhanced metrics
            self.learning_dir = os.path.join(self.models_dir, 'learning_history')
            os.makedirs(self.learning_dir, exist_ok=True)
            
            # Learning history file paths
            self.learning_history_file = os.path.join(self.learning_dir, 'learning_history.csv')
            self.learning_metrics_file = os.path.join(self.learning_dir, 'learning_metrics.json')
            
            # Enhanced learning status tracking
            self.learning_status = {
                'last_learning': None,
                'cycles_completed': 0,
                'initial_accuracy': None,
                'current_accuracy': None,
                'improvement_rate': None,
                'last_adjustments': [],
                'performance_history': [],
                'model_versions': [],
                'feature_modes': [],
                'training_metrics': {
                    'avg_training_time': None,
                    'best_accuracy': None,
                    'worst_accuracy': None,
                    'stability_score': None
                }
            }
            
            # Load learning status and validate configuration
            self.load_learning_status()
            self._validate_configuration()
            
            print("DrawHandler initialized successfully")
            print(f"Using models directory: {self.models_dir}")
            print(f"Predictions directory: {self.predictions_dir}")
            
        except Exception as e:
            print(f"Error initializing DrawHandler: {e}")
            traceback.print_exc()
            raise

    def handle_prediction_pipeline(self, historical_data=None):
        """Coordinates the prediction pipeline process with enhanced analysis"""
        try:
            self.pipeline_status['timestamp'] = datetime.now()
            
            if historical_data is None:
                historical_data = self._load_historical_data()
            
            # 1. Data Preparation Stage
            self.pipeline_status['stage'] = 'data_preparation'
            processed_data = self._prepare_pipeline_data(historical_data)
            
            # 1.5. Analysis Stage (Perform analysis after data preparation)
            self.pipeline_status['stage'] = 'data_analysis'

            # Format data for DataAnalysis
            formatted_draws = []
            for _, row in historical_data.iterrows():
                try:
                    numbers = []
                    for i in range(1, 21):
                        col = f'number{i}'
                        if col in row and pd.notnull(row[col]):
                            num = int(float(row[col]))
                            if 1 <= num <= 80:
                                numbers.append(num)
                    
                    # Only process if we have all 20 numbers
                    if len(numbers) == 20:
                        # Format date string in required format
                        if isinstance(row['date'], pd.Timestamp):
                            date_str = row['date'].strftime('%H:%M  %d-%m-%Y')
                        else:
                            date_str = str(row['date']).strip()
                        
                        # Create tuple with exactly 2 elements
                        draw_tuple = (date_str, sorted(numbers))
                        formatted_draws.append(draw_tuple)
                except Exception as e:
                    print(f"DEBUG: Error formatting draw: {e}")
                    continue

            analyzer = DataAnalysis(formatted_draws)
            analysis_results = analyzer.get_analysis_results()
            
            # Ensure predictor has pipeline_data initialized for compatibility
            if not hasattr(self.predictor, 'pipeline_data'):
                self.predictor.pipeline_data = {}
                
            # Pass analysis results to predictor
            self.predictor.pipeline_data['analysis_context'] = analysis_results
            
            # 2. Prediction Stage
            self.pipeline_status['stage'] = 'prediction'
            model_path = self._get_latest_model()
            if model_path:
                model_files = [
                    f"{model_path}_prob_model.pkl",
                    f"{model_path}_pattern_model.pkl",
                    f"{model_path}_scaler.pkl"
                ]
                if all(os.path.exists(file) for file in model_files):
                    # NEW: Synchronize feature mode before loading
                    self._load_and_synchronize_feature_mode()
                    
                    print(f"✓ Model found: {os.path.basename(model_path)}")
                    load_success = self.predictor.load_models(model_path)
                    
                    if not load_success:
                        print("Model loading failed. Attempting retraining...")
                        if self.train_ml_models():
                            return self._run_prediction(processed_data)
                        
                    # Get predictions and handle returns properly
                    result = self._run_prediction(processed_data)
                    if result and len(result) == 3:
                        numbers, probs, analysis = result
                        if numbers is not None:
                            # Save to consolidated format
                            if hasattr(self.predictor, 'save_prediction'):
                                next_draw_time = get_next_draw_time(datetime.now())
                                success = self.predictor.save_prediction(
                                    prediction=numbers,
                                    probabilities=probs,
                                    next_draw_time=next_draw_time
                                )
                                if not success:
                                    print("WARNING: Failed to save to consolidated format")
                    
                            self.pipeline_status['success'] = True
                            return numbers, probs, analysis
                    
                else:
                    print("Model files incomplete. Attempting retraining...")
                    if self.train_ml_models():
                        return self._run_prediction(processed_data)
            else:
                print("No model found. Attempting training...")
                if self.train_ml_models():
                    return self._run_prediction(processed_data)
            
            self.pipeline_status['error'] = "Failed to generate predictions"
            return None, None, None
                
        except Exception as e:
            print(f"Error in prediction pipeline: {e}")
            self.pipeline_status['error'] = str(e)
            self.pipeline_status['success'] = False
            return None, None, None

    def _validate_configuration(self):
        """Validate and verify all required configurations"""
        try:
            # Verify paths exist
            required_paths = [self.csv_file, self.models_dir, self.predictions_dir]
            for path in required_paths:
                if not os.path.exists(os.path.dirname(path)):
                    os.makedirs(os.path.dirname(path))
                    print(f"Created directory: {os.path.dirname(path)}")
            
            # Verify predictor initialization
            if not hasattr(self, 'predictor') or self.predictor is None:
                self.predictor = LotteryPredictor()
                print("Initialized new LotteryPredictor instance")
            
            # Initialize metrics tracking file if it doesn't exist
            metrics_file = os.path.join(self.learning_dir, 'performance_metrics.json')
            if not os.path.exists(metrics_file):
                initial_metrics = {
                    'version': '1.0',
                    'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'metrics': {
                        'predictions': 0,
                        'successful_predictions': 0,
                        'average_accuracy': 0.0,
                        'last_update': None
                    }
                }
                with open(metrics_file, 'w') as f:
                    json.dump(initial_metrics, f, indent=4)
                print("Initialized performance metrics tracking")
            
            return True
            
        except Exception as e:
            print(f"Error in configuration validation: {e}")
            return False

    def train_ml_models(self, force_retrain=False, use_combined_features=True):
        """Train or retrain ML models"""
        try:
            self.pipeline_status['stage'] = 'model_training'
            
            # Load historical data
            historical_data = self._load_historical_data()
            if historical_data is None or len(historical_data) < 6:
                raise ValueError("Insufficient historical data for training")

            print(f"Loaded {len(historical_data)} draws for training")
            
            # Initialize predictor with combined features setting if it doesn't exist
            if not hasattr(self, 'predictor') or self.predictor is None:
                self.predictor = LotteryPredictor(use_combined_features=use_combined_features)
            
            # Initialize pipeline_data if needed
            if not hasattr(self.predictor, 'pipeline_data'):
                self.predictor.pipeline_data = {}
            
            # Set feature mode in pipeline_data
            self.predictor.pipeline_data['use_combined_features'] = use_combined_features
        
            # Prepare data for training
            features, labels = self.predictor.prepare_data(historical_data)
            if features is None or labels is None or len(features) == 0:
                raise ValueError("Failed to prepare training data")
        
            print(f"Prepared features shape: {features.shape}")
            print(f"Prepared labels shape: {labels.shape}")
            
            # IMPORTANT: Set the feature model type BEFORE training
            self.predictor.pipeline_data['use_combined_features'] = use_combined_features
        
            # Prepare additional features if using combined approach
            if use_combined_features:
                try:
                    print("\nEnhancing training data with analysis features...")
                
                    # Create DataAnalysis instance for feature generation
                    formatted_draws = []
                    for _, row in historical_data.iterrows():
                        try:
                            numbers = [int(float(row[f'number{i}'])) for i in range(1, 21)]
                            if len(numbers) == 20 and all(1 <= n <= 80 for n in numbers):
                                formatted_draws.append((row['date'], numbers))
                        except Exception as e:
                            print(f"DEBUG: Error formatting draw: {e}")
                
                    if formatted_draws:
                        print("\nDEBUG: Sample of formatted data being passed to predictor:")
                        print(f"First draw: {formatted_draws[0]}")
                        print(f"Number of draws: {len(formatted_draws)}")
                        print(f"Data format: {type(formatted_draws[0])}")
                        analyzer = DataAnalysis(formatted_draws)
                    
                        # Calculate analysis features
                        frequency = analyzer.count_frequency()
                        hot_numbers, cold_numbers = analyzer.hot_and_cold_numbers()
                    
                        # --- Commented out hot number features expansion ---
                        # print("Adding analysis features to training data...")
                        # # Get top 20 hot numbers for feature expansion
                        # top_hot = [num for num, _ in hot_numbers[:20]]
                        # # Create hot number features (20 dimensions)
                        # hot_features = np.zeros((features.shape[0], 20))
                        # # Set values based on frequency (simple method)
                        # for i, num in enumerate(top_hot):
                        #     if 1 <= num <= 80:
                        #         freq = frequency.get(num, 0)
                        #         # Normalize by max frequency
                        #         max_freq = max(frequency.values()) if frequency else 1
                        #         hot_features[:, i] = freq / max_freq
                        # # Combine with original features
                        # enhanced_features = np.hstack([features, hot_features])
                        # print(f"Enhanced training data shape: {enhanced_features.shape}")
                        # # Use enhanced features
                        # features = enhanced_features
                        # --- End of commented-out section ---
                except Exception as e:
                    print(f"Warning: Could not enhance training features: {e}")
                    print("Falling back to base features")
                    use_combined_features = False
                    self.predictor.pipeline_data['use_combined_features'] = False

            # Train models using predictor with prepared data
            training_success = self.predictor.train_models(features, labels)
        
            if training_success:
                # Save feature mode used for training
                feature_mode_file = os.path.join(self.models_dir, 'feature_mode.txt')
                with open(feature_mode_file, 'w') as f:
                    f.write('combined' if use_combined_features else 'base')
                    f.write(f"\nfeature_dim:{features.shape[1]}")
            
                # Save models immediately after successful training
                if not self.predictor.save_models():
                    raise Exception("Failed to save trained models")
                
                self.pipeline_status['success'] = True
                print(f"Models trained and saved successfully with {'combined' if use_combined_features else 'base'} features")
                return True
            else:
                raise Exception("Model training failed")
            
        except Exception as e:
            self.pipeline_status['error'] = str(e)
            print(f"Error in model training: {e}")
            return False


    def _get_latest_model(self):
        """Get the path to the latest model"""
        try:
            # First check in the models directory for timestamp file
            timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    timestamp = f.read().strip()
                    model_path = os.path.join(self.models_dir, f'lottery_predictor_{timestamp}')
                    if os.path.exists(f"{model_path}_prob_model.pkl"):
                        return model_path
            
            # Fallback to searching for model files
            model_files = glob.glob(os.path.join(self.models_dir, "*_prob_model.pkl"))
            if model_files:
                latest = max(model_files, key=os.path.getctime)
                return latest.replace('_prob_model.pkl', '')
                
            return None
                
        except Exception as e:
            print(f"Error getting latest model: {e}")
            return None

    def _load_historical_data(self):
        """Load historical data with validation and enhanced debugging"""
        try:
            print(f"\nDEBUG: Attempting to load historical data from: {self.csv_file}")
            print(f"DEBUG: File exists: {os.path.exists(self.csv_file)}")
            print(f"DEBUG: File size: {os.path.getsize(self.csv_file)} bytes")
            
            # Try to read the first few lines of the file directly
            print("\nDEBUG: First few lines of the file:")
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i < 5:  # Print first 5 lines
                        print(f"Line {i}: {line.strip()}")
            
            # Now try to load with pandas
            print("\nDEBUG: Loading with pandas...")
            df = pd.read_csv(self.csv_file, header=0)
            print(f"DEBUG: DataFrame shape: {df.shape}")
            print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")
            print("\nDEBUG: First row of data:")
            print(df.iloc[0])
                
            # Handle the specific date format with potential spaces
            try:
                # First clean up any extra spaces in the date column
                print("\nDEBUG: Attempting date conversion...")
                print(f"DEBUG: Date column before cleaning: {df['date'].head()}")
                
                df['date'] = df['date'].str.strip()
                print(f"DEBUG: Date column after cleaning: {df['date'].head()}")
                
                # Convert using the exact format from your file
                df['date'] = pd.to_datetime(df['date'], format='%H:%M  %d-%m-%Y')
                print("DEBUG: Date conversion successful with double space format")
                print(f"DEBUG: Converted dates: {df['date'].head()}")
                
            except Exception as e:
                print(f"WARNING: Initial date conversion issue: {e}")
                try:
                    # Fallback: Try with single space if double space fails
                    print("\nDEBUG: Attempting fallback date conversion...")
                    df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y')
                    print("DEBUG: Date conversion successful with fallback format")
                    print(f"DEBUG: Converted dates: {df['date'].head()}")
                except Exception as e2:
                    print(f"WARNING: Fallback date conversion failed: {e2}")
                    print("\nDEBUG: Sample of problematic dates:")
                    print(df['date'].head(10))
                    
            return df
                
        except Exception as e:
            print(f"Error loading historical data: {e}")
            traceback.print_exc()
            return None

    def _prepare_pipeline_data(self, data):
        """Prepare data for prediction pipeline"""
        try:
            if data is None:
                return None
            
            # Basic data cleaning
            data = data.copy()
            data = data.dropna(subset=['date'])
            
            # Extract date features
            data = extract_date_features(data)
            
            # Ensure all number columns exist
            for col in self.number_cols:
                if col not in data.columns:
                    data[col] = 0
                    
            # Add additional preparation for compatibility with new predictor
            if hasattr(self.predictor, 'pipeline_data') and self.predictor.pipeline_data.get('use_combined_features', False):
                # Convert numerical columns to float
                number_cols = [f'number{i}' for i in range(1, 21)]
                for col in number_cols:
                    data[col] = data[col].astype(float)
                
                # Add timestamp-based features
                data['timestamp'] = pd.to_datetime(data['date'])
                data['hour_of_day'] = data['timestamp'].dt.hour
                data['day_of_week'] = data['timestamp'].dt.dayofweek
                data['week_of_year'] = data['timestamp'].dt.isocalendar().week
                
            return data
                
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None

    def _run_prediction(self, data):
        try:
            print("\nDEBUG: Starting prediction run...")
            analysis_results = {}
            
            # Calculate next draw time at the start
            current_time = datetime.now()
            next_draw_time = get_next_draw_time(current_time)
            print(f"\nGenerating prediction for draw at: {next_draw_time}")
            
            # Load historical data
            historical_data = self._load_historical_data()
            if historical_data is not None:
                print(f"DEBUG: Loaded historical data shape: {historical_data.shape}")
                
                # Format data for prediction
                print("\nDEBUG: Starting data formatting for prediction...")
                formatted_draws = []
                
                for idx, row in historical_data.iterrows():
                    try:
                        # Extract numbers
                        numbers = []
                        for i in range(1, 21):
                            col = f'number{i}'
                            if col in row and pd.notnull(row[col]):
                                num = int(float(row[col]))
                                if 1 <= num <= 80:
                                    numbers.append(num)
                        
                        # Only process if we have all 20 numbers
                        if len(numbers) == 20:
                            # Format date string
                            date_str = str(row['date']).strip()
                            # Create tuple with exactly 2 elements
                            draw_tuple = (date_str, sorted(numbers))
                            formatted_draws.append(draw_tuple)
                            
                            if idx < 2:  # Debug output for first 2 draws
                                print(f"\nDEBUG: Draw {idx} formatted:")
                                print(f"Date: {date_str}")
                                print(f"Numbers: {sorted(numbers)}")
                                print(f"Tuple format: {draw_tuple}")
                        
                    except Exception as e:
                        print(f"DEBUG: Error formatting draw {idx}: {e}")
                        continue
                
                print(f"\nDEBUG: Total formatted draws: {len(formatted_draws)}")
                
                if formatted_draws:
                    print("\nDEBUG: Creating DataAnalysis instance...")
                    try:
                        analyzer = DataAnalysis(formatted_draws)
                        print("DEBUG: Successfully created DataAnalysis instance")
                        analysis_results = analyzer.get_analysis_results()
                    except Exception as e:
                        print(f"ERROR in analysis: {e}")
                        traceback.print_exc()
                    
                    # Get predictions
                    print("\nDEBUG: Getting predictions...")
                    
                    # Ensure predictor has pipeline_data initialized
                    if not hasattr(self.predictor, 'pipeline_data'):
                        self.predictor.pipeline_data = {}
                    
                    # Pass analysis results and next draw time to predictor
                    self.predictor.pipeline_data.update({
                        'analysis_context': analysis_results,
                        'next_draw_time': next_draw_time,
                        'prediction_metadata': {
                            'generation_time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'target_draw_time': next_draw_time,
                            'data_records': len(data) if isinstance(data, pd.DataFrame) else 0,
                            'historical_records': len(historical_data) if historical_data is not None else 0
                        }
                    })
                    
                    # Get prediction
                    try:
                        print("DEBUG: Calling predictor.predict()...")
                        # FIXED: Pass the original DataFrame, not the formatted draws list
                        prediction_result = self.predictor.predict(data)  # Changed from formatted_draws to data
                        
                        if prediction_result is None:
                            print("ERROR: Predictor returned None")
                            return None, None, None
                        
                        predicted_numbers, probabilities, context = prediction_result
                        return predicted_numbers, probabilities, analysis_results
                        
                    except Exception as e:
                        print(f"ERROR in prediction: {e}")
                        traceback.print_exc()
                        return None, None, None
                
                else:
                    print("ERROR: No valid draws found for analysis!")
                    return None, None, None
                    
            else:
                print("ERROR: No historical data available")
                return None, None, None
                
        except Exception as e:
            print(f"\nERROR in prediction run: {e}")
            traceback.print_exc()
            return None, None, None

    def _handle_pipeline_results(self, predictions, probabilities, analysis_results):
        """Handle the results from the prediction pipeline with enhanced tracking and integration"""
        try:
            print("\nDEBUG: Starting pipeline results handler")
            
            # Validate inputs
            if predictions is None or probabilities is None:
                print("No valid predictions to handle")
                self.pipeline_status['success'] = False
                self.pipeline_status['error'] = "Invalid prediction results"
                return False

            # Format timestamp and next draw time
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            next_draw_time = get_next_draw_time(datetime.now())

            # Convert and validate predictions
            if isinstance(predictions, np.ndarray):
                predictions = predictions.tolist()
            predictions = sorted(predictions)

            # Handle probabilities
            if isinstance(probabilities, np.ndarray):
                probabilities = probabilities.tolist()

            # Create probability mapping
            if len(probabilities) == 80:  # Full distribution
                prob_map = {num: probabilities[num-1] for num in predictions}
            else:  # Direct mapping
                prob_map = dict(zip(predictions, probabilities))

            # Store results in predictor's pipeline data
            self.predictor.pipeline_data.update({
                'latest_prediction': {
                    'numbers': predictions,
                    'probabilities': prob_map,
                    'timestamp': timestamp,
                    'next_draw_time': next_draw_time
                }
            })

            # Update pipeline status
            self.pipeline_status.update({
                'success': True,
                'last_successful_run': timestamp,
                'performance_metrics': {
                    'accuracy': None,  # Will be updated by evaluator
                    'reliability': len(predictions) == 20,  # Basic validation
                    'last_update': timestamp
                }
            })

            # Display results
            formatted_numbers = ','.join(map(str, predictions))
            print(f"\nPredicted numbers for next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}:")
            print(f"Numbers: {formatted_numbers}")
            
            # Display probabilities
            print("\nProbabilities for each predicted number:")
            for num in predictions:
                print(f"Number {num}: {prob_map.get(num, 1.0/len(predictions)):.4f}")

            # Handle analysis results
            if analysis_results:
                print("\nAnalysis Results:")
                if 'hot_cold' in analysis_results and analysis_results['hot_cold']:
                    hot_numbers = [num for num, _ in analysis_results['hot_cold'][0][:4]]
                    print(f"Top 4 hot numbers: {','.join(map(str, hot_numbers))}")
                
                # Store analysis metrics
                self.pipeline_data.update({
                    'latest_analysis': {
                        'timestamp': timestamp,
                        'metrics': analysis_results
                    }
                })

            # Save results to predictor's internal storage
            try:
                self.predictor.save_predictions_to_csv(
                    predicted_numbers=predictions,
                    probabilities=probabilities,
                    timestamp=timestamp
                )
            except Exception as e:
                print(f"Warning: Could not save predictions: {e}")

            return True

        except Exception as e:
            print(f"Error handling pipeline results: {e}")
            self.pipeline_status.update({
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            traceback.print_exc()
            return False
    
    def load_learning_status(self):
        """Load or initialize learning status tracking with enhanced metrics and validation"""
        try:
            print("\nLoading learning status...")
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check if learning metrics file exists
            if os.path.exists(self.learning_metrics_file):
                with open(self.learning_metrics_file, 'r') as f:
                    loaded_status = json.load(f)
                    
                # Validate loaded data structure
                required_keys = {
                    'last_learning', 'cycles_completed', 'initial_accuracy',
                    'current_accuracy', 'improvement_rate', 'last_adjustments',
                    'performance_history', 'model_versions', 'feature_modes',
                    'training_metrics'
                }
                
                # Check for missing keys and initialize them if needed
                for key in required_keys:
                    if key not in loaded_status:
                        print(f"Warning: Missing key '{key}' in loaded status, initializing...")
                        loaded_status[key] = self.learning_status[key]
                
                # Update learning status with loaded data
                self.learning_status.update(loaded_status)
                print("Loaded existing learning metrics")
                
                # Validate training metrics
                if not isinstance(self.learning_status['training_metrics'], dict):
                    self.learning_status['training_metrics'] = {
                        'avg_training_time': None,
                        'best_accuracy': None,
                        'worst_accuracy': None,
                        'stability_score': None
                    }
            else:
                # Initialize new learning metrics with enhanced tracking
                self.learning_status.update({
                    'last_learning': current_time,
                    'cycles_completed': 0,
                    'initial_accuracy': None,
                    'current_accuracy': None,
                    'improvement_rate': None,
                    'last_adjustments': [],
                    'performance_history': [],
                    'model_versions': [],
                    'feature_modes': [],
                    'training_metrics': {
                        'avg_training_time': None,
                        'best_accuracy': None,
                        'worst_accuracy': None,
                        'stability_score': None
                    },
                    'metadata': {
                        'created_at': current_time,
                        'last_modified': current_time,
                        'version': '2.0'
                    }
                })
                
                # Create initial metrics file
                self._save_learning_status()
                print("Initialized new learning metrics")
            
            # Verify predictor has required attributes
            if not hasattr(self.predictor, 'pipeline_data'):
                self.predictor.pipeline_data = {}
            
            # Pass relevant learning status to predictor
            self.predictor.pipeline_data['learning_history'] = {
                'cycles_completed': self.learning_status['cycles_completed'],
                'current_accuracy': self.learning_status['current_accuracy'],
                'feature_modes': self.learning_status['feature_modes'],
                'last_learning': self.learning_status['last_learning'],
                'performance_metrics': self.learning_status['training_metrics']
            }
            
            # Update metadata
            self.learning_status['metadata'] = {
                'last_loaded': current_time,
                'version': '2.0',
                'predictor_version': getattr(self.predictor, 'version', 'unknown')
            }
            
            return True
            
        except Exception as e:
            print(f"Error loading learning status: {e}")
            traceback.print_exc()
            return False

    def _save_learning_status(self):
        """Save current learning status with enhanced validation and backup"""
        try:
            print("\nSaving learning status...")
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Update metadata before saving
            self.learning_status['metadata'] = {
                'last_modified': current_time,
                'version': '2.0',
                'predictor_version': getattr(self.predictor, 'version', 'unknown')
            }
            
            # Create backup of existing file if it exists
            if os.path.exists(self.learning_metrics_file):
                backup_file = self.learning_metrics_file.replace('.json', f'_backup_{current_time.replace(":", "-")}.json')
                try:
                    import shutil
                    shutil.copy2(self.learning_metrics_file, backup_file)
                    print(f"Created backup: {os.path.basename(backup_file)}")
                except Exception as e:
                    print(f"Warning: Could not create backup: {e}")
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.learning_metrics_file), exist_ok=True)
            
            # Validate data before saving
            required_keys = {
                'last_learning', 'cycles_completed', 'initial_accuracy',
                'current_accuracy', 'improvement_rate', 'last_adjustments',
                'performance_history', 'model_versions', 'feature_modes',
                'training_metrics'
            }
            
            missing_keys = [key for key in required_keys if key not in self.learning_status]
            if missing_keys:
                raise ValueError(f"Missing required keys in learning status: {missing_keys}")
            
            # Save with pretty printing
            with open(self.learning_metrics_file, 'w') as f:
                json.dump(self.learning_status, f, indent=4)
            
            print(f"Learning status saved successfully")
            
            # Clean up old backups (keep last 5)
            try:
                backup_files = glob.glob(self.learning_metrics_file.replace('.json', '_backup_*.json'))
                if len(backup_files) > 5:
                    oldest_files = sorted(backup_files)[:-5]
                    for file in oldest_files:
                        os.remove(file)
                    print(f"Cleaned up {len(oldest_files)} old backup files")
            except Exception as e:
                print(f"Warning: Could not clean up old backups: {e}")
            
            return True
            
        except Exception as e:
            print(f"Error saving learning status: {e}")
            traceback.print_exc()
            return False

    def apply_learning_from_evaluations(self):
        """Apply continuous learning with enhanced analysis and model adjustments"""
        try:
            # Get current and display cycles completed
            cycles_completed = self.learning_status.get('cycles_completed', 0)
            print("\nLearning cycle metrics:")
            print(f"- Cycles completed: {cycles_completed}")
            
            # Handle initial accuracy - add this check
            if self.learning_status.get('initial_accuracy') is None:
                # If no initial accuracy, set it from current accuracy
                evaluator = PredictionEvaluator()
                stats = evaluator.get_performance_stats()
                if stats and 'avg_accuracy' in stats:
                    self.learning_status['initial_accuracy'] = stats.get('avg_accuracy', 0)
                    print(f"- Initial accuracy: {self.learning_status['initial_accuracy']:.2f}% (newly set)")
                else:
                    self.learning_status['initial_accuracy'] = 0
                    print("- Initial accuracy: Not available yet")
            else:
                # Safe print of existing value
                print(f"- Initial accuracy: {self.learning_status['initial_accuracy']:.2f}%")
                
            # Safe printing for current accuracy
            current_acc = self.learning_status.get('current_accuracy', 0)
            if current_acc is None:
                current_acc = 0
            print(f"- Current accuracy: {current_acc:.2f}%")
            
            # Ultra-defensive approach for improvement rate
            try:
                # First get the value with a default
                imp_rate = self.learning_status.get('improvement_rate', 0)
                # Force convert to float, handling None case
                imp_rate = float(imp_rate if imp_rate is not None else 0)
                print(f"- Improvement rate: {imp_rate:.2f}%")
            except (TypeError, ValueError):
                # Catch absolutely any formatting or conversion issues
                print("- Improvement rate: 0.00%")
            
            print("\nApplying continuous learning from evaluation results...")
            start_time = datetime.now()
            
            # Initialize or update learning cycle tracking
            cycle_tracking = {
                'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'cycle_number': self.learning_status['cycles_completed'] + 1,
                'stages_completed': [],
                'metrics': {}
            }

            # Initialize the evaluator
            evaluator = PredictionEvaluator()
            
            # Get performance statistics with enhanced metrics
            stats = evaluator.get_performance_stats()
            if not stats or stats.get('total_predictions', 0) < 5:
                print("Not enough evaluation data for learning (need at least 5 predictions)")
                return False
            
            # Extract and validate insights
            insights = {
                'problematic_numbers': list(stats.get('most_frequently_missed', {}).keys()),
                'successful_numbers': list(stats.get('most_frequent_correct', {}).keys()),
                'recent_trend': stats.get('recent_trend', 0),
                'average_accuracy': stats.get('average_accuracy', 0),
                'consistency_score': stats.get('consistency_score', 0),
                'prediction_confidence': stats.get('prediction_confidence', 0)
            }
            
            # Log insights
            print("\nEvaluation insights:")
            print(f"- Problematic numbers: {insights['problematic_numbers']}")
            print(f"- Successful numbers: {insights['successful_numbers']}")
            print(f"- Recent trend: {insights['recent_trend']:.3f} ({'improving' if insights['recent_trend'] > 0 else 'declining'})")
            print(f"- Average accuracy: {insights['average_accuracy']:.2f}%")
            print(f"- Consistency score: {insights['consistency_score']:.2f}")
            print(f"- Prediction confidence: {insights['prediction_confidence']:.2f}")
            
            # Create adjustment plan with enhanced tracking
            adjustments = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'adjustments_made': [],
                'metrics_before': {
                    'accuracy': insights['average_accuracy'],
                    'trend': insights['recent_trend'],
                    'consistency': insights['consistency_score']
                }
            }
            
            # Make model adjustments based on insights
            adjustment_success = self._adjust_model_parameters(
                insights['problematic_numbers'],
                insights['successful_numbers'],
                insights['recent_trend'],
                insights['average_accuracy'],
                adjustments
            )
            
            if adjustment_success:
                print("\nModel adjustments applied successfully")
            else:
                print("\nNo model adjustments were necessary")
            
            # Save learning metadata with enhanced tracking
            metadata = {
                'cycle_number': cycle_tracking['cycle_number'],
                'duration': (datetime.now() - start_time).total_seconds(),
                'insights': insights,
                'adjustments': adjustments,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            self._save_learning_metadata(stats, adjustments, metadata)
            
            # Update learning status with enhanced metrics
            self.learning_status.update({
                'last_learning': adjustments['timestamp'],
                'cycles_completed': cycle_tracking['cycle_number'],
                'current_accuracy': insights['average_accuracy'],
                'last_adjustments': adjustments['adjustments_made'],
                'training_metrics': {
                    'avg_training_time': stats.get('avg_training_time', self.learning_status['training_metrics']['avg_training_time']),
                    'best_accuracy': max(insights['average_accuracy'], 
                                       self.learning_status['training_metrics'].get('best_accuracy', 0) or 0),
                    'worst_accuracy': min(insights['average_accuracy'], 
                                        self.learning_status['training_metrics'].get('worst_accuracy', 100) or 100),
                    'stability_score': insights['consistency_score']
                }
            })
            
            # Update initial accuracy if not set
            if self.learning_status['initial_accuracy'] is None:
                self.learning_status['initial_accuracy'] = insights['average_accuracy']
            
            # Calculate improvement rate
            if self.learning_status['initial_accuracy'] and self.learning_status['initial_accuracy'] > 0:
                self.learning_status['improvement_rate'] = (
                    (insights['average_accuracy'] - self.learning_status['initial_accuracy']) 
                    / self.learning_status['initial_accuracy'] * 100
                )
            
            # Update predictor's pipeline data
            self.predictor.pipeline_data.update({
                'learning_status': {
                    'last_cycle': metadata,
                    'current_metrics': self.learning_status['training_metrics'],
                    'improvement_rate': self.learning_status['improvement_rate']
                }
            })
            
            # Save updated learning status
            self._save_learning_status()
            
            # Print summary
            print("\nLearning cycle completed:")
            print(f"- Total learning cycles: {self.learning_status['cycles_completed']}")
            print(f"- Current accuracy: {self.learning_status['current_accuracy']:.2f}%")
            print(f"- Total improvement: {self.learning_status['improvement_rate']:.2f}%")
            print(f"- Cycle duration: {metadata['duration']:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Error applying learning: {e}")
            traceback.print_exc()
            return False

    def _adjust_model_parameters(self, problematic_numbers, successful_numbers, trend, accuracy, adjustments):
        """Make specific adjustments to model parameters with enhanced tracking and validation"""
        # Define timestamp ONCE at the start
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        try:
            print(f"Current accuracy: {accuracy:.2f}%")
            print("\n=== Model Parameter Adjustment ===")
            print("\nAdjusting model parameters...")
            print(f"Performance trend: {trend}")
            print(f"Problematic numbers: {problematic_numbers[:5]}...")
            print(f"Successful numbers: {successful_numbers[:5]}...")
            
            # Initialize adjustment tracking
            adjustment_tracking = {
                'start_time': timestamp,
                'adjustments_applied': [],
                'parameters_modified': set(),
                'metrics_before': {
                    'accuracy': accuracy,
                    'trend': trend
                }
            }
            
            # Check if predictor and pipeline_data exist
            if not hasattr(self, 'predictor') or self.predictor is None:
                self.predictor = LotteryPredictor()
                
            # Ensure pipeline_data exists
            if not hasattr(self.predictor, 'pipeline_data'):
                self.predictor.pipeline_data = {}
            
            # Now safely track original parameters
            original_params = {
                'number_boosts': self.predictor.pipeline_data.get('number_boosts', None),
                'prediction_weights': self.predictor.pipeline_data.get('prediction_weights', None),
                'feature_mode': self.predictor.pipeline_data.get('use_combined_features', False)
            }
            
            needs_adjustment = False
            
            # 1. Create number boosts with enhanced logic
            if problematic_numbers:
                boost_array = np.ones(80)
                max_boost = 1.15  # Maximum 15% boost
                min_boost = 1.05  # Minimum 5% boost
                
                # Calculate dynamic boost based on accuracy
                boost_factor = min_boost + (max_boost - min_boost) * (1 - min(accuracy, 50) / 50)
                
                for num in problematic_numbers[:10]:  # Take top 10 most missed
                    if 1 <= num <= 80:
                        boost_array[num-1] = boost_factor
                
                self.predictor.pipeline_data['number_boosts'] = boost_array
                adjustments['adjustments_made'].append(
                    f"Boosted problematic numbers with factor {boost_factor:.3f}: {problematic_numbers[:10]}"
                )
                adjustment_tracking['parameters_modified'].add('number_boosts')
                needs_adjustment = True
            
            # 2. Adjust model weights based on trend and accuracy
            weights = {}
            if trend < 0 or accuracy < 30:  # More aggressive adjustment for poor performance
                # Calculate dynamic weights based on performance
                prob_weight = 0.5 - (0.1 * (trend if trend < 0 else 0))  # Increase probability weight for negative trends
                pattern_weight = 1 - prob_weight
                
                weights = {
                    'prob_weight': max(0.3, min(0.7, prob_weight)),  # Clamp between 0.3 and 0.7
                    'pattern_weight': max(0.3, min(0.7, pattern_weight))
                }
                
                self.predictor.pipeline_data['prediction_weights'] = weights
                adjustments['adjustments_made'].append(
                    f"Modified model weights: prob={weights['prob_weight']:.2f}, pattern={weights['pattern_weight']:.2f}"
                )
                adjustment_tracking['parameters_modified'].add('prediction_weights')
                needs_adjustment = True
            
            # 3. Enhanced feature mode adjustments
            if accuracy < 20 or (trend < -0.1 and accuracy < 30):
                # Enable enhanced feature extraction and analysis
                self.predictor.pipeline_data['use_enhanced_features'] = True
                self.predictor.pipeline_data['use_combined_features'] = True
                
                # Adjust feature weights
                self.predictor.pipeline_data['feature_weights'] = {
                    'frequency': 0.4,
                    'pattern': 0.3,
                    'analysis': 0.3
                }
                
                adjustments['adjustments_made'].append(
                    "Enabled enhanced features and adjusted feature weights"
                )
                adjustment_tracking['parameters_modified'].add('feature_mode')
                needs_adjustment = True
                
                # Consider retraining if accuracy is very low
                if accuracy < 10:
                    if hasattr(self.predictor, 'training_status'):
                        self.predictor.training_status['require_retraining'] = True
                        adjustments['adjustments_made'].append("Marked for retraining due to low accuracy")
            
            # 4. Save adjusted model if changes were made
            if needs_adjustment:
                # Create new model path with adjustment identifier
                model_path = os.path.join(
                    self.models_dir,
                    f'lottery_predictor_adjusted_{timestamp}'
                )
                
                # Save the adjusted model
                if self.predictor.save_models(path_prefix=model_path):
                    # Save adjustment metadata
                    adjustment_file = os.path.join(self.models_dir, 'model_adjustments.txt')
                    with open(adjustment_file, 'a') as f:
                        f.write(f"\n--- Adjustments made at {adjustments['timestamp']} ---\n")
                        for adjustment in adjustments['adjustments_made']:
                            f.write(f"- {adjustment}\n")
                        f.write(f"Parameters modified: {', '.join(adjustment_tracking['parameters_modified'])}\n")
                    
                    print(f"Model adjusted and saved as: {os.path.basename(model_path)}")
                    
                    # Update timestamp file
                    timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
                    with open(timestamp_file, 'w') as f:
                        f.write(timestamp)
                    
                    # Store adjustment metrics
                    self.predictor.pipeline_data['last_adjustment'] = {
                        'timestamp': timestamp,
                        'changes': adjustments['adjustments_made'],
                        'parameters': list(adjustment_tracking['parameters_modified']),
                        'metrics_before': adjustment_tracking['metrics_before']
                    }
                    
                    # Compare with original parameters to log what changed
                    final_params = {
                        'number_boosts': self.predictor.pipeline_data.get('number_boosts', None),
                        'prediction_weights': self.predictor.pipeline_data.get('prediction_weights', None),
                        'feature_mode': self.predictor.pipeline_data.get('use_combined_features', False)
                    }
                    
                    # Log what changed
                    for param_name, original_value in original_params.items():
                        final_value = final_params.get(param_name)
                        if original_value != final_value:
                            print(f"Changed parameter {param_name}:")
                            print(f"  Before: {original_value}")
                            print(f"  After:  {final_value}")
                            
                            # Add to adjustments history
                            adjustments['adjustments_made'].append(
                                f"Changed {param_name} from {original_value} to {final_value}"
                            )
                    
                    return True
            else:
                print("No adjustments needed at this time")
                return False
                    
        except Exception as e:
            print(f"Error adjusting model parameters: {e}")
            traceback.print_exc()
            return False

    def _save_learning_metadata(self, stats, adjustments, metadata=None):
        """Save enhanced metadata about the learning process with improved tracking"""
        try:
            print("\nSaving learning metadata...")
            current_time = datetime.now()
            
            # Prepare base metadata
            base_metadata = {
                'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'stats': {
                    'accuracy': stats.get('average_accuracy', 0),
                    'trend': stats.get('recent_trend', 0),
                    'best_prediction': stats.get('best_prediction', 0),
                    'total_predictions': stats.get('total_predictions', 0),
                    'success_rate': stats.get('success_rate', 0),
                    'consistency_score': stats.get('consistency_score', 0),
                    'prediction_confidence': stats.get('prediction_confidence', 0)
                },
                'adjustments': adjustments.get('adjustments_made', []),
                'model_state': {
                    'feature_mode': self.predictor.pipeline_data.get('use_combined_features', False),
                    'weights': self.predictor.pipeline_data.get('prediction_weights', {}),
                    'enhanced_features': self.predictor.pipeline_data.get('use_enhanced_features', False)
                }
            }
            
            # Merge with provided metadata if any
            if metadata:
                base_metadata.update(metadata)
            
            # Create learning history DataFrame
            history_data = {
                'timestamp': [base_metadata['timestamp']],
                'accuracy': [base_metadata['stats']['accuracy']],
                'trend': [base_metadata['stats']['trend']],
                'success_rate': [base_metadata['stats']['success_rate']],
                'consistency': [base_metadata['stats']['consistency_score']],
                'confidence': [base_metadata['stats']['prediction_confidence']],
                'adjustments': [str(base_metadata['adjustments'])],
                'feature_mode': [str(base_metadata['model_state']['feature_mode'])]
            }
            
            df = pd.DataFrame(history_data)
            
            # Save to CSV with proper handling
            try:
                if os.path.exists(self.learning_history_file):
                    df.to_csv(self.learning_history_file, mode='a', header=False, index=False)
                else:
                    os.makedirs(os.path.dirname(self.learning_history_file), exist_ok=True)
                    df.to_csv(self.learning_history_file, index=False)
            except Exception as e:
                print(f"Warning: Could not save to CSV: {e}")
            
            # Save detailed metadata to JSON
            detailed_metadata_file = os.path.join(
                self.learning_dir,
                f'learning_metadata_{current_time.strftime("%Y%m%d_%H%M%S")}.json'
            )
            
            # Add this converter function
            def convert_to_serializable(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, datetime):
                    return obj.strftime('%Y-%m-%d %H:%M:%S')
                return obj
            
            try:
                os.makedirs(os.path.dirname(detailed_metadata_file), exist_ok=True)
                # Then use it when saving JSON
                with open(detailed_metadata_file, 'w') as f:
                    json.dump(base_metadata, f, indent=4, default=convert_to_serializable)
            except Exception as e:
                print(f"Warning: Could not save detailed metadata: {e}")
            
            # Clean up old metadata files (keep last 10)
            try:
                metadata_files = glob.glob(os.path.join(self.learning_dir, 'learning_metadata_*.json'))
                if len(metadata_files) > 10:
                    for old_file in sorted(metadata_files)[:-10]:
                        os.remove(old_file)
                    print(f"Cleaned up {len(metadata_files) - 10} old metadata files")
            except Exception as e:
                print(f"Warning: Could not clean up old metadata files: {e}")
            
            # Update predictor's pipeline data with latest metadata
            self.predictor.pipeline_data.update({
                'latest_learning_metadata': base_metadata,
                'learning_history_file': self.learning_history_file,
                'last_metadata_update': current_time.strftime('%Y-%m-%d %H:%M:%S')
            })
            
            print(f"Learning metadata saved successfully")
            print(f"History file: {os.path.basename(self.learning_history_file)}")
            print(f"Detailed metadata: {os.path.basename(detailed_metadata_file)}")
            
            return True
            
        except Exception as e:
            print(f"Error saving learning metadata: {e}")
            traceback.print_exc()
            return False

    def run_continuous_learning_cycle(self):
        """Run a complete continuous learning cycle with enhanced tracking and validation"""
        try:
            current_time = datetime.now()
            print(f"\nStarting continuous learning cycle at {current_time.strftime('%Y-%m-%d %H:%M:%S')}...")
            
            # Initialize cycle tracking
            cycle_tracking = {
                'start_time': current_time,
                'cycle_number': self.learning_status['cycles_completed'] + 1,
                'stages': {
                    'evaluation': {'status': 'pending', 'start': None, 'end': None, 'error': None},
                    'learning': {'status': 'pending', 'start': None, 'end': None, 'error': None},
                    'testing': {'status': 'pending', 'start': None, 'end': None, 'error': None}
                },
                'metrics': {},
                'errors': []
            }

            # Update pipeline tracking
            self.pipeline_tracking.update({
                'start_time': current_time,
                'stages_completed': [],
                'current_stage': 'initialization',
                'error': None
            })

            # Step 1: Evaluate past predictions
            print("\nStep 1: Evaluating past predictions...")
            cycle_tracking['stages']['evaluation']['start'] = datetime.now()
            self.pipeline_tracking['current_stage'] = 'evaluation'
            
            try:
                evaluator = PredictionEvaluator()
                evaluation_results = evaluator.evaluate_past_predictions()
                
                if evaluation_results:
                    cycle_tracking['stages']['evaluation']['status'] = 'completed'
                    cycle_tracking['metrics']['evaluation'] = evaluation_results
                    self.pipeline_tracking['stages_completed'].append('evaluation')
                else:
                    raise ValueError("Evaluation failed to produce results")
                    
            except Exception as e:
                error_msg = f"Evaluation stage failed: {str(e)}"
                cycle_tracking['stages']['evaluation']['status'] = 'failed'
                cycle_tracking['stages']['evaluation']['error'] = error_msg
                cycle_tracking['errors'].append(error_msg)
                print(f"Error in evaluation stage: {e}")
                
            finally:
                cycle_tracking['stages']['evaluation']['end'] = datetime.now()

            # Step 2: Apply learning from evaluations
            print("\nStep 2: Applying learning from evaluations...")
            cycle_tracking['stages']['learning']['start'] = datetime.now()
            self.pipeline_tracking['current_stage'] = 'learning'
            
            try:
                learning_success = self.apply_learning_from_evaluations()
                
                if learning_success:
                    cycle_tracking['stages']['learning']['status'] = 'completed'
                    self.pipeline_tracking['stages_completed'].append('learning')
                    print("Learning stage completed successfully")
                else:
                    raise ValueError("Learning stage failed to complete")
                    
            except Exception as e:
                error_msg = f"Learning stage failed: {str(e)}"
                cycle_tracking['stages']['learning']['status'] = 'failed'
                cycle_tracking['stages']['learning']['error'] = error_msg
                cycle_tracking['errors'].append(error_msg)
                print(f"Error in learning stage: {e}")
                
            finally:
                cycle_tracking['stages']['learning']['end'] = datetime.now()

            # Step 3: Test the improved model
            print("\nStep 3: Testing improved model...")
            cycle_tracking['stages']['testing']['start'] = datetime.now()
            self.pipeline_tracking['current_stage'] = 'testing'
            
            try:
                # Reset predictor status
                if hasattr(self.predictor, 'training_status'):
                    self.predictor.training_status['model_loaded'] = False
                
                # Load and test adjusted model
                model_path = self._get_latest_model()
                if model_path:
                    print(f"Loading adjusted model: {os.path.basename(model_path)}")
                    load_success = self.predictor.load_models(model_path)
                    
                    if load_success:
                        print("Model loaded successfully, generating test prediction...")
                        
                        # Store original pipeline tracking
                        original_tracking = self.pipeline_tracking.copy()
                        
                        try:
                            # Generate test prediction
                            predictions, probabilities, analysis = self.handle_prediction_pipeline()
                            
                            if predictions is not None:
                                cycle_tracking['stages']['testing']['status'] = 'completed'
                                cycle_tracking['metrics']['test_prediction'] = {
                                    'numbers': predictions,
                                    'confidence': np.mean(probabilities) if isinstance(probabilities, (list, np.ndarray)) else None,
                                    'analysis_results': bool(analysis)
                                }
                                self.pipeline_tracking['stages_completed'].append('testing')
                                print(f"Generated test prediction: {sorted(predictions)}")
                            else:
                                raise ValueError("Test prediction failed")
                                
                        finally:
                            # Restore original pipeline tracking
                            self.pipeline_tracking = original_tracking
                    else:
                        raise ValueError("Failed to load adjusted model")
                else:
                    raise ValueError("No adjusted model found")
                    
            except Exception as e:
                error_msg = f"Testing stage failed: {str(e)}"
                cycle_tracking['stages']['testing']['status'] = 'failed'
                cycle_tracking['stages']['testing']['error'] = error_msg
                cycle_tracking['errors'].append(error_msg)
                print(f"Error in testing stage: {e}")
                
            finally:
                cycle_tracking['stages']['testing']['end'] = datetime.now()

            # Calculate cycle metrics
            cycle_duration = (datetime.now() - cycle_tracking['start_time']).total_seconds()
            successful_stages = sum(1 for stage in cycle_tracking['stages'].values() 
                                  if stage['status'] == 'completed')
            
            # Update cycle tracking with final metrics
            cycle_tracking.update({
                'end_time': datetime.now(),
                'duration': cycle_duration,
                'success_rate': (successful_stages / len(cycle_tracking['stages'])) * 100,
                'total_errors': len(cycle_tracking['errors'])
            })

            # Save cycle results
            try:
                cycle_results_file = os.path.join(
                    self.learning_dir,
                    f'learning_cycle_{current_time.strftime("%Y%m%d_%H%M%S")}.json'
                )
                
                os.makedirs(os.path.dirname(cycle_results_file), exist_ok=True)
                with open(cycle_results_file, 'w') as f:
                    json.dump(cycle_tracking, f, indent=4, default=str)
                    
                print(f"\nCycle results saved to: {os.path.basename(cycle_results_file)}")
            except Exception as e:
                print(f"Warning: Could not save cycle results: {e}")

            # Print cycle summary
            print("\nContinuous learning cycle complete!")
            print(f"Duration: {cycle_duration:.2f} seconds")
            print(f"Success rate: {cycle_tracking['success_rate']:.1f}%")
            print(f"Stages completed: {', '.join(self.pipeline_tracking['stages_completed'])}")
            if cycle_tracking['errors']:
                print(f"Errors encountered: {len(cycle_tracking['errors'])}")
            
            return len(cycle_tracking['errors']) == 0
            
        except Exception as e:
            print(f"Critical error in learning cycle: {e}")
            traceback.print_exc()
            return False

    def get_learning_metrics(self):
        """Get current learning metrics for display"""
        return {
            'cycles_completed': self.learning_status['cycles_completed'],
            'last_learning': self.learning_status['last_learning'],
            'initial_accuracy': self.learning_status['initial_accuracy'],
            'current_accuracy': self.learning_status['current_accuracy'],
            'improvement_rate': self.learning_status['improvement_rate'],
            'last_adjustments': self.learning_status['last_adjustments']
        }

    def _validate_draw_time(self, time_str):
        """Validate the draw time format and values"""
        try:
            # Parse the time string
            if ':' not in time_str:
                return False
                
            hour, minute = map(int, time_str.strip().split(':'))
            
            # Validate hour and minute ranges
            if not (0 <= hour <= 23):
                return False
            if not (0 <= minute <= 59):
                return False
                
            return True
        except ValueError:
            return False

    def save_models(self, path_prefix=None):
        """Save models with proper timestamp handling"""
        try:
            if path_prefix is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                path_prefix = os.path.join(self.models_dir, f'lottery_predictor_{timestamp}')
            else:
                # Extract timestamp from path or generate new one if needed
                if 'adjusted_' in path_prefix:
                    timestamp = path_prefix.split('adjusted_')[-1]
                else:
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Ensure predictor exists
            if not hasattr(self, 'predictor') or self.predictor is None:
                print("Error: Predictor not initialized")
                return False
                
            # Call the predictor's save_models method
            save_success = self.predictor.save_models(path_prefix=path_prefix)
            
            if save_success:
                # Update timestamp file
                timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
                with open(timestamp_file, 'w') as f:
                    f.write(timestamp)
                    
                print(f"Models saved successfully to: {path_prefix}")
                return True
            else:
                print("Error: Predictor failed to save models")
                return False
                
        except Exception as e:
            print(f"Error saving models: {e}")
            traceback.print_exc()
            return False

    def _load_and_synchronize_feature_mode(self):
        """Load feature mode from saved model and synchronize with predictor"""
        try:
            feature_mode_file = os.path.join(self.models_dir, 'feature_mode.txt')
            
            if os.path.exists(feature_mode_file):
                with open(feature_mode_file, 'r') as f:
                    content = f.read().strip().split('\n')
                    saved_mode = 'combined' in content[0]
                    
                    # Synchronize settings with predictor
                    self.predictor.use_combined_features = saved_mode
                    self.predictor.pipeline_data['use_combined_features'] = saved_mode
                    print(f"Synchronized feature mode from saved model: {'combined' if saved_mode else 'base'}")
                    return True
            else:
                # Create the file with current settings if it doesn't exist
                with open(feature_mode_file, 'w') as f:
                    current_mode = 'combined' if self.predictor.use_combined_features else 'base'
                    f.write(f"{current_mode}\n")
                    f.write(f"feature_dim:{164 if current_mode == 'combined' else 84}")
                print(f"Created feature mode file with current mode: {current_mode}")
                return False
                
        except Exception as e:
            print(f"Warning: Feature mode synchronization issue: {e}")
            return False

    def should_retrain_model(self):
        """Check if model should be retrained based on time passed"""
        try:
            # Get last training timestamp from file
            timestamp_file = os.path.join(self.models_dir, 'model_timestamp.txt')
            if os.path.exists(timestamp_file):
                with open(timestamp_file, 'r') as f:
                    timestamp_str = f.read().strip()
                    try:
                        last_train = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        hours_since_training = (datetime.now() - last_train).total_seconds() / 3600
                        
                        # Only retrain once per day (24 hours)
                        should_retrain = hours_since_training > 24
                        
                        if should_retrain:
                            print(f"Model is {hours_since_training:.1f} hours old. Scheduled retraining needed.")
                        else:
                            print(f"Model is {hours_since_training:.1f} hours old. No retraining needed yet.")
                            
                        return should_retrain
                    except ValueError:
                        print(f"Invalid timestamp format: {timestamp_str}")
                        return True
            return True  # If no timestamp file, should retrain
        except Exception as e:
            print(f"Error checking retraining schedule: {e}")
            return True  # Retrain on error for safety

def load_data(file_path=None):
    """Load and preprocess data from CSV"""
    if file_path is None:
        file_path = PATHS['HISTORICAL_DATA']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
        
    df = pd.read_csv(file_path)
    number_cols = [f'number{i}' for i in range(1, 21)]  # Define number_cols here
    
    try:
        df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y', errors='coerce')
        df.loc[df['date'].isna(), 'date'] = pd.to_datetime(df.loc[df['date'].isna(), 'date'], errors='coerce')
    except Exception as e:
        print(f"Warning: Date conversion issue: {e}")
        
    try:
        df[number_cols] = df[number_cols].astype(float)
    except Exception as e:
        print(f"Warning: Could not process number columns: {e}")
        
    for col in number_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
            
    return df

def extract_date_features(df):
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['days_since_first_draw'] = (df['date'] - df['date'].min()).dt.days
    return df

def get_next_draw_time(current_time):
    """Calculate the next draw time"""
    try:
        # Round to nearest 5 minutes
        minute = (current_time.minute // 5 * 5 + 5) % 60
        hour = current_time.hour + (current_time.minute // 5 * 5 + 5) // 60
        
        next_time = current_time.replace(hour=hour % 24, minute=minute, second=0, microsecond=0)
        
        # If we've wrapped to the next day, add a day
        if hour >= 24:
            next_time += timedelta(days=1)
            
        # Return formatted string
        return next_time.strftime('%H:%M  %d-%m-%Y')  # Note the double space
    except Exception as e:
        raise ValueError(f"Error calculating next draw time: {e}")

def save_top_4_numbers_to_excel(top_4_numbers, file_path=None):
    if file_path is None:
        file_path = os.path.join(PATHS['PREDICTIONS_DIR'], 'top_4.xlsx')
        # or if you want it one level up:
        # file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS_DIR']), 'top_4.xlsx')
    df = pd.DataFrame({'Top 4 Numbers': top_4_numbers})
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_excel(file_path, index=False)

def evaluate_numbers(historical_data):
    """Evaluate numbers based on criteria other than frequency.
    For simplicity, this example assumes a dummy evaluation function.
    Replace this with your actual evaluation logic.
    """
    number_evaluation = {i: 0 for i in range(1, 81)}
    for index, row in historical_data.iterrows():
        for i in range(1, 21):
            number = row[f'number{i}']
            number_evaluation[number] += 1  # Dummy evaluation logic
    sorted_numbers = sorted(number_evaluation, key=number_evaluation.get, reverse=True)
    return sorted_numbers[:4]  # Return top 4 numbers

def train_and_predict():
    try:
        handler = DrawHandler()    
        
        # First ensure models are trained
        print("Checking/Training models...")
        if not handler.train_ml_models():
            raise Exception("Model training failed")
        print("Models ready")
        
        # Generate prediction using pipeline
        print("\nGenerating predictions...")
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        if predictions is not None:
            current_time = datetime.now()
            
            # Safely get next draw time
            try:
                next_draw_time = get_next_draw_time(current_time)
            except ValueError as e:
                print(f"Warning: Error getting next draw time: {e}")
                next_draw_time = current_time + timedelta(minutes=5)
                next_draw_time = next_draw_time.replace(second=0, microsecond=0)
            
            # Format and display predictions
            formatted_numbers = ','.join(map(str, sorted(predictions)))
            print(f"\nPredicted numbers for next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}:")
            print(f"Numbers: {formatted_numbers}")
            
            # Safely handle probabilities
            if probabilities is not None:
                print("\nProbabilities for each predicted number:")
                # Convert probabilities to list if it's numpy array
                prob_list = probabilities.tolist() if hasattr(probabilities, 'tolist') else list(probabilities)
                
                # Ensure probabilities match predictions length
                if len(prob_list) == 80:  # If we have probabilities for all numbers
                    pred_probs = [prob_list[num - 1] for num in predictions]
                else:  # If we have probabilities just for predictions
                    pred_probs = prob_list[:len(predictions)]
                    
                # Display probabilities
                for num, prob in zip(sorted(predictions), pred_probs):
                    print(f"Number {num}: {prob:.4f}")

            # Save predictions with proper datetime handling
            prediction_time = next_draw_time.strftime('%Y-%m-%d %H:%M:00')
            
            # Ensure we have valid probabilities for saving
            save_probs = pred_probs if 'pred_probs' in locals() else [1.0] * len(predictions)
            
            handler.save_predictions_to_csv(
                predictions,
                save_probs,
                prediction_time
            )
            
            # Safely handle analysis results
            if analysis and isinstance(analysis, dict):
                hot_numbers = []
                if 'hot_cold' in analysis and analysis['hot_cold']:
                    hot_cold_data = analysis['hot_cold']
                    if isinstance(hot_cold_data, tuple) and len(hot_cold_data) > 0:
                        hot_numbers = [num for num, _ in hot_cold_data[0][:4]]
                elif 'hot_numbers' in analysis:
                    hot_numbers = analysis['hot_numbers'][:4]
                
                if hot_numbers:
                    top_4_file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS']), 'top_4.xlsx')
                    try:
                        save_top_4_numbers_to_excel(hot_numbers, top_4_file_path)
                        print(f"\nTop 4 numbers based on analysis: {','.join(map(str, hot_numbers))}")
                    except Exception as e:
                        print(f"Warning: Could not save top 4 numbers: {e}")
            
            return predictions, pred_probs if 'pred_probs' in locals() else save_probs, analysis
        else:
            print("\nFailed to generate predictions")
            return None, None, None
            
    except Exception as e:
        print(f"\nError in prediction process: {str(e)}")
        traceback.print_exc()
        return None, None, None

def perform_complete_analysis(draws):
    """Perform all analyses and save to Excel"""
    try:
        if not draws:
            collector = KinoDataCollector()    
            draws = collector.fetch_latest_draws()
        
        if draws:
            analysis = DataAnalysis(draws)    
            
            # Perform all analyses
            analysis_data = {
                'frequency': analysis.get_top_numbers(20),
                'suggested_numbers': analysis.suggest_numbers(),
                'common_pairs': analysis.find_common_pairs(),
                'consecutive_numbers': analysis.find_consecutive_numbers(),
                'range_analysis': analysis.number_range_analysis(),
                'hot_cold_numbers': analysis.hot_and_cold_numbers()
            }
            
            # Save to Excel file using config path
            excel_path = PATHS['ANALYSIS']
            os.makedirs(os.path.dirname(excel_path), exist_ok=True)
            
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                # Frequency Analysis
                pd.DataFrame({
                    'Top 20 Numbers': analysis_data['frequency'],
                    'Frequency Count': [analysis.count_frequency().get(num, 0) for num in analysis_data['frequency']]
                }).to_excel(writer, sheet_name='Frequency Analysis', index=False)
                
                # Suggested Numbers
                pd.DataFrame({
                    'Suggested Numbers': analysis_data['suggested_numbers']
                }).to_excel(writer, sheet_name='Suggested Numbers', index=False)
                
                # Common Pairs
                pd.DataFrame(analysis_data['common_pairs'], 
                           columns=['Pair', 'Frequency']
                ).to_excel(writer, sheet_name='Common Pairs', index=False)
                
                # Consecutive Numbers
                pd.DataFrame({
                    'Consecutive Sets': [str(x) for x in analysis_data['consecutive_numbers']]
                }).to_excel(writer, sheet_name='Consecutive Numbers', index=False)
                
                # Range Analysis
                pd.DataFrame(analysis_data['range_analysis'].items(),
                           columns=['Range', 'Count']
                ).to_excel(writer, sheet_name='Range Analysis', index=False)
                
                # Hot and Cold Numbers
                hot, cold = analysis_data['hot_cold_numbers']
                pd.DataFrame({
                    'Hot Numbers': hot,
                    'Cold Numbers': cold
                }).to_excel(writer, sheet_name='Hot Cold Analysis', index=False)
            
            print(f"\nComplete analysis saved to: {excel_path}")
            return True
    except Exception as e:
        print(f"\nError in complete analysis: {str(e)}")
        return False

def test_pipeline_integration():
    """Test the integrated prediction pipeline"""
    try:
        handler = DrawHandler()
        pipeline_status = {
            'data_collection': False,
            'analysis': False,
            'prediction': False,
            'evaluation': False
        }
        
        # 1. Data Collection
        print("\nStep 1: Collecting data...")
        collector = KinoDataCollector()
        draws = collector.fetch_latest_draws()
        if draws:
            pipeline_status['data_collection'] = True
            print("✓ Data collection successful")
            
            # Save draws to CSV
            for draw_date, numbers in draws:
                handler.save_draw_to_csv(draw_date, numbers)

            # 2. Analysis
            print("\nStep 2: Performing analysis...")
            if perform_complete_analysis(draws):
                pipeline_status['analysis'] = True
                print("✓ Analysis complete and saved")

            # 3. ML Prediction
            print("\nStep 3: Generating prediction...")
            predictions, probabilities, analysis = handler.handle_prediction_pipeline()
            if predictions is not None:
                pipeline_status['prediction'] = True
                print("✓ Prediction generated")
                # Display prediction results
                formatted_numbers = ','.join(map(str, predictions))
                print(f"Predicted numbers: {formatted_numbers}")
                if analysis and 'hot_numbers' in analysis:
                    print(f"Hot numbers: {analysis['hot_numbers'][:10]}")

            # 4. Evaluation
            print("\nStep 4: Evaluating predictions...")
            evaluator = PredictionEvaluator()
            evaluator.evaluate_past_predictions()
            pipeline_status['evaluation'] = True
            print("✓ Evaluation complete")
        
        return pipeline_status

    except Exception as e:
        print(f"\nError in pipeline: {str(e)}")
        return pipeline_status

def main():
    handler = DrawHandler()
    
    try:
        # First ensure models are trained
        print("Checking/Training models...")
        if handler.train_ml_models():
            print("Models ready")
        
        # Generate prediction using pipeline
        print("\nGenerating predictions...")
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        if predictions is not None:
            print("\n=== Prediction Results ===")
            print(f"Predicted Numbers: {sorted(predictions)}")
            
            # Display probabilities for predicted numbers
            print("\nProbabilities for predicted numbers:")
            for num, prob in zip(sorted(predictions), 
                               [probabilities[num-1] for num in predictions]):
                print(f"Number {num}: {prob:.4f}")
            
            if analysis:
                print("\n=== Analysis Results ===")
                for key, value in analysis.items():
                    if key != 'clusters':  # Skip clusters for cleaner output
                        print(f"\n{key.replace('_', ' ').title()}:")
                        print(value)
                        
    except Exception as e:
        print(f"\nError in pipeline execution: {str(e)}")
        print(f"Pipeline Status: {handler.pipeline_status}")
