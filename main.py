# Add at the top of main.py
import sys
import os
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config.paths import PATHS, ensure_directories
from src.lottery_predictor import LotteryPredictor
from src.data_collector_selenium import KinoDataCollector
from src.data_analysis import DataAnalysis
from src.draw_handler import DrawHandler
from src.prediction_evaluator import PredictionEvaluator

# 1. System Initialization
def initialize_system():
    """Initialize system with both LotteryPredictor and DrawHandler"""
    try:
        # Ensure all directories exist
        ensure_directories()
        
        # Initialize both core components
        predictor = LotteryPredictor(
            numbers_range=(1, 80),
            numbers_to_draw=20,
            use_combined_features=True
        )
        handler = DrawHandler()
        
        # Print initialization status
        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")
        
        print("System Components Initialized:")
        print("- LotteryPredictor: Ready")
        print("- DrawHandler: Ready")
        print(f"- Models Directory: {PATHS['MODELS_DIR']}")
        print(f"- Predictions Directory: {PATHS['PREDICTIONS']}")
        
        return {
            'start_time': datetime.now(),
            'system_ready': True,
            'predictor': predictor,
            'handler': handler
        }
    except Exception as e:
        print(f"System initialization failed: {e}")
        traceback.print_exc()
        return {
            'start_time': datetime.now(),
            'system_ready': False,
            'predictor': None,
            'handler': None
        }

def get_next_draw_time(current_time):
    """Calculate the next draw time based on current time"""
    try:
        # Round to nearest 5 minutes
        minutes = (current_time.minute // 5 * 5 + 5) % 60
        hour = current_time.hour + (current_time.minute // 5 * 5 + 5) // 60
        
        # Create next draw time
        next_time = current_time.replace(
            hour=hour % 24,
            minute=minutes,
            second=0,
            microsecond=0
        )
        
        # If we've wrapped to the next day
        if hour >= 24:
            next_time += timedelta(days=1)
        
        return next_time
        
    except Exception as e:
        print(f"Error calculating next draw time: {e}")
        # Fallback to simple 5-minute increment
        return current_time + timedelta(minutes=5)

# 2. Data Loading and Processing
def load_data(file_path=None):
    """Load and preprocess historical data with enhanced validation"""
    try:
        # Use default path if none provided
        if file_path is None:
            file_path = PATHS['HISTORICAL_DATA']
            
        print(f"\nDEBUG: Loading data from {file_path}")
        
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file {file_path} not found")
            
        # Load CSV with header
        df = pd.read_csv(file_path, header=0)
        print(f"DEBUG: Initial data shape: {df.shape}")
        
        # Clean and convert dates with enhanced error handling
        try:
            # First clean up any extra spaces in date column
            df['date'] = df['date'].str.strip()
            # Try exact format from your file first
            df['date'] = pd.to_datetime(df['date'], format='%H:%M  %d-%m-%Y')
            print("DEBUG: Date conversion successful")
            
        except Exception as e:
            print(f"WARNING: Initial date conversion failed: {e}")
            try:
                # Fallback to single space format
                df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y')
                print("DEBUG: Date conversion successful with fallback format")
            except Exception as e2:
                print(f"WARNING: Date conversion failed: {e2}")
                return None
        
        # Process number columns
        number_cols = [f'number{i+1}' for i in range(20)]
        try:
            # Convert to numeric and validate ranges
            for col in number_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                invalid_mask = ~df[col].between(1, 80)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    print(f"WARNING: Found {invalid_count} invalid numbers in {col}")
                    # Fill invalid numbers with mode of valid numbers
                    valid_mode = df.loc[~invalid_mask, col].mode()
                    if not valid_mode.empty:
                        df.loc[invalid_mask, col] = valid_mode[0]
                    else:
                        print(f"ERROR: No valid numbers found in column {col}")
                        return None
        except Exception as e:
            print(f"ERROR: Could not process number columns: {e}")
            return None
            
        # Sort by date descending (newest first)
        df = df.sort_values('date', ascending=False)
        
        print(f"DEBUG: Final data shape: {df.shape}")
        print(f"DEBUG: Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}")
        traceback.print_exc()
        return None

def extract_date_features(df):
    """Extract temporal features from date column with enhanced processing"""
    try:
        print("\nExtracting date features...")
        
        # Basic date features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['hour'] = df['date'].dt.hour
        df['minute'] = df['date'].dt.minute
        
        # Additional features used by LotteryPredictor
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Calculate days since first draw (used in pattern analysis)
        df['days_since_first_draw'] = (df['date'] - df['date'].min()).dt.days
        
        print("Extracted features:")
        print("- Basic time features (hour, minute)")
        print("- Calendar features (day, month, weekday)")
        print("- Advanced features (day of year, days since first)")
        
        # Validate extracted features
        for col in ['day_of_week', 'hour', 'minute', 'month', 'day', 'day_of_year', 'days_since_first_draw']:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                print(f"WARNING: Found {null_count} null values in {col}")
                # Fill nulls with median
                df[col] = df[col].fillna(df[col].median())
        
        return df
        
    except Exception as e:
        print(f"ERROR in feature extraction: {e}")
        traceback.print_exc()
        # Return original dataframe if extraction fails
        return df
# 3. Data Collection
def run_data_collector_standalone():
    """Run the data collector in standalone mode with enhanced logging"""
    try:
        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")
        
        print("\n--- Running Data Collector ---")
        collector = KinoDataCollector(debug=True)
        
        # First try to sort existing data
        print("\nSorting historical draws...")
        if collector.sort_historical_draws():
            print("✓ Historical draws sorted successfully")
        else:
            print("! Warning: Could not sort historical draws")
        
        # Then fetch new draws
        print("\nFetching latest draws...")
        draws = collector.fetch_latest_draws(num_draws=24)  # Default to 24 draws for consistency
        
        if draws:
            print("\nCollected draws:")
            for draw_date, numbers in draws:
                print(f"Date: {draw_date}")
                print(f"Numbers: {', '.join(map(str, sorted(numbers)))}")
            
            # Sort again after collecting new draws
            print("\nSorting updated historical draws...")
            if collector.sort_historical_draws():
                print("✓ Historical draws successfully sorted from newest to oldest")
                print(f"Last successful draw: {collector.collection_status['last_successful_draw']}")
            else:
                print("! Warning: Could not sort updated draws")
                
            # Print collection status
            print("\nCollection Status:")
            print(f"- Success: {'✓' if collector.collection_status['success'] else '✗'}")
            print(f"- Draws collected: {collector.collection_status['draws_collected']}")
            if collector.collection_status['last_error']:
                print(f"- Last error: {collector.collection_status['last_error']}")
            
            return draws
        else:
            print("\n✗ No draws collected")
            print(f"Error: {collector.collection_status['last_error']}")
            return None
            
    except Exception as e:
        print(f"\n✗ Error in data collection: {str(e)}")
        traceback.print_exc()
        return None

def save_top_4_numbers_to_excel(top_4_numbers, file_path=None, timestamp=None):
    """Save top 4 numbers to Excel with enhanced metadata"""
    try:
        # Get current timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        # Use default path if none provided
        if file_path is None:
            file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS']), 'top_4.xlsx')
            
        print(f"\nSaving top 4 numbers at {timestamp}")
        print(f"Numbers: {', '.join(map(str, top_4_numbers))}")
        
        # Create DataFrame with enhanced information
        df = pd.DataFrame({
            'Top 4 Numbers': top_4_numbers,
            'Timestamp': [timestamp] * 4,
            'Rank': range(1, 5)  # Add ranking
        })
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save with metadata sheet
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            # Save numbers
            df.to_excel(writer, sheet_name='Top Numbers', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame({
                'Property': ['Timestamp', 'User', 'Total Numbers', 'File Version'],
                'Value': [
                    timestamp,
                    os.getenv('USER', 'Mihai-Edward'),
                    len(top_4_numbers),
                    '2.0'
                ]
            })
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
            
        print(f"✓ Saved successfully to: {file_path}")
        print(f"- File contains {len(top_4_numbers)} numbers")
        print(f"- Added metadata sheet with timestamp and user info")
        
        return True
        
    except Exception as e:
        print(f"✗ Error saving top 4 numbers: {e}")
        traceback.print_exc()
        return False
def evaluate_numbers(historical_data):
    """Evaluate numbers based on comprehensive criteria"""
    try:
        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")
        
        print("Starting number evaluation...")
        
        # Initialize evaluation scores
        number_evaluation = {i: {'score': 0, 'metrics': {}} for i in range(1, 81)}
        
        # Get recent draws (last 24)
        recent_data = historical_data.head(24)
        
        print(f"Analyzing {len(recent_data)} recent draws...")
        
        # Calculate frequency
        for index, row in recent_data.iterrows():
            for i in range(1, 21):
                number = row[f'number{i}']
                if 1 <= number <= 80:
                    number_evaluation[number]['score'] += 1
                    number_evaluation[number]['metrics']['frequency'] = number_evaluation[number]['score']
        
        # Calculate hot/cold status
        avg_frequency = sum(d['score'] for d in number_evaluation.values()) / len(number_evaluation)
        for num in number_evaluation:
            if number_evaluation[num]['score'] > avg_frequency:
                number_evaluation[num]['metrics']['status'] = 'hot'
            else:
                number_evaluation[num]['metrics']['status'] = 'cold'
        
        # Calculate pair patterns
        for index, row in recent_data.iterrows():
            numbers = [row[f'number{i}'] for i in range(1, 21)]
            for num in numbers:
                if 1 <= num <= 80:
                    # Check for pairs
                    for other in numbers:
                        if other != num and abs(other - num) == 1:
                            number_evaluation[num]['score'] += 0.5
                            number_evaluation[num]['metrics']['pairs'] = \
                                number_evaluation[num]['metrics'].get('pairs', 0) + 1
        
        # Sort numbers by score
        sorted_numbers = sorted(
            number_evaluation.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        # Get top 4 numbers
        top_4 = [num for num, _ in sorted_numbers[:4]]
        
        print("\nEvaluation Results:")
        print(f"Top 4 numbers: {', '.join(map(str, top_4))}")
        print("\nDetailed metrics for top numbers:")
        for num in top_4:
            print(f"\nNumber {num}:")
            print(f"- Score: {number_evaluation[num]['score']}")
            print(f"- Status: {number_evaluation[num]['metrics']['status']}")
            print(f"- Frequency: {number_evaluation[num]['metrics']['frequency']}")
            if 'pairs' in number_evaluation[num]['metrics']:
                print(f"- Pair patterns: {number_evaluation[num]['metrics']['pairs']}")
        
        return top_4
        
    except Exception as e:
        print(f"\n✗ Error in number evaluation: {str(e)}")
        traceback.print_exc()
        return []
def perform_complete_analysis(draws=None):
    """Perform comprehensive analysis using DataAnalysis class"""
    try:
        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")
        
        print("Starting comprehensive analysis...")
        
        # If no draws provided, load from historical_draws.csv
        if not draws:
            historical_data = load_data(PATHS['HISTORICAL_DATA'])
            if historical_data is None:
                print("\n✗ No historical data available for analysis")
                return False
                
            print(f"\nLoaded {len(historical_data)} historical draws")
            
            # Use only last 24 draws for consistency
            recent_data = historical_data.head(24)
            print(f"Using last {len(recent_data)} draws for analysis")
            
            # Convert DataFrame to draws format expected by DataAnalysis
            draws = [(row['date'], [row[f'number{i}'] for i in range(1, 21)])
                    for _, row in recent_data.iterrows()]
        
        # Initialize analysis
        print("\nInitializing analysis components...")
        analysis = DataAnalysis(draws)
        
        # Perform comprehensive analysis
        print("\nGenerating analysis results...")
        analysis_data = {
            'frequency': analysis.count_frequency(),
            'hot_cold': analysis.hot_and_cold_numbers(),
            'common_pairs': analysis.find_common_pairs(),
            'range_analysis': analysis.number_range_analysis(),
            'sequences': analysis.sequence_pattern_analysis(),
            'clusters': analysis.cluster_analysis()
        }
        
        # Save to Excel using config path
        excel_path = PATHS['ANALYSIS']
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        
        print("\nSaving analysis to Excel...")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            # Frequency Analysis
            freq_df = pd.DataFrame([(num, count) for num, count in analysis_data['frequency'].items()],
                                 columns=['Number', 'Frequency'])
            freq_df.sort_values('Frequency', ascending=False).to_excel(writer, 
                sheet_name='Frequency Analysis', index=False)
            
            # Hot and Cold Numbers
            hot_numbers, cold_numbers = analysis_data['hot_cold']
            hot_cold_df = pd.DataFrame({
                'Hot Numbers': [x[0] for x in hot_numbers[:20]],
                'Hot Count': [x[1] for x in hot_numbers[:20]],
                'Cold Numbers': [x[0] for x in cold_numbers[:20]],
                'Cold Count': [x[1] for x in cold_numbers[:20]]
            })
            hot_cold_df.to_excel(writer, sheet_name='Hot Cold Analysis', index=False)
            
            # Common Pairs
            pairs_df = pd.DataFrame(analysis_data['common_pairs'],
                                  columns=['Pair', 'Frequency'])
            pairs_df.to_excel(writer, sheet_name='Common Pairs', index=False)
            
            # Range Analysis
            range_df = pd.DataFrame(list(analysis_data['range_analysis'].items()),
                                  columns=['Range', 'Count'])
            range_df.to_excel(writer, sheet_name='Range Analysis', index=False)
            
            # Sequence Analysis
            seq_df = pd.DataFrame(analysis_data['sequences'][:20],
                                columns=['Sequence', 'Frequency'])
            seq_df.to_excel(writer, sheet_name='Sequences', index=False)
            
            # Clusters
            clusters_df = pd.DataFrame([
                {'Cluster': k, 'Numbers': ', '.join(map(str, v))}
                for k, v in analysis_data['clusters'].items()
            ])
            clusters_df.to_excel(writer, sheet_name='Clusters', index=False)
            
            # Add metadata sheet
            metadata_df = pd.DataFrame({
                'Property': [
                    'Analysis Date',
                    'Draws Analyzed',
                    'User',
                    'Analysis Version'
                ],
                'Value': [
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    len(draws),
                    os.getenv('USER', 'Mihai-Edward'),
                    '2.0'
                ]
            })
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        print(f"\n✓ Complete analysis saved to: {excel_path}")
        print("\nAnalysis Summary:")
        print(f"- Total draws analyzed: {len(draws)}")
        print(f"- Hot numbers identified: {len(hot_numbers)}")
        print(f"- Common pairs found: {len(analysis_data['common_pairs'])}")
        print(f"- Number ranges analyzed: {len(analysis_data['range_analysis'])}")
        print(f"- Sequence patterns found: {len(analysis_data['sequences'])}")
        print(f"- Clusters created: {len(analysis_data['clusters'])}")
        
        return True
            
    except Exception as e:
        print(f"\n✗ Error in complete analysis: {str(e)}")
        traceback.print_exc()
        return False
def check_and_train_model():
    """Check if a trained model exists and train if needed using DrawHandler"""
    try:
        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")
        
        print("Initializing model check and training process...")
        
        # Initialize both components for proper model management
        predictor = LotteryPredictor(
            numbers_range=(1, 80),
            numbers_to_draw=20,
            use_combined_features=True
        )
        handler = DrawHandler()
        
        print("\nChecking model status...")
        
        # Check model files
        model_files = {
            'base': os.path.join(PATHS['MODELS_DIR'], 'base_model.joblib'),
            'ensemble': os.path.join(PATHS['MODELS_DIR'], 'ensemble_model.joblib'),
            'meta': os.path.join(PATHS['MODELS_DIR'], 'meta_model.joblib')
        }
        
        models_exist = all(os.path.exists(path) for path in model_files.values())
        
        if models_exist:
            print("✓ Existing models found")
            print("Checking model freshness...")
            
            # Get the oldest model file's timestamp
            oldest_time = min(os.path.getmtime(path) for path in model_files.values())
            model_age = datetime.now().timestamp() - oldest_time
            
            if model_age > 86400:  # 24 hours in seconds
                print(f"! Models are {model_age/3600:.1f} hours old")
                print("Initiating retraining...")
                needs_training = True
            else:
                print(f"✓ Models are fresh ({model_age/3600:.1f} hours old)")
                needs_training = False
        else:
            print("! No existing models found")
            print("Initiating initial training...")
            needs_training = True
        
        if needs_training:
            print("\nStarting model training pipeline...")
            if handler.train_ml_models():
                print("\n✓ Model training completed successfully")
                print("\nTrained Models:")
                for model_type, path in model_files.items():
                    if os.path.exists(path):
                        size = os.path.getsize(path) / 1024  # KB
                        print(f"- {model_type}: {size:.1f} KB")
                return True
            else:
                print("\n✗ Model training failed")
                return False
        else:
            print("\n✓ Models are up to date")
            return True
            
    except Exception as e:
        print(f"\n✗ Error checking/training model: {e}")
        traceback.print_exc()
        return False
def train_and_predict():
    """Generate predictions using both LotteryPredictor and DrawHandler"""
    try:
        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}")
        
        print("\nInitializing prediction pipeline...")
        print("Pipeline stages:")
        print("- Model initialization")
        print("- Data preparation")
        print("- Feature engineering")
        print("- Model prediction")
        print("- Post-processing")
        
        # Initialize both components
        predictor = LotteryPredictor(
            numbers_range=(1, 80),
            numbers_to_draw=20,
            use_combined_features=True
        )
        handler = DrawHandler()
        
        print("\nChecking/Training models...")
        if handler.train_ml_models():
            print("✓ Models ready")
            
            print("\nGenerating predictions...")
            try:
                # Load historical data
                historical_data = load_data(PATHS['HISTORICAL_DATA'])
                if historical_data is not None:
                    print(f"\nLoaded {len(historical_data)} historical draws")
                    
                    # Use last 24 draws for analysis
                    recent_draws = historical_data.head(24)
                    print(f"Using last {len(recent_draws)} draws for analysis")
                    
                    # Convert recent draws for handler
                    draws_for_analysis = [(row['date'], [row[f'number{i+1}'] for i in range(20)])
                                        for _, row in recent_draws.iterrows()]
                    
                    # Get predictions from both components
                    predictions = predictor.predict_next_draw()
                    probabilities = predictor.get_prediction_probabilities()
                    analysis = handler.analyze_draws(draws_for_analysis)
                    
                    if predictions is not None:
                        # Calculate next draw time
                        next_draw = get_next_draw_time(datetime.now())
                        
                        print("\n=== Prediction Results ===")
                        print(f"Next Draw Time: {next_draw.strftime('%Y-%m-%d %H:%M')}")
                        print(f"Predicted Numbers: {', '.join(map(str, sorted(predictions)))}")
                        
                        if probabilities is not None:
                            print("\n=== Probabilities ===")
                            for num, prob in zip(sorted(predictions), probabilities):
                                print(f"Number {num:2d}: {prob:.4f}")
                        
                        if analysis:
                            print("\n=== Analysis Context ===")
                            if 'hot_cold' in analysis:
                                hot, cold = analysis['hot_cold']
                                print("\nHot Numbers (Top 5):")
                                for num, freq in hot[:5]:
                                    print(f"Number {num}: {freq} times")
                                print("\nCold Numbers (Top 5):")
                                for num, freq in cold[:5]:
                                    print(f"Number {num}: {freq} times")
                            
                            if 'common_pairs' in analysis:
                                print("\nMost Common Pairs (Top 3):")
                                for pair, count in analysis['common_pairs'][:3]:
                                    print(f"Pair {pair}: {count} times")
                        
                        # Save predictions
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        prediction_path = os.path.join(PATHS['PREDICTIONS'], f'prediction_{timestamp}.csv')
                        metadata_path = os.path.join(PATHS['PREDICTIONS'], 'metadata', f'prediction_{timestamp}_metadata.json')
                        
                        # Save using both methods
                        predictor.save_prediction(predictions, prediction_path)
                        handler.save_prediction_metadata(metadata_path, predictions, probabilities, analysis)
                        
                        print(f"\n✓ Predictions saved:")
                        print(f"- CSV: {prediction_path}")
                        print(f"- Metadata: {metadata_path}")
                        
                        return predictions, probabilities, analysis
                    else:
                        print("\n✗ Failed to generate predictions")
                        return None, None, None
                else:
                    print("\n✗ No historical data available")
                    return None, None, None
                    
            except Exception as e:
                print(f"\n✗ Error during prediction: {str(e)}")
                traceback.print_exc()
                return None, None, None
        else:
            print("\n✗ Model training failed")
            return None, None, None
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        traceback.print_exc()
        return None, None, None
def test_pipeline_integration():
    """Test the integrated prediction pipeline with enhanced monitoring"""
    try:
        print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")

        handler = DrawHandler()
        pipeline_status = {
            'data_collection': False,
            'analysis': False,
            'prediction': False,
            'evaluation': False,
            'timestamps': {},
            'metrics': {}
        }

        # Initialize predictor
        predictor = LotteryPredictor(
            numbers_range=(1, 80),
            numbers_to_draw=20,
            use_combined_features=True
        )

        # 1. Data Collection
        print("\nStep 1: Testing data collection...")
        pipeline_status['timestamps']['collection_start'] = datetime.now()
        
        historical_data = handler.load_historical_data()
        if historical_data is not None and not historical_data.empty:
            pipeline_status['data_collection'] = True
            pipeline_status['metrics']['data_rows'] = len(historical_data)
            print(f"✓ Data collection successful ({len(historical_data)} draws loaded)")
            
            # 2. Analysis
            print("\nStep 2: Testing analysis integration...")
            pipeline_status['timestamps']['analysis_start'] = datetime.now()
            
            if perform_complete_analysis():
                pipeline_status['analysis'] = True
                print("✓ Analysis complete and saved")
                
                # Store analysis metrics
                analysis_file = PATHS['ANALYSIS']
                if os.path.exists(analysis_file):
                    pipeline_status['metrics']['analysis_file'] = os.path.getsize(analysis_file)

            # 3. Prediction Generation
            print("\nStep 3: Testing prediction pipeline...")
            pipeline_status['timestamps']['prediction_start'] = datetime.now()
            
            # Use both predictor and handler for predictions
            predictions = predictor.predict_next_draw()
            probabilities = predictor.get_prediction_probabilities()
            
            # Get recent draws for analysis
            recent_draws = historical_data.head(24)
            draws_for_analysis = [(row['date'], [row[f'number{i}'] for i in range(1, 21)])
                                for _, row in recent_draws.iterrows()]
            
            analysis = handler.analyze_draws(draws_for_analysis)
            
            if predictions is not None:
                pipeline_status['prediction'] = True
                pipeline_status['metrics']['prediction_count'] = len(predictions)
                print("✓ Prediction generated successfully")
                
                # Display prediction summary
                print(f"\nPredicted numbers: {', '.join(map(str, sorted(predictions)))}")
                if analysis and 'hot_numbers' in analysis:
                    print(f"Top hot numbers: {[num for num, _ in analysis['hot_numbers'][:5]]}")

            # 4. Evaluation
            print("\nStep 4: Testing evaluation system...")
            pipeline_status['timestamps']['evaluation_start'] = datetime.now()
            
            evaluator = PredictionEvaluator()
            evaluator.evaluate_past_predictions()
            
            # Check if evaluation results exist
            if os.path.exists(evaluator.results_file):
                pipeline_status['evaluation'] = True
                print("✓ Evaluation complete")
                
                # Load and store evaluation metrics
                try:
                    eval_df = pd.read_excel(evaluator.results_file)
                    pipeline_status['metrics']['total_evaluations'] = len(eval_df)
                    pipeline_status['metrics']['average_accuracy'] = eval_df['Number_Correct'].mean() / 20 * 100
                except Exception as e:
                    print(f"Note: Could not load evaluation metrics: {e}")

        # Calculate execution times
        pipeline_status['timestamps']['completion'] = datetime.now()
        
        # Print detailed pipeline report
        print("\n=== Pipeline Integration Test Report ===")
        print(f"Test completed at: {pipeline_status['timestamps']['completion'].strftime('%Y-%m-%d %H:%M:%S')}")
        print("\nStage Status:")
        for stage, status in {k: v for k, v in pipeline_status.items() 
                            if k not in ['timestamps', 'metrics']}.items():
            print(f"{stage.replace('_', ' ').title()}: {'✓' if status else '✗'}")
        
        print("\nMetrics:")
        if 'data_rows' in pipeline_status['metrics']:
            print(f"- Historical draws processed: {pipeline_status['metrics']['data_rows']}")
        if 'prediction_count' in pipeline_status['metrics']:
            print(f"- Numbers predicted: {pipeline_status['metrics']['prediction_count']}")
        if 'average_accuracy' in pipeline_status['metrics']:
            print(f"- Average accuracy: {pipeline_status['metrics']['average_accuracy']:.2f}%")
        
        print("\nExecution Times:")
        start_time = pipeline_status['timestamps']['collection_start']
        for stage, time in pipeline_status['timestamps'].items():
            if stage != 'collection_start':
                duration = (time - start_time).total_seconds()
                print(f"- {stage.replace('_', ' ').title()}: {duration:.2f} seconds")

        return pipeline_status

    except Exception as e:
        print(f"\nError in pipeline integration test: {str(e)}")
        traceback.print_exc()
        return None
# 7. Main Program Entry Point
def main():
    try:
        # Initialize system
        system_status = initialize_system()
        if not system_status['system_ready']:
            print("System initialization failed.")
            return
        
        print(f"\nSystem initialized at {system_status['start_time']}")
        print(f"User: {os.getenv('USER', 'Mihai-Edward')}")
        
        # Initialize core components
        handler = DrawHandler()
        predictor = LotteryPredictor(
            numbers_range=(1, 80),
            numbers_to_draw=20,
            use_combined_features=True
        )
        
        def display_menu():
            """Display the main menu with categories"""
            print("\n==========================")
            print("    KINO Draw Analyzer    ")
            print("==========================")
            print("DATA COLLECTION:")
            print("3. Update Historical Data")
            print("\nANALYSIS:")
            print("8. Complete Analysis & Save")
            print("\nPREDICTION:")
            print("9. Get ML prediction")
            print("\nEVALUATION:")
            print("10. Evaluate prediction accuracy")
            print("\nSYSTEM:")
            print("11. Run pipeline test")
            print("12. Run continuous learning cycle")
            print("13. Exit")
            print("==========================\n")

        def handle_data_collection():
            """Handle option 3: Update Historical Data"""
            print("\nUpdating historical data...")
            collector = KinoDataCollector(debug=True)
            draws = collector.fetch_latest_draws()
            
            if collector.collection_status['success']:
                print(f"\n✓ Successfully collected {collector.collection_status['draws_collected']} draws")
                if collector.sort_historical_draws():
                    print("✓ Historical data sorted successfully")
                    print(f"Last successful draw: {collector.collection_status['last_successful_draw']}")
                else:
                    print("! Warning: Could not sort historical data")
            else:
                print(f"\n✗ Data collection failed: {collector.collection_status['last_error']}")

        def handle_analysis():
            """Handle option 8: Complete Analysis"""
            print("\nPerforming complete analysis...")
            success = perform_complete_analysis()
            if success:
                print("\n✓ Complete analysis performed and saved successfully")
            else:
                print("\n✗ Failed to perform complete analysis")

        def handle_prediction():
            """Handle option 9: ML Prediction"""
            try:
                # Load and prepare historical data
                historical_data = handler.load_historical_data()
                if historical_data is None:
                    print("\nNo historical data available")
                    return

                # Get last 24 draws for analysis
                recent_draws = historical_data.tail(24)
                draws_for_analysis = []
                for _, row in recent_draws.iterrows():
                    try:
                        numbers = [int(row[f'number{i}']) for i in range(1, 21)]
                        if len(numbers) == 20:
                            draws_for_analysis.append((row['date'], numbers))
                    except Exception as e:
                        print(f"DEBUG: Error formatting draw: {e}")

                # Generate predictions
                if handler.train_ml_models():
                    predictions = predictor.predict_next_draw()
                    probabilities = predictor.get_prediction_probabilities()
                    analysis = handler.analyze_draws(draws_for_analysis)

                    if predictions is not None:
                        display_prediction_results(predictions, probabilities, analysis)
                    else:
                        print("\n✗ Failed to generate predictions")
                else:
                    print("\n✗ Model training failed")

            except Exception as e:
                print(f"\n✗ Error during prediction: {str(e)}")
                traceback.print_exc()

        def handle_prediction():
            """Handle option 9: ML Prediction"""
            try:
                # 1. Initial Output
                print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")
                
                print("\nInitializing prediction pipeline...")
                print("Pipeline stages:")
                print("- Model initialization")
                print("- Data preparation")
                print("- Feature engineering")
                print("- Model prediction")
                print("- Post-processing")
                
                # 2. Initialize both components
                predictor = LotteryPredictor(
                    numbers_range=(1, 80),
                    numbers_to_draw=20,
                    use_combined_features=True
                )
                handler = DrawHandler()
                
                print("\nChecking/Training models...")
                if handler.train_ml_models():
                    print("✓ Models ready")
                    
                    print("\nGenerating predictions...")
                    try:
                        # Load historical data using the main.py function
                        historical_data = load_data(PATHS['HISTORICAL_DATA'])
                        if historical_data is not None:
                            print(f"\nLoaded {len(historical_data)} historical draws")
                            
                            # Use last 24 draws for analysis
                            recent_draws = historical_data.head(24)
                            print(f"Using last {len(recent_draws)} draws for analysis")
                            
                            # Convert recent draws for handler
                            draws_for_analysis = [(row['date'], [row[f'number{i}'] for i in range(1, 21)])
                                                for _, row in recent_draws.iterrows()]
                            
                            # Get predictions from both components
                            predictions = predictor.predict_next_draw()
                            probabilities = predictor.get_prediction_probabilities()
                            analysis = handler.analyze_draws(draws_for_analysis)
                            
                            if predictions is not None:
                                next_draw = get_next_draw_time(datetime.now())
                                
                                print("\n=== Prediction Results ===")
                                print(f"Next Draw Time: {next_draw.strftime('%Y-%m-%d %H:%M')}")
                                print(f"Predicted Numbers: {', '.join(map(str, sorted(predictions)))}")
                                
                                if probabilities is not None:
                                    print("\n=== Probabilities ===")
                                    for num, prob in zip(sorted(predictions), probabilities):
                                        print(f"Number {num:2d}: {prob:.4f}")
                                
                                if analysis:
                                    print("\n=== Analysis Context ===")
                                    if 'hot_cold' in analysis:
                                        hot, cold = analysis['hot_cold']
                                        print("Hot Numbers (Top 5):")
                                        for num, freq in hot[:5]:
                                            print(f"Number {num}: {freq} times")
                                        print("\nCold Numbers (Top 5):")
                                        for num, freq in cold[:5]:
                                            print(f"Number {num}: {freq} times")
                                    
                                    if 'common_pairs' in analysis:
                                        print("\nMost Common Pairs (Top 3):")
                                        for pair, count in analysis['common_pairs'][:3]:
                                            print(f"Pair {pair}: {count} times")
                                
                                print("\n✓ Prediction completed successfully!")
                            else:
                                print("\n✗ Failed to generate predictions")
                        else:
                            print("\n✗ No historical data available")
                            
                    except Exception as e:
                        print(f"\n✗ Error in prediction generation: {e}")
                        traceback.print_exc()
                else:
                    print("\n✗ Model training failed")

            except Exception as e:
                print(f"\n✗ Error during prediction: {str(e)}")
                traceback.print_exc()
        
        def handle_pipeline_test():
            """Handle option 11: Pipeline Test"""
            print("\nRunning complete pipeline test...")
            print("This will execute steps 3->8->9->10 in sequence")
            confirm = input("Continue? (y/n): ")
            if confirm.lower() == 'y':
                status = test_pipeline_integration()
                if status and all(v for k, v in status.items() if k not in ['timestamps', 'metrics']):
                    print("\n✓ Complete pipeline test successful!")
                else:
                    print("\n✗ Some pipeline steps failed. Check the results above.")

        def handle_learning_cycle():
            """Handle option 12: Continuous Learning"""
            print("\nRunning continuous learning cycle...")
            if handler.run_continuous_learning_cycle():
                display_learning_metrics(handler.get_learning_metrics())
            else:
                print("\n✗ Continuous learning cycle failed")

        # Main program loop
        while True:
            display_menu()
            try:
                choice = input("Choose an option (3,8-13): ")
                
                options = {
                    '3': handle_data_collection,
                    '8': handle_analysis,
                    '9': handle_prediction,
                    '10': handle_evaluation,
                    '11': handle_pipeline_test,
                    '12': handle_learning_cycle,
                    '13': lambda: sys.exit(0)
                }
                
                if choice in options:
                    options[choice]()
                else:
                    print("\nInvalid option. Please choose 3,8-13")

            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                continue
            except Exception as e:
                print(f"\nError processing option: {str(e)}")
                traceback.print_exc()
                print("\nPlease try again.")

    except Exception as e:
        print(f"\nCritical error in main program: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()