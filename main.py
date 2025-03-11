import sys
import os
import traceback
import pandas as pd
import numpy as np
from lottery_predictor import LotteryPredictor
from datetime import datetime, timedelta
import pytz
from data_collector_selenium import KinoDataCollector
from data_analysis import DataAnalysis
from draw_handler import DrawHandler
from prediction_evaluator import PredictionEvaluator
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.paths import PATHS, ensure_directories

# 1. System Initialization
def initialize_system():
    """Initialize system and ensure all directories exist"""
    ensure_directories()
    return {
        'start_time': datetime.now(),
        'system_ready': True
    }

def get_next_draw_time(current_time):
    """Calculate the next draw time based on current time"""
    minutes = (current_time.minute // 5 + 1) * 5
    next_draw_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
    return next_draw_time

# 2. Data Loading and Processing
def load_data(file_path=None):
    """Load and preprocess historical data"""
    if file_path is None:
        file_path = PATHS['HISTORICAL_DATA']
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    df = pd.read_csv(file_path)
    try:
        df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y', errors='coerce')
        df.loc[df['date'].isna(), 'date'] = pd.to_datetime(df.loc[df['date'].isna(), 'date'], errors='coerce')
    except Exception as e:
        print(f"Warning: Date conversion issue: {e}")
    
    number_cols = [f'number{i+1}' for i in range(20)]
    try:
        df[number_cols] = df[number_cols].astype(float)
    except Exception as e:
        print(f"Warning: Could not process number columns: {e}")
    
    for col in number_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)

    return df

def extract_date_features(df):
    """Extract temporal features from date column"""
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    return df
# 3. Data Collection
def run_data_collector_standalone():
    """Run the data collector in standalone mode"""
    print("\n--- Running Data Collector ---")
    
    collector = KinoDataCollector(debug=True)
    
    # First try to sort existing data
    print("\nSorting historical draws...")
    collector.sort_historical_draws()
    
    # Then fetch new draws
    draws = collector.fetch_latest_draws(num_draws=24)  # Default value from original
    if draws:
        print("\nCollected draws:")
        for draw_date, numbers in draws:
            print(f"Date: {draw_date}, Numbers: {', '.join(map(str, numbers))}")
        
        # Sort again after collecting new draws
        print("\nSorting updated historical draws...")
        if collector.sort_historical_draws():
            print("Historical draws successfully sorted from newest to oldest")
        else:
            print("Error occurred while sorting draws")
            
    print("\nCollection Status:", collector.collection_status)
    
    return draws

def save_top_4_numbers_to_excel(top_4_numbers, file_path=None):
    """Save top 4 numbers to Excel file"""
    if file_path is None:
        file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS']), 'top_4.xlsx')
    
    df = pd.DataFrame({'Top 4 Numbers': top_4_numbers})
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_excel(file_path, index=False)

def evaluate_numbers(historical_data):
    """
    Evaluate numbers based on criteria other than frequency.
    For simplicity, this example assumes a dummy evaluation function.
    Replace this with your actual evaluation logic.
    """
    number_evaluation = {i: 0 for i in range(1, 81)}
    for index, row in historical_data.iterrows():
        for i in range(1, 21):
            number = row[f'number{i}']
            number_evaluation[number] += 1  # Dummy evaluation logic

    # Sort numbers by evaluation score in descending order
    sorted_numbers = sorted(number_evaluation, key=number_evaluation.get, reverse=True)
    return sorted_numbers[:4]  # Return top 4 numbers
# 4. Data Analysis
def perform_complete_analysis(draws=None):
    """Perform comprehensive analysis using DataAnalysis class"""
    try:
        # If no draws provided, load from historical_draws.csv
        if not draws:
            historical_data = load_data(PATHS['HISTORICAL_DATA'])
            if historical_data is None:
                print("\nNo historical data available for analysis")
                return False
            
            # Convert DataFrame to draws format expected by DataAnalysis
            draws = [(row['date'], [row[f'number{i}'] for i in range(1, 21)])
                    for _, row in historical_data.iterrows()]
        
        # Initialize analysis
        analysis = DataAnalysis(draws)
        
        # Perform comprehensive analysis
        analysis_data = {
            'frequency': analysis.count_frequency(),
            'hot_cold': analysis.hot_and_cold_numbers(),
            'common_pairs': analysis.find_common_pairs(),
            'range_analysis': analysis.number_range_analysis()
        }
        
        # Save to Excel using config path
        excel_path = PATHS['ANALYSIS']
        os.makedirs(os.path.dirname(excel_path), exist_ok=True)
        
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
                'Count': [x[1] for x in hot_numbers[:20]],
                'Cold Numbers': [x[0] for x in cold_numbers[:20]],
                'Count': [x[1] for x in cold_numbers[:20]]
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
        
        print(f"\nComplete analysis saved to: {excel_path}")
        return True
            
    except Exception as e:
        print(f"\nError in complete analysis: {str(e)}")
        traceback.print_exc()
        return False
def check_and_train_model():
    """Check if a trained model exists and train if needed using DrawHandler"""
    try:
        handler = DrawHandler()
        if handler.train_ml_models():
            print("✓ Model training completed successfully")
            return True
        else:
            print("✗ Model training failed")
            return False
            
    except Exception as e:
        print(f"Error checking/training model: {e}")
        return False
def train_and_predict():
    """Generate predictions using last 24 draws for analysis like lottery_predictor.py"""
    try:
        print("\nInitializing prediction pipeline...")
        
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
                    print(f"Loaded {len(historical_data)} historical draws")
                    
                    # Only use last 24 draws for analysis
                    recent_draws = historical_data.tail(24)
                    print(f"Using last {len(recent_draws)} draws for analysis")
                    
                    # Convert recent draws to the format expected by handler
                    draws_for_analysis = [(row['date'], [row[f'number{i+1}'] for i in range(20)])
                                        for _, row in recent_draws.iterrows()]
                    
                    # Generate prediction using handler's pipeline with recent draws
                    predictions, probabilities, analysis = handler.handle_prediction_pipeline(draws_for_analysis)
                    
                    if predictions is not None:
                        print("\n=== Prediction Results ===")
                        print(f"Predicted numbers: {', '.join(map(str, sorted(predictions)))}")
                        
                        if analysis:
                            print("\n=== Analysis Context ===")
                            if 'frequency' in analysis:
                                print(f"frequency: {analysis['frequency']}")
                            if 'hot_cold' in analysis:
                                print(f"hot_cold: {analysis['hot_cold']}")
                            if 'common_pairs' in analysis:
                                print(f"common_pairs: {analysis['common_pairs']}")
                            if 'range_analysis' in analysis:
                                print(f"range_analysis: {analysis['range_analysis']}")
                        
                        return predictions, probabilities, analysis
                    else:
                        print("\n✗ Failed to generate predictions")
                        return None, None, None
                else:
                    print("\nNo historical data available")
                    return None, None, None
                    
            except Exception as e:
                print(f"\n✗ Error in prediction pipeline: {str(e)}")
                traceback.print_exc()
                return None, None, None
        else:
            print("\n✗ Model training failed")
            return None, None, None
            
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        traceback.print_exc()
        return None, None, None
            
    except Exception as e:
        print(f"\nDEBUG: Critical error: {str(e)}")
        print("DEBUG: Stack trace:")
        traceback.print_exc()
        return None, None, None
def test_pipeline_integration():
    """Test the integrated prediction pipeline with enhanced monitoring"""
    try:
        handler = DrawHandler()
        pipeline_status = {
            'data_collection': False,
            'analysis': False,
            'prediction': False,
            'evaluation': False,
            'timestamps': {},
            'metrics': {}
        }

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
            
            predictions, probabilities, analysis = handler.handle_prediction_pipeline()
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
        return pipeline_status

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
        
        # Initialize core components once
        handler = DrawHandler()
        
        while True:
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

            try:
                choice = input("Choose an option (3,8-13): ")
                
                if choice == '3':
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
                
                elif choice == '8':
                    print("\nPerforming complete analysis...")
                    success = perform_complete_analysis()
                    if success:
                        print("\n✓ Complete analysis performed and saved successfully")
                    else:
                        print("\n✗ Failed to perform complete analysis")
                
                elif choice == '9':
                    print(f"\nCurrent Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}\n")
                    
                    print("\nInitializing prediction pipeline...")
                    print("Pipeline initialization complete with stages:")
                    print("- data_preparation")
                    print("- feature_engineering")
                    print("- model_prediction")
                    print("- post_processing\n")
                    
                    try:
                        # First initialize LotteryPredictor
                        print("Initialized LotteryPredictor:")
                        print("- Number range: (1, 80)")
                        print("- Numbers to draw: 20")
                        print("- Number of classes: 80")
                        print("- Using combined features: True\n")
                        
                        predictor = LotteryPredictor(
                            numbers_range=(1, 80),
                            numbers_to_draw=20,
                            use_combined_features=True
                        )
                        
                        # Then initialize DrawHandler and load its settings
                        print("Loading learning status...")
                        handler = DrawHandler()
                        print("Loaded existing learning metrics")
                        print("DrawHandler initialized successfully")
                        print(f"Using models directory: {PATHS['MODELS_DIR']}")
                        print(f"Predictions directory: {PATHS['PREDICTIONS']}\n")
                        
                        print("Checking/Training models...")
                        if handler.train_ml_models():
                            print("✓ Models ready")
                            
                            print("\nGenerating predictions...")
                            # Load and prepare data
                            historical_data = load_data(PATHS['HISTORICAL_DATA'])
                            if historical_data is not None:
                                print(f"DEBUG: Loaded {len(historical_data)} historical draws")
                                
                                # Use last 24 draws for analysis
                                recent_draws = historical_data.tail(24)
                                print(f"Using last {len(recent_draws)} draws for analysis")
                                
                                # Convert recent draws for analysis
                                draws_for_analysis = [(row['date'], [row[f'number{i+1}'] for i in range(20)])
                                                    for _, row in recent_draws.iterrows()]
                                
                                # Use both predictor and handler for predictions
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
                                    
                                    # Save predictions
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    prediction_path = os.path.join(PATHS['PREDICTIONS'], f'prediction_{timestamp}.csv')
                                    metadata_path = os.path.join(PATHS['PREDICTIONS'], 'metadata', f'prediction_{timestamp}_metadata.json')
                                    
                                    # Save using both predictor and handler methods
                                    predictor.save_prediction(predictions, prediction_path)
                                    handler.save_prediction_metadata(metadata_path, predictions, probabilities, analysis)
                                    
                                    print(f"\nPredictions saved successfully:")
                                    print(f"- CSV file: {prediction_path}")
                                    print(f"- Metadata: {metadata_path}")
                                    
                                    print("\n✓ Prediction completed successfully!")
                                else:
                                    print("\n✗ Failed to generate predictions")
                            else:
                                print("\nNo historical data available")
                                
                        else:
                            print("\n✗ Model training failed")
                            
                    except Exception as e:
                        print(f"\n✗ Error during prediction: {str(e)}")
                        traceback.print_exc()
                
                elif choice == '10':
                    print("\nStarting prediction evaluation...")
                    try:
                        evaluator = PredictionEvaluator()
                        evaluator.evaluate_past_predictions()
                        print("\n✓ Evaluation complete")
                    except Exception as e:
                        print(f"\n✗ Error during evaluation: {e}")
                
                elif choice == '11':
                    print("\nRunning complete pipeline test...")
                    print("This will execute steps 3->8->9->10 in sequence")
                    confirm = input("Continue? (y/n): ")
                    if confirm.lower() == 'y':
                        status = test_pipeline_integration()
                        if all(v for k, v in status.items() if k not in ['timestamps', 'metrics']):
                            print("\n✓ Complete pipeline test successful!")
                        else:
                            print("\n✗ Some pipeline steps failed. Check the results above.")
                
                elif choice == '12':
                    print("\nRunning continuous learning cycle...")
                    if handler.run_continuous_learning_cycle():
                        metrics = handler.get_learning_metrics()
                        print("\nLearning Cycle Results:")
                        print(f"- Learning cycles completed: {metrics['cycles_completed']}")
                        if 'current_accuracy' in metrics:
                            print(f"- Current accuracy: {metrics['current_accuracy']:.2f}%")
                        if 'improvement_rate' in metrics:
                            print(f"- Total improvement: {metrics['improvement_rate']:.2f}%")
                        if 'last_adjustments' in metrics and metrics['last_adjustments']:
                            print("\nRecent model adjustments:")
                            for adj in metrics['last_adjustments']:
                                print(f"- {adj}")
                    else:
                        print("\n✗ Continuous learning cycle failed")
                
                elif choice == '13':
                    print("\nExiting program...")
                    sys.exit(0)
                
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