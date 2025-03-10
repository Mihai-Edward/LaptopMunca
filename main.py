import sys
import os
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

def initialize_system():
    """Initialize system and ensure all directories exist"""
    ensure_directories()
    return {
        'start_time': datetime.now(),
        'system_ready': True
    }

def check_and_train_model():
    """Check if a trained model exists and train if needed using DrawHandler"""
    try:
        handler = DrawHandler()
        model_path = handler._get_latest_model()
        
        if not model_path:
            print("No trained model found. Training new model...")
            if handler.train_ml_models():
                print("✓ Model training completed successfully")
                return True
            else:
                print("✗ Model training failed")
                return False
        else:
            # Verify model files exist
            model_files = [
                f"{model_path}_prob_model.pkl",
                f"{model_path}_pattern_model.pkl",
                f"{model_path}_scaler.pkl"
            ]
            if all(os.path.exists(file) for file in model_files):
                print(f"✓ Model found: {os.path.basename(model_path)}")
                return True
            else:
                print("Model files incomplete. Attempting retraining...")
                return handler.train_ml_models()
                
    except Exception as e:
        print(f"Error checking/training model: {e}")
        return False

def load_data(file_path=None):
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
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    return df

def get_next_draw_time(current_time):
    minutes = (current_time.minute // 5 + 1) * 5
    next_draw_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
    return next_draw_time

def save_top_4_numbers_to_excel(top_4_numbers, file_path=None):
    if file_path is None:
        file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS']), 'top_4.xlsx')
    
    df = pd.DataFrame({'Top 4 Numbers': top_4_numbers})
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_excel(file_path, index=False)

def save_standardized_prediction(predictions, next_draw_time, probabilities=None):
    """
    Save predictions in standardized format to a single file (predictions.csv)
    Format: Date,Predicted_Numbers,Probabilities
    """
    try:
        # Create predictions directory if it doesn't exist
        predictions_dir = os.path.dirname(PATHS['PREDICTIONS'])
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Prepare the data in standardized format
        sorted_predictions = sorted(predictions)
        formatted_date = next_draw_time.strftime('%H:%M %d-%m-%Y')
        
        # Create data dictionary for DataFrame
        prediction_data = {
            'Date': [formatted_date],
            'Predicted_Numbers': [','.join(map(str, sorted_predictions))]
        }
        
        # Add probabilities if they exist
        if probabilities is not None:
            probability_str = ','.join(f"{probabilities[num-1]:.4f}" for num in sorted_predictions)
            prediction_data['Probabilities'] = [probability_str]
        else:
            prediction_data['Probabilities'] = ['']
        
        # Create DataFrame
        df = pd.DataFrame(prediction_data)
        
        # Save to single predictions file
        prediction_file = PATHS['PREDICTIONS']
        mode = 'a' if os.path.exists(prediction_file) else 'w'
        header = not os.path.exists(prediction_file)
        df.to_csv(prediction_file, mode=mode, header=header, index=False)
        
        print(f"\nPrediction saved to: {prediction_file}")
        print(f"Date: {formatted_date}")
        print(f"Numbers: {','.join(map(str, sorted_predictions))}")
        if probabilities is not None:
            print("Probabilities included")
        
        return True
        
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

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

def train_and_predict():
    try:
        handler = DrawHandler()
        
        # First ensure models are trained
        print("Checking/Training models...")
        if handler.train_ml_models():
            print("Models ready")
        else:
            raise Exception("Model training failed")
        
        # Generate prediction using pipeline
        print("\nGenerating predictions...")
        predictions, probabilities, analysis = handler.handle_prediction_pipeline()
        
        if predictions is not None:
            # Format and display predictions
            formatted_numbers = ','.join(map(str, sorted(predictions)))
            next_draw_time = get_next_draw_time(datetime.now())
            print(f"\nPredicted numbers for next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}:")
            print(f"Numbers: {formatted_numbers}")
            
            # Display probabilities in a readable format
            print("\nProbabilities for each predicted number:")
            for num, prob in zip(sorted(predictions), 
                               [probabilities[num - 1] for num in predictions]):
                print(f"Number {num}: {prob:.4f}")

            # Save predictions in standardized format (remove duplicate saving)
            save_standardized_prediction(predictions, next_draw_time, probabilities)

            # Handle top 4 numbers if available
            if analysis and 'hot_numbers' in analysis:
                top_4_numbers = analysis['hot_numbers'][:4]
                top_4_file_path = os.path.join(os.path.dirname(PATHS['PREDICTIONS']), 'top_4.xlsx')
                save_top_4_numbers_to_excel(top_4_numbers, top_4_file_path)
                print(f"\nTop 4 numbers based on analysis: {','.join(map(str, top_4_numbers))}")
            
            return predictions, probabilities, analysis
        else:
            print("\nFailed to generate predictions")
            return None, None, None
            
    except Exception as e:
        print(f"\nError in prediction process: {str(e)}")
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

def run_data_collector_standalone():
    """
    Run the data collector in standalone mode just like running the data_collector_selenium.py script directly.
    This function exactly mimics the behavior of the __main__ block in data_collector_selenium.py.
    """
    print("\n--- Running Data Collector in Standalone Mode ---")
    
    # Create a collector instance
    collector = KinoDataCollector()
    
    # First try to sort existing data
    print("\nSorting historical draws...")
    collector.sort_historical_draws()
    
    # Then fetch new draws
    draws = collector.fetch_latest_draws()
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
    print("\n--- Data Collection Complete ---")
    
    return draws

def main():
    # Initialize system
    system_status = initialize_system()
    if not system_status['system_ready']:
        print("System initialization failed.")
        return
    
    print(f"\nSystem initialized at {system_status['start_time']}")
    
    # Initialize core components
    collector = KinoDataCollector()
    handler = DrawHandler()  # Initialize handler once here
    draws = None

   
    while True:
        print("\n==========================")
        print("3. Fetch latest draws from lotostats.ro")
        print("8. Complete Analysis & Save")
        print("9. Get ML prediction")
        print("10. Evaluate prediction accuracy")
        print("11. Run pipeline test")
        print("12. Run continuous learning cycle")  # New option
        print("13. Exit")  # Updated exit option
        print("==========================\n")

        try:
            choice = input("Choose an option (3,8-13): ")
            
            if choice == '3':
                # Run data collector in standalone mode to mimic behavior of data_collector_selenium.py
                draws = run_data_collector_standalone()
            
            elif choice == '8':
                success = perform_complete_analysis(draws)
                if success:
                    print("\nComplete analysis performed and saved successfully")
                else:
                    print("\nFailed to perform complete analysis")
            
            elif choice == '9':
                # Use the existing handler instead of creating a new one in check_and_train_model
                                # Use the existing handler instead of creating a new one in check_and_train_model
                if check_and_train_model():
                    print("\nGenerating ML prediction for next draw...")
                    predictions, probabilities, analysis = train_and_predict()
                    if predictions is not None:
                        print("\nPrediction process completed successfully!")
                        # Results are already displayed in train_and_predict
                    else:
                        print("\nPrediction generation failed after model training")
                else:
                    print("\nFailed to prepare model for predictions")
            
            elif choice == '10':
                print("\nStarting prediction evaluation...")
                try:
                    evaluator = PredictionEvaluator()
                    evaluator.evaluate_past_predictions()
                    print("Evaluation complete")
                except Exception as e:
                    print(f"\nError during evaluation: {e}")
                    
            elif choice == '11':
                print("\nRunning complete pipeline...")
                print("This will execute steps 3->8->9->10 in sequence")
                confirm = input("Continue? (y/n): ")
                if confirm.lower() == 'y':
                    # Use the existing handler in test_pipeline_integration
                    status = test_pipeline_integration()
                    print("\nPipeline Test Results:")
                    for step, success in status.items():
                        print(f"{step}: {'✓' if success else '✗'}")
                    
                    if all(status.values()):
                        print("\nComplete pipeline test successful!")
                    else:
                        print("\nSome pipeline steps failed. Check the results above.")
            
             
            elif choice == '12':
                print("\nRunning continuous learning cycle...")
                if handler.run_continuous_learning_cycle():
                    metrics = handler.get_learning_metrics()
                    print("\nContinuous Learning Results:")
                    print(f"- Learning cycles completed: {metrics['cycles_completed']}")
                    print(f"- Current prediction accuracy: {metrics['current_accuracy']:.2f}%" if metrics['current_accuracy'] else "- Current accuracy: Not available")
                    print(f"- Total improvement: {metrics['improvement_rate']:.2f}%" if metrics['improvement_rate'] else "- Total improvement: Not available")
                    print("\nMost recent adjustments:")
                    if metrics['last_adjustments']:
                        for adj in metrics['last_adjustments']:
                            print(f"- {adj}")
                    else:
                        print("- No adjustments made")
                else:
                    print("\nContinuous learning cycle did not complete successfully")
            elif choice == '13':
                print("\nExiting program...")
                sys.exit(0)
            else:
                print("\nInvalid option. Please choose 3,8-13")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()