import pandas as pd
from lottery_predictor import LotteryPredictor

def load_excel_file(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        print(f"Successfully loaded {file_path}")
        
        # List all sheet names
        sheet_names = xls.sheet_names
        print(f"Sheet names: {sheet_names}")
        
        # Load each sheet into a DataFrame and inspect
        for sheet in sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet)
            print(f"\nSheet: {sheet}")
            print(df.head())  # Print the first few rows of each sheet
        return xls
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def main():
    # Correct path to the analysis_results.xlsx file
    analysis_file_path = 'C:/Users/MihaiNita/OneDrive - Prime Batteries/Desktop/versiuni_de_care_nu_ma_ating/Versiune1.4/data/processed/analysis_results.xlsx'
    
    # Load and inspect the analysis_results.xlsx file
    load_excel_file(analysis_file_path)
    
    # Historical draws data (as provided in the conversation)
    historical_draws = """
    date,number1,number2,number3,number4,number5,number6,number7,number8,number9,number10,number11,number12,number13,number14,number15,number16,number17,number18,number19,number20
    07:55  11-03-2025,5,16,18,19,21,24,31,33,34,44,47,55,56,57,59,62,67,73,74,75
    07:50  11-03-2025,2,7,10,18,24,25,28,30,32,39,46,50,55,56,60,64,65,76,78,79
    07:45  11-03-2025,1,3,9,29,33,34,39,46,50,53,55,59,61,64,65,66,69,72,74,77
    07:40  11-03-2025,13,21,22,23,27,28,29,32,35,37,39,40,42,50,52,61,69,71,76,77
    07:35  11-03-2025,3,6,9,12,15,24,26,32,34,36,37,39,45,46,47,59,60,65,66,70
    07:30  11-03-2025,4,9,10,11,22,26,30,40,42,44,45,56,60,64,67,69,70,72,75,76
    07:25  11-03-2025,10,12,15,23,26,29,34,37,38,47,48,52,55,57,59,66,68,70,71,77
    07:20  11-03-2025,8,12,14,15,16,19,22,28,33,40,41,47,54,57,59,61,69,71,74,80
    07:15  11-03-2025,6,7,14,18,22,23,25,26,31,32,33,36,46,47,51,66,68,69,72,79
    07:10  11-03-2025,80,5,17,18,20,25,26,28,35,36,46,58,60,62,63,64,71,74,76,80
    07:05  11-03-2025,6,10,11,17,18,19,21,23,28,31,43,46,48,50,55,56,64,76,78,80
    07:00  11-03-2025,1,8,10,11,14,15,16,17,20,21,46,48,52,60,61,67,68,73,75,76
    """
    
    # Convert historical draws to DataFrame
    from io import StringIO
    historical_data = pd.read_csv(StringIO(historical_draws.strip()), delimiter=',')
    historical_data['date'] = historical_data['date'].str.strip()  # Remove leading/trailing spaces
    historical_data['date'] = pd.to_datetime(historical_data['date'], format='%H:%M %d-%m-%Y')
    
    # Initialize LotteryPredictor with additional debugging
    class DebugLotteryPredictor(LotteryPredictor):
        def _create_feature_vector(self, window):
            print(f"Creating feature vector for window:\n{window}")
            feature_vector = super()._create_feature_vector(window)
            print(f"Created feature vector: {feature_vector}")
            return feature_vector
        
        def prepare_data(self, historical_data):
            try:
                return super().prepare_data(historical_data)
            except Exception as e:
                print(f"Error in prepare_data: {e}")
                raise e
    
    predictor = DebugLotteryPredictor()
    
    # Run prepare_data and print debugging information
    try:
        features, labels = predictor.prepare_data(historical_data)
        
        if features is not None and labels is not None:
            print("\nData Preparation Successful!")
            print(f"Features shape: {features.shape}")
            print(f"Labels shape: {labels.shape}")
        else:
            print("\nData Preparation Failed!")
    except Exception as e:
        print(f"Exception during data preparation: {e}")

if __name__ == "__main__":
    main()