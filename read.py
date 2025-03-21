import pickle
import os
from datetime import datetime
from pprint import pprint

def read_prediction_file(file_path):
    """Read and display prediction evaluation data from pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print("\n=== Prediction Evaluation Data ===\n")
        
        if isinstance(data, list):
            print(f"Found {len(data)} predictions\n")
            for idx, pred in enumerate(data, 1):
                print(f"\nPrediction #{idx}:")
                print("-" * 50)
                pprint(pred, indent=2, width=80)
        else:
            print("Single prediction record:")
            print("-" * 50)
            pprint(data, indent=2, width=80)
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    # Prediction evaluation file path
    eval_file = r"C:\Users\MihaiNita\OneDrive - Prime Batteries\Desktop\versiuni_de_care_nu_ma_ating\Versiune1.4\predictions\prediction_evaluations.pkl"
    
    if os.path.exists(eval_file):
        read_prediction_file(eval_file)
    else:
        print(f"File not found: {eval_file}")