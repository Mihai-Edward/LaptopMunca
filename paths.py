import os
import sys
import platform
from pathlib import Path
from datetime import datetime

# Calculate BASE_DIR dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Platform check for driver
IS_WINDOWS = platform.system().lower() == 'windows'
DRIVER_FILENAME = "msedgedriver.exe" if IS_WINDOWS else "msedgedriver"

# Paths dictionary with all necessary paths
PATHS = {
    'BASE_DIR': BASE_DIR,
    'SRC_DIR': os.path.join(BASE_DIR, "src"),
    'DATA_DIR': os.path.join(BASE_DIR, "data"),
    'MODELS_DIR': os.path.join(BASE_DIR, "models"),
    'PREDICTIONS_DIR': os.path.join(BASE_DIR, "predictions"),
    'DRIVER': os.path.join(BASE_DIR, "drivers", DRIVER_FILENAME),
    'DRIVERS_DIR': os.path.join(BASE_DIR, "drivers"),
    
    # Data files
    'HISTORICAL_DATA': os.path.join(BASE_DIR, "src", "historical_draws.csv"),
    'PROCESSED_DATA': os.path.join(BASE_DIR, "data", "processed"),
    'ANALYSIS_RESULTS': os.path.join(BASE_DIR, "data", "processed", "analysis_results.xlsx"),
    
    # Model files
    'MODEL_METADATA': os.path.join(BASE_DIR, "models", "model_metadata.pkl"),
    'MODEL_PREDICTIONS': os.path.join(BASE_DIR, "predictions", "prediction_history.pkl"),
    'EVALUATION_RESULTS': os.path.join(BASE_DIR, "predictions", "prediction_evaluations.pkl"),
}

def get_project_root():
    """Get the project root directory"""
    return Path(BASE_DIR)

def ensure_directories():
    """Ensure all required directories exist"""
    try:
        # Create main directories
        required_dirs = [
            os.path.join(BASE_DIR, "data", "raw"),
            os.path.join(BASE_DIR, "data", "processed"),
            os.path.join(BASE_DIR, "models"),
            os.path.join(BASE_DIR, "predictions"),
            os.path.join(BASE_DIR, "src"),
            os.path.join(BASE_DIR, "drivers"),
        ]
        
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
            
        # For files, create their parent directories
        for path in PATHS.values():
            if not path.endswith('_DIR'):
                directory = os.path.dirname(path)
                os.makedirs(directory, exist_ok=True)
                print(f"Ensured parent directory exists for: {path}")
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False

def validate_paths():
    """Validate all paths exist and are accessible"""
    for name, path in PATHS.items():
        if not os.path.exists(os.path.dirname(path)):
            print(f"Warning: Directory does not exist: {os.path.dirname(path)}")
            continue
        
        if not os.access(os.path.dirname(path), os.W_OK):
            print(f"Warning: Path is not writable: {path}")
            continue
    return True

def get_absolute_path(path_key):
    """Get absolute path for a key"""
    if path_key not in PATHS:
        raise KeyError(f"Unknown path key: {path_key}")
    return os.path.abspath(PATHS[path_key])

def print_system_info():
    """Print system information"""
    print("\nSystem Information:")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Base Directory: {BASE_DIR}")
    # Use actual current time
    current_time = datetime.now().strftime('%H:%M  %d-%m-%Y')
    print(f"Current Date and Time: {current_time}")  # Add this line to print the time
    print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}")

if __name__ == "__main__":
    print_system_info()
    ensure_directories()
    if validate_paths():
        print("\nAll paths are valid and accessible")
        print("\nProject directory structure:")
        for name, path in PATHS.items():
            print(f"- {name}: {path}")
        print("\nProject is ready to run on this system")