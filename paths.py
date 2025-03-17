import os
import sys
import platform
from pathlib import Path
from datetime import datetime

from pandas import ExcelFile  # Added this import which was missing

# Calculate BASE_DIR dynamically
# This takes the directory of the current file (paths.py) and goes up one level to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Platform check for driver
IS_WINDOWS = platform.system().lower() == 'windows'
DRIVER_FILENAME = "msedgedriver.exe" if IS_WINDOWS else "msedgedriver"

# Complete PATHS dictionary with all required paths
PATHS = {
    'HISTORICAL_DATA': os.path.join(BASE_DIR, "src", "historical_draws.csv"),
    'PREDICTIONS_DIR': os.path.join(BASE_DIR, "data", "processed", "predictions"),
    # Fixed path - removed extra 'processed' directory
    'ALL_PREDICTIONS_FILE': os.path.join(BASE_DIR, "data", "processed", "all_predictions.xlsx"),
    'PREDICTIONS_METADATA_DIR': os.path.join(BASE_DIR, "data", "processed", "metadata"),
    'ANALYSIS': os.path.join(BASE_DIR, "data", "processed", "analysis_results.xlsx"),
    'MODELS_DIR': os.path.join(BASE_DIR, "models"),
    'DRIVER': os.path.join(BASE_DIR, "drivers", DRIVER_FILENAME),
    'PROCESSED_DIR': os.path.join(BASE_DIR, "data", "processed"),
    'SRC_DIR': os.path.join(BASE_DIR, "src"),
    'CONFIG_DIR': os.path.join(BASE_DIR, "config"),
    'LEARNING_DIR': os.path.join(BASE_DIR, "models", "learning_history"),
    'LEARNING_HISTORY_FILE': os.path.join(BASE_DIR, "models", "learning_history", "learning_history.csv"),
    'LEARNING_METRICS_FILE': os.path.join(BASE_DIR, "models", "learning_history", "learning_metrics.json"),
}

def get_project_root():
    """Get the project root directory"""
    return Path(BASE_DIR)

def ensure_directories():
    """Ensure all required directories exist"""
    try:
        # Define required directory structure
        required_dirs = [
            os.path.join(BASE_DIR, "data"),
            os.path.join(BASE_DIR, "data", "processed"),
            os.path.join(BASE_DIR, "data", "processed", "predictions"),
            os.path.join(BASE_DIR, "data", "processed", "metadata"),
            os.path.join(BASE_DIR, "models"),
            os.path.join(BASE_DIR, "drivers"),
            os.path.join(BASE_DIR, "src"),
            os.path.join(BASE_DIR, "config"),
            os.path.join(BASE_DIR, "models", "learning_history"),
        ]
        
        # Create core directories
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
        
        # Create parent directories for file paths
        file_paths = [path for name, path in PATHS.items() if not name.endswith('_DIR')]
        for file_path in file_paths:
            directory = os.path.dirname(file_path)
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured parent directory exists for: {file_path}")
        
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        return False

def validate_paths():
    """Validate all paths exist and are accessible"""
    missing_paths = []
    inaccessible_paths = []
    
    if not os.path.exists(BASE_DIR):
        print(f"\nERROR: Base directory does not exist: {BASE_DIR}")
        return False
    
    for name, path in PATHS.items():
        dir_path = path if name.endswith('_DIR') else os.path.dirname(path)
        
        if not os.path.exists(dir_path):
            missing_paths.append(f"{name}: {dir_path}")
            continue
            
        try:
            # For directories, test write permission in the directory
            if name.endswith('_DIR'):
                test_file = os.path.join(dir_path, '.test')
            else:
                test_file = os.path.join(os.path.dirname(path), '.test')
                
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except (IOError, OSError):
            inaccessible_paths.append(f"{name}: {dir_path}")
    
    if missing_paths or inaccessible_paths:
        if missing_paths:
            print("\nThe following paths are missing:")
            for path in missing_paths:
                print(f"- {path}")
        if inaccessible_paths:
            print("\nThe following paths are not writable:")
            for path in inaccessible_paths:
                print(f"- {path}")
        return False
        
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
    print(f"Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Current User's Login: {os.getenv('USER', 'Mihai-Edward')}")

def get_relative_path(path):
    """Get relative path from project root"""
    return os.path.relpath(path, BASE_DIR)

def get_predictions_path(timestamp=None):
    """Get the path for predictions with optional timestamp"""
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PATHS['PREDICTIONS_DIR'], f'prediction_{timestamp}.csv')

def get_metadata_path(timestamp=None):
    """Get the path for metadata with optional timestamp"""
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(PATHS['PREDICTIONS_METADATA_DIR'], f'prediction_{timestamp}_metadata.json')

if __name__ == "__main__":
    from datetime import datetime
    #print_system_info()
    ensure_directories()
    if validate_paths():
        #print("\nAll paths are valid and accessible")
        #print("\nProject directory structure:")
        for name, path in PATHS.items():
            print(f"- {name}: {path}")
        #print("\nProject is ready to run on this system")
        
        # Test prediction and metadata paths
        test_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #print(f"\nTest prediction path: {get_predictions_path(test_timestamp)}")
        #print(f"Test metadata path: {get_metadata_path(test_timestamp)}")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(PATHS['ALL_PREDICTIONS_FILE']), exist_ok=True)
    else:
        print("\nSome paths have issues. Please check the warnings above")