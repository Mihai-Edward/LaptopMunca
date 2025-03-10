import os
import sys
import platform
from pathlib import Path

# Keep original BASE_DIR definition for compatibility
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keep original platform check
IS_WINDOWS = platform.system().lower() == 'windows'
DRIVER_FILENAME = "msedgedriver.exe" if IS_WINDOWS else "msedgedriver"

# Keep your original PATHS dictionary but make it more robust
PATHS = {
    'HISTORICAL_DATA': os.path.join(BASE_DIR, "src", "historical_draws.csv"),
    'PREDICTIONS': os.path.join(BASE_DIR, "data", "processed", "predictions.csv"),
    'ANALYSIS': os.path.join(BASE_DIR, "data", "processed", "analysis_results.xlsx"),
    'MODELS_DIR': os.path.join(BASE_DIR, "models"),
    'DRIVER': os.path.join(BASE_DIR, "drivers", DRIVER_FILENAME),
    'PROCESSED_DIR': os.path.join(BASE_DIR, "data", "processed"),
    'SRC_DIR': os.path.join(BASE_DIR, "src"),
}

# Add new helper function but don't change existing ones
def get_project_root():
    """Get the project root directory in a reliable way"""
    return Path(BASE_DIR).resolve()

def ensure_directories():
    """Ensure all required directories exist with proper structure"""
    try:
        # Keep your original required directories
        required_dirs = [
            os.path.join(BASE_DIR, "data"),
            os.path.join(BASE_DIR, "data", "processed"),
            os.path.join(BASE_DIR, "models"),
            os.path.join(BASE_DIR, "drivers"),
            os.path.join(BASE_DIR, "src"),
            os.path.join(BASE_DIR, "config")
        ]
        
        # Create core directories
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
        
        # Create path directories
        for name, path in PATHS.items():
            directory = os.path.dirname(path)
            os.makedirs(directory, exist_ok=True)
            print(f"Ensured path exists for {name}: {directory}")
        
        return True
    except Exception as e:
        print(f"Error creating directories: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"BASE_DIR: {BASE_DIR}")
        return False

# Keep all your original functions
def validate_paths():
    """Validate that all required paths exist and are accessible"""
    missing_paths = []
    inaccessible_paths = []
    
    if not os.path.exists(BASE_DIR):
        print(f"\nERROR: Base directory does not exist: {BASE_DIR}")
        return False
    
    for name, path in PATHS.items():
        dir_path = os.path.dirname(path)
        
        if not os.path.exists(dir_path):
            missing_paths.append(f"{name}: {dir_path}")
            continue
            
        try:
            test_file = os.path.join(dir_path, '.test')
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
    """Get the absolute path for a given path key"""
    if path_key not in PATHS:
        raise KeyError(f"Unknown path key: {path_key}")
    return os.path.abspath(PATHS[path_key])

def print_system_info():
    """Print system information for debugging"""
    print("\nSystem Information:")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"User Home: {os.path.expanduser('~')}")

# Add new helper function for relative paths
def get_relative_path(path):
    """Convert absolute path to relative path from project root"""
    return os.path.relpath(path, BASE_DIR)

if __name__ == "__main__":
    print_system_info()
    ensure_directories()
    if validate_paths():
        print("\nAll paths are valid and accessible")
        print("\nProject is ready to run on this system")
    else:
        print("\nSome paths have issues. Please check the warnings above")