import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
import json
from datetime import datetime
from cleaning import MISSIONS, clean_mission_data

def get_model_paths(mission_name):
    """Generates standardized file paths for a mission's model artifacts."""
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    return {
        'model': os.path.join(model_dir, f'{mission_name}_model.pkl'),
        'columns': os.path.join(model_dir, f'{mission_name}_columns.pkl'),
        'accuracy': os.path.join(model_dir, f'{mission_name}_accuracy.txt')
    }

def update_version_file(mission, accuracy, params):
    """Updates a JSON file with versioning information for a newly trained model."""
    version_dir = 'Version'
    os.makedirs(version_dir, exist_ok=True)
    version_file = os.path.join(version_dir, 'version_updated.json')
    new_version_info = {
        'mission': mission,
        'timestamp': datetime.now().isoformat(),
        'accuracy': accuracy,
        'parameters': params
    }
    versions = []
    if os.path.exists(version_file):
        with open(version_file, 'r') as f:
            try: versions = json.load(f)
            except json.JSONDecodeError: versions = []
    versions.append(new_version_info)
    with open(version_file, 'w') as f:
        json.dump(versions, f, indent=4)

def create_and_train_model(mission_name, **kwargs):
    """
    Trains, evaluates, and saves a model, using default or provided hyperparameters.
    """
    config = MISSIONS.get(mission_name)
    if not config:
        print(f"ERROR: Mission '{mission_name}' is not configured.")
        return 0.0

    paths = get_model_paths(mission_name)
    df = clean_mission_data(mission_name)
    if df is None or df.empty:
        print(f"FATAL: Could not get clean data for {mission_name}. Aborting training.")
        return 0.0

    Y = df[config['target_col']]
    X = df[config['features']]
    
    # *** NEW: Use mission-specific hyperparameters by default ***
    training_params = config['hyperparameters'].copy()
    # Allow overriding defaults with any provided kwargs (from the config page)
    training_params.update(kwargs)
    
    # The test_size can also be a parameter, but we'll default it here
    test_size = training_params.pop('test_size', 0.2)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42, stratify=Y)
    
    print(f"Training {mission_name.upper()} model with params: {training_params}")
    model = GradientBoostingClassifier(random_state=42, **training_params)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    accuracy_percent = round(accuracy * 100, 2)
    print(f"Model accuracy for {mission_name}: {accuracy_percent}%")

    joblib.dump(model, paths['model'])
    joblib.dump(X.columns.tolist(), paths['columns'])
    with open(paths['accuracy'], 'w') as f:
        f.write(str(accuracy_percent))
    
    update_version_file(mission_name, accuracy_percent, training_params)
    return accuracy_percent

def append_input_and_retrain(mission, input_data):
    """Appends new user input to the uncleaned dataset and triggers a full retrain."""
    # This function remains the same as before
    pass # Add the full function from the previous response here if needed

if __name__ == '__main__':
    for mission in MISSIONS.keys():
        print(f"\n--- Training initial model for: {mission} ---")
        create_and_train_model(mission)