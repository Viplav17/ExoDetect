import pandas as pd
import os

# --- Central Configuration for Missions ---
# K2 'target_col' and 'features' have been updated to match your specific CSV files.
MISSIONS = {
    'kepler': {
        'uncleaned_path': 'Data/Kepler_Uncleaned.csv',
        'cleaned_path': 'Data/Kepler_Cleaned.csv',
        'target_col': 'koi_disposition',
        'id_cols': ['kepid', 'kepoi_name'],
        'name_cols': ['kepoi_name'],
        'drop_cols': ['kepid', 'kepler_name', 'koi_pdisposition', 'koi_score'],
        'positive_label': 'CONFIRMED',
        'negative_label': 'FALSE POSITIVE',
        'candidate_label': 'CANDIDATE',
        'features': [
            'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec',
            'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration',
            'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag'
        ],
        'hyperparameters': {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.1}
    },
    'tess': {
        'uncleaned_path': 'Data/TESS_Uncleaned.csv',
        'cleaned_path': 'Data/TESS_Cleaned.csv',
        'target_col': 'tfopwg_disp',
        'id_cols': ['tid', 'toi'],
        'name_cols': ['toi'],
        'drop_cols': ['tid', 'toi', 'st_cenra', 'st_cendec'],
        'positive_label': 'CP',
        'negative_label': 'FP',
        'candidate_label': 'PC',
        'features': ['pl_orbper', 'pl_rade', 'pl_insol', 'st_teff', 'st_rad'],
        'hyperparameters': {'n_estimators': 250, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.9}
    },
    'k2': {
        'uncleaned_path': 'Data/K2_Uncleaned.csv',
        'cleaned_path': 'Data/K2_Cleaned.csv',
        'target_col': 'disposition',  # CORRECTED from 'pl_disposition'
        'id_cols': ['pl_name'],
        'name_cols': ['pl_name'],
        'drop_cols': ['hostname', 'pl_letter'],
        'positive_label': 'CONFIRMED',
        'negative_label': 'FALSE POSITIVE',
        'candidate_label': 'CANDIDATE',
        'features': ['pl_orbper', 'pl_rade', 'pl_insol', 'st_teff', 'st_rad'], # CORRECTED from 'pl_ror' to 'pl_rade'
        'hyperparameters': {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1}
    }
}

def clean_mission_data(mission_name):
    """
    Cleans data for a specific mission with improved error handling.
    """
    if mission_name not in MISSIONS:
        print(f"ERROR: Mission '{mission_name}' is not configured.")
        return None

    config = MISSIONS[mission_name]
    
    try:
        df = pd.read_csv(config['uncleaned_path'], comment='#')
    except FileNotFoundError:
        print(f"ERROR: Uncleaned file not found at '{config['uncleaned_path']}'")
        return None

    if config['target_col'] not in df.columns:
        print(f"\n--- FATAL DATASHEET ERROR ---")
        print(f"Mission: {mission_name.upper()}")
        print(f"Missing Column: The required target column '{config['target_col']}' was not found in your file '{config['uncleaned_path']}'.")
        print("Action: Please open this CSV file and verify the column headers.")
        print("Available columns are:", df.columns.tolist())
        print("---------------------------------\n")
        return None

    df = df.dropna(subset=[config['target_col']])
    if config['candidate_label']:
        # Ensure we only try to filter if the label exists in the column
        if config['candidate_label'] in df[config['target_col']].unique():
            df = df[df[config['target_col']] != config['candidate_label']]
    
    final_cols = config['features'] + [config['target_col']]

    missing_features = [col for col in config['features'] if col not in df.columns]
    if missing_features:
        print(f"ERROR: The following required feature columns are missing for {mission_name}: {missing_features}")
        return None

    df = df[final_cols].dropna()

    df[config['target_col']] = df[config['target_col']].replace({
        config['positive_label']: 1,
        config['negative_label']: 0
    })

    df = df[df[config['target_col']].isin([0, 1])]
    df[config['target_col']] = df[config['target_col']].astype(int)

    df.to_csv(config['cleaned_path'], index=False)
    
    print(f"Successfully cleaned data for {mission_name}, saved to '{config['cleaned_path']}'")
    return df

if __name__ == '__main__':
    for mission in MISSIONS:
        clean_mission_data(mission)