from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
import aimodel

app = Flask(__name__)

# --- Load the Trained Model and Columns ---
try:
    model = joblib.load('kepler_model.pkl')
    training_columns = joblib.load('training_columns.pkl')
    print("Model and columns loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Model files not found. Please run aimodel.py first to train and save the model.")
    model = None
    training_columns = None

# --- Load the Uncleaned Data for Archive Search ---
try:
    # Use low_memory=False and comment='#' to handle scientific CSV files
    uncleaned_df = pd.read_csv('Data/Kepler_Uncleaned.csv', low_memory=False, comment='#')
    print("Uncleaned Kepler dataset loaded for archive search.")
    
    # --- Self-Adapting Column Finder ---
    possible_name_cols = ['kepoi_name', 'koi_name', 'kepler_name']
    possible_id_cols = ['kepid', 'Kepler ID', 'id']
    
    name_col = next((col for col in possible_name_cols if col in uncleaned_df.columns), None)
    id_col = next((col for col in possible_id_cols if col in uncleaned_df.columns), None)
    
    print(f"Archive search will use Name Column: '{name_col}' and ID Column: '{id_col}'")

except FileNotFoundError:
    uncleaned_df = None
    name_col = None
    id_col = None
    print("WARNING: 'Data/Kepler_Uncleaned.csv' not found. Archive search will be disabled.")

# --- Define Website Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, processes it, and returns a prediction."""
    global model, training_columns # Ensure we can update the global model instance
    # --- Re-load the model in case it was retrained ---
    try:
        model = joblib.load('kepler_model.pkl')
        training_columns = joblib.load('training_columns.pkl')
    except FileNotFoundError:
        return jsonify({'error': 'Model files not found. Please ensure the model is trained.'}), 500

    try:
        form_data = request.form.to_dict()
        input_df = pd.DataFrame([form_data])
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Ensure all training columns are present, adding missing ones with a value of 0
        for col in training_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df_reordered = input_df[training_columns]

        prediction = model.predict(input_df_reordered)
        probability = model.predict_proba(input_df_reordered)
        
        result = int(prediction[0])
        confidence = float(probability[0][result])

        output_folder = 'input_folder'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        input_df.to_csv(os.path.join(output_folder, 'last_input.csv'), index=False)
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    """ Retrains the model with new hyperparameters. """
    try:
        data = request.json
        n_estimators = int(data.get('n_estimators', 100))
        max_depth = int(data.get('max_depth', 20)) if data.get('max_depth') else None
        test_size = float(data.get('test_size', 0.2))

        print(f"Retraining model with params: n_estimators={n_estimators}, max_depth={max_depth}, test_size={test_size}")

        accuracy = aimodel.retrain_model_with_params(n_estimators=n_estimators, max_depth=max_depth, test_size=test_size)
        
        # After retraining, we need to reload the model for subsequent predictions
        global model, training_columns
        model = joblib.load('kepler_model.pkl')
        training_columns = joblib.load('training_columns.pkl')

        return jsonify({'message': 'Model retrained successfully!', 'new_accuracy': round(accuracy * 100, 3)})

    except Exception as e:
        print(f"Retraining Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search_archive', methods=['POST'])
def search_archive():
    """Searches the uncleaned CSV for a given ID or name."""
    if uncleaned_df is None or (name_col is None and id_col is None):
        return jsonify({'error': 'Archive data or key columns are not available on the server.'}), 500

    try:
        search_query = request.form.get('query', '').strip()
        
        if not search_query:
            return jsonify({'error': 'Please provide an ID or name to search.'}), 400

        search_conditions = []
        if name_col:
            search_conditions.append(uncleaned_df[name_col].str.contains(search_query, case=False, na=False))
        if id_col:
            # Ensure the ID column is treated as a string for searching
            search_conditions.append(uncleaned_df[id_col].astype(str).str.contains(search_query, case=False, na=False))
        
        if not search_conditions:
             return jsonify({'error': 'No searchable columns (like kepid or kepoi_name) found in the CSV.'}), 500

        # Combine conditions with an OR operation
        combined_conditions = pd.Series(False, index=uncleaned_df.index)
        for condition in search_conditions:
            combined_conditions |= condition

        result_row = uncleaned_df[combined_conditions]

        if not result_row.empty:
            found_data = result_row.iloc[0].fillna('').to_dict()
            return jsonify(found_data)
        else:
            return jsonify({'error': f"No data found for ID or name: '{search_query}'"}), 404

    except Exception as e:
        print(f"An error occurred during search: {e}")
        return jsonify({'error': str(e)}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

