from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import os
import aimodel
from cleaning import MISSIONS

app = Flask(__name__)

# --- Global Dictionaries to cache loaded models and data ---
models = {}
training_columns = {}
model_accuracies = {}
uncleaned_data = {}

def load_mission_critical_data():
    """
    Loads all models, columns, accuracies, and uncleaned data into memory at startup.
    If model files are not found for a mission, it triggers training for that mission.
    """
    global models, training_columns, model_accuracies, uncleaned_data
    print("--- Loading all mission-critical data at startup ---")
    for mission in MISSIONS.keys():
        paths = aimodel.get_model_paths(mission)
        try:
            models[mission] = joblib.load(paths['model'])
            training_columns[mission] = joblib.load(paths['columns'])
            with open(paths['accuracy'], 'r') as f:
                model_accuracies[mission] = f.read().strip()
            print(f"Loaded model for {mission.upper()}. Accuracy: {model_accuracies[mission]}%")
        except FileNotFoundError:
            print(f"WARNING: Model files for {mission.upper()} not found. Training now.")
            accuracy = aimodel.create_and_train_model(mission)
            if accuracy > 0:
                models[mission] = joblib.load(paths['model'])
                training_columns[mission] = joblib.load(paths['columns'])
                model_accuracies[mission] = str(accuracy)
            else:
                models[mission], training_columns[mission], model_accuracies[mission] = None, None, "N/A"

        try:
            config = MISSIONS[mission]
            uncleaned_data[mission] = pd.read_csv(config['uncleaned_path'], low_memory=False, comment='#')
            print(f"Uncleaned {mission.upper()} dataset loaded for archive search.")
        except FileNotFoundError:
            uncleaned_data[mission] = None
            print(f"WARNING: Uncleaned data for {mission.upper()} not found. Archive search disabled.")

@app.route('/')
def home():
    """
    Renders the main homepage (index.html) and passes the model accuracies to it.
    """
    return render_template('index.html', accuracies=model_accuracies)

@app.route('/config')
def config_page():
    """
    Renders the model configuration and retraining page (config.html).
    """
    return render_template('config.html', accuracies=model_accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives form data, processes it with the appropriate model, returns a prediction,
    and triggers a background task to append the input and retrain the model.
    """
    try:
        form_data = request.form.to_dict()
        mission = form_data.get('mission')

        if not mission or models.get(mission) is None:
            return jsonify({'error': f'Model for mission "{mission}" is not loaded.'}), 500

        model = models[mission]
        columns = training_columns[mission]
        
        input_df = pd.DataFrame([form_data])
        input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        input_df_reordered = pd.DataFrame(columns=columns)
        input_df_reordered = pd.concat([input_df_reordered, input_df], ignore_index=True, sort=False).fillna(0)
        input_df_reordered = input_df_reordered[columns]

        prediction = model.predict(input_df_reordered)
        probability = model.predict_proba(input_df_reordered)
        
        result = int(prediction[0])
        confidence = float(probability[0][result])

        # This part handles adding the input to the dataset and retraining
        # In a production app, this should be a background task (e.g., using Celery)
        aimodel.append_input_and_retrain(mission, form_data)
        
        return jsonify({'prediction': result, 'confidence': round(confidence * 100, 2)})
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Handles requests from the config page to retrain a model with new hyperparameters.
    """
    try:
        data = request.json
        mission = data.get('mission')
        n_estimators = int(data.get('n_estimators', 100))
        max_depth = int(data.get('max_depth', 3))
        test_size = float(data.get('test_size', 0.2))

        if not mission or mission not in MISSIONS:
            return jsonify({'error': 'Invalid mission specified.'}), 400

        print(f"Retraining {mission.upper()} with params: n_estimators={n_estimators}, max_depth={max_depth}")
        new_accuracy = aimodel.create_and_train_model(mission, n_estimators, max_depth, test_size)
        
        # Reload the updated model and accuracy into memory
        paths = aimodel.get_model_paths(mission)
        models[mission] = joblib.load(paths['model'])
        model_accuracies[mission] = str(new_accuracy)

        return jsonify({
            'message': f'{mission.upper()} model retrained!',
            'new_accuracy': new_accuracy
        })
    except Exception as e:
        print(f"Retraining Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/search_archive', methods=['POST'])
def search_archive():
    """
    Searches the uncleaned CSV for a given mission and query (ID or name).
    """
    mission = request.form.get('mission')
    
    if not mission or uncleaned_data.get(mission) is None:
        return jsonify({'error': f'Archive for {mission.upper()} is unavailable.'}), 500
    
    df = uncleaned_data[mission]
    config = MISSIONS[mission]
    searchable_cols = config.get('id_cols', []) + config.get('name_cols', [])

    try:
        query = request.form.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Please provide a search query.'}), 400

        conditions = pd.Series(False, index=df.index)
        for col in searchable_cols:
            if col in df.columns:
                conditions |= df[col].astype(str).str.contains(query, case=False, na=False)
        
        result_row = df[conditions]

        if not result_row.empty:
            found_data = result_row.iloc[0].fillna('').to_dict()
            return jsonify(found_data)
        else:
            return jsonify({'error': f"No data found for '{query}' in {mission.upper()}."}), 404
    except Exception as e:
        print(f"Archive Search Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_mission_critical_data()
    app.run(debug=True)