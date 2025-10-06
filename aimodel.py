# aimodel.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import joblib 
import cleaning

# The cleaning script is no longer needed here, we assume the cleaned file exists.

def create_and_train_model():
    """
    Reads the cleaned data, trains the model, evaluates it,
    and saves the model and training columns to disk.
    """
    try:
        file_path = 'Data/Kepler_Cleaned.csv'
        File = pd.read_csv(file_path)
    except FileNotFoundError:
        print("ERROR: 'Data/Kepler_Cleaned.csv' not found.")
        print("Please run cleaning.py first to generate the cleaned data file.")
        return

    # These columns are not features for the model
    File = File.drop(columns=['kepoi_name'], errors='ignore')

    # Define Target (Y) and Features (X)
    Y = File['koi_disposition']
    X = File.drop(columns=['koi_disposition'])

    # Split data for training and testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Initialize and train the model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    print("Training the model...")
    model.fit(X_train, Y_train)
    print("Model training completed.")

    # Evaluate the model
    print("Evaluating the model...")
    Y_pred = model.predict(X_test)
    print("Model evaluation completed.")

    mse = mean_squared_error(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Accuracy: {accuracy}')

    # --- CRITICAL: SAVE THE MODEL AND COLUMNS ---
    print("Saving model to 'kepler_model.pkl'...")
    joblib.dump(model, 'kepler_model.pkl')
    print("Model saved successfully.")

    print("Saving training columns to 'training_columns.pkl'...")
    joblib.dump(X.columns.tolist(), 'training_columns.pkl') # Save the column names
    print("Columns saved successfully.")

    return model

# This ensures the code only runs when you execute the script directly
if __name__ == '__main__':

    create_and_train_model()