import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import cleaning

def More_Cleaning():
    cleaning.Clean_File()
    file_path = 'Data/Kepler_Cleaned.csv'
    File = pd.read_csv(file_path)

    File = File.drop(columns=['koi_tce_delivname', 'kepoi_name'], errors = 'ignore')
    Y = File.iloc[:, 0]
    X = File.drop(File.columns[0], axis=1)
    return Y, X, File

def Create_Model():
    Y, X, File = More_Cleaning()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    print("Training the model...")
    model.fit(X_train, Y_train)
    print("Model training completed.")
    print("Evaluating the model...")
    Y_pred = model.predict(X_test)
    print("Model evaluation completed.")

    mse = mean_squared_error(Y_test, Y_pred)
    accuracy = accuracy_score(Y_test, Y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Accuracy: {accuracy}')
    return model