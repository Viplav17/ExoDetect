import pandas as pd
import numpy as np

file_path = 'Data/Kepler_Uncleaned.csv'

def Read_File():
    try:
        File = pd.read_csv(file_path, comment='#')
    except FileNotFoundError:
        print("File not found!")
    return File

def Clean_File(File):
    error_cols = [col for col in File.columns if '_err' in col]
    File = File.drop(columns = ['kepid', 'kepoi_name', 'kepler_name', 'koi_pdisposition', 'koi_score'] + error_cols, errors = 'ignore')
    File = File.replace("CANDIDATE", np.nan)
    File = File.dropna()

    File = File.replace("CONFIRMED", 1)
    File = File.replace("FALSE POSITIVE", 0)
    
    return File

Cleaned_File = Clean_File(Read_File())
Cleaned_File.to_csv('Data/Kepler_Cleaned.csv', index = False)
