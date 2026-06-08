import os
import pandas as pd

def load_data(file_path):
    """
    Load the dataset from the specified file path.
    
    Parameters:
    - file_path: str, path to the dataset file
    
    Returns:
    - DataFrame containing the loaded dataset
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Assuming the dataset is in CSV format
    data = pd.read_csv(file_path)
    return data

def load_raw_data(directory):
    """
    Load all raw data files from the specified directory.
    
    Parameters:
    - directory: str, path to the directory containing raw data files
    
    Returns:
    - List of DataFrames containing the loaded datasets
    """
    dataframes = []
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = load_data(file_path)
            dataframes.append(df)
    return dataframes

def load_processed_data(file_path):
    """
    Load the processed dataset from the specified file path.
    
    Parameters:
    - file_path: str, path to the processed dataset file
    
    Returns:
    - DataFrame containing the loaded processed dataset
    """
    return load_data(file_path)