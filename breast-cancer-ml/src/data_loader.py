def load_data(file_path):
    import pandas as pd
    
    # Load the dataset from the specified file path
    data = pd.read_csv(file_path)
    
    return data

def get_data(file_path):
    # Load the data using the load_data function
    data = load_data(file_path)
    
    # Return the data in a usable format
    return data