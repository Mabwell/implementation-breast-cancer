def load_config(config_file):
    import json
    with open(config_file, 'r') as f:
        return json.load(f)

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)

def load_model(model_path):
    import joblib
    return joblib.load(model_path)

def create_directory(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

def log_message(message):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(message)