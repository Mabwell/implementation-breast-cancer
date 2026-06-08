def save_model(model, filename):
    import joblib
    joblib.dump(model, filename)

def load_model(filename):
    import joblib
    return joblib.load(filename)

def save_scaler(scaler, filename):
    import joblib
    joblib.dump(scaler, filename)

def load_scaler(filename):
    import joblib
    return joblib.load(filename)

def log_metrics(metrics, filename):
    import json
    with open(filename, 'w') as f:
        json.dump(metrics, f)