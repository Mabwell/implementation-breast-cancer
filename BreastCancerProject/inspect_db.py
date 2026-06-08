import os
import json
from sqlalchemy import create_engine, text

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")
engine = create_engine(f"sqlite:///{DB_PATH}")

with engine.connect() as conn:
    row = conn.execute(text(
        "SELECT id, patient_name, prediction_result, probability_score, features_json, timestamp "
        "FROM prediction_record ORDER BY timestamp DESC LIMIT 1"
    )).fetchone()

if not row:
    print("No prediction records found.")
else:
    rec = row._mapping
    print(f"ID: {rec['id']}")
    print(f"Patient: {rec['patient_name']}")
    print(f"Result: {rec['prediction_result']}")
    print(f"Probability: {rec.get('probability_score')}")
    print(f"Timestamp: {rec.get('timestamp')}")
    try:
        features = json.loads(rec['features_json'])
        print("Sample features (first 6):")
        for k in list(features.keys())[:6]:
            print(f"  {k}: {features[k]}")
    except Exception:
        pass