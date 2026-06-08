# BreastCancerProject

Design and Implementation of an AI-Based System for Early Prediction and Detection of Breast Cancer  
Small, self-contained Flask + scikit-learn project using the Breast Cancer Wisconsin (Diagnostic) dataset.

## Project structure
BreastCancerProject/
- dataset/data.csv               (optional local copy)
- model/
  - cancer_model.pkl             (trained LogisticRegression)
  - scaler.pkl                   (StandardScaler)
- static/css/style.css
- templates/*.html               (base, login, home, predict, dashboard)
- app.py                         (Flask backend)
- database.db                    (SQLite, auto-created)
- train_model.py                 (train and serialize model)
- README.md

## Prerequisites
- Windows machine (development)
- Python 3.8+ installed (ensure python.exe is accessible)
- Recommended: use the full python path if python is not on PATH:
  `C:\Users\<you>\AppData\Local\Programs\Python\Python314\python.exe`

## Install dependencies
Open VS Code integrated terminal (PowerShell) and run:

# If `python` is available:
```powershell
python -m pip install --upgrade pip
python -m pip install scikit-learn joblib flask sqlalchemy flask_sqlalchemy flask-login
```

# Or (use full path to your python.exe if needed):
```powershell
& 'C:\Users\USER 4\AppData\Local\Programs\Python\Python314\python.exe' -m pip install --upgrade pip
& 'C:\Users\USER 4\AppData\Local\Programs\Python\Python314\python.exe' -m pip install scikit-learn joblib flask sqlalchemy flask_sqlalchemy flask-login
```

## STEP 1 — Train the model
From project root:

```powershell
cd "C:\Users\USER 4\Documents\Breast Cancer\BreastCancerProject"
python .\train_model.py
# or using full python path:
& 'C:\Users\USER 4\AppData\Local\Programs\Python\Python314\python.exe' .\train_model.py
```

Expected output: evaluation metrics printed and two files created in `model/`:
- cancer_model.pkl
- scaler.pkl

## STEP 2 — Run the Flask app
From project root:

```powershell
cd "C:\Users\USER 4\Documents\Breast Cancer\BreastCancerProject"
# run with python on PATH:
python .\app.py
# or with full path:
& 'C:\Users\USER 4\AppData\Local\Programs\Python\Python314\python.exe' .\app.py
```

Open http://127.0.0.1:5000 in a browser. Workflow:
- Register a user → Login
- Home shows model metrics
- Predict → enter patient name + 30 feature values → submit
- Dashboard → view history (Benign green, Malignant soft red)

## Notes for presentation / grading
- Model: Logistic Regression (scikit-learn). Metrics (Accuracy, Precision, Recall, Confusion Matrix) are printed by `train_model.py` and shown on the Home page.
- Database: SQLite via SQLAlchemy. `database.db` is created automatically on first run.
- Security: passwords hashed using werkzeug. Replace `app.config["SECRET_KEY"]` before any deployment.
- Frontend: Bootstrap 5, minimal JS, medical/clean styling in `static/css/style.css`.

## Troubleshooting
- "Python was not found" — restart terminal after installing Python or use the full python.exe path shown by:
  `Get-ChildItem "$env:LocalAppData\Programs" -Filter python.exe -Recurse`
- If templates fail (TemplateNotFound), ensure `templates/` exists under project root and files match names in `app.py`.
- If model/scaler missing, run `train_model.py` and confirm `model/` contains the two .pkl files.

## Next actions (suggested)
- Generate README slides or brief report for academic submission.
- Optionally add simple unit tests for `train_model.py` and