import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import joblib

def train_model(X_train, y_train):
    """
    Train and return (model, scaler). API used by unit tests.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler

def evaluate_model(y_true, y_pred):
    """
    Simple evaluator used by unit tests — returns accuracy.
    """
    return accuracy_score(y_true, y_pred)

def _cli_main():
    """
    CLI: load sklearn dataset, train, evaluate, and save artifacts into ../models
    """
    data = load_breast_cancer()
    X = data.data
    target_names = data.target_names
    malignant_idx = int(np.where(target_names == 'malignant')[0][0])
    y = np.where(data.target == malignant_idx, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model, scaler = train_model(X_train, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion matrix:")
    print(cm)

    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "cancer_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")

if __name__ == "__main__":
    _cli_main()