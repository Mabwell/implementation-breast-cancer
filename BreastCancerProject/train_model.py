"""
train_model.py
Loads the Breast Cancer Wisconsin (Diagnostic) dataset from scikit-learn,
preprocesses features, trains a Logistic Regression classifier, evaluates it,
and saves the trained model and scaler to the model/ directory.

Usage:
    python train_model.py
"""
import os
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from joblib import dump

def main():
    # Load dataset from scikit-learn
    data = load_breast_cancer()
    X = data.data
    y_orig = data.target
    target_names = data.target_names  # e.g., array(['malignant','benign'], dtype='<U9')
    feature_names = data.feature_names

    # Map labels to specification: Malignant=1, Benign=0
    malignant_idx = int(np.where(target_names == 'malignant')[0][0])
    y = np.where(y_orig == malignant_idx, 1, 0)

    # Basic info
    print("Dataset loaded.")
    print(f"Number of samples: {X.shape[0]}, Number of features: {X.shape[1]}")
    print("Using features:", ", ".join(feature_names[:5]) + ", ...")

    # Handle missing values if any (this dataset has none, but keep robust)
    if np.isnan(X).any():
        print("Missing values found. Imputing with column means.")
        col_means = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_means, inds[1])
    else:
        print("No missing values detected.")

    # Split data: 80% train, 20% test, stratified by label for balanced classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Logistic Regression
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # Predictions and evaluation on test set
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability for class=1 (Malignant)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Print evaluation metrics
    print("\nEvaluation on test set:")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("Confusion Matrix (rows=true class [Benign=0, Malignant=1], cols=predicted):")
    print(cm)

    # Ensure model directory exists and save model + scaler
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "cancer_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    dump(model, model_path)
    dump(scaler, scaler_path)

    print(f"\nSaved trained model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")
    print("STEP 1 complete: model and scaler are serialized and ready for the Flask app.")

if __name__ == "__main__":
    main()