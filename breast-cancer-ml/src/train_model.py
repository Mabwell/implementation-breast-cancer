import os
import sys
import argparse
import numpy as np
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def train_model(X_train, y_train):
    """Train and return (model, scaler)."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_train_scaled, y_train)
    return model, scaler


def save_artifacts(model, scaler, out_dir):
    """Save model and scaler to out_dir (create directory if needed)."""
    # If a file exists at out_dir path remove it so we can create a directory
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        os.remove(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "cancer_model.pkl")
    scaler_path = os.path.join(out_dir, "scaler.pkl")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model_path, scaler_path


def evaluate_and_print(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print("Confusion matrix:")
    print(cm)


def _cli_main(argv=None):
    """CLI: train on sklearn dataset and save artifacts."""
    parser = argparse.ArgumentParser(description="Train breast cancer model")
    parser.add_argument(
        "--out-dir",
        "-o",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")),
        help="Directory to write model and scaler",
    )
    args = parser.parse_args(argv)

    data = load_breast_cancer()
    X = data.data
    # Robust mapping: if target_names include 'malignant', map accordingly; otherwise use numeric target
    try:
        target_names = data.target_names
        if "malignant" in target_names:
            malignant_idx = int(np.where(target_names == "malignant")[0][0])
            y = np.where(data.target == malignant_idx, 1, 0)
        else:
            y = data.target
    except Exception:
        y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model, scaler = train_model(X_train, y_train)

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    evaluate_and_print(y_test, y_pred)

    model_path, scaler_path = save_artifacts(model, scaler, args.out_dir)
    print(f"Saved model -> {model_path}")
    print(f"Saved scaler -> {scaler_path}")


if __name__ == "__main__":
    try:
        _cli_main()
    except Exception as e:
        print("Training failed:", e, file=sys.stderr)
        sys.exit(1)