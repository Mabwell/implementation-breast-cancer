"""
app.py
Flask backend for the BreastCancerProject.

Features:
- SQLite (SQLAlchemy) for User and PredictionRecord
- Authentication with Flask-Login
- Password hashing with werkzeug.security
- Loads trained scaler and Logistic Regression model from model/
- Routes: /, /login, /register, /home, /predict (GET/POST), /dashboard, /logout
- Saves prediction history in the database

Save this file in your project root and ensure templates/ and static/ folders
match the project structure defined in the specification.
"""
import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from joblib import load
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from predict_api import bp as predict_api_bp, map_category, map_risk_advice, make_tumor_illustration
import uuid

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "database.db")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "cancer_model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

app = Flask(__name__, static_folder="static", template_folder="templates")
# NOTE: replace the secret key for presentation/deployment
app.config["SECRET_KEY"] = "change_this_secret_for_presentation"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

app.register_blueprint(predict_api_bp)

# Use the sklearn dataset feature names to keep form order consistent with training
FEATURE_NAMES = list(load_breast_cancer().feature_names)  # 30 feature names

# -------------------------
# Database models
# -------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    predictions = db.relationship("PredictionRecord", backref="user", lazy=True)

class PredictionRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_name = db.Column(db.String(200), nullable=False)
    # store a few major features explicitly for quick display
    mean_radius = db.Column(db.Float, nullable=True)
    mean_texture = db.Column(db.Float, nullable=True)
    mean_perimeter = db.Column(db.Float, nullable=True)
    mean_area = db.Column(db.Float, nullable=True)
    mean_smoothness = db.Column(db.Float, nullable=True)
    # store full feature vector (JSON string)
    features_json = db.Column(db.Text, nullable=False)
    prediction_result = db.Column(db.String(20), nullable=False)  # 'Benign' or 'Malignant'
    probability_score = db.Column(db.Float, nullable=False)
    risk_category = db.Column(db.String(20), nullable=True)
    advice_text = db.Column(db.Text, nullable=True)
    tumor_image = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create DB tables if they don't exist
with app.app_context():
    db.create_all()

# -------------------------
# Load model & scaler
# -------------------------
model = None
scaler = None
try:
    scaler = load(SCALER_PATH)
    model = load(MODEL_PATH)
except Exception as e:
    app.logger.error(f"Could not load model or scaler: {e}")
    # The app will still run; pages should handle missing model gracefully.

# -------------------------
# Flask-Login user loader
# -------------------------
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -------------------------
# Helper: evaluate saved model (used on /home)
# -------------------------
def evaluate_saved_model():
    if model is None or scaler is None:
        return None
    data = load_breast_cancer()
    X = data.data
    y_orig = data.target
    target_names = data.target_names
    malignant_idx = int(np.where(target_names == "malignant")[0][0])
    y = np.where(y_orig == malignant_idx, 1, 0)
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "confusion_matrix": cm.tolist()}

# -------------------------
# Routes
# -------------------------
@app.route("/")
def index():
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Username and password are required.", "danger")
            return redirect(url_for("register"))
        if User.query.filter_by(username=username).first():
            flash("Username already taken.", "warning")
            return redirect(url_for("register"))
        hashed = generate_password_hash(password)
        user = User(username=username, password=hashed)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for("login"))
    # GET renders login.html with register flag (templates handle form)
    return render_template("login.html", register=True)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash("Logged in successfully.", "success")
            return redirect(url_for("home"))
        flash("Invalid username or password.", "danger")
        return redirect(url_for("login"))
    return render_template("login.html", register=False)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

@app.route("/home")
@login_required
def home():
    metrics = evaluate_saved_model()
    return render_template("home.html", metrics=metrics)

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    if request.method == "POST":
        patient_name = request.form.get("patient_name", "Unknown").strip()
        # Collect features in the same order as FEATURE_NAMES
        features = []
        try:
            for fname in FEATURE_NAMES:
                val = request.form.get(fname)
                if val is None or val == "":
                    raise ValueError(f"Missing feature input: {fname}")
                features.append(float(val))
        except ValueError as e:
            flash(str(e), "danger")
            return redirect(url_for("predict"))
        # Model inference
        x = np.array(features).reshape(1, -1)
        x_scaled = scaler.transform(x)
        pred_label = int(model.predict(x_scaled)[0])
        prob = float(model.predict_proba(x_scaled)[0][1])  # probability of malignant class
        result_text = "Malignant" if pred_label == 1 else "Benign"
        category, _urgency = map_category(prob)
        advice_text = map_risk_advice(category)

        # Generate illustrative diagram (not a real scan) for this prediction
        tumor_fname = f"tumor_{uuid.uuid4().hex}.png"
        tumor_dir = os.path.join(app.static_folder, "tmp")
        os.makedirs(tumor_dir, exist_ok=True)
        tumor_path = os.path.join(tumor_dir, tumor_fname)
        feature_map_preview = {name: float(features[idx]) for idx, name in enumerate(FEATURE_NAMES)}
        try:
            make_tumor_illustration(feature_map_preview.get("mean radius", 10.0), category, prob, tumor_path)
        except Exception as e:
            app.logger.error(f"Could not generate tumor illustration: {e}")
            tumor_fname = None

        # Store record
        feature_map = feature_map_preview
        rec = PredictionRecord(
            user_id=current_user.id,
            patient_name=patient_name,
            mean_radius=feature_map.get("mean radius"),
            mean_texture=feature_map.get("mean texture"),
            mean_perimeter=feature_map.get("mean perimeter"),
            mean_area=feature_map.get("mean area"),
            mean_smoothness=feature_map.get("mean smoothness"),
            features_json=json.dumps(feature_map),
            prediction_result=result_text,
            probability_score=prob,
            risk_category=category,
            advice_text=advice_text,
            tumor_image=tumor_fname,
        )
        db.session.add(rec)
        db.session.commit()
        return render_template(
            "predict.html",
            FEATURE_NAMES=FEATURE_NAMES,
            result=result_text,
            probability=prob,
            patient_name=patient_name,
            advice=advice_text,
            tumor_image=tumor_fname,
        )
    # GET: show the prediction form
    return render_template("predict.html", FEATURE_NAMES=FEATURE_NAMES)

@app.route("/dashboard")
@login_required
def dashboard():
    records = PredictionRecord.query.filter_by(user_id=current_user.id).order_by(PredictionRecord.timestamp.desc()).all()
    return render_template("dashboard.html", records=records)

# Simple 404 handler
@app.errorhandler(404)
def not_found(e):
    return "Not found", 404

# Run the app (development)
if __name__ == "__main__":
    # Debug True is convenient for development; set False for presentation if needed
    app.run(debug=True, host="127.0.0.1", port=5000)