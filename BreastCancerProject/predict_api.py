import os
import uuid
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required on servers without a display
import matplotlib.pyplot as plt
from flask import Blueprint, request, jsonify, current_app, url_for

bp = Blueprint("predict_api", __name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "cancer_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
STATIC_TMP = os.path.join(os.path.dirname(__file__), "static", "tmp")
os.makedirs(STATIC_TMP, exist_ok=True)

_model = None
_scaler = None


def load_artifacts():
    global _model, _scaler
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)
    return _model, _scaler


def map_category(prob, thresholds=(0.3, 0.7)):
    low_th, high_th = thresholds
    if prob < low_th:
        return "Low", "Routine follow-up"
    if prob < high_th:
        return "Medium", "Prompt clinical assessment"
    return "High", "Urgent evaluation recommended"


def map_risk_advice(category):
    advice_map = {
        "Low": (
            "This result suggests a low likelihood of malignancy. Routine follow-up "
            "and standard screening are recommended."
        ),
        "Medium": (
            "This result is inconclusive. A prompt clinical assessment with a "
            "specialist is recommended to confirm or rule out malignancy."
        ),
        "High": (
            "This result suggests a higher likelihood of malignancy. Urgent "
            "evaluation by an oncologist or breast specialist is strongly recommended."
        ),
    }
    base = advice_map.get(category, "Please consult a qualified clinician to interpret this result.")
    return base + " This tool is informational only and not a medical diagnosis."


def top_feature_contributions(model, scaler, X_raw, feature_names=None, topn=5):
    X_scaled = scaler.transform(np.array(X_raw).reshape(1, -1))[0]
    coefs = np.ravel(model.coef_)
    contributions = coefs * X_scaled
    idx = np.argsort(np.abs(contributions))[::-1][:topn]
    result = []
    for i in idx:
        name = feature_names[i] if feature_names is not None and i < len(feature_names) else f"f{i}"
        result.append({"feature": name, "coef": float(coefs[i]), "value": float(X_raw[i]), "contribution": float(contributions[i])})
    return result


def make_probability_gauge(prob, out_path):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    color = "#d9534f" if prob > 0.7 else ("#f0ad4e" if prob > 0.3 else "#5cb85c")
    ax.barh([0], [prob], color=color, height=0.6)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel(f"Probability of malignancy: {prob*100:.1f}%")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x*100)}%"))
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def make_top_features_bar(top_feats, out_path):
    names = [t["feature"] for t in top_feats][::-1]
    contribs = [t["contribution"] for t in top_feats][::-1]
    colors = ["#5cb85c" if c < 0 else "#d9534f" for c in contribs]
    fig, ax = plt.subplots(figsize=(6, max(1.5, 0.4 * len(names))))
    ax.barh(names, contribs, color=colors)
    ax.set_xlabel("Signed contribution (coef x scaled value)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


RADIUS_MIN, RADIUS_MAX = 6.98, 28.11  # typical range for "mean radius" in this dataset

def make_tumor_illustration(mean_radius, category, prob, out_path):
    fig, ax = plt.subplots(figsize=(5, 5.5))
    fig.patch.set_facecolor("white")

    breast = plt.matplotlib.patches.Ellipse((0.5, 0.46), width=0.6, height=0.7, color="#fbe4e1",
                      ec="#d99a92", lw=2.2, zorder=1)
    ax.add_patch(breast)
    inner_shadow = plt.matplotlib.patches.Ellipse((0.5, 0.44), width=0.48, height=0.56, color="#f6cfc9",
                            ec="none", alpha=0.55, zorder=2)
    ax.add_patch(inner_shadow)
    ax.add_patch(plt.Circle((0.5, 0.76), 0.016, color="#c98b8b", zorder=3))

    t = float(np.clip((mean_radius - RADIUS_MIN) / (RADIUS_MAX - RADIUS_MIN), 0, 1))
    rel = 0.035 + t * 0.13

    color_map = {"Low": "#3fa34d", "Medium": "#e2972e", "High": "#cf3a3a"}
    color = color_map.get(category, "#888888")
    tumor = plt.Circle((0.5, 0.43), rel, color=color, ec="#222222", lw=1.2, alpha=0.92, zorder=4)
    ax.add_patch(tumor)

    if rel > 0.06:
        ax.text(0.5, 0.43, f"{mean_radius:.1f}mm", color="white", fontsize=8.5,
                fontweight="bold", ha="center", va="center", zorder=5)
    else:
        ax.text(0.5, 0.43 - rel - 0.025, f"{mean_radius:.1f}mm", color="#444444", fontsize=8.5,
                fontweight="bold", ha="center", va="top", zorder=5)

    ax.text(0.5, 0.97, "Illustrative Tumor Diagram", fontsize=13, fontweight="bold",
            ha="center", va="top", color="#333333")

    legend_y = 0.91
    for i, (cat, c) in enumerate(color_map.items()):
        x = 0.20 + i * 0.20
        ax.add_patch(plt.Circle((x, legend_y), 0.015, color=c, ec="#222222", lw=0.7, zorder=6))
        ax.text(x + 0.03, legend_y, cat, fontsize=8, va="center", ha="left", color="#444444")

    ax.text(0.5, 0.07, f"Risk: {category}  •  Malignancy probability: {prob*100:.1f}%",
            fontsize=10.5, fontweight="bold", ha="center", va="bottom", color=color)
    ax.text(0.5, 0.015, "Generated from input measurements — not an actual medical scan",
            fontsize=7.5, ha="center", va="bottom", color="#777777", style="italic")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor="white")
    plt.close(fig)

@bp.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True, silent=True)
    if not payload:
        return jsonify({"error": "invalid json payload"}), 400

    features = payload.get("features")
    if features is None:
        return jsonify({"error": "missing 'features' array"}), 400

    try:
        model, scaler = load_artifacts()
    except Exception as e:
        return jsonify({"error": "failed to load model artifacts", "detail": str(e)}), 500

    X = np.array(features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])
    category, urgency_text = map_category(prob)

    feature_names = payload.get("feature_names")
    top_feats = top_feature_contributions(model, scaler, X[0], feature_names=feature_names, topn=6)

    uid = uuid.uuid4().hex
    gauge_fname = f"prob_gauge_{uid}.png"
    top_fname = f"top_features_{uid}.png"
    gauge_path = os.path.join(STATIC_TMP, gauge_fname)
    top_path = os.path.join(STATIC_TMP, top_fname)

    try:
        make_probability_gauge(prob, gauge_path)
        make_top_features_bar(top_feats, top_path)
    except Exception:
        current_app.logger.exception("image generation failed")
        gauge_fname = None
        top_fname = None

    images = []
    if gauge_fname:
        images.append(url_for("static", filename=f"tmp/{gauge_fname}"))
    if top_fname:
        images.append(url_for("static", filename=f"tmp/{top_fname}"))

    summary = (
        f"Model result: {prob*100:.1f}% probability of malignancy ({category} risk). "
        f"Top contributors: {', '.join([t['feature'] for t in top_feats[:3]])}. "
        f"Suggested action: {urgency_text}. "
        "This tool is informational only and not a diagnosis. Consult a qualified clinician."
    )

    resp = {
        "probability": prob,
        "category": category,
        "urgency": urgency_text,
        "top_features": top_feats,
        "images": images,
        "summary": summary,
    }
    return jsonify(resp)