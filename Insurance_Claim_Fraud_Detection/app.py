from pathlib import Path
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
ARTIFACT = joblib.load(Path(__file__).resolve().parent / "insurance_random_forest_regularized.pkl")
MODEL = ARTIFACT["model"]
FEATURES = ARTIFACT["all_encoded_features"]

CATEGORIES = {
    "incident_severity": ["Major Damage", "Minor Damage", "Total Loss", "Trivial Damage"],
    "collision_type": ["Front Collision", "Rear Collision", "Side Collision", "Unknown"],
    "incident_type": ["Multi-vehicle Collision", "Parked Car", "Single Vehicle Collision", "Vehicle Theft"],
    "policy_state": ["ILLINOIS", "INDIANA", "OHIO"],
    "authorities_contacted": ["Ambulance", "Fire", "Other", "Police", "None"],
    "policy_csl": ["100/300", "250/500", "500/1000"],
}
NUMERIC = ["policy_age_years", "umbrella_limit", "property_claim", "injury_claim", "total_claim_amount", "bodily_injuries", "vehicle_claim"]
REQUIRED = list(CATEGORIES) + NUMERIC


def encode_claim(payload):
    missing = [name for name in REQUIRED if name not in payload or payload[name] in (None, "")]
    if missing:
        raise ValueError("Missing fields: " + ", ".join(missing))
    for name, options in CATEGORIES.items():
        if payload[name] not in options:
            raise ValueError(f"Invalid {name}. Choose one of: {', '.join(options)}")
    numeric = {}
    for name in NUMERIC:
        try:
            numeric[name] = float(payload[name])
        except (ValueError, TypeError):
            raise ValueError(f"{name} must be a number")
        if not (-float('inf') < numeric[name] < float('inf')):
            raise ValueError(f"{name} must be finite")
        if numeric[name] < 0:
            raise ValueError(f"{name} cannot be negative")
    if numeric["bodily_injuries"] != int(numeric["bodily_injuries"]):
        raise ValueError("bodily_injuries must be a whole number")

    # The saved model expects pre-encoded columns. Initialize the entire exact
    # training schema, then set the feature indicators requested in the UI.
    row = {name: 0.0 for name in FEATURES}
    for name in ["umbrella_limit", "property_claim", "injury_claim", "total_claim_amount", "bodily_injuries", "vehicle_claim"]:
        if name in row:
            row[name] = numeric[name]
    # policy_age_years is not a model-training column; retained in the form only.
    for field in ["incident_severity", "collision_type", "incident_type", "policy_state", "authorities_contacted", "policy_csl"]:
        key = f"{field}_{payload[field]}"
        if key in row:
            row[key] = 1.0
    # The trained model also contains numerous fields unavailable to this form.
    # They default to 0, but none are in this model's 15 selected features.
    return pd.DataFrame([row], columns=FEATURES)


@app.get("/")
def home():
    return render_template("index.html", categories=CATEGORIES)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Provide a valid JSON object"}), 400
    try:
        data = encode_claim(payload)
        cls = int(MODEL.predict(data)[0])
        probabilities = MODEL.predict_proba(data)[0]
        probability_by_class = {int(key): float(value) for key, value in zip(MODEL.classes_, probabilities)}
        actual = payload.get("actual_fraud_reported", "")
        if actual not in ("", None, "Yes", "No"):
            raise ValueError("actual_fraud_reported must be Yes, No, or empty")
        return jsonify({
            "predicted_fraud_reported": "Yes" if cls == 1 else "No",
            "fraud_probability": round(probability_by_class.get(1, 0) * 100, 2),
            "not_fraud_probability": round(probability_by_class.get(0, 0) * 100, 2),
            "actual_fraud_reported": actual or None,
            "prediction_correct": (cls == int(actual == "Yes")) if actual else None,
            "model": "Random Forest",
        })
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction could not be completed"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
