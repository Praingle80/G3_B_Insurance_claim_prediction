# ClaimGuard Insurance Fraud Prediction (Flask)

This project uses the provided `insurance_random_forest_regularized.pkl` directly. No retraining is needed.

## Run

From a terminal in this folder:

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

macOS/Linux:
```bash
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**.

## API

`POST http://127.0.0.1:5000/predict` with JSON:

```json
{
  "incident_severity": "Major Damage",
  "policy_age_years": 5,
  "collision_type": "Rear Collision",
  "incident_type": "Single Vehicle Collision",
  "policy_state": "OHIO",
  "authorities_contacted": "Police",
  "umbrella_limit": 0,
  "policy_csl": "250/500",
  "property_claim": 10000,
  "injury_claim": 15000,
  "total_claim_amount": 50000,
  "bodily_injuries": 1,
  "vehicle_claim": 25000,
  "actual_fraud_reported": "No"
}
```

## Model compatibility

The uploaded pickle contains a `Pipeline(SelectKBest, RandomForestClassifier)` wrapped in a dictionary. It expects the **full pre-encoded column schema** stored as `all_encoded_features`. This Flask app creates those columns and one-hot encodes the chosen categorical fields without fitting a new encoder.

The trained feature selector selected 15 columns: `total_claim_amount`, `injury_claim`, `property_claim`, `vehicle_claim`, three `incident_type` indicators, `collision_type_Rear Collision`, four `incident_severity` indicators, and three `authorities_contacted` indicators. Therefore, changes to the other displayed inputs do not affect this particular saved model. Specifically, `policy_age_years` does not exist in the saved input schema. The actual fraud label is used only to compare results, not predict.

This is a local educational/demo app. Do not use `debug=True` for public deployment. Load pickle/joblib files only from sources you trust and use a compatible scikit-learn environment if your installed version differs from the model's training version.
