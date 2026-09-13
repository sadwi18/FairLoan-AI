import joblib
import pandas as pd


# Load trained model
model = joblib.load("models/loan_default_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")


def predict_default(applicant_data):

    # Convert input into DataFrame
    input_data = pd.DataFrame([applicant_data])

    # Keep the same feature order used during training
    input_data = input_data[feature_names]

    # Default probability
    probability = model.predict_proba(input_data)[0][1]

    # Prediction
    prediction = model.predict(input_data)[0]

    # Risk level
    if probability < 0.30:
        risk = "Low Risk"
    elif probability < 0.60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    return {
        "prediction": int(prediction),
        "default_probability": float(probability),
        "risk_level": risk
    }


# Test applicant
applicant = {
    "credit_limit": 50000,
    "education": 2,
    "marriage": 1,
    "age": 25,

    "pay_sep": 0,
    "pay_aug": 0,
    "pay_jul": 0,
    "pay_jun": 0,
    "pay_may": 0,
    "pay_apr": 0,

    "bill_sep": 40000,
    "bill_aug": 38000,
    "bill_jul": 35000,
    "bill_jun": 32000,
    "bill_may": 30000,
    "bill_apr": 28000,

    "payment_sep": 2000,
    "payment_aug": 2500,
    "payment_jul": 3000,
    "payment_jun": 2500,
    "payment_may": 3000,
    "payment_apr": 2500
}


result = predict_default(applicant)


print("=" * 70)
print("FAIRLOAN AI - LOAN DEFAULT PREDICTION")
print("=" * 70)

print("\nPrediction:", result["prediction"])
print(
    "Default Probability:",
    f"{result['default_probability'] * 100:.2f}%"
)
print("Risk Level:", result["risk_level"])