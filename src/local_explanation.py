import joblib
import pandas as pd
import shap
import numpy as np


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load("models/loan_default_model.pkl")
feature_names = joblib.load("models/feature_names.pkl")


# ============================================================
# APPLICANT DATA
# ============================================================

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


# ============================================================
# PREPARE INPUT
# ============================================================

input_data = pd.DataFrame([applicant])

# Keep exactly the same feature order used during training
input_data = input_data[feature_names]


# ============================================================
# PREDICTION
# ============================================================

probability = model.predict_proba(input_data)[0][1]
prediction = model.predict(input_data)[0]


# ============================================================
# RISK LEVEL
# ============================================================

if probability < 0.30:
    risk = "Low Risk"
elif probability < 0.60:
    risk = "Medium Risk"
else:
    risk = "High Risk"


# ============================================================
# SHAP EXPLANATION
# ============================================================

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(input_data)


# ============================================================
# HANDLE DIFFERENT SHAP OUTPUT FORMATS
# ============================================================

if isinstance(shap_values, list):

    # Older SHAP versions
    values = np.asarray(shap_values[1])[0]

else:

    # Newer SHAP versions
    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 3:
        # Shape: (samples, features, classes)
        values = shap_array[0, :, 1]

    elif shap_array.ndim == 2:
        # Shape: (samples, features)
        values = shap_array[0]

    else:
        values = shap_array.flatten()


# ============================================================
# SAFETY CHECK
# ============================================================

if len(values) != len(feature_names):
    raise ValueError(
        f"SHAP values contain {len(values)} values, "
        f"but there are {len(feature_names)} features."
    )


# ============================================================
# CREATE EXPLANATION TABLE
# ============================================================

explanation = pd.DataFrame({
    "feature": feature_names,
    "value": input_data.iloc[0].values,
    "shap_value": values
})


# Calculate absolute importance
explanation["absolute_shap"] = explanation["shap_value"].abs()


# Sort by strongest influence
explanation = explanation.sort_values(
    "absolute_shap",
    ascending=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 70)
print("FAIRLOAN AI - INDIVIDUAL EXPLANATION")
print("=" * 70)

print(
    "\nPrediction:",
    "Default" if prediction == 1 else "No Default"
)

print(
    f"Default Probability: {probability * 100:.2f}%"
)

print("Risk Level:", risk)


print("\nTop 10 Factors Influencing This Prediction")
print("-" * 70)


for _, row in explanation.head(10).iterrows():

    if row["shap_value"] > 0:
        direction = "increases default risk"
    else:
        direction = "reduces default risk"

    print(
        f"{row['feature']:15} "
        f"Value: {str(row['value']):10} | "
        f"SHAP: {row['shap_value']: .5f} | "
        f"{direction}"
    )