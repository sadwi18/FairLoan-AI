from fastapi import FastAPI
from pydantic import BaseModel, Field
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
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="FairLoan AI API",
    description=(
        "Explainable and fairness-aware credit default "
        "risk prediction API."
    ),
    version="1.0.0"
)


# ============================================================
# INPUT SCHEMA
# ============================================================

class Applicant(BaseModel):

    credit_limit: float = Field(ge=0)
    education: int = Field(ge=1, le=4)
    marriage: int = Field(ge=1, le=3)
    age: int = Field(ge=18, le=100)

    pay_sep: int
    pay_aug: int
    pay_jul: int
    pay_jun: int
    pay_may: int
    pay_apr: int

    bill_sep: float = Field(ge=0)
    bill_aug: float = Field(ge=0)
    bill_jul: float = Field(ge=0)
    bill_jun: float = Field(ge=0)
    bill_may: float = Field(ge=0)
    bill_apr: float = Field(ge=0)

    payment_sep: float = Field(ge=0)
    payment_aug: float = Field(ge=0)
    payment_jul: float = Field(ge=0)
    payment_jun: float = Field(ge=0)
    payment_may: float = Field(ge=0)
    payment_apr: float = Field(ge=0)


# ============================================================
# HELPER FUNCTION
# ============================================================

def prepare_input(applicant: Applicant):

    data = applicant.model_dump()

    input_data = pd.DataFrame([data])

    input_data = input_data[feature_names]

    return input_data


def get_risk_level(probability):

    if probability < 0.30:
        return "Low Risk"

    elif probability < 0.60:
        return "Medium Risk"

    else:
        return "High Risk"


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "FairLoan AI API is running",
        "version": "1.0",
        "endpoints": [
            "/predict",
            "/explain",
            "/fairness",
            "/docs"
        ]
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(applicant: Applicant):

    input_data = prepare_input(applicant)

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]

    risk = get_risk_level(probability)

    return {
        "prediction": (
            "Default"
            if prediction == 1
            else "No Default"
        ),
        "default_probability": round(
            float(probability),
            4
        ),
        "default_probability_percent": round(
            float(probability) * 100,
            2
        ),
        "risk_level": risk
    }


# ============================================================
# EXPLAINABILITY ENDPOINT
# ============================================================

@app.post("/explain")
def explain(applicant: Applicant):

    input_data = prepare_input(applicant)

    probability = model.predict_proba(
        input_data
    )[0][1]

    prediction = model.predict(
        input_data
    )[0]

    risk = get_risk_level(probability)


    # SHAP
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(
        input_data
    )


    # Handle different SHAP versions
    if isinstance(shap_values, list):

        values = np.asarray(
            shap_values[1]
        )[0]

    else:

        shap_array = np.asarray(
            shap_values
        )

        if shap_array.ndim == 3:

            values = shap_array[0, :, 1]

        elif shap_array.ndim == 2:

            values = shap_array[0]

        else:

            values = shap_array.flatten()


    # Explanation table
    explanation = pd.DataFrame({

        "feature": feature_names,

        "value": input_data.iloc[0].values,

        "shap_value": values

    })


    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )


    explanation = explanation.sort_values(
        "absolute_shap",
        ascending=False
    )


    # Top 10 explanations
    top_features = []

    for _, row in explanation.head(10).iterrows():

        top_features.append({

            "feature": row["feature"],

            "value": row["value"],

            "shap_value": round(
                float(row["shap_value"]),
                5
            ),

            "effect": (
                "increases default risk"
                if row["shap_value"] > 0
                else "reduces default risk"
            )

        })


    return {

        "prediction": (
            "Default"
            if prediction == 1
            else "No Default"
        ),

        "default_probability_percent": round(
            float(probability) * 100,
            2
        ),

        "risk_level": risk,

        "top_factors": top_features

    }


# ============================================================
# FAIRNESS ENDPOINT
# ============================================================

@app.get("/fairness")
def fairness():

    return {

        "sensitive_attribute": "sex",

        "before_mitigation": {

            "demographic_parity_difference": 0.0331,

            "equalized_odds_difference": 0.0320

        },

        "after_mitigation": {

            "demographic_parity_difference": 0.0075,

            "equalized_odds_difference": 0.0561

        },

        "performance": {

            "accuracy_before": 0.7802,

            "accuracy_after": 0.8128,

            "recall_before": 0.5727,

            "recall_after": 0.3497,

            "f1_before": 0.5354,

            "f1_after": 0.4525

        }

    }