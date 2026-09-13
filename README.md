# FairLoan AI

### Explainable & Fair Credit Default Risk Prediction System

FairLoan AI is an end-to-end machine learning application that predicts **next-month credit-card payment default risk**, explains each prediction, and audits whether the model treats applicant groups differently. It combines Machine Learning, Explainable AI (SHAP), Fairness Evaluation (Fairlearn), FastAPI, and Streamlit into a deployed web application.

> **Note:** Despite the name, the underlying dataset is the **UCI Default of Credit Card Clients** dataset. The model predicts next-month credit-card payment default risk, not general personal-loan default. Built for educational/demonstration purposes — not a real-world lending decision system.

---

## 🌐 Live Application

| | |
|---|---|
| **Streamlit Dashboard** | https://fairloan-ai-2ew6outepx76gqozcjvjwb.streamlit.app |
| **FastAPI Backend** | https://fairloan-ai-wqon.onrender.com |
| **Swagger UI (API docs)** | https://fairloan-ai-wqon.onrender.com/docs |

---

## Problem Statement

Risk prediction models are more useful, and more trustworthy, when they can also answer:

1. **Why did the model make this prediction?**
2. **Does the model behave differently across applicant groups?**

FairLoan AI addresses both by combining default risk prediction with SHAP-based explainability and a Fairlearn-based fairness audit and mitigation experiment, all served through a REST API and an interactive dashboard.

---

## Key Features

### 1. Credit Default Risk Prediction
Predicts whether an applicant is likely to default on their next month's credit-card payment, returning a default/no-default label, default probability, and risk category:

| Probability | Risk Level |
|---|---|
| < 30% | Low Risk |
| 30% – 60% | Medium Risk |
| ≥ 60% | High Risk |

*(Project-defined demonstration thresholds — not validated lending policy.)*

### 2. Explainable AI (SHAP)
Rather than only returning a prediction, the system explains which features drove it and in which direction — global feature importance, individual prediction explanations, and top risk-increasing/decreasing factors.

```
Prediction: No Default
Default Probability: 33.63%
Top Factors:
  pay_sep     → Decreases risk
  credit_limit → Increases risk
  pay_aug     → Decreases risk
  payment_jul → Decreases risk
```

### 3. Fairness Audit (Fairlearn)
Evaluates potential disparities across the `SEX` sensitive attribute — which is kept separate for evaluation and **excluded** from the model's predictive features. Metrics: Selection Rate, TPR, FPR, Demographic Parity Difference, Equalized Odds Difference.

### 4. Fairness Mitigation
Applies Fairlearn's `ThresholdOptimizer` with an Equalized Odds constraint as a post-processing mitigation experiment, and reports the resulting trade-offs.

---

## Fairness Results

| Metric | Before Mitigation | After Mitigation |
|---|---|---|
| Accuracy | 78.02% | 81.28% |
| Recall | 57.27% | 34.97% |
| F1-score | 53.54% | 45.25% |
| Demographic Parity Difference | 0.0331 | 0.0075 |
| Equalized Odds Difference | 0.0320 | 0.0561 |

**Interpretation:** Mitigation cut Demographic Parity Difference from 0.0331 → 0.0075, but Equalized Odds Difference rose (0.0320 → 0.0561) and recall/F1 dropped. This illustrates a core fairness-ML principle: improving one fairness criterion doesn't guarantee improvement on every criterion or on predictive performance — mitigation choices need to match the deployment's actual objectives.

---

## Model

**Random Forest Classifier** (primary model)
```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)
```
Chosen over Logistic Regression for stronger recall, F1-score, and ROC-AUC. `class_weight="balanced"` compensates for the imbalanced target (default rate ≈ 22%).

| Metric | Random Forest | Logistic Regression (baseline) |
|---|---|---|
| Accuracy | 78.02% | 80.80% |
| Precision | 50.26% | 68.74% |
| Recall | 57.27% | 24.19% |
| F1-score | 53.54% | 35.79% |
| ROC-AUC | 77.32% | 70.64% |

Logistic Regression edges out on accuracy, but Random Forest is substantially better at catching the minority default class (recall, F1).

**Confusion Matrix (Random Forest)**
```
                 Predicted
              No Default  Default
Actual
No Default        3921       752
Default             567       760
```

**Top SHAP Features:** `pay_sep`, `pay_aug`, `credit_limit`, `pay_jul`, `pay_jun`, `payment_aug`, `payment_sep`, `payment_jul`, `bill_sep`, `pay_may` — repayment-status variables dominate.

---

## Dataset

**UCI Default of Credit Card Clients** — 30,000 records, 23 explanatory variables (credit limit, demographics, repayment history, bill amounts, prior payments).

- Target: `default` (0 = No Default, 1 = Default) — 23,364 vs 6,636 (22.12% default rate)
- Sensitive attribute: `SEX` (1 = Male, 2 = Female) — retained separately for fairness evaluation, not used as a model input
- Split: 80/20 stratified train/test (24,000 / 6,000)

---

## System Architecture

```
                     USER
                      │
                      ▼
          ┌─────────────────────────┐
          │   Streamlit Dashboard   │
          │  Prediction · Explain-  │
          │  ability · Fairness     │
          └────────────┬────────────┘
                       │ HTTPS
                       ▼
          ┌─────────────────────────┐
          │     FastAPI Backend     │
          │ /predict /explain       │
          │ /fairness               │
          └────────────┬────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  Random Forest      SHAP        Fairlearn
        │              │              │
        ▼              ▼              ▼
   Prediction    Explanation      Fairness
        │              │              │
        └──────────────┴──────────────┘
                       ▼
                Final Results
```

**End-to-end flow:** applicant info → Streamlit → FastAPI → feature prep → Random Forest prediction → risk category → SHAP explanation → Fairlearn fairness check → results rendered in Streamlit.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python |
| ML | Scikit-learn (Random Forest, Logistic Regression) |
| Explainability | SHAP |
| Fairness | Fairlearn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Plotly |
| Backend | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Deployment | Render, Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## Project Structure

```
FairLoan-AI/
├── data/
│   └── credit_default.xls
├── models/
│   ├── loan_default_model.pkl
│   └── feature_names.pkl
├── output/                      # EDA charts + SHAP summary
├── src/
│   ├── data_preprocessing.py
│   ├── eda.py
│   ├── visual_eda.py
│   ├── train_test_split.py
│   ├── train_model.py
│   ├── train_random_forest.py
│   ├── save_model.py
│   ├── predict.py
│   ├── explain_model.py
│   ├── local_explanation.py
│   ├── fairness_audit.py
│   ├── fairness_mitigation.py
│   └── test_data.py
├── app.py                       # Streamlit frontend
├── main.py                      # FastAPI backend
├── requirements.txt
└── .gitignore
```

---

## API Endpoints

**`GET /`** — health check

**`POST /predict`** — returns a default-risk prediction
```json
{
  "prediction": "No Default",
  "probability": 0.3363,
  "probability_percent": 33.63,
  "risk": "Medium Risk"
}
```

**`POST /explain`** — returns a SHAP-based explanation for the applicant's prediction

**`GET /fairness`** — returns the fairness audit and mitigation results
```json
{
  "before_mitigation": {
    "demographic_parity_difference": 0.0331,
    "equalized_odds_difference": 0.032
  },
  "after_mitigation": {
    "demographic_parity_difference": 0.0075,
    "equalized_odds_difference": 0.0561
  }
}
```

---

## Run Locally

```bash
# 1. Clone
git clone https://github.com/sadwi18/FairLoan-AI.git
cd FairLoan-AI

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Backend**
```bash
uvicorn main:app --reload
# → http://127.0.0.1:8000  (docs at /docs)
```

**Frontend** (second terminal, same venv)
```bash
streamlit run app.py
```

The Streamlit app reads the backend URL from the `API_URL` environment variable, falling back to `http://127.0.0.1:8000` for local development — so the same code runs locally and in production.

---

## Design Decisions

- **`ID` removed** — it's a record identifier, not a predictive signal, and including it risks learning arbitrary patterns.
- **`SEX` excluded from features** — kept only as a sensitive attribute for fairness evaluation, so outcomes can be checked across groups without using the attribute to predict.
- **`class_weight="balanced"`** — the target is imbalanced (≈78% no-default / 22% default), so balancing gives the minority default class fair weight during training.
- **SHAP** — accuracy alone doesn't explain *why* a prediction was made; SHAP attributes each prediction to contributing features.
- **Fairlearn** — provides standard metrics and mitigation techniques for evaluating and addressing group-level disparities.

---

## Responsible AI & Limitations

FairLoan AI is an educational/research project and should **not** be used for real-world loan approval, credit approval, underwriting, or automated lending decisions.

- Trained on historical data, which may encode existing biases or population-specific patterns
- May not generalize to all populations or lending environments
- Fairness is assessed via a specific set of statistical metrics on one sensitive attribute — other metrics or attributes could yield different conclusions
- SHAP explains model behavior, not causal relationships
- Risk thresholds are demonstration-only
- Production use would require further validation, monitoring, and legal/compliance review

---

## Future Enhancements

- Additional fairness metrics and protected attributes
- Probability calibration
- Model & data drift monitoring
- Additional models (XGBoost, LightGBM)
- Automated retraining pipeline
- API authentication and rate limiting
- CI/CD via GitHub Actions
- Cost-sensitive threshold optimization

---

## Academic Value

Demonstrates practical implementation of: supervised ML, classification on imbalanced data, EDA, feature engineering, model evaluation, explainable AI (SHAP), fairness metrics and mitigation, REST API development, Streamlit app development, and cloud deployment.

---

## Reference

Yeh, I. C., & Lien, C. H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients.* UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

---

## Author

**Korada Satya Sadwika**
B.Tech – Computer Science (Artificial Intelligence & Machine Learning)
GitHub: [@sadwi18](https://github.com/sadwi18)

If you find this project useful, consider starring the repo ⭐
