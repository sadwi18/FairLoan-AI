# FairLoan AI
## Explainable & Fair Credit Default Risk Prediction System
FairLoan AI is an end-to-end machine learning application that predicts **next-month credit-card payment
default risk** while making each prediction explainable and evaluating whether the model produces different
outcomes across applicant groups.
The project combines **Machine Learning, Explainable AI (XAI), Fairness Evaluation, FastAPI, Streamlit, and
Cloud Deployment** into a complete web-based application.
> **Important:** Although the project is named **FairLoan AI**, the underlying dataset is the **UCI Default
of Credit Card Clients** dataset. Therefore, the model predicts the risk of **next-month credit-card payment
default**, not general personal-loan default. This project is intended for educational and demonstration
purposes and should not be used as a real-world lending decision system.
---
## 🌐 Live Application

### Streamlit Dashboard
**Live App:**  
https://fairloan-ai-2ew6outepx76gqozcjvjwb.streamlit.app
### FastAPI Backend
**API:**  
https://fairloan-ai-wqon.onrender.com
### Interactive API Documentation
**Swagger UI:**  
https://fairloan-ai-wqon.onrender.com/docs
---
## ■ Problem Statement
Credit-risk prediction systems can provide useful risk estimates, but two important questions must also be
addressed:
1. **Why did the model make this prediction?**
2. **Does the model behave differently for different applicant groups?**
FairLoan AI addresses these challenges by combining:
- Credit default risk prediction
- Probability-based risk assessment
- SHAP-based model explainability
- Global and local explanations
- Fairness auditing using Fairlearn
- Fairness mitigation experiments
- REST API integration
- Interactive web dashboard
- Cloud deployment
---
## ■ Key Features
### 1. Credit Default Risk Prediction
The system predicts whether an applicant is likely to default on their **next month's credit-card payment**.
The application provides:
- Default / No Default prediction
- Default probability
- Risk percentage
- Risk category
### Project-defined Risk Categories
| Probability | Risk Level |
|---|---|
| < 30% | Low Risk |
| 30% – 60% | Medium Risk |
| ≥ 60% | High Risk |
> These thresholds are project-defined demonstration thresholds and have not been validated as real-world
lending policy thresholds.
---
### 2. Explainable AI using SHAP
FairLoan AI uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.
Instead of only showing:
> "Default predicted."
the system provides information about:
> "Which features contributed to this prediction and in which direction?"
### Explainability includes:
- Global feature importance
- Individual prediction explanations
- Top factors influencing the prediction
- Factors increasing risk
- Factors decreasing risk
- SHAP contribution values
### Example
```text
Prediction: No Default
Default Probability: 33.63%
Important Factors:
pay_sep → Decreases risk
credit_limit → Increases risk
pay_aug → Decreases risk
payment_jul → Decreases risk
```
> SHAP values explain the behavior of the trained model. They do not establish causal relationships.
---
### 3. Fairness Audit using Fairlearn
The project evaluates potential fairness disparities across the `SEX` sensitive attribute.
The sensitive attribute is:
- Kept separately for fairness evaluation
- Excluded from the model's predictive feature set
The fairness analysis evaluates:
- Selection Rate
- True Positive Rate (TPR)
- False Positive Rate (FPR)
- Demographic Parity Difference
- Equalized Odds Difference
---
### 4. Fairness Mitigation
FairLoan AI also experiments with a fairness-aware post-processing technique using:
**Fairlearn `ThresholdOptimizer`**
with:
```text
Equalized Odds
```
as the fairness constraint.
The mitigation experiment demonstrates the trade-offs between:
- Predictive performance
- Fairness
- Recall
- F1-score
- Group-level outcomes
---
## ■ Fairness Results
### Before Mitigation
| Metric | Result |
|---|---:|
| Accuracy | 78.02% |
| Recall | 57.27% |
| F1-score | 53.54% |
| Demographic Parity Difference | 0.0331 |
| Equalized Odds Difference | 0.0320 |
### After Mitigation
| Metric | Result |
|---|---:|
| Accuracy | 81.28% |
| Recall | 34.97% |
| F1-score | 45.25% |
| Demographic Parity Difference | 0.0075 |
| Equalized Odds Difference | 0.0561 |
### Interpretation
The mitigation experiment reduced the **Demographic Parity Difference** from:
```text
0.0331 → 0.0075
```
However, the **Equalized Odds Difference** increased:
```text
0.0320 → 0.0561
```
At the same time, recall and F1-score decreased.
This demonstrates an important practical machine learning fairness principle:
> **Improving one fairness criterion does not necessarily improve every fairness criterion or predictiveperformance metric.**
Therefore, fairness mitigation should be evaluated according to the specific objectives and constraints of a
real deployment rather than being treated as an automatic improvement.
---
# ■ Machine Learning Model
The primary model used in the project is a:
## Random Forest Classifier
Configuration:
```python
RandomForestClassifier(
 n_estimators=200,
 max_depth=12,
 random_state=42,
 class_weight="balanced",
 n_jobs=-1
)
```
### Why Random Forest?
Random Forest was selected as the primary model because it provided stronger:
- Recall
- F1-score
- ROC-AUC
than the Logistic Regression baseline used in the project.
The `class_weight="balanced"` setting was used because the target classes are imbalanced.
---
# ■ Model Performance
The model was evaluated on a held-out test set.
## Random Forest Results
| Metric | Score |
|---|---:|
| Accuracy | 78.02% |
| Precision | 50.26% |
| Recall | 57.27% |
| F1-score | 53.54% |
| ROC-AUC | 77.32% |
### Confusion Matrix
```text
 Predicted
 No Default Default
Actual
No Default 3921 752
Default 567 760
```
---
# ■ Baseline Model
A Logistic Regression model was also trained as a baseline.
## Logistic Regression Results
| Metric | Score |
|---|---:|
| Accuracy | 80.80% |
| Precision | 68.74% |
| Recall | 24.19% |
| F1-score | 35.79% |
| ROC-AUC | 70.64% |
Although Logistic Regression achieved higher accuracy, the Random Forest provided substantially better recall
and F1-score for identifying the minority default class.
---
# ■ Explainability Results
Global SHAP analysis identified several highly influential features.
### Top Features
| Rank | Feature |
|---:|---|
| 1 | `pay_sep` |
| 2 | `pay_aug` |
| 3 | `credit_limit` |
| 4 | `pay_jul` |
| 5 | `pay_jun` |
| 6 | `payment_aug` |
| 7 | `payment_sep` |
| 8 | `payment_jul` |
| 9 | `bill_sep` |
| 10 | `pay_may` |
The repayment-status variables are particularly influential in the trained model.
---
# ■ Dataset
The project uses the:
## UCI Default of Credit Card Clients Dataset
### Dataset Information
- 30,000 records
- 23 explanatory variables
- Credit limit information
- Demographic information
- Repayment history
- Bill amounts
- Previous payment amounts
- Default target variable
### Target Variable
```text
default = 0 → No Default
default = 1 → Default
```
### Target Distribution
```text
No Default : 23,364
Default : 6,636
```
Default rate:
```text
22.12%
```
### Sensitive Attribute
The `SEX` attribute contains:
```text
1 → Male
2 → Female
```
The sensitive attribute is retained separately for fairness evaluation and is not used as a predictive input
feature.
---
# ■ Data Preprocessing
The preprocessing pipeline includes:
- Loading the UCI dataset
- Correctly handling the dataset header
- Renaming columns for readability
- Separating the target variable
- Separating the sensitive attribute
- Removing the `ID` field from model inputs
- Checking missing values
- Checking duplicate records
- Creating training and testing datasets
- Stratified train-test splitting
### Dataset Split
```text
Training Set : 24,000 records
Testing Set : 6,000 records
```
The split uses:
```python
train_test_split(
 test_size=0.20,
 random_state=42,
 stratify=y
)
```
---
# ■ Exploratory Data Analysis
The project includes exploratory analysis of:
- Default distribution
- Age distribution
- Average credit limit
- Credit limit vs. age
- Default rate by sex
- Default rate by education
- Default vs. non-default distribution
Generated visualizations are stored in:
```text
output/
```
---
# ■■ System Architecture
```text
 USER
 ■
▼
 ■■■■■■■■■■■■■■■■■■■■■■■
 ■ Streamlit Dashboard ■
 ■ ■
 ■ • Prediction ■
 ■ • Explainability ■
 ■ • Fairness ■
 ■■■■■■■■■■■■■■■■■■■■■■■
 ■
■ HTTPS
▼
 ■■■■■■■■■■■■■■■■■■■■■■■
 ■ FastAPI Backend ■
 ■ ■
 ■ /predict ■
 ■ /explain ■
 ■ /fairness ■
 ■■■■■■■■■■■■■■■■■■■■■■■
 ■
 ■■■■■■■■■■■■■■■■■■■■■■■■■
 ■ ■ ■
 ▼ ▼ ▼
 Random Forest SHAP Fairlearn
 ■ ■ ■
 ▼ ▼ ▼
 Prediction Explanation Fairness
 ■ ■ ■
 ■■■■■■■■■■■■■■■■■■■■■■■■■
 ▼
 Final Results
```
---
# ■ End-to-End Workflow
```text
1. User enters applicant information
 ↓
2. Streamlit collects input
 ↓
3. Input is sent to FastAPI
 ↓
4. FastAPI prepares the features
 ↓
5. Random Forest predicts default probability
 ↓
6. Probability is converted into a risk category
 ↓
7. SHAP explains important factors
 ↓
8. Fairness metrics evaluate group-level behavior
 ↓
9. Results are displayed in Streamlit
```
---
# ■■ Technology Stack
## Programming Language
- Python
## Machine Learning
- Scikit-learn
- Random Forest
- Logistic Regression
## Explainable AI
- SHAP
## Fairness
- Fairlearn
## Data Processing
- Pandas
- NumPy
## Visualization
- Matplotlib
- Plotly
## Backend
- FastAPI
- Uvicorn
- Pydantic
## Frontend
- Streamlit
## Deployment
- Render
- Streamlit Community Cloud
## Version Control
- Git
- GitHub
---
# ■ Project Structure
```text
FairLoan-AI/
■
■■■ data/
■ ■■■ credit_default.xls
■
■■■ models/
■ ■■■ loan_default_model.pkl
■ ■■■ feature_names.pkl
■
■■■ output/
■ ■■■ age_distribution.html
■ ■■■ average_credit_limit.html
■ ■■■ credit_limit_vs_age.html
■ ■■■ default_rate_by_education.html
■ ■■■ default_rate_by_sex.html
■ ■■■ default_vs_nondefault.html
■ ■■■ shap_summary.png
■
■■■ src/
■ ■■■ data_preprocessing.py
■ ■■■ eda.py
■ ■■■ explain_model.py
■ ■■■ fairness_audit.py
■ ■■■ fairness_mitigation.py
■ ■■■ local_explanation.py
■ ■■■ predict.py
■ ■■■ save_model.py
■ ■■■ test_data.py
■ ■■■ train_model.py
■ ■■■ train_random_forest.py
■ ■■■ train_test_split.py
■ ■■■ visual_eda.py
■
■■■ app.py
■■■ main.py
■■■ README.md
■■■ requirements.txt
■■■ .gitignore
■■■ venv/
```
> `venv/` is excluded from Git using `.gitignore` and should not be uploaded to GitHub.
---
# ■■ FastAPI Endpoints
## `GET /`
Checks whether the API is running.
---
## `POST /predict`
Generates a default-risk prediction.
### Returns
```json
{
 "prediction": "No Default",
 "probability": 0.3363,
 "probability_percent": 33.63,
 "risk": "Medium Risk"
}
```
---
## `POST /explain`
Generates an individual SHAP-based explanation for the applicant.
The response contains the most influential factors contributing to the model prediction.
---
## `GET /fairness`
Returns the project's fairness audit and mitigation results.
Example structure:
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
# ■ Run the Project Locally
## Step 1: Clone the Repository
```bash
git clone https://github.com/sadwi18/FairLoan-AI.git
```
Navigate into the project:
```bash
cd FairLoan-AI
```
---
## Step 2: Create a Virtual Environment
```bash
python -m venv venv
```
---
## Step 3: Activate the Virtual Environment
### Windows
```bash
venv\Scripts\activate
```
### macOS/Linux
```bash
source venv/bin/activate
```
---
## Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```
---
# ■■ Run the FastAPI Backend
Start the backend:
```bash
uvicorn main:app --reload
```
The API will be available at:
```text
http://127.0.0.1:8000
```
Open Swagger UI:
```text
http://127.0.0.1:8000/docs
```
---
# ■■ Run the Streamlit Frontend
Open a second terminal and activate the virtual environment.
Then run:
```bash
streamlit run app.py
```
The dashboard will open in your browser.
---
# ■ Deployment Architecture
The deployed application uses a separated frontend/backend architecture:
```text
 User
 ■
▼
 Streamlit Community Cloud
 ■
 HTTPS
 ■
▼
 Render Web Service
 ■
▼
 FastAPI Backend
 ■
 ■■■■■■■■■■■■■■■■■■■■■■■■■■■
 ▼ ▼ ▼
 Random Forest SHAP Fairlearn
 ■ ■ ■
 ■■■■■■■■■■■■■■■■■■■■■■■■■■■
 ▼
 Prediction Results
```
The Streamlit application reads the backend URL from the `API_URL` environment variable.
For local development, it falls back to:
```text
http://127.0.0.1:8000
```
---
# ■ Environment Configuration
The Streamlit application uses an environment variable for the FastAPI backend URL.
```text
API_URL
```
For the deployed application:
```text
API_URL = "https://fairloan-ai-wqon.onrender.com"
```
For local development, the application falls back to:
```text
http://127.0.0.1:8000
```
This allows the same Streamlit application to work both locally and in production.
---
# ■ Important Design Decisions
## Why is `ID` removed?
The dataset ID is an identifier rather than a meaningful predictive feature. Including it could introduce
arbitrary patterns that do not represent applicant behavior.
---
## Why is `SEX` excluded from prediction?
`SEX` is treated as a sensitive attribute for fairness analysis.
It is kept separately so the project can evaluate whether model outcomes differ across groups without
directly using the sensitive attribute as a model feature.
---
## Why use `class_weight="balanced"`?
The target is imbalanced:
```text
No Default → 77.88%
Default → 22.12%
```
Using balanced class weights gives greater importance to the minority default class during training.
---
## Why use SHAP?
Accuracy alone does not explain a model's decision.
SHAP helps identify which features contributed most to an individual prediction and provides a framework for
understanding feature contributions.
---
## Why use Fairlearn?
Fairlearn provides metrics and mitigation techniques for evaluating and addressing potential disparities in
machine learning systems.
---
# ■■ Responsible AI & Limitations
FairLoan AI is an **educational and research-oriented project**.
It should **not** be used as a standalone system for:
- Real-world loan approval
- Credit approval
- Financial underwriting
- Automated lending decisions
### Important limitations
1. The model is trained on historical credit-card default data.
2. Historical data may contain existing biases or population-specific patterns.
3. The dataset may not represent all populations or lending environments.
4. Fairness is evaluated using selected statistical metrics.
5. The current fairness analysis focuses on one sensitive attribute.
6. Different fairness metrics can produce different conclusions.
7. SHAP explains model behavior but does not establish causality.
8. Risk thresholds used in this project are demonstration thresholds.
9. Production deployment would require extensive validation and governance.
10. Real-world financial applications require appropriate legal, privacy, security, compliance, and domain
review.
---
# ■ Future Enhancements
Potential future improvements include:
- Additional fairness metrics
- Evaluation across additional protected attributes where appropriate
- Probability calibration
- Model drift detection
- Data drift monitoring
- Model performance monitoring
- Additional ML models such as XGBoost and LightGBM
- Automated retraining pipelines
- API authentication
- API rate limiting
- Secure logging
- Automated fairness reports
- Model versioning
- CI/CD using GitHub Actions
- Cost-sensitive threshold optimization
- Improved production monitoring
---
# ■ Project Outcomes
FairLoan AI demonstrates how a machine learning system can go beyond simple prediction by incorporating:
```text
Prediction
 +
Explainability
 +
Fairness
 +
Mitigation
 +
API
 +
Interactive Dashboard
 +
Cloud Deployment
```
This creates a more transparent and responsible ML workflow.
---
# ■ Academic Value
The project demonstrates practical implementation of:
- Supervised Machine Learning
- Classification
- Imbalanced Data Handling
- Exploratory Data Analysis
- Feature Engineering
- Model Evaluation
- Explainable AI
- SHAP
- Fairness Metrics
- Fairness Mitigation
- REST API Development
- Streamlit Application Development
- Cloud Deployment
- Git and GitHub
---
# ■ References
### Dataset
Yeh, I. C., & Lien, C. H. (2009).
**The comparisons of data mining techniques for the predictive accuracy of probability of default of credit
card clients.**
UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients
### Libraries
- Scikit-learn
- SHAP
- Fairlearn
- FastAPI
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
---
# ■■■ Author
## Korada Satya Sadwika
**B.Tech – Computer Science (Artificial Intelligence & Machine Learning)**
GitHub:
https://github.com/sadwi18
---
# ■ Project
If you find this project useful or interesting, consider giving the repository a ■ on GitHub.
**FairLoan AI — Making Credit Risk Prediction More Explainable and Fair.**
