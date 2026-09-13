import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from data_preprocessing import prepare_data


# ==================================================
# 1. LOAD DATA
# ==================================================

X, y, sensitive_feature = prepare_data(
    "data/credit_default.xls"
)


# ==================================================
# 2. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X,
    y,
    sensitive_feature,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ==================================================
# 3. FEATURE SCALING
# ==================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ==================================================
# 4. TRAIN LOGISTIC REGRESSION
# ==================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)


# ==================================================
# 5. MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(X_test_scaled)

y_probability = model.predict_proba(X_test_scaled)[:, 1]


# ==================================================
# 6. MODEL EVALUATION
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(y_test, y_probability)

cm = confusion_matrix(y_test, y_pred)


# ==================================================
# 7. DISPLAY RESULTS
# ==================================================

print("=" * 60)
print("FAIRLOAN AI - LOGISTIC REGRESSION")
print("=" * 60)

print("\nModel Performance")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")


print("\nConfusion Matrix")

print(cm)


print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Default",
            "Default"
        ]
    )
)