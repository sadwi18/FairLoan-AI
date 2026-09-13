import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference
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
# 3. TRAIN RANDOM FOREST
# ==================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


# ==================================================
# 4. PREDICTIONS
# ==================================================

y_pred = model.predict(X_test)


# ==================================================
# 5. FAIRNESS METRICS
# ==================================================

metrics = {
    "Selection Rate": selection_rate,
    "True Positive Rate": true_positive_rate,
    "False Positive Rate": false_positive_rate
}


metric_frame = MetricFrame(
    metrics=metrics,
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_test
)


# ==================================================
# 6. DISPLAY GROUP METRICS
# ==================================================

print("=" * 60)
print("FAIRLOAN AI - FAIRNESS AUDIT")
print("=" * 60)

print("\nMetrics by Sex")

print(metric_frame.by_group)


# ==================================================
# 7. OVERALL METRICS
# ==================================================

print("\nOverall Metrics")

print(metric_frame.overall)


# ==================================================
# 8. DEMOGRAPHIC PARITY
# ==================================================

dp_difference = demographic_parity_difference(
    y_test,
    y_pred,
    sensitive_features=sensitive_test
)


print("\nDemographic Parity Difference:")
print(round(dp_difference, 4))


# ==================================================
# 9. EQUALIZED ODDS
# ==================================================

eo_difference = equalized_odds_difference(
    y_test,
    y_pred,
    sensitive_features=sensitive_test
)


print("\nEqualized Odds Difference:")
print(round(eo_difference, 4))


# ==================================================
# 10. GROUP COMPARISON
# ==================================================

print("\nInterpretation Guide")

print(
    "\nDemographic Parity Difference:"
    "\nCloser to 0 → more similar positive prediction rates."
)

print(
    "\nEqualized Odds Difference:"
    "\nCloser to 0 → more similar error/true-positive rates."
)

print(
    "\nImportant:"
    "\nThese metrics identify disparities; they do not by themselves "
    "prove that discrimination exists."
)