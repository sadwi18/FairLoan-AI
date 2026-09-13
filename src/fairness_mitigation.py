import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from fairlearn.postprocessing import ThresholdOptimizer
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
# 4. ORIGINAL MODEL PREDICTIONS
# ==================================================

y_pred_original = model.predict(X_test)


# ==================================================
# 5. FAIRNESS METRICS FUNCTION
# ==================================================

metrics = {
    "Selection Rate": selection_rate,
    "True Positive Rate": true_positive_rate,
    "False Positive Rate": false_positive_rate
}


def evaluate_model(y_true, y_pred, sensitive):
    """Calculate performance and fairness metrics."""

    metric_frame = MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    dp_difference = demographic_parity_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive
    )

    eo_difference = equalized_odds_difference(
        y_true,
        y_pred,
        sensitive_features=sensitive
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        dp_difference,
        eo_difference,
        metric_frame
    )


# ==================================================
# 6. EVALUATE ORIGINAL MODEL
# ==================================================

(
    accuracy_before,
    precision_before,
    recall_before,
    f1_before,
    dp_before,
    eo_before,
    frame_before
) = evaluate_model(
    y_test,
    y_pred_original,
    sensitive_test
)


# ==================================================
# 7. FAIRNESS MITIGATION
# ==================================================

threshold_optimizer = ThresholdOptimizer(
    estimator=model,
    constraints="equalized_odds",
    predict_method="predict_proba",
    prefit=True
)

threshold_optimizer.fit(
    X_train,
    y_train,
    sensitive_features=sensitive_train
)


# ==================================================
# 8. MITIGATED PREDICTIONS
# ==================================================

y_pred_mitigated = threshold_optimizer.predict(
    X_test,
    sensitive_features=sensitive_test,
    random_state=42
)


# ==================================================
# 9. EVALUATE MITIGATED MODEL
# ==================================================

(
    accuracy_after,
    precision_after,
    recall_after,
    f1_after,
    dp_after,
    eo_after,
    frame_after
) = evaluate_model(
    y_test,
    y_pred_mitigated,
    sensitive_test
)


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("=" * 70)
print("FAIRLOAN AI - FAIRNESS MITIGATION")
print("=" * 70)


print("\nBEFORE MITIGATION")

print(f"Accuracy              : {accuracy_before:.4f}")
print(f"Precision             : {precision_before:.4f}")
print(f"Recall                : {recall_before:.4f}")
print(f"F1 Score              : {f1_before:.4f}")
print(f"Demographic Parity    : {dp_before:.4f}")
print(f"Equalized Odds        : {eo_before:.4f}")


print("\nMetrics by Sex - BEFORE")

print(frame_before.by_group)


print("\nAFTER MITIGATION")

print(f"Accuracy              : {accuracy_after:.4f}")
print(f"Precision             : {precision_after:.4f}")
print(f"Recall                : {recall_after:.4f}")
print(f"F1 Score              : {f1_after:.4f}")
print(f"Demographic Parity    : {dp_after:.4f}")
print(f"Equalized Odds        : {eo_after:.4f}")


print("\nMetrics by Sex - AFTER")

print(frame_after.by_group)


print("\nFAIRNESS IMPROVEMENT")

print(
    f"Demographic Parity Change: "
    f"{dp_before:.4f} → {dp_after:.4f}"
)

print(
    f"Equalized Odds Change: "
    f"{eo_before:.4f} → {eo_after:.4f}"
)


print("\nPerformance Change")

print(
    f"Accuracy: "
    f"{accuracy_before:.4f} → {accuracy_after:.4f}"
)

print(
    f"Recall: "
    f"{recall_before:.4f} → {recall_after:.4f}"
)

print(
    f"F1 Score: "
    f"{f1_before:.4f} → {f1_after:.4f}"
)