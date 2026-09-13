import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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
# 4. CREATE SHAP EXPLAINER
# ==================================================

explainer = shap.TreeExplainer(model)


# Use a sample of the test data for faster SHAP analysis
X_sample = X_test.sample(
    min(1000, len(X_test)),
    random_state=42
)

shap_values = explainer.shap_values(X_sample)


# ==================================================
# 5. GLOBAL FEATURE IMPORTANCE
# ==================================================

print("=" * 60)
print("FAIRLOAN AI - SHAP EXPLAINABILITY")
print("=" * 60)

print("\nMost Important Features")

importance = pd.DataFrame({
    "feature": X_sample.columns,
    "importance": abs(shap_values[:, :, 1]).mean(axis=0)
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print(importance.head(10))


# ==================================================
# 6. SHAP SUMMARY PLOT
# ==================================================

plt.figure()

shap.summary_plot(
    shap_values[:, :, 1],
    X_sample,
    show=False
)

plt.title("SHAP Feature Importance - Default Prediction")

plt.tight_layout()

plt.savefig(
    "output/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nSHAP summary plot saved:")
print("output/shap_summary.png")