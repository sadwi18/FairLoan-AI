import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import prepare_data


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/credit_default.xls"

X, y, sensitive_feature = prepare_data(DATA_PATH)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X,
    y,
    sensitive_feature,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# TRAIN FINAL RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(model, "models/loan_default_model.pkl")


# Save feature names
joblib.dump(list(X.columns), "models/feature_names.pkl")


print("=" * 70)
print("FINAL MODEL SAVED SUCCESSFULLY")
print("=" * 70)

print("\nModel: Random Forest")
print("Number of trees:", model.n_estimators)
print("Max depth:", model.max_depth)

print("\nFeatures used:")
for feature in X.columns:
    print("-", feature)

print("\nModel saved to:")
print("models/loan_default_model.pkl")

print("\nFeature names saved to:")
print("models/feature_names.pkl")
