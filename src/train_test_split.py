from sklearn.model_selection import train_test_split

from data_preprocessing import prepare_data


# ---------------------------------------------
# Load and prepare data
# ---------------------------------------------

X, y, sensitive_feature = prepare_data(
    "data/credit_default.xls"
)


# ---------------------------------------------
# Train/Test Split
# ---------------------------------------------

X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
    X,
    y,
    sensitive_feature,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ---------------------------------------------
# Display results
# ---------------------------------------------

print("=" * 50)
print("FAIRLOAN AI - TRAIN TEST SPLIT")
print("=" * 50)

print("\nOriginal dataset:")
print("X:", X.shape)
print("y:", y.shape)

print("\nTraining data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTesting data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

print("\nSensitive feature:")
print("sensitive_train:", sensitive_train.shape)
print("sensitive_test:", sensitive_test.shape)


# ---------------------------------------------
# Target distribution
# ---------------------------------------------

print("\nTarget distribution - Training:")
print(y_train.value_counts())
print(
    (y_train.value_counts(normalize=True) * 100).round(2)
)

print("\nTarget distribution - Testing:")
print(y_test.value_counts())
print(
    (y_test.value_counts(normalize=True) * 100).round(2)
)


# ---------------------------------------------
# Feature columns
# ---------------------------------------------

print("\nPrediction features:")
print(X.columns.tolist())