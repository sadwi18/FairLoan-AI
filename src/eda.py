import pandas as pd

from data_preprocessing import load_data


# Load dataset
df = load_data("data/credit_default.xls")


print("=" * 50)
print("FAIRLOAN AI - DATASET ANALYSIS")
print("=" * 50)


# 1. Dataset size
print("\n1. Dataset Shape")
print(df.shape)


# 2. Column names
print("\n2. Columns")
print(df.columns.tolist())


# 3. Missing values
print("\n3. Missing Values")
print(df.isnull().sum())


# 4. Duplicate rows
print("\n4. Duplicate Rows")
print(df.duplicated().sum())


# 5. Target distribution
print("\n5. Default Distribution")
print(df["default"].value_counts())

print("\nDefault Percentage")
print(df["default"].value_counts(normalize=True) * 100)


# 6. Basic statistics
print("\n6. Numerical Summary")
print(df.describe())


# 7. Sex distribution
print("\n7. Sex Distribution")
print(df["sex"].value_counts())


# 8. Default rate by sex
print("\n8. Default Rate by Sex")
print(
    df.groupby("sex")["default"]
    .mean()
    .mul(100)
    .round(2)
)


# 9. Average credit limit by default status
print("\n9. Average Credit Limit by Default Status")
print(
    df.groupby("default")["credit_limit"]
    .mean()
    .round(2)
)


# 10. Average age by default status
print("\n10. Average Age by Default Status")
print(
    df.groupby("default")["age"]
    .mean()
    .round(2)
)