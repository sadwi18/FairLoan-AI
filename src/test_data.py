import pandas as pd

# Load the dataset
df = pd.read_excel("data/credit_default.xls", header=1)

print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)