import pandas as pd


def load_data(filepath):
    """Load and prepare the credit default dataset."""

    # Read Excel file
    df = pd.read_excel(filepath, header=1)

    # Rename columns
    df = df.rename(columns={
        "ID": "id",
        "LIMIT_BAL": "credit_limit",
        "SEX": "sex",
        "EDUCATION": "education",
        "MARRIAGE": "marriage",
        "AGE": "age",

        "PAY_0": "pay_sep",
        "PAY_2": "pay_aug",
        "PAY_3": "pay_jul",
        "PAY_4": "pay_jun",
        "PAY_5": "pay_may",
        "PAY_6": "pay_apr",

        "BILL_AMT1": "bill_sep",
        "BILL_AMT2": "bill_aug",
        "BILL_AMT3": "bill_jul",
        "BILL_AMT4": "bill_jun",
        "BILL_AMT5": "bill_may",
        "BILL_AMT6": "bill_apr",

        "PAY_AMT1": "payment_sep",
        "PAY_AMT2": "payment_aug",
        "PAY_AMT3": "payment_jul",
        "PAY_AMT4": "payment_jun",
        "PAY_AMT5": "payment_may",
        "PAY_AMT6": "payment_apr",

        "default payment next month": "default"
    })

    return df


def prepare_data(filepath):
    """Prepare features, target, and fairness attribute."""

    df = load_data(filepath)

    # Keep sex separately for fairness analysis
    sensitive_feature = df["sex"].copy()

    # Remove ID and sensitive feature from prediction features
    X = df.drop(
        columns=["id", "sex", "default"]
    )

    # Target
    y = df["default"]

    return X, y, sensitive_feature