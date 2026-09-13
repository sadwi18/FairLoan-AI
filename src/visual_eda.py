import pandas as pd
import plotly.express as px

from data_preprocessing import load_data


# Load dataset
df = load_data("data/credit_default.xls")


# Convert sex codes to readable labels
df["sex_label"] = df["sex"].map({
    1: "Male",
    2: "Female"
})


# --------------------------------------------------
# 1. Default vs Non-Default
# --------------------------------------------------

default_counts = (
    df["default"]
    .map({
        0: "No Default",
        1: "Default"
    })
    .value_counts()
    .reset_index()
)

default_counts.columns = ["status", "count"]

fig1 = px.bar(
    default_counts,
    x="status",
    y="count",
    title="Default vs Non-Default Applicants",
    text_auto=True
)

fig1.write_html("output/default_vs_nondefault.html")


# --------------------------------------------------
# 2. Default Rate by Sex
# --------------------------------------------------

sex_default = (
    df.groupby("sex_label")["default"]
    .mean()
    .mul(100)
    .reset_index()
)

sex_default.columns = ["sex", "default_rate"]

fig2 = px.bar(
    sex_default,
    x="sex",
    y="default_rate",
    title="Default Rate by Sex",
    text_auto=".2f"
)

fig2.update_yaxes(title="Default Rate (%)")

fig2.write_html("output/default_rate_by_sex.html")


# --------------------------------------------------
# 3. Average Credit Limit by Default Status
# --------------------------------------------------

credit_default = (
    df.groupby("default")["credit_limit"]
    .mean()
    .reset_index()
)

credit_default["status"] = credit_default["default"].map({
    0: "No Default",
    1: "Default"
})

fig3 = px.bar(
    credit_default,
    x="status",
    y="credit_limit",
    title="Average Credit Limit by Default Status",
    text_auto=".2f"
)

fig3.update_yaxes(title="Average Credit Limit")

fig3.write_html("output/average_credit_limit.html")


# --------------------------------------------------
# 4. Default Rate by Education
# --------------------------------------------------

education_default = (
    df.groupby("education")["default"]
    .mean()
    .mul(100)
    .reset_index()
)

education_default["education"] = (
    education_default["education"]
    .astype(str)
)

fig4 = px.bar(
    education_default,
    x="education",
    y="default",
    title="Default Rate by Education",
    text_auto=".2f"
)

fig4.update_yaxes(title="Default Rate (%)")

fig4.write_html("output/default_rate_by_education.html")


# --------------------------------------------------
# 5. Age Distribution by Default Status
# --------------------------------------------------

fig5 = px.box(
    df,
    x="default",
    y="age",
    title="Age Distribution by Default Status",
    labels={
        "default": "Default",
        "age": "Age"
    }
)

fig5.write_html("output/age_distribution.html")


# --------------------------------------------------
# 6. Credit Limit vs Age
# --------------------------------------------------

sample_df = df.sample(
    min(5000, len(df)),
    random_state=42
)

fig6 = px.scatter(
    sample_df,
    x="age",
    y="credit_limit",
    color="default",
    title="Credit Limit vs Age",
    labels={
        "age": "Age",
        "credit_limit": "Credit Limit",
        "default": "Default"
    }
)

fig6.write_html("output/credit_limit_vs_age.html")


print("\n========================================")
print("6 EDA CHARTS CREATED SUCCESSFULLY")
print("========================================")

print("\nCharts saved in the output folder:")
print("1. default_vs_nondefault.html")
print("2. default_rate_by_sex.html")
print("3. average_credit_limit.html")
print("4. default_rate_by_education.html")
print("5. age_distribution.html")
print("6. credit_limit_vs_age.html")