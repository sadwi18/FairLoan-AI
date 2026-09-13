import os
import requests
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FairLoan AI",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================



API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


# ============================================================
# DATASET CODE MAPPINGS
# ============================================================

EDUCATION_OPTIONS = {
    1: "1 — Graduate School",
    2: "2 — University",
    3: "3 — High School",
    4: "4 — Others"
}

MARRIAGE_OPTIONS = {
    1: "1 — Married",
    2: "2 — Single",
    3: "3 — Others"
}

PAY_STATUS_OPTIONS = {
    -2: "-2 — No consumption",
    -1: "-1 — Paid in full",
    0: "0 — Revolving credit / no delay",
    1: "1 — 1 month delay",
    2: "2 — 2 months delay",
    3: "3 — 3 months delay",
    4: "4 — 4 months delay",
    5: "5 — 5 months delay",
    6: "6 — 6 months delay",
    7: "7 — 7 months delay",
    8: "8 — 8 months delay",
    9: "9 — 9+ months delay"
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_risk_info(probability):
    """
    Returns risk category and UI information based on
    project-defined probability thresholds.
    """

    probability_percent = probability * 100

    if probability_percent < 30:
        return {
            "level": "Low Risk",
            "emoji": "🟢",
            "message": "The model estimates a relatively low probability of default."
        }

    elif probability_percent < 60:
        return {
            "level": "Medium Risk",
            "emoji": "🟡",
            "message": "The model estimates a moderate probability of default."
        }

    else:
        return {
            "level": "High Risk",
            "emoji": "🔴",
            "message": "The model estimates a relatively high probability of default."
        }


def explain_feature(feature):
    """
    Converts technical feature names into user-friendly descriptions.
    """

    descriptions = {

        "credit_limit":
            "Credit limit assigned to the applicant.",

        "education":
            "Applicant's education category.",

        "marriage":
            "Applicant's marital status.",

        "age":
            "Applicant's age.",

        "pay_sep":
            "September repayment status.",

        "pay_aug":
            "August repayment status.",

        "pay_jul":
            "July repayment status.",

        "pay_jun":
            "June repayment status.",

        "pay_may":
            "May repayment status.",

        "pay_apr":
            "April repayment status.",

        "bill_sep":
            "September bill amount.",

        "bill_aug":
            "August bill amount.",

        "bill_jul":
            "July bill amount.",

        "bill_jun":
            "June bill amount.",

        "bill_may":
            "May bill amount.",

        "bill_apr":
            "April bill amount.",

        "payment_sep":
            "September previous payment.",

        "payment_aug":
            "August previous payment.",

        "payment_jul":
            "July previous payment.",

        "payment_jun":
            "June previous payment.",

        "payment_may":
            "May previous payment.",

        "payment_apr":
            "April previous payment."
    }

    return descriptions.get(
        feature,
        "Feature used by the machine-learning model."
    )


def format_feature_name(feature):
    """
    Makes technical feature names easier to read.
    """

    replacements = {
        "credit_limit": "Credit Limit",
        "education": "Education",
        "marriage": "Marital Status",
        "age": "Age",

        "pay_sep": "September Repayment",
        "pay_aug": "August Repayment",
        "pay_jul": "July Repayment",
        "pay_jun": "June Repayment",
        "pay_may": "May Repayment",
        "pay_apr": "April Repayment",

        "bill_sep": "September Bill",
        "bill_aug": "August Bill",
        "bill_jul": "July Bill",
        "bill_jun": "June Bill",
        "bill_may": "May Bill",
        "bill_apr": "April Bill",

        "payment_sep": "September Payment",
        "payment_aug": "August Payment",
        "payment_jul": "July Payment",
        "payment_jun": "June Payment",
        "payment_may": "May Payment",
        "payment_apr": "April Payment"
    }

    return replacements.get(
        feature,
        feature.replace("_", " ").title()
    )


# ============================================================
# HEADER
# ============================================================

st.title("💳 FairLoan AI")

st.markdown(
    """
    ### Explainable & Fair Credit Default Risk Prediction System

    Predict credit default risk, understand the factors influencing each
    prediction, and evaluate fairness across demographic groups.
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("👤 Applicant Information")

st.sidebar.info(
    "Enter the applicant's financial information and click "
    "**Predict Default Risk**."
)


# ============================================================
# BASIC INFORMATION
# ============================================================

with st.sidebar.expander(
    "📋 Basic Information",
    expanded=True
):

    credit_limit = st.number_input(
        "Credit Limit",
        min_value=0.0,
        value=50000.0,
        step=5000.0,
        help="Applicant's assigned credit limit."
    )

    education = st.selectbox(
        "Education",
        options=list(EDUCATION_OPTIONS.keys()),
        format_func=lambda x: EDUCATION_OPTIONS[x],
        index=1
    )

    marriage = st.selectbox(
        "Marital Status",
        options=list(MARRIAGE_OPTIONS.keys()),
        format_func=lambda x: MARRIAGE_OPTIONS[x],
        index=0
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25,
        step=1
    )


# ============================================================
# REPAYMENT HISTORY
# ============================================================

with st.sidebar.expander(
    "📅 Repayment History",
    expanded=False
):

    st.caption(
        "Positive values represent payment delays. "
        "Larger values indicate longer delays."
    )

    pay_sep = st.selectbox(
        "September",
        options=list(PAY_STATUS_OPTIONS.keys()),
        format_func=lambda x: PAY_STATUS_OPTIONS[x],
        index=2
    )

    pay_aug = st.selectbox(
        "August",
        options=list(PAY_STATUS_OPTIONS.keys()),
        format_func=lambda x: PAY_STATUS_OPTIONS[x],
        index=2
    )

    pay_jul = st.selectbox(
        "July",
        options=list(PAY_STATUS_OPTIONS.keys()),
        format_func=lambda x: PAY_STATUS_OPTIONS[x],
        index=2
    )

    pay_jun = st.selectbox(
        "June",
        options=list(PAY_STATUS_OPTIONS.keys()),
        format_func=lambda x: PAY_STATUS_OPTIONS[x],
        index=2
    )

    pay_may = st.selectbox(
        "May",
        options=list(PAY_STATUS_OPTIONS.keys()),
        format_func=lambda x: PAY_STATUS_OPTIONS[x],
        index=2
    )

    pay_apr = st.selectbox(
        "April",
        options=list(PAY_STATUS_OPTIONS.keys()),
        format_func=lambda x: PAY_STATUS_OPTIONS[x],
        index=2
    )


# ============================================================
# BILL AMOUNTS
# ============================================================

with st.sidebar.expander(
    "💰 Bill Amounts",
    expanded=False
):

    bill_sep = st.number_input(
        "September Bill",
        min_value=0.0,
        value=40000.0,
        step=1000.0
    )

    bill_aug = st.number_input(
        "August Bill",
        min_value=0.0,
        value=38000.0,
        step=1000.0
    )

    bill_jul = st.number_input(
        "July Bill",
        min_value=0.0,
        value=35000.0,
        step=1000.0
    )

    bill_jun = st.number_input(
        "June Bill",
        min_value=0.0,
        value=32000.0,
        step=1000.0
    )

    bill_may = st.number_input(
        "May Bill",
        min_value=0.0,
        value=30000.0,
        step=1000.0
    )

    bill_apr = st.number_input(
        "April Bill",
        min_value=0.0,
        value=28000.0,
        step=1000.0
    )


# ============================================================
# PREVIOUS PAYMENTS
# ============================================================

with st.sidebar.expander(
    "💵 Previous Payments",
    expanded=False
):

    payment_sep = st.number_input(
        "September Payment",
        min_value=0.0,
        value=2000.0,
        step=500.0
    )

    payment_aug = st.number_input(
        "August Payment",
        min_value=0.0,
        value=2500.0,
        step=500.0
    )

    payment_jul = st.number_input(
        "July Payment",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

    payment_jun = st.number_input(
        "June Payment",
        min_value=0.0,
        value=2500.0,
        step=500.0
    )

    payment_may = st.number_input(
        "May Payment",
        min_value=0.0,
        value=3000.0,
        step=500.0
    )

    payment_apr = st.number_input(
        "April Payment",
        min_value=0.0,
        value=2500.0,
        step=500.0
    )


# ============================================================
# APPLICANT DATA
# ============================================================

applicant = {

    "credit_limit": credit_limit,
    "education": education,
    "marriage": marriage,
    "age": age,

    "pay_sep": pay_sep,
    "pay_aug": pay_aug,
    "pay_jul": pay_jul,
    "pay_jun": pay_jun,
    "pay_may": pay_may,
    "pay_apr": pay_apr,

    "bill_sep": bill_sep,
    "bill_aug": bill_aug,
    "bill_jul": bill_jul,
    "bill_jun": bill_jun,
    "bill_may": bill_may,
    "bill_apr": bill_apr,

    "payment_sep": payment_sep,
    "payment_aug": payment_aug,
    "payment_jul": payment_jul,
    "payment_jun": payment_jun,
    "payment_may": payment_may,
    "payment_apr": payment_apr
}


# ============================================================
# RESET RESULTS WHEN INPUT CHANGES
# ============================================================

current_signature = tuple(
    applicant.values()
)

if "last_applicant" not in st.session_state:

    st.session_state.last_applicant = (
        current_signature
    )

if st.session_state.last_applicant != current_signature:

    st.session_state.prediction = None
    st.session_state.explanation = None

    st.session_state.last_applicant = (
        current_signature
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.sidebar.button(
    "🔍 Predict Default Risk",
    type="primary",
    use_container_width=True
):

    try:

        with st.spinner(
            "Analyzing applicant and generating explanation..."
        ):

            # --------------------------------------------
            # PREDICTION API
            # --------------------------------------------

            prediction_response = requests.post(
                f"{API_URL}/predict",
                json=applicant,
                timeout=30
            )

            prediction_response.raise_for_status()

            st.session_state.prediction = (
                prediction_response.json()
            )


            # --------------------------------------------
            # EXPLANATION API
            # --------------------------------------------

            explanation_response = requests.post(
                f"{API_URL}/explain",
                json=applicant,
                timeout=30
            )

            explanation_response.raise_for_status()

            st.session_state.explanation = (
                explanation_response.json()
            )


        st.success(
            "Prediction completed successfully."
        )

    except requests.exceptions.ConnectionError:

        st.error(
            """
            ❌ **FastAPI connection failed.**

            Make sure the backend is running:

            `uvicorn main:app --reload`
            """
        )

    except requests.exceptions.HTTPError as e:

        st.error(
            f"❌ FastAPI returned an error: {e}"
        )

    except Exception as e:

        st.error(
            f"❌ Something went wrong: {e}"
        )


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🎯 Prediction",
        "🔎 Explainability",
        "⚖️ Fairness"
    ]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab1:

    st.header("🎯 Default Risk Prediction")

    prediction = st.session_state.get(
        "prediction"
    )


    # ========================================================
    # NO PREDICTION YET
    # ========================================================

    if prediction is None:

        st.info(
            "👈 Enter applicant information in the sidebar "
            "and click **Predict Default Risk**."
        )

        st.markdown(
            """
            ## How FairLoan AI works

            **1. Applicant Data**

            Financial and repayment information is provided.

            **2. Machine Learning Prediction**

            A Random Forest model estimates the probability
            of credit default.

            **3. Explainability**

            SHAP identifies the features that contributed most
            to the prediction.

            **4. Fairness**

            Fairness metrics evaluate whether model outcomes
            differ across demographic groups.
            """
        )

    else:

        probability = prediction.get(
            "default_probability",
            prediction.get(
                "probability",
                0
            )
        )

        probability_percent = prediction.get(
            "default_probability_percent",
            probability * 100
        )

        prediction_label = prediction.get(
            "prediction",
            "Unknown"
        )

        risk_level = prediction.get(
            "risk_level",
            get_risk_info(probability)["level"]
        )

        risk_info = get_risk_info(
            probability
        )


        # ====================================================
        # MAIN RESULT
        # ====================================================

        st.subheader("Model Result")


        result_col1, result_col2 = st.columns(
            [1, 2]
        )


        # ----------------------------------------------------
        # RISK LEVEL
        # ----------------------------------------------------

        with result_col1:

            st.markdown(
                f"""
                <div style="
                    padding: 25px;
                    border-radius: 15px;
                    border: 1px solid #444;
                    text-align: center;
                ">

                <div style="font-size: 48px;">
                    {risk_info["emoji"]}
                </div>

                <h2>{risk_level}</h2>

                <p>{risk_info["message"]}</p>

                </div>
                """,
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # PROBABILITY
        # ----------------------------------------------------

        with result_col2:

            st.metric(
                "Probability of Default",
                f"{probability_percent:.2f}%"
            )

            st.progress(
                min(
                    max(
                        probability_percent / 100,
                        0
                    ),
                    1
                )
            )

            st.caption(
                f"Model prediction: **{prediction_label}**"
            )


        st.divider()


        # ====================================================
        # RISK INTERPRETATION
        # ====================================================

        if probability_percent < 30:

            st.success(
                f"""
                🟢 **Low Risk**

                The model estimates a
                **{probability_percent:.2f}% probability**
                of default.
                """
            )

        elif probability_percent < 60:

            st.warning(
                f"""
                🟡 **Medium Risk**

                The model estimates a
                **{probability_percent:.2f}% probability**
                of default.
                """
            )

        else:

            st.error(
                f"""
                🔴 **High Risk**

                The model estimates a
                **{probability_percent:.2f}% probability**
                of default.
                """
            )


        st.caption(
            "Risk categories use project-defined probability "
            "thresholds and are not validated lending-policy "
            "or regulatory thresholds."
        )


        # ====================================================
        # APPLICANT SUMMARY
        # ====================================================

        st.subheader(
            "👤 Applicant Summary"
        )

        summary1, summary2, summary3, summary4 = (
            st.columns(4)
        )


        with summary1:

            st.metric(
                "Credit Limit",
                f"₹{credit_limit:,.0f}"
            )


        with summary2:

            st.metric(
                "Age",
                f"{age} years"
            )


        with summary3:

            st.write("**Education**")

            st.write(
                EDUCATION_OPTIONS[education]
            )


        with summary4:

            st.write("**Marital Status**")

            st.write(
                MARRIAGE_OPTIONS[marriage]
            )


        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        st.divider()

        st.subheader(
            "🤖 Model Information"
        )

        model1, model2, model3 = st.columns(3)

        with model1:

            st.write("**Algorithm**")
            st.write("Random Forest")

        with model2:

            st.write("**Explainability**")
            st.write("SHAP")

        with model3:

            st.write("**Fairness Framework**")
            st.write("Fairlearn")


# ============================================================
# TAB 2 — EXPLAINABILITY
# ============================================================

with tab2:

    st.header(
        "🔎 Why Did the Model Make This Prediction?"
    )

    explanation = st.session_state.get(
        "explanation"
    )

    prediction = st.session_state.get(
        "prediction"
    )


    if explanation is None:

        st.info(
            "Run a prediction first to see the "
            "SHAP-based explanation."
        )

    else:

        st.markdown(
            """
            ### Understanding SHAP

            SHAP values show how individual features
            contributed to the model's prediction.

            - **Positive SHAP value** → pushes the prediction
              toward higher default risk.
            - **Negative SHAP value** → pushes the prediction
              toward lower default risk.
            """
        )


        # ====================================================
        # TOP FACTORS
        # ====================================================

        top_factors = explanation.get(
            "top_factors",
            []
        )


        if top_factors:

            factor_data = []


            for factor in top_factors:

                feature = factor.get(
                    "feature",
                    ""
                )

                shap_value = factor.get(
                    "shap_value",
                    factor.get(
                        "value",
                        0
                    )
                )

                factor_data.append(
                    {
                        "Feature": format_feature_name(
                            feature
                        ),

                        "Technical Feature": feature,

                        "SHAP Value": shap_value,

                        "Description": explain_feature(
                            feature
                        )
                    }
                )


            factor_df = pd.DataFrame(
                factor_data
            )


            # =================================================
            # SHAP CHART
            # =================================================

            st.subheader(
                "📊 Top Contributing Factors"
            )

            chart_df = factor_df[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ].set_index(
                "Feature"
            )


            st.bar_chart(
                chart_df[
                    "SHAP Value"
                ]
            )


            st.caption(
                "Values above zero increase the model's "
                "default prediction; values below zero "
                "decrease it."
            )


            # =================================================
            # POSITIVE / NEGATIVE FACTORS
            # =================================================

            positive_factors = factor_df[
                factor_df["SHAP Value"] > 0
            ]

            negative_factors = factor_df[
                factor_df["SHAP Value"] < 0
            ]


            factor_col1, factor_col2 = st.columns(2)


            # -------------------------------------------------
            # RISK-INCREASING
            # -------------------------------------------------

            with factor_col1:

                st.subheader(
                    "⬆️ Factors Increasing Risk"
                )

                if len(positive_factors) == 0:

                    st.success(
                        "No positive SHAP contributions "
                        "were found among the displayed factors."
                    )

                else:

                    for _, row in positive_factors.iterrows():

                        st.markdown(
                            f"""
                            **{row["Feature"]}**

                            SHAP: `{row["SHAP Value"]:.5f}`

                            {row["Description"]}
                            """
                        )

                        st.divider()


            # -------------------------------------------------
            # RISK-REDUCING
            # -------------------------------------------------

            with factor_col2:

                st.subheader(
                    "⬇️ Factors Reducing Risk"
                )

                if len(negative_factors) == 0:

                    st.info(
                        "No negative SHAP contributions "
                        "were found among the displayed factors."
                    )

                else:

                    for _, row in negative_factors.iterrows():

                        st.markdown(
                            f"""
                            **{row["Feature"]}**

                            SHAP: `{row["SHAP Value"]:.5f}`

                            {row["Description"]}
                            """
                        )

                        st.divider()


            # =================================================
            # DETAILED TABLE
            # =================================================

            st.subheader(
                "📋 Detailed Explanation"
            )

            display_df = factor_df[
                [
                    "Feature",
                    "SHAP Value",
                    "Description"
                ]
            ].copy()


            display_df["SHAP Value"] = (
                display_df["SHAP Value"]
                .map(
                    lambda x: f"{x:.5f}"
                )
            )


            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


        else:

            st.warning(
                "No SHAP explanation was returned by the API."
            )


        # ====================================================
        # EXPLANATION NOTE
        # ====================================================

        st.info(
            """
            ⚠️ **Important:** SHAP explains the relationship
            between the input features and the model's output.
            It does not establish that a feature caused the
            applicant's financial outcome.
            """
        )


# ============================================================
# TAB 3 — FAIRNESS
# ============================================================

with tab3:

    st.header(
        "⚖️ Fairness Audit"
    )

    st.markdown(
        """
        FairLoan AI evaluates whether model outcomes differ
        between the two sex groups represented in the dataset.
        """
    )


    try:

        fairness_response = requests.get(
            f"{API_URL}/fairness",
            timeout=30
        )

        fairness_response.raise_for_status()

        fairness_data = (
            fairness_response.json()
        )


        # ====================================================
        # EXTRACT DATA
        # ====================================================

        before_data = fairness_data.get(
            "before_mitigation",
            {}
        )

        after_data = fairness_data.get(
            "after_mitigation",
            {}
        )

        performance_data = fairness_data.get(
            "performance",
            {}
        )


        # ====================================================
        # FAIRNESS METRICS
        # ====================================================

        st.subheader(
            "Fairness Metrics"
        )


        fairness_table = pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "Recall",
                    "F1 Score",
                    "Demographic Parity Difference",
                    "Equalized Odds Difference"
                ],

                "Before Mitigation": [

                    performance_data.get(
                        "accuracy_before",
                        0
                    ),

                    performance_data.get(
                        "recall_before",
                        0
                    ),

                    performance_data.get(
                        "f1_before",
                        0
                    ),

                    before_data.get(
                        "demographic_parity_difference",
                        0
                    ),

                    before_data.get(
                        "equalized_odds_difference",
                        0
                    )
                ],

                "After Mitigation": [

                    performance_data.get(
                        "accuracy_after",
                        0
                    ),

                    performance_data.get(
                        "recall_after",
                        0
                    ),

                    performance_data.get(
                        "f1_after",
                        0
                    ),

                    after_data.get(
                        "demographic_parity_difference",
                        0
                    ),

                    after_data.get(
                        "equalized_odds_difference",
                        0
                    )
                ]
            }
        )


        formatted_table = (
            fairness_table.copy()
        )


        for column in [
            "Before Mitigation",
            "After Mitigation"
        ]:

            formatted_table[column] = (
                formatted_table[column]
                .map(
                    lambda x: f"{x:.4f}"
                )
            )


        st.dataframe(
            formatted_table,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # PERFORMANCE
        # ====================================================

        st.subheader(
            "📊 Performance Comparison"
        )


        pc1, pc2, pc3 = st.columns(3)


        accuracy_before = performance_data.get(
            "accuracy_before",
            0
        )

        accuracy_after = performance_data.get(
            "accuracy_after",
            0
        )

        recall_before = performance_data.get(
            "recall_before",
            0
        )

        recall_after = performance_data.get(
            "recall_after",
            0
        )

        f1_before = performance_data.get(
            "f1_before",
            0
        )

        f1_after = performance_data.get(
            "f1_after",
            0
        )


        with pc1:

            st.metric(
                "Accuracy",
                f"{accuracy_before:.2%}",
                delta=f"{accuracy_after - accuracy_before:.2%}"
            )


        with pc2:

            st.metric(
                "Recall",
                f"{recall_before:.2%}",
                delta=f"{recall_after - recall_before:.2%}"
            )


        with pc3:

            st.metric(
                "F1 Score",
                f"{f1_before:.2%}",
                delta=f"{f1_after - f1_before:.2%}"
            )


        # ====================================================
        # FAIRNESS COMPARISON
        # ====================================================

        st.subheader(
            "⚖️ Fairness Comparison"
        )


        dp_before = before_data.get(
            "demographic_parity_difference",
            0
        )

        dp_after = after_data.get(
            "demographic_parity_difference",
            0
        )

        eo_before = before_data.get(
            "equalized_odds_difference",
            0
        )

        eo_after = after_data.get(
            "equalized_odds_difference",
            0
        )


        fc1, fc2 = st.columns(2)


        with fc1:

            st.metric(
                "Demographic Parity Difference",
                f"{dp_after:.4f}",
                delta=f"{dp_after - dp_before:.4f}"
            )


        with fc2:

            st.metric(
                "Equalized Odds Difference",
                f"{eo_after:.4f}",
                delta=f"{eo_after - eo_before:.4f}"
            )


        # ====================================================
        # INTERPRETATION
        # ====================================================

        st.subheader(
            "📌 Interpretation"
        )


        st.markdown(
            f"""
            **Before mitigation:**

            - Demographic Parity Difference:
              **{dp_before:.4f}**
            - Equalized Odds Difference:
              **{eo_before:.4f}**
            - Accuracy:
              **{accuracy_before:.2%}**
            - Recall:
              **{recall_before:.2%}**
            - F1 Score:
              **{f1_before:.4f}**

            **After mitigation:**

            - Demographic Parity Difference:
              **{dp_after:.4f}**
            - Equalized Odds Difference:
              **{eo_after:.4f}**
            - Accuracy:
              **{accuracy_after:.2%}**
            - Recall:
              **{recall_after:.2%}**
            - F1 Score:
              **{f1_after:.4f}**
            """
        )


        st.info(
            """
            The fairness mitigation experiment reduced
            demographic parity disparity from **0.0331**
            to **0.0075**, but reduced recall and increased
            equalized-odds disparity.

            This demonstrates the trade-off between fairness
            objectives and predictive performance.
            """
        )


        st.warning(
            """
            A model should not be considered completely fair
            based on a single fairness metric. Fairness must be
            evaluated in the context of the application's
            objectives, population, and deployment setting.
            """
        )


    except requests.exceptions.ConnectionError:

        st.error(
            """
            ❌ Could not connect to FastAPI.

            Start the backend using:

            `uvicorn main:app --reload`
            """
        )

    except Exception as e:

        st.error(
            f"❌ Could not load fairness results: {e}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FairLoan AI | Explainable & Fair Credit Default Risk Prediction"
)

st.caption(
    "UCI Default of Credit Card Clients | "
    "Random Forest | SHAP | Fairlearn"
)