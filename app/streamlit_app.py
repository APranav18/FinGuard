import json
import pickle
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DATA_PATH = PROJECT_ROOT / "data" / "Churn_Modelling.csv"


def apply_black_red_theme() -> None:
    """Inject a black and red visual theme layer for cards and controls."""
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top left, #1a0004 0%, #0b0b0d 45%, #070709 100%);
        }
        div[data-testid="stMetric"] {
            background: rgba(193, 18, 31, 0.12);
            border: 1px solid rgba(193, 18, 31, 0.45);
            border-radius: 12px;
            padding: 10px;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div {
            background-color: #131317 !important;
            border-color: #7f1d1d !important;
        }
        div[data-testid="stSlider"] [role="slider"] {
            background-color: #c1121f !important;
        }
        h1, h2, h3 {
            color: #ffccd2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model file not found. Run training first: python main.py")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_metadata():
    if not METADATA_PATH.exists():
        return {
            "categorical_options": {
                "Gender": ["Male", "Female"],
                "Geography": ["France", "Germany", "Spain"],
            },
            "best_model_name": "Unknown",
            "prediction_threshold": 0.5,
            "is_tuned": False,
        }
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_reference_profile() -> pd.Series | None:
    """Load average numeric customer profile for visual comparison."""
    if not DATA_PATH.exists():
        return None

    df = pd.read_csv(DATA_PATH)
    numeric_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "EstimatedSalary",
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]
    if not available_cols:
        return None

    return df[available_cols].mean(numeric_only=True)


@st.cache_data
def load_customer_records() -> pd.DataFrame | None:
    """Load full customer records for account/name lookup."""
    if not DATA_PATH.exists():
        return None
    return pd.read_csv(DATA_PATH)


def clamp(value: float, min_value: int, max_value: int) -> int:
    return int(max(min_value, min(max_value, round(value))))


def infer_behavior_from_customer_row(row: pd.Series) -> dict:
    """Infer digital interaction behavior from static customer profile fields."""
    is_active = int(row.get("IsActiveMember", 0))
    num_products = int(row.get("NumOfProducts", 1))
    balance = float(row.get("Balance", 0.0))
    age = int(row.get("Age", 40))
    has_card = int(row.get("HasCrCard", 1))

    logins_per_week = clamp(3 + (2 * num_products) + (2 if is_active else -1) + (1 if has_card else 0), 0, 40)
    online_txn_per_month = clamp((logins_per_week * 2.2) + (3 if is_active else -2), 0, 120)
    avg_session_minutes = clamp(4 + (2 if is_active else 0) + (1 if num_products >= 2 else 0), 1, 60)
    feature_adoption = clamp(1 + (1 if num_products >= 2 else 0) + (1 if is_active else 0), 1, 5)
    feature_research_per_month = clamp((feature_adoption * 1.2) + (1 if is_active else 0), 0, 30)
    support_tickets = clamp((2 if not is_active else 1) + (1 if age >= 60 else 0), 0, 20)
    failed_login_events = clamp((1 if age >= 60 else 0) + (1 if not is_active else 0), 0, 30)

    # High-balance users tend to engage more in financial tracking journeys.
    if balance >= 100000:
        online_txn_per_month = clamp(online_txn_per_month + 4, 0, 120)
        feature_research_per_month = clamp(feature_research_per_month + 1, 0, 30)

    return {
        "logins_per_week": logins_per_week,
        "online_txn_per_month": online_txn_per_month,
        "avg_session_minutes": avg_session_minutes,
        "feature_adoption": feature_adoption,
        "feature_research_per_month": feature_research_per_month,
        "support_tickets": support_tickets,
        "failed_login_events": failed_login_events,
    }


def initialize_input_state(metadata):
    """Initialize widget state once so lookup can prefill values."""
    defaults = {
        "credit_score": 650,
        "age": 40,
        "tenure": 5,
        "balance": 60000.0,
        "num_products": 2,
        "has_card": 1,
        "is_active": 1,
        "estimated_salary": 50000.0,
        "gender": metadata.get("categorical_options", {}).get("Gender", ["Male", "Female"])[0],
        "geography": metadata.get("categorical_options", {}).get("Geography", ["France", "Germany", "Spain"])[0],
        "logins_per_week": 8,
        "online_txn_per_month": 20,
        "avg_session_minutes": 8,
        "feature_adoption": 3,
        "feature_research_per_month": 4,
        "support_tickets": 1,
        "failed_login_events": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def populate_state_from_customer_row(row: pd.Series):
    """Map a dataset row into profile and behavior widget state."""
    st.session_state.credit_score = int(row.get("CreditScore", 650))
    st.session_state.age = int(row.get("Age", 40))
    st.session_state.tenure = int(row.get("Tenure", 5))
    st.session_state.balance = float(row.get("Balance", 60000.0))
    st.session_state.num_products = int(row.get("NumOfProducts", 2))
    st.session_state.has_card = int(row.get("HasCrCard", 1))
    st.session_state.is_active = int(row.get("IsActiveMember", 1))
    st.session_state.estimated_salary = float(row.get("EstimatedSalary", 50000.0))
    st.session_state.gender = str(row.get("Gender", "Male"))
    st.session_state.geography = str(row.get("Geography", "France"))

    behavior = infer_behavior_from_customer_row(row)
    st.session_state.logins_per_week = behavior["logins_per_week"]
    st.session_state.online_txn_per_month = behavior["online_txn_per_month"]
    st.session_state.avg_session_minutes = behavior["avg_session_minutes"]
    st.session_state.feature_adoption = behavior["feature_adoption"]
    st.session_state.feature_research_per_month = behavior["feature_research_per_month"]
    st.session_state.support_tickets = behavior["support_tickets"]
    st.session_state.failed_login_events = behavior["failed_login_events"]


def render_customer_lookup(customer_df: pd.DataFrame | None):
    """Lookup customer by account number or name and prefill dashboard inputs."""
    st.subheader("Customer Lookup")
    if customer_df is None:
        st.info("Customer lookup unavailable because the dataset file is missing.")
        return

    lookup_col, match_col = st.columns([1.2, 1])
    with lookup_col:
        lookup_text = st.text_input(
            "Enter Account Number (CustomerId) or Customer Name (Surname)",
            placeholder="e.g. 15634602 or Hargrave",
        ).strip()

    matches = customer_df.iloc[0:0]
    if lookup_text:
        if lookup_text.isdigit():
            matches = customer_df[customer_df["CustomerId"].astype(str) == lookup_text]
        if matches.empty:
            matches = customer_df[customer_df["Surname"].astype(str).str.contains(lookup_text, case=False, na=False)]

    selected_row = None
    with match_col:
        if not matches.empty:
            options = [
                f"{r.CustomerId} | {r.Surname} | {r.Geography}"
                for _, r in matches.head(25).iterrows()
            ]
            picked = st.selectbox("Matching Customer", options=options)
            picked_id = picked.split("|", maxsplit=1)[0].strip()
            selected_row = matches[matches["CustomerId"].astype(str) == picked_id].iloc[0]

    if st.button("Auto-Fill From Customer", type="primary"):
        if selected_row is not None:
            populate_state_from_customer_row(selected_row)
            st.success("Customer profile and behavior fields auto-filled.")
        elif lookup_text:
            st.warning("No matching customer found. Try another account number or name.")
        else:
            st.info("Enter an account number or customer name first.")


def build_input_frame(metadata):
    st.subheader("Customer Inputs")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.slider("Credit Score", min_value=300, max_value=900, step=1, key="credit_score")
        age = st.slider("Age", min_value=18, max_value=100, step=1, key="age")
        tenure = st.slider("Tenure (years)", min_value=0, max_value=10, step=1, key="tenure")
        balance = st.slider("Balance", min_value=0.0, max_value=250000.0, step=500.0, key="balance")
        num_products = st.slider("Number of Products", min_value=1, max_value=4, step=1, key="num_products")

    with col2:
        has_card = st.selectbox(
            "Has Credit Card",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="has_card",
        )
        is_active = st.selectbox(
            "Is Active Member",
            options=[1, 0],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="is_active",
        )
        estimated_salary = st.slider(
            "Estimated Salary", min_value=0.0, max_value=250000.0, step=500.0, key="estimated_salary"
        )
        gender_options = metadata.get("categorical_options", {}).get("Gender", ["Male", "Female"])
        geography_options = metadata.get("categorical_options", {}).get("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", options=gender_options, key="gender")
        geography = st.selectbox("Geography", options=geography_options, key="geography")

    input_data = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": has_card,
                "IsActiveMember": is_active,
                "EstimatedSalary": estimated_salary,
                "Gender": gender,
                "Geography": geography,
            }
        ]
    )
    return input_data


def build_interaction_inputs():
    """Collect digital interaction behavior signals from the user."""
    st.subheader("Digital Interaction Behavior")

    col1, col2 = st.columns(2)
    with col1:
        logins_per_week = st.slider(
            "Online App Logins per Week",
            min_value=0,
            max_value=40,
            step=1,
            key="logins_per_week",
        )
        online_txn_per_month = st.slider(
            "Online Transactions per Month",
            min_value=0,
            max_value=120,
            step=1,
            key="online_txn_per_month",
        )
        avg_session_minutes = st.slider(
            "Average Session Duration (minutes)",
            min_value=1,
            max_value=60,
            step=1,
            key="avg_session_minutes",
        )

    with col2:
        feature_adoption = st.slider(
            "Digital Feature Adoption (1=Low, 5=High)",
            min_value=1,
            max_value=5,
            step=1,
            key="feature_adoption",
        )
        feature_research_per_month = st.slider(
            "Researches New Features per Month",
            min_value=0,
            max_value=30,
            step=1,
            key="feature_research_per_month",
        )
        support_tickets = st.slider(
            "Digital Support Tickets (last 3 months)",
            min_value=0,
            max_value=20,
            step=1,
            key="support_tickets",
        )
        failed_login_events = st.slider(
            "Failed Login Events (last 30 days)",
            min_value=0,
            max_value=30,
            step=1,
            key="failed_login_events",
        )

    return {
        "logins_per_week": logins_per_week,
        "online_txn_per_month": online_txn_per_month,
        "avg_session_minutes": avg_session_minutes,
        "feature_adoption": feature_adoption,
        "feature_research_per_month": feature_research_per_month,
        "support_tickets": support_tickets,
        "failed_login_events": failed_login_events,
    }


def get_behavior_weight_mode() -> tuple[str, float]:
    """Return selected behavior weighting mode and its multiplier."""
    mode = st.radio(
        "Behavior Weighting Mode",
        options=["Conservative", "Balanced", "Aggressive"],
        index=1,
        horizontal=True,
        help="Controls how strongly digital interaction signals influence churn probability.",
    )

    multiplier_map = {
        "Conservative": 0.15,
        "Balanced": 0.30,
        "Aggressive": 0.45,
    }
    return mode, multiplier_map[mode]


def apply_behavior_adjustment(base_probability: float, behavior: dict, behavior_weight: float) -> tuple[float, float]:
    """Adjust base churn probability using interaction behavior heuristics.

    Returns adjusted probability and behavior risk score in [-1, 1].
    """
    score = 0.0

    if behavior["logins_per_week"] <= 2:
        score += 0.18
    elif behavior["logins_per_week"] >= 10:
        score -= 0.12

    if behavior["online_txn_per_month"] <= 5:
        score += 0.16
    elif behavior["online_txn_per_month"] >= 25:
        score -= 0.10

    if behavior["avg_session_minutes"] <= 3:
        score += 0.08
    elif behavior["avg_session_minutes"] >= 10:
        score -= 0.06

    if behavior["feature_adoption"] <= 2:
        score += 0.14
    elif behavior["feature_adoption"] >= 4:
        score -= 0.10

    if behavior["feature_research_per_month"] <= 1:
        score += 0.10
    elif behavior["feature_research_per_month"] >= 6:
        score -= 0.07

    if behavior["support_tickets"] >= 5:
        score += 0.12
    elif behavior["support_tickets"] == 0:
        score -= 0.03

    if behavior["failed_login_events"] >= 5:
        score += 0.09

    score = max(-1.0, min(1.0, score))
    adjusted = base_probability + (behavior_weight * score)
    adjusted = max(0.01, min(0.99, adjusted))
    return adjusted, score


def build_probability_chart(churn_probability: float, stay_probability: float):
    chart_df = pd.DataFrame(
        {
            "Outcome": ["Stay", "Churn"],
            "Probability": [stay_probability, churn_probability],
        }
    )
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Outcome:N", title="Prediction Outcome"),
            y=alt.Y("Probability:Q", title="Probability", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("Outcome:N", scale=alt.Scale(domain=["Stay", "Churn"], range=["#2E8B57", "#C0392B"])),
            tooltip=[alt.Tooltip("Outcome:N"), alt.Tooltip("Probability:Q", format=".2%")],
        )
        .properties(height=260)
    )


def build_interaction_frequency_chart(behavior: dict):
    freq_df = pd.DataFrame(
        {
            "Metric": [
                "App Logins / Week",
                "Online Txns / Month",
                "Feature Research / Month",
            ],
            "Frequency": [
                behavior["logins_per_week"],
                behavior["online_txn_per_month"],
                behavior["feature_research_per_month"],
            ],
        }
    )
    return (
        alt.Chart(freq_df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Metric:N", title="Engagement Metric"),
            y=alt.Y("Frequency:Q", title="Frequency"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    domain=["App Logins / Week", "Online Txns / Month", "Feature Research / Month"],
                    range=["#1F77B4", "#2E8B57", "#8E44AD"],
                ),
                legend=None,
            ),
            tooltip=["Metric:N", "Frequency:Q"],
        )
        .properties(height=260)
    )


def build_feature_comparison_chart(input_df: pd.DataFrame, reference_profile: pd.Series | None):
    if reference_profile is None:
        return None

    compare_cols = ["CreditScore", "Age", "Balance", "NumOfProducts", "EstimatedSalary"]
    rows = []
    for feature in compare_cols:
        if feature in input_df.columns and feature in reference_profile.index:
            rows.append({"Feature": feature, "Profile": "Current Input", "Value": float(input_df.iloc[0][feature])})
            rows.append({"Feature": feature, "Profile": "Dataset Average", "Value": float(reference_profile[feature])})

    if not rows:
        return None

    feature_df = pd.DataFrame(rows)
    return (
        alt.Chart(feature_df)
        .mark_bar()
        .encode(
            x=alt.X("Feature:N", sort=compare_cols),
            y=alt.Y("Value:Q"),
            color=alt.Color("Profile:N", scale=alt.Scale(domain=["Current Input", "Dataset Average"], range=["#1F77B4", "#FF7F0E"])),
            xOffset="Profile:N",
            tooltip=["Feature:N", "Profile:N", alt.Tooltip("Value:Q", format=",.2f")],
        )
        .properties(height=280)
    )


def derive_customer_demerits_and_actions(
    input_df: pd.DataFrame,
    churn_probability: float,
    behavior: dict,
    reference_profile: pd.Series | None,
):
    """Generate profile-based churn pain points and actionable bank recommendations."""
    row = input_df.iloc[0]
    demerits = []
    actions = []

    if churn_probability >= 0.6:
        demerits.append("High overall churn risk based on current profile.")
        actions.append("Trigger an urgent retention workflow with a relationship manager call.")

    if behavior["logins_per_week"] <= 2:
        demerits.append("Very low weekly app engagement with digital channels.")
        actions.append("Start a digital engagement campaign with personalized reminders and rewards.")

    if behavior["online_txn_per_month"] <= 5:
        demerits.append("Low monthly digital transaction frequency indicates weak platform stickiness.")
        actions.append("Offer cashback for bill pay, UPI, and online transfers to increase usage habits.")

    if behavior["feature_research_per_month"] <= 1:
        demerits.append("Rarely explores new banking features and updates.")
        actions.append("Send personalized feature discovery nudges and short in-app demos.")

    if behavior["feature_adoption"] <= 2:
        demerits.append("Customer is under-using key online banking features.")
        actions.append("Run guided feature tours and offer a one-click setup for auto-pay and smart alerts.")

    if behavior["support_tickets"] >= 5 or behavior["failed_login_events"] >= 5:
        demerits.append("Repeated service friction in digital journeys (support/login issues).")
        actions.append("Provide proactive technical support and fast-track issue resolution.")

    if float(row["IsActiveMember"]) == 0:
        demerits.append("Low engagement: customer is not an active member.")
        actions.append("Offer app onboarding support and rewards for monthly digital activity.")

    if float(row["NumOfProducts"]) <= 1:
        demerits.append("Limited product adoption may reduce switching cost.")
        actions.append("Bundle a second product with fee waivers and loyalty points.")

    if float(row["CreditScore"]) < 550:
        demerits.append("Lower credit score can create dissatisfaction with lending options.")
        actions.append("Provide tailored credit-building plans and transparent loan eligibility guidance.")

    if float(row["Balance"]) > 120000 and float(row["IsActiveMember"]) == 0:
        demerits.append("High balance but low engagement suggests value leakage risk.")
        actions.append("Assign a premium advisor and offer personalized wealth-management benefits.")

    if reference_profile is not None and "EstimatedSalary" in reference_profile.index:
        if float(row["EstimatedSalary"]) > float(reference_profile["EstimatedSalary"]) and float(row["NumOfProducts"]) <= 1:
            demerits.append("Higher-income customer with shallow product depth may find limited value.")
            actions.append("Recommend premium account features and personalized investment products.")

    if float(row["Age"]) >= 55 and float(row["NumOfProducts"]) <= 1:
        demerits.append("Mature customer segment may expect more advisory-led experiences.")
        actions.append("Provide proactive financial planning check-ins and retirement-focused offers.")

    if not demerits:
        demerits.append("No major churn pain points detected from current inputs.")
        actions.append("Maintain engagement through loyalty rewards and periodic service quality follow-ups.")

    return demerits[:6], actions[:6]


def main():
    st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="centered")
    apply_black_red_theme()
    st.title("Customer Churn Prediction in Banking")
    st.write("Interactive dashboard with live churn prediction and visual analytics.")

    metadata = load_metadata()
    model = load_model()
    reference_profile = load_reference_profile()
    customer_df = load_customer_records()
    prediction_threshold = float(metadata.get("prediction_threshold", 0.5))

    initialize_input_state(metadata)

    st.caption(f"Best trained model: {metadata.get('best_model_name', 'Unknown')}")
    st.caption(
        f"Threshold: {prediction_threshold:.3f} | Tuned: {'Yes' if metadata.get('is_tuned', False) else 'No'}"
    )

    render_customer_lookup(customer_df)

    input_df = build_input_frame(metadata)
    behavior = build_interaction_inputs()
    mode_label, behavior_weight = get_behavior_weight_mode()

    # Real-time prediction: updates on every input change.
    base_churn_probability = float(model.predict_proba(input_df)[0][1])
    adjusted_churn_probability, behavior_score = apply_behavior_adjustment(
        base_churn_probability,
        behavior,
        behavior_weight,
    )
    stay_probability = 1.0 - adjusted_churn_probability
    prediction = 1 if adjusted_churn_probability >= prediction_threshold else 0

    result_col, metric_col = st.columns([1.2, 1])
    with result_col:
        if prediction == 1:
            st.error("Prediction: CHURN")
        else:
            st.success("Prediction: NOT CHURN")
    with metric_col:
        st.metric(label="Adjusted Churn Probability", value=f"{adjusted_churn_probability * 100:.2f}%")
        st.metric(label="Stay Probability", value=f"{stay_probability * 100:.2f}%")
        st.caption(f"Base model probability: {base_churn_probability * 100:.2f}%")
        st.caption(f"Behavior risk score: {behavior_score:+.2f}")
        st.caption(f"Behavior mode: {mode_label} (weight {behavior_weight:.2f})")

    st.subheader("Prediction Charts")
    st.altair_chart(build_probability_chart(adjusted_churn_probability, stay_probability), width="stretch")

    st.subheader("Interaction and Feature Discovery Frequency")
    st.altair_chart(build_interaction_frequency_chart(behavior), width="stretch")
    freq_col1, freq_col2, freq_col3 = st.columns(3)
    with freq_col1:
        st.metric("Interactions (Logins/Week)", behavior["logins_per_week"])
    with freq_col2:
        st.metric("Online Txns/Month", behavior["online_txn_per_month"])
    with freq_col3:
        st.metric("Feature Research/Month", behavior["feature_research_per_month"])

    feature_chart = build_feature_comparison_chart(input_df=input_df, reference_profile=reference_profile)
    if feature_chart is not None:
        st.subheader("Input vs Dataset Average")
        st.altair_chart(feature_chart, width="stretch")
    else:
        st.info("Feature comparison chart is unavailable because the dataset reference profile is missing.")

    st.subheader("Customer Pain Points and Retention Plan")
    demerits, actions = derive_customer_demerits_and_actions(
        input_df=input_df,
        churn_probability=adjusted_churn_probability,
        behavior=behavior,
        reference_profile=reference_profile,
    )

    pain_col, action_col = st.columns(2)
    with pain_col:
        st.markdown("### Likely Customer Demerits")
        for item in demerits:
            st.write(f"- {item}")
    with action_col:
        st.markdown("### What Bank Can Do to Retain")
        for item in actions:
            st.write(f"- {item}")


if __name__ == "__main__":
    main()
