
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import xgboost as xgb

from pathlib import Path


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Credence AI | Risk Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================



st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --green: #174D38;
    --green-dark: #103B2B;
    --green-mid: #286B4E;
    --green-light: #EAF3ED;
    --white: #FFFFFF;
}

.stApp {
    background-color: #FFFFFF;
    background-image:
        radial-gradient(circle at 5% 12%, rgba(23,77,56,.07), transparent 25%),
        radial-gradient(circle at 95% 75%, rgba(23,77,56,.06), transparent 25%);
    color: var(--green);
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.brand {
    font-family: 'Manrope', sans-serif;
    font-size: 29px;
    font-weight: 800;
    color: var(--green);
    letter-spacing: -1.4px;
}

.brand span {
    color: var(--green-mid);
}

.nav-label {
    text-align: right;
    color: var(--green);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding-top: 12px;
}

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(120deg, #103B2B 0%, #174D38 65%, #286B4E 100%);
    padding: 48px 45px;
    border-radius: 22px;
    margin: 28px 0 38px;
    color: #FFFFFF;
    box-shadow: 0 16px 40px rgba(16,59,43,.15);
}

.hero::before {
    content: "";
    position: absolute;
    width: 360px;
    height: 360px;
    border: 1px solid rgba(255,255,255,.13);
    border-radius: 50%;
    right: -90px;
    top: -170px;
}

.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    border: 1px solid rgba(255,255,255,.12);
    border-radius: 50%;
    right: 20px;
    bottom: -190px;
}

.hero-layout {
    display: grid;
    grid-template-columns: minmax(0, 1.4fr) minmax(220px, .8fr);
    gap: 35px;
    align-items: center;
    position: relative;
    z-index: 1;
}

.hero-tag {
    color: #FFFFFF;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    margin-bottom: 17px;
}

.hero h1 {
    display: block;
    color: #FFFFFF !important;
    font-family: 'Manrope', sans-serif;
    font-size: 46px;
    font-weight: 800;
    line-height: 1.25;
    letter-spacing: -1px;
    margin: 0 0 18px 0;
    overflow-wrap: normal;
}
.hero h1 span {
    display: block;
    color: #FFFFFF !important;
    margin-top: 8px;
    font-size: 30px;
    line-height: 1.3;
}

.hero p {
    color: #FFFFFF !important;
    font-size: 15px;
    line-height: 1.8;
    max-width: 590px;
    margin: 0;
}

.hero-brand {
    font-family: 'Manrope', sans-serif;
    color: #FFFFFF;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 3px;
    margin-top: 25px;
}

.hero-visual {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 230px;
}

.hero-visual svg {
    width: 100%;
    max-width: 270px;
    height: auto;
}

.section-heading {
    color: var(--green);
    font-family: 'Manrope', sans-serif;
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 5px;
}

.section-description {
    color: var(--green-mid);
    font-size: 13px;
    margin-bottom: 20px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #D5E5DA;
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 5px 20px rgba(23,77,56,.035);
}

.stNumberInput label,
.stSelectbox label {
    color: var(--green) !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}

.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border: 1px solid #BBD4C4 !important;
    border-radius: 9px !important;
    color: var(--green) !important;
    min-height: 43px;
}

.stNumberInput input:focus,
.stSelectbox div[data-baseweb="select"] > div:focus-within {
    border-color: var(--green) !important;
    box-shadow: 0 0 0 2px #EAF3ED !important;
}

.stButton > button {
    border-radius: 10px;
    min-height: 48px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    transition: all .2s ease;
}

.stButton > button[kind="primary"] {
    background: var(--green);
    color: #FFFFFF;
    border: 1px solid var(--green);
}

.stButton > button[kind="primary"]:hover {
    background: var(--green-dark);
    border-color: var(--green-dark);
    transform: translateY(-2px);
}

.stButton > button[kind="secondary"] {
    background: #FFFFFF;
    color: var(--green);
    border: 1px solid #A9C8B5;
}

.stButton > button[kind="secondary"]:hover {
    background: var(--green-light);
    color: var(--green);
    border-color: var(--green);
}

.result-card {
    background: #FFFFFF;
    border: 1px solid #C8DDCF;
    border-top: 4px solid var(--green);
    border-radius: 14px;
    padding: 23px;
    min-height: 125px;
    box-shadow: 0 5px 18px rgba(23,77,56,.05);
}

.result-label {
    color: var(--green-mid);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

.result-value {
    color: var(--green);
    font-family: 'Manrope', sans-serif;
    font-size: 25px;
    font-weight: 800;
    margin-top: 12px;
}

.risk-high, .risk-low {
    background: var(--green-light);
    border: 1px solid #BBD4C4;
    border-left: 5px solid var(--green);
    border-radius: 13px;
    padding: 22px;
    color: var(--green);
}

.risk-title {
    color: var(--green);
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 8px;
}

.risk-description {
    color: var(--green);
    font-size: 13px;
    line-height: 1.7;
}

.stProgress > div > div > div > div {
    background-color: var(--green);
}

.stProgress > div > div {
    background-color: var(--green-light);
}

.footer {
    text-align: center;
    color: var(--green);
    font-size: 12px;
    font-weight: 600;
    padding-top: 22px;
    letter-spacing: .6px;
}

hr {
    border-color: #D5E5DA;
}

@media(max-width: 700px) {
    .hero {
        padding: 30px 23px;
    }
    .hero-layout {
        grid-template-columns: 1fr;
    }
    .hero h1 {
        font-size: 35px;
    }
    .hero-visual {
        min-height: 150px;
    }
}
</style>
""", unsafe_allow_html=True)



# =========================================================
# 3. PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DATA_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"


# =========================================================
# 4. LOAD DATA AND MODEL
# =========================================================

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_components():

    preprocessor = joblib.load(
        MODEL_DIR / "credit_preprocessor.pkl"
    )

    classifier = xgb.XGBClassifier()

    classifier.load_model(
        MODEL_DIR / "credit_xgboost.json"
    )

    with open(
        MODEL_DIR / "calibration_parameters.json",
        "r"
    ) as file:
        calibration = json.load(file)

    threshold = float(
        joblib.load(MODEL_DIR / "best_threshold.pkl")
    )

    return preprocessor, classifier, calibration, threshold


try:

    data = load_data()

    preprocessor, classifier, calibration, threshold = (
        load_components()
    )

except Exception as e:

    st.error(f"Unable to load project files: {e}")
    st.stop()


# =========================================================
# 5. SESSION STATE
# =========================================================

input_keys = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_grade",
    "loan_amnt",
    "loan_int_rate",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

if "assessment_result" not in st.session_state:
    st.session_state.assessment_result = None

if "assessment_error" not in st.session_state:
    st.session_state.assessment_error = None


# =========================================================
# 6. RESET FUNCTION
# =========================================================

def reset_form():

    for key in input_keys:
        st.session_state[key] = None

    st.session_state.assessment_result = None
    st.session_state.assessment_error = None


# =========================================================
# 7. ASSESSMENT FUNCTION
# =========================================================

def assess_application():

    values = {
        key: st.session_state.get(key)
        for key in input_keys
    }

    try:

        # Validate employment length

        if values["person_emp_length"] > values["person_age"]:

            st.session_state.assessment_error = (
                "Employment length cannot exceed applicant age."
            )

            return

        # Calculate loan-to-income ratio

        loan_percent_income = (
            values["loan_amnt"] / values["person_income"]
        )

        # Prepare applicant data

        applicant = pd.DataFrame([{

            "person_age": values["person_age"],

            "person_income": values["person_income"],

            "person_home_ownership": values[
                "person_home_ownership"
            ],

            "person_emp_length": values["person_emp_length"],

            "loan_intent": values["loan_intent"],

            "loan_grade": values["loan_grade"],

            "loan_amnt": values["loan_amnt"],

            "loan_int_rate": values["loan_int_rate"],

            "loan_percent_income": loan_percent_income,

            "cb_person_default_on_file": values[
                "cb_person_default_on_file"
            ],

            "cb_person_cred_hist_length": values[
                "cb_person_cred_hist_length"
            ]

        }])

        # STEP 1: Preprocessing

        processed_data = preprocessor.transform(applicant)

        # STEP 2: XGBoost prediction

        raw_probability = classifier.predict_proba(
            processed_data
        )[0][1]

        # STEP 3: Probability calibration

        a = calibration["a"]
        b = calibration["b"]

        probability = 1 / (
            1 + np.exp(
                np.clip(
                    a * raw_probability + b,
                    -500,
                    500
                )
            )
        )

        # STEP 4: Threshold optimization

        prediction = int(probability >= threshold)

        # Store results without clearing applicant details

        st.session_state.assessment_result = {
            "probability": float(probability),
            "threshold": float(threshold),
            "prediction": prediction
        }

        st.session_state.assessment_error = None

    except Exception as e:

        st.session_state.assessment_error = (
            f"Assessment failed: {e}"
        )


# =========================================================
# 8. TOP NAVIGATION
# =========================================================

st.markdown(
    '<div style="height: 35px;"></div>',
    unsafe_allow_html=True
)

header_left, header_right = st.columns([3, 2])

with header_left:
    st.markdown(
        '<div class="brand">◈ CREDENCE AI</div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="nav-label">FINANCIAL RISK INTELLIGENCE</div>',
        unsafe_allow_html=True
    )

st.markdown(
    '<div style="height: 12px;"></div>',
    unsafe_allow_html=True
)

st.divider()



# =========================================================
# 9. HERO SECTION
# =========================================================

st.markdown(
    "<h1 style='color:green;'>CREDENCE AI TEST</h1>",
    unsafe_allow_html=True
)



# =========================================================
# 10. APPLICANT INFORMATION
# =========================================================

st.markdown(
    '<div class="section-heading">Applicant Profile</div>'
    '<div class="section-description">'
    'Enter the applicant information to generate a risk assessment.'
    '</div>',
    unsafe_allow_html=True
)


with st.container(border=True):

    st.markdown("#### Personal & Financial Details")

    col1, col2 = st.columns(2)

    with col1:

        st.number_input(
            "Applicant Age",
            min_value=18,
            max_value=100,
            value=None,
            step=1,
            placeholder="Enter age",
            key="person_age"
        )

        st.number_input(
            "Annual Income (₹)",
            min_value=1,
            value=None,
            step=10000,
            placeholder="Enter annual income",
            key="person_income"
        )

        st.selectbox(
            "Home Ownership",
            options=sorted(
                data["person_home_ownership"].dropna().unique()
            ),
            index=None,
            placeholder="Select home ownership",
            key="person_home_ownership"
        )

        st.number_input(
            "Employment Length (Years)",
            min_value=0.0,
            max_value=60.0,
            value=None,
            step=0.5,
            placeholder="Enter employment length",
            key="person_emp_length"
        )

        st.selectbox(
            "Loan Purpose",
            options=sorted(
                data["loan_intent"].dropna().unique()
            ),
            index=None,
            placeholder="Select loan purpose",
            key="loan_intent"
        )

    with col2:

        st.selectbox(
            "Loan Grade",
            options=sorted(
                data["loan_grade"].dropna().unique()
            ),
            index=None,
            placeholder="Select loan grade",
            key="loan_grade"
        )

        st.number_input(
            "Loan Amount (₹)",
            min_value=1,
            value=None,
            step=10000,
            placeholder="Enter loan amount",
            key="loan_amnt"
        )

        st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=None,
            step=0.1,
            placeholder="Enter interest rate",
            key="loan_int_rate"
        )

        st.selectbox(
            "Previous Default on File",
            options=sorted(
                data["cb_person_default_on_file"].dropna().unique()
            ),
            index=None,
            placeholder="Select previous default status",
            key="cb_person_default_on_file"
        )

        st.number_input(
            "Credit History Length (Years)",
            min_value=0.0,
            value=None,
            step=0.5,
            placeholder="Enter credit history length",
            key="cb_person_cred_hist_length"
        )


# =========================================================
# 11. BUTTONS
# =========================================================

all_fields_filled = all(
    st.session_state.get(key) is not None
    for key in input_keys
)

st.write("")

button1, button2 = st.columns([3, 1])

with button1:

    st.button(
        "Assess Application  →",
        type="primary",
        use_container_width=True,
        disabled=not all_fields_filled,
        on_click=assess_application
    )

with button2:

    st.button(
        "↻ Reset Form",
        type="secondary",
        use_container_width=True,
        on_click=reset_form
    )


if not all_fields_filled:

    st.caption(
        "Complete all applicant fields to enable the assessment."
    )


if st.session_state.assessment_error:

    st.error(st.session_state.assessment_error)


# =========================================================
# 12. ASSESSMENT RESULTS
# =========================================================

result = st.session_state.assessment_result

if result is not None:

    probability = result["probability"]
    threshold_value = result["threshold"]
    prediction = result["prediction"]

    st.divider()

    st.markdown(
        '<div class="section-heading">Risk Assessment Overview</div>'
        '<div class="section-description">'
        'Machine learning assessment based on the submitted applicant profile.'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Default Probability</div>
            <div class="result-value">{probability * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Optimized Threshold</div>
            <div class="result-value">{threshold_value * 100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:

        risk_label = "HIGH RISK" if prediction == 1 else "LOW RISK"

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Risk Classification</div>
            <div class="result-value">{risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.markdown("**Default Probability Indicator**")

    st.progress(probability)
    if prediction == 1:

        st.error(
            "Higher Default Risk Identified — The model classifies "
            "this applicant as having a higher probability of loan default. "
            "Further financial review is recommended."
        )

    else:

         
        st.success(
            "Lower Default Risk Identified — The model classifies "
            "this applicant as having a lower probability of loan default."
        )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.divider()

st.markdown("""
<div class="footer">
    CREDENCE AI · FINANCIAL RISK INTELLIGENCE
   
</div>
""", unsafe_allow_html=True)