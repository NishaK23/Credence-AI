
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
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: #F3F5F1;
    color: #26352D;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.brand {
    font-family: 'DM Sans', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #244536;
    letter-spacing: -0.8px;
}

.brand span {
    color: #8A9D83;
}

.hero {
    background: #E4EBE2;
    padding: 42px;
    border-radius: 18px;
    margin: 25px 0 35px 0;
    border: 1px solid #D4DED1;
}

.hero-tag {
    color: #66816B;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    margin-bottom: 16px;
}

.hero h1 {
    color: #244536;
    font-family: 'Playfair Display', serif;
    font-size: 39px;
    line-height: 1.2;
    margin: 0 0 14px 0;
}

.hero p {
    color: #5E7063;
    font-size: 15px;
    line-height: 1.8;
    max-width: 720px;
    margin: 0;
}

.section-heading {
    color: #244536;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 5px;
}

.section-description {
    color: #78877B;
    font-size: 13px;
    margin-bottom: 20px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #E0E6DE;
    border-radius: 14px;
}

.stNumberInput label,
.stSelectbox label {
    color: #405849 !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}

.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div {
    background: #FAFBF9 !important;
    border-color: #DCE4DA !important;
    border-radius: 8px !important;
    color: #26352D !important;
}

.stButton > button {
    border-radius: 9px;
    height: 47px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
}

.stButton > button[kind="primary"] {
    background: #315B45;
    color: white;
    border: 1px solid #315B45;
}

.stButton > button[kind="primary"]:hover {
    background: #244536;
    border-color: #244536;
}

.stButton > button[kind="secondary"] {
    background: #E4EBE2;
    color: #315B45;
    border: 1px solid #CFDCCD;
}

.result-card {
    background: white;
    border: 1px solid #E0E6DE;
    border-radius: 14px;
    padding: 23px;
    min-height: 125px;
}

.result-label {
    color: #78877B;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.result-value {
    color: #244536;
    font-size: 26px;
    font-weight: 700;
    margin-top: 12px;
}

.risk-high {
    background: #F8EAE7;
    border: 1px solid #E8C9C1;
    border-radius: 13px;
    padding: 22px;
    color: #934B40;
}

.risk-low {
    background: #E7F0E7;
    border: 1px solid #CBDDCA;
    border-radius: 13px;
    padding: 22px;
    color: #386347;
}

.risk-title {
    font-size: 20px;
    font-weight: 700;
    margin-bottom: 8px;
}

.risk-description {
    font-size: 13px;
    line-height: 1.7;
}

.footer {
    text-align: center;
    color: #89968B;
    font-size: 12px;
    padding-top: 20px;
}

hr {
    border-color: #E0E6DE;
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

header_left, header_right = st.columns([3, 1])

with header_left:

    st.markdown(
        '<div class="brand">◈ CREDENCE <span>AI</span></div>',
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        '<div style="text-align:right; color:#8B7868; '
        'font-size:12px; padding-top:10px;">'
        'FINANCIAL RISK INTELLIGENCE</div>',
        unsafe_allow_html=True
    )


st.divider()


# =========================================================
# 9. HERO SECTION
# =========================================================

st.markdown("""
<div class="hero">

    <div class="hero-tag">
        INTELLIGENT CREDIT ASSESSMENT
    </div>

    <h1>Make smarter lending decisions.</h1>

    <p>
        Evaluate applicant information using machine learning,
        calibrated default probabilities and an optimized
        decision threshold. Gain a clearer understanding of
        potential credit risk before making lending decisions.
    </p>

</div>
""", unsafe_allow_html=True)


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

        st.markdown("""
        <div class="risk-high">

            <div class="risk-title">
                Higher Default Risk Identified
            </div>

            <div class="risk-description">
                The model classifies this applicant as having
                a higher probability of loan default based on
                the selected decision threshold. Further
                financial review is recommended.
            </div>

        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="risk-low">

            <div class="risk-title">
                Lower Default Risk Identified
            </div>

            <div class="risk-description">
                The model classifies this applicant as having
                a lower probability of loan default based on
                the selected decision threshold.
            </div>

        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "This machine learning assessment is a decision-support "
        "tool and should not be treated as an automatic lending decision."
    )


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.markdown("""
<div class="footer">

    CREDENCE AI · FINANCIAL RISK INTELLIGENCE

    <br><br>

    Machine Learning &nbsp; • &nbsp;
    Probability Calibration &nbsp; • &nbsp;
    Threshold Optimization

</div>
""", unsafe_allow_html=True)
