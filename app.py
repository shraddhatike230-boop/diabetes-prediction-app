import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Diabetes AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================
st.markdown("""
<style>

/* ---------- APP BACKGROUND ---------- */
[data-testid="stAppViewContainer"] {
    background: #f5f7fb;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ---------- SIDEBAR ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #171b3a, #292b5f);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    padding: 38px 35px;
    border-radius: 25px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 15px 35px rgba(79,70,229,0.22);
}

.hero-title {
    font-size: 43px;
    font-weight: 800;
    margin: 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 17px;
    margin-top: 8px;
    opacity: 0.9;
}

.hero-badge {
    display: inline-block;
    margin-top: 18px;
    padding: 7px 15px;
    border-radius: 30px;
    background: rgba(255,255,255,0.16);
    font-size: 13px;
}

/* ---------- SECTION ---------- */
.section-heading {
    font-size: 25px;
    font-weight: 800;
    color: #20233d;
    margin-top: 25px;
    margin-bottom: 15px;
}

/* ---------- MODEL CARDS ---------- */
.info-card {
    background: white;
    border-radius: 18px;
    padding: 20px;
    border: 1px solid #e9eaf2;
    box-shadow: 0 7px 20px rgba(20,25,50,0.06);
    min-height: 125px;
}

.info-label {
    color: #777b91;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
}

.info-value {
    color: #20233d;
    font-size: 25px;
    font-weight: 800;
    margin-top: 10px;
}

.info-icon {
    font-size: 25px;
}

/* ---------- INPUT BOX ---------- */
.input-wrapper {
    background: white;
    border-radius: 20px;
    padding: 8px 18px 18px 18px;
    border: 1px solid #e8e9f0;
    box-shadow: 0 7px 20px rgba(20,25,50,0.05);
}

/* ---------- BUTTON ---------- */
.stButton > button {
    width: 100%;
    height: 55px;
    border-radius: 14px;
    border: none;
    font-size: 17px;
    font-weight: 800;
    background: linear-gradient(90deg, #4f46e5, #7c3aed);
    color: white;
    box-shadow: 0 8px 20px rgba(79,70,229,0.22);
    transition: 0.2s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 25px rgba(79,70,229,0.30);
}

/* ---------- RESULT ---------- */
.result-high {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1px solid #fecdd3;
    border-radius: 22px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.result-low {
    background: linear-gradient(135deg, #ecfdf5, #d1fae5);
    border: 1px solid #a7f3d0;
    border-radius: 22px;
    padding: 28px;
    text-align: center;
    margin-top: 20px;
}

.result-heading {
    font-size: 30px;
    font-weight: 850;
}

.result-prob {
    font-size: 22px;
    font-weight: 700;
    margin-top: 8px;
}

/* ---------- RISK BOX ---------- */
.risk-box {
    background: white;
    border-radius: 18px;
    padding: 22px;
    border: 1px solid #e8e9f0;
    box-shadow: 0 7px 20px rgba(20,25,50,0.05);
    margin-top: 18px;
}

.risk-title {
    font-size: 19px;
    font-weight: 800;
    color: #20233d;
}

.risk-text {
    color: #65697d;
    line-height: 1.6;
    margin-top: 8px;
}

/* ---------- FOOTER ---------- */
.footer {
    text-align: center;
    margin-top: 45px;
    padding: 25px;
    color: #777b91;
    border-top: 1px solid #dddfea;
}

.small-note {
    color: #777b91;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================
data = pd.read_csv("data.csv")

data.columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

# =========================================================
# MODEL TRAINING
# =========================================================
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

accuracy = model.score(X_test_scaled, y_test)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.markdown("## 🩺 Diabetes AI")

    st.markdown("---")

    st.markdown("### 📌 Dashboard")

    st.write("🏠 Risk Prediction")
    st.write("📊 Model Information")
    st.write("👤 Patient Analysis")

    st.markdown("---")

    st.markdown("### 🤖 AI Model")

    st.write("Algorithm")
    st.write("Logistic Regression")

    st.write("Features")
    st.write("8 Patient Parameters")

    st.markdown("---")

    st.caption(
        "Educational ML project\n\n"
        "Not a medical diagnosis."
    )

# =========================================================
# HERO SECTION
# =========================================================
st.html("""
<div class="hero">
<div class="hero-title">🩺 Diabetes AI Predictor</div>
<div class="hero-subtitle">Intelligent diabetes risk assessment powered by Machine Learning</div>
<div class="hero-badge">🤖 Logistic Regression &nbsp; • &nbsp; 📊 Data Analysis &nbsp; • &nbsp; ⚡ Instant Prediction</div>
</div>
""")

# =========================================================
# INTRO
# =========================================================
st.markdown("""
<div class="small-note">
Enter the patient's information below and let the trained machine-learning
model estimate the diabetes risk probability.
</div>
""", unsafe_allow_html=True)

# =========================================================
# MODEL OVERVIEW
# =========================================================
st.markdown(
    '<div class="section-heading">📊 Model Overview</div>',
    unsafe_allow_html=True
)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-icon">🎯</div>
        <div class="info-label">Model Accuracy</div>
        <div class="info-value">{accuracy * 100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="info-card">
        <div class="info-icon">📁</div>
        <div class="info-label">Dataset Records</div>
        <div class="info-value">{len(data)}</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🤖</div>
        <div class="info-label">Algorithm</div>
        <div class="info-value">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="info-card">
        <div class="info-icon">🧬</div>
        <div class="info-label">Input Features</div>
        <div class="info-value">8</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MODEL PERFORMANCE
# =========================================================
st.markdown(
    '<div class="section-heading">📈 Model Performance</div>',
    unsafe_allow_html=True
)

p1, p2, p3 = st.columns(3)

with p1:
    st.metric("Training Records", len(X_train))

with p2:
    st.metric("Testing Records", len(X_test))

with p3:
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

st.info(
    "The model uses Logistic Regression and StandardScaler "
    "to analyze the patient input features."
)

st.markdown(
    '<div class="section-heading">👤 Patient Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="input-wrapper">',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "🤰 Number of Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "🩸 Glucose Level",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    blood_pressure = st.number_input(
        "💓 Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "📏 Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )

with col2:

    insulin = st.number_input(
        "💉 Insulin",
        min_value=0,
        max_value=900,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "⚖️ BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "🧬 Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5,
        step=0.01
    )

    age = st.number_input(
        "🎂 Age",
        min_value=1,
        max_value=120,
        value=25,
        step=1
    )

st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# PREDICT BUTTON
# =========================================================
st.markdown("###")

predict = st.button("🔍  ANALYZE DIABETES RISK")

# =========================================================
# PREDICTION
# =========================================================
if predict:

    patient = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    # Scale patient data
    patient_scaled = scaler.transform(patient)

    # Prediction
    prediction = model.predict(patient_scaled)[0]

    # Probability
    probability = model.predict_proba(patient_scaled)[0][1]

    # =====================================================
    # SUMMARY
    # =====================================================
    st.markdown(
        '<div class="section-heading">📋 Prediction Summary</div>',
        unsafe_allow_html=True
    )

    s1, s2, s3, s4 = st.columns(4)

    with s1:
        st.metric("🎂 Age", age)

    with s2:
        st.metric("🩸 Glucose", glucose)

    with s3:
        st.metric("⚖️ BMI", f"{bmi:.1f}")

    with s4:
        st.metric("💓 Blood Pressure", blood_pressure)

    # =====================================================
    # RESULT
    # =====================================================
    if prediction == 1:

        st.html(f"""
<div class="result-heading">
    🔴 HIGHER PREDICTED RISK
</div>

<div class="result-prob">
    Estimated probability: {probability * 100:.2f}%
</div>
""")

    else:

        st.markdown(f"""
        <div class="result-low">

            <div class="result-heading">
                🟢 LOWER PREDICTED RISK
            </div>

            <div class="result-prob">
                Estimated probability: {probability * 100:.2f}%
            </div>

        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # PROBABILITY
    # =====================================================
    st.markdown("### 📈 Risk Probability")

    st.progress(float(probability))

    left, right = st.columns(2)

    with left:
        st.caption("0% — Lower model probability")

    with right:
        st.caption("100% — Higher model probability")

    # =====================================================
    # INPUT SNAPSHOT
    # =====================================================
    st.markdown(
        '<div class="section-heading">🔎 Patient Input Snapshot</div>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Parameter": [
            "Glucose",
            "Blood Pressure",
            "BMI",
            "Age"
        ],
        "Value": [
            glucose,
            blood_pressure,
            bmi,
            age
        ]
    })

    st.bar_chart(
        chart_data.set_index("Parameter")
    )

    # =====================================================
    # INTERPRETATION
    # =====================================================
    st.markdown(
        '<div class="section-heading">💡 Prediction Interpretation</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

       st.html("""
<div class="risk-title">
    🔴 Model indicates a higher predicted risk
</div>

<div class="risk-text">
    Based on the information entered, the trained Logistic Regression
    model produced a higher diabetes probability.
    The result should be treated as an educational machine-learning
    prediction and not as a medical diagnosis.
</div>
""")

    else:

        st.markdown("""
        <div class="risk-box">

            <div class="risk-title">
                🟢 Model indicates a lower predicted risk
            </div>

            <div class="risk-text">
                Based on the information entered, the trained Logistic
                Regression model produced a lower diabetes probability.
                The result should be treated as an educational machine
                learning prediction and not as a medical diagnosis.
            </div>

        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # MEDICAL DISCLAIMER
    # =====================================================
    st.warning(
        "⚠️ Educational use only: The probability is generated by a "
        "machine-learning model trained on the supplied dataset. "
        "It is not a medical diagnosis or a substitute for professional "
        "medical advice."
    )

# =========================================================
# ABOUT MODEL
# =========================================================
st.markdown("###")

with st.expander("🧠 About the Machine Learning Model"):

    st.markdown("""
    ### 🤖 Logistic Regression

    This project uses **Logistic Regression**, a supervised machine
    learning classification algorithm.

    The model analyzes these 8 features:

    | Feature | Description |
    |---|---|
    | Pregnancies | Number of pregnancies |
    | Glucose | Glucose measurement |
    | Blood Pressure | Blood pressure measurement |
    | Skin Thickness | Skin thickness measurement |
    | Insulin | Insulin measurement |
    | BMI | Body Mass Index |
    | Diabetes Pedigree | Diabetes-related family history indicator |
    | Age | Patient age |

    ### ⚙️ Processing Pipeline

    **Dataset → Train/Test Split → StandardScaler → Logistic Regression → Prediction**

    The input features are standardized using `StandardScaler`
    before being passed to the Logistic Regression model.
    """)

# =========================================================
# PROJECT INFORMATION
# =========================================================
with st.expander("📚 Project Information"):

    st.write("""
    **Project:** Diabetes AI Predictor

    **Technology Stack**
    - Python
    - Streamlit
    - Pandas
    - Scikit-learn
    - Logistic Regression
    - StandardScaler

    **Project Type:** Machine Learning Classification

    **Purpose:** Educational demonstration of a diabetes-risk
    prediction workflow.
    """)

# =========================================================
# FOOTER
# =========================================================
st.html("""
<div class="footer">

    <div style="font-size:22px;font-weight:800;">
        🩺 Diabetes AI Predictor
    </div>

    <div style="margin-top:8px;">
        Built with Python • Streamlit • Scikit-learn
    </div>

    <div style="margin-top:12px;font-size:12px;">
        ⚠️ For educational purposes only • Not a medical diagnosis
    </div>

</div>
""")