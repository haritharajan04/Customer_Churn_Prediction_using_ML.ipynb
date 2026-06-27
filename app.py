
import streamlit as st
import pandas as pd

import joblib

model_data = joblib.load("customer_churn_model.joblib")
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a customer is likely to churn.")

# Load model


model = model_data["model"]

# Load encoders
with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

st.header("Customer Information")

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox("Gender", ["Male","Female"])

    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0,1],
        format_func=lambda x: "Yes" if x==1 else "No"
    )

    Partner = st.selectbox("Partner",["Yes","No"])

    Dependents = st.selectbox("Dependents",["Yes","No"])

    tenure = st.slider(
        "Tenure (Months)",
        0,
        72,
        12
    )

    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes","No"]
    )

    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["No","Yes","No phone service"]
    )

    InternetService = st.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )

    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
    )

with col2:

    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes","No","No internet service"]
    )

    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
    )

    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes","No","No internet service"]
    )

    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes","No","No internet service"]
    )

    Contract = st.selectbox(
        "Contract",
        ["Month-to-month","One year","Two year"]
    )

    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )

    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=1000.0
    )

if st.button("Predict Churn"):

    input_data = {
        "gender":gender,
        "SeniorCitizen":SeniorCitizen,
        "Partner":Partner,
        "Dependents":Dependents,
        "tenure":tenure,
        "PhoneService":PhoneService,
        "MultipleLines":MultipleLines,
        "InternetService":InternetService,
        "OnlineSecurity":OnlineSecurity,
        "OnlineBackup":OnlineBackup,
        "DeviceProtection":DeviceProtection,
        "TechSupport":TechSupport,
        "StreamingTV":StreamingTV,
        "StreamingMovies":StreamingMovies,
        "Contract":Contract,
        "PaperlessBilling":PaperlessBilling,
        "PaymentMethod":PaymentMethod,
        "MonthlyCharges":MonthlyCharges,
        "TotalCharges":TotalCharges
    }

    input_df = pd.DataFrame([input_data])

    # Encode categorical features
    for column, encoder in encoders.items():
        input_df[column] = encoder.transform(input_df[column])

    prediction = model.predict(input_df)

    probability = model.predict_proba(input_df)

    st.divider()

    if prediction[0] == 1:
        st.error("⚠ This customer is likely to churn.")
    else:
        st.success("✅ This customer is unlikely to churn.")

    st.subheader("Prediction Probability")

    st.write(
        f"Probability of Churn: **{probability[0][1]*100:.2f}%**"
    )

    st.progress(float(probability[0][1]))
