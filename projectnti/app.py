import streamlit as st
import pandas as pd
import joblib

model = joblib.load("random_forest_model.pkl")
columns = joblib.load("columns.pkl")
st.title("Credit Risk Prediction")
age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=30
)

income = st.number_input(
    "Income",
    min_value=0,
    value=50000
)

emp_length = st.number_input(
    "Employment Length",
    min_value=0,
    value=5
)

loan_amount = st.number_input(
    "Loan Amount",
    min_value=0,
    value=10000
)

interest = st.number_input(
    "Interest Rate",
    min_value=0.0,
    value=10.5
)

loan_percent = st.number_input(
    "Loan Percent Income",
    min_value=0.0,
    value=0.20
)

credit_history = st.number_input(
    "Credit History Length",
    min_value=0,
    value=5
)
home = st.selectbox(
    "Home Ownership",
    [
        "RENT",
        "OWN",
        "MORTGAGE",
        "OTHER"
    ]
)
intent = st.selectbox(
    "Loan Intent",
    [
        "EDUCATION",
        "MEDICAL",
        "VENTURE",
        "PERSONAL",
        "HOMEIMPROVEMENT",
        "DEBTCONSOLIDATION"
    ]
) 
grade = st.selectbox(
    "Loan Grade",
    [
        "A","B","C","D","E","F","G"
    ]
)
default = st.selectbox(
    "Default On File",
    [
        "N","Y"
    ]
)
predict = st.button("Predict")
if predict:
    new_customer = pd.DataFrame({
    "person_age": [age],
    "person_income": [income],
    "person_home_ownership": [home],
    "person_emp_length": [emp_length],
    "loan_intent": [intent],
    "loan_grade": [grade],
    "loan_amnt": [loan_amount],
    "loan_int_rate": [interest],
    "loan_percent_income": [loan_percent],
    "cb_person_default_on_file": [default],
    "cb_person_cred_hist_length": [credit_history]
})
    new_customer["loan_grade"] = new_customer["loan_grade"].map({
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6
})
    new_customer["cb_person_default_on_file"] = (
    new_customer["cb_person_default_on_file"]
    .map({
        "N": 0,
        "Y": 1
    })
)
    new_customer = pd.get_dummies(
    new_customer,
    columns=[
        "loan_intent",
        "person_home_ownership"
    ],
    drop_first=True
)
    new_customer = new_customer.reindex(
    columns=columns,
    fill_value=0
)
    prediction = model.predict(new_customer)
    if prediction[0] == 1:
      st.error("⚠️ High Credit Risk")
    else:
      st.success("✅ Low Credit Risk")
