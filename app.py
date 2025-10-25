import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the saved model
with open('attrition_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load employee dataset (for searching by ID)
try:
    df = pd.read_csv('general_data.csv')
except FileNotFoundError:
    df = pd.DataFrame()
    st.warning("⚠️ 'general_data.csv' not found. Search mode will be disabled.")

# Define the features
numerical_cols = ['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction',
                  'WorkLifeBalance', 'Age', 'DistanceFromHome', 'Education', 'JobLevel', 'MonthlyIncome',
                  'NumCompaniesWorked', 'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
                  'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager']

categorical_cols = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']

# App title
st.title("🧩 Employee Attrition Prediction")

# Sidebar for input mode
st.sidebar.header("🔍 Input Mode")
mode = st.sidebar.radio("Choose how you want to predict:", ["Manual Entry", "Search by Employee ID"])

input_df = pd.DataFrame()

# ===========================
# Search by Employee ID
# ===========================
if mode == "Search by Employee ID":
    if not df.empty:
        emp_id = st.sidebar.selectbox("Select Employee ID", df["EmployeeID"].unique())
        emp_data = df[df["EmployeeID"] == emp_id].iloc[0]

        st.subheader(f"Employee Details for ID: {emp_id}")
        st.dataframe(emp_data)

        input_df = pd.DataFrame([emp_data])

    else:
        st.error("Employee data not available. Please ensure 'general_data.csv' is in the app directory.")

# ===========================
# Manual Entry Mode
# ===========================
else:
    st.header("Numerical Features")
    num_inputs = {}
    for col in numerical_cols:
        if col in ['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction', 'JobSatisfaction',
                   'WorkLifeBalance', 'Education', 'JobLevel', 'StockOptionLevel']:
            num_inputs[col] = st.number_input(col, min_value=1, max_value=5, value=3)
        elif col == 'Age':
            num_inputs[col] = st.number_input(col, min_value=18, max_value=100, value=30)
        elif col == 'DistanceFromHome':
            num_inputs[col] = st.number_input(col, min_value=1, max_value=30, value=5)
        elif col == 'MonthlyIncome':
            num_inputs[col] = st.number_input(col, min_value=1000, max_value=200000, value=5000)
        elif col == 'NumCompaniesWorked':
            num_inputs[col] = st.number_input(col, min_value=0, max_value=10, value=1)
        elif col == 'PercentSalaryHike':
            num_inputs[col] = st.number_input(col, min_value=0, max_value=30, value=10)
        elif col == 'TotalWorkingYears':
            num_inputs[col] = st.number_input(col, min_value=0, max_value=50, value=5)
        elif col == 'TrainingTimesLastYear':
            num_inputs[col] = st.number_input(col, min_value=0, max_value=10, value=2)
        elif col == 'YearsAtCompany':
            num_inputs[col] = st.number_input(col, min_value=0, max_value=50, value=3)
        elif col == 'YearsSinceLastPromotion':
            num_inputs[col] = st.number_input(col, min_value=0, max_value=20, value=1)
        elif col == 'YearsWithCurrManager':
            num_inputs[col] = st.number_input(col, min_value=0, max_value=20, value=2)

    st.header("Categorical Features")
    cat_inputs = {}
    cat_inputs['BusinessTravel'] = st.selectbox('BusinessTravel', options=['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    cat_inputs['Department'] = st.selectbox('Department', options=['Sales', 'Research & Development', 'Human Resources'])
    cat_inputs['EducationField'] = st.selectbox('EducationField', options=['Life Sciences', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources', 'Other'])
    cat_inputs['Gender'] = st.selectbox('Gender', options=['Male', 'Female'])
    cat_inputs['JobRole'] = st.selectbox('JobRole', options=['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                                             'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    cat_inputs['MaritalStatus'] = st.selectbox('MaritalStatus', options=['Single', 'Married', 'Divorced'])

    input_data = {**num_inputs, **cat_inputs}
    input_df = pd.DataFrame([input_data])

# ===========================
# Prediction Section
# ===========================
st.markdown("---")
st.subheader("🔮 Attrition Prediction")

if not input_df.empty and st.button("Predict Attrition"):
    try:
        # Ensure all required columns exist
        all_features = numerical_cols + categorical_cols
        for col in all_features:
            if col not in input_df.columns:
                # Fill with neutral/default values
                if col in numerical_cols:
                    input_df[col] = 3  # midpoint for satisfaction/rating type features
                else:
                    input_df[col] = "Unknown"

        # Reorder columns
        input_df = input_df[all_features]

        # Predict
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1]

        if prediction[0] == 1:
            st.error(f"⚠️ High Risk of Attrition! (Probability: {prob:.2f})")
        else:
            st.success(f"✅ Low Risk of Attrition. (Probability: {prob:.2f})")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
