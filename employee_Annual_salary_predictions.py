
import streamlit as st
import pandas as pd
from xgboost import XGBRegressor

st.set_page_config(page_title="Salary Predictor")

@st.cache_resource
def load_model():
    model = XGBRegressor()
    model.load_model("xgb_model.json")
    return model

model = load_model()

st.title("Employee Salary Predictor")
st.write("Enter employee details to predict salary.")

# Simple inputs
age = st.number_input("Age", 18, 65, 28)

experience = st.number_input(
    "Experience (Years)", 0.0, 33.0, 6.0
)

education = st.selectbox(
    "Education",
    ["Master", "Bachelor", "Diploma", "PhD"]
)

job_level = st.selectbox(
    "Job Level",
    ["Junior", "Lead", "Mid", "Senior", "Manager"]
)

skill_score = st.slider(
    "Skill Score", 50, 100, 75
)

# Predict salary
if st.button("Predict Salary"):

    input_df = pd.DataFrame([{
        "Age": float(age),
        "Gender": 0,
        "Education": {
            "Master": 0,
            "Bachelor": 1,
            "Diploma": 2,
            "PhD": 3
        }[education],
        "Experience_Years": float(experience),
        "Department": 3,
        "Job_Level": {
            "Junior": 0,
            "Lead": 1,
            "Mid": 2,
            "Senior": 3,
            "Manager": 4
        }[job_level],
        "Performance_Rating": 3,
        "Certifications": 4,
        "Overtime_Hours": 15,
        "Remote_Work": 0,
        "Company_Tenure": 6,
        "Projects_Completed": 15,
        "Skill_Score": float(skill_score)
    }])

    salary = model.predict(input_df)[0]

    st.success(f"Predicted Salary: {salary:.2f} LPA")
