import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# 1. Page Configuration
st.set_page_config(page_title="Exam Score Predictor", page_icon="🎓", layout="centered")

st.title("🎓 Smart Student Exam Score Predictor")
st.markdown("Predict student final exam scores based on weekly study hours using **Simple Linear Regression**.")
st.markdown("---")

# 2. Load the Saved Model safely
@st.cache_resource
def load_model():
    return joblib.load("exam_score_model.pkl")

model = load_model()

# 3. Historical Data for Graph
data = {
    "Study_Hours": [1.5, 2.0, 3.0, 3.5, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0],
    "Exam_Score": [48, 52, 58, 63, 70, 74, 81, 86, 92, 97]
}
df = pd.DataFrame(data)

# 4. Sidebar Controls for User Input
study_hours = st.sidebar.slider("Weekly Study Hours:", min_value=1.0, max_value=12.0, value=6.5, step=0.5)

# 5. Prediction Logic
if st.button("🚀 Calculate Estimated Score"):
    input_df = pd.DataFrame([[study_hours]], columns=["Study_Hours"])
    prediction = model.predict(input_df)[0]
    
    # Cap prediction between 0 and 100 for realistic score display
    prediction = max(0.0, min(100.0, prediction))
    
    st.success("Prediction Completed Successfully!")
    st.metric(label="Estimated Final Exam Score", value=f"{prediction:.2f}%")

    # 6. Regression Plot
    st.subheader("📈 Regression Analysis Graph")
    X_range = np.linspace(1, 10, 100).reshape(-1, 1)
    X_range_df = pd.DataFrame(X_range, columns=["Study_Hours"])
    y_range = model.predict(X_range_df)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(df["Study_Hours"], df["Exam_Score"], color="#2980b9", s=60, label="Historical Data")
    ax.plot(X_range, y_range, color="#e74c3c", linewidth=2, label="Regression Line")
    ax.scatter([study_hours], [prediction], color="#27ae60", s=180, marker="*", label="Your Prediction")
    ax.set_xlabel("Weekly Study Hours")
    ax.set_ylabel("Final Exam Score (%)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    st.pyplot(fig)