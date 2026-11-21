
import streamlit as st
import pandas as pd
import joblib
import json
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL, DATASET AND METRICS
# --------------------------------------------------

MODEL_PATH = "models/student_model.pkl"
DATA_PATH = "data/student_performance.csv"
METRICS_PATH = "models/metrics.json"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please run train_model.py first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found. Please run train_model.py first.")
    st.stop()

if not os.path.exists(METRICS_PATH):
    st.error("Metrics file not found. Please run train_model.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

with open(METRICS_PATH, "r") as file:
    metrics = json.load(file)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("🎓 Student Performance")
st.sidebar.write("Prediction System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Performance",
        "Prediction History",
        "Dataset"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application predicts student performance "
    "using a Machine Learning model."
)

# ==================================================
# 1. DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("🎓 Student Performance Dashboard")

    st.write(
        "Welcome to the Student Performance Prediction System. "
        "Monitor student records and evaluate the trained ML model."
    )

    # ----------------------------------------------
    # DATASET OVERVIEW
    # ----------------------------------------------

    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Students",
            len(data)
        )

    with col2:
        st.metric(
            "Average Final Score",
            round(data["final_score"].mean(), 2)
        )

    with col3:
        st.metric(
            "Average Attendance",
            f"{data['attendance'].mean():.2f}%"
        )

    st.markdown("---")

    # ----------------------------------------------
    # MODEL EVALUATION
    # ----------------------------------------------

    st.subheader("📈 Model Evaluation")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "MAE",
            round(metrics["MAE"], 2)
        )

    with col2:
        st.metric(
            "RMSE",
            round(metrics["RMSE"], 2)
        )

    with col3:
        st.metric(
            "Testing R²",
            round(metrics["R2 Score"], 4)
        )

    with col4:
        st.metric(
            "Training R²",
            round(metrics["Training R2"], 4)
        )

    # ----------------------------------------------
    # MODEL OPTIMIZATION RESULTS
    # ----------------------------------------------

    st.subheader("🔍 Model Optimization Results")

    col5, col6 = st.columns(2)

    with col5:
        st.metric(
            "Cross-validation MAE",
            round(metrics["Cross Validation MAE"], 2)
        )

    with col6:
        st.write("**Best Model Parameters**")
        st.json(metrics["Best Parameters"])

    st.markdown("---")

    # ----------------------------------------------
    # STUDENT PERFORMANCE CHART
    # ----------------------------------------------

    st.subheader("📊 Student Performance Distribution")

    st.bar_chart(
        data.head(20).set_index(data.head(20).index)["final_score"]
    )

    # ----------------------------------------------
    # STUDY HOURS VS FINAL SCORE
    # ----------------------------------------------

    st.subheader("📚 Study Hours vs Final Score")

    st.scatter_chart(
        data,
        x="study_hours",
        y="final_score"
    )

    # ----------------------------------------------
    # ATTENDANCE VS FINAL SCORE
    # ----------------------------------------------

    st.subheader("📅 Attendance vs Final Score")

    st.scatter_chart(
        data,
        x="attendance",
        y="final_score"
    )


# ==================================================
# 2. PREDICT PERFORMANCE
# ==================================================

elif page == "Predict Performance":

    st.title("🔮 Predict Student Performance")

    st.write(
        "Enter student details below to predict the expected final score."
    )

    col1, col2 = st.columns(2)

    with col1:

        study_hours = st.slider(
            "Study Hours Per Day",
            min_value=1,
            max_value=10,
            value=5
        )

        attendance = st.slider(
            "Attendance Percentage",
            min_value=40,
            max_value=100,
            value=75
        )

    with col2:

        previous_score = st.slider(
            "Previous Exam Score",
            min_value=30,
            max_value=100,
            value=70
        )

        assignments_completed = st.slider(
            "Assignments Completed",
            min_value=1,
            max_value=10,
            value=7
        )

    st.markdown("---")

    if st.button("Predict Performance", type="primary"):

        input_data = pd.DataFrame(
            [[
                study_hours,
                attendance,
                previous_score,
                assignments_completed
            ]],
            columns=[
                "study_hours",
                "attendance",
                "previous_score",
                "assignments_completed"
            ]
        )

        prediction = model.predict(input_data)[0]

        prediction = max(0, min(100, prediction))

        # ------------------------------------------
        # PERFORMANCE CATEGORY
        # ------------------------------------------

        if prediction >= 75:
            category = "Excellent"
            message = "Excellent performance! Keep it up."

        elif prediction >= 50:
            category = "Average"
            message = "Good effort! There is room for improvement."

        else:
            category = "Needs Improvement"
            message = "More focus and practice are recommended."

        # ------------------------------------------
        # DISPLAY RESULT
        # ------------------------------------------

        st.subheader("📊 Prediction Result")

        st.metric(
            "Predicted Final Score",
            f"{prediction:.2f} / 100"
        )

        st.progress(int(prediction))

        st.success(f"Performance Category: {category}")

        st.info(message)

        # ------------------------------------------
        # SAVE PREDICTION IN SESSION HISTORY
        # ------------------------------------------

        record = {
            "Study Hours": study_hours,
            "Attendance": attendance,
            "Previous Score": previous_score,
            "Assignments": assignments_completed,
            "Predicted Score": round(prediction, 2),
            "Category": category
        }

        st.session_state.history.append(record)


# ==================================================
# 3. PREDICTION HISTORY
# ==================================================

elif page == "Prediction History":

    st.title("📋 Prediction History")

    st.write(
        "View the predictions made during the current application session."
    )

    if len(st.session_state.history) > 0:

        history_df = pd.DataFrame(st.session_state.history)

        st.subheader("Previous Predictions")

        st.dataframe(
            history_df,
            use_container_width=True
        )

        st.subheader("📊 Prediction Score Chart")

        st.bar_chart(
            history_df.set_index(history_df.index)["Predicted Score"]
        )

        if st.button("Clear Prediction History"):

            st.session_state.history = []

            st.rerun()

    else:

        st.info("No prediction history available yet.")

        st.write(
            "Go to the Predict Performance page and make a prediction first."
        )


# ==================================================
# 4. DATASET
# ==================================================

elif page == "Dataset":

    st.title("📁 Student Performance Dataset")

    st.write(
        "Explore the student records used to train and evaluate the model."
    )

    st.subheader("Dataset Records")

    st.dataframe(
        data,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("📊 Dataset Statistics")

    st.dataframe(
        data.describe(),
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Total Records:**", len(data))

    with col2:
        st.write("**Total Columns:**", len(data.columns))


# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Student Performance Prediction System | "
    "Developed using Python, Streamlit and Machine Learning"
)
