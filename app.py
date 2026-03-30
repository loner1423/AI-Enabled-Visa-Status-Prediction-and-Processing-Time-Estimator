import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
country_avg = pickle.load(open("country_avg.pkl", "rb"))
visa_avg = pickle.load(open("visa_avg.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# OPTIONAL: load dataset for dashboard (if available)
try:
    df_dashboard = pd.read_csv("USVISA.csv")
    df_dashboard["case_received_date"] = pd.to_datetime(df_dashboard["case_received_date"], errors='coerce')
    df_dashboard["decision_date"] = pd.to_datetime(df_dashboard["decision_date"], errors='coerce')
    df_dashboard["processing_days"] = (df_dashboard["decision_date"] - df_dashboard["case_received_date"]).dt.days
    df_dashboard = df_dashboard.dropna(subset=["processing_days"])
except:
    df_dashboard = None

st.set_page_config(layout="wide")
st.title("🌍 AI Visa Processing Time Predictor")

st.markdown("""
### ✨ About This Project
This AI-powered system predicts visa processing time using historical data and machine learning.

🔍 **What it does:**
- Estimates processing time based on country, visa type, and application date
- Uses advanced ML model (XGBoost)
- Applies feature engineering like seasonality and historical averages

⚙️ **Key Features:**
- Real-time prediction
- Smart handling of unknown inputs
- Data-driven insights

📊 **Use Case:**
Helps applicants plan travel, job joining, and documentation timelines efficiently.
""")

st.divider()

# ================= DASHBOARD =================
st.subheader("📊 Insights Dashboard")

if df_dashboard is not None:
    col1, col2, col3 = st.columns(3)

    col1.metric("Avg Processing Days", round(df_dashboard["processing_days"].mean(), 2))
    col2.metric("Max Days", int(df_dashboard["processing_days"].max()))
    col3.metric("Min Days", int(df_dashboard["processing_days"].min()))

    colA, colB = st.columns(2)

    with colA:
        fig1, ax1 = plt.subplots()
        sns.histplot(df_dashboard["processing_days"], kde=True, ax=ax1)
        ax1.set_title("Processing Time Distribution")
        st.pyplot(fig1)

    with colB:
        top_countries = df_dashboard.groupby("foreign_worker_info_birth_country")["processing_days"].mean().sort_values().tail(10)
        fig2, ax2 = plt.subplots()
        top_countries.plot(kind="barh", ax=ax2)
        ax2.set_title("Top Countries by Avg Processing Time")
        st.pyplot(fig2)

    # Monthly trend
    df_dashboard["month"] = df_dashboard["case_received_date"].dt.month
    monthly_avg = df_dashboard.groupby("month")["processing_days"].mean()

    fig3, ax3 = plt.subplots()
    monthly_avg.plot(ax=ax3)
    ax3.set_title("Monthly Trend of Processing Time")
    st.pyplot(fig3)

else:
    st.warning("Dashboard data not available")

st.divider()

# ================= PREDICTION =================
st.subheader("🎯 Predict Processing Time")

country = st.selectbox("Country", list(encoders["foreign_worker_info_birth_country"].classes_))
visa = st.selectbox("Visa Type", list(encoders["class_of_admission"].classes_))
date = st.date_input("Application Date")


def safe_encode(col, value):
    le = encoders[col]
    if value not in le.classes_:
        return -1
    return le.transform([value])[0]


def preprocess(data):
    df = pd.DataFrame([data])

    df["application_date"] = pd.to_datetime(df["application_date"])
    df["application_month"] = df["application_date"].dt.month
    df["application_year"] = df["application_date"].dt.year
    df["processing_weekday"] = df["application_date"].dt.weekday

    df["season"] = df["application_month"].apply(lambda x: "Peak" if x in [1,2,12] else "Off-Peak")

    df["country_avg_processing"] = df["foreign_worker_info_birth_country"].map(country_avg).fillna(np.mean(list(country_avg.values())))
    df["visa_avg_processing"] = df["class_of_admission"].map(visa_avg).fillna(np.mean(list(visa_avg.values())))

    # Encoding
    df["foreign_worker_info_birth_country"] = df["foreign_worker_info_birth_country"].apply(lambda x: safe_encode("foreign_worker_info_birth_country", x))
    df["class_of_admission"] = df["class_of_admission"].apply(lambda x: safe_encode("class_of_admission", x))
    df["season"] = df["season"].apply(lambda x: safe_encode("season", x))

    df = df.select_dtypes(include=[np.number])

    # ALIGN FEATURES
    for col in features:
        if col not in df.columns:
            df[col] = 0

    df = df[features]

    df = scaler.transform(df)

    return df


def get_confidence(pred):
    # simple heuristic confidence score
    if pred < 30:
        return 0.9
    elif pred < 90:
        return 0.75
    else:
        return 0.6


if st.button("Predict"):
    try:
        data = {
            "foreign_worker_info_birth_country": country,
            "class_of_admission": visa,
            "application_date": str(date)
        }

        processed = preprocess(data)
        pred = model.predict(processed)[0]

        st.success(f"Estimated Processing Time: {round(pred,2)} days")

        # Confidence meter
        confidence = get_confidence(pred)
        st.progress(int(confidence * 100))
        st.info(f"Prediction Confidence: {int(confidence * 100)}%")

        # Range
        st.write(f"Estimated Range: {round(pred*0.9,2)} - {round(pred*1.1,2)} days")

    except Exception as e:
        st.error(f"Error: {str(e)}")
