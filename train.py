import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv("USVISA.csv")

# Date processing
df["case_received_date"] = pd.to_datetime(df["case_received_date"], errors='coerce')
df["decision_date"] = pd.to_datetime(df["decision_date"], errors='coerce')

# Target
df["processing_days"] = (df["decision_date"] - df["case_received_date"]).dt.days

df = df[(df["processing_days"] >= 0) & (df["processing_days"] <= 554)]

# Features
df["application_month"] = df["case_received_date"].dt.month
df["application_year"] = df["case_received_date"].dt.year
df["processing_weekday"] = df["case_received_date"].dt.weekday

df["season"] = df["application_month"].apply(lambda x: "Peak" if x in [1,2,12] else "Off-Peak")

# Aggregations
country_avg = df.groupby("foreign_worker_info_birth_country")["processing_days"].mean()
visa_avg = df.groupby("class_of_admission")["processing_days"].mean()

df["country_avg_processing"] = df["foreign_worker_info_birth_country"].map(country_avg)
df["visa_avg_processing"] = df["class_of_admission"].map(visa_avg)

# Fill NA safely
df.fillna(0, inplace=True)

# Encoding
encoders = {}
for col in ["class_of_admission", "foreign_worker_info_birth_country", "season"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# Prepare data
X = df.drop(["processing_days", "case_received_date", "decision_date"], axis=1, errors='ignore')
y = df["processing_days"]

X = X.select_dtypes(include=[np.number])

# SAVE FEATURE COLUMNS (CRITICAL)
feature_columns = X.columns.tolist()
pickle.dump(feature_columns, open("features.pkl", "wb"))

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6)
model.fit(X_train, y_train)

# Save everything
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(encoders, open("encoders.pkl", "wb"))
pickle.dump(country_avg.to_dict(), open("country_avg.pkl", "wb"))
pickle.dump(visa_avg.to_dict(), open("visa_avg.pkl", "wb"))

