
# ====================== MILESTONE 1 DATA COLLECTION & PREPROCESSING
import pandas as pd              # For data manipulation
import numpy as np               # For numerical operations
import matplotlib.pyplot as plt  # For visualization
import seaborn as sns            # For statistical plots
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor
# 1. Load the Dataset

df = pd.read_csv("hello.csv")  # Load original dataset
print("Initial Dataset Shape:", df.shape)
print("\nInitial Missing Values:\n", df.isnull().sum())


# 2. Convert Date Columns
# Convert string dates to datetime format
# errors='coerce' converts invalid values into NaT
df["case_received_date"] = pd.to_datetime(df["case_received_date"], errors='coerce')
df["decision_date"] = pd.to_datetime(df["decision_date"], errors='coerce')
# 3. Handle Missing Values
# Fill missing education values with 'Unknown'
df["foreign_worker_info_education"].fillna("Unknown", inplace=True)

# Fill missing categorical values if present
categorical_columns = ["case_status","employer_country"]
for col in categorical_columns:
    if col in df.columns:
        df[col].fillna("Unknown", inplace=True)


# 4. Generate Target Variable
# Calculate number of days between submission and decision
if "processing_days" not in df.columns:
    df["processing_days"] = (df["decision_date"] - df["case_received_date"]).dt.days

# Remove negative processing times (invalid records)
df = df[df["processing_days"] >= 0]


print("\nShape Before Outlier Removal:", df.shape)


# 5. Remove Outliers (IQR Method)
# IQR is suitable for skewed real-world data

Q1 = df["processing_days"].quantile(0.25)  # 25th percentile
Q3 = df["processing_days"].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1

# Define acceptable range
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Keep only records within range
df = df[
    (df["processing_days"] >= lower_bound) &
    (df["processing_days"] <= upper_bound)
]

print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Shape After Outlier Removal:", df.shape)


# 6. Save Cleaned Dataset

df.fillna(df.mode().iloc[0], inplace=True)
df.to_csv("cleaned_visa_dataset.csv", index=False)

#============== Milestone 1 Completed ======================

# ====== MILESTONE 2 EDA & FEATURE ENGINEERING =============================
import os  # To create folders
# Create folder to store graphs
output_folder = "visualizations"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 1. Basic Statistical Summary

print("\nProcessing Days Summary Statistics:")
print(df["processing_days"].describe())


# 2. Distribution Plot

plt.figure(figsize=(8,5))
sns.histplot(df["processing_days"], kde=True)
plt.title("Distribution of Visa Processing Days")
plt.xlabel("Processing Days")
plt.ylabel("Frequency")
plt.show()
plt.savefig(os.path.join(output_folder, "distribution_processing_days.png"), dpi=300)
plt.close()


# 3. Boxplot for Outlier Verification

plt.figure(figsize=(8,4))
sns.boxplot(x=df["processing_days"])
plt.title("Boxplot After Outlier Removal")
plt.show()
plt.savefig(os.path.join(output_folder, "Boxplot After Outlier Removal.png"), dpi=300)
plt.close()


# 4. Feature Engineering

# Feature 1: Application Month
df["application_month"] = df["case_received_date"].dt.month


# Feature 2: Seasonal Index
# Peak season assumed: Jan, Feb, Dec

df["season"] = df["application_month"].apply(
    lambda x: "Peak" if x in [1, 2, 12] else "Off-Peak"
)


# Feature 3: Country-Specific Average Processing Time
df["application_month"] = df["case_received_date"].dt.month
df["application_year"] = df["case_received_date"].dt.year
df["processing_weekday"] = df["case_received_date"].dt.weekday
country_avg = df.groupby(
    "foreign_worker_info_birth_country"
)["processing_days"].mean()

df["country_avg_processing"] = df[
    "foreign_worker_info_birth_country"
].map(country_avg)


# Feature 4: Visa-Type Average Processing Time
visa_avg = df.groupby(
    "class_of_admission"
)["processing_days"].mean()

df["visa_avg_processing"] = df[
    "class_of_admission"
].map(visa_avg)


# 5. Correlation Analysis

corr_matrix = df[[
    "processing_days",
    "application_month",
    "country_avg_processing",
    "visa_avg_processing"
]].corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
plt.savefig(os.path.join(output_folder, "correlation_heatmap.png"), dpi=300)
plt.close()


plt.figure()
plt.hist(df["processing_days"], bins=5)
plt.title("Distribution of Visa Processing Days")
plt.xlabel("Processing Days")
plt.ylabel("Number of Applications")
plt.show()
plt.savefig(os.path.join(output_folder, "distribution_processing_days.png"), dpi=300)
plt.close()

country_avg = df.groupby("foreign_worker_info_birth_country")["processing_days"].mean()
plt.figure()
plt.bar(country_avg.index, country_avg.values)
plt.title("Average Processing Days by Country")
plt.xlabel("Country")
plt.ylabel("Average Processing Days")
plt.show()
plt.savefig(os.path.join(output_folder, "average_processing_days_by_country.png"), dpi=300)
plt.close()

visa_avg = df.groupby("class_of_admission")["processing_days"].mean()
plt.figure()
plt.bar(visa_avg.index, visa_avg.values)
plt.title("Average Processing Days by Visa Type")
plt.xlabel("Visa Type")
plt.ylabel("Average Processing Days")
plt.show()
plt.savefig(os.path.join(output_folder, "average_processing_days_by_visa_type.png"), dpi=300)
plt.close()

plt.figure()
plt.scatter(df["application_month"], df["processing_days"])
plt.title("Processing Days vs Application Month")
plt.xlabel("Application Month")
plt.ylabel("Processing Days")
plt.show()
plt.savefig(os.path.join(output_folder, "processing_days_vs_application_month.png"), dpi=300)
plt.close()

monthly_avg = df.groupby("application_month")["processing_days"].mean()
plt.figure()
plt.plot(monthly_avg.index, monthly_avg.values)
plt.title("Monthly Trend of Processing Days")
plt.xlabel("Month")
plt.ylabel("Average Processing Days")
plt.show()
plt.savefig(os.path.join(output_folder, "monthly_trend_processing_days.png"), dpi=300)
plt.close()


# 6. Save Final Feature Dataset


df.to_csv("final_feature_engineered_dataset.csv", index=False)
print("Final dataset with engineered features saved as 'final_feature_engineered_dataset.csv'.")
# ============= Milestone 2 Completed ==============
# ============ ENCODING CATEGORICAL VARIABLES ===============================================

encoder = LabelEncoder()
categorical_cols = [
    "class_of_admission",
    "foreign_worker_info_birth_country",
    "employer_country",
    "season"
    "case_status"
]
for col in categorical_cols:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])

# Remove date columns
df = df.drop(["case_received_date","decision_date"], axis=1, errors="ignore")

# Keep only numeric columns
df = df.select_dtypes(include=[np.number])


# ================ TRAIN TEST SPLIT ====================================
X = df.drop("processing_days", axis=1)
y = df["processing_days"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# ============== FEATURE SCALING ========================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

print("\nModel Performance")
model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# ========== HYPERPARAMETER TUNING ==============================================

print("\nHyperparameter Tuning")
param_grid = {
    "n_estimators":[200,300],
    "max_depth":[5,6,8],
    "learning_rate":[0.01,0.05,0.1]
}

grid = GridSearchCV(XGBRegressor(), param_grid, cv=5)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# ======SAVE MODEL===================================================
with open("visa_processing_model.pkl","wb") as f:
    pickle.dump(best_model,f)

print("\nModel Saved Successfully")
