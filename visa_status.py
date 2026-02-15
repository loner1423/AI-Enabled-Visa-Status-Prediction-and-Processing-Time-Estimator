#import packages

import csv
import numpy as np
import pandas as pd # used to store and manipulate the data in table form
from datetime import datetime # helps to convert the date strings into the real date objects

df = pd.read_csv("VISASTATUSDATASET.csv")

print("Original DataFrame with Missing Values:\n", df)

miss= df.isnull().sum()

# isnull-> finds the missing values
# sum() -> counts them in colm-wise
print("\nMissing values count:\n", miss)

# mode()[0]-> returns the most frequ date in the colm
# # fillna()-> replaces the missing values inside the df
df["RECEIVED_DATE"].fillna(df["RECEIVED_DATE"].mode()[0], inplace=True)
df["DECISION_DATE"].fillna(df["DECISION_DATE"].mode()[0], inplace=True)

df["CASE_STATUS"].fillna("Unknown", inplace=True)
df["EMPLOYER_STATE"].fillna("Unknown", inplace=True)

df["RECEIVED_DATE"] = pd.to_datetime(df["RECEIVED_DATE"])
df["DECISION_DATE"] = pd.to_datetime(df["DECISION_DATE"])

print(df)

df["processing_days"] = (df["DECISION_DATE"] - df["RECEIVED_DATE"]).dt.days
df["processing_days"].fillna(df["processing_days"].mean(),inplace=True)
print("\nAfter calculating processing days:\n", df)

df_encoded = pd.get_dummies(df, columns=["CASE_STATUS","VISA_CLASS","EMPLOYER_STATE"])
print("\nEncoded DataFrame:\n", df_encoded)


df.fillna(df.mode().iloc[0], inplace=True)

miss= df.isnull().sum()
print("\nMissing values count:\n", miss)
df.to_csv("AIVISASTATUSDATASET.csv",index=False)
# converts the text to numeric columns
# one-hot encoding - as the ML model requires the numeric data

