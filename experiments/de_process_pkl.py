"""Process the DE pkl file (fixing mu and lambda)"""

import numpy as np
import pandas as pd


df = pd.read_pickle("de_final_30d.pkl")
print(len(df))

df5 = pd.read_pickle("de_final.pkl")
print(len(df5))

df = df.drop(columns=["Unnamed: 0"])


# replacing stuff to fix
df["mutation_reference"] = df["mutation_reference"].replace(np.nan, "nan")
df["adaptation_method"] = df["adaptation_method"].replace(np.nan, "nan")
df["lambda_"] = df["lambda_"].replace(np.nan, "nan")
df["lambda_"] = df["lambda_"].replace("2", 2.0)
df["lambda_"] = df["lambda_"].replace("10", 10.0)

df.loc[(df["lambda_"] == 10.0) & (df["dim"] == 30), "lambda_"] = 300
df.loc[(df["lambda_"] == 10.0) & (df["dim"] == 5), "lambda_"] = 50
df.loc[(df["lambda_"] == 2.0) & (df["dim"] == 30), "lambda_"] = 60
df.loc[(df["lambda_"] == 2.0) & (df["dim"] == 5), "lambda_"] = 10
df.loc[(df["lambda_"] == "nan") & (df["dim"] == 5), "lambda_"] = 8
df.loc[(df["lambda_"] == "nan") & (df["dim"] == 30), "lambda_"] = 14


df5.loc[df5["dim"] == 30] = df.loc[:]

df5.to_pickle("de_final_processed_new.pkl")
print(df["lambda_"].describe())
