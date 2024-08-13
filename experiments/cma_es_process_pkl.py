"""Process the pickle file with all results (fixes nan in lambda_ setting etc.)"""
import numpy as np
import pandas as pd


dfall = pd.read_pickle("cma_final30d.pkl")
print(len(dfall))

df5 = pd.read_pickle("cma_final_processed.pkl")
print(len(df5))

dfall = dfall.drop(columns=["Unnamed: 0"])
dfall["lambda_"] = dfall["lambda_"].replace(np.nan, "nan")
dfall["mu"] = dfall["mu"].replace(np.nan, "nan")

dfall = dfall.fillna("nan")

# remove all mu > lambda
#dfall.loc[(dfall["lambda_"] == "nan") & (dfall["dim"] == 5), "lambda_"] = 8.0
dfall.loc[(dfall["lambda_"] == "nan") & (dfall["dim"] == 30), "lambda_"] = 14.0
dfall.loc[dfall["mu"] == "nan", "mu"] = dfall.loc[dfall["mu"] == "nan", "lambda_"] // 2
dfall["mu"] = dfall["mu"].astype(float)

dfall = dfall[dfall["mu"] <= dfall["lambda_"]]
dfall = (
    dfall.groupby(
        [
            "fid",
            "iid",
            "dim",
            "seed",
            "active",
            "base_sampler",
            "covariance",
            "elitist",
            "lambda_",
            "local_restart",
            "mirrored",
            "mu",
            "step_size_adaptation",
            "weights_option",
        ]
    )
    .agg("mean")
    .reset_index()
)

#i = df5[(df5[df5["dim"] == 30)].index
df5.loc[df5["dim"] == 30] = dfall.loc[:]

df5.describe()
df5.to_pickle("cma_final_processed_new.pkl")
print(len(df5))