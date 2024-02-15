"""
Processes the CMA-ES data into images and tables using the ioh-xplain package.
"""
import pandas as pd

from config import cmaes_explainer

data_file = "cma_final_processed.pkl"
df = pd.read_pickle(data_file)

cmaes_explainer.load_results(data_file)
# use aucLarge for D30
cmaes_explainer.df.loc[cmaes_explainer.df["dim"] == 30, "auc"] = cmaes_explainer.df.loc[
    cmaes_explainer.df["dim"] == 30, "aucLarge"
]

df = cmaes_explainer.performance_stats()
cmaes_explainer.to_latex_report(filename="../output/cma_es-report", img_dir="../output/cma_img_new/")
