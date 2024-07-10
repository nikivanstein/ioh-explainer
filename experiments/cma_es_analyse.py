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

# df = cmaes_explainer.performance_stats()
# just for the image coloring we will change lambda from 200 to 30 and mu from 100 to 30 (used for the paper)
# cmaes_explainer.df.loc[cmaes_explainer.df["lambda_"] == 200, "lambda_"] =30
# cmaes_explainer.df.loc[cmaes_explainer.df["mu"] == 100, "mu"] =30
# cmaes_explainer.to_latex_report(False,True,False,False, filename=None, img_dir="../output/cma_img_new/")
cmaes_explainer.to_latex_report(
    filename="../output/cma_es-report3", img_dir="../output/cma_img_new2/"
)
