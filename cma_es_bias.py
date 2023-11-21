"""Check for the hall of fame if they have any structural bias.
"""

import pandas as pd
from config import cmaes_explainer

data_file = "cma_final_processed.pkl"
df = pd.read_pickle(data_file)

cmaes_explainer.load_results(data_file)
#use aucLarge for D30
cmaes_explainer.df.loc[cmaes_explainer.df["dim"] == 30,'auc'] = cmaes_explainer.df.loc[cmaes_explainer.df["dim"] == 30,'aucLarge']

hall_of_fame = cmaes_explainer.analyse_best("cma_es-hall-of-fame.tex", True, "bias_plots/", True, "/data/neocortex/cma_data/", 10)
print(hall_of_fame)