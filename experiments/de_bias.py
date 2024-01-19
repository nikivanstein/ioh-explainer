"""Check for the hall of fame if they have any structural bias.
"""

from config import de_explainer

data_file = "de_final_processed.pkl"  # read in modular DE data
de_explainer.load_results(data_file)
# use aucLarge for D30
de_explainer.df.loc[de_explainer.df["dim"] == 30, "auc"] = de_explainer.df.loc[
    de_explainer.df["dim"] == 30, "aucLarge"
]

hall_of_fame = de_explainer.analyse_best(
    "../output/de-hall-of-fame.tex",
    False,
    "../output/bias_plots/",
    True,
    "/data/neocortex/de_data/",
    10,
)
print(hall_of_fame)
