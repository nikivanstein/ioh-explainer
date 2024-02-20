"""Analyse the Modular DE results from the pickle file using ioh-xplain
"""

from config import de_explainer

data_file = "de_final_processed.pkl"
de_explainer.load_results(data_file)

# auc Large
de_explainer.df.loc[de_explainer.df["dim"] == 30, "auc"] = de_explainer.df.loc[
    de_explainer.df["dim"] == 30, "aucLarge"
]

# used in the paper to have better distinquishable colors in the shap summary plots for mu and lambda.
# de_explainer.df.loc[de_explainer.df["lambda_"] == 300, "lambda_"] =30
# de_explainer.df.loc[de_explainer.df["lambda_"] == 60, "lambda_"] = 25
# de_explainer.df.loc[de_explainer.df["lambda_"] == 50, "lambda_"] = 20
# de_explainer.to_latex_report(False,True,False,False, filename=None, img_dir="../output/de_img_new/")

de_explainer.to_latex_report(
    filename="../output/mod_de_new", img_dir="../output/de_img_new/"
)
