"""Analyse the Modular DE results from the pickle file using ioh-xplain
"""

from config import de_explainer

data_file = "de_final_processed.pkl"
de_explainer.load_results(data_file)

#auc Large
de_explainer.df.loc[de_explainer.df["dim"] == 30,'auc'] = de_explainer.df.loc[de_explainer.df["dim"] == 30,'aucLarge']
de_explainer.to_latex_report(filename="mod_de_new", img_dir="de_img_new/")