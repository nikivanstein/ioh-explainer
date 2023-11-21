"""Process the pickle file with all results (fixes nan in lambda_ setting etc.)"""
import pandas as pd
import numpy as np
import pandas as pd

dfall = pd.read_pickle("cma_results_cpp.pkl")

dfall = dfall.drop(columns=['Unnamed: 0'])  
dfall['lambda_'] = dfall['lambda_'].replace(np.nan,'nan')
dfall['mu'] = dfall['mu'].replace(np.nan,'nan')

#remove all mu > lambda
dfall.loc[(dfall['lambda_'] == 'nan') & (dfall['dim'] == 5),'lambda_'] = 8.0
dfall.loc[(dfall['lambda_'] == 'nan') & (dfall['dim'] == 30),'lambda_'] = 14.0
dfall.loc[dfall['mu'] == 'nan','mu'] = dfall.loc[dfall['mu'] == 'nan','lambda_'] // 2
dfall['mu'] = dfall['mu'].astype(float)

dfall = dfall[dfall['mu'] <= dfall['lambda_']]
dfall.describe()
dfall.to_pickle("cma_final_processed.pkl")
