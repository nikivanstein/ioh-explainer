import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from ioh_xplainer import explainer
import pandas as pd

from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from IPython.display import display


dfall = pd.read_pickle("cma_results_cpp.pkl")
dfall = dfall.drop(columns=['Unnamed: 0'])  
dfall['lambda_'] = dfall['lambda_'].replace(np.nan,'nan')
dfall['mu'] = dfall['mu'].replace(np.nan,'nan')

#remove all mu > lambda
dfall.loc[(dfall['lambda_'] == 'nan') & (dfall['dim'] == 5),'lambda_'] = 6.0
dfall.loc[(dfall['lambda_'] == 'nan') & (dfall['dim'] == 30),'lambda_'] = 8.0
dfall.loc[dfall['mu'] == 'nan','mu'] = dfall.loc[dfall['mu'] == 'nan','lambda_'] // 2
dfall['mu'] = dfall['mu'].astype(float)

dfall = dfall[dfall['mu'] <= dfall['lambda_']]

#print(dfall['mu'])
dfall.describe()

dfall.to_pickle("cma_final.pkl")
#print(dfall['lambda_'].describe())
