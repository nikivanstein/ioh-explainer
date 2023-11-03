"""Process the DE pkl file (fixing mu and lambda)"""

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

data_file = "de_final.pkl"
df = pd.read_pickle(data_file)

df = df.drop(columns=['Unnamed: 0'])  


#replacing stuff to fix
df['mutation_reference'] = df['mutation_reference'].replace(np.nan,'nan')
df['adaptation_method'] = df['adaptation_method'].replace(np.nan,'nan')
df['lambda_'] = df['lambda_'].replace(np.nan,'nan')
df['lambda_'] = df['lambda_'].replace('2', 2.0)
df['lambda_'] = df['lambda_'].replace('10', 10.0)

df.loc[(df['lambda_'] == 10.0) & (df['dim'] == 30),'lambda_'] = 300
df.loc[(df['lambda_'] == 10.0) & (df['dim'] == 5),'lambda_'] = 50
df.loc[(df['lambda_'] == 2.0) & (df['dim'] == 30),'lambda_'] = 60
df.loc[(df['lambda_'] == 2.0) & (df['dim'] == 5),'lambda_'] = 10
df.loc[(df['lambda_'] == 'nan') & (df['dim'] == 5),'lambda_'] = 8
df.loc[(df['lambda_'] == 'nan') & (df['dim'] == 30),'lambda_'] = 14

df.to_pickle("de_final_processed.pkl")
print(df['lambda_'].describe())