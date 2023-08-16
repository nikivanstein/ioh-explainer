#automatically configure DE usinig either high level features, low level features, or standardized DOEs.
#first collect all the data
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



data_file = "mod_de_all.pkl" #read in modular DE data
df = pd.read_pickle(data_file)

features= ['mutation_base', 'mutation_reference',
       'mutation_n_comps', 'use_archive', 'crossover', 'adaptation_method',
       'lpsr']

config_dict = {}
for f in features:
    config_dict[f] = list(map(str, df[f].unique()))

config_dict['mutation_n_comps'] = [1,2]
config_dict['use_archive'] = [False, True]
config_dict['lpsr'] = [False, True]

#for each fid, iid get the best configuration  (mean?)
cs = ConfigurationSpace(config_dict)

print(cs)

de_explainer = explainer(None, 
                 cs , 
                 algname="mod-DE",
                 dims = [5,30],#, 10, 20, 40 
                 fids = np.arange(1,25), #,5
                 iids = df['iid'].unique(), #20 
                 reps = len( df['seed'].unique()), 
                 sampling_method = "grid",  #or random
                 grid_steps_dict = {},
                 sample_size = None,  #only used with random method
                 budget = 10000, #10000
                 seed = 1,
                 verbose = True)


de_explainer.load_results(data_file)

new_df = pd.DataFrame(columns=['dim','fid','iid',*features, 'auc'])
new_df_fidonly = pd.DataFrame(columns=['dim','fid','iid',*features, 'auc']) #don't care about instances
for dim in de_explainer.dims:
    dim_df = de_explainer.df[de_explainer.df['dim'] == dim].copy()
    for fid in de_explainer.fids:
        fid_df = dim_df[dim_df['fid'] == fid]

        conf, aucs = de_explainer._get_single_best(fid_df)
        conf['dim'] = dim
        conf['fid'] = fid
        conf['iid'] = iid
        conf['auc'] = aucs['auc'].mean()
        new_df_fidonly.loc[len(new_df_fidonly)] = conf

        for iid in fid_df['iid'].unique():
            iid_df = fid_df[fid_df['iid'] == iid]
            #get best performing conf
            conf, aucs = de_explainer._get_single_best(iid_df)
            conf['dim'] = dim
            conf['fid'] = fid
            conf['iid'] = iid
            conf['auc'] = aucs['auc'].mean()
            new_df.loc[len(new_df)] = conf
print(new_df_fidonly)

#now replace fid, iid with features instead, 
