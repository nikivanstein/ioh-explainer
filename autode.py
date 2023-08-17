#automatically configure DE usinig either high level features, low level features, or standardized DOEs.


## different models we can make:
# ELA/DOE + config --> AUC
# ELA ---> config
# high-level --> config
# DOE --> config

#first collect all the data
import os
import pandas as pd
import re
import numpy as np
from tqdm import tqdm
from ioh_xplainer import explainer
import pandas as pd
import ioh
from scipy.stats import qmc
from ConfigSpace import ConfigurationSpace
from ConfigSpace.util import generate_grid
from IPython.display import display

from pflacco.sampling import create_initial_sample

from pflacco.classical_ela_features import calculate_ela_distribution, calculate_ela_meta, calculate_nbc, calculate_dispersion, calculate_information_content, calculate_ela_level, calculate_cm_grad
from pflacco.misc_features import calculate_fitness_distance_correlation
from pflacco.local_optima_network_features import compute_local_optima_network, calculate_lon_features



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

sample_size = 1000 #fixed

new_doe_df = pd.DataFrame(columns=['dim','fid','iid', 'DOE',*features, 'auc'])
new_ela_df = pd.DataFrame(columns=['dim','fid','iid',*features, 'auc'])
new_df_fidonly = pd.DataFrame(columns=['dim','fid','multimodal', 'global structure', 'funnel',*features, 'auc']) #don't care about instances
for dim in de_explainer.dims:
    dim_df = de_explainer.df[de_explainer.df['dim'] == dim].copy()
    
    X = create_initial_sample(dim, lower_bound = -5, upper_bound = 5, n=sample_size, seed=42)
    

    for fid in tqdm(de_explainer.fids):
        fid_df = dim_df[dim_df['fid'] == fid]

        conf, aucs = de_explainer._get_single_best(fid_df)
        conf['dim'] = dim
        f = fid
        conf['auc'] = aucs['auc'].mean()
        #add high level features
        if f in [1,2,5,6,7,10,11,12,13,14]: #verify!!
            conf['multimodal'] = 0
            conf['global structure'] = 0
            conf['funnel'] = 1
        elif f in [3,4]:
            conf['multimodal'] = 2
            conf['global structure'] = 3
            conf['funnel'] = 1
        elif f in [8,9]:
            conf['multimodal'] = 1
            conf['global structure'] = 0
            conf['funnel'] = 1
        elif f in [15,19]:
            conf['multimodal'] = 2
            conf['global structure'] = 3
            conf['funnel'] = 1
        elif f in [16]:
            conf['multimodal'] = 2
            conf['global structure'] = 2
            conf['funnel'] = 0
        elif f in [17,18]:
            conf['multimodal'] = 2
            conf['global structure'] = 2
            conf['funnel'] = 1
        elif f in [20]:
            conf['multimodal'] = 2
            conf['global structure'] = -1
            conf['funnel'] = 1
        elif f in [21]:
            conf['multimodal'] = 2
            conf['global structure'] = 0
            conf['funnel'] = 0
        elif f in [22]:
            conf['multimodal'] = 1
            conf['global structure'] = 0
            conf['funnel'] = 0
        elif f in [23]:
            conf['multimodal'] = 3
            conf['global structure'] = 0
            conf['funnel'] = 0
        elif f in [24]:
            conf['multimodal'] = 3
            conf['global structure'] = 1
            conf['funnel'] = 1
        new_df_fidonly.loc[len(new_df_fidonly)] = conf

        for iid in fid_df['iid'].unique():
            iid_df = fid_df[fid_df['iid'] == iid]
            #get best performing conf
            conf, aucs = de_explainer._get_single_best(iid_df)
            conf['dim'] = dim
            conf['fid'] = fid
            conf['iid'] = iid
            conf['auc'] = aucs['auc'].mean()

            #ELA
            

            func = ioh.get_problem(fid, dimension=dim, instance=iid)
            y = X.apply(lambda x: func(x), axis = 1)
            y = (y - np.min(y)) / (np.max(y)  - np.min(y))

            #doe = (y.flatten() - np.min(y)) / (
            #    np.max(y) - np.min(y)
            #)
            conf2 = conf.copy()
            conf.update(y)
            new_doe_df.loc[len(new_doe_df)] = conf

            conf2.update(calculate_ela_meta(X, y))
            conf2.update(calculate_ela_distribution(X, y))
            conf2.update(calculate_ela_level(X, y))
            conf2.update(calculate_nbc(X, y))
            conf2.update(calculate_dispersion(X, y))
            conf2.update(calculate_information_content(X, y, seed = 100))

            #all dictionairies! yeaa
            new_ela_df.loc[len(new_ela_df)] = conf2
print(new_df_fidonly)
print(new_ela_df)
print(new_doe_df)
#now replace fid, iid with features instead, 
#build multiple decision trees .. visualise -- multi-output tree vs single output trees

new_df_fidonly.to_pickle("highlevel-features.pkl")

new_ela_df.to_pickle("ela-features.pkl")

new_doe_df.to_pickle("doe-features.pkl")