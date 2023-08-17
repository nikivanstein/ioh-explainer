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

m = 8 #(2**m)

new_df = pd.DataFrame(columns=['dim','fid','iid', 'DOE',*features, 'auc'])
new_df_fidonly = pd.DataFrame(columns=['dim','fid','multimodal', 'global structure', 'funnel',*features, 'auc']) #don't care about instances
for dim in de_explainer.dims:
    dim_df = de_explainer.df[de_explainer.df['dim'] == dim].copy()
    
    sampler = qmc.Sobol(d=dim, scramble=False, seed=42)
    sample = sampler.random_base2(m=m)
    sample = sample * 10 - 5
    for fid in de_explainer.fids:
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
            bbob_y = np.asarray(list(map(func, sample)))
            doe = (bbob_y.flatten() - np.min(bbob_y)) / (
                np.max(bbob_y) - np.min(bbob_y)
            )
            conf['DOE'] = doe

            X = sample
            y = bbob_y
            conf.update(calculate_ela_meta(X, y))
            conf.update(calculate_ela_distribution(X, y))
            conf.update(calculate_ela_level(X, y))
            conf.update(calculate_nbc(X, y))
            conf.update(calculate_dispersion(X, y))
            conf.update(calculate_information_content(X, y, seed = 100))

            #all dictionairies! yeaa
            a = aaaa
            

            new_df.loc[len(new_df)] = conf
print(new_df_fidonly)

print(new_df)
#now replace fid, iid with features instead, 
#build multiple decision trees .. visualise -- multi-output tree vs single output trees