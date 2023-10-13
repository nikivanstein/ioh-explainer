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


data_file = "cma_results_cpp.pkl"
features = ['covariance', 'elitist', 'mirrored', 'base_sampler', 'weights_option', 'local_restart', 'step_size_adaptation', 'lambda_', 'mu']
df = pd.read_pickle(data_file)


config_dict = {}
for f in features:
    config_dict[f] = list(map(str, df[f].unique()))

config_dict['elitist'] = [False, True]
config_dict['active'] = [False, True]
config_dict['covariance'] = [False, True]

print(config_dict)
print( df['dim'].unique())
print( df['seed'].unique())
print( df['iid'].unique())

cs = ConfigurationSpace(config_dict)

print(cs)
print( df['dim'].unique())
cmaes_explainer = explainer(None, 
                 cs , 
                 algname="mod-CMA",
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


cmaes_explainer.load_results(data_file)

sample_size = 1000 #fixed

new_doe_df = []
new_ela_df = []
new_df_fidonly = []
for dim in cmaes_explainer.dims:
    dim_df = cmaes_explainer.df[cmaes_explainer.df['dim'] == dim].copy()
    
    X = create_initial_sample(dim, lower_bound = -5, upper_bound = 5, n=sample_size, seed=42)
    

    for fid in tqdm(cmaes_explainer.fids):
        fid_df = dim_df[dim_df['fid'] == fid]

        for iid in fid_df['iid'].unique():
            iid_df = fid_df[fid_df['iid'] == iid]
            #get best performing conf
            conf, aucs = cmaes_explainer._get_single_best(iid_df)
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
            new_doe_df.append(conf)

            conf2.update(calculate_ela_meta(X, y))
            conf2.update(calculate_ela_distribution(X, y))
            conf2.update(calculate_ela_level(X, y))
            conf2.update(calculate_nbc(X, y))
            conf2.update(calculate_dispersion(X, y))
            conf2.update(calculate_information_content(X, y, seed = 100))

            #all dictionairies! yeaa
            new_ela_df.append(conf2)

#now replace fid, iid with features instead, 
#build multiple decision trees .. visualise -- multi-output tree vs single output trees

new_ela_df = pd.DataFrame.from_records(new_ela_df)
new_ela_df.to_pickle("ela-features-cma.pkl")

new_doe_df = pd.DataFrame.from_records(new_doe_df)
new_doe_df.to_pickle("doe-features-cma.pkl")

print(new_ela_df)
print(new_doe_df)
