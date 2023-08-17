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
from time import monotonic

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

sample_size1 = 500 #fixed
use_sample_size2 = False
sample_size2 = 100 #fixed

ela_auc_df1 = pd.DataFrame(columns=['dim', *features, 'auc'])
ela_auc_df2 = pd.DataFrame(columns=['dim', *features, 'auc'])

for index, row in tqdm(de_explainer.df.iterrows(),  total=de_explainer.df.shape[0]):
    #let's create different random samples to make it "robust". Let's try different sizes as well
    X1 = create_initial_sample(row['dim'], lower_bound = -5, upper_bound = 5, n=sample_size1, seed=index)
    if use_sample_size2:
        X2 = create_initial_sample(row['dim'], lower_bound = -5, upper_bound = 5, n=sample_size2, seed=index)
    
    func = ioh.get_problem(row['fid'], dimension=row['dim'], instance=row['iid'])
    y1 = X1.apply(lambda x: func(x), axis = 1)
    y1 = (y1 - np.min(y1)) / (np.max(y1)  - np.min(y1))

    if use_sample_size2:
        y2= X2.apply(lambda x: func(x), axis = 1)
        y2 = (y2 - np.min(y2)) / (np.max(y2)  - np.min(y2))

    conf = row[features]
    conf['dim'] = row['dim']
    conf2 = conf.copy()

    # start_time = monotonic()
    # conf.update(calculate_ela_meta(X1, y1))
    # print(f"Run time calculate_ela_meta {monotonic() - start_time} seconds")
    # start_time = monotonic()
    #start_time = monotonic()
    conf.update(calculate_ela_distribution(X1, y1))
    # print(f"Run time calculate_ela_distribution {monotonic() - start_time} seconds")
    # start_time = monotonic()

    conf.update(calculate_nbc(X1, y1))
    # print(f"Run time calculate_nbc {monotonic() - start_time} seconds")
    # start_time = monotonic()

    #conf.update(calculate_ela_level(X1, y1))
    # print(f"Run time calculate_ela_level {monotonic() - start_time} seconds")
    # start_time = monotonic()


    conf.update(calculate_dispersion(X1, y1))
    # print(f"Run time calculate_dispersion {monotonic() - start_time} seconds")
    # start_time = monotonic()

    # conf.update(calculate_information_content(X1, y1, seed = 100))
    # print(f"Run time calculate_information_content {monotonic() - start_time} seconds")
    # start_time = monotonic()

    ela_auc_df1.loc[len(ela_auc_df1)] = conf

    if use_sample_size2:
        #conf2.update(calculate_ela_meta(X2, y2))
        conf2.update(calculate_ela_distribution(X2, y2))
        conf2.update(calculate_nbc(X2, y2))
        conf2.update(calculate_dispersion(X2, y2))
        #conf2.update(calculate_ela_level(X2, y2))
        #conf2.update(calculate_information_content(X2, y2, seed = 100))

        ela_auc_df2.loc[len(ela_auc_df2)] = conf2
            
print(ela_auc_df1)
if use_sample_size2:
    print(ela_auc_df2)
#now replace fid, iid with features instead, 
#build multiple decision trees .. visualise -- multi-output tree vs single output trees

ela_auc_df1.to_pickle("ela_auc_df1.pkl")
if use_sample_size2:
    ela_auc_df2.to_pickle("ela_auc_df2.pkl")
